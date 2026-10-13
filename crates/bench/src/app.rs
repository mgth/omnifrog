//! The window's state, and what runs behind it.

use crate::cli;
use egui::Ui;
use omnifrog_bench::batch;
use omnifrog_bench::film::{self, Film};
use omnifrog_bench::listing::{self, Row};
use omnifrog_bench::measure;
use omnifrog_bench::report::{self, Column, Report, ZoneReport};
use omnifrog_bench::set::{Set, Side, Zone};
use omnifrog_core::config::Config;
use omnifrog_core::models::Catalogue;
use omnifrog_core::progress;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};

pub const TITLE: &str = "omnifrog bench";
/// What the list of outputs calls the choice of none.
pub const DEFAULT_OUTPUT: &str = "System default";
const OUTPUT_WIDTH: f32 = 240.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Rate,
    Report,
    Model,
    Film,
    Cascades,
}

/// What the report tab shows, and how.
pub struct ReportView {
    pub side: Side,
    /// Every zone, or the one chosen on the left.
    pub all_zones: bool,
    /// The column each side is sorted by, and whether downwards; in the
    /// order of [`Side::BOTH`].
    pub sort: [(Column, bool); 2],
    /// The table, made again when the set, the scope or the sort changes —
    /// not at every frame.
    pub rows: Vec<Report>,
    /// The zones of the scope where a side has a single reference.
    pub single: Vec<(String, String)>,
}

/// What the model tab shows: the report the other way round, one model,
/// chosen in the left column, and a row for each zone. The side is the
/// report's.
pub struct ModelView {
    /// The model shown; the first there is until one is chosen.
    pub key: Option<String>,
    /// Every key rated or measured in the set, in the catalogue's order.
    pub keys: Vec<String>,
    /// As the report's: a column and a direction for each side. The
    /// column of the names is the zones' own order.
    pub sort: [(Column, bool); 2],
    /// One row for each zone of the set, made again when the report is.
    pub rows: Vec<ZoneReport>,
    /// The model over all the zones: its line in the report.
    pub mean: Report,
}

/// What a batch on its thread tells the window.
enum Told {
    Event(batch::Event),
    Finished(Result<bool, String>),
    /// What the core says while it extracts a film.
    Core(progress::Event),
    /// The extraction ended: the film's name in the bench.
    Extracted(Result<String, String>),
}

/// What the film tab holds: the films of the bench, the one chosen, and the
/// range of it being tried.
pub struct FilmView {
    pub films: Vec<Film>,
    /// What each takes on disk, in bytes; read with the list, not at every
    /// frame.
    pub sizes: Vec<u64>,
    pub chosen: Option<usize>,
    pub object: Option<u32>,
    pub channel: Option<String>,
    /// The range, as typed.
    pub from: String,
    pub to: String,
    /// The kind of passage the next zone is, by its place in the list.
    pub kind: usize,
    /// The range opened: a zone that is not in the set.
    pub trial: Option<Zone>,
    /// Where the trial's element of the original is over it.
    pub trajectory: Vec<(u64, [f64; 3])>,
    /// Remove was clicked once: the next click does it.
    pub removing: bool,
}

/// A batch under way.
pub struct Job {
    told: Receiver<Told>,
    stop: Arc<AtomicBool>,
    /// How far it is, from 0 to 1: models loaded and zones separated, then
    /// zones measured.
    pub progress: f32,
    /// The model in hand: its place, out of how many, the zones it has to
    /// do and those done.
    model: (usize, usize, usize, usize),
}

impl Job {
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    pub fn stopping(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }
}

pub struct App {
    pub dir: PathBuf,
    pub set: Set,
    pub catalogue: Option<Catalogue>,
    /// The catalogue's keys, in its order.
    pub order: Vec<String>,
    pub tab: Tab,
    /// The zone shown, by its place in the set.
    pub selected: Option<usize>,
    /// The rows of the zone shown: read from the disk when it is chosen, not
    /// at every frame.
    pub rows: Vec<Row>,
    pub report: ReportView,
    pub model: ModelView,
    pub listen: crate::listen::Listen,
    pub film: FilmView,
    /// Remove was clicked once on the zone shown: the next click does it.
    pub removing_zone: bool,
    /// The cascade being made: its two models and its mode.
    pub cascade: (
        Option<String>,
        Option<String>,
        omnifrog_bench::set::CascadeMode,
    ),
    /// Whether the film tab was the one shown at the last frame: the zone
    /// listened to is not the same there.
    in_film: bool,
    pub job: Option<Job>,
    /// What the bottom line says: the last thing that happened or went
    /// wrong, or nothing.
    pub status: String,
    pub ctx: egui::Context,
}

impl App {
    pub fn new(ctx: &egui::Context, dir: PathBuf, tab: Tab) -> Self {
        let mut status = String::new();
        let catalogue = match Catalogue::load() {
            Ok(catalogue) => Some(catalogue),
            Err(e) => {
                status = format!("Catalogue: {e}");
                None
            }
        };
        let order = cli::catalogue_keys(catalogue.as_ref());
        let mut app = Self {
            dir,
            set: Set::default(),
            catalogue,
            order,
            tab,
            selected: None,
            rows: Vec::new(),
            report: ReportView {
                side: Side::Vo,
                all_zones: true,
                sort: [(Column::SiSdr, true); 2],
                rows: Vec::new(),
                single: Vec::new(),
            },
            model: ModelView {
                key: None,
                keys: Vec::new(),
                sort: [(Column::Model, false); 2],
                rows: Vec::new(),
                mean: Report::default(),
            },
            listen: crate::listen::Listen::default(),
            film: FilmView {
                films: Vec::new(),
                sizes: Vec::new(),
                chosen: None,
                object: None,
                channel: None,
                from: String::new(),
                to: String::new(),
                kind: 0,
                trial: None,
                trajectory: Vec::new(),
                removing: false,
            },
            removing_zone: false,
            cascade: (None, None, omnifrog_bench::set::CascadeMode::Rest),
            in_film: tab == Tab::Film,
            job: None,
            status,
            ctx: ctx.clone(),
        };
        app.reload();
        app.refresh_films(None);
        app
    }

    /// Read the set again, keeping the zone shown if it is still there.
    pub fn reload(&mut self) {
        let shown = self
            .selected
            .and_then(|i| self.set.zones.get(i))
            .map(|z| z.id.clone());
        match Set::load(&self.dir) {
            Ok(set) => self.set = set,
            Err(e) => self.status = e.to_string(),
        }
        let place = shown
            .and_then(|id| self.set.zones.iter().position(|z| z.id == id))
            .or((!self.set.zones.is_empty()).then_some(0));
        self.select(place);
    }

    pub fn select(&mut self, place: Option<usize>) {
        let another = self.selected != place;
        if another {
            self.removing_zone = false;
        }
        self.selected = place;
        if another && self.tab != Tab::Film {
            self.refocus();
        } else {
            self.refresh_rows();
        }
    }

    /// The zone listened to: the trial under the film tab, the zone chosen
    /// on the left everywhere else.
    pub fn zone_shown(&self) -> Option<&Zone> {
        match self.tab {
            Tab::Film => self.film.trial.as_ref(),
            _ => self.selected.and_then(|i| self.set.zones.get(i)),
        }
    }

    /// The zone listened to is another: the player lets go, the rows are
    /// read again, and there is something to look at before anything plays
    /// — what is left of the element, the stem the original's side is
    /// judged on.
    pub fn refocus(&mut self) {
        self.listen_reset();
        self.refresh_rows();
        if self.zone_shown().is_some() {
            self.listen.kind = Some(crate::listen::Kind::Stem(Side::Vo.listened()));
            self.load_waves();
        }
    }

    /// Read the rows of the zone shown again, and the report with them.
    pub fn refresh_rows(&mut self) {
        self.rows = match self.zone_shown() {
            Some(zone) => listing::rows(&self.dir, &self.set, zone, &self.order),
            None => Vec::new(),
        };
        self.refresh_report();
    }

    /// The zones the report is over.
    fn scope(&self) -> Vec<&Zone> {
        match (self.report.all_zones, self.selected) {
            (false, Some(i)) => self.set.zones.get(i).into_iter().collect(),
            _ => self.set.zones.iter().collect(),
        }
    }

    /// Make the report's table again.
    pub fn refresh_report(&mut self) {
        let side = self.report.side;
        let (column, descending) = self.report.sort[side as usize];
        let zones = self.scope();
        let mut rows = report::table(&self.set, &zones, &self.order, side);
        report::sort(&mut rows, side, column, descending);
        let single = report::single_reference(&zones)
            .into_iter()
            .map(|(zone, sides)| {
                let sides: Vec<&str> = sides.iter().map(|s| s.label()).collect();
                (zone.label(), sides.join(", "))
            })
            .collect();
        self.report.rows = rows;
        self.report.single = single;
        self.refresh_model();
    }

    /// Make the model tab's table again: every zone of the set, whatever
    /// the report's scope.
    pub fn refresh_model(&mut self) {
        let side = self.report.side;
        let (column, descending) = self.model.sort[side as usize];
        let zones: Vec<&Zone> = self.set.zones.iter().collect();
        let keys = report::keys(&self.set, &zones, &self.order);
        // A model that is no longer in the set leaves its place to the
        // first one.
        if !self
            .model
            .key
            .as_ref()
            .is_some_and(|key| keys.contains(key))
        {
            self.model.key = keys.first().cloned();
        }
        let (mut rows, mean) = match &self.model.key {
            Some(key) => (
                report::by_zone(&self.set, &zones, key),
                report::of(&self.set, &zones, key),
            ),
            None => (Vec::new(), Report::default()),
        };
        report::sort(&mut rows, side, column, descending);
        self.model.keys = keys;
        self.model.rows = rows;
        self.model.mean = mean;
    }

    /// Show a model zone by zone.
    pub fn show_model(&mut self, key: &str) {
        self.model.key = Some(key.to_string());
        self.tab = Tab::Model;
        self.refresh_model();
    }

    /// Sort the model's zones by a column, as [`App::sort_by`] sorts the
    /// report's models.
    pub fn sort_model_by(&mut self, column: Column) {
        let sort = &mut self.model.sort[self.report.side as usize];
        *sort = if sort.0 == column {
            (column, !sort.1)
        } else {
            (column, column.higher_is_better())
        };
        self.refresh_model();
    }

    /// Sort the side shown by a column: the best first, the other way
    /// round when it already is.
    pub fn sort_by(&mut self, column: Column) {
        let sort = &mut self.report.sort[self.report.side as usize];
        *sort = if sort.0 == column {
            (column, !sort.1)
        } else {
            (column, column.higher_is_better())
        };
        self.refresh_report();
    }

    /// Take a key in the next batch or leave it out, and keep that.
    pub fn check(&mut self, key: &str, checked: bool) {
        self.set.checked.insert(key.to_string(), checked);
        if let Err(e) = self.set.save(&self.dir) {
            self.status = e.to_string();
        }
    }

    /// Remove the zone chosen on the left, with its files.
    pub fn remove_zone(&mut self) {
        let Some(id) = self
            .selected
            .and_then(|i| self.set.zones.get(i))
            .map(|z| z.id.clone())
        else {
            return;
        };
        self.removing_zone = false;
        // The player may hold one of its files.
        self.listen_reset();
        self.status = match self.set.remove_zone(&self.dir, &id) {
            Ok(()) => format!("{id} removed."),
            Err(e) => e.to_string(),
        };
        self.selected = None;
        let first = (!self.set.zones.is_empty()).then_some(0);
        self.select(first);
    }

    /// Add the cascade the form holds to the set, and keep that.
    pub fn add_cascade(&mut self) {
        let (Some(first), Some(second)) = (self.cascade.0.clone(), self.cascade.1.clone()) else {
            self.status = "A cascade is two models.".to_string();
            return;
        };
        let cascade = omnifrog_bench::set::Cascade {
            first,
            second,
            mode: self.cascade.2,
            kept: false,
        };
        let key = cascade.key();
        self.status = if self.set.add_cascade(cascade) {
            format!("{key} added: Compute, in the report, runs it.")
        } else {
            format!("The set already has {key}.")
        };
        self.save_set();
    }

    pub fn remove_cascade(&mut self, key: &str) {
        if self.set.remove_cascade(key) {
            self.status = format!("{key} removed; its outputs stay where they are.");
            self.save_set();
        }
    }

    pub fn keep_cascade(&mut self, key: &str, kept: bool) {
        if let Some(cascade) = self.set.cascades.iter_mut().find(|c| c.key() == key) {
            cascade.kept = kept;
            self.save_set();
        }
    }

    fn save_set(&mut self) {
        if let Err(e) = self.set.save(&self.dir) {
            self.status = e.to_string();
        }
    }

    /// Choose the variants the next batch computes, and keep that.
    pub fn set_variants(&mut self, variants: omnifrog_bench::set::Variants) {
        self.set.variants = variants;
        if let Err(e) = self.set.save(&self.dir) {
            self.status = e.to_string();
        }
    }

    /// A model's name, or its key when the catalogue does not have it.
    pub fn model_name(&self, key: &str) -> String {
        self.catalogue
            .as_ref()
            .and_then(|c| c.get(key).ok())
            .map_or(key.to_string(), |m| m.name.clone())
    }

    /// What a key is called: its model's name — its two models' for a
    /// cascade — and the variant if it is one.
    pub fn name(&self, key: &str) -> String {
        let (model, variant) = omnifrog_bench::keys::split(key);
        let name = match omnifrog_bench::keys::stages(model) {
            Some((first, second, mode)) => format!(
                "{} then {} on its {}",
                self.model_name(first),
                self.model_name(second),
                match mode {
                    omnifrog_bench::set::CascadeMode::Rest => "rest",
                    omnifrog_bench::set::CascadeMode::Voice => "voice",
                }
            ),
            None => self.model_name(model),
        };
        let name = name.as_str();
        match variant {
            Some(omnifrog_bench::keys::CANCEL) => format!("{name} — duplicates cancelled"),
            Some(_) => format!("{name} — duplicates cancelled, by step"),
            None => name.to_string(),
        }
    }

    /// Measure the checked outputs of the report's zones again.
    pub fn measure(&mut self) {
        self.batch(false);
    }

    /// Separate what the checked models lack on the report's zones, then
    /// measure.
    pub fn compute(&mut self) {
        self.batch(true);
    }

    /// A batch on a thread of its own. The thread works on the set as the
    /// disk has it — the window saves every change at once — and saves as
    /// it goes.
    fn batch(&mut self, separate: bool) {
        if self.job.is_some() {
            return;
        }
        let ids: Vec<String> = self.scope().iter().map(|z| z.id.clone()).collect();
        if ids.is_empty() {
            self.status = "No zone to work on.".to_string();
            return;
        }
        let (tx, told) = channel();
        let stop = Arc::new(AtomicBool::new(false));
        let (dir, ctx, flag) = (self.dir.clone(), self.ctx.clone(), stop.clone());
        std::thread::spawn(move || {
            let say = {
                let (tx, ctx) = (tx.clone(), ctx.clone());
                move |event: batch::Event| {
                    let _ = tx.send(Told::Event(event));
                    ctx.request_repaint();
                }
            };
            let result = (|| -> omnifrog_bench::Result<bool> {
                let mut set = Set::load(&dir)?;
                if !separate {
                    return measure::batch(&dir, &mut set, &ids, &flag, |e| {
                        say(batch::Event::Measure(e))
                    });
                }
                let config = Config::load()?;
                let catalogue = Catalogue::load()?;
                let all = batch::models_of(&catalogue);
                let models = batch::checked(&set, &all);
                let mut start = batch::worker(&config, &catalogue);
                let checked = set.checked.clone();
                let bases = move |key: &str| checked.get(key).copied().unwrap_or(true);
                batch::compute(
                    &dir, &mut set, &ids, &models, &all, &bases, &mut start, &flag, &say,
                )
            })();
            let _ = tx.send(Told::Finished(result.map_err(|e| e.to_string())));
            ctx.request_repaint();
        });
        self.status = if separate {
            "Computing…"
        } else {
            "Measuring…"
        }
        .to_string();
        self.job = Some(Job {
            told,
            stop,
            progress: 0.0,
            model: (0, 1, 1, 0),
        });
    }

    /// Read the films of the bench again, and choose one by name — or keep
    /// the one chosen.
    pub fn refresh_films(&mut self, choose: Option<&str>) {
        let name = choose.map(str::to_string).or_else(|| {
            self.film
                .chosen
                .and_then(|i| self.film.films.get(i))
                .map(|f| f.name.clone())
        });
        self.film.films = film::films(&self.dir);
        self.film.sizes = self.film.films.iter().map(|f| f.size(&self.dir)).collect();
        let place = name
            .and_then(|name| self.film.films.iter().position(|f| f.name == name))
            .or((!self.film.films.is_empty()).then_some(0));
        self.choose_film(place);
    }

    pub fn choose_film(&mut self, place: Option<usize>) {
        let view = &mut self.film;
        if view.chosen == place && place.is_some() {
            return;
        }
        view.chosen = place;
        view.removing = false;
        view.trial = None;
        view.trajectory.clear();
        let chosen = place.and_then(|i| view.films.get(i));
        // The first object and the centre of the dub: where the dialogue
        // most often is.
        view.object = chosen.and_then(|f| {
            f.elements
                .iter()
                .find(|e| e.object)
                .or(f.elements.first())
                .map(|e| e.id)
        });
        view.channel = chosen.and_then(|f| {
            f.channels
                .iter()
                .find(|c| c.as_str() == "C")
                .or(f.channels.first())
                .cloned()
        });
        if self.tab == Tab::Film {
            self.refocus();
        }
    }

    /// Extract a film whole into the bench, on a thread of its own.
    pub fn extract(&mut self, source: PathBuf) {
        if self.job.is_some() {
            return;
        }
        let (tx, told) = channel();
        let (dir, ctx) = (self.dir.clone(), self.ctx.clone());
        std::thread::spawn(move || {
            let say = {
                let (tx, ctx) = (tx.clone(), ctx.clone());
                move |event: progress::Event| {
                    let _ = tx.send(Told::Core(event));
                    ctx.request_repaint();
                }
            };
            let result = Config::load()
                .and_then(|config| film::extract(&config, &dir, &source, None, None, &say))
                .map(|film| film.name)
                .map_err(|e| e.to_string());
            let _ = tx.send(Told::Extracted(result));
            ctx.request_repaint();
        });
        self.status = "Extracting…".to_string();
        self.job = Some(Job {
            told,
            stop: Arc::new(AtomicBool::new(false)),
            progress: 0.0,
            model: (0, 1, 1, 0),
        });
    }

    /// Remove the film chosen from the bench.
    pub fn remove_film(&mut self) {
        let Some(film) = self.film.chosen.and_then(|i| self.film.films.get(i)) else {
            return;
        };
        self.status = match film::remove(&self.dir, film) {
            Ok(()) => format!("{} removed.", film.name),
            Err(e) => e.to_string(),
        };
        self.film.chosen = None;
        self.refresh_films(None);
    }

    /// Open the range typed: cut it out of the film, and listen to it.
    pub fn open_trial(&mut self) {
        let view = &self.film;
        let Some(film) = view.chosen.and_then(|i| view.films.get(i)) else {
            return;
        };
        let (Some(object), Some(channel)) = (view.object, view.channel.clone()) else {
            return;
        };
        let range = omnifrog_bench::set::parse_clock(&view.from)
            .zip(omnifrog_bench::set::parse_clock(&view.to));
        let Some((from, to)) = range else {
            self.status = "A range is two times: 12:02 and 12:40, say.".to_string();
            return;
        };
        match film::trial(&self.dir, film, object, &channel, from, to) {
            Ok(trial) => {
                self.film.trajectory = film::trajectory_of(&self.dir, film, &trial);
                self.film.trial = Some(trial);
                self.status.clear();
                self.refocus();
            }
            Err(e) => self.status = e.to_string(),
        }
    }

    /// Run every checked model on the range opened, on a thread of its own.
    pub fn separate_trial(&mut self) {
        let (Some(trial), None) = (self.film.trial.clone(), &self.job) else {
            return;
        };
        let trajectory = self.film.trajectory.clone();
        let (tx, told) = channel();
        let stop = Arc::new(AtomicBool::new(false));
        let (dir, ctx, flag) = (self.dir.clone(), self.ctx.clone(), stop.clone());
        let checked = self.set.checked.clone();
        std::thread::spawn(move || {
            let say = {
                let (tx, ctx) = (tx.clone(), ctx.clone());
                move |event: batch::Event| {
                    let _ = tx.send(Told::Event(event));
                    ctx.request_repaint();
                }
            };
            let result = (|| -> omnifrog_bench::Result<bool> {
                let config = Config::load()?;
                let catalogue = Catalogue::load()?;
                let models: Vec<batch::Model> = batch::models_of(&catalogue)
                    .into_iter()
                    .filter(|m| checked.get(&m.key).copied().unwrap_or(true))
                    .collect();
                let target = batch::Target {
                    zone: &trial,
                    trajectory: &trajectory,
                };
                let mut start = batch::worker(&config, &catalogue);
                let (done, _) = batch::separate(&dir, &[target], &models, &mut start, &flag, &say)?;
                Ok(done)
            })();
            let _ = tx.send(Told::Finished(result.map_err(|e| e.to_string())));
            ctx.request_repaint();
        });
        self.status = "Separating…".to_string();
        self.job = Some(Job {
            told,
            stop,
            progress: 0.0,
            model: (0, 1, 1, 0),
        });
    }

    /// Make a zone of the range opened: of its loop, or of all of it.
    pub fn add_zone(&mut self) {
        let Some(trial) = self.film.trial.clone() else {
            return;
        };
        let kind = omnifrog_bench::set::KINDS[self.film.kind.min(7)];
        let range = self.listen.player.looped();
        self.status = match film::add_zone(&self.dir, &mut self.set, &trial, range, kind) {
            Ok(id) => format!("Zone {id} added: rate it under Rate."),
            Err(e) => e.to_string(),
        };
        self.refresh_report();
    }

    /// Take in what the batch said since the last frame.
    fn drain(&mut self) {
        let Some(job) = &mut self.job else {
            return;
        };
        let mut finished = None;
        let mut extracted = None;
        let mut changed = false;
        while let Ok(told) = job.told.try_recv() {
            let within = |job: &Job, zone: f64| {
                let (index, total, zones, done) = job.model;
                (index as f32 + (done as f32 + zone as f32) / zones.max(1) as f32)
                    / total.max(1) as f32
            };
            match told {
                Told::Event(batch::Event::Model {
                    key,
                    zones,
                    index,
                    total,
                }) => {
                    job.model = (index, total, zones, 0);
                    job.progress = within(job, 0.0);
                    self.status = format!("Loading {key} ({}/{total})…", index + 1);
                }
                Told::Event(batch::Event::Zone { key, zone, done }) => {
                    job.progress = within(job, done);
                    self.status = format!("{key} on {zone}…");
                }
                Told::Event(batch::Event::Separated { .. }) => {
                    job.model.3 += 1;
                    job.progress = within(job, 0.0);
                    changed = true;
                }
                Told::Event(batch::Event::Variant { key, zone, .. }) => {
                    self.status = format!("{key} on {zone}");
                    changed = true;
                }
                Told::Event(batch::Event::Failed { key, why }) => {
                    self.status = format!("{key}: {why}");
                }
                Told::Event(batch::Event::Note(text)) => self.status = text,
                Told::Event(batch::Event::Measure(measure::Event::Zone { id, index, total })) => {
                    job.progress = index as f32 / total.max(1) as f32;
                    self.status = format!("Measuring {id} ({}/{total})…", index + 1);
                }
                Told::Event(batch::Event::Measure(measure::Event::Measured { .. })) => {
                    changed = true;
                }
                Told::Finished(result) => finished = Some(result),
                Told::Core(progress::Event::Message { text }) => self.status = text,
                Told::Core(progress::Event::Fraction {
                    what,
                    done,
                    total: Some(total),
                    ..
                }) if total > 0 => {
                    job.progress = done as f32 / total as f32;
                    self.status = format!("Extracting: {what}");
                }
                Told::Core(_) => {}
                Told::Extracted(result) => extracted = Some(result),
            }
        }
        if let Some(result) = extracted {
            self.job = None;
            match result {
                Ok(name) => {
                    self.status = format!("{name} extracted.");
                    self.film.chosen = None;
                    self.refresh_films(Some(&name));
                }
                Err(e) => self.status = e,
            }
            return;
        }
        if let Some(result) = finished {
            self.job = None;
            self.status = match result {
                Ok(true) => "Done.".to_string(),
                Ok(false) => "Stopped; what was done is kept.".to_string(),
                Err(e) => e,
            };
            changed = true;
        }
        if changed {
            self.batch_changed();
        }
    }

    /// A batch wrote something: read the set again, whatever tab is shown.
    /// The batch's thread saved `bench.json` itself; a set kept here from
    /// before it would be written over its measures at the next thing
    /// changed in the window. What is being listened to stays as it is.
    pub fn batch_changed(&mut self) {
        self.reload();
        if self.tab == Tab::Film && self.listen.waves.len() < self.rows.len() {
            self.load_waves();
        }
    }

    fn top_bar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tab, Tab::Rate, "Rate");
            ui.selectable_value(&mut self.tab, Tab::Report, "Report");
            ui.selectable_value(&mut self.tab, Tab::Model, "Model");
            ui.selectable_value(&mut self.tab, Tab::Film, "Film");
            ui.selectable_value(&mut self.tab, Tab::Cascades, "Cascades");
            ui.separator();
            ui.weak(self.dir.display().to_string());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(self.job.is_none(), egui::Button::new("Reload"))
                    .clicked()
                {
                    self.status.clear();
                    self.reload();
                }
                ui.separator();
                self.output_choice(ui);
                ui.weak("Output");
            });
        });
    }

    /// Where the sound goes. One width whatever is chosen: nothing beside
    /// it moves. The outputs are asked of the system when the list opens.
    fn output_choice(&mut self, ui: &mut Ui) {
        let chosen = self.listen.chosen.clone();
        let mut pick = None;
        let open = egui::ComboBox::from_id_salt("output")
            .width(OUTPUT_WIDTH)
            .selected_text(chosen.as_ref().map_or(DEFAULT_OUTPUT, |c| c.name.as_str()))
            .truncate()
            .show_ui(ui, |ui| {
                if !self.listen.listed {
                    self.list_outputs();
                }
                if ui
                    .selectable_label(chosen.is_none(), DEFAULT_OUTPUT)
                    .clicked()
                {
                    pick = Some(None);
                }
                for choice in &self.listen.choices {
                    let is_chosen = chosen.as_ref().is_some_and(|c| c.is(choice));
                    if ui.selectable_label(is_chosen, &choice.name).clicked() {
                        pick = Some(Some(choice.clone()));
                    }
                }
                // The one chosen stays in the list when it is unplugged:
                // it is what Play will ask for.
                if let Some(gone) = chosen.as_ref().filter(|_| !self.output_listed()) {
                    let _ = ui.selectable_label(true, format!("{} – not there", gone.name));
                }
            })
            .inner
            .is_some();
        self.listen.listed = open;
        if let Some(choice) = pick {
            self.choose_output(choice);
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        self.draw(ui);
    }
}

impl App {
    /// One frame of the window. Apart from the frame eframe hands over,
    /// which nothing here uses: the tests draw the window without one.
    pub fn draw(&mut self, ui: &mut Ui) {
        self.drain();
        self.drain_listen();
        egui::Panel::top("top-bar").show(ui, |ui| {
            ui.add_space(4.0);
            self.top_bar(ui);
            ui.add_space(2.0);
        });
        // Under the film tab the zone listened to is the range tried, not
        // the zone chosen on the left: going in or out changes it.
        let in_film = self.tab == Tab::Film;
        if in_film != self.in_film {
            self.in_film = in_film;
            self.refocus();
        }
        // One line, always there: nothing above the lists moves when
        // something is said.
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.label(if self.status.is_empty() {
                    " "
                } else {
                    self.status.as_str()
                });
            });
            ui.add_space(2.0);
        });
        egui::Panel::left("zones")
            .resizable(true)
            .default_size(280.0)
            .min_size(220.0)
            // The column lists what the tab's lines are not: the models
            // under the tab that has a line for each zone.
            .show(ui, |ui| match self.tab {
                Tab::Model => crate::view_model::list(self, ui),
                _ => crate::view_zones::list(self, ui),
            });
        egui::CentralPanel::default().show(ui, |ui| match self.tab {
            Tab::Rate => crate::view_rate::ui(self, ui),
            Tab::Film => crate::view_film::ui(self, ui),
            Tab::Cascades => crate::view_cascades::ui(self, ui),
            Tab::Report => crate::view_report::ui(self, ui),
            Tab::Model => crate::view_model::ui(self, ui),
        });
    }
}
