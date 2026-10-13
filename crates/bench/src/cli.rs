//! The bench without its window.

use clap::{Parser, Subcommand};
use omnifrog_bench::batch;
use omnifrog_bench::film;
use omnifrog_bench::import;
use omnifrog_bench::listing;
use omnifrog_bench::measure;
use omnifrog_bench::report::{self, Column};
use omnifrog_bench::set::{self, Set, Side, Status, Zone};
use omnifrog_core::config::Config;
use omnifrog_core::models::Catalogue;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "omnifrog-bench",
    version,
    about = "The model test bench of omnifrog. Without a subcommand: the window."
)]
pub struct Args {
    /// The bench directory; by default `_bench` in omnifrog's work directory.
    #[arg(long, global = true, value_name = "DIR")]
    pub dir: Option<PathBuf>,
    /// The tab the window opens on.
    #[arg(long, value_enum, default_value = "rate")]
    pub tab: TabArg,
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum TabArg {
    Rate,
    Report,
    Model,
    Film,
    Cascades,
}

#[derive(Subcommand)]
pub enum Command {
    /// Bring in a legacy test set: a `_JeuxEssai` directory.
    Import {
        /// The `_JeuxEssai`, the directory that holds `jeu.json`.
        from: PathBuf,
    },
    /// List the zones of the set, or the outputs of one.
    List {
        /// A zone's id: list its outputs rather than the zones.
        #[arg(long, value_name = "ID")]
        zone: Option<String>,
    },
    /// Measure every checked output against the references, and save.
    Measure {
        /// One zone rather than all of them.
        #[arg(long, value_name = "ID")]
        zone: Option<String>,
        /// Measure again, compare with what the set and the reference files
        /// hold, and write nothing.
        #[arg(long)]
        dry_run: bool,
    },
    /// Separate what the models lack on the zones, then measure. It takes
    /// the device: one separation at a time on a machine.
    Separate {
        /// One zone rather than all of them.
        #[arg(long, value_name = "ID")]
        zone: Option<String>,
        /// These models, by key, rather than the ones the set has checked.
        #[arg(long, value_name = "KEYS", value_delimiter = ',')]
        models: Vec<String>,
    },
    /// Extract a film whole into the bench: every element of its original
    /// and every channel of its dub, kept for the zones cut from it.
    Extract {
        film: PathBuf,
        /// The original track, among the audio tracks from 1; by default
        /// the one a run would take.
        #[arg(long, value_name = "N")]
        vo: Option<u32>,
        /// The dubbed track; by default the one a run would take.
        #[arg(long, value_name = "N")]
        vf: Option<u32>,
    },
    /// The films extracted in the bench, and what they take.
    Films {
        /// Remove this one: its extraction and its trials. Its zones stay.
        #[arg(long, value_name = "NAME")]
        remove: Option<String>,
    },
    /// Cut a zone out of a film of the bench. With `--models`, those are
    /// first run on the range and some seconds either side of it, and the
    /// zone gets their outputs: a short zone heard in its context.
    AddZone {
        /// The film, by its name in the bench (see `films`).
        #[arg(long)]
        film: String,
        /// The element of the original, by its id.
        #[arg(long)]
        object: u32,
        /// The channel of the dub, by its name.
        #[arg(long)]
        channel: String,
        /// Where the zone starts in the film: `12:02`, `1:02:03.5`, seconds.
        #[arg(long)]
        from: String,
        #[arg(long)]
        to: String,
        #[arg(long, default_value = "Other")]
        kind: String,
        /// Models to run before the zone is cut, by key.
        #[arg(long, value_name = "KEYS", value_delimiter = ',')]
        models: Vec<String>,
        /// Seconds the models hear either side of the zone.
        #[arg(long, default_value_t = 15.0)]
        context: f64,
    },
    /// The cascades of the set: two models one after the other. Without an
    /// option, lists them.
    Cascade {
        /// Add one: the model that separates, then the one that works one
        /// of its stems again.
        #[arg(long, num_args = 2, value_names = ["FIRST", "SECOND"])]
        add: Vec<String>,
        /// The stem the second model works again: the rest, against a
        /// leak, or the voice, against a lack.
        #[arg(long, value_enum, default_value = "rest")]
        mode: ModeArg,
        /// Take one out, by its key. Its outputs stay.
        #[arg(long, value_name = "KEY")]
        remove: Option<String>,
    },
    /// Remove a zone from the set, with all its files. There is no way back.
    RemoveZone {
        /// The zone's id (see `list`).
        id: String,
    },
    /// The models summed up over the zones; or, with `--model`, one model
    /// zone by zone.
    Report {
        /// One zone rather than all of them.
        #[arg(long, value_name = "ID")]
        zone: Option<String>,
        /// One model, with a line for each zone, rather than a line for
        /// each model: `--sort model` is then the zones' own order.
        #[arg(long, value_name = "KEY", conflicts_with = "zone")]
        model: Option<String>,
        /// The original's side (what is left of the element) or the dub's
        /// (the voice of the channel).
        #[arg(long, value_enum, default_value = "vo")]
        side: SideArg,
        /// The column to sort by; the best first.
        #[arg(long, value_enum, default_value = "si-sdr")]
        sort: SortArg,
    },
}

#[derive(Clone, Copy, clap::ValueEnum)]
pub enum ModeArg {
    Rest,
    Voice,
}

#[derive(Clone, Copy, clap::ValueEnum)]
pub enum SideArg {
    Vo,
    Vf,
}

#[derive(Clone, Copy, clap::ValueEnum)]
pub enum SortArg {
    Model,
    SiSdr,
    Residue,
    Loss,
    Artifacts,
    Gap,
    Excess,
    Lack,
    Refs,
    Rejects,
}

/// Where the bench keeps its set: `--dir`, else beside omnifrog's jobs.
/// A configuration that does not read is an error, never the default work
/// directory: the bench would open an empty set somewhere else and look as if
/// it had lost its zones.
pub fn bench_dir(asked: Option<&Path>) -> Result<PathBuf, String> {
    match asked {
        Some(dir) => Ok(dir.to_path_buf()),
        None => Config::load()
            .map(|config| config.work_dir().join(set::DIR))
            .map_err(|e| format!("{e}\n(or say where the bench is with --dir)")),
    }
}

/// The catalogue's keys, in its order; none when it does not load, and every
/// key is then foreign.
pub fn catalogue_keys(catalogue: Option<&Catalogue>) -> Vec<String> {
    catalogue
        .map(|c| c.models.iter().map(|m| m.key.clone()).collect())
        .unwrap_or_default()
}

pub fn run(command: Command, dir: &Path) -> Result<(), String> {
    let catalogue = Catalogue::load().map_err(|e| e.to_string())?;
    match command {
        Command::Import { from } => import_set(&from, dir, &catalogue),
        Command::List { zone } => list(dir, zone.as_deref(), &catalogue),
        Command::Measure { zone, dry_run } => measure_set(dir, zone.as_deref(), dry_run),
        Command::Separate { zone, models } => separate(dir, zone.as_deref(), &models, &catalogue),
        Command::Cascade { add, mode, remove } => {
            cascade(dir, &catalogue, &add, mode, remove.as_deref())
        }
        Command::RemoveZone { id } => {
            let mut set = Set::load(dir).map_err(|e| e.to_string())?;
            set.remove_zone(dir, &id).map_err(|e| e.to_string())?;
            println!("{id} removed");
            Ok(())
        }
        Command::Extract { film, vo, vf } => extract(dir, &film, vo, vf),
        Command::Films { remove } => films(dir, remove.as_deref()),
        Command::AddZone {
            film,
            object,
            channel,
            from,
            to,
            kind,
            models,
            context,
        } => add_zone(
            dir,
            &catalogue,
            &film,
            (object, &channel),
            (&from, &to),
            &kind,
            &models,
            context,
        ),
        Command::Report {
            zone,
            model,
            side,
            sort,
        } => {
            let side = match side {
                SideArg::Vo => Side::Vo,
                SideArg::Vf => Side::Vf,
            };
            let column = match sort {
                SortArg::Model => Column::Model,
                SortArg::SiSdr => Column::SiSdr,
                SortArg::Residue => Column::Residue,
                SortArg::Loss => Column::Loss,
                SortArg::Artifacts => Column::Artifacts,
                SortArg::Gap => Column::Gap,
                SortArg::Excess => Column::Excess,
                SortArg::Lack => Column::Lack,
                SortArg::Refs => Column::Refs,
                SortArg::Rejects => Column::Rejects,
            };
            match model {
                Some(key) => report_model(dir, &key, side, column).map(|table| {
                    let _ = std::io::stdout().lock().write_all(table.as_bytes());
                }),
                None => report_set(dir, zone.as_deref(), side, column, &catalogue),
            }
        }
    }
}

fn import_set(from: &Path, dir: &Path, catalogue: &Catalogue) -> Result<(), String> {
    println!("{} → {}", from.display(), dir.display());
    let report = import::run(
        from,
        dir,
        |key| catalogue.get(key).is_ok(),
        |zone| println!("  {zone}"),
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{} zone(s) in, {} already here; {} key(s), {} file(s), {:.1} MB",
        report.zones.len(),
        report.skipped.len(),
        report.keys.len(),
        report.files,
        report.bytes as f64 / 1e6
    );
    if !report.foreign.is_empty() {
        let foreign: Vec<&str> = report.foreign.iter().map(String::as_str).collect();
        println!(
            "not in the catalogue, kept as they are: {}",
            foreign.join(", ")
        );
    }
    for warning in &report.warnings {
        println!("warning: {warning}");
    }
    Ok(())
}

fn list(dir: &Path, zone: Option<&str>, catalogue: &Catalogue) -> Result<(), String> {
    // Written through a handle whose errors are dropped: `list | head` closes
    // the pipe early, and that is no failure.
    let mut out = std::io::stdout().lock();
    let set = Set::load(dir).map_err(|e| e.to_string())?;
    let order = catalogue_keys(Some(catalogue));
    let Some(id) = zone else {
        let _ = writeln!(out, "{} — {} zone(s)", dir.display(), set.zones.len());
        for zone in &set.zones {
            let rows = listing::rows(dir, &set, zone, &order);
            let complete = rows.iter().filter(|r| r.files() == 5).count();
            let measured = rows.iter().filter(|r| r.measure.is_some()).count();
            let _ = writeln!(out, "{}", zone.id);
            let _ = writeln!(out, "  {} · {}", zone.label(), zone.kind);
            let _ = writeln!(
                out,
                "  {} output(s), {complete} complete, {measured} measured · ★ {} VO, {} VF · ✗ {} VO, {} VF",
                rows.len(),
                zone.vo.refs.len(),
                zone.vf.refs.len(),
                zone.vo.rejects.len(),
                zone.vf.rejects.len()
            );
        }
        return Ok(());
    };
    let zone = set
        .zone(id)
        .ok_or_else(|| format!("no zone `{id}` in {}", dir.display()))?;
    let _ = writeln!(out, "{} · {}", zone.label(), zone.kind);
    for row in listing::rows(dir, &set, zone, &order) {
        let mark = |side: Side| match row.status(side) {
            Status::Reference => "★".to_string(),
            Status::Rejected(Some(reason)) => format!("✗ {}", reason.label()),
            Status::Rejected(None) => "✗".to_string(),
            Status::Neutral => "·".to_string(),
        };
        let si_sdr = |side: Side| {
            row.measure
                .as_ref()
                .and_then(|m| m.stems.get(&side.listened()))
                .and_then(|m| m.si_sdr)
                .map_or("—".to_string(), |v| format!("{v:.1}"))
        };
        let _ = writeln!(
            out,
            "  {:<28} {}/5  VO {:<12} {:>6}   VF {:<12} {:>6}{}",
            row.key,
            row.files(),
            mark(Side::Vo),
            si_sdr(Side::Vo),
            mark(Side::Vf),
            si_sdr(Side::Vf),
            if row.foreign {
                "   (not in the catalogue)"
            } else {
                ""
            }
        );
    }
    Ok(())
}

/// The zones a subcommand works on: the one named, or all of them.
fn scope<'a>(set: &'a Set, dir: &Path, zone: Option<&str>) -> Result<Vec<&'a Zone>, String> {
    match zone {
        Some(id) => set
            .zone(id)
            .map(|zone| vec![zone])
            .ok_or_else(|| format!("no zone `{id}` in {}", dir.display())),
        None => Ok(set.zones.iter().collect()),
    }
}

fn measure_set(dir: &Path, zone: Option<&str>, dry_run: bool) -> Result<(), String> {
    let mut set = Set::load(dir).map_err(|e| e.to_string())?;
    let ids: Vec<String> = scope(&set, dir, zone)?
        .iter()
        .map(|z| z.id.clone())
        .collect();
    if dry_run {
        let mut all = measure::Difference::default();
        for id in &ids {
            let zone = set.zone(id).expect("an id of the set");
            let check = measure::check(dir, &set, zone).map_err(|e| e.to_string())?;
            let references: Vec<String> = Side::BOTH
                .iter()
                .filter_map(|&side| {
                    let (unlike, total) = check.references[side as usize]?;
                    Some(match unlike {
                        0 => format!("{} reference identical", side.label()),
                        _ => format!(
                            "{} reference: {unlike} of {total} samples differ",
                            side.label()
                        ),
                    })
                })
                .collect();
            println!("{id}");
            println!(
                "  {} output(s), {} value(s): largest difference {:.1e} dB{}, {} unlike, {} not measured before",
                check.outputs,
                check.measures.values,
                check.measures.largest,
                check
                    .worst
                    .as_ref()
                    .filter(|(_, worst)| *worst > 0.0)
                    .map_or(String::new(), |(key, _)| format!(" ({key})")),
                check.measures.unlike,
                check.unmeasured
            );
            if !references.is_empty() {
                println!("  {}", references.join(" · "));
            }
            all.add(check.measures);
        }
        println!(
            "{} zone(s), {} value(s): largest difference {:.1e} dB, {} unlike. Nothing written.",
            ids.len(),
            all.values,
            all.largest,
            all.unlike
        );
        return Ok(());
    }
    let stop = std::sync::atomic::AtomicBool::new(false);
    measure::batch(dir, &mut set, &ids, &stop, |event| match event {
        measure::Event::Zone { id, index, total } => println!("[{}/{total}] {id}", index + 1),
        measure::Event::Measured { outputs, .. } => println!("  {outputs} output(s) measured"),
    })
    .map_err(|e| e.to_string())?;
    Ok(())
}

fn report_set(
    dir: &Path,
    zone: Option<&str>,
    side: Side,
    column: Column,
    catalogue: &Catalogue,
) -> Result<(), String> {
    let set = Set::load(dir).map_err(|e| e.to_string())?;
    let zones = scope(&set, dir, zone)?;
    let order = catalogue_keys(Some(catalogue));
    let mut reports = report::table(&set, &zones, &order, side);
    report::sort(&mut reports, side, column, column.higher_is_better());
    let mut out = std::io::stdout().lock();
    let stem = side.listened();
    let _ = writeln!(
        out,
        "{} — {} over {} zone(s), by {}",
        side.label(),
        stem.name(),
        zones.len(),
        match column {
            Column::Model => "the catalogue's order",
            other => other.label(),
        }
    );
    for (zone, sides) in report::single_reference(&zones) {
        let sides: Vec<&str> = sides.iter().map(|s| s.label()).collect();
        let _ = writeln!(
            out,
            "warning: a single ★ on {} ({}): the measure there favours that model's family",
            zone.id,
            sides.join(", ")
        );
    }
    let _ = writeln!(
        out,
        "{:<28} {:>7} {:>7} {:>7} {:>7} {:>6} {:>6} {:>6} {:>7} {:>3} {:>3} {:>5} {:>7}",
        "key",
        "SI-SDR",
        "residue",
        "loss",
        "artif.",
        "gap",
        "excess",
        "lack",
        "other",
        "★",
        "✗",
        "zones",
        "time"
    );
    for report in &reports {
        let _ = writeln!(
            out,
            "{:<28} {} {:>3} {:>3} {:>5} {}",
            report.key,
            metrics(report, side),
            report.refs[side as usize],
            report.rejects[side as usize],
            report.mean(stem).n,
            cell(report.seconds, 6, 1) + "s",
        );
    }
    Ok(())
}

fn cell(value: Option<f64>, width: usize, decimals: usize) -> String {
    match value {
        Some(value) => format!("{value:>width$.decimals$}"),
        None => format!("{:>width$}", "—"),
    }
}

/// The eight numbers of a report's line, from SI-SDR to the other stem's.
fn metrics(report: &report::Report, side: Side) -> String {
    let stem = side.listened();
    let mean = report.mean(stem);
    let si_sdr = if report.only_reference(side) {
        format!("{:>7}", "ref.")
    } else {
        cell(mean.si_sdr, 7, 1)
    };
    format!(
        "{si_sdr} {} {} {} {} {} {} {}",
        cell(mean.residue, 7, 1),
        cell(mean.loss, 7, 1),
        cell(mean.artifacts, 7, 1),
        cell(mean.gap, 6, 2),
        cell(mean.excess, 6, 2),
        cell(mean.lack, 6, 2),
        cell(report.mean(stem.other()).si_sdr, 7, 1),
    )
}

/// One model zone by zone, and its mean over them at the end, as the text
/// to print. The models it knows are the report's and the window's: a key
/// the set holds anything of, were it only the time its outputs took.
pub fn report_model(dir: &Path, key: &str, side: Side, column: Column) -> Result<String, String> {
    let set = Set::load(dir).map_err(|e| e.to_string())?;
    let zones: Vec<&Zone> = set.zones.iter().collect();
    if !report::keys(&set, &zones, &[]).iter().any(|k| k == key) {
        return Err(format!(
            "{key} is neither measured nor rated on any zone of {}",
            dir.display()
        ));
    }
    let mut rows = report::by_zone(&set, &zones, key);
    let mean = report::of(&set, &zones, key);
    report::sort(&mut rows, side, column, column.higher_is_better());
    let mut out = Vec::new();
    let width = rows.iter().map(|r| r.zone.len()).max().unwrap_or(0).max(4);
    let _ = writeln!(
        out,
        "{} — {} of {key} zone by zone, by {}",
        side.label(),
        side.listened().name(),
        match column {
            Column::Model => "the zones' order",
            other => other.label(),
        }
    );
    let _ = writeln!(
        out,
        "{:<width$} {:>7} {:>7} {:>7} {:>7} {:>6} {:>6} {:>6} {:>7} {:>1} {:<9} {:>7}",
        "zone",
        "SI-SDR",
        "residue",
        "loss",
        "artif.",
        "gap",
        "excess",
        "lack",
        "other",
        "★",
        "✗",
        "time"
    );
    for row in &rows {
        let report = &row.report;
        let rejected = match report.rejects[side as usize] {
            0 => "·",
            _ => set::Reason::ALL
                .iter()
                .zip(report.reasons[side as usize])
                .find(|(_, n)| *n > 0)
                .map_or("✗", |(reason, _)| reason.label()),
        };
        let _ = writeln!(
            out,
            "{:<width$} {} {} {rejected:<9} {}",
            row.zone,
            metrics(report, side),
            if report.refs[side as usize] > 0 {
                "★"
            } else {
                "·"
            },
            cell(report.seconds, 6, 1) + "s",
        );
    }
    let _ = writeln!(
        out,
        "{:<width$} {} {} {:<9} {}",
        format!("mean ({})", mean.mean(side.listened()).n),
        metrics(&mean, side),
        mean.refs[side as usize],
        mean.rejects[side as usize],
        cell(mean.seconds, 6, 1) + "s",
    );
    Ok(String::from_utf8_lossy(&out).into_owned())
}

fn separate(
    dir: &Path,
    zone: Option<&str>,
    keys: &[String],
    catalogue: &Catalogue,
) -> Result<(), String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    let mut set = Set::load(dir).map_err(|e| e.to_string())?;
    let ids: Vec<String> = scope(&set, dir, zone)?
        .iter()
        .map(|z| z.id.clone())
        .collect();
    let all = batch::models_of(catalogue);
    let models = if keys.is_empty() {
        batch::checked(&set, &all)
    } else {
        // A model of the catalogue, or a cascade of the set.
        for key in keys {
            if !set.cascades.iter().any(|c| &c.key() == key) {
                catalogue.get(key).map_err(|e| e.to_string())?;
            }
        }
        all.iter()
            .filter(|m| keys.contains(&m.key))
            .cloned()
            .collect()
    };
    let stop = std::sync::atomic::AtomicBool::new(false);
    let mut start = batch::worker(&config, catalogue);
    // The variants: of the models named, or of every checked output.
    let named = keys.to_vec();
    let checked = set.checked.clone();
    let bases = move |key: &str| match named.is_empty() {
        true => checked.get(key).copied().unwrap_or(true),
        false => named.iter().any(|k| k == key),
    };
    batch::compute(
        dir,
        &mut set,
        &ids,
        &models,
        &all,
        &bases,
        &mut start,
        &stop,
        &|event| match event {
            batch::Event::Model {
                key,
                zones,
                index,
                total,
            } => println!("[{}/{total}] {key}: {zones} zone(s)", index + 1),
            batch::Event::Zone { .. } => {}
            batch::Event::Separated { zone, seconds, .. } => println!("  {zone}: {seconds:.1} s"),
            batch::Event::Variant {
                key,
                zone,
                removed_db,
            } => println!("  {key} on {zone}: {removed_db:.1} dB taken out of the dub's voice"),
            batch::Event::Failed { key, why } => println!("  {key} failed: {why}"),
            batch::Event::Note(text) => println!("  {text}"),
            batch::Event::Measure(measure::Event::Zone { id, .. }) => println!("measuring {id}"),
            batch::Event::Measure(measure::Event::Measured { .. }) => {}
        },
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// What the core says as it works, on the terminal: its messages, and a
/// fraction every few percent rather than at every block.
fn narrate() -> impl Fn(omnifrog_core::progress::Event) + Sync {
    let last = std::sync::atomic::AtomicU64::new(u64::MAX);
    move |event| match event {
        omnifrog_core::progress::Event::Message { text } => println!("  {text}"),
        omnifrog_core::progress::Event::Fraction {
            what,
            done,
            total: Some(total),
            ..
        } if total > 0 => {
            let twentieth = done * 20 / total;
            if last.swap(twentieth, std::sync::atomic::Ordering::Relaxed) != twentieth {
                println!("  {what}: {} %", twentieth * 5);
            }
        }
        _ => {}
    }
}

fn extract(dir: &Path, source: &Path, vo: Option<u32>, vf: Option<u32>) -> Result<(), String> {
    let config = Config::load().map_err(|e| e.to_string())?;
    println!(
        "{} → {}",
        source.display(),
        dir.join(film::FILMS_DIR).display()
    );
    let film =
        film::extract(&config, dir, source, vo, vf, &narrate()).map_err(|e| e.to_string())?;
    println!(
        "{}: {} element(s), {} channel(s) of the dub, {}, {:.1} GB",
        film.name,
        film.elements.len(),
        film.channels.len(),
        set::clock(film.frames),
        film.size(dir) as f64 / 1e9
    );
    Ok(())
}

fn films(dir: &Path, remove: Option<&str>) -> Result<(), String> {
    let all = film::films(dir);
    if let Some(name) = remove {
        let film = all
            .iter()
            .find(|f| f.name == name)
            .ok_or_else(|| format!("no film `{name}` in {}", dir.display()))?;
        let size = film.size(dir);
        film::remove(dir, film).map_err(|e| e.to_string())?;
        println!("{name} removed: {:.1} GB", size as f64 / 1e9);
        return Ok(());
    }
    let mut out = std::io::stdout().lock();
    for film in &all {
        let objects: Vec<String> = film
            .elements
            .iter()
            .filter(|e| e.object)
            .map(|e| e.id.to_string())
            .collect();
        let bed: Vec<String> = film
            .elements
            .iter()
            .filter(|e| !e.object)
            .map(|e| format!("{} {}", e.id, e.name))
            .collect();
        let _ = writeln!(
            out,
            "{}\n  {} · {:.1} GB{}\n  tracks {} and {} of {}\n  bed: {}\n  objects: {}\n  dub: {}",
            film.name,
            set::clock(film.frames),
            film.size(dir) as f64 / 1e9,
            if film.complete(dir) {
                ""
            } else {
                " · INCOMPLETE"
            },
            film.original,
            film.dub,
            film.source,
            bed.join(", "),
            objects.join(", "),
            film.channels.join(", ")
        );
    }
    if all.is_empty() {
        let _ = writeln!(out, "no film in {}", dir.display());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn add_zone(
    dir: &Path,
    catalogue: &Catalogue,
    name: &str,
    (object, channel): (u32, &str),
    (from, to): (&str, &str),
    kind: &str,
    keys: &[String],
    context: f64,
) -> Result<(), String> {
    let film = film::films(dir)
        .into_iter()
        .find(|f| f.name == name)
        .ok_or_else(|| format!("no film `{name}` in {}; see `films`", dir.display()))?;
    let time = |text: &str| set::parse_clock(text).ok_or_else(|| format!("`{text}` is not a time"));
    let (from, to) = (time(from)?, time(to)?.min(film.frames));
    if to <= from {
        return Err("an empty range".to_string());
    }
    // Without models nothing hears the context: the trial is the zone.
    let margin = if keys.is_empty() {
        0
    } else {
        (context.max(0.0) * set::RATE as f64) as u64
    };
    let start = from.saturating_sub(margin);
    let end = (to + margin).min(film.frames);
    let trial = film::trial(dir, &film, object, channel, start, end).map_err(|e| e.to_string())?;
    if !keys.is_empty() {
        for key in keys {
            catalogue.get(key).map_err(|e| e.to_string())?;
        }
        let config = Config::load().map_err(|e| e.to_string())?;
        let models: Vec<batch::Model> = batch::models_of(catalogue)
            .into_iter()
            .filter(|m| keys.contains(&m.key))
            .collect();
        let trajectory = film::trajectory_of(dir, &film, &trial);
        let target = batch::Target {
            zone: &trial,
            trajectory: &trajectory,
        };
        let stop = std::sync::atomic::AtomicBool::new(false);
        let mut start_worker = batch::worker(&config, catalogue);
        println!(
            "{} – {} of {name}, object {object} / channel {channel}",
            set::clock(start),
            set::clock(end)
        );
        batch::separate(
            dir,
            &[target],
            &models,
            &mut start_worker,
            &stop,
            &|event| match event {
                batch::Event::Model {
                    key, index, total, ..
                } => println!("[{}/{total}] {key}", index + 1),
                batch::Event::Separated { seconds, .. } => println!("  {seconds:.1} s"),
                batch::Event::Failed { key, why } => println!("  {key} failed: {why}"),
                batch::Event::Note(text) => println!("  {text}"),
                _ => {}
            },
        )
        .map_err(|e| e.to_string())?;
    }
    let mut set = Set::load(dir).map_err(|e| e.to_string())?;
    let id = film::add_zone(
        dir,
        &mut set,
        &trial,
        Some((from - start, to - start)),
        kind,
    )
    .map_err(|e| e.to_string())?;
    println!("{id}");
    Ok(())
}

fn cascade(
    dir: &Path,
    catalogue: &Catalogue,
    add: &[String],
    mode: ModeArg,
    remove: Option<&str>,
) -> Result<(), String> {
    let mut set = Set::load(dir).map_err(|e| e.to_string())?;
    if let [first, second] = add {
        for key in [first, second] {
            catalogue.get(key).map_err(|e| e.to_string())?;
        }
        let cascade = set::Cascade {
            first: first.clone(),
            second: second.clone(),
            mode: match mode {
                ModeArg::Rest => set::CascadeMode::Rest,
                ModeArg::Voice => set::CascadeMode::Voice,
            },
            kept: false,
        };
        let key = cascade.key();
        if !set.add_cascade(cascade) {
            return Err(format!("the set already has `{key}`"));
        }
        set.save(dir).map_err(|e| e.to_string())?;
        println!("{key} added; `separate` runs it where `{first}` has its outputs");
        return Ok(());
    }
    if let Some(key) = remove {
        if !set.remove_cascade(key) {
            return Err(format!("no cascade `{key}` in the set"));
        }
        set.save(dir).map_err(|e| e.to_string())?;
        println!("{key} removed; its outputs stay where they are");
        return Ok(());
    }
    let mut out = std::io::stdout().lock();
    for cascade in &set.cascades {
        let key = cascade.key();
        let done = set
            .zones
            .iter()
            .filter(|zone| omnifrog_bench::layout::outputs_present(dir, zone, &key))
            .count();
        let _ = writeln!(
            out,
            "{key:<28} {done}/{} zone(s){}{}",
            set.zones.len(),
            if set.is_checked(&key) {
                ""
            } else {
                " · unchecked"
            },
            if cascade.kept { " · kept" } else { "" }
        );
    }
    if set.cascades.is_empty() {
        let _ = writeln!(
            out,
            "no cascade; `cascade --add FIRST SECOND [--mode voice]` makes one"
        );
    }
    Ok(())
}
