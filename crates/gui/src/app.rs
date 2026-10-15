//! The window's state, and what loads in the background.

use crate::run::{Kind, Run};
use crate::task::{self, Request};
use egui::Ui;
use omnifrog_core::config::{Config, Inference};
use omnifrog_core::doctor::{self, Check, Status};
use omnifrog_core::models::Catalogue;
use omnifrog_core::probe::{self, Media};
use omnifrog_core::setup::{self, Device, Recommendation};
use omnifrog_core::{pipeline, progress};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Film,
    Setup,
}

/// What a background thread hands back.
pub enum Loaded {
    Device(Device, Recommendation),
    Checks(Vec<Check>),
    Probe {
        path: PathBuf,
        result: Result<Media, String>,
        chosen: Option<(u32, u32)>,
    },
}

/// What the film tab asks of a run, beside the tracks. The format and its
/// options are the configuration's, kept from one session to the next.
pub struct DubForm {
    pub mkv: bool,
    pub trial: bool,
    pub seconds: f64,
    /// Where in the film a trial starts, in seconds.
    pub start: f64,
    pub gain_db: f64,
    pub cancel_duplicates: bool,
    pub out_dir: Option<PathBuf>,
    pub restart: bool,
}

pub struct App {
    pub tab: Tab,
    pub config: Config,
    pub config_error: Option<String>,
    pub catalogue: Option<Catalogue>,
    pub device: Option<(Device, Recommendation)>,
    pub checks: Option<Vec<Check>>,
    pub checking: bool,
    /// Why the last file chosen for a tool was refused.
    pub tool_refused: Option<String>,

    pub source: Option<PathBuf>,
    pub media: Option<Result<Media, String>>,
    pub probing: bool,
    pub original: Option<u32>,
    pub dub: Option<u32>,
    pub form: DubForm,

    pub run: Option<Run>,
    loaded_tx: Sender<Loaded>,
    loaded_rx: Receiver<Loaded>,
    ctx: egui::Context,
}

impl App {
    pub fn new(ctx: &egui::Context, film: Option<PathBuf>) -> Self {
        let (config, config_error) = match Config::load() {
            Ok(config) => (config, None),
            Err(e) => (Config::default(), Some(e.to_string())),
        };
        let catalogue = Catalogue::load().ok();
        let (loaded_tx, loaded_rx) = channel();
        let form = DubForm {
            mkv: false,
            trial: false,
            seconds: 300.0,
            start: 0.0,
            gain_db: 0.0,
            cancel_duplicates: config.cancel_duplicates,
            out_dir: None,
            restart: false,
        };
        let mut app = Self {
            tab: Tab::Film,
            config,
            config_error,
            catalogue,
            device: None,
            checks: None,
            checking: false,
            tool_refused: None,
            source: None,
            media: None,
            probing: false,
            original: None,
            dub: None,
            form,
            run: None,
            loaded_tx,
            loaded_rx,
            ctx: ctx.clone(),
        };
        app.detect_device();
        app.recheck();
        if let Some(film) = film {
            app.open(film);
        }
        app
    }

    /// Run `work` on a thread of its own and hand its result to the window.
    fn background(&self, work: impl FnOnce() -> Loaded + Send + 'static) {
        let tx = self.loaded_tx.clone();
        let ctx = self.ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(work());
            ctx.request_repaint();
        });
    }

    fn detect_device(&self) {
        self.background(|| {
            let device = setup::device();
            let advice = setup::recommend(&device);
            Loaded::Device(device, advice)
        });
    }

    /// Run every check again; they start the tools and the venv's Python,
    /// which takes a second or two.
    pub fn recheck(&mut self) {
        if self.checking {
            return;
        }
        self.checking = true;
        let config = self.config.clone();
        self.background(move || Loaded::Checks(doctor::run(&config)));
    }

    /// Read the film's tracks, and which two a run would take.
    pub fn open(&mut self, path: PathBuf) {
        self.source = Some(path.clone());
        self.media = None;
        self.original = None;
        self.dub = None;
        self.probing = true;
        let config = self.config.clone();
        self.background(move || {
            let result = probe::media(&config, &path, &progress::quiet).map_err(|e| e.to_string());
            let chosen = result.as_ref().ok().and_then(|media| {
                let options = pipeline::Options {
                    source: path.clone(),
                    ..Default::default()
                };
                pipeline::choose(media, &options)
                    .ok()
                    .map(|(o, d)| (o.index, d.index))
            });
            Loaded::Probe {
                path,
                result,
                chosen,
            }
        });
    }

    /// Write the configuration, and check again what depends on it.
    pub fn save_config(&mut self) {
        match self.config.save() {
            Ok(()) => self.config_error = None,
            Err(e) => self.config_error = Some(e.to_string()),
        }
        self.catalogue = Catalogue::load().ok();
        self.recheck();
    }

    /// Everything a run needs is there.
    pub fn ready(&self) -> bool {
        self.checks
            .as_ref()
            .is_some_and(|checks| !checks.iter().any(|c| c.status == Status::Missing))
    }

    /// The models that are not all on disk, among the two selected.
    pub fn models_to_fetch(&self) -> Vec<String> {
        let Some(catalogue) = &self.catalogue else {
            return Vec::new();
        };
        let mut keys = Vec::new();
        for key in [&self.config.models.vo, &self.config.models.vf] {
            if let Ok(model) = catalogue.get(key)
                && !model.local(&self.config).all_present()
                && !keys.contains(key)
            {
                keys.push(key.clone());
            }
        }
        keys
    }

    pub fn busy(&self) -> bool {
        self.run.as_ref().is_some_and(Run::active)
    }

    /// The whole machine made ready: the recommended inference settings
    /// (unless the configuration already has its own), the environment for
    /// the detected device, the two models.
    pub fn auto_setup(&mut self, force: bool) {
        let Some((device, advice)) = &self.device else {
            return;
        };
        let inference = (self.config.inference == Inference::default()
            || self.config.inference == advice.inference)
            .then_some(advice.inference);
        let request = Request::Setup {
            gpu: device.gpu.label().to_string(),
            force,
            inference,
            fetch: vec![self.config.models.vo.clone(), self.config.models.vf.clone()],
        };
        self.start(Kind::Setup, "Setting up", request);
    }

    pub fn fetch(&mut self, keys: Vec<String>) {
        let title = format!("Downloading {}", keys.join(", "));
        self.start(Kind::Fetch, title, Request::Fetch { keys });
    }

    pub fn dub(&mut self) {
        let Some(source) = self.source.clone() else {
            return;
        };
        // The worker reads the configuration from disk.
        self.save_config();
        let request = Request::Dub(task::Dub {
            source: source.clone(),
            original: self.original,
            dub: self.dub,
            seconds: self.form.trial.then_some(self.form.seconds),
            start: (self.form.trial && self.form.start > 0.0).then_some(self.form.start),
            mkv: self.form.mkv && !self.form.trial,
            encoder: Some(self.config.encoder),
            gain_db: self.form.gain_db,
            cancel_duplicates: Some(self.form.cancel_duplicates),
            out_dir: self.form.out_dir.clone(),
            restart: self.form.restart,
        });
        let name = source
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        self.start(Kind::Dub, name, request);
    }

    fn start(&mut self, kind: Kind, title: impl Into<String>, request: Request) {
        if self.busy() {
            return;
        }
        self.run = Some(Run::start(kind, title, &request, &self.ctx));
    }

    fn drain(&mut self) {
        while let Ok(loaded) = self.loaded_rx.try_recv() {
            match loaded {
                Loaded::Device(device, advice) => self.device = Some((device, advice)),
                Loaded::Checks(checks) => {
                    // A first launch on a machine not set up opens where
                    // that is done, unless a film was asked for.
                    let first = self.checks.is_none();
                    self.checks = Some(checks);
                    self.checking = false;
                    if first && !self.ready() && self.source.is_none() {
                        self.tab = Tab::Setup;
                    }
                }
                Loaded::Probe {
                    path,
                    result,
                    chosen,
                } => {
                    // A film opened since supersedes this one.
                    if self.source.as_ref() != Some(&path) {
                        continue;
                    }
                    self.probing = false;
                    if let Some((original, dub)) = chosen {
                        self.original = Some(original);
                        self.dub = Some(dub);
                    }
                    self.media = Some(result);
                }
            }
        }
        if let Some(run) = &mut self.run
            && run.pump()
            && matches!(run.kind, Kind::Setup | Kind::Fetch)
        {
            // setup writes the venv and MSST into the configuration.
            if let Ok(config) = Config::load() {
                self.config = config;
            }
            self.catalogue = Catalogue::load().ok();
            self.recheck();
        }
    }

    fn top_bar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.heading("omnifrog");
            ui.add_space(12.0);
            ui.selectable_value(&mut self.tab, Tab::Film, "Film");
            let setup_label = if self.checks.is_some() && !self.ready() {
                "Setup ⚠"
            } else {
                "Setup"
            };
            ui.selectable_value(&mut self.tab, Tab::Setup, setup_label);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let Some((device, _)) = &self.device {
                    ui.weak(device.describe());
                } else {
                    ui.spinner();
                }
            });
        });
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        self.drain();
        let dropped: Vec<PathBuf> = ui.ctx().input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf())
                .filter(|p| !p.as_os_str().is_empty())
                .collect()
        });
        if let Some(path) = dropped.into_iter().next()
            && !self.busy()
        {
            self.tab = Tab::Film;
            self.open(path);
        }

        egui::Panel::top("top-bar").show(ui, |ui| {
            ui.add_space(4.0);
            self.top_bar(ui);
            ui.add_space(2.0);
        });
        if let Some(run) = &mut self.run {
            egui::Panel::bottom("run")
                .resizable(true)
                .default_size(170.0)
                .min_size(90.0)
                .show(ui, |ui| {
                    ui.add_space(6.0);
                    run.ui(ui);
                    ui.add_space(4.0);
                });
            // Time moves while the worker is quiet: the clock and the
            // estimate have to be redrawn anyway.
            if run.active() {
                ui.ctx()
                    .request_repaint_after(std::time::Duration::from_millis(500));
            }
        }
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| match self.tab {
                    Tab::Film => crate::view_film::ui(self, ui),
                    Tab::Setup => crate::view_setup::ui(self, ui),
                });
        });
    }
}
