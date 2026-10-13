//! The setup tab: the device and what suits it, what is installed, the
//! two models, the folders — and one button that does all of it.

use crate::app::App;
use egui::{Color32, RichText, Ui};
use omnifrog_core::config::{Config, Engine};
use omnifrog_core::doctor::Status;
use omnifrog_core::models::Role;
use omnifrog_core::setup::Gpu;
use omnifrog_core::tools::Tool;
use std::path::PathBuf;

const GREEN: Color32 = Color32::from_rgb(90, 190, 110);
const AMBER: Color32 = Color32::from_rgb(220, 170, 60);
const RED: Color32 = Color32::from_rgb(230, 90, 80);

pub fn ui(app: &mut App, ui: &mut Ui) {
    ui.add_space(6.0);
    if let Some(error) = &app.config_error {
        ui.colored_label(RED, format!("Configuration: {error}"));
    }
    device(app, ui);
    ui.add_space(10.0);
    automatic(app, ui);
    ui.add_space(12.0);
    checks(app, ui);
    ui.add_space(12.0);
    models(app, ui);
    ui.add_space(12.0);
    folders(app, ui);
    ui.add_space(8.0);
    ui.weak(format!("Configuration file: {}", Config::path().display()));
}

fn device(app: &mut App, ui: &mut Ui) {
    ui.heading("This machine");
    let Some((device, advice)) = app.device.clone() else {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("Looking at the graphics card…");
        });
        return;
    };
    egui::Grid::new("device")
        .num_columns(2)
        .spacing([14.0, 6.0])
        .show(ui, |ui| {
            ui.label("Device");
            ui.label(RichText::new(device.describe()).strong());
            ui.end_row();
            ui.label("Suits it");
            ui.vertical(|ui| {
                ui.label(advice.inference.describe());
                ui.small(RichText::new(&advice.reason).weak());
            });
            ui.end_row();
            ui.label("Configured");
            ui.horizontal(|ui| {
                ui.label(app.config.inference.describe());
                if app.config.inference != advice.inference
                    && ui
                        .button("Use what suits the device")
                        .on_hover_text(
                            "an estimate from one card; `omnifrog bench-vram` measures this one",
                        )
                        .clicked()
                {
                    app.config.inference = advice.inference;
                    app.save_config();
                }
            });
            ui.end_row();
        });
    let note = match device.gpu {
        Gpu::Cpu if cfg!(windows) => Some(
            "No NVIDIA card: torch has no build for other cards on Windows, so the processor runs \
             the models — many hours a film.",
        ),
        Gpu::Cpu => {
            Some("No GPU torch can use: the processor runs the models — many hours a film.")
        }
        Gpu::Mps => {
            Some("Apple silicon: the models run through Metal (mps), a first for this program.")
        }
        _ => None,
    };
    if let Some(note) = note {
        ui.colored_label(AMBER, note);
    }
    if app.config.inference.engine == Engine::Msst {
        ui.small(
            RichText::new("MSST: the least memory, about three times slower than pymss.").weak(),
        );
    }
}

fn automatic(app: &mut App, ui: &mut Ui) {
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal(|ui| {
            let enabled = !app.busy() && app.device.is_some();
            if ui
                .add_enabled(
                    enabled,
                    egui::Button::new(RichText::new("⚙  Set up automatically").size(15.0))
                        .min_size(egui::vec2(200.0, 30.0)),
                )
                .clicked()
            {
                app.auto_setup(false);
            }
            ui.vertical(|ui| {
                let gpu = app
                    .device
                    .as_ref()
                    .map(|(d, _)| d.gpu.label())
                    .unwrap_or("…");
                ui.label(format!(
                    "Builds the Python environment with torch for {gpu} and MSST, downloads the \
                     two models, and sizes the inference for the card. Several gigabytes; what \
                     is already there is kept."
                ));
                if ui
                    .add_enabled(
                        enabled,
                        egui::Button::new("Rebuild the environment").small(),
                    )
                    .on_hover_text("remove the venv and MSST and install them again")
                    .clicked()
                {
                    app.auto_setup(true);
                }
            });
        });
    });
}

fn checks(app: &mut App, ui: &mut Ui) {
    ui.horizontal(|ui| {
        ui.heading("What is installed");
        if app.checking {
            ui.spinner();
        } else if ui.small_button("Check again").clicked() {
            app.recheck();
        }
    });
    let Some(checks) = app.checks.clone() else {
        return;
    };
    let mut browse: Option<Tool> = None;
    let mut fetch: Option<Vec<String>> = None;
    // Rows rather than a grid: a grid gives a wrapped column the width of
    // its narrowest cell, and a path then breaks at every slash.
    let busy = app.busy();
    for check in checks.iter().filter(|c| c.name != "device") {
        let (mark, color) = match check.status {
            Status::Ok => ("✔", GREEN),
            Status::Warning => ("?", AMBER),
            Status::Missing => ("✖", RED),
        };
        ui.horizontal_top(|ui| {
            let row = ui.spacing().interact_size.y;
            ui.add_sized(
                [16.0, row],
                egui::Label::new(RichText::new(mark).color(color)),
            );
            ui.allocate_ui_with_layout(
                egui::vec2(92.0, row),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    ui.set_width(92.0);
                    ui.label(&check.name);
                },
            );
            let detail = (ui.available_width() - 96.0).max(160.0);
            ui.allocate_ui_with_layout(
                egui::vec2(detail, row),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    ui.set_width(detail);
                    ui.add_space(3.0);
                    ui.add(egui::Label::new(RichText::new(&check.detail).small()).wrap());
                },
            );
            let tool = Tool::here().find(|t| t.name() == check.name);
            if let Some(tool) = tool {
                if ui
                    .small_button("Choose…")
                    .on_hover_text(format!("where {} is", tool.name()))
                    .clicked()
                {
                    browse = Some(tool);
                }
            } else if check.name.starts_with("model")
                && check.status != Status::Ok
                && ui
                    .add_enabled(!busy, egui::Button::new("Download").small())
                    .clicked()
            {
                fetch = Some(app.models_to_fetch());
            }
        });
    }
    if let Some(tool) = browse
        && let Some(path) = rfd::FileDialog::new().pick_file()
    {
        *tool_slot(&mut app.config, tool) = Some(path);
        app.save_config();
    }
    if let Some(keys) = fetch
        && !keys.is_empty()
    {
        app.fetch(keys);
    }
    let missing_tools: Vec<Tool> = checks
        .iter()
        .filter(|c| c.status == Status::Missing)
        .filter_map(|c| Tool::here().find(|t| t.name() == c.name))
        .collect();
    if !missing_tools.is_empty() {
        ui.add_space(4.0);
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(RichText::new("Where to get them").strong());
            for tool in missing_tools {
                ui.label(format!("{}: {}", tool.name(), where_to_get(tool)));
            }
            if cfg!(windows) {
                ui.small(
                    RichText::new(
                        "A program in a tools folder beside omnifrog-gui.exe is found without \
                         being chosen.",
                    )
                    .weak(),
                );
            }
        });
    }
}

fn models(app: &mut App, ui: &mut Ui) {
    ui.heading("Models");
    let Some(catalogue) = app.catalogue.clone() else {
        ui.colored_label(RED, "The model catalogue could not be read.");
        return;
    };
    let mut changed = false;
    // What is on disk is read against the configuration as it was.
    let config = app.config.clone();
    egui::Grid::new("models")
        .num_columns(2)
        .spacing([14.0, 6.0])
        .show(ui, |ui| {
            for (label, hint, key, preferred) in [
                (
                    "Original",
                    "takes the dialogue out of the original's elements",
                    &mut app.config.models.vo,
                    Role::Objects,
                ),
                (
                    "Dub",
                    "takes the dialogue out of the dubbed track",
                    &mut app.config.models.vf,
                    Role::Voices,
                ),
            ] {
                ui.label(label).on_hover_text(hint);
                let current = catalogue
                    .get(key)
                    .map(|m| m.name.clone())
                    .unwrap_or_else(|_| format!("{key} (not in the catalogue)"));
                egui::ComboBox::from_id_salt(label)
                    .width(420.0)
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        // The models trained for this side first; the role is
                        // a hint, not a rule.
                        let mut list: Vec<_> = catalogue.visible().collect();
                        list.sort_by_key(|m| {
                            (m.role != preferred && m.role != Role::Both, m.name.clone())
                        });
                        for model in list {
                            let on_disk = if model.local(&config).all_present() {
                                "  ✔"
                            } else {
                                ""
                            };
                            let text =
                                format!("{}  ·  {}{on_disk}", model.name, model.role.label());
                            if ui
                                .selectable_label(*key == model.key, text)
                                .on_hover_text(&model.notes)
                                .clicked()
                                && *key != model.key
                            {
                                *key = model.key.clone();
                                changed = true;
                            }
                        }
                    });
                ui.end_row();
            }
        });
    if changed {
        app.save_config();
    }
    let to_fetch = app.models_to_fetch();
    if !to_fetch.is_empty() {
        ui.horizontal(|ui| {
            ui.colored_label(AMBER, format!("Not on disk: {}", to_fetch.join(", ")));
            if ui
                .add_enabled(!app.busy(), egui::Button::new("Download"))
                .clicked()
            {
                app.fetch(to_fetch);
            }
        });
    }
}

fn folders(app: &mut App, ui: &mut Ui) {
    ui.heading("Folders");
    let mut changed = false;
    egui::Grid::new("folders")
        .num_columns(3)
        .spacing([14.0, 6.0])
        .show(ui, |ui| {
            let work = app.config.work_dir();
            let models = app.config.models_dir();
            for (label, hint, current, slot) in [
                (
                    "Work",
                    "the jobs' temporaries: 30 to 45 GB a film",
                    work,
                    &mut app.config.work_dir,
                ),
                (
                    "Models",
                    "checkpoints and their configurations",
                    models,
                    &mut app.config.models_dir,
                ),
            ] {
                ui.label(label).on_hover_text(hint);
                ui.label(current.display().to_string());
                ui.horizontal(|ui| {
                    if ui.small_button("Choose…").clicked()
                        && let Some(dir) =
                            rfd::FileDialog::new().set_directory(&current).pick_folder()
                    {
                        *slot = Some(dir);
                        changed = true;
                    }
                    if slot.is_some() && ui.small_button("default").clicked() {
                        *slot = None;
                        changed = true;
                    }
                });
                ui.end_row();
            }
            ui.label("Keep");
            ui.checkbox(
                &mut app.config.keep_work,
                "keep a job's temporaries after it succeeds",
            )
            .changed()
            .then(|| changed = true);
            ui.end_row();
        });
    if changed {
        app.save_config();
    }
}

fn tool_slot(config: &mut Config, tool: Tool) -> &mut Option<PathBuf> {
    let t = &mut config.tools;
    match tool {
        Tool::Ffmpeg => &mut t.ffmpeg,
        Tool::Ffprobe => &mut t.ffprobe,
        Tool::Mkvmerge => &mut t.mkvmerge,
        Tool::Harletty => &mut t.harletty,
        Tool::Harlettizer => &mut t.harlettizer,
        Tool::Dee => &mut t.dee,
        Tool::Wine => &mut t.wine,
    }
}

fn where_to_get(tool: Tool) -> &'static str {
    let windows = cfg!(windows);
    let mac = cfg!(target_os = "macos");
    match tool {
        Tool::Ffmpeg | Tool::Ffprobe if windows => {
            "the \"release essentials\" build from gyan.dev/ffmpeg (ffmpeg.exe and ffprobe.exe)"
        }
        Tool::Ffmpeg | Tool::Ffprobe if mac => "brew install ffmpeg",
        Tool::Ffmpeg | Tool::Ffprobe => "your distribution's ffmpeg package",
        Tool::Mkvmerge if windows => "MKVToolNix, from mkvtoolnix.download",
        Tool::Mkvmerge if mac => "brew install mkvtoolnix",
        Tool::Mkvmerge => "your distribution's mkvtoolnix package",
        Tool::Harletty => {
            "the harletty-cli zip for this system from a harletty-bridge release, \
             github.com/harletty/harletty-bridge"
        }
        Tool::Harlettizer => {
            "the harlettizer release for this system, github.com/harletty/harlettizer"
        }
        Tool::Dee => "only for the Dee encoder: your own copy of it",
        Tool::Wine => "only for the Dee encoder on Linux: your distribution's wine package",
    }
}
