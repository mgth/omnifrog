//! The film tab: a film of the bench, a range of it — one element of its
//! original, one channel of its dub — every model run on that range, and
//! zones cut out of it.
//!
//! Three rows of controls that are always there, enabled or not, then the
//! same listening bench as the rate tab: what one clicks does not move as
//! a film is chosen or a range opened.

use crate::app::App;
use egui::Ui;
use omnifrog_bench::set::{KINDS, clock};

pub fn ui(app: &mut App, ui: &mut Ui) {
    ui.add_space(4.0);
    films(app, ui);
    range(app, ui);
    actions(app, ui);
    ui.add_space(2.0);
    ui.separator();
    if app.film.trial.is_some() {
        crate::view_rate::bench(app, ui, false);
    } else {
        ui.add_space(8.0);
        ui.weak(if app.film.films.is_empty() {
            "No film in the bench yet. Choose one: it is extracted whole, once — every element of \
             its original and every channel of its dub, tens of gigabytes — and kept for the zones \
             cut from it."
        } else {
            "Open a range: an element of the original, a channel of the dub, two times. A few \
             tens of seconds around the passage: the models hear all of it, and the zone is cut \
             from what they made of it."
        });
    }
}

/// The films of the bench, and the way in for another.
fn films(app: &mut App, ui: &mut Ui) {
    let busy = app.job.is_some();
    ui.horizontal(|ui| {
        if ui
            .add_enabled(!busy, egui::Button::new("Extract a film…"))
            .on_hover_text(
                "Extract a film whole into the bench: the tracks a run would take. \
                 Several minutes, and tens of gigabytes.",
            )
            .clicked()
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("Films", &["mkv", "mka", "m2ts", "mp4", "mk3d"])
                .add_filter("All files", &["*"])
                .pick_file()
        {
            app.extract(path);
        }
        let chosen = app.film.chosen.and_then(|i| app.film.films.get(i));
        let label = chosen.map_or("no film".to_string(), |film| film.name.clone());
        let mut choose = None;
        ui.add_enabled_ui(!busy && !app.film.films.is_empty(), |ui| {
            egui::ComboBox::from_id_salt("film")
                .width(420.0)
                .selected_text(label)
                .show_ui(ui, |ui| {
                    for (i, film) in app.film.films.iter().enumerate() {
                        if ui
                            .selectable_label(app.film.chosen == Some(i), &film.name)
                            .clicked()
                        {
                            choose = Some(i);
                        }
                    }
                });
        });
        if choose.is_some() {
            app.choose_film(choose);
        }
        let Some(film) = app.film.chosen.and_then(|i| app.film.films.get(i)) else {
            return;
        };
        let size = app.film.chosen.and_then(|i| app.film.sizes.get(i)).copied();
        ui.weak(format!(
            "{} · {} element(s) · tracks {} and {} · {:.1} GB",
            clock(film.frames),
            film.elements.len(),
            film.original,
            film.dub,
            size.unwrap_or(0) as f64 / 1e9
        ))
        .on_hover_text(&film.source);
        let text = if app.film.removing {
            "Remove it for good?"
        } else {
            "Remove"
        };
        if ui
            .add_enabled(!busy, egui::Button::new(text))
            .on_hover_text(
                "Remove the film's extraction and the ranges tried on it. \
                 The zones cut from it are their own files, and stay.",
            )
            .clicked()
        {
            if app.film.removing {
                app.remove_film();
            } else {
                app.film.removing = true;
            }
        }
    });
}

/// The element, the channel and the two times of the range.
fn range(app: &mut App, ui: &mut Ui) {
    let busy = app.job.is_some();
    let film = app.film.chosen.and_then(|i| app.film.films.get(i)).cloned();
    ui.horizontal(|ui| {
        ui.add_enabled_ui(film.is_some() && !busy, |ui| {
            ui.label("Element");
            let name = |id: u32| {
                film.iter()
                    .flat_map(|f| &f.elements)
                    .find(|e| e.id == id)
                    .map_or(id.to_string(), |e| match e.object {
                        true => format!("{} · object", e.id),
                        false => format!("{} · {}", e.id, e.name),
                    })
            };
            egui::ComboBox::from_id_salt("element")
                .width(130.0)
                .selected_text(app.film.object.map_or("—".to_string(), name))
                .show_ui(ui, |ui| {
                    for element in film.iter().flat_map(|f| &f.elements) {
                        ui.selectable_value(
                            &mut app.film.object,
                            Some(element.id),
                            name(element.id),
                        );
                    }
                });
            ui.label("Channel");
            egui::ComboBox::from_id_salt("channel")
                .width(70.0)
                .selected_text(app.film.channel.clone().unwrap_or("—".to_string()))
                .show_ui(ui, |ui| {
                    for channel in film.iter().flat_map(|f| &f.channels) {
                        ui.selectable_value(&mut app.film.channel, Some(channel.clone()), channel);
                    }
                });
            ui.label("From");
            let from = ui.add(
                egui::TextEdit::singleline(&mut app.film.from)
                    .desired_width(80.0)
                    .hint_text("12:02"),
            );
            ui.label("to");
            let to = ui.add(
                egui::TextEdit::singleline(&mut app.film.to)
                    .desired_width(80.0)
                    .hint_text("12:40"),
            );
            let enter = (from.lost_focus() || to.lost_focus())
                && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if ui
                .button("Open")
                .on_hover_text("Cut this range out of the film and listen to it.")
                .clicked()
                || enter
            {
                app.open_trial();
            }
        });
        if let Some(trial) = &app.film.trial {
            ui.weak(format!(
                "open: object {} / channel {} · {}",
                trial.object,
                trial.channel,
                trial.range()
            ));
        }
    });
}

/// What is done with the range opened: every model run on it, and a zone
/// made of it.
fn actions(app: &mut App, ui: &mut Ui) {
    let open = app.film.trial.is_some();
    ui.horizontal(|ui| {
        match &app.job {
            None => {
                if ui
                    .add_enabled(
                        open,
                        egui::Button::new("Separate all").min_size(egui::vec2(110.0, 0.0)),
                    )
                    .on_hover_text(
                        "Run every model checked in the report on this range; the ones that \
                         already have their outputs here are left. One model at a time; it \
                         takes the device.",
                    )
                    .clicked()
                {
                    app.separate_trial();
                }
            }
            Some(job) => {
                if ui
                    .add_enabled(
                        !job.stopping(),
                        egui::Button::new("Stop").min_size(egui::vec2(110.0, 0.0)),
                    )
                    .on_hover_text(
                        "Ends after the model in hand; what was done is kept. \
                         An extraction is not stopped.",
                    )
                    .clicked()
                {
                    job.stop();
                }
            }
        }
        let progress = app.job.as_ref().map_or(0.0, |job| job.progress);
        ui.add(
            egui::ProgressBar::new(progress)
                .desired_width(180.0)
                .show_percentage(),
        );
        ui.separator();
        ui.add_enabled_ui(open && app.job.is_none(), |ui| {
            ui.label("Kind");
            egui::ComboBox::from_id_salt("kind")
                .width(190.0)
                .selected_text(KINDS[app.film.kind.min(KINDS.len() - 1)])
                .show_ui(ui, |ui| {
                    for (i, kind) in KINDS.iter().enumerate() {
                        ui.selectable_value(&mut app.film.kind, i, *kind);
                    }
                });
            let looped = app.listen.player.looped();
            if ui
                .button("Add zone")
                .on_hover_text(
                    "Make a zone of the loop — drag on the top waveform to set it — or of the \
                     whole range when there is none. Its inputs and every output here are cut \
                     to it; it is rated under Rate.",
                )
                .clicked()
            {
                app.add_zone();
            }
            let start = app.film.trial.as_ref().map_or(0, |trial| trial.start);
            ui.weak(match looped {
                Some((a, b)) => format!("the loop: {} – {}", clock(start + a), clock(start + b)),
                None => "the whole range".to_string(),
            });
        });
    });
}
