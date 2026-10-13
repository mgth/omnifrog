//! The model tab: the report the other way round. The left column lists
//! the models in place of the zones; one is chosen there, and each zone has
//! its line — where the model's mean comes from, and where it fails.

use crate::app::App;
use crate::view_report::{count, hint, metrics};
use egui::{RichText, Ui};
use omnifrog_bench::report::Column;
use omnifrog_bench::set::{Reason, Side};

/// The film's name is cut to this: the numbers stay in the window whatever
/// a film is called. The whole of it is on the left, and under the pointer.
const FILM_WIDTH: f32 = 260.0;

pub fn ui(app: &mut App, ui: &mut Ui) {
    ui.add_space(4.0);
    controls(app, ui);
    ui.add_space(4.0);
    table(app, ui);
}

/// The left column under this tab: the models of the set, in the
/// catalogue's order, where the other tabs have the zones.
pub fn list(app: &mut App, ui: &mut Ui) {
    ui.add_space(4.0);
    ui.weak(format!("{} model(s)", app.model.keys.len()));
    ui.add_space(2.0);
    let mut chosen = None;
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for key in &app.model.keys {
                let text = format!("{}\n{key}", app.name(key));
                let line = ui.selectable_label(app.model.key.as_ref() == Some(key), text);
                if line.clicked() {
                    chosen = Some(key.clone());
                }
            }
        });
    if let Some(key) = chosen {
        app.show_model(&key);
    }
}

/// The side, shared with the report, and the name of the model shown: one
/// line, there whether a model is or not.
fn controls(app: &mut App, ui: &mut Ui) {
    ui.horizontal(|ui| {
        let mut changed = false;
        for side in Side::BOTH {
            let text = match side {
                Side::Vo => "VO — what is left of the element",
                Side::Vf => "VF — the voice of the channel",
            };
            changed |= ui
                .selectable_value(&mut app.report.side, side, text)
                .changed();
        }
        if changed {
            app.refresh_report();
        }
        ui.separator();
        match &app.model.key {
            Some(key) => {
                ui.strong(app.name(key));
                ui.monospace(key);
            }
            None => {
                ui.weak("Nothing rated or measured yet.");
            }
        }
    });
}

/// The report's columns, less the one that counts zones: here a line is one.
const COLUMNS: [Column; 12] = [
    Column::Model,
    Column::SiSdr,
    Column::Residue,
    Column::Loss,
    Column::Artifacts,
    Column::Gap,
    Column::Excess,
    Column::Lack,
    Column::OtherSiSdr,
    Column::Refs,
    Column::Rejects,
    Column::Seconds,
];

fn table(app: &mut App, ui: &mut Ui) {
    let side = app.report.side;
    let (sorted, descending) = app.model.sort[side as usize];
    let mut sort = None;
    let mut choose = None;
    egui::ScrollArea::both()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            egui::Grid::new(("model", side as usize))
                .striped(true)
                .num_columns(COLUMNS.len() + 2)
                .spacing([14.0, 4.0])
                .show(ui, |ui| {
                    for column in COLUMNS {
                        let mut title = match (column, side) {
                            (Column::Model, _) => "Zone".to_string(),
                            (Column::OtherSiSdr, Side::Vo) => "Voice".to_string(),
                            (Column::OtherSiSdr, Side::Vf) => "Rest".to_string(),
                            (Column::Rejects, _) => "🗙".to_string(),
                            _ => column.label().to_string(),
                        };
                        if column == sorted {
                            title.push_str(if descending { " ⏷" } else { " ⏶" });
                        }
                        let header = ui
                            .add(egui::Button::new(RichText::new(title).strong()).frame(false))
                            .on_hover_text(zone_hint(column, side));
                        if header.clicked() {
                            sort = Some(column);
                        }
                        if column == Column::Model {
                            ui.strong("Range");
                            ui.strong("Kind");
                        }
                    }
                    ui.end_row();
                    if app.model.key.is_some() {
                        mean(app, side, ui);
                        ui.end_row();
                    }
                    for row in &app.model.rows {
                        let place = app.set.zones.iter().position(|z| z.id == row.zone);
                        let Some((place, zone)) = place.map(|i| (i, &app.set.zones[i])) else {
                            continue;
                        };
                        let film = egui::Button::selectable(
                            app.selected == Some(place),
                            zone.film.as_str(),
                        )
                        .truncate();
                        let (rect, _) = ui.allocate_exact_size(
                            egui::vec2(FILM_WIDTH, ui.spacing().interact_size.y),
                            egui::Sense::hover(),
                        );
                        let line = ui
                            .new_child(
                                egui::UiBuilder::new()
                                    .max_rect(rect)
                                    .layout(egui::Layout::left_to_right(egui::Align::Center)),
                            )
                            .add(film)
                            .on_hover_text(format!(
                                "{}\nClick to choose this zone: the other tabs show it.",
                                zone.label()
                            ));
                        if line.clicked() {
                            choose = Some(place);
                        }
                        ui.label(format!(
                            "object {} / channel {} · {}",
                            zone.object,
                            zone.channel,
                            zone.range()
                        ));
                        ui.label(&zone.kind);
                        let report = &row.report;
                        metrics(ui, report, side);
                        match report.refs[side as usize] {
                            0 => ui.weak("·"),
                            _ => ui.label("★"),
                        };
                        match report.rejects[side as usize] {
                            0 => ui.weak("·"),
                            _ => ui.label(
                                Reason::ALL
                                    .iter()
                                    .zip(report.reasons[side as usize])
                                    .find(|(_, n)| *n > 0)
                                    .map_or("🗙", |(reason, _)| reason.label()),
                            ),
                        };
                        seconds(ui, report.seconds);
                        ui.end_row();
                    }
                });
        });
    if let Some(column) = sort {
        app.sort_model_by(column);
    }
    if choose.is_some() {
        app.select(choose);
    }
}

/// The model over all the zones, first and whatever the sort: its line in
/// the report, to read each zone against.
fn mean(app: &App, side: Side, ui: &mut Ui) {
    let mean = &app.model.mean;
    ui.strong("Mean");
    ui.weak(format!("{} zone(s)", mean.mean(side.listened()).n))
        .on_hover_text("Zones the means are over.");
    ui.label("");
    metrics(ui, mean, side);
    count(ui, mean.refs[side as usize]);
    let rejects = count(ui, mean.rejects[side as usize]);
    let reasons: Vec<String> = Reason::ALL
        .iter()
        .zip(mean.reasons[side as usize])
        .filter(|(_, n)| *n > 0)
        .map(|(reason, n)| format!("{n} for {}", reason.label()))
        .collect();
    if !reasons.is_empty() {
        rejects.on_hover_text(reasons.join(", "));
    }
    seconds(ui, mean.seconds);
}

fn seconds(ui: &mut Ui, seconds: Option<f64>) {
    match seconds {
        Some(seconds) => ui.monospace(format!("{seconds:.1} s")),
        None => ui.weak("—"),
    };
}

fn zone_hint(column: Column, side: Side) -> &'static str {
    match column {
        Column::Model => "The zones' own order: the set's.",
        Column::Refs => "Whether it is one of the zone's references.",
        Column::Rejects => "Whether it is rejected on the zone, and what for.",
        Column::Seconds => "The time its outputs took on the zone.",
        _ => hint(column, side),
    }
}
