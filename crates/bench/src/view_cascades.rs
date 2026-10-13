//! The cascades tab: two models one after the other.
//!
//! The form is on top and keeps its place; the list is under it.

use crate::app::App;
use egui::Ui;
use omnifrog_bench::layout;
use omnifrog_bench::set::CascadeMode;

pub fn ui(app: &mut App, ui: &mut Ui) {
    let busy = app.job.is_some();
    ui.add_space(4.0);
    ui.weak(
        "The first model separates; the second works one of its stems again, and what it finds \
         there changes side — nothing is lost, it is moved. On the rest, against a leak: the \
         voice still in it joins the voice. On the voice, against a lack: the rest taken with it \
         goes back to the rest.",
    );
    ui.add_space(4.0);
    let models: Vec<(String, String)> = app
        .catalogue
        .iter()
        .flat_map(|c| c.visible())
        .map(|m| (m.key.clone(), m.name.clone()))
        .collect();
    ui.add_enabled_ui(!busy, |ui| {
        ui.horizontal(|ui| {
            let choose = |ui: &mut Ui, id: &str, chosen: &mut Option<String>| {
                let text = chosen
                    .as_ref()
                    .and_then(|key| models.iter().find(|(k, _)| k == key))
                    .map_or("a model".to_string(), |(_, name)| name.clone());
                egui::ComboBox::from_id_salt(id)
                    .width(280.0)
                    .selected_text(text)
                    .show_ui(ui, |ui| {
                        for (key, name) in &models {
                            ui.selectable_value(chosen, Some(key.clone()), name);
                        }
                    });
            };
            choose(ui, "first", &mut app.cascade.0);
            ui.label("then");
            choose(ui, "second", &mut app.cascade.1);
            ui.label("on its");
            ui.selectable_value(&mut app.cascade.2, CascadeMode::Rest, "rest");
            ui.selectable_value(&mut app.cascade.2, CascadeMode::Voice, "voice");
            if ui
                .button("Add")
                .on_hover_text(
                    "Add this cascade to the set. Compute, in the report, runs it on the zones \
                     where its first model has its outputs.",
                )
                .clicked()
            {
                app.add_cascade();
            }
        });
    });
    ui.add_space(4.0);
    ui.separator();

    let mut remove = None;
    let mut keep = None;
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if app.set.cascades.is_empty() {
                ui.weak("No cascade yet.");
                return;
            }
            egui::Grid::new("cascades")
                .striped(true)
                .num_columns(5)
                .spacing([14.0, 4.0])
                .show(ui, |ui| {
                    for title in ["Cascade", "Key", "Zones", "Kept", ""] {
                        ui.strong(title);
                    }
                    ui.end_row();
                    for cascade in &app.set.cascades {
                        let key = cascade.key();
                        ui.label(app.name(&key));
                        ui.monospace(&key);
                        let done = app
                            .set
                            .zones
                            .iter()
                            .filter(|zone| layout::outputs_present(&app.dir, zone, &key))
                            .count();
                        ui.monospace(format!("{done}/{}", app.set.zones.len()))
                            .on_hover_text("Zones it has its outputs on.");
                        let mut kept = cascade.kept;
                        if ui
                            .add_enabled(!busy, egui::Checkbox::without_text(&mut kept))
                            .on_hover_text(
                                "A note that this one is worth keeping; nothing acts on it.",
                            )
                            .changed()
                        {
                            keep = Some((key.clone(), kept));
                        }
                        if ui
                            .add_enabled(!busy, egui::Button::new("Remove"))
                            .on_hover_text("Take it out of the set. Its outputs stay.")
                            .clicked()
                        {
                            remove = Some(key.clone());
                        }
                        ui.end_row();
                    }
                });
        });
    if let Some((key, kept)) = keep {
        app.keep_cascade(&key, kept);
    }
    if let Some(key) = remove {
        app.remove_cascade(&key);
    }
}
