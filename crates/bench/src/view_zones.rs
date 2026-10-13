//! The left column: the zones of the set.

use crate::app::App;
use egui::Ui;

pub fn list(app: &mut App, ui: &mut Ui) {
    ui.add_space(4.0);
    ui.weak(format!("{} zone(s)", app.set.zones.len()));
    ui.add_space(2.0);
    let mut chosen = None;
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for (i, zone) in app.set.zones.iter().enumerate() {
                let text = format!(
                    "{}\nobject {} / channel {} · {}",
                    zone.film,
                    zone.object,
                    zone.channel,
                    zone.range()
                );
                let line = ui
                    .selectable_label(app.selected == Some(i), text)
                    .on_hover_text(&zone.id);
                if line.clicked() {
                    chosen = Some(i);
                }
            }
        });
    if chosen.is_some() {
        app.select(chosen);
    }
}
