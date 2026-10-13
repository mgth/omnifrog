//! The report tab: the models summed up over the zones, one side at a time.

use crate::app::App;
use egui::{Color32, RichText, Ui};
use omnifrog_bench::report::{Column, Report};
use omnifrog_bench::set::{Reason, Side};

const WARNING: Color32 = Color32::from_rgb(220, 170, 60);

pub fn ui(app: &mut App, ui: &mut Ui) {
    ui.add_space(4.0);
    controls(app, ui);
    warning(app, ui);
    ui.add_space(4.0);
    table(app, ui);
}

/// The side, the scope and the batch. One row of fixed height: the progress
/// bar has its place whether a batch runs or not.
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
        ui.separator();
        changed |= ui
            .selectable_value(&mut app.report.all_zones, true, "All zones")
            .changed();
        let one = ui.add_enabled(
            app.selected.is_some(),
            egui::Button::selectable(!app.report.all_zones, "The zone chosen"),
        );
        if one.clicked() && app.report.all_zones {
            app.report.all_zones = false;
            changed = true;
        }
        if changed {
            app.refresh_report();
        }
        ui.separator();
        match &app.job {
            None => {
                if ui
                    .button("Compute")
                    .on_hover_text(
                        "Separate what the checked models lack on these zones, then measure. \
                         One model at a time; it takes the device.",
                    )
                    .clicked()
                {
                    app.compute();
                }
                if ui
                    .button("Metrics only")
                    .on_hover_text(
                        "Measure the checked outputs again against the references. \
                         Nothing is separated.",
                    )
                    .clicked()
                {
                    app.measure();
                }
            }
            Some(job) => {
                // Stop takes the place of both, at their width: the bar
                // after them stays where it is.
                if ui
                    .add_enabled(
                        !job.stopping(),
                        egui::Button::new("Stop").min_size(egui::vec2(150.0, 0.0)),
                    )
                    .on_hover_text("Ends after the zone in hand; what was done is kept.")
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
        // Which variants Compute makes of every checked output, on the
        // zones that have a duplicates context.
        let mut variants = app.set.variants;
        ui.add_enabled_ui(app.job.is_none(), |ui| {
            ui.checkbox(&mut variants.cancel, "+cancel").on_hover_text(
                "The duplicates cancelled in the dub's voice, as a run cancels them; in the \
                 mix, the voice moved by the offset of the two tracks.",
            );
            ui.checkbox(&mut variants.cancel_step, "+cancel-step")
                .on_hover_text(
                    "The same cancellation; in the mix, the voice moved step by step along \
                     the map of the dub, with a 20 ms fade between two steps.",
                );
        });
        if variants != app.set.variants {
            app.set_variants(variants);
        }
    });
}

/// The single-reference warning, on a line that is there either way.
fn warning(app: &App, ui: &mut Ui) {
    let single = &app.report.single;
    let text = match single.len() {
        0 => " ".to_string(),
        1 => format!("⚠ A single ★ on {} ({})", single[0].0, single[0].1),
        n => format!("⚠ A single ★ on {n} zones"),
    };
    let line = ui.colored_label(WARNING, text);
    if !single.is_empty() {
        let zones: Vec<String> = single
            .iter()
            .map(|(zone, sides)| format!("• {zone} — {sides}"))
            .collect();
        line.on_hover_text(format!(
            "A reference made of one model: the measure there favours that model's family \
             rather than what the ear prefers. Add ★ from other families.\n\n{}",
            zones.join("\n")
        ));
    }
}

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
    let (sorted, descending) = app.report.sort[side as usize];
    let busy = app.job.is_some();
    let mut sort = None;
    let mut check = None;
    let mut open = None;
    egui::ScrollArea::both()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            egui::Grid::new(("report", side as usize))
                .striped(true)
                .num_columns(COLUMNS.len() + 3)
                .spacing([14.0, 4.0])
                .show(ui, |ui| {
                    ui.label("");
                    for column in COLUMNS {
                        let mut title = match (column, side) {
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
                            .on_hover_text(hint(column, side));
                        if header.clicked() {
                            sort = Some(column);
                        }
                        if column == Column::Model {
                            ui.strong("Key");
                        }
                    }
                    ui.strong("Zones");
                    ui.end_row();
                    for report in &app.report.rows {
                        let mut checked = app.set.is_checked(&report.key);
                        if ui
                            .add_enabled(!busy, egui::Checkbox::without_text(&mut checked))
                            .on_hover_text("Taken in the next batch.")
                            .changed()
                        {
                            check = Some((report.key.clone(), checked));
                        }
                        if row(app, report, side, ui) {
                            open = Some(report.key.clone());
                        }
                        ui.end_row();
                    }
                });
        });
    if let Some(column) = sort {
        app.sort_by(column);
    }
    if let Some((key, checked)) = check {
        app.check(&key, checked);
    }
    if let Some(key) = open {
        app.show_model(&key);
    }
}

/// One model's line. Whether its name was clicked: the model is then shown
/// zone by zone.
fn row(app: &App, report: &Report, side: Side, ui: &mut Ui) -> bool {
    let mean = report.mean(side.listened());
    let open = ui
        .add(egui::Button::new(app.name(&report.key)).frame(false))
        .on_hover_text("Show this model zone by zone.")
        .clicked();
    ui.monospace(&report.key);
    metrics(ui, report, side);
    count(ui, report.refs[side as usize]);
    let rejects = count(ui, report.rejects[side as usize]);
    if report.rejects[side as usize] > 0 {
        let reasons: Vec<String> = Reason::ALL
            .iter()
            .zip(report.reasons[side as usize])
            .filter(|(_, n)| *n > 0)
            .map(|(reason, n)| format!("{n} for {}", reason.label()))
            .collect();
        if !reasons.is_empty() {
            rejects.on_hover_text(reasons.join(", "));
        }
    }
    match report.seconds {
        Some(seconds) => ui.monospace(format!("{seconds:.1} s")),
        None => ui.weak("—"),
    };
    ui.monospace(format!("{}", mean.n))
        .on_hover_text("Zones the means are over.");
    open
}

/// The eight numbers of a line, from SI-SDR to the other stem's.
pub fn metrics(ui: &mut Ui, report: &Report, side: Side) {
    let mean = report.mean(side.listened());
    if report.only_reference(side) {
        ui.weak("ref.").on_hover_text(
            "The only reference wherever it was measured: never compared with itself.",
        );
    } else {
        number(ui, mean.si_sdr, 1);
    }
    number(ui, mean.residue, 1);
    number(ui, mean.loss, 1);
    number(ui, mean.artifacts, 1);
    number(ui, mean.gap, 2);
    number(ui, mean.excess, 2);
    number(ui, mean.lack, 2);
    number(ui, report.mean(side.listened().other()).si_sdr, 1);
}

fn number(ui: &mut Ui, value: Option<f64>, decimals: usize) {
    match value {
        Some(value) => ui.monospace(format!("{value:.decimals$}")),
        None => ui.weak("—"),
    };
}

pub fn count(ui: &mut Ui, n: usize) -> egui::Response {
    match n {
        0 => ui.weak("·"),
        n => ui.monospace(format!("{n}")),
    }
}

pub fn hint(column: Column, side: Side) -> &'static str {
    match (column, side) {
        (Column::Model, _) => "The catalogue's order.",
        (Column::SiSdr, _) => {
            "Signal to error against the reference, blind to gain; dB, higher is closer."
        }
        (Column::Residue, Side::Vo) => "Voice left in the rest; dB, lower is better.",
        (Column::Residue, Side::Vf) => "Rest left in the voice; dB, lower is better.",
        (Column::Loss, _) => "The part of the reference that is missing; dB, lower is better.",
        (Column::Artifacts, _) => "What is neither stem of the reference; dB, lower is better.",
        (Column::Gap, _) => "Gap to the reference by bands; dB, lower is closer.",
        (Column::Excess, _) => "The part of the gap where the output is louder: a leak.",
        (Column::Lack, _) => "The part of the gap where the output is quieter: a lack.",
        (Column::OtherSiSdr, Side::Vo) => "SI-SDR of the voice taken out of the element.",
        (Column::OtherSiSdr, Side::Vf) => "SI-SDR of what is left of the channel.",
        (Column::Refs, _) => "Zones where it is one of the references.",
        (Column::Rejects, _) => "Zones where it is rejected.",
        (Column::Seconds, _) => "Mean time its outputs took on a zone.",
    }
}
