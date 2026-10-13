//! The rate tab: every output of the zone chosen, to listen to side by side
//! and to rate.
//!
//! Nothing above the rows changes height: the transport and the top
//! waveform have their place whether anything is loaded or not, and the
//! clocks have a fixed width. One clicks down a column of rows in a series;
//! the row under the pointer must still be the one aimed at.

use crate::app::App;
use crate::listen::{HEIGHT, Kind, Loaded, Wave};
use egui::{Align, Color32, Layout, Rect, RichText, Sense, Stroke, Ui, pos2, vec2};
use omnifrog_bench::listing::Row;
use omnifrog_bench::set::{RATE, Reason, Side, Status, Stem};
use omnifrog_bench::wave;

const STAR: Color32 = Color32::from_rgb(240, 190, 60);
const REJECT: Color32 = Color32::from_rgb(230, 90, 80);
const EXCESS: Color32 = Color32::from_rgb(255, 59, 48);
const LACK: Color32 = Color32::from_rgb(255, 208, 0);
const LOOP_START: Color32 = Color32::from_rgb(80, 200, 120);
const LOOP_END: Color32 = Color32::from_rgb(230, 90, 80);
// The two marks, in glyphs egui's own fonts have.
const STAR_MARK: &str = "★";
const REJECT_MARK: &str = "🗙";

// The widths of a row's cells: the top of the tab is laid out on the same
// ones, so the top waveform sits exactly over the rows'.
const NAME: f32 = 230.0;
const MARK: f32 = 36.0;
const BUTTON: f32 = 58.0;
const GAP: f32 = 6.0;
const LEFT: f32 = NAME + 4.0 * MARK + 5.0 * BUTTON + 10.0 * GAP;

/// A cell of exactly this size, whatever goes in it: the rows stay in
/// columns, and the top waveform over the rows'.
fn cell<R>(ui: &mut Ui, size: egui::Vec2, layout: Layout, add: impl FnOnce(&mut Ui) -> R) -> R {
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect).layout(layout));
    add(&mut child)
}

/// What a click in the tab asks for; done once the tab is drawn, when
/// nothing borrows the app any more.
enum Ask {
    Toggle,
    Original(Side),
    Output(String, Kind),
    Blend(String, Stem),
    Rate(String, Side, Status),
    Seek(u64),
    Loop(Option<(u64, u64)>),
}

pub fn ui(app: &mut App, ui: &mut Ui) {
    let Some(zone) = app.selected.and_then(|i| app.set.zones.get(i)) else {
        ui.add_space(8.0);
        ui.weak(if app.set.zones.is_empty() {
            "No zone yet. `omnifrog-bench import <_JeuxEssai>` brings in a legacy test set."
        } else {
            "Choose a zone."
        });
        return;
    };
    ui.add_space(4.0);
    let mut remove = false;
    let (busy, removing) = (app.job.is_some(), app.removing_zone);
    ui.horizontal(|ui| {
        ui.heading(zone.label());
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let text = if removing {
                "Remove it for good?"
            } else {
                "Remove zone"
            };
            remove = ui
                .add_enabled(!busy, egui::Button::new(text))
                .on_hover_text(
                    "Remove this zone from the set, with its inputs, every output, its \
                     references and its measures. There is no way back.",
                )
                .clicked();
        });
    });
    ui.horizontal(|ui| {
        ui.label(&zone.kind);
        if let (Some(vo), Some(vf)) = (&zone.lang_vo, &zone.lang_vf) {
            ui.weak(format!("· VO {vo}, VF {vf}"));
        }
        for side in Side::BOTH {
            let rating = zone.rating(side);
            ui.weak(format!(
                "· {}: {} {STAR_MARK}, {} {REJECT_MARK}",
                side.label(),
                rating.refs.len(),
                rating.rejects.len()
            ));
        }
        if let Some(context) = &zone.duplicates {
            ui.weak(format!(
                "· duplicates: dub leads by {:.0} ms, {} step(s)",
                context.dub_lead_ms,
                context.steps.len()
            ));
        }
    });
    ui.add_space(4.0);
    if remove && removing {
        app.remove_zone();
        return;
    }
    if remove {
        app.removing_zone = true;
    }
    bench(app, ui, true);
}

/// The listening bench on the zone shown: the transport and the input's
/// waveform, then one row per output. `rating` is whether the rows can be
/// rated — a range tried on a film is not a zone of the set yet.
pub fn bench(app: &mut App, ui: &mut Ui, rating: bool) {
    let start = app.zone_shown().map_or(0, |zone| zone.start);
    let mut asks: Vec<Ask> = Vec::new();
    if ui.input(|i| i.key_pressed(egui::Key::Space)) && ui.memory(|m| m.focused().is_none()) {
        asks.push(Ask::Toggle);
    }
    ui.spacing_mut().item_spacing.x = GAP;
    let height = app.listen.height;
    let frames = app
        .listen
        .original
        .as_ref()
        .map_or(app.listen.player.len(), |w| w.frames) as u64;
    let position = app.listen.player.position();
    let looped = app.listen.player.looped();

    // The transport, over the cells of the rows; the input's waveform, over
    // theirs.
    ui.horizontal(|ui| {
        cell(
            ui,
            vec2(LEFT, height),
            Layout::left_to_right(Align::Center),
            |ui| {
                transport(app, ui, start, position, looped, &mut asks);
            },
        );
        top_wave(app, ui, height, frames, position, looped, &mut asks);
    });
    titles(ui);
    ui.separator();

    let busy = app.job.is_some() || !rating;
    let rows = app.rows.len();
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show_rows(ui, height, rows, |ui, range| {
            for row in &app.rows[range] {
                ui.horizontal(|ui| {
                    cell(
                        ui,
                        vec2(LEFT, height),
                        Layout::left_to_right(Align::Center),
                        |ui| {
                            cells(app, row, busy, ui, &mut asks);
                        },
                    );
                    let (rect, response) = ui.allocate_exact_size(
                        vec2(ui.available_width().max(80.0), height),
                        Sense::click(),
                    );
                    paint(
                        ui,
                        rect,
                        app.listen.waves.get(&row.key),
                        app.listen.two_colours,
                    );
                    marks(ui, rect, frames, position, looped);
                    if let Some(frame) = clicked(&response, rect, frames) {
                        asks.push(Ask::Seek(frame));
                    }
                });
            }
        });

    for ask in asks {
        match ask {
            Ask::Toggle => app.toggle_play(),
            Ask::Original(side) => app.listen_original(side),
            Ask::Output(key, kind) => app.listen_output(&key, kind),
            Ask::Blend(key, stem) => app.listen_blend(&key, stem),
            Ask::Rate(key, side, status) => app.rate(&key, side, status),
            Ask::Seek(frame) => {
                // A click outside the loop lets go of it.
                if looped.is_some_and(|(a, b)| frame < a || frame >= b) {
                    app.listen.player.set_loop(None);
                }
                app.listen.player.seek(frame);
            }
            Ask::Loop(range) => app.listen.player.set_loop(range),
        }
    }
    if app.listen.player.playing() {
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(33));
    }
}

/// Frames of the zone as a time of the film, to the hundredth: `12:03.45`.
fn clock(frames: u64) -> String {
    let hundredths = frames * 100 / RATE as u64;
    let (minutes, seconds, rest) = (hundredths / 6000, hundredths / 100 % 60, hundredths % 100);
    if minutes >= 60 {
        format!(
            "{}:{:02}:{seconds:02}.{rest:02}",
            minutes / 60,
            minutes % 60
        )
    } else {
        format!("{minutes}:{seconds:02}.{rest:02}")
    }
}

fn transport(
    app: &mut App,
    ui: &mut Ui,
    start: u64,
    position: u64,
    looped: Option<(u64, u64)>,
    asks: &mut Vec<Ask>,
) {
    let playing = app.listen.player.playing();
    if ui
        .add_sized([34.0, 26.0], egui::Button::selectable(playing, if playing { "⏸" } else { "▶" }))
        .on_hover_text("Play or pause (Space). Only this starts and stops: a click on an output changes what plays, not whether.")
        .clicked()
    {
        asks.push(Ask::Toggle);
    }
    // Clocks of a fixed width: the buttons after them do not move as the
    // time runs.
    let time = |ui: &mut Ui, text: String, colour: Color32, hint: &str| {
        ui.add_sized(
            [78.0, 20.0],
            egui::Label::new(RichText::new(text).monospace().color(colour)),
        )
        .on_hover_text(hint);
    };
    let plain = ui.visuals().text_color();
    time(
        ui,
        clock(start + position),
        plain,
        "Where the player is, in the film's own time.",
    );
    match looped {
        Some((a, b)) => {
            time(ui, clock(start + a), LOOP_START, "Start of the loop.");
            time(ui, clock(start + b), LOOP_END, "End of the loop.");
        }
        None => {
            time(
                ui,
                "—".to_string(),
                LOOP_START,
                "No loop: drag on the top waveform to set one.",
            );
            time(
                ui,
                "—".to_string(),
                LOOP_END,
                "No loop: drag on the top waveform to set one.",
            );
        }
    }
    if ui
        .add_enabled(looped.is_some(), egui::Button::new("Whole"))
        .on_hover_text("Let go of the loop: play the whole zone.")
        .clicked()
    {
        asks.push(Ask::Loop(None));
    }
    for side in Side::BOTH {
        let on = app.listen.loaded == Some(Loaded::Original(side));
        if ui
            .add(egui::Button::selectable(on, format!("▶ {}", side.label())))
            .on_hover_text(match side {
                Side::Vo => "The element of the original, before any model.",
                Side::Vf => "The channel of the dub, before any model.",
            })
            .clicked()
        {
            asks.push(Ask::Original(side));
        }
    }
    ui.checkbox(&mut app.listen.two_colours, "2 colours").on_hover_text(
        "Waveforms tinted by their gap to the reference: red where the output is louder (a leak), \
         yellow where it is quieter (a lack). Off: their sum, in red.",
    );
    if let Some(Ok(output)) = &app.listen.output {
        ui.weak(format!("{} Hz", output.rate))
            .on_hover_text(&output.name);
    }
}

/// The input's waveform. A click goes there, a drag sets the loop, a double
/// click lets go of it, the wheel sets the height of every waveform.
fn top_wave(
    app: &mut App,
    ui: &mut Ui,
    height: f32,
    frames: u64,
    position: u64,
    looped: Option<(u64, u64)>,
    asks: &mut Vec<Ask>,
) {
    let (rect, response) = ui.allocate_exact_size(
        vec2(ui.available_width().max(80.0), height),
        Sense::click_and_drag(),
    );
    paint(
        ui,
        rect,
        app.listen.original.as_ref(),
        app.listen.two_colours,
    );
    marks(ui, rect, frames, position, looped);
    let at = |x: f32| -> u64 {
        let share = ((x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
        (share as f64 * frames as f64) as u64
    };
    if frames > 0 {
        if response.drag_started() {
            app.listen.drag_from = response.interact_pointer_pos().map(|p| at(p.x));
        }
        if response.dragged()
            && let (Some(from), Some(pointer)) =
                (app.listen.drag_from, response.interact_pointer_pos())
        {
            let to = at(pointer.x);
            if from.abs_diff(to) > RATE as u64 / 50 {
                asks.push(Ask::Loop(Some((from.min(to), from.max(to)))));
            }
        }
        if response.drag_stopped() {
            app.listen.drag_from = None;
        }
        if response.double_clicked() {
            asks.push(Ask::Loop(None));
        } else if let Some(frame) = clicked(&response, rect, frames) {
            asks.push(Ask::Seek(frame));
        }
    }
    if response.hovered() {
        // The wheel's own events, one step each: the smoothed scroll would
        // go on stepping for several frames.
        let wheel: f32 = ui.input(|i| {
            i.events
                .iter()
                .filter_map(|event| match event {
                    egui::Event::MouseWheel { delta, .. } => Some(delta.y),
                    _ => None,
                })
                .sum()
        });
        if wheel != 0.0 {
            let (low, high, step) = HEIGHT;
            app.listen.height = (app.listen.height + step * wheel.signum()).clamp(low, high);
        }
    }
    response.on_hover_text(
        "Click: go there. Drag: loop over that range. Double click: the whole zone. \
         Wheel: the height of every waveform.",
    );
}

/// The frame a plain click on a waveform points at.
fn clicked(response: &egui::Response, rect: Rect, frames: u64) -> Option<u64> {
    if !response.clicked() || frames == 0 {
        return None;
    }
    let x = response.interact_pointer_pos()?.x;
    let share = ((x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
    Some(((share as f64 * frames as f64) as u64).min(frames - 1))
}

fn titles(ui: &mut Ui) {
    ui.horizontal(|ui| {
        let title = |ui: &mut Ui, width: f32, text: &str, hint: &str| {
            cell(
                ui,
                vec2(width, 18.0),
                Layout::left_to_right(Align::Center),
                |ui| {
                    ui.strong(text).on_hover_text(hint);
                },
            );
        };
        title(
            ui,
            NAME,
            "Model",
            "Under the name: the SI-SDR and the gap of the stem each side is judged on.",
        );
        for side in Side::BOTH {
            title(
                ui,
                2.0 * MARK + GAP,
                side.label(),
                "★: a reference of this side. 🗙: rejected, and why.",
            );
        }
        for kind in Kind::ALL {
            title(
                ui,
                BUTTON,
                kind.label(),
                "The loudness of the file, in LKFS. Click: play it from where the player is. \
                 Ctrl+click a rest and a voice, of two models if you like: hear them together.",
            );
        }
    });
}

/// The cells of a row: the model and its measures, its rating on each side,
/// its five files.
fn cells(app: &App, row: &Row, busy: bool, ui: &mut Ui, asks: &mut Vec<Ask>) {
    cell(
        ui,
        vec2(NAME, ui.available_height()),
        Layout::top_down(Align::Min),
        |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            let name = RichText::new(app.name(&row.key));
            let label =
                egui::Label::new(if row.foreign { name.italics() } else { name }).truncate();
            ui.add(label).on_hover_text(if row.foreign {
                format!("{} — not in the catalogue: never computed again.", row.key)
            } else {
                match placed(app, &row.key) {
                    Some(placed) => format!("{}\n{placed}", row.key),
                    None => row.key.clone(),
                }
            });
            ui.add(egui::Label::new(RichText::new(measures(row)).weak().small()).truncate());
        },
    );
    for side in Side::BOTH {
        let status = row.status(side);
        let reference = status == Status::Reference;
        let star = egui::Button::new(RichText::new(STAR_MARK).color(if reference {
            STAR
        } else {
            ui.visuals().weak_text_color()
        }));
        if ui
            .add_enabled(!busy, star.min_size(vec2(MARK, 24.0)))
            .on_hover_text(format!(
                "{}: one of the outputs the others are measured against.",
                side.label()
            ))
            .clicked()
        {
            let next = if reference {
                Status::Neutral
            } else {
                Status::Reference
            };
            asks.push(Ask::Rate(row.key.clone(), side, next));
        }
        let (text, colour) = match status {
            Status::Rejected(Some(reason)) => (
                format!("{REJECT_MARK}{}", reason.label()[..1].to_uppercase()),
                REJECT,
            ),
            Status::Rejected(None) => (REJECT_MARK.to_string(), REJECT),
            _ => (REJECT_MARK.to_string(), ui.visuals().weak_text_color()),
        };
        cell(
            ui,
            vec2(MARK, 24.0),
            Layout::left_to_right(Align::Center),
            |ui| {
                if busy {
                    ui.disable();
                }
                ui.menu_button(RichText::new(text).color(colour), |ui| {
                    for reason in Reason::ALL {
                        let (label, hint) = match reason {
                            Reason::Leak => ("Leak", "Some of the other signal is left."),
                            Reason::Lack => ("Lack", "The wanted signal is eaten into."),
                            Reason::Artifacts => ("Artifacts", "The sound is damaged."),
                        };
                        if ui.button(label).on_hover_text(hint).clicked() {
                            asks.push(Ask::Rate(
                                row.key.clone(),
                                side,
                                Status::Rejected(Some(reason)),
                            ));
                            ui.close();
                        }
                    }
                    if ui.button("Rejected, no reason").clicked() {
                        asks.push(Ask::Rate(row.key.clone(), side, Status::Rejected(None)));
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Not rejected").clicked() {
                        asks.push(Ask::Rate(row.key.clone(), side, Status::Neutral));
                        ui.close();
                    }
                });
            },
        );
    }
    for kind in Kind::ALL {
        let on = app.listen.loaded
            == Some(Loaded::Output {
                key: row.key.clone(),
                kind,
            });
        let blended = matches!(kind, Kind::Stem(stem) if app.listen.blended(&row.key, stem));
        let text = match kind.loudness(row) {
            Some(lkfs) => format!("{lkfs:.1}"),
            None => "▶".to_string(),
        };
        let text = if blended { format!("+ {text}") } else { text };
        let button = ui.add_enabled(
            kind.present(row),
            egui::Button::selectable(on || blended, RichText::new(text).monospace())
                .min_size(vec2(BUTTON, 24.0)),
        );
        if button.clicked() {
            let ctrl = ui.input(|i| i.modifiers.command);
            asks.push(match (ctrl, kind) {
                (true, Kind::Stem(stem)) => Ask::Blend(row.key.clone(), stem),
                _ => Ask::Output(row.key.clone(), kind),
            });
        }
    }
}

/// Where a variant put the dub's voice in its mix, for the zone shown.
fn placed(app: &App, key: &str) -> Option<String> {
    let (_, variant) = omnifrog_bench::keys::split(key);
    let zone = app.zone_shown()?;
    let context = zone.duplicates.as_ref()?;
    match variant? {
        omnifrog_bench::keys::CANCEL => Some(format!(
            "In its mix, the voice is moved by the offset of the two tracks: {:.0} ms.",
            context.dub_lead_ms
        )),
        _ => {
            let frames = zone.end.unwrap_or(zone.start).saturating_sub(zone.start) as i64;
            Some(format!(
                "In its mix, the voice is moved by the step of the dub's map: {}.",
                omnifrog_bench::variants::steps_over(&context.steps, frames)
            ))
        }
    }
}

/// The line under a model's name: on each side, the SI-SDR and the gap of
/// the stem that side is judged on.
fn measures(row: &Row) -> String {
    let Some(measure) = &row.measure else {
        return "not measured".to_string();
    };
    Side::BOTH
        .iter()
        .filter_map(|&side| {
            let metric = measure.stems.get(&side.listened())?;
            Some(match (metric.si_sdr, metric.gap) {
                (Some(si_sdr), Some(gap)) => {
                    format!("{} {si_sdr:.1} dB · gap {gap:.2}", side.label())
                }
                (Some(si_sdr), None) => format!("{} {si_sdr:.1} dB", side.label()),
                _ if metric.reference => format!("{} ref.", side.label()),
                _ => format!("{} —", side.label()),
            })
        })
        .collect::<Vec<_>>()
        .join("   ")
}

fn blend(from: Color32, to: Color32, share: f32) -> Color32 {
    let mix = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * share) as u8;
    Color32::from_rgb(
        mix(from.r(), to.r()),
        mix(from.g(), to.g()),
        mix(from.b(), to.b()),
    )
}

/// A waveform: one line per pixel column from the lowest to the highest
/// sample it stands for, tinted by the gap to the reference there.
fn paint(ui: &Ui, rect: Rect, wave: Option<&Wave>, two_colours: bool) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    let Some(wave) = wave else {
        return;
    };
    let base = ui.visuals().weak_text_color();
    let width = rect.width().max(1.0) as usize;
    let (middle, half) = (rect.center().y, rect.height() / 2.0 - 1.0);
    for column in 0..width {
        let (low, high) = wave::span(&wave.envelope, column, width);
        let colour = match &wave.gaps {
            Some(gaps) => {
                let from = (column * wave.frames / width) as u64;
                let to = ((column + 1) * wave.frames / width) as u64;
                let (excess, lack) = gaps.max_over(from, to);
                let tint = wave::tint(excess, lack, two_colours);
                blend(base, blend(LACK, EXCESS, tint.excess), tint.strength)
            }
            None => base,
        };
        let x = rect.left() + column as f32 + 0.5;
        let top = middle - high.clamp(-1.0, 1.0) * half;
        let bottom = middle - low.clamp(-1.0, 1.0) * half;
        painter.line_segment(
            [pos2(x, top.min(bottom - 1.0)), pos2(x, bottom)],
            Stroke::new(1.0, colour),
        );
    }
}

/// The loop's two ends and where the player is, over a waveform.
fn marks(ui: &Ui, rect: Rect, frames: u64, position: u64, looped: Option<(u64, u64)>) {
    if frames == 0 {
        return;
    }
    let painter = ui.painter_at(rect);
    let x = |frame: u64| rect.left() + rect.width() * (frame as f64 / frames as f64) as f32;
    if let Some((a, b)) = looped {
        painter.vline(x(a), rect.y_range(), Stroke::new(1.0, LOOP_START));
        painter.vline(x(b), rect.y_range(), Stroke::new(1.0, LOOP_END));
    }
    painter.vline(
        x(position.min(frames)),
        rect.y_range(),
        Stroke::new(1.0, ui.visuals().strong_text_color()),
    );
}
