//! The film tab: the file, its two tracks, the run's options, Start.

use crate::app::{App, Tab};
use crate::run::clock;
use egui::{Color32, RichText, Ui};
use omnifrog_core::config::{Encoder, IAMF_BITS, IamfCodec};
use std::time::Duration;

pub fn ui(app: &mut App, ui: &mut Ui) {
    ui.add_space(6.0);
    if let Some(error) = &app.config_error {
        ui.colored_label(
            Color32::from_rgb(230, 90, 80),
            format!("Configuration: {error}"),
        );
    }
    if let Some(checks) = &app.checks
        && !app.ready()
    {
        let missing = checks
            .iter()
            .filter(|c| c.status == omnifrog_core::doctor::Status::Missing)
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(Color32::from_rgb(220, 170, 60), "⚠ Not ready to run:");
                ui.label(format!("missing {missing}."));
                if ui.button("Open Setup").clicked() {
                    app.tab = Tab::Setup;
                }
            });
        });
        ui.add_space(6.0);
    }

    ui.horizontal(|ui| {
        let busy = app.busy();
        if ui
            .add_enabled(!busy, egui::Button::new("Choose a film…"))
            .clicked()
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("Films", &["mkv", "mka", "m2ts", "mp4", "mk3d"])
                .add_filter("All files", &["*"])
                .pick_file()
        {
            app.open(path);
        }
        match &app.source {
            Some(path) => {
                ui.label(RichText::new(path.display().to_string()).strong());
            }
            None => {
                ui.weak("or drop one on the window");
            }
        }
    });
    ui.add_space(6.0);

    if app.probing {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("Reading the tracks…");
        });
        return;
    }
    let Some(media) = &app.media else {
        return;
    };
    let media = match media {
        Ok(media) => media.clone(),
        Err(error) => {
            ui.colored_label(Color32::from_rgb(230, 90, 80), error);
            return;
        }
    };

    ui.weak(format!(
        "{}{} audio tracks · {} subtitles · {}",
        media
            .title
            .as_deref()
            .map(|t| format!("{t} · "))
            .unwrap_or_default(),
        media.audio.len(),
        media.subtitles,
        if media.duration > 0.0 {
            clock(Duration::from_secs_f64(media.duration))
        } else {
            "duration unknown".into()
        }
    ));
    ui.add_space(4.0);

    ui.heading("Tracks");
    ui.label(
        "The original is the spatial track whose dialogue is replaced; the dub is the channel \
         track the new dialogue is taken from.",
    );
    ui.add_space(4.0);
    egui::Grid::new("tracks")
        .striped(true)
        .num_columns(4)
        .spacing([14.0, 6.0])
        .show(ui, |ui| {
            ui.strong("Original");
            ui.strong("Dub");
            ui.strong("#");
            ui.strong("Track");
            ui.end_row();
            for track in &media.audio {
                let spatial = track.is_spatial();
                ui.add_enabled_ui(spatial, |ui| {
                    ui.radio_value(&mut app.original, Some(track.index), "")
                        .on_disabled_hover_text("no scene in this track: nothing to re-voice");
                });
                ui.add_enabled_ui(!spatial, |ui| {
                    ui.radio_value(&mut app.dub, Some(track.index), "")
                        .on_disabled_hover_text("a scene: the dub is taken from a channel track");
                });
                ui.label(track.index.to_string());
                ui.vertical(|ui| {
                    let summary = track.summary();
                    let text = RichText::new(summary.trim_end()).monospace();
                    if Some(track.index) == app.original || Some(track.index) == app.dub {
                        ui.label(text.strong());
                    } else {
                        ui.label(text);
                    }
                    if let Some(why) = &track.probe_error {
                        ui.small(
                            RichText::new(format!("the decoder could not read it: {why}"))
                                .color(Color32::from_rgb(220, 170, 60)),
                        );
                    }
                });
                ui.end_row();
            }
        });
    if media.audio.iter().all(|t| !t.is_spatial()) {
        ui.colored_label(
            Color32::from_rgb(230, 90, 80),
            "No spatial track (TrueHD Atmos, E-AC-3 JOC, DTS:X, Auro-3D) in this file.",
        );
    }

    ui.add_space(10.0);
    ui.heading("Output");
    let form = &mut app.form;
    let config = &mut app.config;
    let mut save = false;
    egui::Grid::new("options")
        .num_columns(2)
        .spacing([14.0, 8.0])
        .show(ui, |ui| {
            ui.label("Write");
            ui.horizontal(|ui| {
                ui.radio_value(&mut form.mkv, false, ".mka — the new track alone");
                ui.add_enabled_ui(!form.trial, |ui| {
                    ui.radio_value(&mut form.mkv, true, ".mkv — the whole film with it")
                        .on_disabled_hover_text("a trial writes an .mka to listen to");
                });
            });
            ui.end_row();

            ui.label("Trial");
            ui.horizontal(|ui| {
                if ui.checkbox(&mut form.trial, "only").changed() && form.trial {
                    form.mkv = false;
                }
                ui.add_enabled(
                    form.trial,
                    egui::DragValue::new(&mut form.seconds)
                        .range(10.0..=36_000.0)
                        .speed(5.0)
                        .suffix(" s"),
                );
                ui.add_enabled_ui(form.trial, |ui| {
                    ui.label("from");
                    let end = if media.duration > 0.0 {
                        media.duration
                    } else {
                        36_000.0
                    };
                    ui.add(
                        egui::DragValue::new(&mut form.start)
                            .range(0.0..=end)
                            .speed(1.0)
                            .custom_formatter(|s, _| time_of_day(s))
                            .custom_parser(parse_time),
                    )
                    .on_hover_text(
                        "where in the film the trial starts, as h:mm:ss; it starts at the \
                         original's first keyframe there, a tenth of a second later at most",
                    );
                    if form.start > 0.0 && ui.small_button("beginning").clicked() {
                        form.start = 0.0;
                    }
                });
            });
            ui.end_row();

            ui.label("Format");
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    let before = config.encoder;
                    ui.radio_value(&mut config.encoder, Encoder::Hz, "TrueHD · harlettizer")
                        .on_hover_text(
                            "TrueHD Atmos that keeps the original's elements and pans the \
                             new voices onto them",
                        );
                    ui.radio_value(&mut config.encoder, Encoder::Dee, "TrueHD · Dee")
                        .on_hover_text(
                            "TrueHD Atmos by Dee, an encoder you supply, which re-clusters \
                             the whole scene",
                        );
                    // One choice for IAMF, which starts as the overlay;
                    // what becomes of the voices is chosen under it.
                    let iamf = ui.radio(config.encoder.is_iamf(), "IAMF").on_hover_text(
                        "the open format browsers and televisions decode, by harlettizer: \
                             the master's elements as IAMF objects, no TrueHD anywhere",
                    );
                    if iamf.clicked() && !config.encoder.is_iamf() {
                        config.encoder = Encoder::Iamf;
                    }
                    save |= config.encoder != before;
                });
                if config.encoder.is_iamf() {
                    ui.horizontal(|ui| {
                        let before = config.encoder;
                        ui.label("Voices");
                        ui.radio_value(
                            &mut config.encoder,
                            Encoder::Iamf,
                            "panned onto the objects",
                        )
                        .on_hover_text(
                            "the same objects as the TrueHD overlay, the voices projected \
                                 onto them the same way, under the same guards",
                        );
                        ui.radio_value(&mut config.encoder, Encoder::IamfBed, "on the bed")
                            .on_hover_text(
                                "the voices put on the fixed bed, the original's other objects \
                                 left independent; written beside the other, as …-iamf-lit.mka",
                            );
                        save |= config.encoder != before;
                    });
                    ui.horizontal(|ui| {
                        let before = config.iamf_codec;
                        ui.label("Codec");
                        ui.radio_value(&mut config.iamf_codec, IamfCodec::Flac, "FLAC")
                            .on_hover_text("lossless, about half the size of the samples");
                        let bits_before = config.iamf_bits;
                        ui.add_enabled_ui(config.iamf_codec == IamfCodec::Flac, |ui| {
                            egui::ComboBox::from_id_salt("flac-bits")
                                .selected_text(bits_label(config.iamf_bits))
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut config.iamf_bits,
                                        None,
                                        bits_label(None),
                                    )
                                    .on_hover_text(
                                        "measured on the original: 16 for a master at 16 bits or \
                                         fewer, 24 otherwise; FLAC drops the bits no sample sets",
                                    );
                                    for bits in IAMF_BITS {
                                        ui.selectable_value(
                                            &mut config.iamf_bits,
                                            Some(bits),
                                            bits_label(Some(bits)),
                                        );
                                    }
                                })
                                .response
                                .on_disabled_hover_text("FLAC only: Opus has no depth");
                        });
                        save |= config.iamf_bits != bits_before;
                        ui.radio_value(&mut config.iamf_codec, IamfCodec::Opus, "Opus")
                            .on_hover_text(
                                "lossy, what a browser or a streaming service is sent; the \
                                 file is named for its bitrate, …-iamf-opus64.mka",
                            );
                        save |= config.iamf_codec != before;
                        let opus = config.iamf_codec == IamfCodec::Opus;
                        let rate = ui
                            .add_enabled(
                                opus,
                                egui::DragValue::new(&mut config.iamf_bitrate)
                                    .range(6..=256)
                                    .speed(1.0)
                                    .suffix(" kbit/s a channel"),
                            )
                            .on_hover_text(
                                "a coupled pair is coded at twice it and the LFE at a quarter: \
                                 64 puts a 7.1.4 bed at about 720 kbit/s; YouTube's IAMF runs \
                                 at about 46",
                            )
                            .on_disabled_hover_text("Opus only: FLAC is lossless");
                        save |= rate.drag_stopped() || rate.lost_focus();
                    });
                    ui.horizontal(|ui| {
                        ui.label("Options");
                        let edit = ui
                            .add(
                                egui::TextEdit::singleline(&mut config.iamf_opts)
                                    .hint_text("harlettizer's defaults")
                                    .desired_width(320.0),
                            )
                            .on_hover_text(
                                "passed to `harlettizer iamf` as they are, after the codec: \
                                 --bits 16|24 for FLAC; for the overlay, its guards as for \
                                 TrueHD (--overlay-cost 0, …). A --codec or --bitrate here \
                                 replaces the choice above",
                            );
                        save |= edit.lost_focus();
                    });
                }
            });
            ui.end_row();

            ui.label("Voices");
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut form.gain_db)
                        .range(-12.0..=12.0)
                        .speed(0.1)
                        .fixed_decimals(1)
                        .suffix(" dB"),
                )
                .on_hover_text("on top of the levelling against the original's dialogue");
                ui.checkbox(&mut form.cancel_duplicates, "cancel the duplicates")
                    .on_hover_text(
                        "move the dub's voices onto the original's clock and take out the music \
                         and effects they still carry",
                    );
            });
            ui.end_row();

            ui.label("Into");
            ui.horizontal(|ui| {
                match &form.out_dir {
                    Some(dir) => {
                        ui.label(dir.display().to_string());
                        if ui.small_button("beside the film").clicked() {
                            form.out_dir = None;
                        }
                    }
                    None => {
                        ui.label("beside the film");
                    }
                }
                if ui.small_button("Choose…").clicked()
                    && let Some(dir) = rfd::FileDialog::new().pick_folder()
                {
                    form.out_dir = Some(dir);
                }
            });
            ui.end_row();

            ui.label("");
            ui.checkbox(
                &mut form.restart,
                "start over, whatever an earlier run left",
            )
            .on_hover_text(
                "otherwise a run picks up at the first step not done with these settings",
            );
            ui.end_row();
        });

    if save {
        app.save_config();
    }

    ui.add_space(12.0);
    let why_not = if app.busy() {
        Some("a task is running")
    } else if !app.ready() {
        Some("the setup is not complete")
    } else if app.original.is_none() || app.dub.is_none() {
        Some("choose the original and the dub")
    } else {
        None
    };
    let start = ui.add_enabled(
        why_not.is_none(),
        egui::Button::new(RichText::new("▶  Start").size(16.0)).min_size(egui::vec2(140.0, 32.0)),
    );
    if let Some(why) = why_not {
        start.on_disabled_hover_text(why);
    } else if start.clicked() {
        app.dub();
    }
}

/// What the FLAC depth menu says for a choice.
fn bits_label(bits: Option<u32>) -> String {
    match bits {
        None => "the source's depth".to_string(),
        Some(bits) => format!("{bits} bits"),
    }
}

/// Seconds into the film as `h:mm:ss`.
fn time_of_day(seconds: f64) -> String {
    let s = seconds.max(0.0).round() as u64;
    format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

/// `h:mm:ss`, `mm:ss` or seconds.
fn parse_time(text: &str) -> Option<f64> {
    let parts: Vec<&str> = text.trim().split(':').collect();
    if parts.len() > 3 {
        return None;
    }
    let mut seconds = 0.0;
    for part in parts {
        let value: f64 = part.trim().parse().ok()?;
        if value < 0.0 {
            return None;
        }
        seconds = seconds * 60.0 + value;
    }
    Some(seconds)
}
