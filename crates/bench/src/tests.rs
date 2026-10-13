//! The window without a screen: a synthetic zone, the tabs drawn for real,
//! and what a click does asked of the app directly. Nothing here opens an
//! audio device.

use crate::app::{App, Tab};
use crate::listen::{Kind, Loaded};
use crate::output::Choice;
use crate::prefs::Prefs;
use omnifrog_bench::film::{self, Element, Film};
use omnifrog_bench::outputs::{self, Split};
use omnifrog_bench::report::Column;
use omnifrog_bench::set::{RATE, Reason, Set, Side, Status, Stem, Zone};
use omnifrog_bench::{audio, layout};
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            0.25 * ((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
        })
        .collect()
}

/// A zone of one second with two complete outputs and a third that only
/// has the dub's side.
fn bench(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("omnifrog-bench-app-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let zone = Zone {
        id: "zone".into(),
        film: "Film".into(),
        object: 3,
        channel: "C".into(),
        start: 60 * RATE as u64,
        end: Some(61 * RATE as u64),
        kind: "Clean dialogue".into(),
        ..Zone::default()
    };
    let n = RATE as usize;
    for side in Side::BOTH {
        audio::write(&layout::extract(&dir, &zone, side), &noise(1, n), RATE).unwrap();
    }
    for (key, seed, stems) in [
        ("one", 10, &Stem::ALL[..]),
        ("two", 20, &Stem::ALL[..]),
        ("half", 30, &[Stem::VoiceVf, Stem::RestVf][..]),
    ] {
        for (i, stem) in stems.iter().enumerate() {
            let samples = noise(seed + i as u64, n);
            audio::write(&layout::output(&dir, &zone, key, *stem), &samples, RATE).unwrap();
        }
        audio::write(&layout::mix(&dir, &zone, key), &noise(seed + 9, n), RATE).unwrap();
    }
    let mut set = Set::default();
    set.zones.push(zone);
    set.save(&dir).unwrap();
    dir
}

fn frame(ctx: &egui::Context, app: &mut App) {
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1500.0, 900.0),
        )),
        ..Default::default()
    };
    // No renderer to hand the font texture to.
    let mut output = ctx.run_ui(input, |ui| app.draw(ui));
    output.textures_delta.clear();
}

/// Draw until the waveforms asked for have come back from their thread.
fn settle(ctx: &egui::Context, app: &mut App) {
    let until = Instant::now() + Duration::from_secs(10);
    loop {
        frame(ctx, app);
        if !app.listen.waves.is_empty() || Instant::now() > until {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn both_tabs_draw_a_zone_and_an_empty_bench() {
    let dir = bench("draw");
    let ctx = egui::Context::default();
    let mut app = App::new(&ctx, dir.clone(), Tab::Rate);
    assert_eq!(app.selected, Some(0));
    assert_eq!(app.rows.len(), 3);
    settle(&ctx, &mut app);
    // What is left of the element is shown before anything is played; the
    // output without that stem has no waveform.
    assert_eq!(app.listen.kind, Some(Kind::Stem(Stem::RestVo)));
    assert_eq!(app.listen.waves.len(), 2);
    assert!(app.listen.original.is_some());
    frame(&ctx, &mut app);
    app.tab = Tab::Report;
    frame(&ctx, &mut app);
    // Nothing was played: no device was opened.
    assert!(app.listen.output.is_none());

    let empty = dir.join("nothing-here");
    let mut app = App::new(&ctx, empty, Tab::Rate);
    frame(&ctx, &mut app);
    app.tab = Tab::Report;
    frame(&ctx, &mut app);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_click_loads_a_file_and_keeps_the_position() {
    let dir = bench("load");
    let ctx = egui::Context::default();
    let mut app = App::new(&ctx, dir.clone(), Tab::Rate);
    app.listen.player.seek(1234);
    app.listen_output("one", Kind::Stem(Stem::VoiceVf));
    assert_eq!(
        app.listen.loaded,
        Some(Loaded::Output {
            key: "one".into(),
            kind: Kind::Stem(Stem::VoiceVf)
        })
    );
    assert_eq!(app.listen.player.len(), RATE as usize);
    assert_eq!(app.listen.player.position(), 1234);
    assert!(!app.listen.player.playing());
    // The rows now show that kind of file: all three have it.
    settle(&ctx, &mut app);
    assert_eq!(app.listen.kind, Some(Kind::Stem(Stem::VoiceVf)));
    assert_eq!(app.listen.waves.len(), 3);

    app.listen_original(Side::Vf);
    assert_eq!(app.listen.loaded, Some(Loaded::Original(Side::Vf)));
    assert_eq!(app.listen.player.position(), 1234);
    // A file that is not there says so and changes nothing.
    app.listen_output("half", Kind::Stem(Stem::RestVo));
    assert_eq!(app.listen.loaded, Some(Loaded::Original(Side::Vf)));
    assert!(!app.status.is_empty());

    // Another zone, or none: the player lets go.
    app.select(None);
    assert_eq!(app.listen.loaded, None);
    assert!(app.listen.player.is_empty());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_rest_and_a_voice_play_together() {
    let dir = bench("blend");
    let ctx = egui::Context::default();
    let mut app = App::new(&ctx, dir.clone(), Tab::Rate);
    // The file playing joins the blend it is the other half of.
    app.listen_output("one", Kind::Stem(Stem::RestVo));
    app.listen_blend("two", Stem::VoiceVf);
    assert_eq!(app.listen.loaded, Some(Loaded::Blend));
    assert!(app.listen.blended("one", Stem::RestVo));
    assert!(app.listen.blended("two", Stem::VoiceVf));
    // Another voice takes the place of the first.
    app.listen_blend("half", Stem::VoiceVf);
    assert!(!app.listen.blended("two", Stem::VoiceVf));
    assert_eq!(app.listen.loaded, Some(Loaded::Blend));
    // Taken out again, the rest plays alone, still chosen.
    app.listen_blend("half", Stem::VoiceVf);
    assert_eq!(
        app.listen.loaded,
        Some(Loaded::Output {
            key: "one".into(),
            kind: Kind::Stem(Stem::RestVo)
        })
    );
    assert!(app.listen.blended("one", Stem::RestVo));
    // A plain click forgets the blend.
    app.listen_output("two", Kind::Mix);
    assert!(app.listen.blend.iter().all(Option::is_none));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_rating_is_kept_at_once() {
    let dir = bench("rate");
    let ctx = egui::Context::default();
    let mut app = App::new(&ctx, dir.clone(), Tab::Rate);
    app.rate("one", Side::Vf, Status::Reference);
    let one = app.rows.iter().find(|row| row.key == "one").unwrap();
    assert_eq!(one.status(Side::Vf), Status::Reference);
    assert!(app.status.contains("measure it again"));
    let zone = app.set.zones[0].clone();
    assert!(layout::reference_present(&dir, &zone, Side::Vf));
    let rejected = Status::Rejected(Some(Reason::Lack));
    app.rate("two", Side::Vf, rejected);
    assert!(app.status.is_empty());
    // A reference needs its files: said, and nothing changed.
    app.rate("half", Side::Vo, Status::Reference);
    assert!(app.status.contains("no VO outputs"));
    let saved = Set::load(&dir).unwrap();
    assert_eq!(saved.zones[0].vf.refs, ["one"]);
    assert_eq!(saved.zones[0].vf.status("two"), rejected);
    assert!(saved.zones[0].vo.refs.is_empty());
    frame(&ctx, &mut app);
    fs::remove_dir_all(&dir).unwrap();
}

/// A film of ten seconds in a bench that has no zone: what an extraction
/// leaves, without the scene's own files.
fn bench_with_a_film(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("omnifrog-bench-app-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let film = Film {
        name: "Some.Film".into(),
        source: "/films/Some Film.mkv".into(),
        original: 1,
        dub: 2,
        elements: vec![
            Element {
                id: 3,
                name: "C".into(),
                object: false,
            },
            Element {
                id: 12,
                name: "12".into(),
                object: true,
            },
        ],
        channels: vec!["L".into(), "C".into()],
        frames: 10 * RATE as u64,
        ..Film::default()
    };
    let n = 10 * RATE as usize;
    let into = film.extract_dir(&dir);
    audio::write(&into.join("vo_3.wav"), &noise(3, n), RATE).unwrap();
    audio::write(&into.join("vo_12.wav"), &noise(12, n), RATE).unwrap();
    audio::write(&into.join("vf_L.wav"), &noise(20, n), RATE).unwrap();
    audio::write(&into.join("vf_C.wav"), &noise(21, n), RATE).unwrap();
    fs::write(
        film.dir(&dir).join(film::FILE),
        serde_json::to_string(&film).unwrap(),
    )
    .unwrap();
    dir
}

#[test]
fn a_range_of_a_film_is_opened_separated_and_made_a_zone() {
    let dir = bench_with_a_film("film");
    let ctx = egui::Context::default();
    let mut app = App::new(&ctx, dir.clone(), Tab::Film);
    frame(&ctx, &mut app);
    // The film is there, with the first object and the centre of the dub.
    assert_eq!(app.film.chosen, Some(0));
    assert_eq!(app.film.object, Some(12));
    assert_eq!(app.film.channel.as_deref(), Some("C"));
    assert!(app.zone_shown().is_none());

    // Nothing typed, or nonsense: said, and nothing opened.
    app.open_trial();
    assert!(app.film.trial.is_none() && !app.status.is_empty());
    app.film.from = "0:02".into();
    app.film.to = "0:08".into();
    app.open_trial();
    let trial = app.film.trial.clone().expect("a range opened");
    assert_eq!(
        (trial.start, trial.end),
        (2 * RATE as u64, Some(8 * RATE as u64))
    );
    assert_eq!(
        app.zone_shown().map(|z| z.id.clone()),
        Some(trial.id.clone())
    );
    assert!(app.rows.is_empty());
    frame(&ctx, &mut app);
    // The input can be heard before any model ran.
    app.listen_original(Side::Vo);
    assert_eq!(app.listen.player.len(), 6 * RATE as usize);

    // A model's outputs land on the range: a row for it.
    let mut split = Split {
        voice: 0.5,
        off_by: 0,
    };
    outputs::make(&dir, &trial, "model", RATE, &[], &mut split, &mut |_| {}).unwrap();
    app.refresh_rows();
    assert_eq!(app.rows.len(), 1);
    app.listen_output("model", Kind::Stem(Stem::VoiceVf));
    settle(&ctx, &mut app);
    // A range tried is not rated.
    app.rate("model", Side::Vf, Status::Reference);
    assert!(app.rows[0].status(Side::Vf) == Status::Neutral);

    // The loop becomes a zone of the set, with the model's outputs.
    app.listen
        .player
        .set_loop(Some((RATE as u64, 3 * RATE as u64)));
    app.film.kind = 3;
    app.add_zone();
    assert_eq!(app.set.zones.len(), 1, "{}", app.status);
    let zone = app.set.zones[0].clone();
    assert_eq!(zone.id, "Some.Film_o12_cC_3");
    assert_eq!(zone.kind, "Whisper");
    assert!(layout::outputs_present(&dir, &zone, "model"));
    assert_eq!(Set::load(&dir).unwrap().zones.len(), 1);

    // Out of the film tab, the zone listened to is the set's again.
    app.tab = Tab::Rate;
    app.select(Some(0));
    frame(&ctx, &mut app);
    assert_eq!(
        app.zone_shown().map(|z| z.id.clone()),
        Some(zone.id.clone())
    );
    assert_eq!(app.rows.len(), 1);
    assert!(app.listen.loaded.is_none());
    app.rate("model", Side::Vf, Status::Reference);
    assert_eq!(app.rows[0].status(Side::Vf), Status::Reference);
    app.tab = Tab::Film;
    frame(&ctx, &mut app);
    assert_eq!(app.zone_shown().map(|z| z.id.clone()), Some(trial.id));

    // The film removed, its zone stays.
    app.remove_film();
    assert!(app.film.films.is_empty() && app.film.trial.is_none());
    assert!(layout::zone_dir(&dir, &zone).is_dir());
    frame(&ctx, &mut app);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_cascade_is_added_named_kept_and_removed() {
    let dir = bench("cascades");
    let ctx = egui::Context::default();
    let mut app = App::new(&ctx, dir.clone(), Tab::Cascades);
    frame(&ctx, &mut app);
    // Half a form: said.
    app.add_cascade();
    assert!(app.set.cascades.is_empty() && !app.status.is_empty());
    app.cascade.0 = Some("one".into());
    app.cascade.1 = Some("two".into());
    app.cascade.2 = omnifrog_bench::set::CascadeMode::Voice;
    app.add_cascade();
    assert_eq!(app.set.cascades.len(), 1);
    let key = app.set.cascades[0].key();
    assert_eq!(key, "one~two~voice");
    // The same again is refused; the name says what it is.
    app.add_cascade();
    assert_eq!(app.set.cascades.len(), 1);
    assert_eq!(app.name(&key), "one then two on its voice");
    assert_eq!(
        app.name("one~two~voice+cancel"),
        "one then two on its voice — duplicates cancelled"
    );
    app.keep_cascade(&key, true);
    assert!(Set::load(&dir).unwrap().cascades[0].kept);
    frame(&ctx, &mut app);
    app.remove_cascade(&key);
    assert!(Set::load(&dir).unwrap().cascades.is_empty());
    frame(&ctx, &mut app);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_zone_is_removed_from_the_window() {
    let dir = bench("remove");
    let ctx = egui::Context::default();
    let mut app = App::new(&ctx, dir.clone(), Tab::Rate);
    app.listen_output("one", Kind::Mix);
    frame(&ctx, &mut app);
    app.remove_zone();
    assert!(app.set.zones.is_empty() && app.selected.is_none());
    assert!(app.rows.is_empty() && app.listen.loaded.is_none());
    assert!(!dir.join("zone").exists());
    assert!(Set::load(&dir).unwrap().zones.is_empty());
    frame(&ctx, &mut app);
    // Nothing chosen: nothing to remove.
    app.remove_zone();
    fs::remove_dir_all(&dir).unwrap();
}

/// A batch that ends while another tab than the report is shown: what its
/// thread saved must be what the window holds, or the next thing changed
/// here writes the old set over it.
#[test]
fn what_a_batch_saved_is_read_again_whatever_tab_is_shown() {
    let dir = bench("stale");
    let ctx = egui::Context::default();
    let mut app = App::new(&ctx, dir.clone(), Tab::Report);
    assert!(app.set.measures.is_empty());
    // The batch's thread, as far as the disk goes: a measure, saved.
    let mut saved = Set::load(&dir).unwrap();
    let measure = omnifrog_bench::set::Measure {
        seconds: 12.5,
        ..Default::default()
    };
    saved
        .measures
        .entry("zone".into())
        .or_default()
        .insert("one".into(), measure);
    saved.save(&dir).unwrap();
    // Meanwhile the film tab was opened; the batch ends there.
    app.tab = Tab::Film;
    frame(&ctx, &mut app);
    app.batch_changed();
    assert_eq!(
        app.set.measure("zone", "one").map(|m| m.seconds),
        Some(12.5)
    );
    // Back in the report, a checkbox: the measure is still on disk.
    app.tab = Tab::Report;
    frame(&ctx, &mut app);
    app.check("two", false);
    let after = Set::load(&dir).unwrap();
    assert_eq!(after.measure("zone", "one").map(|m| m.seconds), Some(12.5));
    assert!(!after.is_checked("two"));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn an_output_chosen_is_kept_and_opens_nothing() {
    let dir = bench("output");
    let kept = dir.join("window.json");
    let ctx = egui::Context::default();
    let mut app = App::new(&ctx, dir.clone(), Tab::Rate);
    app.remember_in(kept.clone());
    assert_eq!(app.listen.chosen, None);
    let headphones = Choice {
        id: "host:headphones".into(),
        name: "Headphones".into(),
    };
    app.choose_output(Some(headphones.clone()));
    // Nothing was playing: nothing is opened for having chosen.
    assert!(app.listen.output.is_none());
    assert_eq!(Prefs::load(&kept).output, Some(headphones.clone()));
    frame(&ctx, &mut app);
    // The list stays closed: the system was not asked for its outputs.
    assert!(!app.listen.listed && app.listen.choices.is_empty());

    // The next window finds the choice again.
    let mut next = App::new(&ctx, dir.clone(), Tab::Rate);
    next.remember_in(kept.clone());
    assert_eq!(next.listen.chosen, Some(headphones));
    frame(&ctx, &mut next);
    next.choose_output(None);
    assert_eq!(Prefs::load(&kept), Prefs::default());
    assert!(next.listen.output.is_none());

    // A file that does not read is nothing chosen, not a refusal to start.
    fs::write(&kept, "not json").unwrap();
    assert_eq!(Prefs::load(&kept), Prefs::default());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_model_is_shown_zone_by_zone() {
    let dir = bench("model");
    let ctx = egui::Context::default();
    let mut app = App::new(&ctx, dir.clone(), Tab::Model);
    // Nothing rated or measured: no model to show, and the tab draws.
    assert_eq!(app.model.key, None);
    frame(&ctx, &mut app);

    // A rating is enough for a model to have its lines, at once.
    app.rate("one", Side::Vf, Status::Reference);
    app.rate("two", Side::Vf, Status::Rejected(Some(Reason::Lack)));
    assert_eq!(app.model.keys, ["one", "two"]);
    assert_eq!(app.model.key.as_deref(), Some("one"));
    assert_eq!(app.model.rows.len(), app.set.zones.len());
    assert_eq!(app.model.rows[0].zone, "zone");
    assert_eq!(app.model.rows[0].report.refs, [0, 1]);
    app.report.side = Side::Vf;
    app.refresh_report();
    frame(&ctx, &mut app);

    // From the report, a model's name opens it here.
    app.tab = Tab::Report;
    app.show_model("two");
    assert_eq!(app.tab, Tab::Model);
    assert_eq!(app.model.rows[0].report.rejects, [0, 1]);
    assert_eq!(app.model.mean.rejects, [0, 1]);
    frame(&ctx, &mut app);

    // A column sorts the best first, then the other way round.
    app.sort_model_by(Column::SiSdr);
    assert_eq!(app.model.sort[Side::Vf as usize], (Column::SiSdr, true));
    app.sort_model_by(Column::SiSdr);
    assert_eq!(app.model.sort[Side::Vf as usize], (Column::SiSdr, false));
    // The other side keeps its own.
    assert_eq!(app.model.sort[Side::Vo as usize], (Column::Model, false));

    // A model the set no longer has leaves its place to the first.
    app.show_model("gone");
    assert_eq!(app.model.key.as_deref(), Some("one"));
    frame(&ctx, &mut app);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn an_output_the_system_renamed_is_still_the_one_chosen() {
    let dir = bench("renamed");
    let kept = dir.join("window.json");
    let ctx = egui::Context::default();
    let mut app = App::new(&ctx, dir.clone(), Tab::Rate);
    app.remember_in(kept.clone());
    let choice = |id: &str, name: &str| Choice {
        id: id.into(),
        name: name.into(),
    };
    app.choose_output(Some(choice("host:headphones", "Headphones")));

    // The same id under another name: it is there, and its name is the
    // one it has now, kept at once.
    let speakers = choice("host:speakers", "Speakers");
    let renamed = choice("host:headphones", "Headset");
    app.outputs_listed(vec![speakers.clone(), renamed.clone()]);
    assert!(app.output_listed());
    assert_eq!(app.listen.chosen, Some(renamed.clone()));
    assert_eq!(Prefs::load(&kept).output, Some(renamed.clone()));
    // Choosing it again, whatever it is called, is choosing nothing new.
    app.choose_output(Some(choice("host:headphones", "Headphones")));
    assert_eq!(app.listen.chosen, Some(renamed.clone()));
    assert!(app.listen.output.is_none());

    // Unplugged, it is no longer listed and stays the one chosen.
    app.outputs_listed(vec![speakers]);
    assert!(!app.output_listed());
    assert_eq!(app.listen.chosen, Some(renamed));
    frame(&ctx, &mut app);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_model_of_which_only_the_time_is_left_is_still_reported() {
    let dir = bench("timed");
    let mut set = Set::load(&dir).unwrap();
    // What a change of references leaves of a measure: the time the
    // outputs took, and no number.
    let timed = omnifrog_bench::set::Measure {
        seconds: 12.5,
        ..Default::default()
    };
    set.measures
        .entry("zone".into())
        .or_default()
        .insert("one".into(), timed);
    set.save(&dir).unwrap();
    for side in Side::BOTH {
        let table = crate::cli::report_model(&dir, "one", side, Column::SiSdr).unwrap();
        let lines: Vec<&str> = table.lines().collect();
        // The title, the columns, the zone, the mean.
        assert_eq!(lines.len(), 4);
        assert!(lines[2].starts_with("zone ") && lines[2].ends_with("12.5s"));
        assert!(lines[3].starts_with("mean (0)") && lines[3].ends_with("12.5s"));
    }
    // The window lists it too.
    let ctx = egui::Context::default();
    let app = App::new(&ctx, dir.clone(), Tab::Model);
    assert_eq!(app.model.keys, ["one"]);
    assert_eq!(app.model.rows[0].report.seconds, Some(12.5));
    // A key the set holds nothing of is said so.
    let error = crate::cli::report_model(&dir, "two", Side::Vo, Column::SiSdr).unwrap_err();
    assert!(error.contains("neither measured nor rated"));
    fs::remove_dir_all(&dir).unwrap();
}
