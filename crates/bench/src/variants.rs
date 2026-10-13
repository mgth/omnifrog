//! The variants computed from a model's outputs: the duplicates cancelled.
//!
//! A dubbed track carries the original's music and effects a few tens of
//! milliseconds late, and a model that lifts the dub's voice leaves some of
//! them in it: heard against the original's own, an echo. A run cancels
//! them (see the core's `duplicates`); a variant does the same on a zone,
//! from the outputs of one model, so the model can be judged as a run
//! would use it.
//!
//! The reference is the run's: the original's music and effects — the sum
//! of its other elements, written with the zone's context, and what this
//! model left of the element tested. The files of a variant, under the
//! key of the model and a suffix:
//!
//! - the voice and the rest of the original: the model's, copied;
//! - the voice of the dub: cancelled, still on the dub's clock, so it
//!   compares sample for sample with the model's own;
//! - the rest of the dub: the model's, plus what left the voice — voice and
//!   rest are still the channel;
//! - the mix: the rest of the original and the cancelled voice *moved onto
//!   the original's clock*, at the gain of the model's own mix. One gain
//!   for the model and its variants: from one row to the next, only the
//!   cancellation and the place of the voice change.
//!
//! `+cancel` moves the voice by the offset of the two tracks. `+cancel-step`
//! moves it by the step of the dub's map nearest each sample, with a 20 ms
//! fade between two steps: the map measures the dub's music and effects
//! against the original's, and whether it holds for the voices too is what
//! the ear is asked.

use crate::layout::{self, Loudness};
use crate::set::{RATE, Stem, Step, Zone};
use crate::{Error, Result, audio, keys, outputs};
use omnifrog_core::duplicates;
use std::fs;
use std::path::Path;

/// Frames of the fade between two steps of the voice moved step by step.
pub const FADE: i64 = RATE as i64 / 50;

/// Whether a zone has what a variant takes: its context, and the sum of
/// the other elements of the original.
pub fn possible(dir: &Path, zone: &Zone) -> bool {
    zone.duplicates
        .as_ref()
        .is_some_and(|c| !c.steps.is_empty())
        && layout::others(dir, zone).is_file()
}

/// Add a voice to a mix: `mix[i] += ratio · voice[i + offset(i)]`. The
/// offset is the tracks' (`steps` none), or that of the step nearest the
/// sample, faded over [`FADE`] frames centred on the border of two steps
/// that differ — a plain jump would cut a syllable.
pub fn add_voice(mix: &mut [f32], voice: &[f32], ratio: f64, lead: i64, steps: Option<&[Step]>) {
    let at = |j: i64| -> f64 {
        if j >= 0 && (j as usize) < voice.len() {
            voice[j as usize] as f64
        } else {
            0.0
        }
    };
    let steps = steps.unwrap_or(&[]);
    if steps.is_empty() {
        for (i, mix) in mix.iter_mut().enumerate() {
            *mix += (at(i as i64 + lead) * ratio) as f32;
        }
        return;
    }
    let border = |k: usize| (steps[k].centre + steps[k + 1].centre) / 2;
    let mut k = 0;
    for (i, mix) in mix.iter_mut().enumerate() {
        let i = i as i64;
        // The nearest step: their centres go up.
        while k + 1 < steps.len() && i >= border(k) {
            k += 1;
        }
        let here = steps[k].offset as i64;
        let mut value = at(i + here);
        if k + 1 < steps.len()
            && steps[k + 1].offset != steps[k].offset
            && i >= border(k) - FADE / 2
        {
            // Before the border with the next: on the way to its step.
            let w = (i - (border(k) - FADE / 2)) as f64 / FADE as f64;
            value = (1.0 - w) * value + w * at(i + steps[k + 1].offset as i64);
        } else if k > 0 && steps[k - 1].offset != steps[k].offset && i < border(k - 1) + FADE / 2 {
            // After the border with the one before: still arriving.
            let w = (i - (border(k - 1) - FADE / 2)) as f64 / FADE as f64;
            value = (1.0 - w) * at(i + steps[k - 1].offset as i64) + w * value;
        }
        *mix += (value * ratio) as f32;
    }
}

/// Make a variant of `base` on a zone: `suffix` is [`keys::CANCEL`] or
/// [`keys::CANCEL_STEP`]. The model's four stems must be there, and so
/// must the zone's context. Returns the loudness of the variant's files
/// and the energy taken out of the dub's voice, in dB below it.
pub fn make(dir: &Path, zone: &Zone, base: &str, suffix: &str) -> Result<(Loudness, f64)> {
    let context = zone
        .duplicates
        .as_ref()
        .filter(|c| !c.steps.is_empty())
        .ok_or_else(|| Error::msg(format!("{}: no duplicates context", zone.id)))?;
    for stem in Stem::ALL {
        let path = layout::output(dir, zone, base, stem);
        if !path.is_file() {
            return Err(Error::msg(format!(
                "{}: the model's outputs are incomplete; separate it first",
                path.display()
            )));
        }
    }
    let from = |stem| layout::output(dir, zone, base, stem);
    let others = layout::others(dir, zone);
    if omnifrog_core::wav::frames(&others) == 0
        || omnifrog_core::wav::frames(&others) != omnifrog_core::wav::frames(&from(Stem::RestVo))
    {
        return Err(Error::msg(format!(
            "{}: the sum of the other elements of the original is absent, or not the zone's length",
            others.display()
        )));
    }
    let key = format!("{base}{suffix}");
    // Made aside and put in place whole, as any output.
    outputs::staged(dir, zone, &key, |staging| {
        let to = |stem| layout::output(dir, zone, staging, stem);
        for stem in [Stem::VoiceVo, Stem::RestVo] {
            fs::copy(from(stem), to(stem)).map_err(|e| Error::io(from(stem), e))?;
        }
        let steps: Vec<duplicates::Step> =
            context.steps.iter().map(|s| (s.centre, s.offset)).collect();
        let removed = duplicates::cancel_extract(
            &steps,
            &[others, from(Stem::RestVo)],
            &from(Stem::VoiceVf),
            &to(Stem::VoiceVf),
        )?;

        let before = audio::read(&from(Stem::VoiceVf))?;
        let after = audio::read(&to(Stem::VoiceVf))?;
        // What left the voice joins the rest.
        let mut rest = audio::read(&from(Stem::RestVf))?;
        for (rest, (before, after)) in rest.iter_mut().zip(before.iter().zip(&after)) {
            *rest += before - after;
        }
        audio::write(&to(Stem::RestVf), &rest, RATE)?;

        // The gain of the model's own mix: the one that mix was made with
        // when it was kept — a zone cut out of a longer range has the mix
        // of that range, and levels measured on the cut would give another
        // gain, so the variant would change the volume even where nothing
        // is cancelled. Without it (an imported output), from the model's
        // loudness, the dub's voice as it was before the cancellation.
        let mut base_loudness = Loudness::load(dir, zone, base);
        let ratio = match base_loudness.mix_gain() {
            Some(ratio) => ratio,
            None => {
                if base_loudness.of(Stem::VoiceVo).is_none()
                    || base_loudness.of(Stem::VoiceVf).is_none()
                {
                    base_loudness = outputs::loudness_of(dir, zone, base)?;
                }
                match (
                    base_loudness.of(Stem::VoiceVo),
                    base_loudness.of(Stem::VoiceVf),
                ) {
                    (Some(vo), Some(vf)) => 10f64.powf((vo - vf) / 20.0),
                    _ => 1.0,
                }
            }
        };
        // > 0: the dub starts before the original, its voice arrives late;
        // the dub's sample i + lead goes to i.
        let lead = (context.dub_lead_ms * RATE as f64 / 1000.0).round_ties_even() as i64;
        let mut mix = audio::read(&from(Stem::RestVo))?;
        let by_step = (suffix == keys::CANCEL_STEP).then_some(context.steps.as_slice());
        add_voice(&mut mix, &after, ratio, lead, by_step);
        audio::write(&layout::mix(dir, zone, staging), &mix, RATE)?;

        let mut loudness = outputs::loudness_of(dir, zone, staging)?;
        loudness.set_mix_gain(ratio);
        loudness.save(dir, zone, staging)?;
        Ok((loudness, removed))
    })
}

/// The steps that touch a zone of `frames` frames, in ms, each once: where
/// a variant put the voice.
pub fn steps_over(steps: &[Step], frames: i64) -> String {
    let mut seen: Vec<i32> = Vec::new();
    for (i, step) in steps.iter().enumerate() {
        let ends_after_start = i + 1 >= steps.len() || (step.centre + steps[i + 1].centre) / 2 > 0;
        let starts_before_end = i == 0 || (steps[i - 1].centre + step.centre) / 2 < frames;
        if ends_after_start && starts_before_end && !seen.contains(&step.offset) {
            seen.push(step.offset);
        }
    }
    seen.iter()
        .map(|offset| format!("{:.1} ms", *offset as f64 * 1000.0 / RATE as f64))
        .collect::<Vec<_>>()
        .join(" / ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outputs::tests::noise;
    use crate::set::DuplicatesContext;

    fn step(centre: i64, offset: i32) -> Step {
        Step { centre, offset }
    }

    #[test]
    fn a_voice_is_added_at_the_tracks_offset() {
        let voice = [1.0, 2.0, 3.0, 4.0];
        let mut mix = [10.0f32; 4];
        add_voice(&mut mix, &voice, 0.5, 1, None);
        // The dub's sample i + 1 goes to i; past its end, nothing.
        assert_eq!(mix, [11.0, 11.5, 12.0, 10.0]);
        let mut mix = [0.0f32; 4];
        add_voice(&mut mix, &voice, 1.0, -2, Some(&[]));
        assert_eq!(mix, [0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn a_voice_moved_by_step_fades_from_one_step_to_the_next() {
        // A ramp, so a sample says where it was read: two steps, 0 then
        // 100, their border at 5000.
        let voice: Vec<f32> = (0..20_000).map(|i| i as f32).collect();
        let steps = [step(0, 0), step(10_000, 100)];
        let mut mix = vec![0.0f32; 10_000];
        add_voice(&mut mix, &voice, 1.0, 0, Some(&steps));
        let half = (FADE / 2) as usize;
        // Well before the border: the first step; well after: the second.
        assert_eq!(mix[1000], 1000.0);
        assert_eq!(mix[5000 - half - 1], (5000 - half - 1) as f32);
        assert_eq!(mix[5000 + half], (5000 + half + 100) as f32);
        assert_eq!(mix[9000], 9100.0);
        // On the border: half-way between the two.
        assert_eq!(mix[5000], 5050.0);
        // Two steps alike: no fade, one offset throughout.
        let steps = [step(0, 7), step(10_000, 7)];
        let mut mix = vec![0.0f32; 10_000];
        add_voice(&mut mix, &voice, 1.0, 0, Some(&steps));
        assert_eq!((mix[4999], mix[5000]), (5006.0, 5007.0));
    }

    #[test]
    fn the_steps_that_touch_a_zone_are_told_once() {
        let steps = [
            step(-960_000, 48),
            step(-720_000, 48),
            step(-240_000, 96),
            step(0, 96),
            step(240_000, 144),
            step(720_000, 480),
        ];
        // A zone of two seconds: the steps whose reach crosses [0, 96000).
        assert_eq!(steps_over(&steps, 96_000), "2.0 ms");
        assert_eq!(steps_over(&steps, 480_000), "2.0 ms / 3.0 ms");
        assert_eq!(steps_over(&[], 96_000), "");
    }

    /// A zone of three seconds where the dub's voice carries the original's
    /// music and effects twenty samples late.
    #[test]
    fn a_variant_takes_the_duplicate_out_of_the_voice_and_keeps_the_channel_whole() {
        let dir =
            std::env::temp_dir().join(format!("omnifrog-bench-variant-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let n = 3 * RATE as usize;
        let mut zone = Zone {
            id: "zone".into(),
            object: 1,
            channel: "C".into(),
            ..Zone::default()
        };
        let effects = noise(1, n, 0.2);
        let speech = noise(2, n, 0.2);
        let late: Vec<f32> = (0..n)
            .map(|i| if i >= 20 { effects[i - 20] } else { 0.0 })
            .collect();
        let voice: Vec<f32> = speech.iter().zip(&late).map(|(s, e)| s + 0.5 * e).collect();
        let put = |stem, samples: &[f32]| {
            audio::write(&layout::output(&dir, &zone, "model", stem), samples, RATE).unwrap();
        };
        put(Stem::VoiceVo, &noise(3, n, 0.1));
        put(Stem::RestVo, &effects);
        put(Stem::VoiceVf, &voice);
        put(Stem::RestVf, &noise(4, n, 0.05));
        // The other elements of the original: silent here.
        duplicates::write_sum(&[], n as u64, &layout::others(&dir, &zone)).unwrap();

        // No context yet: said.
        assert!(!possible(&dir, &zone));
        assert!(make(&dir, &zone, "model", keys::CANCEL).is_err());
        zone.duplicates = Some(DuplicatesContext {
            dub_lead_ms: 0.0,
            steps: vec![step(0, 20), step(240_000, 20)],
            others: String::new(),
        });
        assert!(possible(&dir, &zone));

        let (loudness, removed) = make(&dir, &zone, "model", keys::CANCEL).unwrap();
        assert!(layout::outputs_present(&dir, &zone, "model+cancel"));
        assert!(loudness.mix().is_some());
        // Half the effects were in the voice — a fifth of its energy, 7 dB
        // below it. Against a voice that is noise too, the cancellation
        // takes out a good part of that, not all.
        assert!((-14.0..-6.0).contains(&removed), "{removed}");
        // The voice is closer to the speech than it was.
        let after =
            audio::read(&layout::output(&dir, &zone, "model+cancel", Stem::VoiceVf)).unwrap();
        let error = |x: &[f32]| -> f64 {
            x.iter()
                .zip(&speech)
                .skip(4800)
                .take(n - 9600)
                .map(|(x, s)| ((x - s) as f64).powi(2))
                .sum()
        };
        let (was, is) = (error(&voice), error(&after));
        assert!(is < 0.75 * was, "{is} {was}");
        // Voice and rest are still the channel.
        let rest_before = audio::read(&layout::output(&dir, &zone, "model", Stem::RestVf)).unwrap();
        let rest_after =
            audio::read(&layout::output(&dir, &zone, "model+cancel", Stem::RestVf)).unwrap();
        let voice_read = audio::read(&layout::output(&dir, &zone, "model", Stem::VoiceVf)).unwrap();
        for i in [5000, 60_000, 120_000] {
            let whole = voice_read[i] + rest_before[i];
            assert!(
                (after[i] + rest_after[i] - whole).abs() < 3.0 / 8_388_608.0,
                "{i}"
            );
        }
        // The original's side is the model's, as it was.
        assert_eq!(
            fs::read(layout::output(&dir, &zone, "model+cancel", Stem::RestVo)).unwrap(),
            fs::read(layout::output(&dir, &zone, "model", Stem::RestVo)).unwrap()
        );
        // A model without its stems: said, and nothing left.
        fs::remove_file(layout::output(&dir, &zone, "model", Stem::RestVf)).unwrap();
        assert!(make(&dir, &zone, "model", keys::CANCEL_STEP).is_err());
        assert!(!layout::output_dir(&dir, &zone, "model+cancel-step").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    /// A zone cut out of a longer range: its mix was made at the gain the
    /// levels had over the range, and the levels measured on the zone would
    /// give another. With nothing to cancel and no offset, the variant's
    /// mix must be the model's own.
    #[test]
    fn a_variant_mixes_at_the_gain_the_model_s_own_mix_was_made_with() {
        let dir = std::env::temp_dir().join(format!("omnifrog-bench-gain-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let n = 3 * RATE as usize;
        let zone = Zone {
            id: "zone".into(),
            object: 1,
            channel: "C".into(),
            duplicates: Some(DuplicatesContext {
                dub_lead_ms: 0.0,
                steps: vec![step(0, 0), step(240_000, 0)],
                others: String::new(),
            }),
            ..Zone::default()
        };
        let silence = vec![0.0f32; n];
        let voice = noise(2, n, 0.1);
        let put = |stem, samples: &[f32]| {
            audio::write(&layout::output(&dir, &zone, "model", stem), samples, RATE).unwrap();
        };
        put(Stem::VoiceVo, &noise(3, n, 0.2));
        put(Stem::RestVo, &silence);
        put(Stem::VoiceVf, &voice);
        put(Stem::RestVf, &silence);
        duplicates::write_sum(&[], n as u64, &layout::others(&dir, &zone)).unwrap();
        // The mix as the range made it, at a gain of about 1.46; measured
        // on the zone, the two voices would say 2.
        let gain = 1.458f64;
        let mix: Vec<f32> = voice.iter().map(|v| (*v as f64 * gain) as f32).collect();
        audio::write(&layout::mix(&dir, &zone, "model"), &mix, RATE).unwrap();
        let mut loudness = outputs::loudness_of(&dir, &zone, "model").unwrap();
        let measured = loudness.of(Stem::VoiceVo).unwrap() - loudness.of(Stem::VoiceVf).unwrap();
        assert!((measured - 6.0).abs() < 0.2, "{measured}");
        loudness.set_mix_gain(gain);
        loudness.save(&dir, &zone, "model").unwrap();

        let (variant, removed) = make(&dir, &zone, "model", keys::CANCEL).unwrap();
        // Nothing to cancel against silence.
        assert!(removed < -60.0, "{removed}");
        assert!((variant.mix_gain().unwrap() - gain).abs() < 1e-9);
        let theirs = audio::read(&layout::mix(&dir, &zone, "model")).unwrap();
        let ours = audio::read(&layout::mix(&dir, &zone, "model+cancel")).unwrap();
        let worst = theirs
            .iter()
            .zip(&ours)
            .skip(4800)
            .take(n - 9600)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-4, "{worst}");
        // Without the gain kept — an imported output — the levels decide,
        // as they did when it was made.
        let mut bare = Loudness::load(&dir, &zone, "model");
        bare.0.remove(layout::MIX_GAIN);
        bare.save(&dir, &zone, "model").unwrap();
        let (variant, _) = make(&dir, &zone, "model", keys::CANCEL).unwrap();
        assert!((variant.mix_gain().unwrap() - 2.0).abs() < 0.05);
        fs::remove_dir_all(&dir).unwrap();
    }
}
