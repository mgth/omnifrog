//! The five files a model makes of a zone.
//!
//! The zone's two inputs go to the model as a run sends an element: the very
//! same samples under a header that says the rate the model was trained at.
//! What comes back is folded to mono, squared up with the input it came
//! from and written at 24 bits: the voice and the rest of the original's
//! element, those of the dub's channel. The fifth file is what a run would
//! make of them — the rest of the original with the voice of the dub
//! brought to the level of the voice it replaces — and the loudness of all
//! five is kept beside them.
//!
//! The model hears the zone alone here, with nothing around it. On a zone
//! of a dozen seconds that is close to what it makes of the same seconds
//! inside a longer range (within −31 dB on the one compared). On a zone
//! shorter than the model's window it is not: on one of two seconds, the
//! dub's voice came out some 40 dB under what the same model gave it with
//! the film around. Lengthening the input by mirroring it up to the window
//! was tried and did not bring the voice back. The outputs of a very short
//! zone are better cut from the separation of a longer range, which is how
//! the zones of an imported legacy set were made.

use crate::layout::{self, Loudness};
use crate::set::{RATE, Side, Stem, Zone};
use crate::{Error, Result, audio, keys};
use omnifrog_core::worker::{Stems, Worker};
use omnifrog_core::{loudness, wav};
use std::fs;
use std::path::{Path, PathBuf};

/// What separates inputs into stems: the core's worker, or a stand-in where
/// there is no model to run.
pub trait Separate {
    /// Each input leaves a voice and a rest in `out_dir`; `fraction` is
    /// told how far the request is.
    fn separate(
        &mut self,
        inputs: &[PathBuf],
        out_dir: &Path,
        fraction: &mut dyn FnMut(f64),
    ) -> Result<Vec<Stems>>;
}

impl Separate for Worker {
    fn separate(
        &mut self,
        inputs: &[PathBuf],
        out_dir: &Path,
        fraction: &mut dyn FnMut(f64),
    ) -> Result<Vec<Stems>> {
        Worker::separate(self, inputs, out_dir, fraction)
    }
}

/// The loudness of a file, when there is enough sound in it to have one.
pub fn lkfs(path: &Path, surround: bool) -> Result<Option<f64>> {
    let measure = loudness::of(path, surround, None)?;
    Ok(measure.valid().then_some(measure.lkfs))
}

/// Where an element of the original is over a zone, when the bench knows:
/// pairs of (frame of the zone, position). Empty: weighted as a front
/// element, which is what a zone cut out of its film has to settle for.
pub type Trajectory = [(u64, [f64; 3])];

/// The loudness of the original's voice, weighted by where the object is
/// at each instant, as a run weights it.
fn lkfs_along(path: &Path, trajectory: &Trajectory) -> Result<Option<f64>> {
    if trajectory.is_empty() {
        return lkfs(path, false);
    }
    let voice = loudness::Voice::object(path, trajectory.to_vec());
    let measure = loudness::measure(&[voice], None, &omnifrog_core::progress::quiet)?;
    Ok(measure.valid().then_some(measure.lkfs))
}

/// The loudness of the five files of an output as they are on disk, for
/// files that were cut rather than made.
pub fn loudness_of(dir: &Path, zone: &Zone, key: &str) -> Result<Loudness> {
    let surround = loudness::is_surround(&zone.channel);
    let mut loudness = Loudness::default();
    for stem in Stem::ALL {
        let path = layout::output(dir, zone, key, stem);
        if path.is_file()
            && let Some(value) = lkfs(&path, stem.side() == Side::Vf && surround)?
        {
            loudness.0.insert(stem.name().to_string(), value);
        }
    }
    let mix = layout::mix(dir, zone, key);
    if mix.is_file()
        && let Some(value) = lkfs(&mix, surround)?
    {
        loudness.0.insert("mix".to_string(), value);
    }
    Ok(loudness)
}

/// A stem as the worker left it, folded to mono and brought to the length
/// of the input it came from: padded with silence, or cut.
fn fitted(path: &Path, frames: usize) -> Result<Vec<f32>> {
    let mut samples = audio::read(path)?;
    samples.resize(frames, 0.0);
    Ok(samples)
}

/// Make the five files of `key` on a zone with a separator loaded with that
/// model, `rate` being the rate the model is told its input is at. They are
/// made aside and take the place of what was there for that key at the end
/// (see [`staged`]); the variants computed from the old outputs go then.
pub fn make(
    dir: &Path,
    zone: &Zone,
    key: &str,
    rate: u32,
    trajectory: &Trajectory,
    separator: &mut dyn Separate,
    fraction: &mut dyn FnMut(f64),
) -> Result<Loudness> {
    let inputs = Side::BOTH.map(|side| layout::extract(dir, zone, side));
    for input in &inputs {
        if !input.is_file() {
            return Err(Error::msg(format!(
                "{}: the zone has lost its input; it cannot be computed again",
                input.display()
            )));
        }
    }
    let frames = [
        wav::frames(&inputs[0]) as usize,
        wav::frames(&inputs[1]) as usize,
    ];
    let made = staged(dir, zone, key, |staging| {
        let out = layout::output_dir(dir, zone, staging);
        let (prepared, stems_dir) = (out.join("_prep"), out.join("_stems"));
        fs::create_dir_all(&prepared).map_err(|e| Error::io(&prepared, e))?;
        let fed = [prepared.join("vo.wav"), prepared.join("vf.wav")];
        for (input, fed) in inputs.iter().zip(&fed) {
            wav::copy_rate(input, fed, rate)?;
        }
        let stems = separator.separate(&fed, &stems_dir, fraction)?;
        let [vo, vf] = stems.as_slice() else {
            return Err(Error::msg(format!(
                "the worker returned {} pair(s) of stems for two inputs",
                stems.len()
            )));
        };
        let stems = [
            fitted(&vo.voice, frames[0])?,
            fitted(&vo.rest, frames[0])?,
            fitted(&vf.voice, frames[1])?,
            fitted(&vf.rest, frames[1])?,
        ];
        // What was fed and what came back are not part of the output.
        remove_dir(&prepared)?;
        remove_dir(&stems_dir)?;
        finish(dir, zone, staging, stems, trajectory)
    })?;
    // The variants computed from the outputs this one replaces came from
    // them.
    for variant in keys::VARIANTS {
        remove_dir(&layout::output_dir(dir, zone, &format!("{key}{variant}")))?;
    }
    Ok(made)
}

/// Make an output aside and put it in place whole. `make` is handed the
/// name to write under — the key with [`layout::PART`] after it — and the
/// directory takes its key, in the place of what was there, only once
/// `make` has returned. A run that fails, or is killed half-way, leaves
/// nothing under the key: an output that is there has every file it will
/// ever have, and one that is not is made again.
pub fn staged<T>(
    dir: &Path,
    zone: &Zone,
    key: &str,
    make: impl FnOnce(&str) -> Result<T>,
) -> Result<T> {
    let staging = format!("{key}{}", layout::PART);
    let aside = layout::output_dir(dir, zone, &staging);
    remove_dir(&aside)?;
    fs::create_dir_all(&aside).map_err(|e| Error::io(&aside, e))?;
    match make(&staging) {
        Ok(made) => {
            let place = layout::output_dir(dir, zone, key);
            remove_dir(&place)?;
            fs::rename(&aside, &place).map_err(|e| Error::io(&place, e))?;
            Ok(made)
        }
        Err(e) => {
            let _ = fs::remove_dir_all(&aside);
            Err(e)
        }
    }
}

/// Write the four stems of an output — the voice and the rest of the
/// original, those of the dub — then its mix and the loudness of all five.
fn finish(
    dir: &Path,
    zone: &Zone,
    key: &str,
    [voice_vo, rest_vo, voice_vf, rest_vf]: [Vec<f32>; 4],
    trajectory: &Trajectory,
) -> Result<Loudness> {
    let surround = loudness::is_surround(&zone.channel);
    let mut loudness = Loudness::default();
    for (stem, samples, surround) in [
        (Stem::VoiceVo, &voice_vo, false),
        (Stem::RestVo, &rest_vo, false),
        (Stem::VoiceVf, &voice_vf, surround),
        (Stem::RestVf, &rest_vf, surround),
    ] {
        let path = layout::output(dir, zone, key, stem);
        audio::write(&path, samples, RATE)?;
        // The original's voice by the object's trajectory when it is
        // known; everything else of the original as a front element, as
        // an imported set's levels were measured.
        let value = match stem {
            Stem::VoiceVo => lkfs_along(&path, trajectory)?,
            _ => lkfs(&path, surround)?,
        };
        if let Some(value) = value {
            loudness.0.insert(stem.name().to_string(), value);
        }
    }
    // The mix: the rest of the original, and the voice of the dub at
    // the level of the voice it replaces.
    let ratio = match (loudness.of(Stem::VoiceVo), loudness.of(Stem::VoiceVf)) {
        (Some(vo), Some(vf)) => 10f64.powf((vo - vf) / 20.0),
        _ => 1.0,
    };
    loudness.set_mix_gain(ratio);
    let mut mix = rest_vo;
    for (mix, voice) in mix.iter_mut().zip(&voice_vf) {
        *mix += (*voice as f64 * ratio) as f32;
    }
    let path = layout::mix(dir, zone, key);
    audio::write(&path, &mix, RATE)?;
    if let Some(value) = lkfs(&path, surround)? {
        loudness.0.insert("mix".to_string(), value);
    }
    loudness.save(dir, zone, key)?;
    Ok(loudness)
}

/// Make the five files of a cascade on a zone from the outputs its first
/// model already has there: the second model, loaded in `separator`, works
/// the first's rest or voice again, and what it finds changes side. The
/// sum of voice and rest stays the first model's: nothing is lost.
pub fn make_cascade(
    dir: &Path,
    zone: &Zone,
    cascade: &crate::set::Cascade,
    rate: u32,
    separator: &mut dyn Separate,
    fraction: &mut dyn FnMut(f64),
) -> Result<Loudness> {
    use crate::set::CascadeMode;
    let key = cascade.key();
    let first = |stem| layout::output(dir, zone, &cascade.first, stem);
    for stem in Stem::ALL {
        if !first(stem).is_file() {
            return Err(Error::msg(format!(
                "{}: the first model of the cascade has no outputs here; separate it first",
                first(stem).display()
            )));
        }
    }
    let again = match cascade.mode {
        CascadeMode::Rest => [first(Stem::RestVo), first(Stem::RestVf)],
        CascadeMode::Voice => [first(Stem::VoiceVo), first(Stem::VoiceVf)],
    };
    let frames = [
        wav::frames(&again[0]) as usize,
        wav::frames(&again[1]) as usize,
    ];
    let made = staged(dir, zone, &key, |staging| {
        let out = layout::output_dir(dir, zone, staging);
        let (prepared, stems_dir) = (out.join("_prep"), out.join("_stems"));
        fs::create_dir_all(&prepared).map_err(|e| Error::io(&prepared, e))?;
        let fed = [prepared.join("vo.wav"), prepared.join("vf.wav")];
        for (input, fed) in again.iter().zip(&fed) {
            wav::copy_rate(input, fed, rate)?;
        }
        let stems = separator.separate(&fed, &stems_dir, fraction)?;
        let [vo, vf] = stems.as_slice() else {
            return Err(Error::msg(format!(
                "the worker returned {} pair(s) of stems for two inputs",
                stems.len()
            )));
        };
        // What the second model found in the stem it was given joins the
        // first model's other stem.
        let sum = |kept: Stem, moved: &Path, frames: usize| -> Result<Vec<f32>> {
            let mut samples = audio::read(&first(kept))?;
            samples.resize(frames, 0.0);
            for (sample, moved) in samples.iter_mut().zip(fitted(moved, frames)?) {
                *sample += moved;
            }
            Ok(samples)
        };
        let stems = match cascade.mode {
            CascadeMode::Rest => [
                sum(Stem::VoiceVo, &vo.voice, frames[0])?,
                fitted(&vo.rest, frames[0])?,
                sum(Stem::VoiceVf, &vf.voice, frames[1])?,
                fitted(&vf.rest, frames[1])?,
            ],
            CascadeMode::Voice => [
                fitted(&vo.voice, frames[0])?,
                sum(Stem::RestVo, &vo.rest, frames[0])?,
                fitted(&vf.voice, frames[1])?,
                sum(Stem::RestVf, &vf.rest, frames[1])?,
            ],
        };
        remove_dir(&prepared)?;
        remove_dir(&stems_dir)?;
        finish(dir, zone, staging, stems, &[])
    })?;
    for variant in keys::VARIANTS {
        remove_dir(&layout::output_dir(dir, zone, &format!("{key}{variant}")))?;
    }
    Ok(made)
}

fn remove_dir(path: &Path) -> Result<()> {
    match fs::remove_dir_all(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(Error::io(path, e)),
        _ => Ok(()),
    }
}

/// A separator for tests and for trying the bench where there is no model:
/// the voice is the input at a gain, the rest what is left of it. Stems are
/// written where the worker writes them, under its names.
pub struct Split {
    pub voice: f32,
    /// Frames the stems are longer (or, negative, shorter) than the input
    /// by, as a model's are when its rate is not the input's.
    pub off_by: i64,
}

impl Separate for Split {
    fn separate(
        &mut self,
        inputs: &[PathBuf],
        out_dir: &Path,
        fraction: &mut dyn FnMut(f64),
    ) -> Result<Vec<Stems>> {
        fs::create_dir_all(out_dir).map_err(|e| Error::io(out_dir, e))?;
        let mut all = Vec::new();
        for (i, input) in inputs.iter().enumerate() {
            let samples = audio::read(input)?;
            let frames = (samples.len() as i64 + self.off_by).max(0) as usize;
            let part = |gain: f32| -> Vec<f32> {
                let mut part: Vec<f32> = samples.iter().map(|s| gain * s).collect();
                part.resize(frames, 0.0);
                part
            };
            let name = input.file_stem().unwrap_or_default().to_string_lossy();
            let stems = Stems {
                voice: out_dir.join(format!("{name}{}", omnifrog_core::worker::VOICE_SUFFIX)),
                rest: out_dir.join(format!("{name}{}", omnifrog_core::worker::REST_SUFFIX)),
            };
            audio::write(&stems.voice, &part(self.voice), RATE)?;
            audio::write(&stems.rest, &part(1.0 - self.voice), RATE)?;
            all.push(stems);
            fraction((i + 1) as f64 / inputs.len() as f64);
        }
        Ok(all)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn noise(seed: u64, n: usize, level: f32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                level * ((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
            })
            .collect()
    }

    /// A zone of two seconds, its two inputs at different levels.
    pub(crate) fn bench(name: &str) -> (PathBuf, Zone) {
        let dir = std::env::temp_dir().join(format!(
            "omnifrog-bench-outputs-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let zone = Zone {
            id: "zone".into(),
            object: 7,
            channel: "C".into(),
            ..Zone::default()
        };
        let n = 2 * RATE as usize;
        audio::write(
            &layout::extract(&dir, &zone, Side::Vo),
            &noise(1, n, 0.25),
            RATE,
        )
        .unwrap();
        audio::write(
            &layout::extract(&dir, &zone, Side::Vf),
            &noise(2, n, 0.0625),
            RATE,
        )
        .unwrap();
        (dir, zone)
    }

    #[test]
    fn a_model_leaves_five_files_and_their_loudness() {
        let (dir, zone) = bench("five");
        // A stale variant of the key, and a stale file of the key itself.
        let stale = layout::output_dir(&dir, &zone, "model+cancel");
        fs::create_dir_all(&stale).unwrap();
        let leftover = layout::output_dir(&dir, &zone, "model").join("old.wav");
        fs::create_dir_all(leftover.parent().unwrap()).unwrap();
        fs::write(&leftover, "").unwrap();

        let mut fractions = Vec::new();
        let mut split = Split {
            voice: 0.5,
            off_by: -100,
        };
        let loudness = make(&dir, &zone, "model", 44_100, &[], &mut split, &mut |f| {
            fractions.push(f)
        })
        .unwrap();
        assert_eq!(fractions, [0.5, 1.0]);
        assert!(layout::outputs_present(&dir, &zone, "model"));
        assert!(!stale.exists());
        assert!(!leftover.exists());
        let out = layout::output_dir(&dir, &zone, "model");
        assert!(!out.join("_prep").exists() && !out.join("_stems").exists());
        assert_eq!(layout::keys_on_disk(&dir, &zone), ["model"]);

        // Every file is the input's length, at the bench's rate, whatever
        // the stems were.
        let n = 2 * RATE as u64;
        for stem in Stem::ALL {
            let format = wav::read_header(&layout::output(&dir, &zone, "model", stem)).unwrap();
            assert_eq!(
                (format.frames, format.sample_rate, format.bits),
                (n, RATE, 24)
            );
        }
        assert_eq!(wav::frames(&layout::mix(&dir, &zone, "model")), n);

        // The dub's voice is 12 dB under the original's: the mix brings it
        // up by as much, onto the rest of the original.
        let (vo, vf) = (
            loudness.of(Stem::VoiceVo).unwrap(),
            loudness.of(Stem::VoiceVf).unwrap(),
        );
        assert!((vo - vf - 12.04).abs() < 0.05, "{vo} {vf}");
        let rest = audio::read(&layout::output(&dir, &zone, "model", Stem::RestVo)).unwrap();
        let voice = audio::read(&layout::output(&dir, &zone, "model", Stem::VoiceVf)).unwrap();
        let mix = audio::read(&layout::mix(&dir, &zone, "model")).unwrap();
        let ratio = 10f64.powf((vo - vf) / 20.0);
        for i in [0, 1000, 50_000] {
            let expected = rest[i] + (voice[i] as f64 * ratio) as f32;
            assert!((mix[i] - expected).abs() < 2.0 / 8_388_608.0, "{i}");
        }
        assert!(loudness.mix().is_some());
        assert_eq!(Loudness::load(&dir, &zone, "model"), loudness);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_cascade_moves_what_the_second_model_finds_and_loses_nothing() {
        use crate::set::{Cascade, CascadeMode};
        let (dir, zone) = bench("cascade");
        let mut first = Split {
            voice: 0.5,
            off_by: 0,
        };
        make(&dir, &zone, "a", RATE, &[], &mut first, &mut |_| {}).unwrap();
        let read = |key: &str, stem| audio::read(&layout::output(&dir, &zone, key, stem)).unwrap();
        let step = 1.0 / 8_388_608.0f32;

        // The second model calls a quarter of what it is given voice.
        let mut second = Split {
            voice: 0.25,
            off_by: 0,
        };
        for (mode, key) in [
            (CascadeMode::Rest, "a~b~rest"),
            (CascadeMode::Voice, "a~b~voice"),
        ] {
            let cascade = Cascade {
                first: "a".into(),
                second: "b".into(),
                mode,
                kept: false,
            };
            assert_eq!(cascade.key(), key);
            let loudness =
                make_cascade(&dir, &zone, &cascade, RATE, &mut second, &mut |_| {}).unwrap();
            assert!(layout::outputs_present(&dir, &zone, key));
            assert!(loudness.mix().is_some());
            let out = layout::output_dir(&dir, &zone, key);
            assert!(!out.join("_prep").exists() && !out.join("_stems").exists());
            for side in Side::BOTH {
                let (a_voice, a_rest) = (read("a", side.voice()), read("a", side.rest()));
                let (voice, rest) = (read(key, side.voice()), read(key, side.rest()));
                for i in [0, 777, 90_000] {
                    // Voice and rest are still what the first model made.
                    let whole = a_voice[i] + a_rest[i];
                    assert!(
                        (voice[i] + rest[i] - whole).abs() <= 3.0 * step,
                        "{key} {i}"
                    );
                    // And the part that moved is the second model's.
                    let expected = match mode {
                        CascadeMode::Rest => 0.75 * a_rest[i],
                        CascadeMode::Voice => a_rest[i] + 0.75 * a_voice[i],
                    };
                    assert!((rest[i] - expected).abs() <= 3.0 * step, "{key} {i}");
                }
            }
        }
        // A first model that was never run here: said, nothing left.
        let orphan = Cascade {
            first: "nobody".into(),
            second: "b".into(),
            mode: CascadeMode::Rest,
            kept: false,
        };
        assert!(make_cascade(&dir, &zone, &orphan, RATE, &mut second, &mut |_| {}).is_err());
        assert!(!layout::output_dir(&dir, &zone, &orphan.key()).exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_zone_without_its_input_or_a_worker_that_fails_leaves_nothing() {
        struct Broken;
        impl Separate for Broken {
            fn separate(
                &mut self,
                _: &[PathBuf],
                _: &Path,
                _: &mut dyn FnMut(f64),
            ) -> Result<Vec<Stems>> {
                Err(Error::msg("out of memory"))
            }
        }
        let (dir, zone) = bench("broken");
        assert!(make(&dir, &zone, "model", RATE, &[], &mut Broken, &mut |_| {}).is_err());
        assert!(!layout::output_dir(&dir, &zone, "model").exists());
        assert!(layout::keys_on_disk(&dir, &zone).is_empty());

        // A run killed after the stems and before the mix: what it left is
        // under no key, so the output is not there and is made again.
        let half = layout::output_dir(&dir, &zone, "model.part");
        for stem in Stem::ALL {
            let path = half.join(layout::stem_file(stem));
            fs::create_dir_all(&half).unwrap();
            fs::write(path, "half").unwrap();
        }
        assert!(!layout::outputs_present(&dir, &zone, "model"));
        assert!(layout::keys_on_disk(&dir, &zone).is_empty());
        let mut whole = Split {
            voice: 0.5,
            off_by: 0,
        };
        make(&dir, &zone, "model", RATE, &[], &mut whole, &mut |_| {}).unwrap();
        assert!(layout::outputs_present(&dir, &zone, "model"));
        assert!(layout::mix(&dir, &zone, "model").is_file());
        assert!(!half.exists());

        fs::remove_file(layout::extract(&dir, &zone, Side::Vf)).unwrap();
        let mut split = Split {
            voice: 0.5,
            off_by: 0,
        };
        let error = make(&dir, &zone, "model", RATE, &[], &mut split, &mut |_| {}).unwrap_err();
        assert!(error.to_string().contains("lost its input"), "{error}");
        fs::remove_dir_all(&dir).unwrap();
    }
}
