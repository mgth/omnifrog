//! ITU-R BS.1770 loudness of a mix of isolated voices.
//!
//! This is the one measurement the whole levelling rests on: the dialogue
//! of the original and the dialogue of the dub are each measured here, and
//! their difference is the gain the dubbed voices are brought up or down by
//! before they are placed. A ratio of two numbers from the same formula, so
//! what matters is that both sides go through it, constant for constant.
//!
//! Two things make it more than a textbook integration. The voices are
//! measured *together*, as the mix they will be heard as, not one by one
//! and averaged. And the channel weighting is not fixed: BS.1770 gives a
//! surround channel +1.5 dB, and an object is not a channel — it moves. So
//! `G(t)` follows each object's trajectory, front while it is in front and
//! surround while it is behind, sample by sample. A dub's channels do not
//! move, and take the weight their name implies.
//!
//! The K filter is two biquads whose coefficients are derived for the
//! actual rate rather than read from the 48 kHz table, so a stem still
//! labelled 44.1 kHz measures right.

use crate::progress::{Event, Progress};
use crate::wav;
use crate::{Error, Result};
use std::path::{Path, PathBuf};

/// BS.1770 constants: the offset of the loudness scale, the absolute gate,
/// and the gate relative to what the absolute one kept.
const OFFSET: f64 = -0.691;
const ABSOLUTE_GATE: f64 = -70.0;
const RELATIVE_GATE: f64 = -10.0;

/// Channel weights: a surround channel counts for 1.5 dB more than a front
/// one. The LFE is never measured and must never be passed.
pub const G_FRONT: f64 = 1.0;
pub const G_SURROUND: f64 = 1.41;
/// Where the front half of the room ends, on the axis that runs from the
/// back of the room (-1) to the screen (+1).
pub const FRONT_LIMIT: f64 = 0.5;

/// What a measurement says. `lkfs` is the integrated, gated loudness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measure {
    pub lkfs: f64,
    /// 400 ms blocks in all, and the ones the gates kept.
    pub blocks: usize,
    pub kept: usize,
    pub seconds: f64,
}

impl Measure {
    /// Whether there was enough sound to measure: a valid loudness is
    /// below zero, and a silent or very short file has none.
    pub fn valid(&self) -> bool {
        self.blocks > 0 && self.lkfs.is_finite()
    }

    /// The loudness, or 0 when there was nothing to measure — which is how
    /// the manifest records "not measured", valid values being negative.
    pub fn or_unmeasured(&self) -> f64 {
        if self.valid() { self.lkfs } else { 0.0 }
    }
}

/// One mono voice of the mix, and how it is weighted.
#[derive(Debug, Clone, PartialEq)]
pub struct Voice {
    pub path: PathBuf,
    /// Where the object is over time, as pairs of (sample, position). Empty
    /// for a channel, which does not move.
    pub trajectory: Vec<(u64, [f64; 3])>,
    /// The weight of a voice that does not move.
    pub surround: bool,
}

impl Voice {
    /// A channel of the dub, weighted by where its speaker is.
    pub fn channel(path: impl Into<PathBuf>, name: &str) -> Self {
        Self {
            path: path.into(),
            trajectory: Vec::new(),
            surround: is_surround(name),
        }
    }

    /// An object of the original, weighted by where it is at each instant.
    pub fn object(path: impl Into<PathBuf>, trajectory: Vec<(u64, [f64; 3])>) -> Self {
        Self {
            path: path.into(),
            trajectory,
            surround: false,
        }
    }

    fn moving(&self) -> bool {
        !self.trajectory.is_empty()
    }
}

/// Whether a channel is a surround one, by its name: the bed's own names
/// (`Lss`, `Rrs`) and the ones other layouts use (`Ls`, `SL`, `BL`). The
/// LFE is not a channel this asks about.
pub fn is_surround(name: &str) -> bool {
    let name = name.trim().to_ascii_uppercase();
    matches!(name.as_str(), "LS" | "RS" | "LSS" | "RSS" | "LRS" | "RRS")
        || name.starts_with('S')
        || name.starts_with('B')
}

/// The weight a position earns: behind the front limit is surround.
fn weight(position: &[f64; 3]) -> f64 {
    if position[1] < FRONT_LIMIT {
        G_SURROUND
    } else {
        G_FRONT
    }
}

/// Measure the mix of `voices`.
///
/// `rate` overrides what the headers say, for stems still labelled at the
/// model's rate; `None` takes the rate of the first voice.
pub fn measure(voices: &[Voice], rate: Option<u32>, progress: Progress) -> Result<Measure> {
    if voices.is_empty() {
        return Ok(Measure {
            lkfs: f64::NEG_INFINITY,
            blocks: 0,
            kept: 0,
            seconds: 0.0,
        });
    }
    let rate = match rate {
        Some(rate) => rate,
        None => wav::read_header(&voices[0].path)?.sample_rate,
    };
    if rate == 0 {
        return Err(Error::msg("a voice states no sample rate"));
    }
    // A segment is 100 ms; a gating block is four of them, overlapping by
    // three quarters.
    let segment = (rate / 10).max(1) as usize;

    // Every voice is independent, and by now every one of them is in the
    // page cache: the filtering is what costs, so it runs one voice per
    // thread.
    let mut energies: Vec<Result<(Vec<f64>, usize)>> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = voices
            .iter()
            .map(|voice| scope.spawn(move || per_segment(voice, rate, segment)))
            .collect();
        for (voice, handle) in voices.iter().zip(handles) {
            energies.push(match handle.join() {
                Ok(result) => result,
                Err(_) => Err(Error::msg(format!(
                    "measuring {} stopped unexpectedly",
                    voice.path.display()
                ))),
            });
        }
    });
    let energies: Vec<(Vec<f64>, usize)> = energies.into_iter().collect::<Result<_>>()?;
    progress(Event::message(format!(
        "measured {} voice{}",
        voices.len(),
        if voices.len() > 1 { "s" } else { "" }
    )));

    // The mix runs as long as its longest voice; the tail of under 100 ms
    // that no voice completes is left out.
    let segments = energies.iter().map(|(_, whole)| *whole).max().unwrap_or(0);
    let seconds = (segments * segment) as f64 / rate as f64;
    if segments < 4 {
        return Ok(Measure {
            lkfs: f64::NEG_INFINITY,
            blocks: 0,
            kept: 0,
            seconds,
        });
    }
    let mut mix = vec![0.0f64; segments];
    for (energy, _) in &energies {
        for (into, from) in mix.iter_mut().zip(energy) {
            *into += from;
        }
    }

    let blocks = segments - 3;
    let denominator = 4.0 * segment as f64;
    let power: Vec<f64> = (0..blocks)
        .map(|j| (mix[j] + mix[j + 1] + mix[j + 2] + mix[j + 3]) / denominator)
        .collect();

    let absolute = 10f64.powf((ABSOLUTE_GATE - OFFSET) / 10.0);
    let (mut loud, mut count) = (0.0, 0usize);
    for p in &power {
        if *p >= absolute {
            loud += p;
            count += 1;
        }
    }
    if count == 0 {
        return Ok(Measure {
            lkfs: f64::NEG_INFINITY,
            blocks,
            kept: 0,
            seconds,
        });
    }
    let mean = OFFSET + 10.0 * (loud / count as f64).log10();
    let relative = 10f64.powf((mean + RELATIVE_GATE - OFFSET) / 10.0);
    let (mut kept_sum, mut kept) = (0.0, 0usize);
    for p in &power {
        if *p >= absolute && *p >= relative {
            kept_sum += p;
            kept += 1;
        }
    }
    if kept == 0 {
        kept_sum = loud;
        kept = count;
    }
    Ok(Measure {
        lkfs: OFFSET + 10.0 * (kept_sum / kept as f64).log10(),
        blocks,
        kept,
        seconds,
    })
}

/// The K-weighted energy of one voice, accumulated per 100 ms segment, and
/// how many of those segments are whole. The last, partial one is included
/// — a longer voice fills the rest of it — but not counted.
fn per_segment(voice: &Voice, rate: u32, segment: usize) -> Result<(Vec<f64>, usize)> {
    let filter = KFilter::new(rate as f64);
    let mut state = FilterState::default();

    // The weight, and the sample at which it may change. A voice that does
    // not move has no next change, so the test below is never true.
    let mut g;
    let mut next = u64::MAX;
    let mut at = 0usize;
    if voice.moving() {
        g = weight(&voice.trajectory[0].1);
        if voice.trajectory.len() > 1 {
            next = voice.trajectory[1].0;
        }
    } else {
        g = if voice.surround { G_SURROUND } else { G_FRONT };
    }

    let mut reader = wav::Reader::open(&voice.path)?;
    let mut block = vec![0f32; wav::BLOCK];
    let mut segments: Vec<f64> = Vec::new();
    let (mut sum, mut count, mut whole, mut position) = (0.0f64, 0usize, 0usize, 0u64);
    loop {
        let read = reader.read_block(&mut block)?;
        if read == 0 {
            break;
        }
        for sample in &block[..read] {
            if position >= next {
                while at + 1 < voice.trajectory.len() && voice.trajectory[at + 1].0 <= position {
                    at += 1;
                }
                g = weight(&voice.trajectory[at].1);
                next = voice
                    .trajectory
                    .get(at + 1)
                    .map(|(sample, _)| *sample)
                    .unwrap_or(u64::MAX);
            }
            let y = filter.step(&mut state, *sample as f64);
            sum += g * y * y;
            position += 1;
            count += 1;
            if count == segment {
                segments.push(sum);
                sum = 0.0;
                count = 0;
                whole += 1;
            }
        }
    }
    if count > 0 {
        segments.push(sum);
    }
    Ok((segments, whole))
}

/// The two biquads of the K weighting, for one sample rate.
///
/// The coefficients are derived rather than tabulated, which reproduces the
/// 48 kHz table of the recommendation to about 1e-16 and is exact at every
/// other rate — so a stem still labelled at the model's 44.1 kHz measures
/// as correctly as one labelled 48.
struct KFilter {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    h1: f64,
    h2: f64,
}

#[derive(Default)]
struct FilterState {
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
    u1: f64,
    u2: f64,
    v1: f64,
    v2: f64,
}

impl KFilter {
    fn new(rate: f64) -> Self {
        // Stage one: the head shelf.
        const F0: f64 = 1_681.974_450_955_532;
        const Q: f64 = 0.707_175_236_955_419_6;
        const GAIN_DB: f64 = 3.999_843_853_973_347;
        let k = (std::f64::consts::PI * F0 / rate).tan();
        let vh = 10f64.powf(GAIN_DB / 20.0);
        let vb = vh.powf(0.499_666_774_154_561_6);
        let a0 = 1.0 + k / Q + k * k;
        // Stage two: the RLB high pass, whose numerator is fixed at
        // [1, -2, 1].
        const F0_HIGH: f64 = 38.135_470_876_024_44;
        const Q_HIGH: f64 = 0.500_327_037_323_877_3;
        let kh = (std::f64::consts::PI * F0_HIGH / rate).tan();
        let a0h = 1.0 + kh / Q_HIGH + kh * kh;
        Self {
            b0: (vh + vb * k / Q + k * k) / a0,
            b1: 2.0 * (k * k - vh) / a0,
            b2: (vh - vb * k / Q + k * k) / a0,
            a1: 2.0 * (k * k - 1.0) / a0,
            a2: (1.0 - k / Q + k * k) / a0,
            h1: 2.0 * (kh * kh - 1.0) / a0h,
            h2: (1.0 - kh / Q_HIGH + kh * kh) / a0h,
        }
    }

    #[inline]
    fn step(&self, s: &mut FilterState, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * s.x1 + self.b2 * s.x2 - self.a1 * s.y1 - self.a2 * s.y2;
        s.x2 = s.x1;
        s.x1 = x;
        s.y2 = s.y1;
        s.y1 = y;
        let v = y - 2.0 * s.u1 + s.u2 - self.h1 * s.v1 - self.h2 * s.v2;
        s.u2 = s.u1;
        s.u1 = y;
        s.v2 = s.v1;
        s.v1 = v;
        v
    }
}

/// The loudness of one file measured on its own, for a caller with no mix.
pub fn of(path: &Path, surround: bool, rate: Option<u32>) -> Result<Measure> {
    measure(
        &[Voice {
            path: path.to_path_buf(),
            trajectory: Vec::new(),
            surround,
        }],
        rate,
        &crate::progress::quiet,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::{BufWriter, Write};

    /// A 1 kHz sine at `db` below full scale, sampled at `rate` and
    /// written as a mono 24-bit WAV whose header says `label` — which is
    /// `rate` unless a test wants the no-resample trick.
    fn tone_at(path: &Path, rate: u32, label: u32, seconds: f64, db: f64) {
        let frames = (rate as f64 * seconds) as u64;
        let amplitude = 10f64.powf(db / 20.0);
        let mut out = BufWriter::new(File::create(path).unwrap());
        wav::write_header(
            &mut out,
            wav::Format {
                channels: 1,
                sample_rate: label,
                bits: 24,
                float: false,
                frames,
                data_offset: 0,
            },
        )
        .unwrap();
        for i in 0..frames {
            let t = i as f64 / rate as f64;
            let value = (amplitude * (2.0 * std::f64::consts::PI * 1000.0 * t).sin() * 8_388_608.0)
                .round()
                .clamp(-8_388_608.0, 8_388_607.0) as i32;
            out.write_all(&value.to_le_bytes()[..3]).unwrap();
        }
        out.flush().unwrap();
    }

    fn tone(path: &Path, rate: u32, seconds: f64, db: f64) {
        tone_at(path, rate, rate, seconds, db);
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("omnifrog-loudness-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn a_thousand_hertz_front_channel_measures_its_own_level() {
        // The calibration everyone checks against: a 1 kHz sine of
        // amplitude -20 dBFS on one front channel is -23 LKFS.
        let path = scratch("front.wav");
        tone(&path, 48000, 3.0, -20.0);
        let measure = of(&path, false, None).unwrap();
        assert!(measure.valid());
        assert!(
            (measure.lkfs + 23.0).abs() < 0.05,
            "{} LKFS is not around -23",
            measure.lkfs
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_surround_channel_counts_for_one_and_a_half_decibels_more() {
        let path = scratch("surround.wav");
        tone(&path, 48000, 3.0, -20.0);
        let front = of(&path, false, None).unwrap().lkfs;
        let surround = of(&path, true, None).unwrap().lkfs;
        assert!(
            (surround - front - 1.5).abs() < 0.05,
            "{surround} against {front}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn two_voices_are_measured_as_the_mix_they_will_be_heard_as() {
        // The same tone on two channels is 3 dB more than on one.
        let path = scratch("one.wav");
        tone(&path, 48000, 3.0, -20.0);
        let one = of(&path, false, None).unwrap().lkfs;
        let two = measure(
            &[Voice::channel(&path, "C"), Voice::channel(&path, "L")],
            None,
            &crate::progress::quiet,
        )
        .unwrap()
        .lkfs;
        assert!((two - one - 3.0103).abs() < 0.01, "{two} against {one}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_stem_labelled_at_the_models_rate_measures_at_the_real_one() {
        // 48 kHz samples under a 44.1 kHz header — what the separation
        // hands back, since the input was labelled at the model's rate so
        // that nothing resampled it. Told the truth, the filter and the
        // segment grid are the right ones and the tone measures -23.
        let path = scratch("mislabelled.wav");
        tone_at(&path, 48000, 44100, 3.0, -20.0);
        let told_the_truth = of(&path, false, Some(48000)).unwrap().lkfs;
        assert!((told_the_truth + 23.0).abs() < 0.05, "{told_the_truth}");
        // Taken at face value the tone is read as 1088 Hz, which the head
        // shelf lifts a little more: close, but not the same number.
        let at_face_value = of(&path, false, None).unwrap().lkfs;
        assert!(at_face_value != told_the_truth);
        assert!((at_face_value + 23.0).abs() < 0.3, "{at_face_value}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn silence_has_no_loudness_rather_than_a_wrong_one() {
        let path = scratch("silence.wav");
        tone(&path, 48000, 3.0, -200.0);
        let measure = of(&path, false, None).unwrap();
        assert!(!measure.valid());
        assert_eq!(measure.or_unmeasured(), 0.0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_object_is_weighted_by_where_it_is_at_each_instant() {
        let path = scratch("moving.wav");
        tone(&path, 48000, 4.0, -20.0);
        let front = measure(
            &[Voice::object(&path, vec![(0, [0.0, 1.0, 0.0])])],
            None,
            &crate::progress::quiet,
        )
        .unwrap()
        .lkfs;
        let behind = measure(
            &[Voice::object(&path, vec![(0, [0.0, -1.0, 0.0])])],
            None,
            &crate::progress::quiet,
        )
        .unwrap()
        .lkfs;
        // Front for the first half, behind for the second: between the two.
        let moving = measure(
            &[Voice::object(
                &path,
                vec![(0, [0.0, 1.0, 0.0]), (96_000, [0.0, -1.0, 0.0])],
            )],
            None,
            &crate::progress::quiet,
        )
        .unwrap()
        .lkfs;
        assert!((behind - front - 1.5).abs() < 0.05);
        assert!(moving > front + 0.3 && moving < behind - 0.3, "{moving}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_surround_channel_is_known_by_its_name_in_any_layout() {
        for name in ["Lss", "Rrs", "SL", "BR", "Ls"] {
            assert!(is_surround(name), "{name}");
        }
        for name in ["L", "R", "C"] {
            assert!(!is_surround(name), "{name}");
        }
    }
}
