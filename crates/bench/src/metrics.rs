//! A separation output measured against a reference.
//!
//! There is no ground truth — nobody has the isolated stems of a film mix —
//! so the reference is the output of the models rated best by ear, and every
//! number here is relative to it: close to the reference, not perfect. That
//! is measurable, repeatable, and enough to rank models and to see a
//! regression.
//!
//! - **Lag**: where the output sits against the reference (cross-correlation,
//!   ±[`LAG_MAX`]), undone before anything else is measured, so a model that
//!   is a few samples late is not marked down for it.
//! - **SI-SDR**: signal to error, blind to gain (Le Roux et al. 2019).
//! - **Residue, loss, artifacts**: by frames of [`FRAME`] samples, the output
//!   is projected by least squares onto the reference's two stems, the target
//!   and the other: `c ≈ a·target + b·other + e`. The residue is the energy of
//!   `b·other` (the other stem leaking in), the loss the energy of the target
//!   that is missing (`a < 1`), the artifacts the energy of `e`. It is
//!   BSS-eval's SDR/SIR/SAR split, without the allowed-distortion filters.
//! - **Gap by bands** ([`gaps`]): how far the output's spectrum is from the
//!   reference's, frame by frame, in two directions — louder and quieter.
//!
//! Some products are made in single precision on purpose: an imported
//! legacy set is measured again here, and the numbers have to be the ones
//! it came with.

use crate::set::{Metric, RATE};
use realfft::RealFftPlanner;
use std::borrow::Cow;

/// How far a lag is searched: ±100 ms.
pub const LAG_MAX: usize = 4800;
/// The lag is first searched on the signals summed by blocks of this many
/// samples, then refined within one block at the full rate.
pub const DECIMATION: usize = 16;
/// Frame of the projection onto the two stems: 200 ms.
pub const FRAME: usize = 9600;
/// The gap's transform and its hop.
pub const FFT: usize = 2048;
pub const HOP: usize = 1024;
/// The gap's bands, logarithmic from 50 Hz to 20 kHz.
pub const BANDS: usize = 24;
/// The least normalised correlation a lag is believed at.
pub const MIN_CORRELATION: f64 = 0.2;
const EPSILON: f64 = 1e-12;

/// What [`compare`] measures. A value is none when it has no meaning: the
/// reference is silent, or the signals are shorter than a frame.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Comparison {
    pub si_sdr: Option<f64>,
    pub residue: Option<f64>,
    pub loss: Option<f64>,
    pub artifacts: Option<f64>,
    /// Samples the output is late on the reference by.
    pub lag: i32,
}

impl Comparison {
    fn none(lag: i32) -> Self {
        Self {
            lag,
            ..Self::default()
        }
    }
}

/// Compare the output `candidate` with the reference `target` — the same
/// stem in the reference — `other` being the reference's other stem: the
/// rest if the target is the voice, and the reverse. All three are mono at
/// the same rate and cut from the same source, so the same length within a
/// few samples.
pub fn compare(target: &[f32], other: &[f32], candidate: &[f32]) -> Comparison {
    let n = target.len().min(other.len()).min(candidate.len());
    if n < FRAME {
        return Comparison::none(0);
    }
    let lag = lag_within(target, candidate, n);
    // Only where the shifted output exists: zeros at an edge would put a
    // ceiling on the SI-SDR.
    let start = (-(lag as i64)).max(0) as usize;
    let end = (n as i64).min(n as i64 - lag as i64) as usize;
    if end < start + FRAME {
        return Comparison::none(lag);
    }
    let r = window(target, start, end, 0);
    let o = window(other, start, end, 0);
    let c = window(candidate, start, end, lag);

    let rr = dot(&r, &r);
    if rr < EPSILON {
        return Comparison::none(lag);
    }
    // SI-SDR: the output projected onto the reference, gain free; what is
    // left is the error.
    let alpha = dot(&c, &r) / rr;
    let target_energy = alpha * alpha * rr;
    let error: f64 = c
        .iter()
        .zip(&r)
        .map(|(c, r)| {
            let e = c - alpha * r;
            e * e
        })
        .sum();
    let si_sdr =
        (10.0 * (target_energy.max(EPSILON) / error.max(EPSILON)).log10()).clamp(-40.0, 60.0);

    let (mut e_target, mut e_other, mut e_artifacts, mut e_loss, mut e_reference) =
        (0.0, 0.0, 0.0, 0.0, 0.0);
    for t0 in (0..c.len()).step_by(FRAME) {
        let t1 = (t0 + FRAME).min(c.len());
        let (mut tt, mut oo, mut to, mut ct, mut co) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in t0..t1 {
            tt += r[i] * r[i];
            oo += o[i] * o[i];
            to += r[i] * o[i];
            ct += c[i] * r[i];
            co += c[i] * o[i];
        }
        let determinant = tt * oo - to * to;
        let (a, b) = if determinant > 1e-9 * (tt * oo).max(EPSILON) {
            (
                (ct * oo - co * to) / determinant,
                (co * tt - ct * to) / determinant,
            )
        } else {
            // The stems are collinear or one is silent over the frame: the
            // target alone, or the other if the target is the silent one.
            (
                if tt > EPSILON { ct / tt } else { 0.0 },
                if tt > EPSILON || oo < EPSILON {
                    0.0
                } else {
                    co / oo
                },
            )
        };
        let mut artifacts = 0.0;
        for i in t0..t1 {
            let e = c[i] - a * r[i] - b * o[i];
            artifacts += e * e;
        }
        e_target += a * a * tt;
        e_other += b * b * oo;
        e_artifacts += artifacts;
        e_reference += tt;
        if a < 1.0 {
            e_loss += (1.0 - a) * (1.0 - a) * tt;
        }
    }
    Comparison {
        si_sdr: Some(si_sdr),
        residue: decibels(e_other, e_target),
        loss: decibels(e_loss, e_reference),
        artifacts: decibels(e_artifacts, e_target),
        lag,
    }
}

/// A ratio of energies in dB, within ±100. No useful part: +100, all of it
/// is unwanted; neither: no value.
fn decibels(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator < EPSILON {
        return (numerator > EPSILON).then_some(100.0);
    }
    Some((10.0 * (numerator.max(EPSILON) / denominator).log10()).clamp(-100.0, 100.0))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// `[start, end)` of a signal shifted by `lag` samples; outside it, zero.
fn window(signal: &[f32], start: usize, end: usize, lag: i32) -> Vec<f64> {
    (start..end)
        .map(|i| {
            let k = i as i64 + lag as i64;
            if k >= 0 && (k as usize) < signal.len() {
                signal[k as usize] as f64
            } else {
                0.0
            }
        })
        .collect()
}

/// An output shifted back by the lag found for it: `y[i] = x[i + lag]`,
/// zero outside. The gap by bands is measured on it, aligned as the SI-SDR
/// was.
pub fn shifted(candidate: &[f32], lag: i32) -> Cow<'_, [f32]> {
    if lag == 0 {
        return Cow::Borrowed(candidate);
    }
    Cow::Owned(
        (0..candidate.len())
            .map(|i| {
                let k = i as i64 + lag as i64;
                if k >= 0 && (k as usize) < candidate.len() {
                    candidate[k as usize]
                } else {
                    0.0
                }
            })
            .collect(),
    )
}

/// How many samples `candidate` is late on `reference`, over their common
/// length. 0 when they do not resemble each other: an output unrelated to
/// the reference is not given a lag at random.
pub fn lag(reference: &[f32], candidate: &[f32]) -> i32 {
    lag_within(reference, candidate, reference.len().min(candidate.len()))
}

/// The lag over the first `end` samples: the peak of the cross-correlation
/// of the decimated signals, then refined at the full rate around it.
fn lag_within(r: &[f32], c: &[f32], end: usize) -> i32 {
    let n = end / DECIMATION;
    if n < 4 {
        return 0;
    }
    let sums = |x: &[f32]| -> Vec<f64> {
        let (blocks, _) = x[..n * DECIMATION].as_chunks::<DECIMATION>();
        blocks
            .iter()
            .map(|block| block.iter().map(|&v| v as f64).sum())
            .collect()
    };
    let (rd, cd) = (sums(r), sums(c));
    let (er, ec) = (dot(&rd, &rd), dot(&cd, &cd));
    if er < EPSILON || ec < EPSILON {
        return 0;
    }
    let reach = (LAG_MAX / DECIMATION) as i64;
    let (mut coarse, mut best) = (0i64, f64::NEG_INFINITY);
    for l in -reach..=reach {
        let from = (-l).max(0) as usize;
        // Nothing overlaps when the shift is longer than the signal.
        let to = (n as i64).min(n as i64 - l).max(0) as usize;
        let mut s = 0.0;
        for i in from..to {
            s += rd[i] * cd[(i as i64 + l) as usize];
        }
        if s > best {
            best = s;
            coarse = l;
        }
    }
    if best / (er * ec).sqrt() < MIN_CORRELATION {
        return 0;
    }
    let centre = coarse * DECIMATION as i64;
    let (mut fine, mut best) = (0i64, f64::NEG_INFINITY);
    for l in centre - DECIMATION as i64..=centre + DECIMATION as i64 {
        let mut s = 0.0f64;
        for (i, &r) in r[..end].iter().enumerate() {
            let k = i as i64 + l;
            if k >= 0 && (k as usize) < c.len() {
                // A single-precision product, as an imported set's numbers were made.
                s += (r * c[k as usize]) as f64;
            }
        }
        if s > best {
            best = s;
            fine = l;
        }
    }
    fine as i32
}

/// The gap of an output to the reference, frame by frame (frame `t` is the
/// samples `[t·HOP, t·HOP + FFT)`), in two directions: where the output is
/// louder than the reference, and where it is quieter. Their sum is the gap.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Gaps {
    pub excess: Vec<f32>,
    pub lack: Vec<f32>,
}

impl Gaps {
    pub fn len(&self) -> usize {
        self.excess.len()
    }

    pub fn is_empty(&self) -> bool {
        self.excess.is_empty()
    }

    /// The most of each direction over the frames that cover the samples
    /// `[from, to)` — one column of a waveform: a short defect stays visible
    /// when the view is zoomed out.
    pub fn max_over(&self, from: u64, to: u64) -> (f32, f32) {
        if self.is_empty() {
            return (0.0, 0.0);
        }
        let last = self.len() as i64 - 1;
        let first = ((from as i64 - FFT as i64) / HOP as i64 + 1).clamp(0, last);
        let end = (to as i64 / HOP as i64).clamp(first, last);
        let (mut excess, mut lack) = (0.0f32, 0.0f32);
        for t in first as usize..=end as usize {
            excess = excess.max(self.excess[t]);
            lack = lack.max(self.lack[t]);
        }
        (excess, lack)
    }

    /// The two directions averaged over the frames, as a measure keeps them.
    /// None when there is no frame.
    pub fn means(&self) -> Option<(f64, f64)> {
        if self.is_empty() {
            return None;
        }
        // Each mean is rounded to single precision, as an imported set's are.
        let mean =
            |x: &[f32]| (x.iter().map(|&v| v as f64).sum::<f64>() / x.len() as f64) as f32 as f64;
        Some((mean(&self.excess), mean(&self.lack)))
    }
}

/// The gap by bands of `candidate` to `reference`, frame by frame.
///
/// A gap of spectra by bands rather than of samples: it is blind to phase (a
/// reverberation a few samples off sounds the same, and a local SDR would
/// count it a large error), and it sees both reasons an output is rejected
/// for — signal too many, a leak, and signal missing, a lack — band by band.
/// [`BANDS`] logarithmic bands from 50 Hz to 20 kHz, close to the ear's
/// scale; levels in dB with a floor 60 dB under the reference's loudest band
/// (silence would otherwise make huge gaps nobody hears); the gap of a frame
/// is the mean over the bands, each weighted by its level above the floor.
/// The output's overall gain is fitted first, as for the SI-SDR: a model a
/// little louder everywhere is not far. Frames are smoothed over three.
///
/// Calibrated on the test set: the references stay under about 2 dB in all
/// on nine frames out of ten, the rejects reach 5 to 30 dB.
pub fn gaps(reference: &[f32], candidate: &[f32]) -> Gaps {
    let n = reference.len().min(candidate.len());
    if n < FFT {
        return Gaps::default();
    }
    let frames = (n - FFT) / HOP + 1;
    let (mut cc, mut cr) = (0.0f64, 0.0f64);
    for i in 0..n {
        cc += candidate[i] as f64 * candidate[i] as f64;
        cr += candidate[i] as f64 * reference[i] as f64;
    }
    // The gain that brings the output closest to the reference.
    let gain = if cc > EPSILON && cr > 0.0 {
        cr / cc
    } else {
        1.0
    };
    let bin = RATE as f64 / FFT as f64;
    let edges: Vec<usize> = (0..=BANDS)
        .map(|b| {
            let hertz = 50.0 * (20_000.0f64 / 50.0).powf(b as f64 / BANDS as f64);
            ((hertz / bin).round_ties_even() as usize).clamp(1, FFT / 2)
        })
        .collect();
    let hann: Vec<f64> = (0..FFT)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / FFT as f64).cos())
        .collect();

    let fft = RealFftPlanner::<f64>::new().plan_fft_forward(FFT);
    let mut input = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();
    let mut scratch = fft.make_scratch_vec();
    let mut bands = |signal: &[f32], scale: f64, out: &mut [f64]| {
        for (i, x) in input.iter_mut().enumerate() {
            *x = signal[i] as f64 * scale * hann[i];
        }
        fft.process_with_scratch(&mut input, &mut spectrum, &mut scratch)
            .expect("buffers of the plan's own sizes");
        for (b, out) in out.iter_mut().enumerate() {
            // A band narrower than a bin still takes the one it starts in.
            let to = edges[b + 1].max(edges[b] + 1);
            *out = spectrum[edges[b]..to].iter().map(|z| z.norm_sqr()).sum();
        }
    };
    let mut er = vec![0.0f64; frames * BANDS];
    let mut ec = vec![0.0f64; frames * BANDS];
    for t in 0..frames {
        let at = t * HOP;
        let row = t * BANDS..(t + 1) * BANDS;
        bands(&reference[at..at + FFT], 1.0, &mut er[row.clone()]);
        bands(&candidate[at..at + FFT], gain, &mut ec[row]);
    }
    let loudest = er.iter().copied().fold(0.0f64, f64::max);
    let mut excess = vec![0.0f32; frames];
    let mut lack = vec![0.0f32; frames];
    if loudest < EPSILON {
        // A silent reference: nothing to compare with.
        return Gaps { excess, lack };
    }
    let floor = loudest * 1e-6;
    let floor_db = 10.0 * floor.log10();
    for t in 0..frames {
        let (mut more, mut less, mut weights) = (0.0, 0.0, 0.0);
        for b in 0..BANDS {
            let lr = 10.0 * er[t * BANDS + b].max(floor).log10();
            let lc = 10.0 * ec[t * BANDS + b].max(floor).log10();
            // An audible band counts for more than one at the floor.
            let weight = lr.max(lc) - floor_db;
            if lc > lr {
                more += weight * (lc - lr);
            } else {
                less += weight * (lr - lc);
            }
            weights += weight;
        }
        if weights > 1e-9 {
            excess[t] = (more / weights) as f32;
            lack[t] = (less / weights) as f32;
        }
    }
    Gaps {
        excess: smoothed(&excess),
        lack: smoothed(&lack),
    }
}

/// Each frame averaged with its two neighbours.
fn smoothed(x: &[f32]) -> Vec<f32> {
    (0..x.len())
        .map(|t| {
            let from = t.saturating_sub(1);
            let to = (t + 1).min(x.len() - 1);
            let mut sum = 0.0f32;
            for value in &x[from..=to] {
                sum += value;
            }
            sum / (to - from + 1) as f32
        })
        .collect()
}

/// Everything a measure keeps of one stem: the comparison, and the gap by
/// bands of the output shifted back by the lag found. `reference` says the
/// output is one of the references of its side — a mark, it is measured
/// like any other.
pub fn metric(target: &[f32], other: &[f32], candidate: &[f32], reference: bool) -> Metric {
    let comparison = compare(target, other, candidate);
    let mut metric = Metric {
        reference,
        si_sdr: comparison.si_sdr,
        residue: comparison.residue,
        loss: comparison.loss,
        artifacts: comparison.artifacts,
        lag: comparison.lag,
        ..Metric::default()
    };
    // A silent reference: a gap would mean nothing, and certainly not
    // "perfect".
    let energy: f64 = target.iter().map(|&v| v as f64 * v as f64).sum();
    if energy < 1e-9 {
        return metric;
    }
    if let Some((excess, lack)) = gaps(target, &shifted(candidate, comparison.lag)).means() {
        metric.excess = Some(excess);
        metric.lack = Some(lack);
        metric.gap = Some(excess + lack);
    }
    metric
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A repeatable noise in ±1, without a dependency for it.
    fn noise(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn tone(hertz: f64, n: usize, level: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                level * (2.0 * std::f64::consts::PI * hertz * i as f64 / RATE as f64).sin() as f32
            })
            .collect()
    }

    fn mix(a: &[f32], gain_a: f32, b: &[f32], gain_b: f32) -> Vec<f32> {
        a.iter()
            .zip(b)
            .map(|(a, b)| gain_a * a + gain_b * b)
            .collect()
    }

    const N: usize = 4 * RATE as usize;

    #[test]
    fn the_reference_itself_is_as_close_as_it_gets() {
        let (target, other) = (noise(1, N), noise(2, N));
        let c = compare(&target, &other, &target);
        assert_eq!(c.lag, 0);
        assert_eq!(c.si_sdr, Some(60.0));
        assert_eq!(c.residue, Some(-100.0));
        assert_eq!(c.loss, Some(-100.0));
        assert_eq!(c.artifacts, Some(-100.0));
        let m = metric(&target, &other, &target, true);
        assert!(m.reference);
        assert_eq!((m.gap, m.excess, m.lack), (Some(0.0), Some(0.0), Some(0.0)));
    }

    #[test]
    fn the_si_sdr_is_blind_to_gain() {
        let (target, other) = (noise(1, N), noise(2, N));
        let quiet: Vec<f32> = target.iter().map(|v| 0.25 * v).collect();
        let c = compare(&target, &other, &quiet);
        assert!(c.si_sdr.unwrap() > 59.0);
        // A quarter of the level is three quarters of the target missing.
        let expected = 20.0 * 0.75f64.log10();
        assert!((c.loss.unwrap() - expected).abs() < 0.01, "{:?}", c.loss);
        // And the gap fits the gain first: a quieter output is not far.
        let m = metric(&target, &other, &quiet, false);
        assert!(m.gap.unwrap() < 1e-3, "{:?}", m.gap);
    }

    #[test]
    fn a_leak_of_the_other_stem_is_the_residue() {
        let (target, other) = (noise(1, N), noise(2, N));
        let leaking = mix(&target, 1.0, &other, 0.1);
        let c = compare(&target, &other, &leaking);
        // A tenth of the other stem: 20 dB under the target.
        assert!((c.residue.unwrap() + 20.0).abs() < 0.1, "{:?}", c.residue);
        assert!((c.si_sdr.unwrap() - 20.0).abs() < 0.1, "{:?}", c.si_sdr);
        assert!(c.artifacts.unwrap() < -60.0, "{:?}", c.artifacts);
        assert!(c.loss.unwrap() < -40.0, "{:?}", c.loss);
    }

    #[test]
    fn what_is_neither_stem_is_artifacts() {
        let (target, other, stranger) = (noise(1, N), noise(2, N), noise(3, N));
        let damaged = mix(&target, 1.0, &stranger, 0.1);
        let c = compare(&target, &other, &damaged);
        assert!(
            (c.artifacts.unwrap() + 20.0).abs() < 0.1,
            "{:?}",
            c.artifacts
        );
        assert!(c.residue.unwrap() < -40.0, "{:?}", c.residue);
    }

    #[test]
    fn a_late_output_is_shifted_back_before_it_is_measured() {
        let (target, other) = (noise(1, N), noise(2, N));
        for late in [37i32, -250, 1601] {
            let moved: Vec<f32> = (0..N as i64)
                .map(|i| {
                    let k = i - late as i64;
                    if (0..N as i64).contains(&k) {
                        target[k as usize]
                    } else {
                        0.0
                    }
                })
                .collect();
            assert_eq!(lag(&target, &moved), late);
            let c = compare(&target, &other, &moved);
            assert_eq!(c.lag, late);
            assert_eq!(c.si_sdr, Some(60.0), "late by {late}");
            assert_eq!(&shifted(&moved, late)[2000..3000], &target[2000..3000]);
        }
    }

    #[test]
    fn a_signal_shorter_than_the_search_is_still_searched() {
        // 64 samples against a search of ±4800: most shifts overlap nothing.
        let short = noise(5, 64);
        assert_eq!(lag(&short, &short), 0);
        let late: Vec<f32> = (0..64)
            .map(|i| if i >= 3 { short[i - 3] } else { 0.0 })
            .collect();
        assert_eq!(lag(&short, &late), 3);
    }

    #[test]
    fn an_unrelated_output_is_given_no_lag() {
        let (target, stranger) = (noise(1, N), noise(3, N));
        assert_eq!(lag(&target, &stranger), 0);
        assert_eq!(lag(&target, &vec![0.0; N]), 0);
        assert_eq!(lag(&target[..40], &target[..40]), 0);
    }

    #[test]
    fn a_silent_reference_or_a_short_one_measures_nothing() {
        let other = noise(2, N);
        let silent = vec![0.0f32; N];
        let m = metric(&silent, &other, &other, false);
        assert_eq!(m.si_sdr, None);
        assert_eq!(m.gap, None);
        let short = noise(1, FRAME - 1);
        assert_eq!(compare(&short, &short, &short), Comparison::none(0));
        assert!(gaps(&short[..FFT - 1], &short[..FFT - 1]).is_empty());
    }

    #[test]
    fn a_silent_other_stem_leaves_the_target_alone() {
        // The projection falls back on the target when the other is silent.
        let target = noise(1, N);
        let silent = vec![0.0f32; N];
        let half: Vec<f32> = target.iter().map(|v| 0.5 * v).collect();
        let c = compare(&target, &silent, &half);
        assert_eq!(c.residue, Some(-100.0));
        assert!((c.loss.unwrap() - 20.0 * 0.5f64.log10()).abs() < 0.01);
    }

    #[test]
    fn the_gap_tells_too_much_from_too_little() {
        // A reference of two tones; one output adds a third, the other
        // loses one.
        let low = tone(300.0, N, 0.3);
        let high = tone(3000.0, N, 0.3);
        let reference = mix(&low, 1.0, &high, 1.0);
        let extra = mix(&reference, 1.0, &tone(900.0, N, 0.3), 1.0);
        let g = gaps(&reference, &extra);
        assert_eq!(g.len(), (N - FFT) / HOP + 1);
        let (excess, lack) = g.means().unwrap();
        assert!(excess > 1.0 && excess > 5.0 * lack, "{excess} {lack}");

        let g = gaps(&reference, &low);
        let (excess, lack) = g.means().unwrap();
        assert!(lack > 1.0 && lack > 5.0 * excess, "{excess} {lack}");
    }

    #[test]
    fn a_short_defect_shows_in_the_column_that_covers_it() {
        let reference = tone(300.0, N, 0.3);
        let mut output = reference.clone();
        let burst = tone(4000.0, 4800, 0.3);
        let at = 2 * RATE as usize;
        for (o, b) in output[at..at + 4800].iter_mut().zip(&burst) {
            *o += b;
        }
        let g = gaps(&reference, &output);
        let (inside, _) = g.max_over(at as u64, at as u64 + 4800);
        let (outside, _) = g.max_over(0, RATE as u64);
        assert!(inside > 1.0, "{inside}");
        assert!(outside < 0.05, "{outside}");
        assert_eq!(Gaps::default().max_over(0, 100), (0.0, 0.0));
    }
}
