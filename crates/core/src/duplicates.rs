//! Duplicates between the original and the dub, cancelled in the dub's voices.
//!
//! The new mix is the original's music and effects (`sons_*`) plus the
//! dub's voices (`voix_*`). A voice model leaves part of the dub's own music
//! and effects in its voices — which is the original's music and effects a
//! second time, a few tens of milliseconds later and with a slightly
//! different timbre: one sharp noise plays twice. Measured on a 2 h 43
//! film (Big Beta 7 on both sides): 165 clear duplicates, heard in a blind
//! test 27 times out of 40 (controls: 3 out of 20), as an echo or a double
//! attack.
//!
//! Three corrections, which do NOT go one without the others:
//!
//! 1. **Offset.** Each track is decoded from its first packet (the
//!    original: its raw stream, through harletty; the dub: ffmpeg, which
//!    does not pad the start with silence): sample 0 of each WAV is the
//!    instant ITS track starts in the container. When the two do not start
//!    together (on that film: original 0.084 s, dub 0.006 s), the dub's
//!    voice laid on the original's clock arrives late by the difference
//!    (78 ms). It is put back. ALONE, that makes things worse: what is
//!    left is the dub's own offset (6 to 36 ms depending on the reel), a
//!    comb filter — a metallic sound (22 passages out of 22 by ear).
//!    Hence 2.
//! 2. **Cancellation.** What looks like the original's music and effects,
//!    shifted by the local offset, is taken out of the dub's voices. The
//!    dubbing offset changes along the film (some forty plateaus from 42 to
//!    90 ms on that film):
//!    - a *map*: cross-correlation of the dub's mix against the original's
//!      (the extraction's elements, music and effects in common), 20 s
//!      windows every 5 s, ±150 ms. A window whose peak stands out
//!      ([`SHARPNESS_MIN`]) keeps its offset; the others take the one most
//!      of their sharp neighbours share (the plateaus are discrete: a median
//!      would invent an offset that exists nowhere);
//!    - *leaks*: the same windows, the dub's voices against the original's
//!      music and effects: up to three more sharp peaks besides the plateau;
//!    - *attacks*: two noises 130 ms apart can be offset differently (67
//!      then 42 ms), so every attack of a voice that has a twin in the music
//!      and effects gets ITS offset, measured on the waveforms and applied
//!      from 20 ms before to 300 ms after it.
//!
//!    The cancellation itself: STFT (1024 / 256, root Hann); in each
//!    time-frequency neighbourhood (3 frames × 5 bins) the voice Y is
//!    projected on the shifted music and effects X: α = ⟨Y, X⟩ / ⟨X, X⟩,
//!    coherence g = |⟨Y, X⟩|² / (⟨X, X⟩ ⟨Y, Y⟩); w(g) α X is taken out (w
//!    rises from [`COHERENCE_LOW`] to [`COHERENCE_HIGH`]), never more than
//!    |Y|. Speech that does not look like the music and effects has a low
//!    coherence and stays.
//! 3. **Remux.** The new track lives on the original's clock, which starts
//!    at the original's `start` in the source; the remux shifts it there
//!    ([`crate::remux`]), with or without this pass.
//!
//! **No pairing of the original's elements with the dub's channels.** Each
//! voice channel is compared with the mono SUM of the original's music and
//! effects: a noise is in it whichever object carries it. The local complex
//! gain α absorbs the panning and the timbre, the coherence says where to
//! take out; sound is sparse in time-frequency, one source usually rules a
//! neighbourhood, so the rest of the sum does not get in the way. Limits:
//! two noises in one neighbourhood that came by different paths (one α
//! cannot stand for both: partial cancellation), and a mono sum cancelling
//! itself (the same sound in opposite phase on two objects, rare).
//! Not taken: rendering the original's music and effects to the dub's
//! channel layout for a reference per channel — heavier (a whole render of
//! the objects) and still imperfect (a dub's mix is not the original
//! rendered to 5.1); to revisit only if the limits above are heard.
//!
//! Validated against a prototype heard in 24 passages out of 24 clean, no
//! dub voice damaged. An attack's own offset is kept when the *normalised*
//! correlation of the waveforms reaches [`ATTACK_CORRELATION_MIN`], as in
//! the prototype: a threshold compared with a correlation left scaled by
//! its FFT size keeps almost every attack.

use crate::progress::{Event, Progress};
use crate::wav::{self, Format};
use crate::{Error, Result};
use rayon::prelude::*;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// The only rate this pass knows: every master set is at it.
pub const RATE: i64 = 48_000;

/// STFT of the cancellation and of the attacks.
const NFFT: usize = 1024;
const HOP: usize = 256;
const BINS: usize = NFFT / 2 + 1;
/// About 5 s per cancellation block, a multiple of the hop.
const BLOCK: usize = 939 * HOP;
/// Either side of a block: its edge frames are whole, so blocks join
/// without a seam.
const MARGIN: usize = 16 * HOP;
/// Blocks cancelled at once, every channel of each in parallel.
const BLOCKS_AT_ONCE: usize = 8;
/// The coherence ramp: nothing taken out below the low end, all of it above
/// the high one.
const COHERENCE_LOW: f64 = 0.3;
const COHERENCE_HIGH: f64 = 0.7;
/// The map: 20 s windows every 5 s, the offset looked for over ±150 ms.
const MAP_WINDOW: i64 = 20 * RATE;
const MAP_STEP: i64 = 5 * RATE;
const SEARCH: i64 = 150 * RATE / 1000;
/// A window is the sum of the 5 s slices it is made of: each slice is read
/// and transformed once, rather than four times in overlapping windows.
const SLICES_PER_WINDOW: usize = (MAP_WINDOW / MAP_STEP) as usize;
/// Windows computed per batch; the slices no later window needs are dropped.
const MAP_BATCH: usize = 64;
/// Two peaks less than 1 ms apart are one peak.
const PEAK_GAP: usize = (RATE / 1000) as usize;
/// Main peak over the best other one. The offset is exact even when the
/// correlation is low (speech rules the mix); what counts is that ONE
/// offset stands out.
const SHARPNESS_MIN: f64 = 1.5;
/// A leak besides the plateau is kept when its peak is at least half the
/// main one.
const LEAK_SHARE_MIN: f64 = 0.5;
/// Sharp windows a doubtful one consults: ±30 s.
const VOTE_REACH: f64 = 30.0;
/// Two offsets within 0.25 ms are the same plateau.
const TOLERANCE: i32 = 12;
/// Attacks: 24 logarithmic bands from 150 Hz to 20 kHz.
const BANDS: usize = 24;
/// Mean rise (dB over 2 frames) that makes an attack.
const RISE_MIN: f64 = 6.0;
/// A twin is looked for in the music and effects within ±28 frames (±150 ms).
const PAIRING: i64 = 28;
/// Rises that look alike (cosine), then waveforms that match (normalised
/// correlation).
const SIMILARITY_MIN: f64 = 0.6;
const ATTACK_CORRELATION_MIN: f64 = 0.5;
/// Cancellation at an attack's own offset: from 20 ms before to 300 ms after.
const BEFORE_ATTACK: i64 = RATE / 50;
const AFTER_ATTACK: i64 = RATE * 3 / 10;
/// Attack frames per block of the attack search: 30 s.
const ATTACK_FRAMES: usize = 5625;
/// Frames computed either side of an attack block.
const ATTACK_EDGE: usize = 4;

/// What the pass did, as the manifest keeps it and a progress line says it.
/// An empty map (`windows == 0`: a range shorter than one map window)
/// means nothing was cancelled: no attacks, no `removed_db`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    /// How far the dub's voices were moved onto the original's clock, in
    /// ms; > 0 = they were late.
    pub offset_ms: f64,
    /// Windows of the map, how many had a sharp peak, how many a leak too.
    pub windows: usize,
    pub sharp: usize,
    pub with_leaks: usize,
    /// The dubbing's offset, from its lowest to its highest plateau, in ms.
    pub lag_min_ms: f64,
    pub lag_max_ms: f64,
    /// Attacks cancelled at their own offset.
    pub attacks: usize,
    /// Energy taken out of each voice channel, in dB below the voice.
    pub removed_db: Vec<(String, f64)>,
}

impl Report {
    pub fn text(&self) -> String {
        if self.windows == 0 {
            return format!(
                "voices moved by {:.1} ms onto the original's clock; range too short for the \
                 dubbing map (under {} s): duplicates not cancelled",
                self.offset_ms,
                MAP_WINDOW / RATE
            );
        }
        let removed = self
            .removed_db
            .iter()
            .map(|(channel, db)| format!("{channel} {db:.1} dB"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "voices moved by {:.1} ms onto the original's clock, dubbing offset {:.0} to {:.0} ms \
             ({}/{} sharp windows, {} with a leak besides), {} attacks at their own offset, \
             removed {removed}",
            self.offset_ms,
            self.lag_min_ms,
            self.lag_max_ms,
            self.sharp,
            self.windows,
            self.with_leaks,
            self.attacks
        )
    }
}

/// One voice object to write: the voice of `channel`, scaled by `gain`,
/// 24-bit, at `path`.
#[derive(Debug, Clone)]
pub struct Output {
    pub channel: String,
    pub path: PathBuf,
    pub gain: f32,
}

/// What the pass reads and writes.
#[derive(Debug, Clone)]
pub struct Plan<'a> {
    /// The original's elements as decoded, LFE aside: the map's reference.
    pub original: &'a [PathBuf],
    /// The dub's channels as decoded, LFE aside.
    pub dub: &'a [PathBuf],
    /// The original's elements without their dialogue, as the new mix
    /// carries them: what is cancelled.
    pub effects: &'a [PathBuf],
    /// Each voice channel of the dub and its `voix_*` file.
    pub voices: &'a [(String, PathBuf)],
    pub outputs: &'a [Output],
    /// Frames every output holds: the original's length.
    pub frames: u64,
    /// Samples the dub's voices are late on the original's clock: the
    /// original's start minus the dub's.
    pub offset: i64,
}

/// Cancel the duplicates and write every voice object, on the original's
/// clock. Returns what was done and how many samples had to be clipped to
/// fit 24 bits.
pub fn cancel(plan: &Plan, progress: Progress) -> Result<(Report, u64)> {
    let n = plan.frames as i64;
    let original = Signal::open(plan.original)?;
    let dub = Signal::open(plan.dub)?;
    let effects = Signal::open(plan.effects)?;
    let voices = plan
        .voices
        .iter()
        .map(|(_, path)| Signal::open(std::slice::from_ref(path)))
        .collect::<Result<Vec<_>>>()?;
    let all_voices = Signal::open(
        &plan
            .voices
            .iter()
            .map(|(_, p)| p.clone())
            .collect::<Vec<_>>(),
    )?;
    let fft = Fft::get();

    // 1. The dubbing map and its leaks, then the vote.
    progress(Event::message("duplicates: map of the dub's offset"));
    let mut map = self::map(&original, &dub, Some((&effects, &all_voices)), 0, n, |r| {
        progress(Event::ratio("duplicates: map", r))
    })?;
    smooth(&mut map);
    // A range shorter than one map window (under 20 s) gives no window, so
    // no offset was measured: nothing is cancelled (at an offset of 0 the
    // music and effects would be taken out where the dub never put them),
    // the dub's voices are only moved onto the original's clock.
    let unmapped = map.is_empty();

    // 2. Attacks: the music and effects', then each voice's that have a twin
    // there, with their own offset.
    let own_offsets = if unmapped {
        vec![Vec::new(); voices.len()]
    } else {
        progress(Event::message("duplicates: attacks"));
        let effect_attacks = attacks(&effects, n, fft)?;
        voices
            .par_iter()
            .map(|voice| {
                let found = attacks(voice, n, fft)?;
                attack_offsets(&found, &effect_attacks, voice, &effects)
            })
            .collect::<Result<Vec<_>>>()?
    };

    // 3. The cancellation, block by block, written out on the original's
    // clock at each voice object's gain.
    progress(Event::message("duplicates: cancelling in the dub's voices"));
    let mut sinks: Vec<Vec<Sink>> = Vec::with_capacity(plan.voices.len());
    for (channel, _) in plan.voices {
        let mut these = Vec::new();
        for output in plan.outputs.iter().filter(|o| &o.channel == channel) {
            these.push(Sink::create(&output.path, output.gain, plan.frames)?);
        }
        sinks.push(these);
    }
    if plan.offset < 0 {
        // The dub starts after the original: its voice is early, delay it.
        for sink in sinks.iter_mut().flatten() {
            sink.zeros((-plan.offset) as u64)?;
        }
    }
    let energy = cancel_blocks(
        &map,
        &effects,
        &voices,
        &own_offsets,
        &mut sinks,
        n,
        plan.offset,
        fft,
        &|done| progress(Event::ratio("duplicates: cancelling", done)),
    )?;
    let mut clipped = 0;
    for sink in sinks.into_iter().flatten() {
        clipped += sink.finish()?;
    }

    let lags: Vec<f64> = map.iter().map(|w| ms(w.smoothed as i64)).collect();
    let report = Report {
        offset_ms: ms(plan.offset),
        windows: map.len(),
        sharp: map.iter().filter(|w| w.sharpness >= SHARPNESS_MIN).count(),
        with_leaks: map.iter().filter(|w| offsets(w).len() > 1).count(),
        lag_min_ms: lags.iter().copied().reduce(f64::min).unwrap_or(0.0),
        lag_max_ms: lags.iter().copied().reduce(f64::max).unwrap_or(0.0),
        attacks: own_offsets.iter().map(Vec::len).sum(),
        removed_db: if unmapped {
            Vec::new()
        } else {
            plan.voices
                .iter()
                .zip(&energy)
                .map(|((channel, _), &(voice, removed))| {
                    (channel.clone(), removed_db(voice, removed))
                })
                .collect()
        },
    };
    Ok((report, clipped))
}

/// The cancellation itself, block by block: every voice rid of the music
/// and effects at the offsets of its block and of its own attacks, written
/// to its sinks `offset` samples earlier. Returns, for each voice, its
/// energy and the energy taken out of it.
#[allow(clippy::too_many_arguments)]
fn cancel_blocks(
    map: &[Window],
    effects: &Signal,
    voices: &[Signal],
    own_offsets: &[Vec<(i64, i32)>],
    sinks: &mut [Vec<Sink>],
    n: i64,
    offset: i64,
    fft: &Fft,
    done: &(dyn Fn(f64) + Sync),
) -> Result<Vec<(f64, f64)>> {
    let channels = voices.len();
    let mut energy = vec![(0f64, 0f64); channels];
    let blocks = (n.max(0) as usize).div_ceil(BLOCK);
    for g0 in (0..blocks).step_by(BLOCKS_AT_ONCE) {
        let count = BLOCKS_AT_ONCE.min(blocks - g0);
        let group = (0..count)
            .into_par_iter()
            .map(|j| {
                let s0 = ((g0 + j) * BLOCK) as i64;
                Block::new(map, effects, s0, BLOCK.min((n - s0) as usize), fft)
            })
            .collect::<Result<Vec<_>>>()?;
        let results = (0..count * channels)
            .into_par_iter()
            .map(|t| {
                let c = t % channels;
                cancel_channel(&group[t / channels], &voices[c], &own_offsets[c], fft)
            })
            .collect::<Result<Vec<_>>>()?;
        for (t, (cancelled, voice, removed)) in results.into_iter().enumerate() {
            let block = &group[t / channels];
            let c = t % channels;
            energy[c].0 += voice;
            energy[c].1 += removed;
            // Offset: sample i of the dub goes to i - offset on the
            // original's clock.
            let skip = (-(block.s0 - offset)).clamp(0, block.nb as i64) as usize;
            for sink in &mut sinks[c] {
                sink.write(&cancelled[MARGIN + skip..MARGIN + block.nb])?;
            }
        }
        done(((g0 + count) * BLOCK) as f64 / n.max(1) as f64);
    }
    Ok(energy)
}

/// How much was taken out of a voice, in dB below it, to a tenth.
fn removed_db(voice: f64, removed: f64) -> f64 {
    if voice > 0.0 {
        (10.0 * (removed.max(1e-30) / voice).log10() * 10.0).round() / 10.0
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// An extract on its own: a few seconds of one element and one channel, as a
// test bench holds them. An extract is shorter than a window of the map, so
// the map is measured on the film around it ([`steps_around`]) and handed
// back ([`cancel_extract`]); the leaks and the attacks are measured on the
// extract itself.

/// One step of the map of the dub around an extract: the centre of a map
/// window in samples from the extract's start (negative: before it), and
/// the smoothed offset of the dub there (> 0: the dub arrives later).
pub type Step = (i64, i32);

/// How far either side of an extract the film is read for its map: the
/// vote of a doubtful window reaches thirty seconds, and a window is twenty
/// long.
const CONTEXT: i64 = 60 * RATE;

/// The map of the dub around the range `[start, end)` of a film: its mixes
/// read [`CONTEXT`] either side, the same vote as for a whole film, and the
/// steps of the windows within one window of the range. `original` and
/// `dub` are the elements and the channels as decoded, LFE aside.
pub fn steps_around(
    original: &[PathBuf],
    dub: &[PathBuf],
    start: i64,
    end: i64,
) -> Result<Vec<Step>> {
    let original = Signal::open(original)?;
    let dub = Signal::open(dub)?;
    let total = original
        .files
        .iter()
        .chain(&dub.files)
        .map(|(_, format)| format.frames as i64)
        .max()
        .unwrap_or(0);
    let from = (start - CONTEXT).max(0);
    let to = (end + CONTEXT).min(total);
    let mut map = self::map(&original, &dub, None, from, to - from, |_| {})?;
    smooth(&mut map);
    Ok(map
        .iter()
        .filter(|w| {
            let at = from + w.centre;
            at >= start - MAP_WINDOW && at <= end + MAP_WINDOW
        })
        .map(|w| (from + w.centre - start, w.smoothed))
        .collect())
}

/// Cancel the duplicates in the voice of an extract, given the steps of the
/// map around it. `effects` is the original's music and effects over the
/// extract; the voice stays on the dub's clock, and `output` is as long as
/// it is. Returns the energy taken out, in dB below the voice.
pub fn cancel_extract(
    steps: &[Step],
    effects: &[PathBuf],
    voice: &Path,
    output: &Path,
) -> Result<f64> {
    if steps.is_empty() {
        return Err(Error::msg(
            "duplicates: no step of the dub's map around the extract (too short a film?)",
        ));
    }
    let effects = Signal::open(effects)?;
    let voice = Signal::open(&[voice.to_path_buf()])?;
    let frames = voice.files[0].1.frames;
    let n = frames as i64;
    let fft = Fft::get();
    let mut map: Vec<Window> = steps
        .iter()
        .map(|&(centre, lag)| Window {
            centre,
            lag,
            smoothed: lag,
            sharpness: 99.0,
            leaks: Vec::new(),
        })
        .collect();
    // The leaks: the part of each window that falls in the extract, when
    // it is a second at least.
    for window in &mut map {
        let from = (window.centre - MAP_WINDOW / 2).max(0);
        let to = (window.centre + MAP_WINDOW / 2).min(n);
        if to - from < RATE {
            continue;
        }
        let span = (to - from) as usize;
        window.leaks = leak_peaks(&correlate(
            &voice.read(from, span + 1)?,
            &effects.read(from - SEARCH, span + 2 * SEARCH as usize + 1)?,
        ));
    }
    let own = attack_offsets(
        &attacks(&voice, n, fft)?,
        &attacks(&effects, n, fft)?,
        &voice,
        &effects,
    )?;
    let mut sinks = vec![vec![Sink::create(output, 1.0, frames)?]];
    let energy = cancel_blocks(
        &map,
        &effects,
        std::slice::from_ref(&voice),
        &[own],
        &mut sinks,
        n,
        0,
        fft,
        &|_| {},
    )?;
    for sink in sinks.into_iter().flatten() {
        sink.finish()?;
    }
    Ok(removed_db(energy[0].0, energy[0].1))
}

/// Write the sum of mono files over `frames` frames, each read from its own
/// start, as a float WAV — a sum of elements goes past ±1. The music and
/// effects of the other elements of the original around an extract.
pub fn write_sum(sources: &[(PathBuf, i64)], frames: u64, output: &Path) -> Result<()> {
    let signals = sources
        .iter()
        .map(|(path, start)| Ok((Signal::open(std::slice::from_ref(path))?, *start)))
        .collect::<Result<Vec<_>>>()?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    let part = output.with_extension("wav.part");
    let file = File::create(&part).map_err(|e| Error::io(&part, e))?;
    let mut writer = BufWriter::new(file);
    wav::write_header(
        &mut writer,
        Format {
            channels: 1,
            sample_rate: RATE as u32,
            bits: 32,
            float: true,
            frames,
            data_offset: 0,
        },
    )
    .map_err(|e| Error::io(&part, e))?;
    let mut at = 0u64;
    while at < frames {
        let count = BLOCK.min((frames - at) as usize);
        let mut sum = vec![0f32; count];
        for (signal, start) in &signals {
            for (sum, sample) in sum.iter_mut().zip(signal.read(start + at as i64, count)?) {
                *sum += sample;
            }
        }
        for sample in &sum {
            writer
                .write_all(&sample.to_le_bytes())
                .map_err(|e| Error::io(&part, e))?;
        }
        at += count as u64;
    }
    writer.flush().map_err(|e| Error::io(&part, e))?;
    drop(writer);
    fs::rename(&part, output).map_err(|e| Error::io(output, e))
}

fn ms(samples: i64) -> f64 {
    samples as f64 * 1000.0 / RATE as f64
}

// ---------------------------------------------------------------------------
// The map: one window is its centre, the dubbing's offset (> 0 = the dub
// arrives later), how sharp its peak is, and the leak peaks of the dub's
// voices (offset, value over the main peak, sharpness).

#[derive(Debug, Clone, Default)]
struct Window {
    centre: i64,
    lag: i32,
    smoothed: i32,
    sharpness: f64,
    leaks: Vec<(i32, f64, f64)>,
}

/// The map over `[origin, origin + n)` of the files, centres counted from
/// `origin`. Without `leaks` — the music and effects and the voices — only
/// the mixes are read: the context of an extract has no separated stems,
/// and its leaks are measured on the extract itself.
fn map(
    original: &Signal,
    dub: &Signal,
    leaks: Option<(&Signal, &Signal)>,
    origin: i64,
    n: i64,
    report: impl Fn(f64),
) -> Result<Vec<Window>> {
    let mut centres = Vec::new();
    let mut c = MAP_WINDOW / 2;
    while c < n - MAP_WINDOW / 2 {
        centres.push(c);
        c += MAP_STEP;
    }
    if centres.is_empty() {
        return Ok(Vec::new());
    }
    // Two real FFTs of 2^19 points per correlation, about 13 MB a thread.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(rayon::current_num_threads().min(16))
        .build()
        .map_err(|e| Error::msg(format!("duplicates: {e}")))?;
    type Slice = (Vec<f64>, Vec<f64>);
    let mut slices: Vec<Option<Slice>> = (0..centres.len() + SLICES_PER_WINDOW - 1)
        .map(|_| None)
        .collect();
    let mut windows = Vec::with_capacity(centres.len());
    let step = MAP_STEP as usize;
    let span = step + 2 * SEARCH as usize + 1;
    for l0 in (0..centres.len()).step_by(MAP_BATCH) {
        let l1 = (l0 + MAP_BATCH).min(centres.len());
        // Slice m = [m·step, (m + 1)·step[; the batch's first three are
        // already there, except in the first batch.
        let wanted: Vec<usize> = (l0..l1 + SLICES_PER_WINDOW - 1)
            .filter(|&m| slices[m].is_none())
            .collect();
        let made = pool.install(|| {
            wanted
                .par_iter()
                .map(|&m| -> Result<(usize, Slice)> {
                    let t0 = origin + (m * step) as i64;
                    let mix =
                        correlate(&dub.read(t0, step + 1)?, &original.read(t0 - SEARCH, span)?);
                    let leak = match leaks {
                        Some((effects, voices)) => correlate(
                            &voices.read(t0, step + 1)?,
                            &effects.read(t0 - SEARCH, span)?,
                        ),
                        None => vec![0.0; mix.len()],
                    };
                    Ok((m, (mix, leak)))
                })
                .collect::<Result<Vec<_>>>()
        })?;
        for (m, slice) in made {
            slices[m] = Some(slice);
        }
        let batch: Vec<Window> = (l0..l1)
            .into_par_iter()
            .map(|i| {
                let first = slices[i].as_ref().expect("computed above");
                let (mut mix, mut leak) = (first.0.clone(), first.1.clone());
                for j in 1..SLICES_PER_WINDOW {
                    let (a, b) = slices[i + j].as_ref().expect("computed above");
                    for k in 0..mix.len() {
                        mix[k] += a[k];
                        leak[k] += b[k];
                    }
                }
                window(centres[i], &mix, &leak)
            })
            .collect();
        windows.extend(batch);
        // No window still to come starts before l1.
        for slice in &mut slices[l0..l1] {
            *slice = None;
        }
        report(l1 as f64 / centres.len() as f64);
    }
    Ok(windows)
}

/// One window of the map from its two correlations: the dub's mix against
/// the original's (the offset and its sharpness), the dub's voices against
/// the original's music and effects (the leaks).
///
/// A window that is all zeros (black credits, silent music and effects,
/// past the end of a file) has no positive peak ([`peaks`] gives
/// `[(0, 0.0)]`): its sharpness is 0, so it does not vote for its
/// neighbours ([`smooth`]) as a perfect offset of 0 would. A lone positive
/// peak, on the contrary, is as sharp as can be (99). Likewise, voices or
/// music and effects silent over the window have no leak at all.
fn window(centre: i64, mix: &[f64], leak: &[f64]) -> Window {
    let peaks = self::peaks(mix, 2);
    let sharpness = if peaks[0].1 <= 0.0 {
        0.0
    } else if peaks.len() > 1 {
        (peaks[0].1 / peaks[1].1).min(99.0)
    } else {
        99.0
    };
    let mut window = Window {
        centre,
        lag: peaks[0].0,
        sharpness,
        ..Window::default()
    };
    window.leaks = leak_peaks(leak);
    window
}

/// The leak peaks of a correlation of the dub's voices against the
/// original's music and effects: up to three, each with its value over the
/// first and how far it stands above the next. None when either side is
/// silent.
fn leak_peaks(leak: &[f64]) -> Vec<(i32, f64, f64)> {
    let peaks = self::peaks(leak, 4);
    let mut leaks = Vec::new();
    if peaks[0].1 > 0.0 {
        for j in 0..peaks.len().min(3) {
            let next = peaks.get(j + 1).map_or(0.0, |p| p.1);
            leaks.push((
                peaks[j].0,
                peaks[j].1 / peaks[0].1,
                if next > 0.0 { peaks[j].1 / next } else { 99.0 },
            ));
        }
    }
    leaks
}

/// The vote: a sharp window keeps its offset; another takes the one most
/// sharp windows within ±[`VOTE_REACH`] share (within [`TOLERANCE`]; on a
/// tie, the nearest group), or the nearest sharp window's.
fn smooth(map: &mut [Window]) {
    let sharp: Vec<(i64, i32)> = map
        .iter()
        .filter(|w| w.sharpness >= SHARPNESS_MIN)
        .map(|w| (w.centre, w.lag))
        .collect();
    for window in map.iter_mut() {
        if window.sharpness >= SHARPNESS_MIN || sharp.is_empty() {
            window.smoothed = window.lag;
            continue;
        }
        let here = window.centre;
        let distance = |centre: i64| (centre - here).abs() as f64 / RATE as f64;
        let nearest = |members: &[(i64, i32)]| {
            members
                .iter()
                .copied()
                .reduce(|a, b| if distance(b.0) < distance(a.0) { b } else { a })
                .expect("never empty")
        };
        let near: Vec<(i64, i32)> = sharp
            .iter()
            .copied()
            .filter(|&(c, _)| distance(c) <= VOTE_REACH)
            .collect();
        if near.is_empty() {
            window.smoothed = nearest(&sharp).1;
            continue;
        }
        let mut votes: Vec<(i32, Vec<(i64, i32)>)> = Vec::new();
        for &(c, lag) in &near {
            match votes.iter_mut().find(|v| (v.0 - lag).abs() <= TOLERANCE) {
                Some(vote) => vote.1.push((c, lag)),
                None => votes.push((lag, vec![(c, lag)])),
            }
        }
        let mut winner = &votes[0].1;
        for (_, members) in &votes[1..] {
            let better = members.len() > winner.len()
                || (members.len() == winner.len()
                    && distance(nearest(members).0) < distance(nearest(winner).0));
            if better {
                winner = members;
            }
        }
        window.smoothed = nearest(winner).1;
    }
}

/// The offsets to cancel around a window: the dubbing's plateau first, then
/// the sharp leaks distinct from it.
fn offsets(window: &Window) -> Vec<i32> {
    let mut list = vec![window.smoothed];
    for &(lag, share, sharpness) in &window.leaks {
        if share >= LEAK_SHARE_MIN
            && sharpness >= SHARPNESS_MIN
            && list.iter().all(|x| (x - lag).abs() > TOLERANCE)
        {
            list.push(lag);
        }
    }
    list
}

/// The window of a map (not empty) nearest a sample.
fn nearest_window(map: &[Window], centre: i64) -> &Window {
    // Regular windows, one every step from the first: the index is
    // computed. (A film's first is half a window in; an extract's steps
    // start wherever its context does.)
    let i = ((centre - map[0].centre) as f64 / MAP_STEP as f64)
        .round_ties_even()
        .clamp(0.0, (map.len() - 1) as f64) as usize;
    &map[i]
}

// ---------------------------------------------------------------------------
// Cross-correlation of first differences (the low end and DC no longer
// weigh): r[k] = Σ a[t] b[t + k] for k from 0 to len(b) - len(a); the offset
// is SEARCH - k (> 0: a arrives after b).

fn correlate(a0: &[f32], b0: &[f32]) -> Vec<f64> {
    let la = a0.len() - 1;
    let lb = b0.len() - 1;
    let size = (la + lb).next_power_of_two();
    let (forward, inverse) = correlation_plans(size);
    let mut a = vec![0f64; size];
    let mut b = vec![0f64; size];
    for t in 0..la {
        a[t] = (a0[t + 1] - a0[t]) as f64;
    }
    for t in 0..lb {
        b[t] = (b0[t + 1] - b0[t]) as f64;
    }
    let mut fa = forward.make_output_vec();
    let mut fb = forward.make_output_vec();
    forward.process(&mut a, &mut fa).expect("sized by its plan");
    forward.process(&mut b, &mut fb).expect("sized by its plan");
    for (b, a) in fb.iter_mut().zip(&fa) {
        *b *= a.conj();
    }
    let last = fb.len() - 1;
    fb[0].im = 0.0;
    fb[last].im = 0.0;
    let mut r = inverse.make_output_vec();
    inverse.process(&mut fb, &mut r).expect("sized by its plan");
    // The inverse is unscaled: bring it back to the plain sum, which the
    // normalised correlation of an attack needs.
    let scale = 1.0 / size as f64;
    r.truncate(lb - la + 1);
    for value in &mut r {
        *value *= scale;
    }
    r
}

/// The `count` highest positive peaks of r, at least [`PEAK_GAP`] apart, as
/// offsets (SEARCH - k); no positive peak (r all zeros) gives `[(0, 0.0)]`.
fn peaks(r: &[f64], count: usize) -> Vec<(i32, f64)> {
    let mut c = r.to_vec();
    let mut found = Vec::new();
    for _ in 0..count {
        let mut k = 0;
        for i in 1..c.len() {
            if c[i] > c[k] {
                k = i;
            }
        }
        if c[k] <= 0.0 || c[k] == f64::NEG_INFINITY {
            break;
        }
        found.push(((SEARCH - k as i64) as i32, c[k]));
        let hi = (k + PEAK_GAP).min(c.len() - 1);
        for value in &mut c[k.saturating_sub(PEAK_GAP)..=hi] {
            *value = f64::NEG_INFINITY;
        }
    }
    if found.is_empty() {
        found.push((0, 0.0));
    }
    found
}

// ---------------------------------------------------------------------------
// Attacks of a signal: local maxima (±3 frames) of the 24 bands' mean rise
// (dB over 2 frames), at least RISE_MIN; bands below -80 dB do not count.
// Streamed in 30 s blocks.

struct Attack {
    frame: i64,
    rise: [f32; BANDS],
}

fn attacks(signal: &Signal, n: i64, fft: &Fft) -> Result<Vec<Attack>> {
    if n < NFFT as i64 {
        return Ok(Vec::new());
    }
    let frames = ((n - NFFT as i64) / HOP as i64 + 1) as usize;
    let floor = 10.0 * ((NFFT * NFFT) as f64 / 4.0).log10() - 80.0;
    let blocks = frames.div_ceil(ATTACK_FRAMES);
    let found = (0..blocks)
        .into_par_iter()
        .map(|ib| -> Result<Vec<Attack>> {
            let t0 = ib * ATTACK_FRAMES;
            // Frames computed [from, to[; attacks kept in [t0, t0 + ATTACK_FRAMES[.
            let from = t0.saturating_sub(ATTACK_EDGE);
            let to = frames.min(t0 + ATTACK_FRAMES + ATTACK_EDGE);
            let y = signal.read((from * HOP) as i64, (to - from - 1) * HOP + NFFT)?;
            let mut buf = vec![0f32; NFFT];
            let mut spectrum = fft.forward.make_output_vec();
            let mut scratch = fft.forward.make_scratch_vec();
            let mut levels = vec![[0f32; BANDS]; to - from];
            for (k, level) in levels.iter_mut().enumerate() {
                for i in 0..NFFT {
                    buf[i] = y[k * HOP + i] * fft.hann[i];
                }
                fft.forward
                    .process_with_scratch(&mut buf, &mut spectrum, &mut scratch)
                    .expect("sized by its plan");
                for (b, &(lo, hi)) in fft.bands.iter().enumerate() {
                    let e: f64 = spectrum[lo..hi]
                        .iter()
                        .map(|z| z.re as f64 * z.re as f64 + z.im as f64 * z.im as f64)
                        .sum();
                    level[b] = (10.0 * (e + 1e-10).log10()) as f32;
                }
            }
            // Mean rise of local frame k; fills `rise` when asked.
            let mean = |k: usize, rise: Option<&mut [f32; BANDS]>| -> f64 {
                if k < 2 {
                    return 0.0;
                }
                let mut each = [0f32; BANDS];
                for b in 0..BANDS {
                    each[b] = if (levels[k][b] as f64) < floor {
                        0.0
                    } else {
                        (levels[k][b] - levels[k - 2][b]).max(0.0)
                    };
                }
                if let Some(rise) = rise {
                    *rise = each;
                }
                each.iter().map(|&m| m as f64).sum::<f64>() / BANDS as f64
            };
            let means: Vec<f64> = (0..to - from).map(|k| mean(k, None)).collect();
            let mut found = Vec::new();
            for t in t0.max(from)..(t0 + ATTACK_FRAMES).min(to) {
                let k = t - from;
                let m = means[k];
                if m < RISE_MIN {
                    continue;
                }
                let peak = (1..=3).all(|d| {
                    let after = if t + d < frames && k + d < means.len() {
                        means[k + d]
                    } else {
                        0.0
                    };
                    let before = if k >= d { means[k - d] } else { 0.0 };
                    m >= after && m > before
                });
                if !peak {
                    continue;
                }
                let mut rise = [0f32; BANDS];
                mean(k, Some(&mut rise));
                found.push(Attack {
                    frame: t as i64,
                    rise,
                });
            }
            Ok(found)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(found.into_iter().flatten().collect())
}

/// Each attack of a voice that has a twin in the music and effects (within
/// ±PAIRING frames, rises alike) gets its own offset: the normalised
/// correlation of the waveforms (first differences, from 10 ms before to
/// 60 ms after the attack) over ±150 ms, kept if it reaches
/// ATTACK_CORRELATION_MIN. (Centre in samples; offset > 0 = the dub arrives
/// later.)
fn attack_offsets(
    voice_attacks: &[Attack],
    effect_attacks: &[Attack],
    voice: &Signal,
    effects: &Signal,
) -> Result<Vec<(i64, i32)>> {
    const BEFORE: i64 = RATE / 100;
    const AFTER: i64 = RATE * 6 / 100;
    let mut found = Vec::new();
    for attack in voice_attacks {
        let lo = effect_attacks.partition_point(|a| a.frame < attack.frame - PAIRING);
        let hi = effect_attacks.partition_point(|a| a.frame < attack.frame + PAIRING + 1);
        if lo >= hi {
            continue;
        }
        let best = effect_attacks[lo..hi]
            .iter()
            .map(|twin| cosine(&attack.rise, &twin.rise))
            .fold(-1.0, f64::max);
        if best < SIMILARITY_MIN {
            continue;
        }
        let c = attack.frame * HOP as i64 + NFFT as i64 / 2;
        let v = voice.read(c - BEFORE, (BEFORE + AFTER + 1) as usize)?;
        let me = effects.read(
            c - BEFORE - SEARCH,
            (BEFORE + AFTER + 2 * SEARCH + 1) as usize,
        )?;
        let (lag, correlation) = normalised_correlation(&v, &me);
        if correlation >= ATTACK_CORRELATION_MIN {
            found.push((c, lag));
        }
    }
    Ok(found)
}

/// The offset (SEARCH - k) that maximises the normalised correlation of v's
/// first differences against each slice of me's, and that correlation.
fn normalised_correlation(v: &[f32], me: &[f32]) -> (i32, f64) {
    let r = correlate(v, me);
    let la = v.len() - 1;
    let d: Vec<f64> = me.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
    let na = v
        .windows(2)
        .map(|w| {
            let x = (w[1] - w[0]) as f64;
            x * x
        })
        .sum::<f64>()
        .sqrt();
    // The sliding energy of me's slice.
    let mut eb: f64 = d[..la].iter().map(|x| x * x).sum();
    let (mut best_k, mut best) = (0, f64::NEG_INFINITY);
    for (k, &value) in r.iter().enumerate() {
        if k > 0 {
            eb += d[k + la - 1] * d[k + la - 1] - d[k - 1] * d[k - 1];
        }
        let c = value / (na * eb.max(0.0).sqrt() + 1e-12);
        if c > best {
            best = c;
            best_k = k;
        }
    }
    ((SEARCH - best_k as i64) as i32, best)
}

fn cosine(a: &[f32; BANDS], b: &[f32; BANDS]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        ab += (x * y) as f64;
        aa += (x * x) as f64;
        bb += (y * y) as f64;
    }
    ab / (aa.sqrt() * bb.sqrt() + 1e-9)
}

// ---------------------------------------------------------------------------
// The cancellation, block by block.

/// One block: its bounds, its offsets (the map's), and the music and effects
/// read ONCE, extended by ±SEARCH: shifted by d (|d| ≤ SEARCH, as every
/// offset of the map and of the attacks is), they are the slice starting at
/// SEARCH - d. X = their STFT at each of the map's offsets, shared by the
/// channels. An empty map gives no offset: nothing to cancel, and neither
/// the music and effects nor X are read (see [`cancel_channel`]).
struct Block {
    s0: i64,
    a0: i64,
    nb: usize,
    ln: usize,
    frames: usize,
    lags: Vec<i32>,
    me: Vec<f32>,
    x: Vec<Vec<Complex<f32>>>,
}

impl Block {
    fn new(map: &[Window], effects: &Signal, s0: i64, nb: usize, fft: &Fft) -> Result<Self> {
        let a0 = s0 - MARGIN as i64;
        let ln = nb + 2 * MARGIN;
        let frames = (ln - NFFT) / HOP + 1;
        if map.is_empty() {
            return Ok(Self {
                s0,
                a0,
                nb,
                ln,
                frames,
                lags: Vec::new(),
                me: Vec::new(),
                x: Vec::new(),
            });
        }
        let lags = offsets(nearest_window(map, s0 + nb as i64 / 2));
        let me = effects.read(a0 - SEARCH, ln + 2 * SEARCH as usize)?;
        let x = lags
            .iter()
            .map(|&d| fft.stft(&me, (SEARCH - d as i64) as usize, 0, frames))
            .collect();
        Ok(Self {
            s0,
            a0,
            nb,
            ln,
            frames,
            lags,
            me,
            x,
        })
    }
}

/// One voice channel over one block: rid of the music and effects at the
/// block's offsets, then at its attacks' own. Returns the block cancelled
/// (margins included) and the energies of the voice and of what was taken
/// out, over the block itself. A block without offsets (empty map) gives the
/// voice back as it is, without the STFT round trip.
fn cancel_channel(
    block: &Block,
    voice: &Signal,
    attacks: &[(i64, i32)],
    fft: &Fft,
) -> Result<(Vec<f32>, f64, f64)> {
    let frames = block.frames;
    let y = voice.read(block.a0, block.ln)?;
    if block.lags.is_empty() {
        let energy = y[MARGIN..MARGIN + block.nb]
            .iter()
            .map(|&v| (v * v) as f64)
            .sum();
        return Ok((y, energy, 0.0));
    }
    let mut spectrum = fft.stft(&y, 0, 0, frames);
    for x in &block.x {
        remove(&mut spectrum, x, frames, 0, frames);
    }
    // The block's attacks whose offset is not already one of the block's:
    // cancelled locally at their own. Only their frames [k0, k1] change, and
    // the 3-frame neighbourhood only reads [k0 - 1, k1 + 1] for them: only
    // that range is computed, not the whole block — the same result, about
    // 15 times less work an attack.
    let lo = attacks.partition_point(|a| a.0 < block.s0);
    let hi = attacks.partition_point(|a| a.0 < block.s0 + block.nb as i64);
    let half = NFFT as i64 / 2;
    for &(c, d) in &attacks[lo..hi] {
        if block.lags.iter().any(|&x| (x - d).abs() <= TOLERANCE) {
            continue;
        }
        let first = (c - BEFORE_ATTACK - block.a0 - half) as f64 / HOP as f64;
        let last = (c + AFTER_ATTACK - block.a0 - half) as f64 / HOP as f64;
        let k0 = first.ceil().max(0.0) as i64;
        let k1 = (last.floor() as i64).min(frames as i64 - 1);
        if k1 < k0 {
            continue;
        }
        let (k0, k1) = (k0 as usize, k1 as usize);
        // Frames computed [f0, f1[; at the block's edges the neighbourhood
        // is replicated as over the whole block.
        let f0 = k0.saturating_sub(1);
        let f1 = frames.min(k1 + 2);
        // |d| ≤ SEARCH (attack_offsets reads the music and effects ±SEARCH
        // around the voice): the shifted ones are read in the block's.
        let x = fft.stft(&block.me, (SEARCH - d as i64) as usize, f0, f1);
        let mut local = spectrum[f0 * BINS..f1 * BINS].to_vec();
        remove(&mut local, &x, f1 - f0, k0 - f0, k1 + 1 - f0);
        // `remove` only touches [k0, k1].
        spectrum[k0 * BINS..(k1 + 1) * BINS]
            .copy_from_slice(&local[(k0 - f0) * BINS..(k1 + 1 - f0) * BINS]);
    }
    let cancelled = fft.istft(&mut spectrum, frames, block.ln);
    let (mut voice_energy, mut removed) = (0f64, 0f64);
    for i in MARGIN..MARGIN + block.nb {
        voice_energy += (y[i] * y[i]) as f64;
        let r = (y[i] - cancelled[i]) as f64;
        removed += r * r;
    }
    Ok((cancelled, voice_energy, removed))
}

/// Coherent subtraction in frames [k0, k1[: Y -= w(g) · α · X, α and g
/// estimated over 3 frames × 5 bins, never more than |Y|.
fn remove(y: &mut [Complex<f32>], x: &[Complex<f32>], frames: usize, k0: usize, k1: usize) {
    let size = frames * BINS;
    let mut sxy_re = vec![0f32; size];
    let mut sxy_im = vec![0f32; size];
    let mut sxx = vec![0f32; size];
    let mut syy = vec![0f32; size];
    for i in 0..size {
        let (yr, yi, xr, xi) = (y[i].re, y[i].im, x[i].re, x[i].im);
        // Y · conj(X)
        sxy_re[i] = yr * xr + yi * xi;
        sxy_im[i] = yi * xr - yr * xi;
        sxx[i] = xr * xr + xi * xi;
        syy[i] = yr * yr + yi * yi;
    }
    for a in [&mut sxy_re, &mut sxy_im, &mut sxx, &mut syy] {
        neighbourhood(a, frames);
    }
    for i in k0 * BINS..k1 * BINS {
        let (px, py) = (sxx[i] as f64, syy[i] as f64);
        let (cr, ci) = (sxy_re[i] as f64, sxy_im[i] as f64);
        let g = (cr * cr + ci * ci) / (px * py + 1e-20);
        let w = ((g - COHERENCE_LOW) / (COHERENCE_HIGH - COHERENCE_LOW)).clamp(0.0, 1.0);
        if w <= 0.0 {
            continue;
        }
        // α = ⟨Y, X⟩ / ⟨X, X⟩
        let (ar, ai) = (cr / (px + 1e-12), ci / (px + 1e-12));
        let (xr, xi) = (x[i].re as f64, x[i].im as f64);
        let (mut pr, mut pi) = (w * (ar * xr - ai * xi), w * (ar * xi + ai * xr));
        let (yr, yi) = (y[i].re as f64, y[i].im as f64);
        let (mc, my) = ((pr * pr + pi * pi).sqrt(), (yr * yr + yi * yi).sqrt());
        if mc > my {
            let f = my / (mc + 1e-12);
            pr *= f;
            pi *= f;
        }
        y[i] = Complex::new((yr - pr) as f32, (yi - pi) as f32);
    }
}

/// Sum over 5 bins then 3 frames, the edges replicated.
fn neighbourhood(a: &mut [f32], frames: usize) {
    let mut t = vec![0f32; a.len()];
    for k in 0..frames {
        let o = k * BINS;
        for f in 0..BINS {
            let mut s = 0f32;
            for d in -2i32..=2 {
                s += a[o + (f as i32 + d).clamp(0, BINS as i32 - 1) as usize];
            }
            t[o + f] = s;
        }
    }
    for k in 0..frames {
        let (p, o, s) = (
            k.saturating_sub(1) * BINS,
            k * BINS,
            (k + 1).min(frames - 1) * BINS,
        );
        for f in 0..BINS {
            a[o + f] = t[p + f] + t[o + f] + t[s + f];
        }
    }
}

// ---------------------------------------------------------------------------
// FFTs.

struct Fft {
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
    /// Periodic root Hann: analysis × synthesis = Hann, which sums to 2 at a
    /// hop of N/4.
    root_hann: Vec<f32>,
    /// Symmetric Hann, for the attacks (the prototype's).
    hann: Vec<f32>,
    /// The attack bands: 24 geometric bounds from 150 Hz to 20 kHz, as FFT
    /// bin indices (at least one bin each).
    bands: Vec<(usize, usize)>,
}

impl Fft {
    fn get() -> &'static Fft {
        static FFT: OnceLock<Fft> = OnceLock::new();
        FFT.get_or_init(|| {
            let mut planner = RealFftPlanner::<f32>::new();
            let tau = 2.0 * std::f64::consts::PI;
            let ratio = (20_000.0f64 / 150.0).powf(1.0 / BANDS as f64);
            // The first bin at or above f.
            let bin = |f: f64| (f * NFFT as f64 / RATE as f64 - 1e-9).ceil() as usize;
            Fft {
                forward: planner.plan_fft_forward(NFFT),
                inverse: planner.plan_fft_inverse(NFFT),
                root_hann: (0..NFFT)
                    .map(|i| (0.5 - 0.5 * (tau * i as f64 / NFFT as f64).cos()).sqrt() as f32)
                    .collect(),
                hann: (0..NFFT)
                    .map(|i| (0.5 - 0.5 * (tau * i as f64 / (NFFT - 1) as f64).cos()) as f32)
                    .collect(),
                bands: (0..BANDS)
                    .map(|i| {
                        let lo = bin(150.0 * ratio.powi(i as i32));
                        let hi = bin(150.0 * ratio.powi(i as i32 + 1));
                        (lo, hi.max(lo + 1))
                    })
                    .collect(),
            }
        })
    }

    /// Frames [k0, k1[ of the signal starting at y[start] (frame k =
    /// y[start + k·HOP ...]), frame k0 first.
    fn stft(&self, y: &[f32], start: usize, k0: usize, k1: usize) -> Vec<Complex<f32>> {
        let mut out = Vec::with_capacity((k1 - k0) * BINS);
        let mut buf = vec![0f32; NFFT];
        let mut spectrum = self.forward.make_output_vec();
        let mut scratch = self.forward.make_scratch_vec();
        for k in k0..k1 {
            let o = start + k * HOP;
            for i in 0..NFFT {
                buf[i] = y[o + i] * self.root_hann[i];
            }
            self.forward
                .process_with_scratch(&mut buf, &mut spectrum, &mut scratch)
                .expect("sized by its plan");
            out.extend_from_slice(&spectrum);
        }
        out
    }

    /// Overlap-add back to `length` samples. Both directions are unscaled:
    /// / NFFT; the overlap: / 2.
    fn istft(&self, z: &mut [Complex<f32>], frames: usize, length: usize) -> Vec<f32> {
        let mut y = vec![0f32; length];
        let mut buf = self.inverse.make_output_vec();
        let mut scratch = self.inverse.make_scratch_vec();
        let scale = 1.0 / (NFFT as f32 * 2.0);
        for k in 0..frames {
            let frame = &mut z[k * BINS..(k + 1) * BINS];
            // A real signal's DC and Nyquist bins have no imaginary part.
            frame[0].im = 0.0;
            frame[BINS - 1].im = 0.0;
            self.inverse
                .process_with_scratch(frame, &mut buf, &mut scratch)
                .expect("sized by its plan");
            let o = k * HOP;
            for i in 0..NFFT {
                y[o + i] += buf[i] * self.root_hann[i] * scale;
            }
        }
        y
    }
}

type CorrelationPlans = (Arc<dyn RealToComplex<f64>>, Arc<dyn ComplexToReal<f64>>);

fn correlation_plans(size: usize) -> CorrelationPlans {
    static PLANNER: OnceLock<Mutex<RealFftPlanner<f64>>> = OnceLock::new();
    let mut planner = PLANNER
        .get_or_init(|| Mutex::new(RealFftPlanner::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    (
        planner.plan_fft_forward(size),
        planner.plan_fft_inverse(size),
    )
}

// ---------------------------------------------------------------------------
// Reading: the sum of mono files over [start, start + n[, zeros outside a
// file (start may be negative).

struct Signal {
    files: Vec<(PathBuf, Format)>,
}

impl Signal {
    fn open(paths: &[PathBuf]) -> Result<Self> {
        let files = paths
            .iter()
            .map(|path| Ok((path.clone(), wav::read_header(path)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { files })
    }

    fn read(&self, start: i64, n: usize) -> Result<Vec<f32>> {
        let mut out = vec![0f32; n];
        let mut bytes = Vec::new();
        for (path, format) in &self.files {
            let from = start.max(0);
            let to = (start + n as i64).min(format.frames as i64);
            if to <= from {
                continue;
            }
            let count = (to - from) as usize;
            let frame = format.bytes_per_frame() as usize;
            bytes.resize(count * frame, 0);
            let mut file = File::open(path).map_err(|e| Error::io(path, e))?;
            file.seek(SeekFrom::Start(
                format.data_offset + from as u64 * frame as u64,
            ))
            .and_then(|_| file.read_exact(&mut bytes))
            .map_err(|e| Error::io(path, e))?;
            let at = (from - start) as usize;
            // Channel 0: the elements are mono.
            for (i, out) in out[at..at + count].iter_mut().enumerate() {
                *out += wav::sample(&bytes[i * frame..], format.bits, format.float);
            }
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// A voice object being written: a mono 24-bit WAV of exactly `frames`
// frames, its gain applied, padded with silence when it is closed. Written
// as `.part` and renamed, so a file there is whole.

struct Sink {
    path: PathBuf,
    part: PathBuf,
    writer: BufWriter<File>,
    gain: f32,
    frames: u64,
    written: u64,
    clipped: u64,
    bytes: Vec<u8>,
}

impl Sink {
    fn create(path: &Path, gain: f32, frames: u64) -> Result<Self> {
        let part = path.with_extension("wav.part");
        let file = File::create(&part).map_err(|e| Error::io(&part, e))?;
        let mut writer = BufWriter::new(file);
        wav::write_header(
            &mut writer,
            Format {
                channels: 1,
                sample_rate: RATE as u32,
                bits: 24,
                float: false,
                frames,
                data_offset: 0,
            },
        )
        .map_err(|e| Error::io(&part, e))?;
        Ok(Self {
            path: path.to_path_buf(),
            part,
            writer,
            gain,
            frames,
            written: 0,
            clipped: 0,
            bytes: Vec::new(),
        })
    }

    fn zeros(&mut self, count: u64) -> Result<()> {
        let count = count.min(self.frames - self.written) as usize;
        self.write(&vec![0f32; count])
    }

    fn write(&mut self, samples: &[f32]) -> Result<()> {
        let take = samples.len().min((self.frames - self.written) as usize);
        self.bytes.clear();
        for &sample in &samples[..take] {
            // As wav::copy_mono writes a levelled voice at full depth.
            let scaled = (sample * self.gain * 8_388_608.0).round();
            if !(-8_388_608.0..=8_388_607.0).contains(&scaled) {
                self.clipped += 1;
            }
            let value = scaled.clamp(-8_388_608.0, 8_388_607.0) as i32;
            self.bytes.extend_from_slice(&value.to_le_bytes()[..3]);
        }
        self.writer
            .write_all(&self.bytes)
            .map_err(|e| Error::io(&self.part, e))?;
        self.written += take as u64;
        Ok(())
    }

    fn finish(mut self) -> Result<u64> {
        let zeros = [0u8; 3 * 4096];
        while self.written < self.frames {
            let take = ((self.frames - self.written) as usize).min(4096);
            self.writer
                .write_all(&zeros[..take * 3])
                .map_err(|e| Error::io(&self.part, e))?;
            self.written += take as u64;
        }
        self.writer.flush().map_err(|e| Error::io(&self.part, e))?;
        drop(self.writer);
        fs::rename(&self.part, &self.path).map_err(|e| Error::io(&self.path, e))?;
        Ok(self.clipped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic noise, so a test needs no crate.
    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn the_correlation_finds_the_offset_and_its_sign() {
        // b is a signal, a the same signal 40 samples later.
        let base = noise(20_000, 1);
        let lag = 40usize;
        let a: Vec<f32> = (0..4_001).map(|t| base[8_000 + t - lag]).collect();
        let b: Vec<f32> = (0..4_001 + 2 * SEARCH as usize)
            .map(|t| base[8_000 - SEARCH as usize + t])
            .collect();
        let peaks = peaks(&correlate(&a, &b), 2);
        assert_eq!(peaks[0].0, lag as i32, "> 0: a arrives after b");
        let (found, correlation) = normalised_correlation(&a, &b);
        assert_eq!(found, lag as i32);
        assert!((correlation - 1.0).abs() < 1e-6, "{correlation}");
    }

    #[test]
    fn the_stft_gives_the_signal_back() {
        let fft = Fft::get();
        let y = noise(40 * HOP, 2);
        let frames = (y.len() - NFFT) / HOP + 1;
        let mut z = fft.stft(&y, 0, 0, frames);
        let back = fft.istft(&mut z, frames, y.len());
        // Whole once the window overlaps itself fully.
        for i in NFFT..y.len() - NFFT {
            assert!((back[i] - y[i]).abs() < 1e-5, "{i}: {} {}", back[i], y[i]);
        }
    }

    #[test]
    fn a_copy_of_the_reference_is_removed_and_the_rest_kept() {
        let fft = Fft::get();
        let n = 60 * HOP;
        let reference = noise(n, 3);
        let other = noise(n, 4);
        let frames = (n - NFFT) / HOP + 1;
        let x = fft.stft(&reference, 0, 0, frames);
        // Only the reference, at half level: taken out almost whole.
        let mut y = fft.stft(
            &reference.iter().map(|v| v * 0.5).collect::<Vec<_>>(),
            0,
            0,
            frames,
        );
        remove(&mut y, &x, frames, 0, frames);
        let left = fft.istft(&mut y, frames, n);
        let energy = |s: &[f32]| {
            s[NFFT..n - NFFT]
                .iter()
                .map(|v| (v * v) as f64)
                .sum::<f64>()
        };
        assert!(
            energy(&left) < 1e-6 * energy(&reference),
            "{}",
            energy(&left)
        );
        // Something unrelated: incoherent, left alone.
        let mut y = fft.stft(&other, 0, 0, frames);
        remove(&mut y, &x, frames, 0, frames);
        let kept = fft.istft(&mut y, frames, n);
        assert!(energy(&kept) > 0.9 * energy(&other));
    }

    #[test]
    fn a_doubtful_window_takes_the_plateau_its_neighbours_share() {
        let window = |centre_s: i64, lag: i32, sharpness: f64| Window {
            centre: centre_s * RATE,
            lag,
            sharpness,
            ..Window::default()
        };
        let mut map = vec![
            window(10, 3_300, 3.0),
            window(15, 3_302, 2.0),
            window(20, 999, 1.1),
            window(25, 2_000, 4.0),
            window(30, 3_305, 2.5),
        ];
        smooth(&mut map);
        let smoothed: Vec<i32> = map.iter().map(|w| w.smoothed).collect();
        // Three sharp windows share ~3300; a median of the raw offsets
        // would have made one up.
        assert_eq!(smoothed, [3_300, 3_302, 3_302, 2_000, 3_305]);
    }

    #[test]
    fn a_leak_counts_when_it_stands_out_and_differs_from_the_plateau() {
        let window = Window {
            smoothed: 3_300,
            leaks: vec![(3_305, 1.0, 2.0), (2_000, 0.8, 1.6), (1_000, 0.3, 9.0)],
            ..Window::default()
        };
        assert_eq!(offsets(&window), [3_300, 2_000]);
    }

    #[test]
    fn digital_silence_is_not_sharp_and_does_not_vote() {
        let zeros = vec![0.0; 2 * SEARCH as usize + 1];
        let silent = |centre_s: i64| window(centre_s * RATE, &zeros, &zeros);
        let quiet = silent(10);
        assert_eq!((quiet.lag, quiet.sharpness), (0, 0.0));
        assert!(quiet.leaks.is_empty(), "{:?}", quiet.leaks);
        // Two silent windows and one sharp one around a doubtful window: had
        // silence counted as a perfect offset of 0, it would outvote 2000.
        let mut map = vec![
            silent(5),
            silent(10),
            Window {
                centre: 15 * RATE,
                lag: 999,
                sharpness: 1.1,
                ..Window::default()
            },
            Window {
                centre: 20 * RATE,
                lag: 2_000,
                sharpness: 4.0,
                ..Window::default()
            },
        ];
        smooth(&mut map);
        assert_eq!(map[2].smoothed, 2_000);
    }

    #[test]
    fn a_lone_positive_peak_is_as_sharp_as_can_be() {
        let mut r = vec![-0.25; 2 * SEARCH as usize + 1];
        r[100] = 3.0;
        let lone = window(0, &r, &r);
        assert_eq!(lone.lag, (SEARCH - 100) as i32);
        assert_eq!(lone.sharpness, 99.0);
        assert_eq!(lone.leaks, [((SEARCH - 100) as i32, 1.0, 99.0)]);
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("omnifrog-duplicates-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A mono 24-bit WAV at RATE: what the pass reads back is exactly what
    /// it writes.
    fn write_wav(path: &Path, samples: &[f32]) {
        let mut sink = Sink::create(path, 1.0, samples.len() as u64).unwrap();
        sink.write(samples).unwrap();
        assert_eq!(sink.finish().unwrap(), 0);
    }

    #[test]
    fn a_range_shorter_than_a_map_window_only_moves_the_voices() {
        let dir = scratch("short");
        let n = 5 * RATE as usize;
        let voice: Vec<f32> = noise(n, 5).iter().map(|v| v * 0.5).collect();
        // The music and effects ARE the voice: cancelled at an offset of 0,
        // it would be gone.
        let file = |name: &str, samples: &[f32]| {
            let path = dir.join(name);
            write_wav(&path, samples);
            path
        };
        let original = [file("original.wav", &noise(n, 6))];
        let dub = [file("dub.wav", &noise(n, 7))];
        let effects = [file("effects.wav", &voice)];
        let voices = [("C".to_string(), file("voice.wav", &voice))];
        let outputs = [Output {
            channel: "C".to_string(),
            path: dir.join("out.wav"),
            gain: 1.0,
        }];
        let offset = 480;
        let (report, clipped) = cancel(
            &Plan {
                original: &original,
                dub: &dub,
                effects: &effects,
                voices: &voices,
                outputs: &outputs,
                frames: n as u64,
                offset,
            },
            &crate::progress::quiet,
        )
        .unwrap();
        assert_eq!(clipped, 0);
        assert_eq!((report.windows, report.attacks), (0, 0));
        assert!(report.removed_db.is_empty());
        assert_eq!(report.offset_ms, 10.0);
        let text = report.text();
        assert!(text.contains("10.0 ms"), "{text}");
        assert!(
            text.contains("under 20 s): duplicates not cancelled"),
            "{text}"
        );
        // The voice as written (24 bits), 10 ms earlier, sample for sample;
        // silence after it.
        let read = |path: &Path| {
            Signal::open(&[path.to_path_buf()])
                .unwrap()
                .read(0, n)
                .unwrap()
        };
        let (out, voice) = (read(&outputs[0].path), read(&voices[0].1));
        let shift = offset as usize;
        let moved = (0..n - shift)
            .filter(|&i| out[i] != voice[i + shift])
            .count();
        assert_eq!(moved, 0, "samples that are not the voice moved");
        assert!(out[n - shift..].iter().all(|&v| v == 0.0));
        let _ = fs::remove_dir_all(&dir);
    }
}
