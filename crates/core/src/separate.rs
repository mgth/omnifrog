//! The two separations: dialogue out of the original, dialogue out of the
//! dub.
//!
//! Both drive the same worker with the same model family and differ only in
//! what they keep. From each of the original's elements the *rest* is kept
//! — music and effects, the film without its voices — and the voice is kept
//! too, but only so the level of the original's dialogue can be measured.
//! From each of the dub's channels only the voice is kept; the rest is the
//! French film's own music and effects, which nothing here wants.
//!
//! Four things shape how it is done.
//!
//! **One element at a time.** A film's element is a gigabyte, and the
//! worker writes float stereo stems. Handing the worker the whole set at
//! once was tens of gigabytes of temporaries; handing it one element leaves
//! at most one element's worth on disk, and every finished element is won
//! the moment it is finished.
//!
//! **No resampling.** The models are trained at 44.1 kHz and MSST would
//! resample a 48 kHz input down and back for every element. The input is
//! copied with nothing changed but the rate its header states, so the model
//! is handed exactly the samples it wants; the stems come back and are
//! re-labelled 48 kHz. A model that really is 48 kHz (Bandit) states so in
//! the catalogue and gets a true header.
//!
//! **Resume by element.** Each finished element is written to a temporary
//! and renamed, so a file that is there is whole. A second run skips every
//! element whose files are there at the right length, and an interruption
//! costs only the element that was in flight. The set is only thrown away
//! when the *parameters* changed, which a signature in the manifest
//! decides.
//!
//! **Two elements at a time, when asked.** The models are stereo and our
//! elements mono: alone, an element is written on both sides. With the
//! `pairs` switches two elements share a pass — the original's by spatial
//! neighbours from the centre, the dub's left with right — which halves the
//! passes and, where both carry the same voice, leaves less of it behind
//! (see [`Plan`]). The dub's centre can also be separated twice, beside L
//! and beside R, and the two estimates averaged.
//!
//! **The original's depth.** An element without its dialogue is written at
//! the depth the original was mastered at, not at the worker's float: the
//! bits below it are arithmetic, and the coder would carry them all (see
//! `depth`). One an earlier version left at 24 bits is rounded where it is
//! rather than separated again.

use crate::config::{Config, Pairs};
use crate::depth;
use crate::extract;
use crate::job::Job;
use crate::loudness::{self, Voice};
use crate::master::{Keyframe, MasterSet, Trajectories};
use crate::models::{self, Model};
use crate::progress::{Event, Progress};
use crate::wav;
use crate::worker::Worker;
use crate::{Error, Result, voices};
use std::f64::consts::TAU;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Everything the pipeline hands around is at this rate, whatever the
/// model was fed.
pub const RATE: u32 = 48_000;

/// What one of the two separations left behind.
#[derive(Debug, Clone, PartialEq)]
pub struct Separated {
    /// What was separated, in order: an element's identifier or a
    /// channel's name.
    pub elements: Vec<String>,
    /// How many of them a previous run had already done.
    pub resumed: usize,
    /// BS.1770 loudness of the mix of voices, or 0 when it could not be
    /// measured — a valid one is negative.
    pub lkfs: f64,
}

/// Take the dialogue out of every element of the original.
///
/// Every waveform of the master set but the LFE is separated: an object
/// because a voice can be anywhere, and a bed channel because on a DTS:X,
/// an Auro-3D or an Atmos master with a real bed the dialogue lives in the
/// centre. No element is skipped on a detection — a voice the detector
/// misses is a voice that stays in the film under the dubbed one.
pub fn original(
    config: &Config,
    job: &Job,
    model: &Model,
    master: &MasterSet,
    pairs: Pairs,
    progress: Progress,
) -> Result<Separated> {
    let into = job.separate_dir();
    fs::create_dir_all(&into).map_err(|e| Error::io(&into, e))?;

    let elements: Vec<u32> = master
        .channels
        .iter()
        .filter(|channel| !channel.is_lfe())
        .map(|channel| channel.id)
        .collect();
    if elements.is_empty() {
        return Err(Error::msg(
            "the master set holds nothing but an LFE; there is no dialogue to take out",
        ));
    }

    let work: Vec<Unit> = elements
        .iter()
        .map(|id| Unit::of(job, Stage::Original, &id.to_string()))
        .collect();
    let trajectories = Trajectories::read(&master.metadata)?;
    let plan = if pairs.original {
        // Where each element is, on average: an object by its trajectory, a
        // bed channel by its speaker. Only its neighbours matter.
        let rears = master
            .channels
            .iter()
            .any(|c| c.name == "Lrs" || c.name == "Rrs");
        let angles: Vec<f64> = elements
            .iter()
            .zip(&work)
            .map(|(id, unit)| {
                let position = match master.channels.iter().find(|c| c.id == *id) {
                    Some(channel) if !channel.object => voices::position(&channel.name, rears),
                    _ => mean_position(trajectories.of(*id), wav::frames(&unit.source))
                        .unwrap_or([0.0, 1.0, 0.0]),
                };
                azimuth(position)
            })
            .collect();
        let plan = Plan::from_centre(&angles);
        progress(Event::message(format!(
            "in pairs, from the centre: {}",
            plan.describe(&work)
        )));
        plan
    } else {
        Plan::alone(work.len())
    };
    let bits = mastered_at(job, progress);
    let resumed = run(
        config,
        job,
        model,
        &work,
        &plan,
        Stage::Original,
        bits,
        progress,
    )?;

    // The level to bring the dub's dialogue to. Measured on the mix of
    // every voice, each weighted by where it is: an object by its
    // trajectory, a bed channel by its speaker.
    //
    // Weighting *every* element of the original by its trajectory would
    // count a bed channel, whose trajectory is empty, as front. It would
    // not show on a TrueHD Atmos master, whose bed is the LFE alone, but on
    // one with a real bed — DTS:X, Auro-3D — it leaves the original's
    // surrounds a decibel and a half light while the dub's very same
    // surrounds are weighted properly, so the two sides of the ratio are
    // not measured the same way. Weighting them both by name closes that:
    // it moved one DTS:X film by 0.08 dB, and it is what BS.1770 says.
    progress(Event::message("measuring the original's dialogue"));
    let voices: Vec<Voice> = elements
        .iter()
        .map(|id| {
            let path = into.join(format!("{}{id}.wav", extract::ORIGINAL_VOICE));
            match master.channels.iter().find(|c| c.id == *id) {
                // A bed channel does not move: it is weighted by its name.
                Some(channel) if !channel.object => Voice::channel(path, &channel.name),
                _ => Voice::object(path, trajectories.of(*id).to_vec()),
            }
        })
        .collect();
    let measure = loudness::measure(&voices, Some(RATE), progress)?;
    progress(Event::message(format!(
        "the original's dialogue: {}",
        report(&measure)
    )));
    Ok(Separated {
        elements: elements.iter().map(u32::to_string).collect(),
        resumed,
        lkfs: measure.or_unmeasured(),
    })
}

/// Take the dialogue out of every channel of the dub, and decide how the
/// voices will be placed.
pub fn dub(
    config: &Config,
    job: &Job,
    model: &Model,
    channels: &[String],
    pairs: Pairs,
    progress: Progress,
) -> Result<(Separated, voices::Analysis)> {
    let into = job.separate_dir();
    fs::create_dir_all(&into).map_err(|e| Error::io(&into, e))?;

    // The LFE carries no dialogue, and is the one channel never re-voiced.
    let mut wanted: Vec<&String> = channels
        .iter()
        .filter(|name| !name.eq_ignore_ascii_case("LFE"))
        .collect();
    wanted.sort_by_key(|name| SPATIAL_ORDER.iter().position(|o| o == name).unwrap_or(99));
    if wanted.is_empty() {
        return Err(Error::msg("the dub has no channel to take dialogue from"));
    }

    let work: Vec<Unit> = wanted
        .iter()
        .map(|name| Unit::of(job, Stage::Dub, name))
        .collect();
    let names: Vec<&str> = wanted.iter().map(|name| name.as_str()).collect();
    let plan = Plan::of_dub(&names, pairs);
    if plan.passes.len() < work.len() || plan.takes.iter().any(|t| t.len() > 1) {
        progress(Event::message(format!(
            "in pairs: {}",
            plan.describe(&work)
        )));
    }
    // The dub was never the master the stream is coded at: nothing says
    // what its depth was, and its voices are mixed before they are coded.
    let resumed = run(config, job, model, &work, &plan, Stage::Dub, 24, progress)?;

    progress(Event::message("comparing the dub's voices"));
    let analysis = voices::analyse(&into, "C", progress)?;
    let path = voices::report_path(&into);
    fs::write(&path, voices::report(&analysis)).map_err(|e| Error::io(&path, e))?;
    progress(Event::message(if analysis.identical {
        "the dub's channels are copies of its centre: one voice, spread over the front"
    } else {
        "the dub's channels are independent: one voice object each"
    }));

    progress(Event::message("measuring the dub's dialogue"));
    let voices: Vec<Voice> = analysis
        .channels
        .iter()
        .map(|channel| {
            Voice::channel(
                into.join(format!("{}{}.wav", extract::DUB_VOICE, channel.name)),
                &channel.name,
            )
        })
        .collect();
    let measure = loudness::measure(&voices, Some(RATE), progress)?;
    progress(Event::message(format!(
        "the dub's dialogue: {}",
        report(&measure)
    )));
    Ok((
        Separated {
            elements: wanted.iter().map(|name| (*name).clone()).collect(),
            resumed,
            lkfs: measure.or_unmeasured(),
        },
        analysis,
    ))
}

/// The order the dub's channels are worked through: round the room from
/// the left, which makes a progress line readable. The LFE is not here.
const SPATIAL_ORDER: [&str; 7] = ["L", "C", "R", "Rss", "Rrs", "Lrs", "Lss"];

/// Which of the two separations, and therefore what is read and what is
/// kept. One place says where every file of a step lives, so the step, the
/// resume and the check that a finished step is still on disk cannot
/// disagree about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Original,
    Dub,
}

impl Stage {
    /// What a progress line calls one of them.
    fn kind(self) -> &'static str {
        match self {
            Stage::Original => "element",
            Stage::Dub => "channel",
        }
    }

    /// What the step's one progress line is called.
    fn label(self) -> &'static str {
        match self {
            Stage::Original => "dialogue out of the original",
            Stage::Dub => "dialogue out of the dub",
        }
    }

    fn source(self, job: &Job, name: &str) -> PathBuf {
        let prefix = match self {
            Stage::Original => extract::ORIGINAL,
            Stage::Dub => extract::DUB,
        };
        job.extract_dir().join(format!("{prefix}{name}.wav"))
    }

    fn outputs(self, job: &Job, name: &str) -> Vec<Output> {
        let dir = job.separate_dir();
        match self {
            Stage::Original => vec![
                Output {
                    path: dir.join(format!("{}{name}.wav", extract::MUSIC_AND_EFFECTS)),
                    stem: Stem::Rest,
                    // What the stream is coded from.
                    form: Form::Mastered,
                },
                Output {
                    path: dir.join(format!("{}{name}.wav", extract::ORIGINAL_VOICE)),
                    stem: Stem::Voice,
                    // The original's voice is only measured, never mixed,
                    // so it is kept as the worker wrote it: float,
                    // unclipped.
                    form: Form::Float,
                },
            ],
            Stage::Dub => vec![Output {
                path: dir.join(format!("{}{name}.wav", extract::DUB_VOICE)),
                stem: Stem::Voice,
                form: Form::Pcm,
            }],
        }
    }

    /// The prefixes a change of parameter throws away.
    pub fn prefixes(self) -> &'static [&'static str] {
        match self {
            Stage::Original => &[
                extract::MUSIC_AND_EFFECTS,
                extract::ORIGINAL_VOICE,
                "_pair_element_",
            ],
            Stage::Dub => &[extract::DUB_VOICE, "_pair_channel_"],
        }
    }

    /// Where a pass's stem is kept while the element it serves waits for
    /// its other estimate: beside the stems, under a name of its own so a
    /// resume finds it.
    fn kept(self, job: &Job, pass: &str, stem: Stem) -> PathBuf {
        job.separate_dir()
            .join(format!("_pair_{}_{pass}.{}.wav", self.kind(), stem.label()))
    }
}

/// Whether every stem a finished separation recorded is still on disk, at
/// the length of the element it came from.
///
/// A step marked done in the manifest is taken at its word only this far:
/// a stem someone deleted, or one a disk left half-written, sends the step
/// back to be run again — where the resume below redoes that stem alone.
pub fn complete(job: &Job, stage: Stage, names: &[String]) -> bool {
    names.iter().all(|name| {
        let frames = wav::frames(&stage.source(job, name));
        frames > 0
            && stage
                .outputs(job, name)
                .iter()
                .all(|output| wav::holds(&output.path, frames))
    })
}

/// One thing to separate, and what to keep of it.
struct Unit {
    /// What a progress line calls it.
    name: String,
    source: PathBuf,
    outputs: Vec<Output>,
}

impl Unit {
    fn of(job: &Job, stage: Stage, name: &str) -> Self {
        Self {
            name: name.to_string(),
            source: stage.source(job, name),
            outputs: stage.outputs(job, name),
        }
    }
}

struct Output {
    path: PathBuf,
    stem: Stem,
    form: Form,
}

/// How an output is written.
#[derive(Clone, Copy, PartialEq)]
enum Form {
    /// As the worker wrote it.
    Float,
    /// 24-bit PCM.
    Pcm,
    /// 24-bit PCM rounded to the depth the original was mastered at.
    Mastered,
}

impl Form {
    fn sample(self, mastered_at: u32) -> wav::Sample {
        match self {
            Form::Float => wav::Sample::Float,
            Form::Pcm => wav::Sample::Pcm(24),
            Form::Mastered => wav::Sample::Pcm(mastered_at),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Stem {
    Voice,
    Rest,
}

impl Stem {
    fn label(self) -> &'static str {
        match self {
            Stem::Voice => "voice",
            Stem::Rest => "rest",
        }
    }
}

/// One pass of the model: an element alone, written on both sides, or two
/// side by side. Indices are into the separation's list of elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pass {
    pub left: usize,
    pub right: Option<usize>,
}

/// Where an element's estimate comes from: one side of a pass's stems, 0
/// the left and 1 the right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Take {
    pub pass: usize,
    pub side: usize,
}

/// What a separation hands the model, and where each element comes back
/// from.
///
/// Every element has one take, or two when it is estimated twice and the two
/// are averaged (the dub's centre, beside L and beside R). Measured on two
/// films with three models:
/// where two elements carry the same voice, sharing a pass gains 0.5 to
/// 3.9 dB of voice and leaves 0.1 to 5.9 dB less of it in the music and
/// effects, in all fourteen cases; where one element carries a voice alone,
/// a partner is anywhere from 4 dB worse to 6 dB better depending on the
/// scene. An element with no voice is never hurt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub passes: Vec<Pass>,
    pub takes: Vec<Vec<Take>>,
}

impl Plan {
    fn empty(elements: usize) -> Self {
        Self {
            passes: Vec::new(),
            takes: vec![Vec::new(); elements],
        }
    }

    fn add(&mut self, left: usize, right: Option<usize>) {
        let pass = self.passes.len();
        self.passes.push(Pass { left, right });
        self.takes[left].push(Take { pass, side: 0 });
        if let Some(right) = right {
            self.takes[right].push(Take { pass, side: 1 });
        }
    }

    /// Every element alone: what the separations did before the pairs.
    pub fn alone(elements: usize) -> Self {
        let mut plan = Self::empty(elements);
        for element in 0..elements {
            plan.add(element, None);
        }
        plan
    }

    /// The original's elements two by two, round the room: neighbours by
    /// azimuth (radians clockwise from the centre of the screen), starting
    /// at the most central element and going first towards whichever of
    /// its two neighbours is nearer. The dialogue is usually around the
    /// centre, so its two main carriers tend to share the first pass; an
    /// odd element out is left alone. Nothing is detected: it is the order
    /// alone that decides.
    pub fn from_centre(angles: &[f64]) -> Self {
        let n = angles.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|a, b| angles[*a].total_cmp(&angles[*b]));
        let off = |a: f64| a.min(TAU - a);
        if let Some(start) =
            (0..n).min_by(|&i, &j| off(angles[order[i]]).total_cmp(&off(angles[order[j]])))
        {
            order.rotate_left(start);
        }
        let gap = |a: usize, b: usize| {
            let d = (angles[a] - angles[b]).abs();
            d.min(TAU - d)
        };
        if n > 2 && gap(order[0], order[n - 1]) < gap(order[0], order[1]) {
            order[1..].reverse();
        }
        let mut plan = Self::empty(n);
        for chunk in order.chunks(2) {
            plan.add(chunk[0], chunk.get(1).copied());
        }
        plan
    }

    /// The dub's channels, named as the extraction wrote them: left with
    /// right (L+R, Lss+Rss, Lrs+Rrs) when `pairs.dub`, the centre alone —
    /// or, with `pairs.centre_twice`, beside L and beside R, which is where
    /// L and R then come from. A channel without its partner is alone.
    pub fn of_dub(names: &[&str], pairs: Pairs) -> Self {
        let pairs = pairs.effective();
        let index = |name: &str| names.iter().position(|n| *n == name);
        let mut plan = Self::empty(names.len());
        let mut taken = vec![false; names.len()];
        if pairs.centre_twice
            && let (Some(l), Some(c), Some(r)) = (index("L"), index("C"), index("R"))
        {
            plan.add(l, Some(c));
            plan.add(c, Some(r));
            for i in [l, c, r] {
                taken[i] = true;
            }
        }
        if pairs.dub {
            for (a, b) in [("L", "R"), ("Lss", "Rss"), ("Lrs", "Rrs")] {
                if let (Some(a), Some(b)) = (index(a), index(b))
                    && !taken[a]
                    && !taken[b]
                {
                    plan.add(a, Some(b));
                    taken[a] = true;
                    taken[b] = true;
                }
            }
        }
        for (i, _) in taken.iter().enumerate().filter(|(_, taken)| !**taken) {
            plan.add(i, None);
        }
        plan
    }

    /// A progress line's account of the passes: `12+11, 18+16, 10`.
    fn describe(&self, units: &[Unit]) -> String {
        self.passes
            .iter()
            .map(|pass| pass_name(pass, units))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn pass_name(pass: &Pass, units: &[Unit]) -> String {
    match pass.right {
        None => units[pass.left].name.clone(),
        Some(right) => format!("{}+{}", units[pass.left].name, units[right].name),
    }
}

/// Where an object is on average over `frames`: each position weighted by
/// how long it is held. `None` for an object that states none.
fn mean_position(trajectory: &[Keyframe], frames: u64) -> Option<[f64; 3]> {
    let mut sum = [0.0; 3];
    let mut weight = 0.0;
    for (i, (at, position)) in trajectory.iter().enumerate() {
        let until = trajectory
            .get(i + 1)
            .map_or(frames.max(*at + 1), |next| next.0);
        let held = until.saturating_sub(*at) as f64;
        for axis in 0..3 {
            sum[axis] += position[axis] * held;
        }
        weight += held;
    }
    (weight > 0.0).then(|| sum.map(|v| v / weight))
}

/// Radians clockwise from the centre of the screen, in [0, 2π): x runs from
/// the left (-1) to the right (+1), y from the back (-1) to the screen (+1).
fn azimuth(position: [f64; 3]) -> f64 {
    let a = position[0].atan2(position[1]);
    if a < 0.0 { a + TAU } else { a }
}

/// The signature's word for how the original is separated.
pub fn mode_original(pairs: Pairs) -> &'static str {
    if pairs.original { "pairs" } else { "mono" }
}

/// The signature's word for how the dub is separated.
pub fn mode_dub(pairs: Pairs) -> &'static str {
    let pairs = pairs.effective();
    match (pairs.dub, pairs.centre_twice) {
        (false, _) => "mono",
        (true, false) => "pairs",
        (true, true) => "pairs+centre2",
    }
}

/// Separate every unit that is not already done, one pass at a time.
///
/// Returns how many units were found already done. The worker is started at
/// the first pass that actually needs separating and shut down on the way
/// out, so a fully resumed step never loads a checkpoint and never takes the
/// device. `mastered_at` is the depth a [`Form::Mastered`] output is written
/// at.
///
/// A pass is skipped when every unit it serves is done — or, for a unit
/// estimated twice, when the pass's stem is already kept aside for the
/// average. A unit estimated once is written the moment its pass is done;
/// one estimated twice is written once both are, and what was kept aside
/// for it goes.
#[allow(clippy::too_many_arguments)]
fn run(
    config: &Config,
    job: &Job,
    model: &Model,
    units: &[Unit],
    plan: &Plan,
    stage: Stage,
    mastered_at: u32,
    progress: Progress,
) -> Result<usize> {
    let kind = stage.kind();
    let into = job.separate_dir();
    let prepared = into.join(format!("_prepared_{kind}"));
    let stems = into.join(format!("_stems_{kind}"));
    // What a previous run was in the middle of; the finished ones are not
    // in here, and neither is what was kept aside for an average.
    discard(&prepared)?;
    discard(&stems)?;

    let mut frames = Vec::with_capacity(units.len());
    let mut finished = vec![false; units.len()];
    let mut done = 0;
    let mut rounded = 0;
    for (u, unit) in units.iter().enumerate() {
        // A source's own length is what a finished output has to match:
        // anything shorter is a run that was interrupted mid-write.
        let length = wav::frames(&unit.source);
        if length == 0 {
            return Err(Error::msg(format!(
                "{}: not there, or not a WAV; the extraction has to run again",
                unit.source.display()
            )));
        }
        frames.push(length);
        if unit
            .outputs
            .iter()
            .all(|output| wav::holds(&output.path, length))
        {
            finished[u] = true;
            done += 1;
            for output in &unit.outputs {
                if output.form == Form::Mastered
                    && !depth::within(&output.path, mastered_at).unwrap_or(false)
                {
                    reround(&output.path, length, mastered_at)?;
                    rounded += 1;
                }
            }
        }
    }
    if done > 0 {
        progress(Event::message(format!(
            "{done} {kind}{} already done",
            if done > 1 { "s" } else { "" }
        )));
    }
    if rounded > 0 {
        progress(Event::message(format!(
            "{rounded} of them written at 24 bits by an earlier version: rounded to {mastered_at}"
        )));
    }

    let names: Vec<String> = plan.passes.iter().map(|p| pass_name(p, units)).collect();
    let kept = |pass: usize, stem: Stem| stage.kept(job, &names[pass], stem);
    let is_kept = |pass: usize, unit: &Unit| {
        unit.outputs
            .iter()
            .all(|output| wav::frames(&kept(pass, output.stem)) > 0)
    };
    // The units that still need this pass.
    let serves = |pass: usize| -> Vec<usize> {
        (0..units.len())
            .filter(|&u| !finished[u] && plan.takes[u].iter().any(|t| t.pass == pass))
            .collect()
    };
    let todo: Vec<usize> = (0..plan.passes.len())
        .filter(|&pass| {
            serves(pass)
                .iter()
                .any(|&u| plan.takes[u].len() == 1 || !is_kept(pass, &units[u]))
        })
        .collect();
    let averaged: Vec<usize> = (0..units.len())
        .filter(|&u| !finished[u] && plan.takes[u].len() > 1)
        .collect();
    if todo.is_empty() && averaged.is_empty() {
        return Ok(done);
    }

    let total = plan.passes.len();
    let behind = total - todo.len();
    let mut worker: Option<Worker> = None;
    let result = (|| -> Result<()> {
        if !todo.is_empty() {
            // Fetched here rather than at the first request: the worker
            // loads the checkpoint as it starts, so it has to be on disk
            // before that.
            let local = models::fetch(model, config, progress)?;
            for (index, &pass) in todo.iter().enumerate() {
                let started = Instant::now();
                let at = behind + index;
                let step = plan.passes[pass];
                progress(Event::message(format!(
                    "{} {}/{total} ({})",
                    if step.right.is_some() { "pair" } else { kind },
                    at + 1,
                    names[pass]
                )));
                empty(&prepared)?;
                empty(&stems)?;

                // The no-resample trick: the very same samples, under a
                // header that says the rate the model was trained at — one
                // element on its own, or two side by side.
                let input = prepared.join(format!("{}.wav", names[pass]));
                match step.right {
                    None => wav::copy_rate(&units[step.left].source, &input, model.rate)?,
                    Some(right) => wav::pair(
                        &units[step.left].source,
                        &units[right].source,
                        &input,
                        model.rate,
                    )?,
                };

                // A worker that died under a request — a GPU reset takes the
                // device's processes down, whoever hung it — is started
                // again for that one pass, once. A request the worker
                // refused, alive, is not retried: it would fail the same way.
                let mut attempts = 0;
                let separated = loop {
                    attempts += 1;
                    let running = match &mut worker {
                        Some(worker) => worker,
                        none => none.insert(Worker::start(config, model, &local, progress)?),
                    };
                    // One line for the whole step, moving through the
                    // passes: `at` are behind it and `fraction` is how far
                    // into this one.
                    match running.separate(std::slice::from_ref(&input), &stems, |fraction| {
                        progress(Event::ratio(
                            stage.label(),
                            (at as f64 + fraction) / total as f64,
                        ));
                    }) {
                        Ok(separated) => {
                            // An element pymss ran out of device memory on
                            // went through MSST: the same stems, slower.
                            for note in running.take_notes() {
                                progress(Event::message(format!("  {note}")));
                            }
                            break separated;
                        }
                        Err(why) if attempts < 2 && !running.is_alive() => {
                            progress(Event::message(format!(
                                "the worker died ({why}); starting it again for {}",
                                names[pass]
                            )));
                            worker = None;
                            empty(&stems)?;
                        }
                        Err(why) => return Err(why),
                    }
                };
                let stem = &separated[0];
                let from = |which: Stem| match which {
                    Stem::Voice => &stem.voice,
                    Stem::Rest => &stem.rest,
                };

                let mut keep = false;
                for u in serves(pass) {
                    if plan.takes[u].len() > 1 {
                        keep = true;
                        continue;
                    }
                    let side = plan.takes[u][0].side;
                    for output in &units[u].outputs {
                        // Alone, the element is on both sides and the fold
                        // is the element; in a pair, it is its own side.
                        let pick = match step.right {
                            None => wav::Pick::Fold(from(output.stem)),
                            Some(_) => wav::Pick::Channel(from(output.stem), side),
                        };
                        write(&[pick], &output.path, frames[u], output.form, mastered_at)?;
                    }
                }
                if keep {
                    // The stems of a pass one element needs twice wait for
                    // the other one; renamed into place whole.
                    for which in [Stem::Voice, Stem::Rest] {
                        let needed = serves(pass).iter().any(|&u| {
                            plan.takes[u].len() > 1
                                && units[u].outputs.iter().any(|o| o.stem == which)
                        });
                        if needed {
                            let to = kept(pass, which);
                            let part = to.with_extension("wav.part");
                            fs::rename(from(which), &part).map_err(|e| Error::io(&part, e))?;
                            fs::rename(&part, &to).map_err(|e| Error::io(&to, e))?;
                        }
                    }
                }
                progress(Event::message(format!(
                    "  {} done in {:.0} s",
                    names[pass],
                    started.elapsed().as_secs_f64()
                )));
            }
        }
        for &u in &averaged {
            let takes = &plan.takes[u];
            progress(Event::message(format!(
                "{}: the average of its {} estimates",
                units[u].name,
                takes.len()
            )));
            for output in &units[u].outputs {
                let paths: Vec<PathBuf> = takes.iter().map(|t| kept(t.pass, output.stem)).collect();
                let picks: Vec<wav::Pick> = takes
                    .iter()
                    .zip(&paths)
                    .map(|(t, path)| wav::Pick::Channel(path, t.side))
                    .collect();
                write(&picks, &output.path, frames[u], output.form, mastered_at)?;
            }
        }
        // Every average is written: what was kept for them has served.
        for pass in 0..plan.passes.len() {
            for which in [Stem::Voice, Stem::Rest] {
                let _ = fs::remove_file(kept(pass, which));
            }
        }
        progress(Event::ratio(stage.label(), 1.0));
        Ok(())
    })();

    // What a smaller card has to hold: `inference.block_seconds` and
    // `inference.batch_size` bring it down.
    if let Some(peak) = worker.as_ref().and_then(Worker::device_peak) {
        progress(Event::message(format!(
            "device memory: {peak:.1} GiB at most, as torch reserved it"
        )));
    }
    // The device goes back before anything else happens, whether the step
    // finished or fell over. The tidying cannot be allowed to hide why:
    // whatever the step said is what comes out.
    drop(worker);
    let _ = discard(&prepared);
    let _ = discard(&stems);
    let _ = clear_parts(&into);
    result?;
    Ok(done)
}

/// Write one output from `picks` (one, or the estimates to average), aside
/// and then renamed: a file under its real name is always whole, which is
/// what the resume trusts.
fn write(
    picks: &[wav::Pick],
    path: &Path,
    frames: u64,
    form: Form,
    mastered_at: u32,
) -> Result<()> {
    let temporary = path.with_extension("wav.part");
    wav::mix_mono(
        picks,
        &temporary,
        frames,
        RATE,
        1.0,
        form.sample(mastered_at),
    )?;
    fs::rename(&temporary, path).map_err(|e| Error::io(path, e))
}

/// The depth the original was mastered at, said out loud.
fn mastered_at(job: &Job, progress: Progress) -> u32 {
    let (bits, why) = depth::of_original(&job.extract_dir());
    progress(Event::message(match why {
        None => format!("the original is mastered at {bits} bits; its elements are kept at {bits}"),
        Some(why) => format!("{why}: the original's elements are kept at {bits} bits"),
    }));
    bits
}

/// Whether every element of the original a finished separation left is at
/// the depth the original was mastered at. One that is not was written by
/// an earlier version, and the step runs again to round it — which takes
/// seconds, not the GPU.
pub fn at_depth(job: &Job, names: &[String]) -> bool {
    let (bits, _) = depth::of_original(&job.extract_dir());
    names.iter().all(|name| {
        Stage::Original
            .outputs(job, name)
            .iter()
            .filter(|output| output.form == Form::Mastered)
            .all(|output| depth::within(&output.path, bits).unwrap_or(false))
    })
}

/// Round a stem to `bits` where it is: aside, then renamed over it.
fn reround(path: &Path, frames: u64, bits: u32) -> Result<()> {
    let temporary = path.with_extension("wav.part");
    wav::copy_mono(path, &temporary, frames, RATE, 1.0, wav::Sample::Pcm(bits))?;
    fs::rename(&temporary, path).map_err(|e| Error::io(path, e))
}

/// The signature the set of stems belongs to. When it changes, what is on
/// disk was made for another question and is thrown away.
///
/// `mode` is how the elements were handed to the model (`mode_original`,
/// `mode_dub`); the one-at-a-time mode adds nothing, so a job made before
/// the pairs keeps the signature it had and its stems.
pub fn signature(
    model: &crate::job::ModelRef,
    source: &Path,
    track: u32,
    seconds: Option<f64>,
    start: Option<f64>,
    mode: &str,
) -> String {
    let mut base = format!(
        "{}|{}|{}|{}|{}",
        model.key,
        model.checkpoint,
        source.display(),
        track,
        seconds.map(|s| s.to_string()).unwrap_or_else(|| "-".into())
    );
    // Only when there is one, so a run from the beginning keeps the
    // signature its stems were written under.
    if let Some(start) = start {
        base.push_str(&format!("|from {start}"));
    }
    if mode == "mono" {
        base
    } else {
        format!("{base}|{mode}")
    }
}

/// Throw away every stem of one of the two separations: what a changed
/// model, track or range makes stale.
pub fn clear(dir: &Path, stage: Stage) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir).map_err(|e| Error::io(dir, e))? {
        let path = entry.map_err(|e| Error::io(dir, e))?.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if stage
            .prefixes()
            .iter()
            .any(|prefix| name.starts_with(prefix))
        {
            fs::remove_file(&path).map_err(|e| Error::io(&path, e))?;
        }
    }
    Ok(())
}

fn report(measure: &loudness::Measure) -> String {
    if measure.valid() {
        format!(
            "{:.2} LKFS ({}/{} blocks kept, {:.0} s)",
            measure.lkfs, measure.kept, measure.blocks, measure.seconds
        )
    } else {
        "not measurable (too short, or silent)".to_string()
    }
}

fn discard(dir: &Path) -> Result<()> {
    if dir.is_dir() {
        fs::remove_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    }
    Ok(())
}

fn empty(dir: &Path) -> Result<()> {
    discard(dir)?;
    fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))
}

/// A rename that never happened leaves a `.part`; nothing reads one.
fn clear_parts(dir: &Path) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir).map_err(|e| Error::io(dir, e))? {
        let path = entry.map_err(|e| Error::io(dir, e))?.path();
        if path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .ends_with(".part")
        {
            let _ = fs::remove_file(&path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::ModelRef;

    fn model() -> ModelRef {
        ModelRef {
            key: "uhav".into(),
            code: "uhav".into(),
            checkpoint: "bs_roformer_voc_hyperacev2.ckpt".into(),
        }
    }

    #[test]
    fn a_signature_changes_with_everything_that_makes_a_stem_stale() {
        let at = |model: &ModelRef, file: &str, track: u32, seconds, mode| {
            signature(model, Path::new(file), track, seconds, None, mode)
        };
        let base = at(&model(), "/films/a.mkv", 1, None, "mono");
        let mut other_model = model();
        other_model.checkpoint = "other.ckpt".into();
        assert_ne!(base, at(&other_model, "/films/a.mkv", 1, None, "mono"));
        assert_ne!(base, at(&model(), "/films/b.mkv", 1, None, "mono"));
        assert_ne!(base, at(&model(), "/films/a.mkv", 2, None, "mono"));
        assert_ne!(base, at(&model(), "/films/a.mkv", 1, Some(120.0), "mono"));
        assert_ne!(base, at(&model(), "/films/a.mkv", 1, None, "pairs"));
        assert_ne!(
            at(&model(), "/films/a.mkv", 1, None, "pairs"),
            at(&model(), "/films/a.mkv", 1, None, "pairs+centre2")
        );
        assert_eq!(base, at(&model(), "/films/a.mkv", 1, None, "mono"));
        // A trial from another point of the film is other stems.
        let from = |start| {
            signature(
                &model(),
                Path::new("/films/a.mkv"),
                1,
                Some(120.0),
                start,
                "mono",
            )
        };
        assert_ne!(from(None), from(Some(779.0)));
        assert_ne!(from(Some(60.0)), from(Some(779.0)));
    }

    #[test]
    fn one_element_at_a_time_keeps_the_signature_it_had_before_the_pairs() {
        assert_eq!(
            signature(&model(), Path::new("/films/a.mkv"), 1, None, None, "mono"),
            "uhav|bs_roformer_voc_hyperacev2.ckpt|/films/a.mkv|1|-"
        );
    }

    fn degrees(values: &[f64]) -> Vec<f64> {
        values.iter().map(|d| d.to_radians()).collect()
    }

    /// Each pass as (left, right) indices.
    fn passes(plan: &Plan) -> Vec<(usize, Option<usize>)> {
        plan.passes.iter().map(|p| (p.left, p.right)).collect()
    }

    /// Every element taken exactly once, from the side of a pass it is on.
    fn each_once(plan: &Plan) {
        for (element, takes) in plan.takes.iter().enumerate() {
            assert_eq!(takes.len(), 1, "element {element}");
            let pass = plan.passes[takes[0].pass];
            let on = if takes[0].side == 0 {
                Some(pass.left)
            } else {
                pass.right
            };
            assert_eq!(on, Some(element));
        }
    }

    #[test]
    fn the_original_is_paired_round_the_room_from_the_centre() {
        // A TrueHD Atmos master's objects over 25 seconds of its opening: 12
        // at the centre, 11 and 10 either side of it, the rest round the room.
        // Indices:             0=12  1=11  2=18  3=16  4=14   5=20   6=19   7=13   8=15   9=17   10=10
        let plan = Plan::from_centre(&degrees(&[
            0.0, 45.0, 49.0, 90.0, 135.0, 137.0, 224.0, 225.0, 270.0, 309.0, 315.0,
        ]));
        each_once(&plan);
        // Nearer neighbour of the centre is 11 (45°) rather than 10 (315°):
        // the ring goes clockwise, and 10 is the odd one out.
        assert_eq!(
            passes(&plan),
            [
                (0, Some(1)),
                (2, Some(3)),
                (4, Some(5)),
                (6, Some(7)),
                (8, Some(9)),
                (10, None)
            ]
        );
    }

    #[test]
    fn the_ring_turns_towards_the_nearer_neighbour_of_the_centre() {
        // The centre's nearer neighbour is on its left this time.
        let plan = Plan::from_centre(&degrees(&[350.0, 60.0, 180.0, 330.0]));
        each_once(&plan);
        assert_eq!(passes(&plan), [(0, Some(3)), (2, Some(1))]);
        // One element, two elements.
        assert_eq!(passes(&Plan::from_centre(&degrees(&[10.0]))), [(0, None)]);
        assert_eq!(
            passes(&Plan::from_centre(&degrees(&[200.0, 10.0]))),
            [(1, Some(0))]
        );
    }

    #[test]
    fn alone_is_one_pass_per_element() {
        let plan = Plan::alone(3);
        each_once(&plan);
        assert_eq!(passes(&plan), [(0, None), (1, None), (2, None)]);
    }

    #[test]
    fn the_dub_is_paired_left_with_right_and_its_centre_stays_alone() {
        let pairs = Pairs {
            dub: true,
            ..Pairs::default()
        };
        // 5.1, in the order the separation works through it.
        let plan = Plan::of_dub(&["L", "C", "R", "Rss", "Lss"], pairs);
        each_once(&plan);
        assert_eq!(passes(&plan), [(0, Some(2)), (4, Some(3)), (1, None)]);
        // 7.1.
        let plan = Plan::of_dub(&["L", "C", "R", "Rss", "Rrs", "Lrs", "Lss"], pairs);
        each_once(&plan);
        assert_eq!(
            passes(&plan),
            [(0, Some(2)), (6, Some(3)), (5, Some(4)), (1, None)]
        );
        // Without the switch, every channel alone.
        let plan = Plan::of_dub(&["L", "C", "R"], Pairs::default());
        assert_eq!(passes(&plan), [(0, None), (1, None), (2, None)]);
    }

    #[test]
    fn the_centre_twice_costs_no_pass_and_is_averaged() {
        let plan = Plan::of_dub(&["L", "C", "R", "Rss", "Lss"], Pairs::ALL);
        assert_eq!(passes(&plan), [(0, Some(1)), (1, Some(2)), (4, Some(3))]);
        // The centre: the right of [L, C] and the left of [C, R].
        assert_eq!(
            plan.takes[1],
            [Take { pass: 0, side: 1 }, Take { pass: 1, side: 0 }]
        );
        assert_eq!(plan.takes[0], [Take { pass: 0, side: 0 }]);
        assert_eq!(plan.takes[2], [Take { pass: 1, side: 1 }]);
        // Asked without the pairs of the dub, it does nothing.
        let alone = Plan::of_dub(
            &["L", "C", "R"],
            Pairs {
                centre_twice: true,
                ..Pairs::default()
            },
        );
        assert_eq!(passes(&alone), [(0, None), (1, None), (2, None)]);
        // A dub without L or R cannot have it either.
        let mono = Plan::of_dub(&["C"], Pairs::ALL);
        assert_eq!(passes(&mono), [(0, None)]);
    }

    #[test]
    fn the_modes_the_signature_is_told() {
        assert_eq!(mode_original(Pairs::default()), "mono");
        assert_eq!(mode_original(Pairs::ALL), "pairs");
        assert_eq!(mode_dub(Pairs::default()), "mono");
        assert_eq!(
            mode_dub(Pairs {
                dub: true,
                ..Pairs::default()
            }),
            "pairs"
        );
        assert_eq!(mode_dub(Pairs::ALL), "pairs+centre2");
        assert_eq!(
            mode_dub(Pairs {
                centre_twice: true,
                ..Pairs::default()
            }),
            "mono"
        );
    }

    #[test]
    fn an_object_is_where_it_spends_its_time() {
        // Front-centre for 900 frames, hard right for 100.
        let track: Vec<Keyframe> = vec![(0, [0.0, 1.0, 0.0]), (900, [1.0, 1.0, 0.0])];
        let mean = mean_position(&track, 1000).unwrap();
        assert!((mean[0] - 0.1).abs() < 1e-12 && (mean[1] - 1.0).abs() < 1e-12);
        assert!(azimuth(mean) > 0.0 && azimuth(mean) < 0.2);
        assert_eq!(mean_position(&[], 1000), None);
        // Left of the centre is just short of a full turn.
        assert!((azimuth([-1.0, 1.0, 0.0]) - 315f64.to_radians()).abs() < 1e-12);
        assert!((azimuth([1.0, 0.0, 0.0]) - 90f64.to_radians()).abs() < 1e-12);
    }

    #[test]
    fn the_channels_are_worked_through_from_the_left() {
        let mut names: Vec<&str> = vec!["Rrs", "C", "LFE", "L", "Lss"];
        names.sort_by_key(|name| SPATIAL_ORDER.iter().position(|o| o == name).unwrap_or(99));
        assert_eq!(names, ["L", "C", "Rrs", "Lss", "LFE"]);
    }
}
