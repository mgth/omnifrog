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
//!
//! **The original begun during the extraction.** Under pymss this machine
//! separates as many of the original's elements as it keeps up with while
//! the extraction writes them, a block at a time ([`Follower`]); this step
//! finishes those first, and makes the others as above.

use crate::config::{Config, Pairs, Remote};
use crate::depth;
use crate::extract;
use crate::job::Job;
use crate::loudness::{self, Voice};
use crate::master::{Keyframe, MasterSet, Trajectories};
use crate::models::{self, Model};
use crate::progress::{Event, Progress};
use crate::remote::{self, Ssh};
use crate::run::Stop;
use crate::wav;
use crate::worker::{Started, Stems, Worker};
use crate::{Error, Result, voices};
use std::collections::VecDeque;
use std::f64::consts::TAU;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

mod follow;
pub use follow::{Followed, Follower, follows};

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
///
/// `followed` is what was separated while the extraction ran (see
/// [`Follower`]): this machine finishes those elements first, from where
/// they are, while the remotes start on the others.
pub fn original(
    config: &Config,
    job: &Job,
    model: &Model,
    master: &MasterSet,
    pairs: Pairs,
    followed: Option<Followed>,
    progress: Progress,
) -> Result<Separated> {
    original_with(config, job, model, master, pairs, followed, None, progress)
}

/// [`original`], this machine's worker started by `start` when there is
/// one: the tests' mock.
#[allow(clippy::too_many_arguments)]
fn original_with(
    config: &Config,
    job: &Job,
    model: &Model,
    master: &MasterSet,
    pairs: Pairs,
    followed: Option<Followed>,
    start: Option<Start>,
    progress: Progress,
) -> Result<Separated> {
    let into = job.separate_dir();
    fs::create_dir_all(&into).map_err(|e| Error::io(&into, e))?;

    let elements = elements_of(master);
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
    let plan = plan_original(master, &elements, &work, pairs, Some(&trajectories))
        .expect("with the trajectories, every element has a place");
    if pairs.original {
        progress(Event::message(format!(
            "in pairs, from the centre: {}",
            plan.describe(&work)
        )));
    }
    // What the extraction's follower made, if it made it of these very
    // elements, cut into these very passes.
    let followed = followed.and_then(|followed| followed.fits(master, &plan, progress));
    let bits = mastered_at(job, progress);
    let resumed = run(
        config,
        job,
        model,
        &work,
        &plan,
        Stage::Original,
        bits,
        followed,
        start,
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

/// The original's elements a separation works on: every waveform but the
/// LFE, in the master set's order.
fn elements_of(master: &MasterSet) -> Vec<u32> {
    master
        .channels
        .iter()
        .filter(|channel| !channel.is_lfe())
        .map(|channel| channel.id)
        .collect()
}

/// The passes the original's `elements` are made in. In pairs, by where
/// each one is on average: an object by its trajectory, a bed channel by
/// its speaker; only its neighbours matter. Without `trajectories` — the
/// extraction is still writing them — an object has no place yet, and a
/// plan that pairs it is `None`.
fn plan_original(
    master: &MasterSet,
    elements: &[u32],
    work: &[Unit],
    pairs: Pairs,
    trajectories: Option<&Trajectories>,
) -> Option<Plan> {
    if !pairs.original {
        return Some(Plan::alone(work.len()));
    }
    let rears = master
        .channels
        .iter()
        .any(|c| c.name == "Lrs" || c.name == "Rrs");
    let angles = elements
        .iter()
        .zip(work)
        .map(|(id, unit)| {
            let position = match master.channels.iter().find(|c| c.id == *id) {
                Some(channel) if !channel.object => voices::position(&channel.name, rears),
                _ => mean_position(trajectories?.of(*id), wav::frames(&unit.source))
                    .unwrap_or([0.0, 1.0, 0.0]),
            };
            Some(azimuth(position))
        })
        .collect::<Option<Vec<f64>>>()?;
    Some(Plan::from_centre(&angles))
}

/// Starts this machine's worker, killed when the stop is raised: what the
/// tests hand in instead of the inference.
type Start<'a> = &'a (dyn Fn(Stop) -> Result<Worker> + Sync);

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
    dub_with(config, job, model, channels, pairs, None, progress)
}

/// [`dub`], this machine's worker started by `start` when there is one.
fn dub_with(
    config: &Config,
    job: &Job,
    model: &Model,
    channels: &[String],
    pairs: Pairs,
    start: Option<Start>,
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
    let resumed = run(
        config,
        job,
        model,
        &work,
        &plan,
        Stage::Dub,
        24,
        None,
        start,
        progress,
    )?;

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

/// How long a busy remote waits before it is asked again.
const BUSY_RETRY: Duration = Duration::from_secs(60);
/// Failures in a row after which a remote is left out of the step.
const REMOTE_FAILURES: u32 = 2;

/// The passes still to make, shared by the devices that make them: this
/// machine's (0) and the remotes'.
///
/// Once nothing is left in line, this machine does not wait for the
/// remotes: it makes again, itself, the oldest pass one of them is on, so
/// a remote that hangs or crawls costs the step nothing. Whichever of the
/// two has its stems first *claims* the pass and writes its outputs; the
/// other is stopped ([`Stop`]) and what it made is dropped.
struct Shared {
    queue: Mutex<Queue>,
    changed: Condvar,
    /// Per device: raised to stop whatever it is running — its pass was
    /// made first elsewhere, or the step is over.
    stops: Vec<Stop>,
}

struct Queue {
    todo: VecDeque<usize>,
    /// Per device: the pass it is on.
    running: Vec<Option<Run>>,
    /// Per pass of the plan: the device that has its stems and writes its
    /// outputs.
    claimed: Vec<Option<usize>>,
    /// Per pass of the plan: its number in this run, once taken.
    numbers: Vec<usize>,
    /// Passes this run has to make.
    total: usize,
    /// Passes taken in this run, for « pair 3/12 ».
    taken: usize,
    /// Passes finished in this run.
    finished: usize,
    /// What stopped the step.
    failed: Option<Error>,
}

/// A device on a pass.
#[derive(Debug, Clone, Copy)]
struct Run {
    pass: usize,
    /// How far into it the device is.
    fraction: f64,
    since: Instant,
}

/// A pass a device takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Took {
    pass: usize,
    /// Its number in this run.
    number: usize,
    /// The device this one makes the pass again beside; `None` for a pass
    /// nobody is on.
    beside: Option<usize>,
}

impl Shared {
    /// `todo`, out of a plan of `passes`, for `devices` devices.
    fn new(todo: VecDeque<usize>, passes: usize, devices: usize) -> Self {
        Self {
            queue: Mutex::new(Queue {
                total: todo.len(),
                todo,
                running: vec![None; devices],
                claimed: vec![None; passes],
                numbers: vec![0; passes],
                taken: 0,
                finished: 0,
                failed: None,
            }),
            changed: Condvar::new(),
            stops: (0..devices).map(|_| Stop::new()).collect(),
        }
    }

    fn queue(&self) -> std::sync::MutexGuard<'_, Queue> {
        self.queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// What stops `device`'s programs.
    fn stop(&self, device: usize) -> Stop {
        self.stops[device].clone()
    }

    /// The next pass for `device`; `None` when the step is over or failed,
    /// or, for a device that does not `wait`, when nothing is left in line.
    /// One that waits makes again the oldest pass another device is on, and
    /// waits only while every pass under way is being written.
    fn take(&self, device: usize, waits: bool) -> Option<Took> {
        let mut queue = self.queue();
        loop {
            if queue.failed.is_some() || queue.finished == queue.total {
                return None;
            }
            if let Some(pass) = queue.todo.pop_front() {
                queue.taken += 1;
                queue.numbers[pass] = queue.taken;
                return Some(self.start(&mut queue, device, pass, None));
            }
            if !waits {
                return None;
            }
            let oldest = queue
                .running
                .iter()
                .enumerate()
                .filter_map(|(d, run)| run.map(|run| (d, run)))
                .filter(|(d, run)| *d != device && queue.claimed[run.pass].is_none())
                .min_by_key(|(_, run)| run.since);
            if let Some((beside, run)) = oldest {
                return Some(self.start(&mut queue, device, run.pass, Some(beside)));
            }
            if queue.running.iter().all(Option::is_none) {
                return None;
            }
            queue = self
                .changed
                .wait(queue)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    fn start(&self, queue: &mut Queue, device: usize, pass: usize, beside: Option<usize>) -> Took {
        queue.running[device] = Some(Run {
            pass,
            fraction: 0.0,
            since: Instant::now(),
        });
        self.stops[device].lower();
        Took {
            pass,
            number: queue.numbers[pass],
            beside,
        }
    }

    /// `device` is this far into its pass; returns the passes done in this
    /// run, the ones under way counted by their fraction — a pass two
    /// devices are on, once, as far as the further one.
    fn at(&self, device: usize, fraction: f64) -> f64 {
        let mut queue = self.queue();
        if let Some(run) = &mut queue.running[device] {
            run.fraction = fraction;
        }
        let runs: Vec<Run> = queue.running.iter().flatten().copied().collect();
        let under_way: f64 = runs
            .iter()
            .enumerate()
            .filter(|(i, run)| runs[..*i].iter().all(|other| other.pass != run.pass))
            .map(|(_, run)| {
                runs.iter()
                    .filter(|other| other.pass == run.pass)
                    .map(|other| other.fraction)
                    .fold(0.0, f64::max)
            })
            .sum();
        queue.finished as f64 + under_way
    }

    /// `device` has the stems of its pass. `true` if it is the first: it
    /// writes the outputs, and the device making it beside it is stopped.
    fn claim(&self, device: usize) -> bool {
        let mut queue = self.queue();
        let Some(run) = queue.running[device] else {
            return false;
        };
        if queue.claimed[run.pass].is_some() {
            return false;
        }
        queue.claimed[run.pass] = Some(device);
        // Back in line meanwhile — given back by a device whose claim
        // failed, while this one was being stopped — it is made now.
        let before = queue.todo.len();
        queue.todo.retain(|&pass| pass != run.pass);
        if queue.todo.len() < before {
            queue.taken += 1;
        }
        for (d, other) in queue.running.iter().enumerate() {
            if d != device && other.is_some_and(|other| other.pass == run.pass) {
                self.stops[d].raise();
            }
        }
        self.changed.notify_all();
        true
    }

    /// `device` wrote the outputs of the pass it claimed.
    fn done(&self, device: usize) {
        let mut queue = self.queue();
        queue.running[device] = None;
        queue.finished += 1;
        if queue.finished == queue.total {
            // Whatever is still under way — a remote being asked what it
            // is, or waiting to be asked again — has nothing left to do.
            self.stop_all();
        }
        self.changed.notify_all();
    }

    /// `device` is off its pass without having made it — busy, failed,
    /// beaten to it, or stopped — and gives back its claim if it had one.
    /// Unless another device made the pass or is still really on it (not
    /// already told to stop), it goes first in line for another.
    fn give_back(&self, device: usize) {
        let mut queue = self.queue();
        let Some(run) = queue.running[device].take() else {
            return;
        };
        if queue.claimed[run.pass] == Some(device) {
            queue.claimed[run.pass] = None;
        }
        // A device the claim just given back had stopped is no fallback:
        // it will give the pass back in turn, and find it in line.
        let elsewhere = queue.claimed[run.pass].is_some()
            || queue.todo.contains(&run.pass)
            || queue.running.iter().enumerate().any(|(d, other)| {
                other.is_some_and(|other| other.pass == run.pass) && !self.stops[d].raised()
            });
        if !elsewhere {
            queue.taken -= 1;
            queue.todo.push_front(run.pass);
        }
        self.changed.notify_all();
    }

    /// A pass that was not in line — one the extraction's follower began
    /// and could not finish — goes in, at the back, to be made whole.
    fn add(&self, pass: usize) {
        let mut queue = self.queue();
        queue.todo.push_back(pass);
        queue.total += 1;
        self.changed.notify_all();
    }

    fn fail(&self, device: usize, why: Error) {
        let mut queue = self.queue();
        queue.running[device] = None;
        queue.failed.get_or_insert(why);
        self.stop_all();
        self.changed.notify_all();
    }

    fn stop_all(&self) {
        for stop in &self.stops {
            stop.raise();
        }
    }

    /// Up to `timeout`, or until there is nothing left to take.
    fn pause(&self, timeout: Duration) {
        let queue = self.queue();
        let _ = self.changed.wait_timeout_while(queue, timeout, |queue| {
            queue.failed.is_none() && !queue.todo.is_empty()
        });
    }

    fn into_failure(self) -> Option<Error> {
        self.queue
            .into_inner()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .failed
    }
}

/// Separate every unit that is not already done: this machine's GPU one
/// pass at a time, and each of the configuration's remotes beside it, the
/// passes going to whichever is free (see [`crate::remote`]).
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
///
/// The passes in `followed` were begun while the extraction ran: they are
/// not in line, and this machine finishes them first with the worker that
/// began them, the remotes starting on the rest meanwhile. One that fails
/// to finish goes back in line, to be made whole.
#[allow(clippy::too_many_arguments)]
fn run(
    config: &Config,
    job: &Job,
    model: &Model,
    units: &[Unit],
    plan: &Plan,
    stage: Stage,
    mastered_at: u32,
    followed: Option<Followed>,
    start: Option<Start>,
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
    let follow_dir = into.join(follow::DIR);
    if followed.is_none() {
        discard(&follow_dir)?;
    }
    let following: Vec<usize> = followed.as_ref().map_or_else(Vec::new, Followed::passes);

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
        .filter(|pass| !following.contains(pass))
        .filter(|&pass| {
            serves(pass)
                .iter()
                .any(|&u| plan.takes[u].len() == 1 || !is_kept(pass, &units[u]))
        })
        .collect();
    let averaged: Vec<usize> = (0..units.len())
        .filter(|&u| !finished[u] && plan.takes[u].len() > 1)
        .collect();
    if todo.is_empty() && averaged.is_empty() && following.is_empty() {
        return Ok(done);
    }
    if !following.is_empty() {
        progress(Event::message(format!(
            "{} {kind}{} followed during the extraction: this machine finishes {} first",
            following.len(),
            if following.len() > 1 { "s" } else { "" },
            if following.len() > 1 { "them" } else { "it" },
        )));
    }

    let total = plan.passes.len();
    let behind = total - todo.len() - following.len();
    // How much of the followed passes is made, counted in passes.
    let tails = Mutex::new(followed.as_ref().map_or(0.0, Followed::made));
    let mut followed = followed;
    let result = (|| -> Result<()> {
        if !todo.is_empty() || followed.is_some() {
            // Fetched here rather than at the first request: the worker
            // loads the checkpoint as it starts, so it has to be on disk
            // before that.
            let local = match start {
                None => Some(models::fetch(model, config, progress)?),
                Some(_) => None,
            };
            let start_here = |stop: Stop| -> Result<Worker> {
                match (start, &local) {
                    (Some(start), _) => start(stop),
                    (None, Some(local)) => {
                        Worker::start_under(config, model, local, progress, Some(stop))
                    }
                    (None, None) => unreachable!("fetched when there is no start"),
                }
            };
            let shared = Shared::new(
                todo.iter().copied().collect(),
                plan.passes.len(),
                1 + config.remotes.len(),
            );
            // With remotes, every line says which machine it is about.
            let named = !config.remotes.is_empty();
            let prefix = |device: usize| -> String {
                match device {
                    _ if !named => String::new(),
                    0 => "[local] ".to_string(),
                    d => format!("[{}] ", config.remotes[d - 1].host),
                }
            };
            let announce = |device: usize, took: Took| {
                let step = plan.passes[took.pass];
                let again = match took.beside {
                    Some(beside) => format!(
                        " again, beside {}: the first to finish keeps it",
                        config.remotes[beside - 1].host
                    ),
                    None => String::new(),
                };
                progress(Event::message(format!(
                    "{}{} {}/{total} ({}){again}",
                    prefix(device),
                    if step.right.is_some() { "pair" } else { kind },
                    behind + took.number,
                    names[took.pass]
                )));
            };
            let dropped = |device: usize, pass: usize| {
                progress(Event::message(format!(
                    "  {}{}: made first by another machine; this one's is dropped",
                    prefix(device),
                    names[pass]
                )));
            };
            // One line for the whole step, moving through the passes: those
            // behind it, those done in this run, and how far into theirs the
            // devices are.
            let fraction = |device: usize, fraction: f64| {
                let done = shared.at(device, fraction);
                let tails = *tails.lock().unwrap_or_else(|p| p.into_inner());
                progress(Event::ratio(
                    stage.label(),
                    (behind as f64 + tails + done) / total as f64,
                ));
            };
            // The no-resample trick: the very same samples, under a header
            // that says the rate the model was trained at — one element on
            // its own, or two side by side. Each device has its own
            // directories.
            let input_of = |pass: usize, device: usize| -> Result<(PathBuf, PathBuf)> {
                let dir = prepared.join(device.to_string());
                let out = stems.join(device.to_string());
                empty(&dir)?;
                empty(&out)?;
                let step = plan.passes[pass];
                let input = dir.join(format!("{}.wav", names[pass]));
                match step.right {
                    None => wav::copy_rate(&units[step.left].source, &input, model.rate)?,
                    Some(right) => wav::pair(
                        &units[step.left].source,
                        &units[right].source,
                        &input,
                        model.rate,
                    )?,
                };
                Ok((input, out))
            };
            // What a pass's stems become: the outputs of the units it alone
            // serves, or stems kept aside for the ones estimated twice.
            let finish = |pass: usize, stem: &Stems| -> Result<()> {
                let step = plan.passes[pass];
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
                Ok(())
            };
            let done_in = |device: usize, pass: usize, started: Instant| {
                progress(Event::message(format!(
                    "  {}{} done in {:.0} s",
                    prefix(device),
                    names[pass],
                    started.elapsed().as_secs_f64()
                )));
            };

            // This machine: one worker for the whole step, as before there
            // were remotes. With nothing left in line, it makes again the
            // passes the remotes are on rather than wait for them (see
            // `Shared`).
            let mut here = || {
                let stop = shared.stop(0);
                let mut worker: Option<Worker> = None;
                // What the extraction's follower began, finished first by the
                // worker that began it: nobody else has the fades it keeps.
                if let Some(mut began) = followed.take() {
                    worker = began.take_worker();
                    let mut left = Vec::new();
                    for stream in began.streams() {
                        let pass = stream.pass();
                        let started = Instant::now();
                        progress(Event::message(format!(
                            "{}{} {} ({}): the rest of it, followed during the extraction",
                            prefix(0),
                            if plan.passes[pass].right.is_some() {
                                "pair"
                            } else {
                                kind
                            },
                            names[pass],
                            stream.describe(),
                        )));
                        let finished = match &mut worker {
                            Some(running) => {
                                stream.finish(running, units, &frames, mastered_at, |made| {
                                    *tails.lock().unwrap_or_else(|p| p.into_inner()) += made;
                                    fraction(0, 0.0);
                                })
                            }
                            None => Err(Error::msg("the worker that followed it is gone")),
                        };
                        match finished {
                            Ok(()) => done_in(0, pass, started),
                            Err(why) => {
                                progress(Event::message(format!(
                                    "  {}{}: {why}; made whole instead",
                                    prefix(0),
                                    names[pass]
                                )));
                                // Made again from nothing, it counts from nothing.
                                *tails.lock().unwrap_or_else(|p| p.into_inner()) -= stream.made();
                                if worker.as_mut().is_some_and(|w| !w.is_alive()) {
                                    worker = None;
                                }
                                left.push(pass);
                            }
                        }
                    }
                    for pass in left {
                        shared.add(pass);
                    }
                    if let Some(worker) = &mut worker {
                        // From here on it serves the step, as a worker it
                        // had started itself would.
                        worker.set_stop(Some(stop.clone()));
                    }
                }
                while let Some(took) = shared.take(0, true) {
                    let pass = took.pass;
                    let started = Instant::now();
                    announce(0, took);
                    // `Ok(false)`: another device made the pass first.
                    let outcome = (|| -> Result<bool> {
                        let (input, out) = input_of(pass, 0)?;
                        // A worker that died under a request — a GPU reset
                        // takes the device's processes down, whoever hung
                        // it — is started again for that one pass, once. A
                        // request the worker refused, alive, is not retried:
                        // it would fail the same way.
                        let mut attempts = 0;
                        let separated = loop {
                            attempts += 1;
                            let running = match &mut worker {
                                Some(worker) => worker,
                                none => none.insert(start_here(stop.clone())?),
                            };
                            match running
                                .separate(std::slice::from_ref(&input), &out, |f| fraction(0, f))
                            {
                                Ok(separated) => {
                                    // An element pymss ran out of device
                                    // memory on went through MSST: the same
                                    // stems, slower.
                                    for note in running.take_notes() {
                                        progress(Event::message(format!("  {note}")));
                                    }
                                    break separated;
                                }
                                // Stopped: killed under the request, or
                                // told before it began and still loaded.
                                Err(_) if stop.raised() => {
                                    if !running.is_alive() {
                                        worker = None;
                                    }
                                    return Ok(false);
                                }
                                Err(why) if attempts < 2 && !running.is_alive() => {
                                    progress(Event::message(format!(
                                        "the worker died ({why}); starting it again for {}",
                                        names[pass]
                                    )));
                                    worker = None;
                                    empty(&out)?;
                                }
                                Err(why) => return Err(why),
                            }
                        };
                        if !shared.claim(0) {
                            return Ok(false);
                        }
                        finish(pass, &separated[0])?;
                        done_in(0, pass, started);
                        Ok(true)
                    })();
                    match outcome {
                        Ok(true) => shared.done(0),
                        Ok(false) => {
                            dropped(0, pass);
                            shared.give_back(0);
                        }
                        // Stopped while it loaded the model.
                        Err(_) if stop.raised() && worker.is_none() => {
                            dropped(0, pass);
                            shared.give_back(0);
                        }
                        Err(why) => {
                            shared.fail(0, why);
                            break;
                        }
                    }
                }
                // What a smaller card has to hold: `inference.block_seconds`
                // and `inference.batch_size` bring it down.
                if let Some(peak) = worker.as_ref().and_then(Worker::device_peak) {
                    progress(Event::message(format!(
                        "{}device memory: {peak:.1} GiB at most, as torch reserved it",
                        prefix(0)
                    )));
                }
                // The device goes back before anything else happens.
                drop(worker);
            };

            // A remote: one worker for the step too, its model loaded once
            // (see `remote`). It never fails the step: what it does not
            // make, this machine makes. Whatever it runs is killed when its
            // stop is raised.
            let far = |device: usize, remote: &Remote| {
                let who = &remote.host;
                let ssh = Ssh::new(who).stopped_by(shared.stop(device));
                let stop = ssh.stop().clone();
                match remote::probe(remote, &ssh) {
                    // The step was over before the remote answered.
                    Err(_) if stop.raised() => return,
                    Err(why) => {
                        progress(Event::message(format!("{who}: left out: {why}")));
                        return;
                    }
                    Ok(probe) => match probe.unfit() {
                        Some(why) => {
                            progress(Event::message(format!("{who}: left out: {why}")));
                            return;
                        }
                        None => progress(Event::message(format!(
                            "{who}: {}, separating beside this machine",
                            probe.describe()
                        ))),
                    },
                }
                /// How a pass went over there.
                enum Made {
                    Yes,
                    /// Another device made it first.
                    Lost,
                    /// The remote's GPU is someone else's: why.
                    Busy(String),
                }
                let mut failures = 0;
                let mut said_busy = false;
                let mut peak: Option<f64> = None;
                let mut worker: Option<Box<Worker>> = None;
                // The most a worker that is let go reserved on its device.
                let mut keep_peak = |worker: &Worker| {
                    if let Some(p) = worker.device_peak() {
                        peak = Some(peak.map_or(p, |q: f64| q.max(p)));
                    }
                };
                while let Some(took) = shared.take(device, false) {
                    let pass = took.pass;
                    let started = Instant::now();
                    let outcome = (|| -> Result<Made> {
                        // Started at the first pass, and again only after
                        // one that killed it.
                        let running = match &mut worker {
                            Some(worker) => worker,
                            none => match Worker::start_remote(
                                remote,
                                &ssh,
                                model,
                                config.inference,
                                progress,
                            )? {
                                Started::Busy(why) => return Ok(Made::Busy(why)),
                                Started::Ready(worker) => none.insert(worker),
                            },
                        };
                        announce(device, took);
                        let (input, out) = input_of(pass, device)?;
                        let separated =
                            running.separate(std::slice::from_ref(&input), &out, |f| {
                                fraction(device, f)
                            })?;
                        for note in running.take_notes() {
                            progress(Event::message(format!("  {}{note}", prefix(device))));
                        }
                        if !shared.claim(device) {
                            return Ok(Made::Lost);
                        }
                        // Stems that do not make outputs are given back
                        // with the claim, and the pass is made here.
                        finish(pass, &separated[0])?;
                        done_in(device, pass, started);
                        Ok(Made::Yes)
                    })();
                    // A worker the pass killed — stopped under the request,
                    // or dead — is started again for the next one; one that
                    // lives on, after a transfer that failed, is kept.
                    if let Some(dead) = worker.take_if(|worker| !worker.is_alive()) {
                        keep_peak(&dead);
                    }
                    match outcome {
                        Ok(Made::Yes) => {
                            shared.done(device);
                            failures = 0;
                            said_busy = false;
                        }
                        Ok(Made::Lost) => {
                            dropped(device, pass);
                            shared.give_back(device);
                        }
                        Err(_) if stop.raised() => {
                            dropped(device, pass);
                            shared.give_back(device);
                        }
                        Ok(Made::Busy(why)) => {
                            shared.give_back(device);
                            if !said_busy {
                                progress(Event::message(format!(
                                    "{who}: {why}; asked again every {} s",
                                    BUSY_RETRY.as_secs()
                                )));
                                said_busy = true;
                            }
                            shared.pause(BUSY_RETRY);
                        }
                        Err(why) => {
                            shared.give_back(device);
                            failures += 1;
                            progress(Event::message(format!(
                                "{who}: {} failed, made here instead: {why}",
                                names[pass]
                            )));
                            if failures >= REMOTE_FAILURES {
                                progress(Event::message(format!(
                                    "{who}: left out for the rest of this step"
                                )));
                                break;
                            }
                        }
                    }
                }
                if let Some(worker) = &worker {
                    keep_peak(worker);
                }
                // Its GPU goes back as the step ends.
                drop(worker);
                if let Some(peak) = peak {
                    progress(Event::message(format!(
                        "{}device memory: {peak:.1} GiB at most, as torch reserved it",
                        prefix(device)
                    )));
                }
            };

            std::thread::scope(|scope| {
                // With everything followed, the remotes have nothing to take.
                for (index, remote) in config.remotes.iter().enumerate() {
                    let far = &far;
                    if !todo.is_empty() {
                        scope.spawn(move || far(index + 1, remote));
                    }
                }
                here();
                // Whatever a remote still runs — asked what it is, or a pass
                // this machine made first — has no step left to serve.
                shared.stop_all();
            });
            if let Some(why) = shared.into_failure() {
                return Err(why);
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

    // The devices went back as their passes ended, whether the step
    // finished or fell over. The tidying cannot be allowed to hide why:
    // whatever the step said is what comes out.
    let _ = discard(&prepared);
    let _ = discard(&stems);
    let _ = clear_parts(&into);
    drop(followed);
    let _ = discard(&follow_dir);
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

    fn next(pass: usize, number: usize) -> Option<Took> {
        Some(Took {
            pass,
            number,
            beside: None,
        })
    }

    fn again(pass: usize, number: usize, beside: usize) -> Option<Took> {
        Some(Took {
            pass,
            number,
            beside: Some(beside),
        })
    }

    #[test]
    fn with_nothing_left_in_line_this_machine_makes_a_remote_pass_again() {
        let shared = Shared::new([0, 1, 2].into(), 3, 2);
        // The remote takes the first, this machine the other two.
        assert_eq!(shared.take(1, false), next(0, 1));
        assert_eq!(shared.take(0, true), next(1, 2));
        assert!(shared.claim(0));
        shared.done(0);
        assert_eq!(shared.take(0, true), next(2, 3));
        assert_eq!(shared.at(0, 0.5), 1.5);
        assert!(shared.claim(0));
        shared.done(0);
        // Nothing left in line and the remote still on its pass: this
        // machine does not wait for it, it makes it again.
        assert_eq!(shared.take(0, true), again(0, 1, 1));
        // A pass two devices are on counts once, as far as the further.
        shared.at(1, 0.8);
        assert_eq!(shared.at(0, 0.3), 2.8);
        // This machine is first: the remote is stopped, its stems dropped.
        let remote = shared.stop(1);
        assert!(!remote.raised());
        assert!(shared.claim(0));
        assert!(remote.raised());
        assert!(!shared.claim(1));
        shared.give_back(1);
        shared.done(0);
        // Every pass made: whatever still runs is stopped, nobody gets more.
        assert!(shared.stop(0).raised());
        assert_eq!(shared.take(1, false), None);
        assert_eq!(shared.take(0, true), None);
        assert!(shared.into_failure().is_none());
    }

    #[test]
    fn a_remote_first_stops_this_machine_which_then_waits_for_the_end() {
        let shared = Shared::new([0].into(), 1, 2);
        assert_eq!(shared.take(1, false), next(0, 1));
        assert_eq!(shared.take(0, true), again(0, 1, 1));
        assert!(shared.claim(1));
        assert!(shared.stop(0).raised());
        assert!(!shared.claim(0));
        shared.give_back(0);
        // The remote writes the outputs; a pass being written is not made
        // again, so this machine waits, and the step is over when it ends.
        std::thread::scope(|scope| {
            let waiting = scope.spawn(|| shared.take(0, true));
            std::thread::sleep(Duration::from_millis(50));
            assert!(!waiting.is_finished());
            shared.done(1);
            assert_eq!(waiting.join().unwrap(), None);
        });
    }

    #[test]
    fn a_pass_a_remote_gives_back_is_made_here() {
        // Busy, or failed: first in line again.
        let shared = Shared::new([0, 1].into(), 2, 2);
        assert_eq!(shared.take(1, false), next(0, 1));
        shared.give_back(1);
        assert_eq!(shared.take(0, true), next(0, 1));
        assert_eq!(shared.take(1, false), next(1, 2));

        // Given back while this machine makes it again: it stays there.
        let shared = Shared::new([0].into(), 1, 2);
        assert_eq!(shared.take(1, false), next(0, 1));
        assert_eq!(shared.take(0, true), again(0, 1, 1));
        shared.give_back(1);
        assert_eq!(shared.take(1, false), None);
        assert!(shared.claim(0));
        shared.done(0);
        assert_eq!(shared.take(0, true), None);

        // Claimed, then its stems made no outputs: given back with the
        // claim, and made here.
        let shared = Shared::new([0].into(), 1, 2);
        assert_eq!(shared.take(1, false), next(0, 1));
        assert!(shared.claim(1));
        shared.give_back(1);
        assert_eq!(shared.take(0, true), next(0, 1));
        assert!(shared.claim(0));
    }

    #[test]
    fn a_claim_given_back_after_it_stopped_the_other_copy_is_made_again() {
        // The remote wins, then its stems make no outputs; this machine's
        // copy was already told to stop. The pass must not be lost between
        // the two.
        let shared = Shared::new([0].into(), 1, 2);
        assert_eq!(shared.take(1, false), next(0, 1));
        assert_eq!(shared.take(0, true), again(0, 1, 1));
        assert!(shared.claim(1));
        assert!(shared.stop(0).raised());
        shared.give_back(1);
        // This machine's stopped copy gives it back in turn: once in line.
        shared.give_back(0);
        assert_eq!(shared.take(0, true), next(0, 1));
        assert_eq!(shared.take(1, false), None);
        assert!(!shared.stop(0).raised());
        assert!(shared.claim(0));
        shared.done(0);
        assert_eq!(shared.take(0, true), None);
        assert!(shared.into_failure().is_none());

        // Or the stopped copy finishes before it is killed: its claim
        // takes the pass out of line again.
        let shared = Shared::new([0, 1].into(), 2, 2);
        assert_eq!(shared.take(0, true), next(0, 1));
        assert_eq!(shared.take(1, false), next(1, 2));
        assert!(shared.claim(0));
        shared.done(0);
        assert_eq!(shared.take(0, true), again(1, 2, 1));
        assert!(shared.claim(1));
        shared.give_back(1);
        assert!(shared.claim(0));
        shared.done(0);
        assert_eq!(shared.take(1, false), None);
        assert_eq!(shared.take(0, true), None);

        // The same, this machine's copy giving it back first.
        let shared = Shared::new([0].into(), 1, 2);
        assert_eq!(shared.take(1, false), next(0, 1));
        assert_eq!(shared.take(0, true), again(0, 1, 1));
        assert!(shared.claim(1));
        shared.give_back(0);
        shared.give_back(1);
        assert_eq!(shared.take(1, false), next(0, 1));
        assert_eq!(shared.take(0, true), again(0, 1, 1));
    }

    #[test]
    fn a_failure_here_stops_every_device() {
        let shared = Shared::new([0, 1, 2].into(), 3, 2);
        assert_eq!(shared.take(1, false), next(0, 1));
        assert_eq!(shared.take(0, true), next(1, 2));
        shared.fail(0, Error::msg("the disk is full"));
        assert!(shared.stop(1).raised());
        assert_eq!(shared.take(1, false), None);
        assert_eq!(shared.take(0, true), None);
        assert_eq!(
            shared.into_failure().map(|e| e.to_string()),
            Some("the disk is full".to_string())
        );
    }

    #[test]
    fn a_busy_remote_pauses_until_the_line_is_empty() {
        let shared = Shared::new([0].into(), 1, 2);
        std::thread::scope(|scope| {
            let started = Instant::now();
            let pausing = scope.spawn(|| shared.pause(Duration::from_secs(30)));
            std::thread::sleep(Duration::from_millis(50));
            assert_eq!(shared.take(0, true), next(0, 1));
            assert!(shared.claim(0));
            shared.done(0);
            pausing.join().unwrap();
            assert!(started.elapsed() < Duration::from_secs(10));
        });
    }

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
