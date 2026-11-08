//! The original and the dub separated while the extraction is still
//! writing them.
//!
//! The extraction reads the whole container and is bound by the disk: a
//! quarter of an hour for a film, during which the GPU has nothing to do.
//! Every waveform grows in step with the others, so no element is whole
//! before the very end; but the worker cuts an element into blocks anyway
//! (`block_seconds`), and a block only needs the frames it covers and a
//! window past them. So the follower hands each machine's worker one block
//! at a time, as the decoder writes them, and keeps what comes back.
//!
//! **Every machine, as many as each keeps up with.** This machine and every
//! remote that is ready (see [`crate::remote`]) follow side by side, each on
//! a thread of its own, taking elements from one line in the plan's order:
//! one at once, then another each time it made a block since, while the
//! extraction is less than [`BEHIND`] blocks ahead of the next in line —
//! about the first two blocks of film. So a machine takes about as many as
//! it keeps up with, a faster one more. Each serves its own in order, the
//! first with a block ready first; one that is [`BEHIND`] blocks behind is
//! let go anyway — the next one, if the others still do not keep up, once
//! another block is made. What is let go, or left in line, is separated
//! whole after the extraction, by any machine, like the elements nobody
//! followed. A remote is sent what its inputs gained before each block, and
//! its stems come back after it; one that is busy or left out follows
//! nothing, and one still loading its model when the decode ends, or still
//! on a block [`FAR_GRACE`] later, is let go rather than hold the
//! extraction's end up: what it followed is made whole after it.
//!
//! **The dub, in the time the original leaves.** The original rarely keeps
//! a GPU busy all through the extraction, and the dub's channels grow in
//! step with it. Each machine has a second worker, with the dub's model,
//! that follows the dub's channels the same way, from a line of their own.
//! The machine's one thread drives both workers, one request at a time —
//! a GPU never separates two things at once (see [`Lock`]) — and the
//! original comes first: its block when one is ready, and its line when
//! there is time for one more. So the dub gets what the original leaves,
//! and what falls behind of it is let go, as above, before anything of the
//! original is. The two workers load side by side with the first blocks,
//! the dub's after the original's.
//!
//! **The same stems.** The worker cuts an element by a plan that is a
//! function of its length alone, whole or growing, and fades one block into
//! the next the same way (`plan_blocs`, `fond_bloc`): an element comes out
//! the same, to rounding, whether it was followed or made whole. Each block
//! is folded into the element's outputs as it comes back — through the same
//! [`wav::Reader`] as a whole stem — so at most one block's stems are ever on
//! disk. The element without its dialogue is kept in float until the end of
//! the extraction, when the depth the original was mastered at is known.
//!
//! **Handed to the next steps.** When the decode ends, what each element
//! still needs is its last block or two, and the fade the worker keeps for
//! it: each machine's worker and what it followed go to the separation they
//! belong to ([`Followed`], [`Handed`]), where each machine finishes its own
//! first, then takes from the line. The original's go to the step after the
//! extraction; the dub's wait for the dub's, loaded, beside it. Each model
//! is loaded once for the extraction and its step, on every machine. What a
//! remote finishes is made again here, whole, if it takes too long, as any
//! pass a remote is on.
//!
//! **Nothing half-made survives.** A decode that fails or is stopped kills
//! every worker at once and drops everything (the extraction runs again
//! from the beginning anyway); a run that ends before a separation does
//! too. Nothing a follower writes is under an output's name until it is
//! whole.
//!
//! Not followed: anything under MSST, which separates whole elements only,
//! or on a CPU, which the decoder needs; the original in pairs when it has
//! objects, whose pairs are drawn from where each one is over the whole
//! film; and the dub in pairs.

use super::{
    Beside, Form, Plan, Stage, Stem, Unit, discard, dub_work, elements_of, pass_name, plan_original,
};
use crate::config::{Config, Engine, Pairs};
use crate::extract;
use crate::job::Job;
use crate::master::MasterSet;
use crate::models::{self, Model};
use crate::progress::{Event, Progress};
use crate::remote::{self, Seat, Ssh};
use crate::run::Stop;
use crate::worker::{Ask, Grid, Lock, REST_SUFFIX, Started, VOICE_SUFFIX, Worker};
use crate::{Error, Result, wav};
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, SystemTime};

/// Where a follower keeps what it grows, in the separation's directory.
pub(super) fn dir_of(stage: Stage) -> &'static str {
    match stage {
        Stage::Original => "_follow_element",
        Stage::Dub => "_follow_channel",
    }
}

/// What a follower follows, in the order a machine serves them.
const STAGES: [Stage; 2] = [Stage::Original, Stage::Dub];
/// Blocks a followed element may be behind the extraction before the last
/// of the followed is let go.
const BEHIND: u64 = 2;
/// How much older than the follower a master set's header may look and
/// still be this extraction's: some file systems keep coarse times.
const FRESH: Duration = Duration::from_secs(2);
/// How long a remote may still take over the block it is on when the
/// decode ends — seconds, normally — before it is let go: one that hangs
/// would otherwise hold the extraction's end, and every GPU, for as long as
/// a worker is given between two words.
const FAR_GRACE: Duration = if cfg!(test) {
    Duration::from_secs(1)
} else {
    Duration::from_secs(60)
};

/// Whether this configuration separates the original during the extraction:
/// under pymss, which cuts an element into blocks.
pub fn follows(config: &Config) -> bool {
    config.inference.engine == Engine::Pymss
}

/// Whether the dub is followed too: under pymss, channel by channel. In
/// pairs, it is separated after the extraction.
pub fn follows_dub(config: &Config, pairs: Pairs) -> bool {
    follows(config) && !pairs.effective().dub
}

/// What a separation's step is called in a line about it.
fn the(stage: Stage) -> &'static str {
    match stage {
        Stage::Original => "the original",
        Stage::Dub => "the dub",
    }
}

/// Starts a stage's worker on this machine, killed when the stop is
/// raised: what the tests hand in instead of the inference.
type Starts<'a> = &'a (dyn Fn(Stage, Stop) -> Result<Worker> + Sync);

/// A stage's worker started on a machine, or why not: `Ok(None)` for one
/// left out, which is said.
type Startup = (usize, Result<Option<Worker>>);

/// Follows the original's elements and the dub's channels as the extraction
/// writes them: one thread runs [`Follower::follow`] while the extraction
/// runs, and the extraction calls [`Follower::decoded`] once the decoder is
/// done.
pub struct Follower<'a> {
    config: &'a Config,
    job: &'a Job,
    /// Per stage, in [`STAGES`]' order: its model.
    models: [&'a Model; 2],
    /// The dub's channels, as the extraction names them: none, the dub is
    /// not followed.
    dub: &'a [String],
    pairs: Pairs,
    /// The tests' mock worker; the inference otherwise.
    start: Option<Starts<'a>>,
    /// How often the decoder's files are looked at.
    poll: Duration,
    /// Per stage, per device: raised, that worker and its commands are
    /// killed.
    stops: [Vec<Stop>; 2],
    /// Per stage, per remote: its worker is loaded, and follows.
    following: [Vec<AtomicBool>; 2],
    /// Per remote: it is done following.
    over: Vec<AtomicBool>,
    /// Per remote: let go, still on a block [`FAR_GRACE`] after the decode
    /// ended.
    late: Vec<AtomicBool>,
    end: Mutex<End>,
    changed: Condvar,
    /// Per machine, the times it found nothing to do once following: the
    /// tests' extraction keeps in step with them.
    #[cfg(test)]
    waits: Vec<std::sync::atomic::AtomicU64>,
    /// How many blocks the tests let the machines make: a GPU that keeps up
    /// with only so much of the extraction, in step with it.
    #[cfg(test)]
    gate: tests::Gate,
}

#[derive(Default)]
struct End {
    /// The decode is over, and whether it succeeded.
    decoded: Option<bool>,
    /// The follower makes no more requests.
    quiet: bool,
}

/// What the machines following one separation share.
struct Common<'f> {
    stage: Stage,
    model: &'f Model,
    plan: Plan,
    units: Vec<Unit>,
    /// Per pass of the plan: the files it reads while the extraction writes
    /// them, its left element's, then its right one's.
    raw: Vec<Vec<PathBuf>>,
    /// Per pass of the plan: its name.
    names: Vec<String>,
    /// What it is followed on, for the separation to check it is what it is
    /// about to make ([`Followed::fits`]).
    identity: Vec<String>,
    /// Where the followed elements grow.
    dir: PathBuf,
    /// The passes no machine took yet, in the plan's order.
    line: Mutex<VecDeque<usize>>,
    /// Several machines: every line says which one it is about.
    named: bool,
    /// What the step's progress line is drawn from: per pass of the plan,
    /// the machine on it and the frames made of it so far. A pass let go,
    /// or followed by a machine left out, loses what was made of it.
    made: Mutex<Vec<Option<(usize, u64)>>>,
    /// The frames an element holds once the decoder is done, as far as the
    /// film's duration says; nought when it does not, and no line is drawn.
    whole: u64,
    /// The passes an earlier run made whole.
    before: usize,
}

impl Common<'_> {
    /// `device` has made `frames` of `pass` so far, or (`None`) what was
    /// made of it is lost; and how far the step is, said.
    fn made(&self, pass: usize, by: Option<(usize, u64)>, progress: Progress) {
        let mut made = self.made.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(slot) = made.get_mut(pass) {
            *slot = by;
        }
        self.tell(&made, progress);
    }

    /// `device` is left out: what it was making is lost.
    fn lost(&self, device: usize, progress: Progress) {
        let mut made = self.made.lock().unwrap_or_else(|p| p.into_inner());
        let mut any = false;
        for slot in made.iter_mut() {
            if slot.is_some_and(|(on, _)| on == device) {
                *slot = None;
                any = true;
            }
        }
        if any {
            self.tell(&made, progress);
        }
    }

    /// How far the separation is, said under its step's own name rather
    /// than the extraction's, so that a front end draws it as that step
    /// started early (see [`crate::job::Step::labelled`]): what an earlier
    /// run made, and what the machines made since, over every pass. The
    /// same measure the step itself reports once it runs, so the line
    /// carries on where this one stopped. Said under the lock, so that two
    /// machines' blocks are said in the order they were made.
    fn tell(&self, made: &[Option<(usize, u64)>], progress: Progress) {
        if self.whole == 0 {
            return;
        }
        let frames: u64 = made.iter().flatten().map(|(_, frames)| *frames).sum();
        let passes = self.plan.passes.len().max(1) as f64;
        progress(Event::ratio(
            self.stage.label(),
            (self.before as f64 + frames as f64 / self.whole as f64) / passes,
        ));
    }

    /// What a line calls its passes: `elements`, `pairs`, `channels`.
    fn kind(&self) -> &'static str {
        match self.stage {
            Stage::Dub => "channels",
            Stage::Original if self.plan.passes.len() < self.units.len() => "pairs",
            Stage::Original => "elements",
        }
    }
}

/// One stage's worker on one machine, and what it has come to.
struct Side {
    /// Its worker, once loaded, and what it follows.
    machine: Option<Machine>,
    /// How the worker cuts an element; `None` until it is loaded, or for
    /// one that follows nothing.
    grid: Option<Grid>,
    /// Its worker is still loading.
    loading: bool,
    /// Its line may still give it a pass.
    open: bool,
    /// What left it out of the extraction, and whether it was stopped
    /// then — the decode failed, or a remote was still loading as it ended:
    /// nothing to say.
    failed: Option<(Error, bool)>,
}

/// What a block request on a side came to.
enum Stepped {
    Made,
    /// The worker is gone, or the decode failed: the side is out.
    Out(Error),
    /// The worker refused that element: it is let go.
    Refused(Error),
}

/// What a machine is called: `device` 0 is this one, `d` the remote
/// `remotes[d - 1]`.
fn name_of(config: &Config, device: usize) -> &str {
    match device {
        0 => "this machine",
        d => &config.remotes[d - 1].host,
    }
}

/// The frames a film's element holds once the extraction is done, as far
/// as the manifest says; nought when it does not.
fn whole_of(job: &Job) -> u64 {
    let manifest = &job.manifest;
    let seconds = extract::wanted(manifest.duration, manifest.start, manifest.seconds);
    (seconds * extract::RATE as f64) as u64
}

/// What the original was followed on: the master set's elements as it read
/// them, the decoder's rank and the identifier.
pub(super) fn identity(master: &MasterSet) -> Vec<String> {
    master
        .channels
        .iter()
        .map(|c| format!("{}:{}", c.index, c.id))
        .collect()
}

impl<'a> Follower<'a> {
    /// A follower of the original with `original`, and of the dub's
    /// `channels` with `dub` when they are followed too ([`follows_dub`]).
    pub fn new(
        config: &'a Config,
        job: &'a Job,
        original: &'a Model,
        dub: &'a Model,
        channels: &'a [String],
        pairs: Pairs,
    ) -> Self {
        let remotes = config.remotes.len();
        let stops = || (0..=remotes).map(|_| Stop::new()).collect();
        let flags = || (0..remotes).map(|_| AtomicBool::new(false)).collect();
        Self {
            config,
            job,
            models: [original, dub],
            dub: channels,
            pairs,
            start: None,
            poll: extract::POLL,
            stops: [stops(), stops()],
            following: [flags(), flags()],
            over: flags(),
            late: flags(),
            end: Mutex::new(End::default()),
            changed: Condvar::new(),
            #[cfg(test)]
            waits: (0..=remotes).map(|_| Default::default()).collect(),
            #[cfg(test)]
            gate: tests::Gate::default(),
        }
    }

    fn end(&self) -> MutexGuard<'_, End> {
        self.end.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn decode_ended(&self) -> Option<bool> {
        self.end().decoded
    }

    /// Nothing to do on `device`: up to a poll, or until the decode ends.
    fn wait(&self, device: usize) {
        #[cfg(test)]
        self.waits[device].fetch_add(1, Ordering::SeqCst);
        #[cfg(not(test))]
        let _ = device;
        self.pause(self.poll);
    }

    /// Up to `timeout`, or until the decode ends.
    fn pause(&self, timeout: Duration) {
        let end = self.end();
        let _ = self
            .changed
            .wait_timeout_while(end, timeout, |end| end.decoded.is_none());
    }

    /// The decode is over: `ok` when it succeeded. Returns once the follower
    /// makes no more requests — after the blocks in flight, which are worth
    /// having, or at once when the decode failed: every worker is killed and
    /// nothing they made is kept. A remote's worker still loading its model
    /// is let go rather than hold the extraction's end up, and so is a
    /// remote still on its block [`FAR_GRACE`] later: the separations make
    /// what it followed whole. Called again, it only waits.
    pub fn decoded(&self, ok: bool) {
        let mut end = self.end();
        if end.decoded.is_none() {
            end.decoded = Some(ok);
            for (stops, following) in self.stops.iter().zip(&self.following) {
                if !ok {
                    stops[0].raise();
                }
                for (stop, following) in stops[1..].iter().zip(following) {
                    if !ok || !following.load(Ordering::SeqCst) {
                        stop.raise();
                    }
                }
            }
            self.changed.notify_all();
        }
        let (waited, _) = self
            .changed
            .wait_timeout_while(end, FAR_GRACE, |end| !end.quiet)
            .unwrap_or_else(|p| p.into_inner());
        end = waited;
        if !end.quiet {
            for remote in 0..self.over.len() {
                if !self.over[remote].load(Ordering::SeqCst) {
                    self.late[remote].store(true, Ordering::SeqCst);
                    for stops in &self.stops {
                        stops[remote + 1].raise();
                    }
                }
            }
        }
        while !end.quiet {
            end = self.changed.wait(end).unwrap_or_else(|p| p.into_inner());
        }
    }

    /// Follow the original, and the dub, until the decode ends. What was
    /// followed of each, with the workers that followed it, when the decode
    /// succeeded; nothing when it failed.
    pub fn follow(&self, progress: Progress) -> Handed {
        /// Says, however the follower ends, that it makes no more requests.
        struct Quiet<'f, 'a>(&'f Follower<'a>);
        impl Drop for Quiet<'_, '_> {
            fn drop(&mut self) {
                self.0.end().quiet = true;
                self.0.changed.notify_all();
            }
        }
        let _quiet = Quiet(self);
        match self.run(progress) {
            Ok(handed) if self.decode_ended() == Some(true) => handed,
            Ok(_) => Handed::default(),
            Err(why) => {
                if self.decode_ended() != Some(false) {
                    progress(Event::message(format!(
                        "nothing is separated during the extraction: {why}"
                    )));
                }
                Handed::default()
            }
        }
    }

    fn run(&self, progress: Progress) -> Result<Handed> {
        let raw = self.job.extract_dir();
        let into = self.job.separate_dir();
        // What a run that stopped left.
        for stage in STAGES {
            discard(&into.join(dir_of(stage)))?;
        }
        let since = SystemTime::now()
            .checked_sub(FRESH)
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let header = raw.join(extract::MASTER).with_extension("atmos");
        // The decoder writes the header as soon as it knows the elements:
        // seconds into the film. One left by an earlier run is not it. The
        // dub's files are this run's by then: ffmpeg opened them before it
        // fed the decoder anything.
        let master = loop {
            if self.decode_ended().is_some() {
                return Ok(Handed::default());
            }
            let fresh = fs::metadata(&header)
                .and_then(|m| m.modified())
                .is_ok_and(|at| at >= since);
            if fresh && let Ok(master) = MasterSet::read(&header) {
                break master;
            }
            self.pause(self.poll);
        };
        let commons = [
            self.original_common(&master, progress)?,
            self.dub_common(progress)?,
        ];
        if commons.iter().all(Option::is_none) {
            return Ok(Handed::default());
        }

        let remotes = &self.config.remotes;
        let on = match remotes.len() {
            0 => "on this machine".to_string(),
            1 => format!("on this machine and on {}", remotes[0].host),
            n => format!("on this machine and on {n} others"),
        };
        let keep = if remotes.is_empty() {
            "it keeps"
        } else {
            "they keep"
        };
        for common in commons.iter().flatten() {
            let wanted = common.line.lock().map_or(0, |line| line.len());
            progress(Event::message(match common.stage {
                Stage::Dub if commons[0].is_some() => format!(
                    "separating the dub too, in the time the original leaves: as many of its \
                     {wanted} channels as {keep} up with"
                ),
                stage => format!(
                    "separating {} as the extraction writes it, {on}: as many of its {wanted} {} \
                     as {keep} up with",
                    the(stage),
                    common.kind()
                ),
            }));
        }
        // Each machine on a thread of its own, this one on this thread.
        let mut machines: Vec<[Option<Machine>; 2]> = std::thread::scope(|scope| {
            let commons = &commons;
            let far: Vec<_> = (1..=remotes.len())
                .map(|device| scope.spawn(move || self.machine(device, commons, progress)))
                .collect();
            let mut machines = vec![self.machine(0, commons, progress)];
            machines.extend(
                far.into_iter()
                    .map(|far| far.join().unwrap_or([None, None])),
            );
            machines
        });
        let mut handed = [None, None];
        for (s, common) in commons.iter().enumerate() {
            let Some(common) = common else { continue };
            let followed = Followed {
                stage: common.stage,
                machines: machines
                    .iter_mut()
                    .filter_map(|machine| machine[s].take())
                    .collect(),
                plan: common.plan.clone(),
                identity: common.identity.clone(),
                dir: common.dir.clone(),
            };
            // Dropped, it takes what it grew with it.
            if self.decode_ended() != Some(true) || followed.machines.is_empty() {
                continue;
            }
            self.tell_followed(common, &followed, progress);
            handed[s] = Some(followed);
        }
        let [original, dub] = handed;
        Ok(Handed { original, dub })
    }

    /// The original, if any of it can be followed and is still to make.
    fn original_common(
        &self,
        master: &MasterSet,
        progress: Progress,
    ) -> Result<Option<Common<'a>>> {
        let elements = elements_of(master);
        let units: Vec<Unit> = elements
            .iter()
            .map(|id| Unit::of(self.job, Stage::Original, &id.to_string()))
            .collect();
        let Some(plan) = plan_original(master, &elements, &units, self.pairs, None) else {
            progress(Event::message(
                "the original's elements are paired by where each one is over the whole film, \
                 known once it is read: they are separated after the extraction",
            ));
            return Ok(None);
        };
        let raw = (0..plan.passes.len())
            .map(|pass| raw_of(&plan, pass, &units, master))
            .collect::<Result<Vec<_>>>()?;
        Ok(self.common(Stage::Original, 0, plan, units, raw, identity(master)))
    }

    /// The dub, if it is followed and any of it is still to make.
    fn dub_common(&self, progress: Progress) -> Result<Option<Common<'a>>> {
        if self.dub.is_empty() {
            return Ok(None);
        }
        if self.pairs.effective().dub {
            progress(Event::message(
                "the dub's channels are separated in pairs: after the extraction",
            ));
            return Ok(None);
        }
        // A dub with nothing to separate is its step's to say.
        let Ok((names, units, plan)) = dub_work(self.job, self.dub, self.pairs) else {
            return Ok(None);
        };
        // The dub's channels are written under their names from the start.
        let raw = plan
            .passes
            .iter()
            .map(|pass| {
                std::iter::once(pass.left)
                    .chain(pass.right)
                    .map(|u| units[u].source.clone())
                    .collect()
            })
            .collect();
        Ok(self.common(Stage::Dub, 1, plan, units, raw, names))
    }

    /// What the machines following `stage`, the `s`-th of [`STAGES`], share,
    /// if a previous run did not
    /// make all of it already: its stems for these parameters are the
    /// separation's to check, at their length.
    fn common(
        &self,
        stage: Stage,
        s: usize,
        plan: Plan,
        units: Vec<Unit>,
        raw: Vec<Vec<PathBuf>>,
        identity: Vec<String>,
    ) -> Option<Common<'a>> {
        let wanted: VecDeque<usize> = (0..plan.passes.len())
            .filter(|&pass| {
                serving(&plan, pass).any(|(u, _)| {
                    !units[u]
                        .outputs
                        .iter()
                        .all(|output| wav::frames(&output.path) > 0)
                })
            })
            .collect();
        if wanted.is_empty() {
            return None;
        }
        let names: Vec<String> = plan.passes.iter().map(|p| pass_name(p, &units)).collect();
        Some(Common {
            stage,
            model: self.models[s],
            before: plan.passes.len() - wanted.len(),
            made: Mutex::new(vec![None; plan.passes.len()]),
            line: Mutex::new(wanted),
            plan,
            units,
            raw,
            names,
            identity,
            dir: self.job.separate_dir().join(dir_of(stage)),
            named: !self.config.remotes.is_empty(),
            whole: whole_of(self.job),
        })
    }

    /// The run log's account of what was followed of a separation.
    fn tell_followed(&self, common: &Common, followed: &Followed, progress: Progress) {
        let streams: Vec<&Stream> = followed
            .machines
            .iter()
            .flat_map(|machine| &machine.streams)
            .collect();
        if streams.is_empty() {
            return;
        }
        let (made, length): (u64, u64) = streams
            .iter()
            .fold((0, 0), |(m, l), s| (m + s.written, l + s.length));
        let split = if common.named {
            let each: Vec<String> = followed
                .machines
                .iter()
                .filter(|machine| !machine.streams.is_empty())
                .map(|machine| match machine.device {
                    0 => format!("{} here", machine.streams.len()),
                    d => format!(
                        "{} on {}",
                        machine.streams.len(),
                        self.config.remotes[d - 1].host
                    ),
                })
                .collect();
            format!(" ({})", each.join(", "))
        } else {
            String::new()
        };
        let (whose, next) = match common.stage {
            Stage::Original => ("the original's", "the next step"),
            Stage::Dub => ("the dub's", "the dub's step"),
        };
        progress(Event::message(format!(
            "during the extraction: {} of {whose} {} {} followed{split}, {:.0} % of them \
             separated; the rest of them is made first in {next}",
            streams.len(),
            common.plan.passes.len(),
            common.kind(),
            100.0 * made as f64 / length.max(1) as f64,
        )));
    }

    /// One machine following, this one (`device` 0) or a remote: per stage,
    /// what it followed and its worker once the decode is over; `None` for
    /// a stage it has no worker of to hand on. What goes wrong on one is
    /// said, and leaves what it followed to be separated after the
    /// extraction.
    fn machine(
        &self,
        device: usize,
        commons: &[Option<Common>; 2],
        progress: Progress,
    ) -> [Option<Machine>; 2] {
        let sides = self.follow_on(device, commons, progress);
        if device > 0 {
            self.over[device - 1].store(true, Ordering::SeqCst);
        }
        // Done: the tests' extraction no longer waits for it.
        #[cfg(test)]
        self.waits[device].store(tests::DONE, Ordering::SeqCst);
        let late = device > 0 && self.late[device - 1].load(Ordering::SeqCst);
        if late && sides.iter().any(|side| side.failed.is_some()) {
            progress(Event::message(format!(
                "{}: still on a block {} s after the extraction ended: let go; what it \
                 followed is separated after the extraction",
                name_of(self.config, device),
                FAR_GRACE.as_secs()
            )));
        }
        let both = commons.iter().all(Option::is_some);
        let [original, dub] = sides;
        let outcome = |s: usize, side: Side| -> Option<Machine> {
            let Some((why, stopped)) = side.failed else {
                return side.machine;
            };
            let common = commons[s].as_ref()?;
            common.lost(device, progress);
            if late || stopped {
                return None;
            }
            let whose = the(common.stage);
            progress(Event::message(if device == 0 && !common.named {
                format!("{whose} is separated after the extraction: {why}")
            } else if both {
                format!(
                    "{}: left out of the extraction for {whose}: {why}; what it followed of it is \
                     separated after it",
                    name_of(self.config, device)
                )
            } else {
                format!(
                    "{}: left out of the extraction: {why}; what it followed is separated after it",
                    name_of(self.config, device)
                )
            }));
            None
        };
        [outcome(0, original), outcome(1, dub)]
    }

    /// `device` following, until the decode ends: its workers are loaded on
    /// a thread of their own, and it serves each as soon as it is there.
    fn follow_on(
        &self,
        device: usize,
        commons: &[Option<Common>; 2],
        progress: Progress,
    ) -> [Side; 2] {
        let mut sides = [0, 1].map(|s| Side {
            machine: None,
            grid: None,
            loading: commons[s].is_some(),
            open: commons[s].is_some(),
            failed: None,
        });
        let wanted = [commons[0].is_some(), commons[1].is_some()];
        let (sender, started) = channel();
        std::thread::scope(|scope| {
            scope.spawn(move || self.start_all(device, wanted, sender, progress));
            self.serve(device, commons, &mut sides, &started, progress);
        });
        // What loaded once the decode was over: handed on as it is.
        while let Ok((s, worker)) = started.try_recv() {
            self.arrived(
                device,
                s,
                &mut sides[s],
                commons[s].as_ref(),
                worker,
                progress,
            );
        }
        if self.decode_ended() != Some(true) {
            return sides;
        }
        // The decoder is done: every file is as long as it will be.
        for side in &mut sides {
            let Some(machine) = &mut side.machine else {
                continue;
            };
            let (begun, idle): (Vec<Stream>, Vec<Stream>) = std::mem::take(&mut machine.streams)
                .into_iter()
                .partition(|stream| stream.blocks > 0);
            for stream in idle {
                machine.let_go(stream);
            }
            machine.streams = begun;
            for stream in &mut machine.streams {
                stream.length = stream.held();
            }
        }
        sides
    }

    /// One block at a time, the original's first: the first element with
    /// one ready first. A pass is taken from a line at once, and another
    /// each time a block was made since, while the line is not too far
    /// behind — the original's line first. A machine that falls behind on
    /// any lets the last of the dub's go, or with none left the last of the
    /// original's, once a block was made since the last went.
    fn serve(
        &self,
        device: usize,
        commons: &[Option<Common>; 2],
        sides: &mut [Side; 2],
        started: &Receiver<Startup>,
        progress: Progress,
    ) {
        let prefix = match device {
            _ if self.config.remotes.is_empty() => String::new(),
            0 => "[local] ".to_string(),
            d => format!("[{}] ", self.config.remotes[d - 1].host),
        };
        let mut may_let_go = true;
        let mut may_take = true;
        'serve: while self.decode_ended().is_none() {
            while let Ok((s, worker)) = started.try_recv() {
                self.arrived(
                    device,
                    s,
                    &mut sides[s],
                    commons[s].as_ref(),
                    worker,
                    progress,
                );
            }
            if sides
                .iter()
                .all(|side| !side.loading && side.machine.is_none())
            {
                // Nothing to follow with.
                return;
            }
            let behind = sides.iter().any(|side| {
                side.grid
                    .zip(side.machine.as_ref())
                    .is_some_and(|(grid, machine)| {
                        machine
                            .streams
                            .iter()
                            .any(|stream| grid.ready(stream.held()) >= stream.blocks + BEHIND)
                    })
            });
            if behind && may_let_go {
                // The dub's last, while it has one: it only ever has the
                // time the original leaves.
                for s in [1, 0] {
                    let (Some(common), Some(machine)) = (&commons[s], &mut sides[s].machine) else {
                        continue;
                    };
                    let Some(stream) = machine.streams.pop() else {
                        continue;
                    };
                    common.made(stream.pass, None, progress);
                    progress(Event::message(format!(
                        "  {prefix}{} let go: {} does not keep up with the extraction on that \
                         many; it is separated after it",
                        stream.base,
                        name_of(self.config, device)
                    )));
                    machine.let_go(stream);
                    may_let_go = false;
                    continue 'serve;
                }
            }
            for s in 0..2 {
                let (Some(common), Some(grid)) = (&commons[s], sides[s].grid) else {
                    continue;
                };
                let Some(machine) = &mut sides[s].machine else {
                    continue;
                };
                let held: Vec<u64> = machine.streams.iter().map(Stream::held).collect();
                let Some(i) = (0..machine.streams.len())
                    .find(|&i| held[i] >= grid.ready_at(machine.streams[i].blocks))
                else {
                    continue;
                };
                #[cfg(test)]
                self.gate.pass();
                let stepped = match machine.split(i) {
                    Err(why) => Stepped::Out(why),
                    Ok((worker, stream)) => {
                        let inputs = stream.raw.clone();
                        match stream.step(worker, &inputs, held[i], false) {
                            Ok(_) => {
                                common.made(stream.pass, Some((device, stream.written)), progress);
                                Stepped::Made
                            }
                            // The decode failed, or the worker is gone: what
                            // it followed goes.
                            Err(why) if self.stops[s][device].raised() || !worker.is_alive() => {
                                Stepped::Out(why)
                            }
                            Err(why) => Stepped::Refused(why),
                        }
                    }
                };
                match stepped {
                    Stepped::Made => {
                        may_let_go = true;
                        may_take = true;
                    }
                    Stepped::Refused(why) => {
                        let stream = machine.streams.remove(i);
                        common.made(stream.pass, None, progress);
                        progress(Event::message(format!(
                            "  {prefix}{}: {why}; it is separated after the extraction",
                            stream.base
                        )));
                        machine.let_go(stream);
                    }
                    Stepped::Out(why) => {
                        let stopped = self.stops[s][device].raised();
                        out(&mut sides[s], common, device, (why, stopped), progress);
                    }
                }
                continue 'serve;
            }
            if may_take {
                for s in 0..2 {
                    let (Some(common), Some(grid), true) =
                        (&commons[s], sides[s].grid, sides[s].open)
                    else {
                        continue;
                    };
                    match self.intake(common, grid) {
                        Ok(Some(stream)) => {
                            if let Some(machine) = &mut sides[s].machine {
                                machine.streams.push(stream);
                            }
                            may_take = false;
                            continue 'serve;
                        }
                        Ok(None) => sides[s].open = false,
                        Err(why) => {
                            let stopped = self.stops[s][device].raised();
                            out(&mut sides[s], common, device, (why, stopped), progress);
                        }
                    }
                }
            }
            if sides.iter().any(|side| side.loading) {
                // Not following yet: the tests' extraction waits for it.
                self.pause(self.poll);
            } else {
                self.wait(device);
            }
        }
    }

    /// A stage's worker on `device`, loaded — or not: taken in.
    fn arrived(
        &self,
        device: usize,
        s: usize,
        side: &mut Side,
        common: Option<&Common>,
        started: Result<Option<Worker>>,
        progress: Progress,
    ) {
        side.loading = false;
        let mut worker = match started {
            Ok(Some(worker)) => worker,
            // Left out, which was said.
            Ok(None) => {
                side.open = false;
                return;
            }
            Err(why) => {
                side.open = false;
                side.failed = Some((why, self.stops[s][device].raised()));
                return;
            }
        };
        if let Some(common) = common
            && self.decode_ended().is_none()
        {
            match worker.grid(common.model.rate) {
                Ok(grid) => side.grid = Some(grid),
                // Loaded all the same: the separation gets it as it is.
                Err(why) => {
                    side.open = false;
                    let whose = the(common.stage);
                    progress(Event::message(if device == 0 && !common.named {
                        format!("{whose} is separated after the extraction: {why}")
                    } else {
                        format!(
                            "{}: follows nothing of {whose} during the extraction: {why}",
                            name_of(self.config, device)
                        )
                    }));
                }
            }
        }
        side.machine = Some(Machine {
            device,
            worker: Some(worker),
            streams: Vec::new(),
        });
    }

    /// `device`'s workers, loaded one after the other, the original's
    /// first, each sent to the thread that serves them as it is: `wanted`
    /// says which stages are followed. On a remote, the second stands
    /// beside the first ([`Seat`]); one that is busy or left out is said,
    /// and the other is not asked for either.
    fn start_all(
        &self,
        device: usize,
        wanted: [bool; 2],
        sender: Sender<Startup>,
        progress: Progress,
    ) {
        let stages = (0..2).filter(|&s| wanted[s]);
        if device == 0 {
            let mut lock: Option<Lock> = None;
            for s in stages {
                let started = self.start_here(s, lock.as_ref(), progress);
                if lock.is_none() {
                    lock = started.as_ref().ok().and_then(Worker::lock);
                }
                let _ = sender.send((s, started.map(Some)));
            }
            return;
        }
        let remote = &self.config.remotes[device - 1];
        let who = &remote.host;
        let first = if wanted[0] { 0 } else { 1 };
        let ssh = Ssh::new(who).stopped_by(self.stops[first][device].clone());
        let fit = match remote::probe(remote, &ssh) {
            Err(why) => {
                let _ = sender.send((first, Err(why)));
                false
            }
            Ok(probe) => match probe.unfit() {
                Some(why) => {
                    progress(Event::message(format!(
                        "{who}: left out of the extraction: {why}"
                    )));
                    false
                }
                None => {
                    progress(Event::message(format!(
                        "{who}: {}, following beside this machine",
                        probe.describe()
                    )));
                    true
                }
            },
        };
        let mut beside: Option<Seat> = None;
        let mut asked = fit;
        for s in stages {
            if !asked {
                let _ = sender.send((s, Ok(None)));
                continue;
            }
            let ssh = Ssh::new(who).stopped_by(self.stops[s][device].clone());
            let started = match Worker::start_remote(
                remote,
                &ssh,
                self.models[s],
                self.config.inference,
                progress,
                beside,
            ) {
                Ok(Started::Busy(why)) => {
                    progress(Event::message(format!(
                        "{who}: {why}; asked again for the separation"
                    )));
                    asked = false;
                    Ok(None)
                }
                Ok(Started::Ready(worker)) => {
                    self.following[s][device - 1].store(true, Ordering::SeqCst);
                    beside = worker.seat();
                    Ok(Some(*worker))
                }
                Err(why) => Err(why),
            };
            let _ = sender.send((s, started));
        }
    }

    /// Stage `s`'s worker on this machine, loaded; `beside` is the lock the
    /// other's holds.
    fn start_here(&self, s: usize, beside: Option<&Lock>, progress: Progress) -> Result<Worker> {
        let stop = self.stops[s][0].clone();
        match self.start {
            Some(start) => start(STAGES[s], stop),
            None => {
                let local = models::fetch(self.models[s], self.config, progress)?;
                Worker::start_under(
                    self.config,
                    self.models[s],
                    &local,
                    progress,
                    Some(stop),
                    beside,
                )
            }
        }
    }

    /// The next pass in line, for a machine that has time for one more,
    /// while it would start fewer than [`BEHIND`] blocks behind the
    /// extraction; `None` once the line is empty or that far behind.
    fn intake(&self, common: &Common, grid: Grid) -> Result<Option<Stream>> {
        let mut line = common.line.lock().unwrap_or_else(|p| p.into_inner());
        let Some(&pass) = line.front() else {
            return Ok(None);
        };
        let held = common.raw[pass]
            .iter()
            .map(|path| wav::frames(path))
            .min()
            .unwrap_or(0);
        if grid.ready(held) >= BEHIND {
            return Ok(None);
        }
        line.pop_front();
        drop(line);
        Stream::new(pass, common, grid).map(Some)
    }

    /// [`Follower::new`], this machine's workers started by `start` and the
    /// files looked at every `poll`: the tests'.
    #[cfg(test)]
    fn with_mock(
        config: &'a Config,
        job: &'a Job,
        model: &'a Model,
        channels: &'a [String],
        pairs: Pairs,
        start: Starts<'a>,
        poll: Duration,
    ) -> Self {
        Self {
            start: Some(start),
            poll,
            ..Self::new(config, job, model, model, channels, pairs)
        }
    }

    #[cfg(test)]
    fn waits(&self, device: usize) -> u64 {
        self.waits[device].load(Ordering::SeqCst)
    }
}

/// A stage's worker on `device` is gone, or the decode failed: what it
/// followed goes, and it follows nothing more. `failed`: why, and whether
/// its stop was raised.
fn out(side: &mut Side, common: &Common, device: usize, failed: (Error, bool), progress: Progress) {
    side.failed = Some(failed);
    side.machine = None;
    side.grid = None;
    side.open = false;
    common.lost(device, progress);
}

/// The units a pass serves, each with its side of the stems: `None` for an
/// element alone, folded.
fn serving(plan: &Plan, pass: usize) -> impl Iterator<Item = (usize, Option<usize>)> + '_ {
    let paired = plan.passes[pass].right.is_some();
    plan.takes.iter().enumerate().filter_map(move |(u, takes)| {
        takes
            .iter()
            .find(|take| take.pass == pass)
            .map(|take| (u, paired.then_some(take.side)))
    })
}

/// The decoder's files one of the original's passes reads while the
/// extraction runs: its left element's, then its right one's.
fn raw_of(plan: &Plan, pass: usize, units: &[Unit], master: &MasterSet) -> Result<Vec<PathBuf>> {
    let step = plan.passes[pass];
    let raw_dir = master.header.parent().unwrap_or(Path::new("."));
    std::iter::once(step.left)
        .chain(step.right)
        .map(|u| {
            let id: u32 = units[u]
                .name
                .parse()
                .map_err(|_| Error::msg(format!("{}: not an element", units[u].name)))?;
            let channel = master
                .channels
                .iter()
                .find(|c| c.id == id)
                .ok_or_else(|| Error::msg(format!("element {id}: not in the master set")))?;
            Ok(raw_dir.join(format!("{}{}.wav", extract::RAW, channel.index)))
        })
        .collect()
}

/// What the extraction's follower hands on: what was followed of each
/// separation, for its own step to finish.
#[derive(Default)]
pub struct Handed {
    pub original: Option<Followed>,
    pub dub: Option<Followed>,
}

/// What was followed of one separation during the extraction, and the
/// workers that followed it, machine by machine: for that separation to
/// finish. Dropped, its workers let their devices go and what they grew
/// goes from the disk.
pub struct Followed {
    stage: Stage,
    machines: Vec<Machine>,
    /// The passes it was cut into.
    plan: Plan,
    /// What it was followed on (see [`identity`], `dub_work`).
    identity: Vec<String>,
    dir: PathBuf,
}

impl Followed {
    /// The passes being followed.
    pub(super) fn passes(&self) -> Vec<usize> {
        self.machines
            .iter()
            .flat_map(|machine| &machine.streams)
            .map(|stream| stream.pass)
            .collect()
    }

    /// How much of them is made, in passes.
    pub(super) fn made(&self) -> f64 {
        self.machines
            .iter()
            .flat_map(|machine| &machine.streams)
            .map(Stream::made)
            .sum()
    }

    /// Itself, if what it followed is what the separation is about to make:
    /// the same elements, from the same waveforms, in the same passes. Its
    /// workers are kept either way.
    pub(super) fn fits(
        mut self,
        identity: &[String],
        plan: &Plan,
        progress: Progress,
    ) -> Option<Self> {
        if !self.passes().is_empty() && (identity != self.identity || *plan != self.plan) {
            progress(Event::message(match self.stage {
                Stage::Original => {
                    "the master set is not the one the extraction was followed on: what was \
                     separated during it is dropped"
                }
                Stage::Dub => {
                    "the dub's channels are not the ones the extraction was followed on: what \
                     was separated of them during it is dropped"
                }
            }));
            for machine in &mut self.machines {
                for stream in std::mem::take(&mut machine.streams) {
                    machine.let_go(stream);
                }
            }
        }
        Some(self)
    }

    /// How many machines followed some of it.
    pub(super) fn following_on(&self) -> usize {
        self.machines
            .iter()
            .filter(|machine| !machine.streams.is_empty())
            .count()
    }

    /// What `device` followed, and its worker, taken out.
    pub(super) fn machine(&mut self, device: usize) -> Option<Machine> {
        let at = self
            .machines
            .iter()
            .position(|machine| machine.device == device)?;
        Some(self.machines.swap_remove(at))
    }

    /// What its workers hold on each machine while they wait for their
    /// step: for another step's workers to stand beside.
    pub fn beside(&self) -> Beside {
        let devices = self
            .machines
            .iter()
            .map(|m| m.device + 1)
            .max()
            .unwrap_or(0);
        let mut beside = Beside {
            lock: None,
            seats: vec![None; devices],
        };
        for machine in &self.machines {
            let Some(worker) = &machine.worker else {
                continue;
            };
            match machine.device {
                0 => beside.lock = worker.lock(),
                d => beside.seats[d] = worker.seat(),
            }
        }
        beside
    }
}

impl Drop for Followed {
    fn drop(&mut self) {
        self.machines.clear();
        let _ = discard(&self.dir);
    }
}

/// What one machine followed of a separation, and its worker. Dropped, the
/// worker lets the device go and what the streams grew goes.
pub(super) struct Machine {
    /// 0: this one; `d`: the remote `remotes[d - 1]`.
    device: usize,
    worker: Option<Worker>,
    streams: Vec<Stream>,
}

impl Machine {
    fn split(&mut self, stream: usize) -> Result<(&mut Worker, &mut Stream)> {
        let worker = self
            .worker
            .as_mut()
            .ok_or_else(|| Error::msg("the worker that followed it is gone"))?;
        Ok((worker, &mut self.streams[stream]))
    }

    /// Stop following `stream`: the worker forgets it, and what it grew goes.
    fn let_go(&mut self, stream: Stream) {
        if let Some(worker) = &mut self.worker
            && worker.is_alive()
        {
            let _ = worker.forget(&stream.base);
        }
        stream.remove();
    }

    pub(super) fn take_worker(&mut self) -> Option<Worker> {
        self.worker.take()
    }

    /// How many passes it followed.
    pub(super) fn followed(&self) -> usize {
        self.streams.len()
    }

    pub(super) fn streams(&mut self) -> &mut [Stream] {
        &mut self.streams
    }
}

impl Drop for Machine {
    fn drop(&mut self) {
        // The device first, then the files: closed before they go, which
        // Windows asks.
        drop(self.worker.take());
        for stream in self.streams.drain(..) {
            stream.remove();
        }
    }
}

/// One pass followed: an element alone, or two side by side.
pub(super) struct Stream {
    pass: usize,
    /// What the worker calls it: the pass's name.
    base: String,
    /// The units it reads, in order: left, then right.
    sides: Vec<usize>,
    /// The decoder's files it reads while the extraction runs.
    raw: Vec<PathBuf>,
    /// Where the worker leaves a block's stems.
    stems: PathBuf,
    grid: Grid,
    rate: u32,
    /// Blocks made.
    blocks: u64,
    /// Frames of the outputs made, from the first.
    written: u64,
    /// The element's frames once the decoder is done; 0 until then.
    length: u64,
    outputs: Vec<Grown>,
}

/// One output of a followed unit, growing.
struct Grown {
    unit: usize,
    side: Option<usize>,
    stem: Stem,
    form: Form,
    /// The output, once it is whole.
    path: PathBuf,
    /// Where it grows meanwhile, in float.
    growing: PathBuf,
    appender: Option<wav::Appender>,
}

impl Stream {
    /// `pass` of `common`'s separation, followed from its first block.
    fn new(pass: usize, common: &Common, grid: Grid) -> Result<Self> {
        let step = common.plan.passes[pass];
        let sides: Vec<usize> = std::iter::once(step.left).chain(step.right).collect();
        let mut outputs = Vec::new();
        for (u, side) in serving(&common.plan, pass) {
            for output in &common.units[u].outputs {
                let growing = common.dir.join(output.path.file_name().unwrap_or_default());
                outputs.push(Grown {
                    unit: u,
                    side,
                    stem: output.stem,
                    form: output.form,
                    path: output.path.clone(),
                    appender: Some(wav::Appender::create(&growing, super::RATE)?),
                    growing,
                });
            }
        }
        Ok(Self {
            pass,
            base: common.names[pass].clone(),
            sides,
            raw: common.raw[pass].clone(),
            stems: common.dir.join("stems"),
            grid,
            rate: common.model.rate,
            blocks: 0,
            written: 0,
            length: 0,
            outputs,
        })
    }

    pub(super) fn pass(&self) -> usize {
        self.pass
    }

    /// How much of it is made, from 0 to 1.
    pub(super) fn made(&self) -> f64 {
        self.written as f64 / self.length.max(1) as f64
    }

    /// « 9 of its 12 blocks made ».
    pub(super) fn describe(&self) -> String {
        format!(
            "{} of its {} blocks made",
            self.blocks,
            self.grid.blocks(self.length)
        )
    }

    /// What every one of its inputs holds now.
    fn held(&self) -> u64 {
        self.raw
            .iter()
            .map(|path| wav::frames(path))
            .min()
            .unwrap_or(0)
    }

    /// The next block, its inputs holding `held` frames (`whole`: and no
    /// more), added to the outputs. `true` once the last one is in.
    fn step(
        &mut self,
        worker: &mut Worker,
        inputs: &[PathBuf],
        held: u64,
        whole: bool,
    ) -> Result<bool> {
        let block = worker.block(
            &Ask {
                base: &self.base,
                out_dir: &self.stems,
                rate: self.rate,
                held,
                whole,
                inputs,
            },
            |_| {},
        )?;
        if block.from != self.written {
            return Err(Error::msg(format!(
                "{}: a block from frame {}, where {} are made",
                self.base, block.from, self.written
            )));
        }
        if block.to > block.from {
            let voice = self.stems.join(format!("{}{VOICE_SUFFIX}", self.base));
            let rest = self.stems.join(format!("{}{REST_SUFFIX}", self.base));
            for grown in &mut self.outputs {
                let stem = match grown.stem {
                    Stem::Voice => &voice,
                    Stem::Rest => &rest,
                };
                let pick = match grown.side {
                    None => wav::Pick::Fold(stem),
                    Some(side) => wav::Pick::Channel(stem, side),
                };
                let appender = grown
                    .appender
                    .as_mut()
                    .ok_or_else(|| Error::msg(format!("{}: already finished", self.base)))?;
                let added = appender.append(pick)?;
                if added != block.to - block.from {
                    return Err(Error::msg(format!(
                        "{}: the stems of frames {} to {} hold {added}",
                        stem.display(),
                        block.from,
                        block.to
                    )));
                }
            }
            let _ = fs::remove_file(&voice);
            let _ = fs::remove_file(&rest);
            self.written = block.to;
            self.blocks += 1;
        }
        Ok(block.last)
    }

    /// What is left of it, from the extraction's finished waveforms, and its
    /// outputs written whole; `made` is told each block's share of the pass.
    /// `claim` is asked before anything is written: `false` (another machine
    /// made the pass first), nothing is, and `Ok(false)` says so.
    pub(super) fn finish(
        &mut self,
        worker: &mut Worker,
        units: &[Unit],
        frames: &[u64],
        mastered_at: u32,
        mut made: impl FnMut(f64),
        claim: impl FnOnce() -> bool,
    ) -> Result<bool> {
        let inputs: Vec<PathBuf> = self
            .sides
            .iter()
            .map(|&u| units[u].source.clone())
            .collect();
        let length = self.sides.iter().map(|&u| frames[u]).min().unwrap_or(0);
        if length != self.length {
            return Err(Error::msg(format!(
                "{} frames, where it was followed to {}",
                length, self.length
            )));
        }
        loop {
            let before = self.written;
            let last = self.step(worker, &inputs, length, true)?;
            made((self.written - before) as f64 / length.max(1) as f64);
            if last {
                break;
            }
            if self.written == before {
                return Err(Error::msg("the worker made nothing of a whole element"));
            }
        }
        if !claim() {
            return Ok(false);
        }
        for grown in &mut self.outputs {
            grown.settle(frames[grown.unit], mastered_at)?;
        }
        Ok(true)
    }

    /// What it grew goes.
    fn remove(self) {
        for grown in self.outputs {
            drop(grown.appender);
            let _ = fs::remove_file(&grown.growing);
        }
        for suffix in [VOICE_SUFFIX, REST_SUFFIX] {
            let _ = fs::remove_file(self.stems.join(format!("{}{suffix}", self.base)));
        }
    }
}

impl Grown {
    /// Written whole under its name: `frames` long, as the separation writes
    /// an output, at the depth the original was mastered at for the element
    /// without its dialogue.
    fn settle(&mut self, frames: u64, mastered_at: u32) -> Result<()> {
        let mut appender = self
            .appender
            .take()
            .ok_or_else(|| Error::msg(format!("{}: already settled", self.path.display())))?;
        if appender.frames() > frames {
            return Err(Error::msg(format!(
                "{}: {} frames grown for an element of {frames}",
                self.growing.display(),
                appender.frames()
            )));
        }
        appender.pad_to(frames)?;
        appender.finish()?;
        if self.form == Form::Float {
            return fs::rename(&self.growing, &self.path).map_err(|e| Error::io(&self.path, e));
        }
        let part = self.path.with_extension("wav.part");
        wav::copy_mono(
            &self.growing,
            &part,
            frames,
            super::RATE,
            1.0,
            self.form.sample(mastered_at),
        )?;
        fs::rename(&part, &self.path).map_err(|e| Error::io(&self.path, e))?;
        let _ = fs::remove_file(&self.growing);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! A run without a GPU: a mock worker — this test binary, started again
    //! as a child that speaks the worker's protocol — and an extraction that
    //! grows the decoder's files as harletty does.

    use super::*;
    use crate::job::{Job, Step};
    use crate::models::Catalogue;
    use crate::progress::quiet;
    use std::collections::HashMap;
    use std::io::{BufRead, BufWriter, Read, Seek, SeekFrom, Write};
    use std::process::Command;
    use std::sync::Arc;

    /// Set in the child's environment: it is the mock, not the tests.
    const MOCK: &str = "OMNIFROG_MOCK_WORKER";
    /// A machine's count of waits once it is done following.
    pub(super) const DONE: u64 = u64::MAX / 2;
    /// Milliseconds the mock spends on a block, as a GPU would.
    const MOCK_BLOCK_MS: &str = "OMNIFROG_MOCK_BLOCK_MS";
    /// `<base>:<block>`: the block the mock fails on.
    const MOCK_FAIL: &str = "OMNIFROG_MOCK_FAIL";
    /// `<base>:<block>`: the block the mock dies on, as a GPU reset kills a
    /// worker.
    const MOCK_DIE: &str = "OMNIFROG_MOCK_DIE";
    /// The mock as a remote's `remote-worker`: the directory it says its
    /// files go to.
    const MOCK_DIR: &str = "OMNIFROG_MOCK_DIR";
    /// Set, the mock as a remote's `remote-worker` says its GPU is busy.
    const MOCK_BUSY: &str = "OMNIFROG_MOCK_BUSY";
    /// `<base>:<block>`, `*` for any: the block the mock hangs on, as a GPU
    /// that stalls.
    const MOCK_HANG: &str = "OMNIFROG_MOCK_HANG";
    /// Set, the mock does not follow: it refuses `GRID`, as MSST or a CPU
    /// does.
    const MOCK_WHOLE_ONLY: &str = "OMNIFROG_MOCK_WHOLE_ONLY";
    /// As [`MOCK_DIE`], for the mock on a remote's second seat alone.
    const MOCK_DIE_BESIDE: &str = "OMNIFROG_MOCK_DIE_BESIDE";
    /// The mock's grid, in frames: a block's step, the fade, the lookahead.
    const STEP: u64 = 8_000;
    const FADE: u64 = 1_000;
    const AHEAD: u64 = FADE + 1;

    /// Not a test: the mock worker, when this binary is started as one.
    #[test]
    fn mock_worker() {
        if std::env::var_os(MOCK).is_some() {
            std::process::exit(serve());
        }
    }

    type Stems = [Vec<Vec<f32>>; 2];

    /// The mock's model: stereo stems — a voice and the rest — that depend
    /// on the block they are made in, as a model's do near its edges, so a
    /// block cut elsewhere or a fade done otherwise shows.
    fn separate_block(x: &[Vec<f32>]) -> Stems {
        let k = (x[0].len() % 997) as f32 * 1e-3;
        let sides = if x.len() == 1 {
            [&x[0], &x[0]]
        } else {
            [&x[0], &x[1]]
        };
        let make = |sign: f32| -> Vec<Vec<f32>> {
            sides
                .iter()
                .map(|c| c.iter().map(|s| 0.5 * s + sign * k).collect())
                .collect()
        };
        [make(1.0), make(-1.0)]
    }

    /// The worker's plan (`plan_blocs`) and fade (`fond_bloc`), in Rust.
    fn plan(n: u64) -> Vec<(u64, u64)> {
        let k = n.saturating_sub(AHEAD) / STEP;
        let mut blocks: Vec<(u64, u64)> =
            (0..k).map(|i| (i * STEP, (i + 1) * STEP + FADE)).collect();
        blocks.push((k * STEP, n));
        blocks
    }

    /// `stems` faded into `tail`; what is written now, and the next tail.
    fn fade(mut stems: Stems, tail: Option<Stems>, last: bool) -> (Stems, Option<Stems>) {
        let f = FADE as usize;
        if let Some(tail) = tail {
            for (stem, old) in stems.iter_mut().zip(&tail) {
                for (channel, old) in stem.iter_mut().zip(old) {
                    for i in 0..f {
                        let w = i as f32 / (f - 1) as f32;
                        channel[i] = old[i] * (1.0 - w) + channel[i] * w;
                    }
                }
            }
        }
        if last {
            return (stems, None);
        }
        let keep = stems[0][0].len() - f;
        let split = |stem: &mut Vec<Vec<f32>>| -> Vec<Vec<f32>> {
            stem.iter_mut().map(|c| c.split_off(keep)).collect()
        };
        let next = [split(&mut stems[0]), split(&mut stems[1])];
        (stems, Some(next))
    }

    /// Frames [from, to) of each input, read past a header that may still
    /// say nothing.
    fn read(inputs: &[&str], from: u64, to: u64) -> std::result::Result<Vec<Vec<f32>>, String> {
        let mut channels = Vec::new();
        for input in inputs {
            let path = Path::new(input);
            let format = wav::read_header(path).map_err(|e| e.to_string())?;
            let size = format.bytes_per_sample() as usize;
            let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
            for c in 0..format.channels as usize {
                let mut samples = Vec::with_capacity((to - from) as usize);
                let frame = format.bytes_per_frame();
                let mut bytes = vec![0u8; ((to - from) * frame) as usize];
                file.seek(SeekFrom::Start(format.data_offset + from * frame))
                    .map_err(|e| e.to_string())?;
                file.read_exact(&mut bytes)
                    .map_err(|e| format!("{input}: {e}"))?;
                for f in bytes.chunks_exact(frame as usize) {
                    samples.push(wav::sample(&f[c * size..], format.bits, format.float));
                }
                channels.push(samples);
            }
        }
        Ok(channels)
    }

    fn write_stems(out: &str, base: &str, rate: u32, stems: &Stems) {
        for (stem, suffix) in stems.iter().zip([VOICE_SUFFIX, REST_SUFFIX]) {
            let path = Path::new(out).join(format!("{base}{suffix}"));
            let mut w = BufWriter::new(fs::File::create(path).unwrap());
            let format = wav::Format {
                channels: 2,
                sample_rate: rate,
                bits: 32,
                float: true,
                frames: stem[0].len() as u64,
                data_offset: 0,
            };
            wav::write_header(&mut w, format).unwrap();
            for (l, r) in stem[0].iter().zip(&stem[1]) {
                w.write_all(&l.to_le_bytes()).unwrap();
                w.write_all(&r.to_le_bytes()).unwrap();
            }
            w.flush().unwrap();
        }
    }

    fn concat(into: &mut Option<Stems>, piece: Stems) {
        match into {
            None => *into = Some(piece),
            Some(stems) => {
                for (stem, add) in stems.iter_mut().zip(piece) {
                    for (channel, add) in stem.iter_mut().zip(add) {
                        channel.extend(add);
                    }
                }
            }
        }
    }

    /// The worker's protocol, on this process's standard streams.
    fn serve() -> i32 {
        let slow = std::env::var(MOCK_BLOCK_MS)
            .ok()
            .and_then(|v| v.parse().ok())
            .map(Duration::from_millis);
        let args: Vec<String> = std::env::args().collect();
        let fail = std::env::var(MOCK_FAIL).ok();
        let die = match args.iter().any(|arg| arg == "--beside") {
            true => std::env::var(MOCK_DIE_BESIDE).ok(),
            false => std::env::var(MOCK_DIE).ok(),
        };
        let hang = std::env::var(MOCK_HANG).ok();
        let whole_only = std::env::var_os(MOCK_WHOLE_ONLY).is_some();
        // A list of `<base>:<block>`, `*` for any base.
        let hits = |spec: &Option<String>, base: &str, k: u64| {
            spec.as_deref().is_some_and(|spec| {
                spec.split(',')
                    .any(|one| one == format!("{base}:{k}") || one == format!("*:{k}"))
            })
        };
        let say = |line: &str| {
            let mut out = std::io::stdout().lock();
            let _ = writeln!(out, "{line}");
            let _ = out.flush();
        };
        // The test harness has printed « test …mock_worker ... » with no end
        // of line: this one ends it, so that what follows is on lines of its
        // own.
        say("");
        // Started as a remote's omnifrog: the two commands this side runs
        // there, after the harness's own arguments.
        if args.iter().any(|arg| arg == "remote-check") {
            say(&format!(
                "OMNIFROG {} {}",
                env!("CARGO_PKG_VERSION"),
                remote::protocol()
            ));
            say("DEVICE Mock GPU");
            say("VRAM 0.00 24.00");
            return 0;
        }
        if args.iter().any(|arg| arg == "remote-worker") {
            if std::env::var_os(MOCK_BUSY).is_some() {
                say("BUSY another run has its GPU");
                return 0;
            }
            let mut dir = PathBuf::from(std::env::var_os(MOCK_DIR).expect("the mock's directory"));
            // The second seat has files of its own.
            if args.iter().any(|arg| arg == "--beside") {
                dir.as_mut_os_string().push("-beside");
            }
            for sub in ["in", "out"] {
                fs::create_dir_all(dir.join(sub)).unwrap();
            }
            say(&format!("DIR {}", dir.display()));
        }
        say("PRET");
        let mut streams: HashMap<String, (u64, Option<Stems>)> = HashMap::new();
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            let fields: Vec<&str> = line.split('\t').collect();
            let answer = match fields[0] {
                "QUIT" => break,
                "GRID" if whole_only => Err("RuntimeError: the mock separates whole only".into()),
                "GRID" => {
                    say(&format!("GRID {STEP} {AHEAD}"));
                    Ok(())
                }
                "DROP" => {
                    streams.remove(fields[1]);
                    Ok(())
                }
                "SEP" => (|| {
                    for input in &fields[2..] {
                        let format =
                            wav::read_header(Path::new(input)).map_err(|e| e.to_string())?;
                        let mut made = None;
                        let mut tail = None;
                        let blocks = plan(format.frames);
                        for (b, &(from, to)) in blocks.iter().enumerate() {
                            let stems = separate_block(&read(&[input], from, to)?);
                            let (piece, next) = fade(stems, tail, b == blocks.len() - 1);
                            tail = next;
                            concat(&mut made, piece);
                        }
                        let base = Path::new(input).file_stem().unwrap().to_string_lossy();
                        write_stems(fields[1], &base, format.sample_rate, &made.unwrap());
                        say(&format!("OK {base}"));
                    }
                    Ok(())
                })(),
                "BLK" => (|| {
                    let (out, base) = (fields[1], fields[2]);
                    let rate: u32 = fields[3].parse().map_err(|_| "rate")?;
                    let held: u64 = fields[4].parse().map_err(|_| "frames")?;
                    let whole = fields[5] == "1";
                    let (k, tail) = streams.remove(base).unwrap_or((0, None));
                    if hits(&fail, base, k) {
                        return Err("RuntimeError: the mock fails here".to_string());
                    }
                    if hits(&die, base, k) {
                        std::process::exit(3);
                    }
                    if hits(&hang, base, k) {
                        std::thread::sleep(Duration::from_secs(600));
                    }
                    let from = k * STEP;
                    let ready = held >= from + STEP + AHEAD;
                    if !ready && !whole {
                        say(&format!("BLK {base} {from} {from}"));
                        streams.insert(base.to_string(), (k, tail));
                        return Ok(());
                    }
                    let to = if ready { from + STEP + FADE } else { held };
                    if let Some(slow) = slow {
                        std::thread::sleep(slow);
                    }
                    let stems = separate_block(&read(&fields[6..], from, to)?);
                    let (piece, next) = fade(stems, tail, !ready);
                    write_stems(out, base, rate, &piece);
                    let made = piece[0][0].len() as u64;
                    say(&format!("BLK {base} {from} {}", from + made));
                    if ready {
                        streams.insert(base.to_string(), (k + 1, next));
                    } else {
                        say(&format!("OK {base}"));
                    }
                    Ok(())
                })(),
                _ => Err(format!("invalid: {line}")),
            };
            match answer {
                Ok(()) => say("FIN"),
                Err(why) => say(&format!("ERR {why}")),
            }
        }
        0
    }

    /// The mock, started for `env`.
    fn mock(env: &[(&str, String)]) -> Command {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "separate::follow::tests::mock_worker",
                "--exact",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(MOCK, "1");
        for (key, value) in env {
            command.env(key, value);
        }
        command
    }

    /// A master set: (decoder's rank, identifier, bed name or `None` for an
    /// object), each element's samples a function of its identifier with
    /// four dead bits — the original mastered at 20; and the dub's channels,
    /// each a function of its rank.
    struct Film {
        elements: Vec<(usize, u32, Option<&'static str>)>,
        dub: Vec<&'static str>,
        frames: u64,
    }

    /// A 5.1 dub, as the extraction names its channels.
    const DUB: [&str; 6] = ["L", "R", "C", "LFE", "Lss", "Rss"];

    impl Film {
        fn header(&self) -> String {
            let mut text = String::from(
                "version: 0.5.1\npresentations:\n  - type: home\n    metadata: vo.atmos.metadata\n    \
                 audio: vo.atmos.audio\n    fps: 24\n    sourceCodec: test\n    bedInstances:\n      \
                 - channels:\n",
            );
            for (_, id, name) in &self.elements {
                if let Some(name) = name {
                    text.push_str(&format!(
                        "          - channel: {name}\n            ID: {id}\n"
                    ));
                }
            }
            text.push_str("    objects:\n");
            for (_, id, name) in &self.elements {
                if name.is_none() {
                    text.push_str(&format!("      - ID: {id}\n"));
                }
            }
            text
        }

        fn metadata(&self) -> String {
            let mut text = String::from("sampleRate: 48000\nevents:\n");
            for (i, (_, id, name)) in self.elements.iter().enumerate() {
                if name.is_none() {
                    let x = (i as f64 / 3.0).sin();
                    text.push_str(&format!(
                        "  - ID: {id}\n    samplePos: 0\n    pos: [{x:.3}, 0.5, 0]\n"
                    ));
                }
            }
            text
        }

        fn sample(id: u32, i: u64) -> [u8; 3] {
            let v = ((i * (u64::from(id) + 3) * 7919) % (1 << 20)) as i32 - (1 << 19);
            let v = v << 4;
            let b = v.to_le_bytes();
            [b[0], b[1], b[2]]
        }

        fn bytes(id: u32, from: u64, to: u64) -> Vec<u8> {
            (from..to).flat_map(|i| Self::sample(id, i)).collect()
        }

        fn wav_header(frames: u64) -> Vec<u8> {
            let mut bytes = Vec::new();
            wav::write_header(
                &mut bytes,
                wav::Format {
                    channels: 1,
                    sample_rate: 48_000,
                    bits: 24,
                    float: false,
                    frames,
                    data_offset: 0,
                },
            )
            .unwrap();
            bytes
        }

        /// What ffmpeg writes as a WAV's header while it streams to it:
        /// both sizes at their maximum, until it ends.
        fn streamed_header() -> Vec<u8> {
            let mut bytes = Self::wav_header(0);
            bytes[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
            let data = bytes.windows(4).position(|w| w == b"data").unwrap();
            bytes[data + 4..data + 8].copy_from_slice(&u32::MAX.to_le_bytes());
            bytes
        }

        /// The dub's channel `name`, and what its samples are drawn from.
        fn dub_paths(&self, dir: &Path) -> Vec<(PathBuf, u32)> {
            self.dub
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    (
                        dir.join(format!("{}{name}.wav", extract::DUB)),
                        40 + i as u32,
                    )
                })
                .collect()
        }

        /// The dub's channels' names.
        fn channels(&self) -> Vec<String> {
            self.dub.iter().map(|name| name.to_string()).collect()
        }

        /// A 5.1 dub beside it.
        fn dubbed(mut self) -> Self {
            self.dub = DUB.to_vec();
            self
        }

        /// The extraction done: `vo_<id>.wav` whole, the dub's
        /// `vf_<name>.wav`, and the header.
        fn extracted(&self, dir: &Path) {
            fs::create_dir_all(dir).unwrap();
            for (_, id, _) in &self.elements {
                let mut bytes = Self::wav_header(self.frames);
                bytes.extend(Self::bytes(*id, 0, self.frames));
                fs::write(dir.join(format!("{}{id}.wav", extract::ORIGINAL)), bytes).unwrap();
            }
            for (path, seed) in self.dub_paths(dir) {
                let mut bytes = Self::wav_header(self.frames);
                bytes.extend(Self::bytes(seed, 0, self.frames));
                fs::write(path, bytes).unwrap();
            }
            fs::write(dir.join("vo.atmos"), self.header()).unwrap();
            fs::write(dir.join("vo.atmos.metadata"), self.metadata()).unwrap();
        }

        /// The extraction under way: the decoder's `canal_<n>.wav` grow
        /// `chunk` frames at a time, sizes left at 0 until the end — as
        /// harletty writes them — and the header after the first chunk.
        /// `until` cuts it short; `begun` is waited for after the first
        /// chunk, so a slow machine starting the mock does not find the
        /// extraction over, and `in_step` after every chunk.
        fn grow(
            &self,
            dir: &Path,
            chunk: u64,
            pause: Duration,
            until: Option<u64>,
            begun: &dyn Fn() -> bool,
            in_step: Option<&dyn Fn()>,
        ) {
            fs::create_dir_all(dir).unwrap();
            let raw = |n: usize| dir.join(format!("{}{n}.wav", extract::RAW));
            let mut files: Vec<fs::File> = self
                .elements
                .iter()
                .map(|(n, ..)| {
                    let mut file = fs::File::create(raw(*n)).unwrap();
                    file.write_all(&Self::wav_header(0)).unwrap();
                    file
                })
                .collect();
            // ffmpeg's: there before the decoder is fed, their sizes at the
            // most until the end.
            let mut dubbed: Vec<(fs::File, u32)> = self
                .dub_paths(dir)
                .into_iter()
                .map(|(path, seed)| {
                    let mut file = fs::File::create(path).unwrap();
                    file.write_all(&Self::streamed_header()).unwrap();
                    (file, seed)
                })
                .collect();
            let end = until.unwrap_or(self.frames);
            let mut at = 0;
            while at < end {
                let to = (at + chunk).min(end);
                for (file, (_, id, _)) in files.iter_mut().zip(&self.elements) {
                    file.write_all(&Self::bytes(*id, at, to)).unwrap();
                    file.flush().unwrap();
                }
                for (file, seed) in &mut dubbed {
                    file.write_all(&Self::bytes(*seed, at, to)).unwrap();
                    file.flush().unwrap();
                }
                if at == 0 {
                    fs::write(dir.join("vo.atmos"), self.header()).unwrap();
                    fs::write(dir.join("vo.atmos.metadata"), self.metadata()).unwrap();
                    let deadline = std::time::Instant::now() + Duration::from_secs(60);
                    while !begun() && std::time::Instant::now() < deadline {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }
                if let Some(in_step) = in_step {
                    in_step();
                }
                at = to;
                std::thread::sleep(pause);
            }
            if until.is_none() {
                // harletty's finish and ffmpeg's: the sizes, once everything
                // is written.
                for file in files
                    .iter_mut()
                    .chain(dubbed.iter_mut().map(|(file, _)| file))
                {
                    file.seek(SeekFrom::Start(0)).unwrap();
                    file.write_all(&Self::wav_header(self.frames)).unwrap();
                }
            }
        }

        /// What the extraction does once the decoder is done.
        fn rename(&self, dir: &Path) {
            for (n, id, _) in &self.elements {
                fs::rename(
                    dir.join(format!("{}{n}.wav", extract::RAW)),
                    dir.join(format!("{}{id}.wav", extract::ORIGINAL)),
                )
                .unwrap();
            }
        }

        fn objects(count: u32, frames: u64) -> Self {
            let mut elements = vec![(0, 3, Some("LFE"))];
            elements.extend((0..count).map(|i| (i as usize + 1, 10 + i, None)));
            Self {
                elements,
                dub: Vec::new(),
                frames,
            }
        }
    }

    /// A separation's outputs: each file's name and bytes.
    type Outputs = Vec<(String, Vec<u8>)>;

    struct Bench {
        root: PathBuf,
        config: crate::config::Config,
        model: Model,
    }

    impl Bench {
        fn new(name: &str) -> Self {
            let root =
                std::env::temp_dir().join(format!("omnifrog-follow-{name}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&root);
            fs::create_dir_all(&root).unwrap();
            Self {
                config: crate::config::Config {
                    work_dir: Some(root.join("work")),
                    ..Default::default()
                },
                model: Catalogue::embedded().unwrap().get("uhav").unwrap().clone(),
                root,
            }
        }

        fn job(&self, name: &str, pairs: Pairs) -> Job {
            let mut manifest = crate::job::tests::manifest();
            manifest.pairs = pairs;
            Job {
                dir: self.root.join(name),
                manifest,
            }
        }

        /// The separation of the original, the mock as this machine's worker.
        fn separate(
            &self,
            job: &Job,
            followed: Option<Followed>,
            beside: &Beside,
            env: &[(&str, String)],
            progress: Progress,
        ) -> super::super::Separated {
            let master = MasterSet::read(&job.extract_dir().join("vo.atmos")).unwrap();
            let start = |stop: Stop| Worker::start_command(mock(env), Some(stop), &quiet);
            super::super::original_with(
                &self.config,
                job,
                &self.model,
                &master,
                job.manifest.pairs,
                followed,
                beside,
                Some(&start),
                progress,
            )
            .unwrap()
        }

        /// The separation of the dub's `channels`, the mock as this
        /// machine's worker.
        fn separate_dub(
            &self,
            job: &Job,
            channels: &[String],
            followed: Option<Followed>,
            env: &[(&str, String)],
            progress: Progress,
        ) -> super::super::Separated {
            let start = |stop: Stop| Worker::start_command(mock(env), Some(stop), &quiet);
            super::super::dub_with(
                &self.config,
                job,
                &self.model,
                channels,
                job.manifest.pairs,
                followed,
                Some(&start),
                progress,
            )
            .unwrap()
            .0
        }

        /// Every output of the original's separation, by name.
        fn outputs(job: &Job) -> Outputs {
            let mut found: Outputs = fs::read_dir(job.separate_dir())
                .unwrap()
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_file())
                .map(|p| {
                    (
                        p.file_name().unwrap().to_string_lossy().into_owned(),
                        fs::read(&p).unwrap(),
                    )
                })
                .collect();
            found.sort();
            found
        }

        /// The reference: the film extracted, then separated whole — its
        /// dub too, when it has one.
        fn whole(&self, film: &Film, pairs: Pairs) -> Outputs {
            let job = self.job("whole", pairs);
            film.extracted(&job.extract_dir());
            self.separate(&job, None, &Beside::default(), &[], &quiet);
            if !film.dub.is_empty() {
                self.separate_dub(&job, &film.channels(), None, &[], &quiet);
            }
            Self::outputs(&job)
        }
    }

    impl Drop for Bench {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    /// What a run said, kept.
    #[derive(Default)]
    struct Said(Mutex<Vec<String>>, Mutex<[Ratios; 2]>);

    /// How far a separation said it was: during the extraction, then in its
    /// own step.
    #[derive(Default)]
    struct Ratios {
        early: Vec<f64>,
        own: Vec<f64>,
        extracted: bool,
    }

    impl Said {
        fn progress(self: &Arc<Self>) -> impl Fn(Event) + Sync + use<> {
            let said = Arc::clone(self);
            move |event| match event {
                Event::Message { text } => said.0.lock().unwrap().push(text),
                Event::Fraction {
                    what,
                    done,
                    total: Some(total),
                    ..
                } => {
                    let Some(s) = STAGES.iter().position(|stage| what == stage.label()) else {
                        return;
                    };
                    let ratios = &mut said.1.lock().unwrap()[s];
                    let ratio = done as f64 / total as f64;
                    if ratios.extracted {
                        ratios.own.push(ratio);
                    } else {
                        ratios.early.push(ratio);
                    }
                }
                _ => {}
            }
        }

        /// What the original's follower said, and what the step said after
        /// it.
        fn ratios(&self) -> (Vec<f64>, Vec<f64>) {
            self.ratios_of(Stage::Original)
        }

        fn ratios_of(&self, stage: Stage) -> (Vec<f64>, Vec<f64>) {
            let s = STAGES.iter().position(|s| *s == stage).unwrap();
            let ratios = &self.1.lock().unwrap()[s];
            (ratios.early.clone(), ratios.own.clone())
        }

        fn extracted(&self) {
            for ratios in self.1.lock().unwrap().iter_mut() {
                ratios.extracted = true;
            }
        }

        fn any(&self, needle: &str) -> bool {
            self.0
                .lock()
                .unwrap()
                .iter()
                .any(|line| line.contains(needle))
        }

        fn lines(&self) -> String {
            self.0.lock().unwrap().join("\n")
        }
    }

    /// The tests' hold on how many blocks the machines make: closed, each
    /// block waits for one the test granted. A GPU that keeps up with only
    /// so much of the extraction, in step with it, wherever the tests run.
    #[derive(Default)]
    pub(super) struct Gate {
        state: Mutex<GateState>,
        changed: Condvar,
    }

    #[derive(Default)]
    struct GateState {
        closed: bool,
        granted: u64,
        used: u64,
        /// A block waits for one.
        waiting: bool,
    }

    impl Gate {
        fn state(&self) -> MutexGuard<'_, GateState> {
            self.state.lock().unwrap_or_else(|p| p.into_inner())
        }

        /// A block about to be asked for: through when the gate is open, or
        /// once one is granted.
        pub(super) fn pass(&self) {
            let mut state = self.state();
            while state.closed && state.used >= state.granted {
                state.waiting = true;
                state = self.changed.wait(state).unwrap_or_else(|p| p.into_inner());
            }
            state.waiting = false;
            state.used += 1;
        }

        fn close(&self) {
            self.state().closed = true;
        }

        /// `blocks` more, from now.
        fn grant(&self, blocks: u64) {
            let mut state = self.state();
            state.granted = state.used + blocks;
            self.changed.notify_all();
        }

        /// What was not used of the last grant is lost, as GPU time is: the
        /// next chunk of film finds none.
        fn revoke(&self) {
            let mut state = self.state();
            state.granted = state.used;
        }

        fn open(&self) {
            self.state().closed = false;
            self.changed.notify_all();
        }

        /// Every block granted is made, and the next one waits.
        fn spent(&self) -> bool {
            let state = self.state();
            state.closed && state.waiting && state.used >= state.granted
        }
    }

    /// What [`follow_with`] ran into: per stage what was made of it during
    /// the extraction, in passes; every output of both separations; and
    /// what the run said.
    struct Ran {
        made: [f64; 2],
        outputs: Outputs,
        said: Arc<Said>,
    }

    /// Follow `film` as it grows, then separate it: what was followed of
    /// the original, the outputs, and what the run said. `in_step`, the
    /// extraction writes a chunk once the follower made every block the
    /// last one let it make, so that what is followed does not depend on
    /// how fast the machine is; otherwise it writes one every `pause`.
    fn follow(
        bench: &Bench,
        film: &Film,
        pairs: Pairs,
        env: Vec<(&'static str, String)>,
        chunk: u64,
        pause: Duration,
        in_step: bool,
    ) -> (f64, Outputs, Arc<Said>) {
        let ran = follow_with(
            bench,
            film,
            pairs,
            [env.clone(), env],
            chunk,
            pause,
            in_step,
            None,
        );
        (ran.made[0], ran.outputs, ran.said)
    }

    /// [`follow`], each stage's worker on this machine started for its
    /// `envs`, the dub too when `film` has one. With `gate`, the machines
    /// make that many blocks for every chunk written, and the extraction
    /// waits for them (`in_step`).
    #[allow(clippy::too_many_arguments)]
    fn follow_with(
        bench: &Bench,
        film: &Film,
        pairs: Pairs,
        envs: [Vec<(&'static str, String)>; 2],
        chunk: u64,
        pause: Duration,
        in_step: bool,
        gate: Option<u64>,
    ) -> Ran {
        let mut job = bench.job("followed", pairs);
        // The manifest's film is the one grown here, which is what the
        // follower measures itself against.
        job.manifest.seconds = Some(film.frames as f64 / extract::RATE as f64);
        let said = Arc::new(Said::default());
        let progress = said.progress();
        let env_of = |stage: Stage| &envs[STAGES.iter().position(|s| *s == stage).unwrap()];
        let start = |stage: Stage, stop: Stop| {
            Worker::start_command(mock(env_of(stage)), Some(stop), &quiet)
        };
        let channels = film.channels();
        let follower = Follower::with_mock(
            &bench.config,
            &job,
            &bench.model,
            &channels,
            pairs,
            &start,
            Duration::from_millis(2),
        );
        if gate.is_some() {
            follower.gate.close();
        }
        let dir = job.extract_dir();
        let handed = std::thread::scope(|scope| {
            let following = scope.spawn(|| follower.follow(&progress));
            let devices = 1 + bench.config.remotes.len();
            // Every machine has its first element, or is done: a slow one
            // starting its worker does not find the extraction over.
            let begun = || {
                following.is_finished()
                    || (0..devices).all(|d| follower.waits(d) > 0)
                    || follower.gate.spent()
            };
            // Caught up once every machine found nothing to do twice since the
            // chunk was written, or is done: the second time, it had seen the
            // chunk. Behind a gate, also once every block granted is made.
            // A follower that never is gets a minute, then the extraction
            // runs on.
            let late = std::cell::Cell::new(false);
            let caught_up = || {
                let seen: Vec<u64> = (0..devices).map(|d| follower.waits(d)).collect();
                if let Some(blocks) = gate {
                    follower.gate.grant(blocks);
                }
                let deadline = std::time::Instant::now() + Duration::from_secs(60);
                while !late.get()
                    && !following.is_finished()
                    && !follower.gate.spent()
                    && (0..devices).any(|d| {
                        let waits = follower.waits(d);
                        waits < DONE && waits < seen[d] + 2
                    })
                {
                    late.set(std::time::Instant::now() >= deadline);
                    std::thread::sleep(Duration::from_millis(1));
                }
                follower.gate.revoke();
            };
            film.grow(
                &dir,
                chunk,
                pause,
                None,
                &begun,
                in_step.then_some(&caught_up as &dyn Fn()),
            );
            follower.gate.open();
            follower.decoded(true);
            following.join().unwrap()
        });
        film.rename(&dir);
        let made = [&handed.original, &handed.dub].map(|f| f.as_ref().map_or(0.0, Followed::made));
        said.extracted();
        let beside = handed
            .dub
            .as_ref()
            .map(Followed::beside)
            .unwrap_or_default();
        bench.separate(&job, handed.original, &beside, &envs[0], &progress);
        drop(beside);
        if !film.dub.is_empty() {
            bench.separate_dub(&job, &channels, handed.dub, &envs[1], &progress);
        }
        for stage in STAGES {
            assert!(
                !job.separate_dir().join(dir_of(stage)).exists(),
                "the follower's files are left"
            );
        }
        Ran {
            made,
            outputs: Bench::outputs(&job),
            said,
        }
    }

    fn same(followed: &[(String, Vec<u8>)], whole: &[(String, Vec<u8>)]) {
        let names =
            |set: &[(String, Vec<u8>)]| set.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
        assert_eq!(names(followed), names(whole));
        for ((name, a), (_, b)) in followed.iter().zip(whole) {
            assert!(a == b, "{name} differs from the one separated whole");
        }
    }

    #[test]
    fn an_element_followed_comes_out_as_it_would_whole() {
        let bench = Bench::new("same");
        let film = Film::objects(3, 60_000);
        let whole = bench.whole(&film, Pairs::default());
        // Six outputs: the rest and the voice of the three objects, the LFE
        // left out.
        assert_eq!(
            whole.len(),
            6,
            "{:?}",
            whole.iter().map(|(n, _)| n).collect::<Vec<_>>()
        );
        let (made, followed, said) = follow(
            &bench,
            &film,
            Pairs::default(),
            Vec::new(),
            2_000,
            Duration::from_millis(2),
            true,
        );
        assert!(
            made > 1.0,
            "{made}: not much was followed\n{}",
            said.lines()
        );
        assert!(said.any("during the extraction: "), "{}", said.lines());
        assert!(said.any("followed during the extraction: this machine finishes them first"));
        same(&followed, &whole);
    }

    /// What the follower said of itself, against what it handed on: it ends
    /// on exactly what was made of the `passes`, and only ever loses what a
    /// machine left out or an element let go took with it.
    fn ends_on_what_was_made(said: &Said, made: f64, passes: usize) {
        let (early, own) = said.ratios();
        let last = *early
            .last()
            .unwrap_or_else(|| panic!("nothing said\n{}", said.lines()));
        assert!(
            (last - made / passes as f64).abs() < 1e-3,
            "it said {last}, and {made} of {passes} passes were made"
        );
        // The step carries on from there to the end.
        assert!(
            own.first().is_some_and(|first| *first >= last - 1e-3),
            "{own:?}"
        );
        assert_eq!(own.last().copied(), Some(1.0));
    }

    /// **Said under the next step's name.** While the extraction runs, the
    /// separation of the original says how far it is as that step does, so
    /// a front end draws it as a bar of its own beside the decoding, rising
    /// with every block, and the step's own bar carries on from it.
    #[test]
    fn the_separation_says_how_far_it_is_during_the_extraction() {
        let bench = Bench::new("said");
        let film = Film::objects(3, 60_000);
        let (made, _, said) = follow(
            &bench,
            &film,
            Pairs::default(),
            Vec::new(),
            2_000,
            Duration::from_millis(2),
            true,
        );
        let (early, _) = said.ratios();
        assert!(early.len() > 3, "{early:?}");
        assert!(
            early.windows(2).all(|pair| pair[1] >= pair[0]),
            "nothing was let go, and it went back: {early:?}"
        );
        assert_eq!(
            Step::labelled(Stage::Original.label()),
            Some(Step::SeparateOriginal)
        );
        ends_on_what_was_made(&said, made, 3);
    }

    #[test]
    fn pairs_of_bed_channels_are_followed_and_pairs_of_objects_wait_for_the_end() {
        let bench = Bench::new("pairs");
        let pairs = Pairs {
            original: true,
            ..Pairs::default()
        };
        // A bed alone: every place is known from the header.
        let bed = Film {
            elements: vec![
                (0, 0, Some("L")),
                (1, 1, Some("R")),
                (2, 2, Some("C")),
                (3, 3, Some("LFE")),
                (4, 4, Some("Lss")),
                (5, 5, Some("Rss")),
            ],
            dub: Vec::new(),
            frames: 50_000,
        };
        let whole = bench.whole(&bed, pairs);
        let (made, followed, said) = follow(
            &bench,
            &bed,
            pairs,
            Vec::new(),
            2_000,
            Duration::from_millis(2),
            true,
        );
        assert!(made > 0.5, "{made}\n{}", said.lines());
        same(&followed, &whole);

        // Objects are placed by where they are over the whole film.
        let bench = Bench::new("pairs-objects");
        let film = Film::objects(4, 40_000);
        let whole = bench.whole(&film, pairs);
        let (made, followed, said) = follow(
            &bench,
            &film,
            pairs,
            Vec::new(),
            2_000,
            Duration::from_millis(1),
            true,
        );
        assert_eq!(made, 0.0);
        assert!(said.any("known once it is read"), "{}", said.lines());
        same(&followed, &whole);
    }

    #[test]
    fn a_machine_that_falls_behind_lets_the_last_elements_go() {
        let bench = Bench::new("behind");
        let film = Film::objects(4, 160_000);
        let whole = bench.whole(&film, Pairs::default());
        // A block takes the mock ten times as long as the decoder takes to
        // write one.
        let env = vec![(MOCK_BLOCK_MS, "20".to_string())];
        let (made, followed, said) = follow(
            &bench,
            &film,
            Pairs::default(),
            env,
            4_000,
            Duration::from_millis(1),
            false,
        );
        assert!(
            said.any("does not keep up with the extraction"),
            "{}",
            said.lines()
        );
        // What was made of an element let go is no longer counted.
        ends_on_what_was_made(&said, made, 4);
        same(&followed, &whole);
    }

    #[test]
    fn an_element_the_worker_fails_on_is_made_whole_after_the_extraction() {
        let bench = Bench::new("fails");
        let film = Film::objects(3, 60_000);
        let whole = bench.whole(&film, Pairs::default());
        let env = vec![(MOCK_FAIL, "11:2".to_string())];
        let (_, followed, said) = follow(
            &bench,
            &film,
            Pairs::default(),
            env,
            2_000,
            Duration::from_millis(2),
            true,
        );
        assert!(said.any("11: the separation failed: RuntimeError: the mock fails here; it is separated after the extraction"), "{}", said.lines());
        same(&followed, &whole);
    }

    /// The lines that say `needle` of a run, in order.
    fn lines_with(said: &Said, needle: &str) -> Vec<String> {
        said.0
            .lock()
            .unwrap()
            .iter()
            .filter(|line| line.contains(needle))
            .cloned()
            .collect()
    }

    /// **The dub, followed too, comes out as it would whole.** Its channels
    /// are followed in the time the original leaves, handed to the dub's
    /// step, finished there first, and every voice is the one a whole
    /// separation writes; the original's are untouched by it.
    #[test]
    fn a_dub_followed_comes_out_as_it_would_whole() {
        let bench = Bench::new("dub-same");
        let film = Film::objects(3, 60_000).dubbed();
        let whole = bench.whole(&film, Pairs::default());
        assert!(
            whole.iter().any(|(n, _)| n == "voix_C.wav"),
            "{:?}",
            whole.iter().map(|(n, _)| n).collect::<Vec<_>>()
        );
        let ran = follow_with(
            &bench,
            &film,
            Pairs::default(),
            [Vec::new(), Vec::new()],
            2_000,
            Duration::from_millis(2),
            true,
            None,
        );
        let said = &ran.said;
        assert!(
            ran.made[0] > 1.0 && ran.made[1] > 1.0,
            "{:?}\n{}",
            ran.made,
            said.lines()
        );
        assert!(
            said.any(
                "separating the dub too, in the time the original leaves: as many of its 5 channels"
            ),
            "{}",
            said.lines()
        );
        assert!(
            said.any("during the extraction: 5 of the dub's 5 channels followed"),
            "{}",
            said.lines()
        );
        assert!(
            said.any("5 channels followed during the extraction: this machine finishes them first")
        );
        // Under the dub's step's own name, as the original's is under its.
        assert_eq!(Step::labelled(Stage::Dub.label()), Some(Step::SeparateDub));
        let (early, own) = said.ratios_of(Stage::Dub);
        assert!(
            early.windows(2).all(|pair| pair[1] >= pair[0]) && early.len() > 3,
            "{early:?}"
        );
        assert!(
            own.first()
                .is_some_and(|first| *first >= early.last().unwrap() - 1e-3),
            "{own:?}"
        );
        assert_eq!(own.last().copied(), Some(1.0));
        same(&ran.outputs, &whole);
    }

    /// **The dub has what the original leaves.** A machine that makes six
    /// blocks for every block of film the extraction writes keeps the
    /// original's four elements and two of the dub's channels; the third it
    /// took falls behind and is let go — nothing of the original is — and
    /// the rest of the dub is never taken. Everything comes out as whole.
    #[test]
    fn a_machine_that_does_not_keep_up_lets_the_dub_go_before_the_original() {
        let bench = Bench::new("dub-behind");
        let film = Film::objects(4, 80_000).dubbed();
        let whole = bench.whole(&film, Pairs::default());
        let ran = follow_with(
            &bench,
            &film,
            Pairs::default(),
            [Vec::new(), Vec::new()],
            STEP,
            Duration::ZERO,
            true,
            Some(6),
        );
        let said = &ran.said;
        assert!(
            said.any("during the extraction: 4 of the original's 4 elements followed"),
            "{}",
            said.lines()
        );
        assert!(
            said.any("during the extraction: 2 of the dub's 5 channels followed"),
            "{}",
            said.lines()
        );
        assert_eq!(
            lines_with(said, " let go: "),
            [
                "  R let go: this machine does not keep up with the extraction on that many; it is \
              separated after it"
            ],
            "{}",
            said.lines()
        );
        same(&ran.outputs, &whole);
    }

    /// **A dub the worker cannot follow.** Its worker separates whole
    /// elements only — MSST, or a CPU: the dub is separated after the
    /// extraction, by the worker that was loaded for it, and the original
    /// is followed as before.
    #[test]
    fn a_dub_that_cannot_be_followed_is_separated_after_the_extraction() {
        let bench = Bench::new("dub-whole-only");
        let film = Film::objects(3, 60_000).dubbed();
        let whole = bench.whole(&film, Pairs::default());
        let ran = follow_with(
            &bench,
            &film,
            Pairs::default(),
            [Vec::new(), vec![(MOCK_WHOLE_ONLY, "1".to_string())]],
            2_000,
            Duration::from_millis(2),
            true,
            None,
        );
        let said = &ran.said;
        assert!(
            said.any(
                "the dub is separated after the extraction: the separation failed: \
                      RuntimeError: the mock separates whole only"
            ),
            "{}",
            said.lines()
        );
        assert_eq!(ran.made[1], 0.0);
        assert!(ran.made[0] > 1.0, "{:?}\n{}", ran.made, said.lines());
        assert!(!said.any("of the dub's 5 channels followed"));
        same(&ran.outputs, &whole);
    }

    /// **The dub in pairs** is separated after the extraction, as before,
    /// and the original is followed beside it.
    #[test]
    fn the_dub_in_pairs_is_separated_after_the_extraction() {
        let bench = Bench::new("dub-pairs");
        let pairs = Pairs {
            dub: true,
            ..Pairs::default()
        };
        let film = Film::objects(3, 60_000).dubbed();
        let whole = bench.whole(&film, pairs);
        let ran = follow_with(
            &bench,
            &film,
            pairs,
            [Vec::new(), Vec::new()],
            2_000,
            Duration::from_millis(2),
            true,
            None,
        );
        assert!(
            ran.said
                .any("the dub's channels are separated in pairs: after the extraction"),
            "{}",
            ran.said.lines()
        );
        assert_eq!(ran.made[1], 0.0);
        assert!(ran.made[0] > 1.0);
        same(&ran.outputs, &whole);
    }

    #[test]
    fn every_pairing_of_the_dub_still_separates_it() {
        // Never followed, but cut by the same plan and run by the same
        // queue. The mock's voice of a channel is the same alone, in a pair
        // or averaged from two: every pairing must write the same files.
        let bench = Bench::new("dub");
        let names = ["L", "R", "C", "LFE", "Lss", "Rss"];
        let mut written = Vec::new();
        for pairs in [
            Pairs::default(),
            Pairs {
                dub: true,
                ..Pairs::default()
            },
            Pairs::ALL,
        ] {
            let job = bench.job(&format!("dub-{}", super::super::mode_dub(pairs)), pairs);
            let dir = job.extract_dir();
            fs::create_dir_all(&dir).unwrap();
            for (i, name) in names.iter().enumerate() {
                let mut bytes = Film::wav_header(30_000);
                bytes.extend(Film::bytes(40 + i as u32, 0, 30_000));
                fs::write(dir.join(format!("{}{name}.wav", extract::DUB)), bytes).unwrap();
            }
            let start = |stop: Stop| Worker::start_command(mock(&[]), Some(stop), &quiet);
            let channels: Vec<String> = names.iter().map(|n| n.to_string()).collect();
            let (separated, _) = super::super::dub_with(
                &bench.config,
                &job,
                &bench.model,
                &channels,
                pairs,
                None,
                Some(&start),
                &quiet,
            )
            .unwrap();
            assert_eq!(separated.elements, ["L", "C", "R", "Rss", "Lss"]);
            written.push(Bench::outputs(&job));
        }
        for set in &written {
            let names: Vec<&str> = set.iter().map(|(n, _)| n.as_str()).collect();
            assert!(
                names.contains(&"voix_C.wav") && names.contains(&"voix_Lss.wav"),
                "{names:?}"
            );
            assert!(!names.iter().any(|n| n.starts_with("_pair_")), "{names:?}");
        }
        let voices = |set: &Outputs| -> Outputs {
            set.iter()
                .filter(|(n, _)| n.starts_with(extract::DUB_VOICE))
                .cloned()
                .collect()
        };
        same(&voices(&written[1]), &voices(&written[0]));
        same(&voices(&written[2]), &voices(&written[0]));
    }

    #[test]
    fn a_failed_extraction_leaves_nothing_of_what_was_followed() {
        let bench = Bench::new("failed");
        let film = Film::objects(3, 60_000).dubbed();
        let job = bench.job("failed", Pairs::default());
        let env = vec![(MOCK_BLOCK_MS, "5".to_string())];
        let start = |_: Stage, stop: Stop| Worker::start_command(mock(&env), Some(stop), &quiet);
        let channels = film.channels();
        let follower = Follower::with_mock(
            &bench.config,
            &job,
            &bench.model,
            &channels,
            Pairs::default(),
            &start,
            Duration::from_millis(2),
        );
        let said = Arc::new(Said::default());
        let progress = said.progress();
        let followed = std::thread::scope(|scope| {
            let following = scope.spawn(|| follower.follow(&progress));
            film.grow(
                &job.extract_dir(),
                2_000,
                Duration::from_millis(2),
                Some(40_000),
                &|| said.any("as the extraction writes it"),
                None,
            );
            follower.decoded(false);
            following.join().unwrap()
        });
        assert!(followed.original.is_none() && followed.dub.is_none());
        for stage in STAGES {
            assert!(!job.separate_dir().join(dir_of(stage)).exists());
        }
        let outputs = fs::read_dir(job.separate_dir())
            .map(|entries| entries.filter_map(|e| e.ok()).count())
            .unwrap_or(0);
        assert_eq!(outputs, 0, "something of the failed run is left");
        // Nothing to say: the extraction says why it failed.
        assert!(
            !said.any("separated after the extraction"),
            "{}",
            said.lines()
        );
    }

    /// A remote that is the mock, reached through this machine's shell, its
    /// files under the bench's `far/`; `env` is its environment. The shell
    /// becomes the mock (`exec`), so that killing one kills the other, as
    /// ssh's end does over there.
    #[cfg(unix)]
    fn far_mock(bench: &Bench, env: &[(&str, String)]) -> crate::config::Remote {
        let mut command = format!(
            "exec env {MOCK}=1 {MOCK_DIR}={}",
            remote::quote(&bench.root.join("far").to_string_lossy())
        );
        for (key, value) in env {
            command.push_str(&format!(" {key}={}", remote::quote(value)));
        }
        command.push_str(&format!(
            " {} separate::follow::tests::mock_worker --exact --nocapture --test-threads=1 --",
            remote::quote(&std::env::current_exe().unwrap().to_string_lossy())
        ));
        crate::config::Remote {
            host: remote::SHELL.to_string(),
            command,
            idle_below_gib: 0.0,
        }
    }

    /// What is left in the remote's directory `sub`.
    #[cfg(unix)]
    fn left_there(bench: &Bench, sub: &str) -> Vec<String> {
        fs::read_dir(bench.root.join("far").join(sub))
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[cfg(unix)]
    #[test]
    fn a_remote_follows_beside_this_machine_and_finishes_what_it_followed() {
        let mut bench = Bench::new("far");
        let film = Film::objects(4, 60_000);
        let whole = bench.whole(&film, Pairs::default());
        bench.config.remotes = vec![far_mock(&bench, &[])];
        let (made, followed, said) = follow(
            &bench,
            &film,
            Pairs::default(),
            Vec::new(),
            2_000,
            Duration::from_millis(2),
            true,
        );
        let far = remote::SHELL;
        assert!(made > 2.0, "{made}\n{}", said.lines());
        assert!(
            said.any("following beside this machine"),
            "{}",
            said.lines()
        );
        // Both followed some: the first in line went here, the second there.
        assert!(said.any(&format!(" on {far})")), "{}", said.lines());
        assert!(said.any("here, "), "{}", said.lines());
        assert!(said.any("each machine finishes its own first"));
        assert!(
            said.any(&format!(
                "[{far}] element 11 (7 of its 8 blocks made): the rest of it"
            )),
            "{}",
            said.lines()
        );
        same(&followed, &whole);
        // Every input and stem carried there went once it had served.
        assert_eq!(
            left_there(&bench, "in"),
            Vec::<String>::new(),
            "{}",
            said.lines()
        );
        assert_eq!(left_there(&bench, "out"), Vec::<String>::new());
    }

    #[cfg(unix)]
    #[test]
    fn a_remote_that_dies_leaves_what_it_followed_to_be_made_whole() {
        let mut bench = Bench::new("far-dies");
        let film = Film::objects(4, 60_000);
        let whole = bench.whole(&film, Pairs::default());
        // The second element in line is the remote's: it dies on its second
        // block.
        bench.config.remotes = vec![far_mock(&bench, &[(MOCK_DIE, "11:1".to_string())])];
        let (made, followed, said) = follow(
            &bench,
            &film,
            Pairs::default(),
            Vec::new(),
            2_000,
            Duration::from_millis(2),
            true,
        );
        let far = remote::SHELL;
        assert!(
            said.any(&format!("{far}: left out of the extraction: ")),
            "{}",
            said.lines()
        );
        assert!(said.any("what it followed is separated after it"));
        // This machine kept following.
        assert!(said.any("followed during the extraction: this machine finishes"));
        // What the remote made of its element is no longer counted.
        ends_on_what_was_made(&said, made, 4);
        same(&followed, &whole);
    }

    /// **A remote whose dub worker dies.** This machine's dub worker
    /// follows nothing, so the dub's channels are the remote's; its worker
    /// for them dies on its second block. What it followed of the dub is
    /// made whole after the extraction; its worker for the original, beside
    /// it, keeps following and hands on what it followed.
    #[cfg(unix)]
    #[test]
    fn a_remote_that_dies_following_the_dub_keeps_following_the_original() {
        let mut bench = Bench::new("far-dub-dies");
        let film = Film::objects(4, 60_000).dubbed();
        let whole = bench.whole(&film, Pairs::default());
        bench.config.remotes = vec![far_mock(&bench, &[(MOCK_DIE_BESIDE, "*:1".to_string())])];
        let ran = follow_with(
            &bench,
            &film,
            Pairs::default(),
            [Vec::new(), vec![(MOCK_WHOLE_ONLY, "1".to_string())]],
            2_000,
            Duration::from_millis(2),
            true,
            None,
        );
        let said = &ran.said;
        let far = remote::SHELL;
        assert!(
            said.any(&format!("{far}: left out of the extraction for the dub: ")),
            "{}",
            said.lines()
        );
        assert!(said.any("what it followed of it is separated after it"));
        // Its worker for the original kept following, and handed on.
        let original = lines_with(said, "of the original's 4 elements followed");
        assert!(
            original.len() == 1 && original[0].contains(&format!(" on {far})")),
            "{}",
            said.lines()
        );
        assert!(
            !said.any("of the dub's 5 channels followed"),
            "{}",
            said.lines()
        );
        // The remote's two workers had a directory each.
        assert!(bench.root.join("far-beside").is_dir());
        same(&ran.outputs, &whole);
    }

    #[cfg(unix)]
    #[test]
    fn a_busy_remote_follows_nothing_and_the_rest_is_made_here() {
        let mut bench = Bench::new("far-busy");
        let film = Film::objects(3, 60_000);
        let whole = bench.whole(&film, Pairs::default());
        bench.config.remotes = vec![far_mock(&bench, &[(MOCK_BUSY, "1".to_string())])];
        let (made, followed, said) = follow(
            &bench,
            &film,
            Pairs::default(),
            Vec::new(),
            2_000,
            Duration::from_millis(2),
            true,
        );
        assert!(
            said.any("another run has its GPU; asked again for the separation"),
            "{}",
            said.lines()
        );
        assert!(made > 1.0, "{made}\n{}", said.lines());
        assert!(!said.any(&format!(" on {})", remote::SHELL)));
        same(&followed, &whole);
    }

    #[cfg(unix)]
    #[test]
    fn a_remote_that_hangs_on_what_it_followed_is_overtaken_here() {
        let mut bench = Bench::new("far-hangs");
        let film = Film::objects(4, 60_000);
        let whole = bench.whole(&film, Pairs::default());
        // The remote's element hangs on its last block, after the
        // extraction.
        bench.config.remotes = vec![far_mock(&bench, &[(MOCK_HANG, "11:7".to_string())])];
        let (_, followed, said) = follow(
            &bench,
            &film,
            Pairs::default(),
            Vec::new(),
            2_000,
            Duration::from_millis(2),
            true,
        );
        let far = remote::SHELL;
        assert!(
            said.any(&format!(
                "(11) again, beside {far}: the first to finish keeps it"
            )),
            "{}",
            said.lines()
        );
        assert!(
            said.any(&format!(
                "[{far}] 11: made first by another machine; this one's is dropped"
            )),
            "{}",
            said.lines()
        );
        same(&followed, &whole);
    }

    #[cfg(unix)]
    #[test]
    fn a_remote_that_hangs_during_the_extraction_is_let_go_when_it_ends() {
        let mut bench = Bench::new("far-stalls");
        let film = Film::objects(4, 60_000);
        let whole = bench.whole(&film, Pairs::default());
        // The remote hangs on the first block of what it takes; the
        // extraction runs on without it.
        bench.config.remotes = vec![far_mock(&bench, &[(MOCK_HANG, "*:0".to_string())])];
        let started = std::time::Instant::now();
        let (_, followed, said) = follow(
            &bench,
            &film,
            Pairs::default(),
            Vec::new(),
            2_000,
            Duration::from_millis(2),
            false,
        );
        assert!(
            said.any(&format!(
                "{}: still on a block 1 s after the extraction ended: let go",
                remote::SHELL
            )),
            "{}",
            said.lines()
        );
        // Not the hour a worker is given between two words.
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "{:?}",
            started.elapsed()
        );
        same(&followed, &whole);
    }
}
