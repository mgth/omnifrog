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
//! first with a block ready first.
//!
//! **Behind, through a burst.** The extraction reads the container at the
//! disk's pace, so the film comes faster where the picture is light, and
//! the element served last falls behind first. A machine with one element
//! of the original [`BEHIND`] blocks behind is watched over a round — as
//! many blocks of the original as it follows elements of it ([`Trend`]):
//! what its inputs hold that it has not separated yet, summed, shrinks over
//! a round when it keeps up, and a burst only holds it back for a while.
//! Only [`STUCK`] rounds in a row over which it did not shrink let the last
//! of the followed go — one round, once one was let go since it fell
//! behind — and only one under half made: one further along costs more
//! made again whole than finished after the extraction, unless the machine
//! already has more left than a whole element and another machine could
//! take it. So a machine never lets one go a block. What is let go, or
//! left in line, is separated whole after the extraction, by any machine,
//! like the elements nobody followed.
//!
//! **A trace.** Every block a machine asks for, how long it took and on
//! what (a remote's transfers, the worker, folding the stems in), every
//! stretch of a few seconds with nothing to do and what it waited for, and
//! every round watched go to the run's log ([`Event::Trace`]): enough to
//! tell a GPU that stalled from an extraction that did.
//!
//! A remote is sent what its inputs gained before each block, and
//! its stems come back after it; one that is busy or left out follows
//! nothing, and one still loading its model when the decode ends, or still
//! on a block [`FAR_GRACE`] later, is let go rather than hold the
//! extraction's end up: what it followed is made whole after it.
//!
//! **The dub, in the gaps the original leaves.** The original rarely keeps
//! a GPU busy all through the extraction, and the dub's channels grow in
//! step with it. Each machine has a second worker, with the dub's model,
//! that follows the dub's channels block by block, from a line of their
//! own. The machine's one thread drives both workers, one request at a
//! time — a GPU never separates two things at once (see [`Lock`]) — and
//! the original comes first: its block when one is ready, its line when
//! there is time for one more, and a block of the dub only when it ends,
//! with room to spare, before the original's next one is ready — and the
//! dub's line, which stays open for as long as a channel taken could still
//! be half made by the decode's end, not the original's two blocks. That room
//! is the original's slack, which it needs: the extraction does not write
//! at one rate, and where the video is lighter the film comes faster — on
//! a film, a dub served whenever the original had nothing ready cost the
//! original element after element there. So the original fares as it
//! would alone; the dub is never let go for falling behind, and what was
//! made of it is kept when the decode ends if it is half of it or more.
//! The two workers load side by side with the first blocks, the dub's
//! after the original's.
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
use crate::worker::{Ask, Grid, Lock, REST_SUFFIX, Started, Took, VOICE_SUFFIX, Worker};
use crate::{Error, Result, wav};
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime};

/// Where a follower keeps what it grows, in the separation's directory.
pub(super) fn dir_of(stage: Stage) -> &'static str {
    match stage {
        Stage::Original => "_follow_element",
        Stage::Dub => "_follow_channel",
    }
}

/// What a follower follows, in the order a machine serves them.
const STAGES: [Stage; 2] = [Stage::Original, Stage::Dub];
/// Blocks a followed element of the original may be behind the extraction
/// before the machine is watched for a round ([`Trend`]); and how far ahead
/// of the next pass in line the extraction may be for a machine to still
/// take it.
const BEHIND: u64 = 2;
/// Rounds in a row over which a machine behind did not catch up before it
/// lets the last of the original's followed go: the round a burst ends in
/// looks like one that does not, and the next one tells. Once one was let
/// go, one round is enough for the next ([`Trend::needed`]).
const STUCK: usize = 2;
/// How long a machine may have nothing to do before the trace says so: a
/// poll or two is not worth a line. The tests' machines wait for
/// milliseconds.
const IDLE_TRACED: Duration = if cfg!(test) {
    Duration::from_millis(1)
} else {
    Duration::from_secs(2)
};
/// How much room a block of the dub needs before the original's next one
/// is ready, over what a block costs: the extraction does not write at one
/// rate — where the video is lighter, the film comes faster — and the
/// original needs that slack.
const DUB_FIT: f64 = 1.25;
/// The share of its blocks a channel of the dub must have made when the
/// decode ends to be finished by the machine that followed it; less, it is
/// let go, and made whole by any.
const DUB_KEPT: f64 = 0.5;
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
    /// What one of its blocks costs: the frames the extraction writes
    /// while the machine makes one, on average.
    cost: Option<u64>,
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

/// Whether a machine behind the extraction catches up with it: what it has
/// left to separate of what its inputs hold — the frames, summed over the
/// elements of the original it follows — read after every block of the
/// original it makes. The first read opens a round of as many blocks as it
/// follows elements, and the round ends on whether that shrank since. Read
/// after a block every time, so that a block's own frames do not look like
/// a burst; over a whole round, so that the element served last, which a
/// burst leaves furthest behind, does not pass for a machine that does not
/// keep up.
#[derive(Default)]
struct Trend {
    /// What was left at the round's start, and the blocks made since.
    mark: Option<(u64, usize)>,
    /// The rounds in a row that ended on no less.
    stuck: usize,
    /// One was let go since the machine fell behind: it is known not to keep
    /// up, and a round that does not shrink is enough for the next.
    shed: bool,
}

/// Where a round is, after a block.
#[derive(Debug, PartialEq, Eq)]
enum Round {
    /// That block opened the first.
    Opened,
    Going,
    /// One ended on less than `was`, what was left when it opened.
    Shrank {
        was: u64,
    },
    /// One ended on no less than `was`: the `rounds`-th in a row.
    Stuck {
        was: u64,
        rounds: usize,
    },
}

impl Trend {
    /// A block was made, the machine behind: `left` to separate, of
    /// `streams` elements. The next round opens where one ends.
    fn made(&mut self, left: u64, streams: usize) -> Round {
        let Some((was, blocks)) = self.mark else {
            self.mark = Some((left, 0));
            return Round::Opened;
        };
        let blocks = blocks + 1;
        if blocks < streams {
            self.mark = Some((was, blocks));
            return Round::Going;
        }
        self.mark = Some((left, 0));
        if left < was {
            self.stuck = 0;
            Round::Shrank { was }
        } else {
            self.stuck += 1;
            Round::Stuck {
                was,
                rounds: self.stuck,
            }
        }
    }

    fn watching(&self) -> bool {
        self.mark.is_some()
    }

    /// The rounds in a row that let the next one go.
    fn needed(&self) -> usize {
        if self.shed { 1 } else { STUCK }
    }

    /// One was let go: a round opens afresh, of one element fewer.
    fn shed(&mut self) {
        *self = Self {
            shed: true,
            ..Self::default()
        };
    }
}

/// How far ahead of the next pass in `common`'s line the extraction may be,
/// in blocks, for a machine to still take it. The original's, [`BEHIND`]: a
/// pass taken later starts too far behind to keep up. The dub's, as long as
/// a channel taken then could still be [`DUB_KEPT`] made by the decode's
/// end: it only has the gaps the original leaves, so starting behind costs
/// the original nothing, and the dub's line opens to a machine only once
/// the original's has closed there.
fn intake_window(common: &Common, grid: Grid) -> u64 {
    match common.stage {
        Stage::Dub if common.whole > 0 => {
            BEHIND.max(((1.0 - DUB_KEPT) * grid.blocks(common.whole) as f64) as u64)
        }
        _ => BEHIND,
    }
}

/// Whether a block of the dub that costs `cost` frames of the extraction
/// fits in a `gap` of frames before the original's next block is ready,
/// with the room [`DUB_FIT`] keeps for the original.
fn dub_fits(cost: u64, gap: u64) -> bool {
    cost as f64 * DUB_FIT <= gap as f64
}

/// Whether a channel of the dub with `blocks` of its `total` made when the
/// decode ends is finished by the machine that followed it ([`DUB_KEPT`]).
fn dub_kept(blocks: u64, total: u64) -> bool {
    blocks > 0 && blocks as f64 >= DUB_KEPT * total as f64
}

/// Whether an element `written` frames into a film of `whole` is kept
/// rather than let go, its machine `left` frames behind what its inputs
/// hold: half made or more, making it again whole costs more than
/// finishing it — unless another machine could take it (`others`) and this
/// one already has more left than a whole element, which it alone would
/// finish after the extraction while that machine made the element whole.
fn kept_half_made(written: u64, whole: u64, left: u64, others: bool) -> bool {
    whole > 0 && written.saturating_mul(2) >= whole && !(others && left >= whole)
}

/// A stretch with nothing to do: since when, and the element nearest a
/// block — its side, its rank there, and what its inputs held then.
struct Idle {
    since: Instant,
    next: Option<(usize, usize, u64)>,
}

/// « 3190.4 s »: the decoder's frames as film.
fn film(frames: u64) -> String {
    format!("{:.1} s", frames as f64 / extract::RATE as f64)
}

/// « 10.42 s ».
fn secs(took: Duration) -> String {
    format!("{:.2} s", took.as_secs_f64())
}

/// How a trace line names a stream of `stage`: the original's by its name,
/// the dub's after « the dub's ».
fn traced(stage: Stage, base: &str) -> String {
    match stage {
        Stage::Original => base.to_string(),
        Stage::Dub => format!("the dub's {base}"),
    }
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
            cost: None,
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
        // The decoder is done: every file is as long as it will be. What
        // was never begun goes, and so does the dub's that is less than
        // half made: whole, it can be made anywhere.
        for (s, side) in sides.iter_mut().enumerate() {
            let (Some(machine), Some(common)) = (&mut side.machine, &commons[s]) else {
                continue;
            };
            let grid = side.grid;
            for stream in &mut machine.streams {
                stream.length = stream.held();
            }
            let (kept, gone): (Vec<Stream>, Vec<Stream>) = std::mem::take(&mut machine.streams)
                .into_iter()
                .partition(|stream| match common.stage {
                    Stage::Original => stream.blocks > 0,
                    Stage::Dub => {
                        grid.is_some_and(|grid| dub_kept(stream.blocks, grid.blocks(stream.length)))
                    }
                });
            for stream in gone {
                if stream.blocks > 0 {
                    self.trace(
                        device,
                        format!(
                            "the decode is over: {} let go, {}: it is made whole after the \
                             extraction",
                            traced(common.stage, &stream.base),
                            stream.describe()
                        ),
                        progress,
                    );
                }
                common.made(stream.pass, None, progress);
                machine.let_go(stream);
            }
            machine.streams = kept;
            if let Some(grid) = grid {
                let left: u64 = machine
                    .streams
                    .iter()
                    .map(|stream| grid.blocks(stream.length).saturating_sub(stream.blocks))
                    .sum();
                self.trace(
                    device,
                    format!(
                        "the decode is over: {} followed{}, {left} blocks of them left",
                        machine.streams.len(),
                        match common.stage {
                            Stage::Original => "",
                            Stage::Dub => " of the dub",
                        }
                    ),
                    progress,
                );
            }
        }
        sides
    }

    /// One block at a time, the original's first: the first element with
    /// one ready first. A pass is taken from the original's line at once,
    /// and another each time a block was made since, while the line is not
    /// too far behind; a machine [`BEHIND`] on the original is watched for
    /// rounds of blocks (`watch`) — as when the original was followed alone.
    ///
    /// The dub has the gaps: a block of it only when the original has none
    /// ready, and only if the block would end before the original's next
    /// one is ready, with room to spare ([`DUB_FIT`]), by what a block
    /// costs that machine in frames of the extraction ([`Side::cost`]). A
    /// burst of film, where the extraction goes faster, raises that cost
    /// and leaves no gap: the original keeps its slack. So it never holds
    /// the original back,
    /// and it is never let go for falling behind: it makes what it can,
    /// and what it took and made too little of is let go when the decode
    /// ends ([`DUB_KEPT`]).
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
        let mut trend = Trend::default();
        let mut may_take = true;
        let mut idle: Option<Idle> = None;
        // A block of the original was made since the last look.
        let mut made_one = false;
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
            if std::mem::take(&mut made_one)
                && let (Some(common), Some(grid), Some(machine)) =
                    (&commons[0], sides[0].grid, &mut sides[0].machine)
            {
                let held: Vec<u64> = machine.streams.iter().map(Stream::held).collect();
                if self.watch(
                    device, machine, &mut trend, grid, &held, common, &prefix, progress,
                ) {
                    continue 'serve;
                }
            }
            // Frames until the original has a block ready again.
            let gap = match (sides[0].grid, &sides[0].machine) {
                (Some(grid), Some(machine)) => machine
                    .streams
                    .iter()
                    .map(|stream| grid.ready_at(stream.blocks).saturating_sub(stream.held()))
                    .min()
                    .unwrap_or(u64::MAX),
                _ => u64::MAX,
            };
            // The original's block, its line, the dub's block, its line.
            for (s, intake) in [(0, false), (0, true), (1, false), (1, true)] {
                let (Some(common), Some(grid)) = (&commons[s], sides[s].grid) else {
                    continue;
                };
                if intake {
                    if !may_take || !sides[s].open {
                        continue;
                    }
                    // The dub takes another only while what it has is not
                    // behind: its gaps would not make that one too.
                    if common.stage == Stage::Dub
                        && sides[s].machine.as_ref().is_some_and(|machine| {
                            machine
                                .streams
                                .iter()
                                .any(|stream| grid.ready(stream.held()) >= stream.blocks + BEHIND)
                        })
                    {
                        continue;
                    }
                    match self.intake(common, grid) {
                        Ok(Some(stream)) => {
                            if let Some(idle) = idle.take() {
                                self.trace_idle(device, sides, commons, idle, progress);
                            }
                            self.trace(
                                device,
                                format!(
                                    "takes {}, {} of film written",
                                    traced(common.stage, &stream.base),
                                    film(stream.held())
                                ),
                                progress,
                            );
                            if let Some(machine) = &mut sides[s].machine {
                                machine.streams.push(stream);
                            }
                            if s == 0 {
                                trend = Trend::default();
                            }
                            may_take = false;
                            continue 'serve;
                        }
                        Ok(None) => {
                            sides[s].open = false;
                            self.trace(
                                device,
                                format!(
                                    "takes no more{}: the line is empty, or {} blocks behind",
                                    match common.stage {
                                        Stage::Original => "",
                                        Stage::Dub => " of the dub",
                                    },
                                    intake_window(common, grid)
                                ),
                                progress,
                            );
                        }
                        Err(why) => {
                            let stopped = self.stops[s][device].raised();
                            out(&mut sides[s], common, device, (why, stopped), progress);
                        }
                    }
                    continue;
                }
                if s == 1 {
                    let cost = sides[1].cost.or(sides[0].cost).unwrap_or(0);
                    if !dub_fits(cost, gap) {
                        continue;
                    }
                }
                let ready = sides[s].machine.as_ref().and_then(|machine| {
                    let held: Vec<u64> = machine.streams.iter().map(Stream::held).collect();
                    (0..machine.streams.len())
                        .find(|&i| held[i] >= grid.ready_at(machine.streams[i].blocks))
                        .map(|i| (i, held[i]))
                });
                let Some((i, held_i)) = ready else {
                    continue;
                };
                if let Some(idle) = idle.take() {
                    self.trace_idle(device, sides, commons, idle, progress);
                }
                let side = &mut sides[s];
                let Some(machine) = &mut side.machine else {
                    continue;
                };
                #[cfg(test)]
                self.gate.pass();
                let asked = Instant::now();
                let stepped = match machine.split(i) {
                    Err(why) => Stepped::Out(why),
                    Ok((worker, stream)) => {
                        let inputs = stream.raw.clone();
                        match stream.step(worker, &inputs, held_i, false) {
                            Ok(made) => {
                                common.made(stream.pass, Some((device, stream.written)), progress);
                                self.trace(
                                    device,
                                    format!(
                                        "{} block {} in {} ({}), {} of film written",
                                        traced(common.stage, &stream.base),
                                        stream.blocks.saturating_sub(1),
                                        secs(asked.elapsed()),
                                        made.describe(),
                                        film(held_i)
                                    ),
                                    progress,
                                );
                                // What the extraction wrote meanwhile.
                                let spent = stream.held().saturating_sub(held_i);
                                side.cost =
                                    Some(side.cost.map_or(spent, |cost| (3 * cost + spent) / 4));
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
                        may_take = true;
                        made_one |= s == 0;
                    }
                    Stepped::Refused(why) => {
                        let stream = machine.streams.remove(i);
                        common.made(stream.pass, None, progress);
                        progress(Event::message(format!(
                            "  {prefix}{}: {why}; it is separated after the extraction",
                            traced(common.stage, &stream.base)
                        )));
                        machine.let_go(stream);
                        if s == 0 {
                            trend = Trend::default();
                        }
                    }
                    Stepped::Out(why) => {
                        let stopped = self.stops[s][device].raised();
                        out(&mut sides[s], common, device, (why, stopped), progress);
                        if s == 0 {
                            trend = Trend::default();
                        }
                    }
                }
                continue 'serve;
            }
            if idle.is_none() {
                // What it waits for: the original's element nearest a
                // block, or the dub's.
                let next = (0..2).find_map(|s| {
                    let (grid, machine) = (sides[s].grid?, sides[s].machine.as_ref()?);
                    (0..machine.streams.len())
                        .map(|i| (i, machine.streams[i].held()))
                        .min_by_key(|&(i, held)| {
                            grid.ready_at(machine.streams[i].blocks)
                                .saturating_sub(held)
                        })
                        .map(|(i, held)| (s, i, held))
                });
                idle = Some(Idle {
                    since: Instant::now(),
                    next,
                });
            }
            if sides.iter().any(|side| side.loading) {
                // Not following yet: the tests' extraction waits for it.
                self.pause(self.poll);
            } else {
                self.wait(device);
            }
        }
        if let Some(idle) = idle.take() {
            self.trace_idle(device, sides, commons, idle, progress);
        }
    }

    /// After a block of the original `device` made, its inputs holding
    /// `held`: whether it is behind the extraction, and once [`STUCK`]
    /// rounds of blocks behind it in a row did not shrink what it has left
    /// to separate ([`Trend`]), the last of the followed let go — unless it
    /// is half made ([`kept_half_made`]). `true` when one was.
    #[allow(clippy::too_many_arguments)]
    fn watch(
        &self,
        device: usize,
        machine: &mut Machine,
        trend: &mut Trend,
        grid: Grid,
        held: &[u64],
        common: &Common,
        prefix: &str,
        progress: Progress,
    ) -> bool {
        let behind = |i: usize| {
            grid.ready(held[i])
                .saturating_sub(machine.streams[i].blocks)
        };
        let worst = (0..machine.streams.len()).max_by_key(|&i| behind(i));
        let Some(worst) = worst.filter(|&i| behind(i) >= BEHIND) else {
            if trend.watching() {
                self.trace(device, "caught up".to_string(), progress);
            }
            *trend = Trend::default();
            return false;
        };
        let streams = machine.streams.len();
        let lag: u64 = machine
            .streams
            .iter()
            .zip(held)
            .map(|(stream, &held)| held.saturating_sub(stream.written))
            .sum();
        let needed = trend.needed();
        let was = match trend.made(lag, streams) {
            Round::Going => return false,
            Round::Opened => {
                self.trace(
                    device,
                    format!(
                        "behind: {} is {} blocks behind, {} left to separate of what its {streams} \
                         elements hold; looked at again in {streams} blocks",
                        machine.streams[worst].base,
                        behind(worst),
                        film(lag)
                    ),
                    progress,
                );
                return false;
            }
            Round::Shrank { was } => {
                self.trace(
                    device,
                    format!(
                        "a round behind: {} left to separate, {} before it: catching up, all kept",
                        film(lag),
                        film(was)
                    ),
                    progress,
                );
                return false;
            }
            Round::Stuck { was, rounds } if rounds < needed => {
                self.trace(
                    device,
                    format!(
                        "a round behind: {} left to separate, {} before it: not catching up, \
                         {rounds} round of {needed} in a row",
                        film(lag),
                        film(was)
                    ),
                    progress,
                );
                return false;
            }
            Round::Stuck { was, .. } => was,
        };
        let last = machine.streams.last().expect("one is behind");
        let others = !self.config.remotes.is_empty();
        if kept_half_made(last.written, common.whole, lag, others) {
            self.trace(
                device,
                format!(
                    "a round behind: {} left to separate, {} before it: not catching up, {needed} \
                     rounds in a row, but {} is half made: kept, and finished after the extraction",
                    film(lag),
                    film(was),
                    last.base
                ),
                progress,
            );
            return false;
        }
        self.trace(
            device,
            format!(
                "a round behind: {} left to separate, {} before it: not catching up, {needed} \
                 rounds in a row: {} let go",
                film(lag),
                film(was),
                last.base
            ),
            progress,
        );
        let stream = machine.streams.pop().expect("one is behind");
        common.made(stream.pass, None, progress);
        progress(Event::message(format!(
            "  {prefix}{} let go: {} does not keep up with the extraction on that many; it is \
             separated after it",
            stream.base,
            name_of(self.config, device)
        )));
        machine.let_go(stream);
        trend.shed();
        true
    }

    /// One line of the trace ([`Event::Trace`]): what `device` did.
    fn trace(&self, device: usize, text: String, progress: Progress) {
        progress(Event::trace(format!(
            "  follow, {}: {text}",
            name_of(self.config, device)
        )));
    }

    /// The end of a stretch `device` had nothing to do: said if it was long
    /// enough to matter, with what it waited for and how far the extraction
    /// got meanwhile.
    fn trace_idle(
        &self,
        device: usize,
        sides: &[Side; 2],
        commons: &[Option<Common>; 2],
        idle: Idle,
        progress: Progress,
    ) {
        let long = idle.since.elapsed();
        if long < IDLE_TRACED {
            return;
        }
        let waited = idle.next.and_then(|(s, i, then)| {
            let (grid, machine, common) = (
                sides[s].grid?,
                sides[s].machine.as_ref()?,
                commons[s].as_ref()?,
            );
            let stream = machine.streams.get(i)?;
            Some(format!(
                "waiting for {} block {}, which needs {} of film: {} written then, {} now",
                traced(common.stage, &stream.base),
                stream.blocks,
                film(grid.ready_at(stream.blocks)),
                film(then),
                film(stream.held())
            ))
        });
        let waited = waited.unwrap_or_else(|| "nothing followed".to_string());
        self.trace(device, format!("idle {}, {waited}", secs(long)), progress);
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
        let mut said: Option<usize> = None;
        let fit = match remote::probe(remote, &ssh) {
            Err(why) => {
                let _ = sender.send((first, Err(why)));
                said = Some(first);
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
            if said == Some(s) {
                continue;
            }
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
    /// while it would start less than [`intake_window`] behind the
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
        if grid.ready(held) >= intake_window(common, grid) {
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
    /// step: for another step's workers to stand beside. This machine's
    /// share of the lock is handed on whether its worker is alive or not —
    /// it is held until the worker is dropped; a remote's worker gone is no
    /// longer there to stand beside.
    pub fn beside(&mut self) -> Beside {
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
        for machine in &mut self.machines {
            let Some(worker) = machine.worker.as_mut() else {
                continue;
            };
            match machine.device {
                0 => beside.lock = worker.lock(),
                d if worker.is_alive() => beside.seats[d] = worker.seat(),
                _ => {}
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

/// A block made: whether it was the last, and how long it took.
struct Made {
    last: bool,
    took: Took,
    /// Folding its stems into the outputs.
    fold: Duration,
}

impl Made {
    /// « worker 10.40 s, fold 0.05 s »; a remote's with its transfers.
    fn describe(&self) -> String {
        let Took { send, run, back } = self.took;
        let fold = secs(self.fold);
        if send.is_zero() && back.is_zero() {
            format!("worker {}, fold {fold}", secs(run))
        } else {
            format!(
                "send {}, worker {}, back {}, fold {fold}",
                secs(send),
                secs(run),
                secs(back)
            )
        }
    }
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
    /// more), added to the outputs.
    fn step(
        &mut self,
        worker: &mut Worker,
        inputs: &[PathBuf],
        held: u64,
        whole: bool,
    ) -> Result<Made> {
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
        let folding = Instant::now();
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
        Ok(Made {
            last: block.last,
            took: block.took,
            fold: folding.elapsed(),
        })
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
            let last = self.step(worker, &inputs, length, true)?.last;
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
    /// Set, the mock as a remote's `remote-worker` holds a block's stems
    /// back until it is asked for the next block of the element — or, for
    /// the last one, asked while another element's input is there, for a
    /// block of another element — and answers for that one only once they
    /// were fetched: they wait in named pipes, which a fetch made before
    /// the next request would wait on in vain.
    const MOCK_HOLD: &str = "OMNIFROG_MOCK_HOLD";
    /// A file the mock as a remote's `remote-worker` adds a line to for
    /// every block it is asked for — `ask <base> <block> sent <bases>`, the
    /// elements whose inputs are in its directory — and, held back
    /// ([`MOCK_HOLD`]), `fetched <base> <block> during <base> <block>`.
    const MOCK_LOG: &str = "OMNIFROG_MOCK_LOG";
    /// A file the mock as a remote's `remote-worker` waits for (a minute at
    /// most) at a block it is asked for while another element's input is
    /// there: the test's word that another machine took that element.
    const MOCK_WAIT_FOR: &str = "OMNIFROG_MOCK_WAIT_FOR";
    /// A file the mock waits for (a minute at most) before every whole
    /// separation but its first.
    const MOCK_SEP_WAIT: &str = "OMNIFROG_MOCK_SEP_WAIT";
    /// Set, the mock as a remote's `remote-worker` loses the voice of every
    /// element's last block once it is made: the fetch that comes for it
    /// fails, and the worker lives on.
    const MOCK_LOSE_LAST: &str = "OMNIFROG_MOCK_LOSE_LAST";
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
        let hang = std::env::var(MOCK_HANG).ok();
        let whole_only = std::env::var_os(MOCK_WHOLE_ONLY).is_some();
        let hold = std::env::var_os(MOCK_HOLD).is_some();
        let log = std::env::var_os(MOCK_LOG).map(PathBuf::from);
        let wait_for = std::env::var_os(MOCK_WAIT_FOR).map(PathBuf::from);
        let sep_wait = std::env::var_os(MOCK_SEP_WAIT).map(PathBuf::from);
        let lose_last = std::env::var_os(MOCK_LOSE_LAST).is_some();
        let mut separations = 0;
        let await_file = |flag: &Path| {
            let deadline = std::time::Instant::now() + Duration::from_secs(60);
            while !flag.exists() && std::time::Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(2));
            }
        };
        let note = |line: String| {
            if let Some(log) = &log
                && let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(log)
            {
                let _ = writeln!(file, "{line}");
            }
        };
        // As a remote's: the directory its inputs are sent to.
        let mut far_dir: Option<PathBuf> = None;
        // A block's stems held back: its element and block, the directory
        // and rate they are written at, the stems, and whether it is the
        // element's last.
        let mut kept_back: Option<(String, u64, String, u32, Stems, bool)> = None;
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
        // As a remote's worker: the seat the remote's own rules give its run
        // (`remote::seat`), under locks of its own, held until it ends.
        let mut second = false;
        let mut _seat = None;
        if args.iter().any(|arg| arg == "remote-worker") {
            if std::env::var_os(MOCK_BUSY).is_some() {
                say("BUSY another run has its GPU");
                return 0;
            }
            let mut dir = PathBuf::from(std::env::var_os(MOCK_DIR).expect("the mock's directory"));
            let run = args
                .iter()
                .position(|arg| arg == "--run")
                .and_then(|at| args.get(at + 1))
                .cloned();
            match remote::seat(&dir.with_extension("locks"), run.as_deref()) {
                Ok((lock, seat)) => {
                    second = remote::seat_of(&seat.to_string_lossy()) == Seat::Second;
                    _seat = Some(lock);
                }
                Err(why) => {
                    say(&format!("BUSY {why}"));
                    return 0;
                }
            }
            // The second seat has files of its own.
            if second {
                dir.as_mut_os_string().push("-beside");
            }
            for sub in ["in", "out"] {
                fs::create_dir_all(dir.join(sub)).unwrap();
            }
            say(&format!("DIR {}", dir.display()));
            far_dir = Some(dir);
        }
        let die = match second {
            true => std::env::var(MOCK_DIE_BESIDE).ok(),
            false => std::env::var(MOCK_DIE).ok(),
        };
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
                    separations += 1;
                    if let Some(flag) = &sep_wait
                        && separations > 1
                    {
                        await_file(flag);
                    }
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
                    // The elements whose inputs are there.
                    let mut sent: Vec<String> = far_dir
                        .as_ref()
                        .and_then(|dir| fs::read_dir(dir.join("in")).ok())
                        .map(|entries| {
                            entries
                                .filter_map(|e| e.ok())
                                .filter(|e| e.path().is_dir())
                                .map(|e| e.file_name().to_string_lossy().into_owned())
                                .collect()
                        })
                        .unwrap_or_default();
                    sent.sort();
                    let another = sent.iter().any(|other| other != base);
                    if far_dir.is_some() {
                        note(format!("ask {base} {k} sent {}", sent.join(",")));
                        if let Some(flag) = &wait_for
                            && another
                        {
                            await_file(flag);
                        }
                    }
                    // A block held back goes out once the next one is asked
                    // for — the next of its element, or for an element's last
                    // one, the first of another — which is answered once it
                    // went.
                    let serving = kept_back
                        .take()
                        .filter(|(b, j, _, _, _, last)| match last {
                            false => b == base && j + 1 == k,
                            true => b != base,
                        })
                        .map(|(b, j, out, rate, piece, _)| {
                            let (went, gone) = channel();
                            let what = format!("{b} {j}");
                            std::thread::spawn(move || {
                                write_stems(&out, &b, rate, &piece);
                                let _ = went.send(());
                            });
                            (what, gone)
                        });
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
                    if let Some((what, gone)) = serving {
                        let went = gone.recv_timeout(Duration::from_secs(60)).is_ok();
                        let fetched = if went { "fetched" } else { "not fetched" };
                        note(format!("{fetched} {what} during {base} {k}"));
                    }
                    // Its directory is the request's, made as the worker does.
                    fs::create_dir_all(out).map_err(|e| e.to_string())?;
                    let made = piece[0][0].len() as u64;
                    // The last block too, when another element was sent
                    // ahead: one comes after it.
                    if hold && (ready || another) {
                        for suffix in [VOICE_SUFFIX, REST_SUFFIX] {
                            let pipe = Path::new(out).join(format!("{base}{suffix}"));
                            let _ = fs::remove_file(&pipe);
                            let piped = Command::new("mkfifo").arg(&pipe).status();
                            if !piped.is_ok_and(|status| status.success()) {
                                return Err(format!("mkfifo {}", pipe.display()));
                            }
                        }
                        kept_back =
                            Some((base.to_string(), k, out.to_string(), rate, piece, !ready));
                    } else {
                        write_stems(out, base, rate, &piece);
                        if lose_last && far_dir.is_some() && !ready {
                            let voice = Path::new(out).join(format!("{base}{VOICE_SUFFIX}"));
                            fs::remove_file(&voice).map_err(|e| e.to_string())?;
                        }
                    }
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
            let start = |stop: Stop| Worker::start_command(mock(env), Some(stop), &quiet);
            self.separate_started(job, followed, beside, &start, progress)
        }

        /// The same, this machine's worker started by `start`.
        fn separate_started(
            &self,
            job: &Job,
            followed: Option<Followed>,
            beside: &Beside,
            start: super::super::Start,
            progress: Progress,
        ) -> super::super::Separated {
            let master = MasterSet::read(&job.extract_dir().join("vo.atmos")).unwrap();
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
                Event::Message { text } | Event::Trace { text } => {
                    said.0.lock().unwrap().push(text)
                }
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

    /// [`follow`], and with `gate`, this machine makes that many blocks for
    /// every chunk written ([`follow_with`]).
    #[allow(clippy::too_many_arguments)]
    fn follow_gated(
        bench: &Bench,
        film: &Film,
        pairs: Pairs,
        env: Vec<(&'static str, String)>,
        chunk: u64,
        pause: Duration,
        in_step: bool,
        gate: Option<&dyn Fn(usize) -> f64>,
    ) -> (f64, Outputs, Arc<Said>) {
        let ran = follow_with(
            bench,
            film,
            pairs,
            [env.clone(), env],
            chunk,
            pause,
            in_step,
            gate,
        );
        (ran.made[0], ran.outputs, ran.said)
    }

    /// [`follow`], each stage's worker on this machine started for its
    /// `envs`, the dub too when `film` has one. With `gate`, the machines
    /// make that many blocks for every chunk written, by its rank — on
    /// average, a share of one carried over to the next chunk — and the
    /// extraction waits for them (`in_step`).
    #[allow(clippy::too_many_arguments)]
    fn follow_with(
        bench: &Bench,
        film: &Film,
        pairs: Pairs,
        envs: [Vec<(&'static str, String)>; 2],
        chunk: u64,
        pause: Duration,
        in_step: bool,
        gate: Option<&dyn Fn(usize) -> f64>,
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
            let owed = std::cell::Cell::new(0.0);
            let chunks = std::cell::Cell::new(0);
            let caught_up = || {
                let seen: Vec<u64> = (0..devices).map(|d| follower.waits(d)).collect();
                if let Some(blocks) = gate {
                    let due = owed.get() + blocks(chunks.get());
                    chunks.set(chunks.get() + 1);
                    owed.set(due.fract());
                    follower.gate.grant(due.trunc() as u64);
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
        let mut handed = handed;
        let beside = handed
            .dub
            .as_mut()
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
    fn a_round_tells_a_machine_that_catches_up_from_one_that_does_not() {
        let mut trend = Trend::default();
        assert!(!trend.watching());
        // Three elements: a round is three blocks after the one that opens it.
        assert_eq!(trend.made(100, 3), Round::Opened);
        assert!(trend.watching());
        assert_eq!(trend.made(140, 3), Round::Going);
        assert_eq!(trend.made(120, 3), Round::Going);
        assert_eq!(
            trend.made(100, 3),
            Round::Stuck {
                was: 100,
                rounds: 1
            }
        );
        // Less than when it opened, though it went up meanwhile.
        assert_eq!(trend.made(150, 3), Round::Going);
        assert_eq!(trend.made(110, 3), Round::Going);
        assert_eq!(trend.made(99, 3), Round::Shrank { was: 100 });
        // The count of rounds in a row starts again.
        for _ in 0..2 {
            trend.made(120, 3);
        }
        assert_eq!(trend.made(99, 3), Round::Stuck { was: 99, rounds: 1 });
        for _ in 0..2 {
            trend.made(120, 3);
        }
        assert_eq!(trend.made(130, 3), Round::Stuck { was: 99, rounds: 2 });
        // Two in a row let one go; then one is enough, until it catches up.
        assert_eq!(trend.needed(), STUCK);
        trend.shed();
        assert!(!trend.watching());
        assert_eq!(trend.needed(), 1);
        assert_eq!(trend.made(130, 2), Round::Opened);
        assert_eq!(Trend::default().needed(), STUCK);
    }

    /// **A dead worker here still hands on its share of the lock.** A GPU
    /// reset kills this machine's dub worker before the next step: the
    /// share it holds is this machine's lock until it is dropped, and the
    /// next step's worker shares it rather than find its own run in the way.
    #[test]
    fn a_dead_worker_here_still_hands_on_its_share_of_this_machine_s_lock() {
        let bench = Bench::new("dead-share");
        let work = bench.root.join("lock");
        let lock = Lock::take(&work).unwrap();
        let mut worker = Worker::start_command(mock(&[]), None, &quiet)
            .unwrap()
            .holding(lock);
        worker.kill_now();
        assert!(!worker.is_alive());
        let mut followed = Followed {
            stage: Stage::Dub,
            machines: vec![Machine {
                device: 0,
                worker: Some(worker),
                streams: Vec::new(),
            }],
            plan: Plan::empty(0),
            identity: Vec::new(),
            dir: bench.root.join("followed"),
        };
        let beside = followed.beside();
        assert!(beside.lock().is_some());
        // Nobody else gets it meanwhile; once both let go, anyone does.
        assert!(Lock::take(&work).is_err());
        drop(beside);
        drop(followed);
        drop(crate::worker::soon(|| Lock::take(&work)).unwrap());
    }

    #[test]
    fn the_dub_has_a_gap_with_room_to_spare_and_is_kept_half_made() {
        // A block of 100 frames needs 125 before the original's next.
        assert!(dub_fits(100, 125));
        assert!(!dub_fits(100, 124));
        assert!(!dub_fits(1, 0));
        assert!(dub_fits(0, 0));
        // Half of a channel's blocks made: finished by that machine.
        assert!(dub_kept(5, 10));
        assert!(!dub_kept(4, 10));
        assert!(dub_kept(6, 11));
        assert!(!dub_kept(5, 11));
        // Nothing made: never kept.
        assert!(!dub_kept(0, 0));
    }

    #[test]
    fn half_made_is_kept_unless_another_machine_would_be_done_sooner() {
        let whole = 1_000;
        // Under half: let go.
        assert!(!kept_half_made(499, whole, 5_000, false));
        // Half made: kept, alone or not, while what is left is less than an
        // element whole.
        assert!(kept_half_made(500, whole, 5_000, false));
        assert!(kept_half_made(500, whole, 999, true));
        // More left than an element whole, and another machine to take it.
        assert!(!kept_half_made(500, whole, 1_000, true));
        // The film's length unknown: nothing is kept for it.
        assert!(!kept_half_made(500, 0, 0, false));
    }

    /// The lines the run said that `needle` is in, in order.
    fn lines_in<'s>(said: &'s [String], needle: &str) -> Vec<&'s String> {
        said.iter().filter(|line| line.contains(needle)).collect()
    }

    /// **A burst the machine catches up on loses nothing.** The film comes
    /// four times as fast as this machine separates it for three blocks of
    /// it: the elements served last fall two and three blocks behind, the
    /// point at which the follower used to let one go every block made. It
    /// catches up once the burst is over, and keeps them all.
    #[test]
    fn a_burst_the_machine_catches_up_on_loses_nothing() {
        let bench = Bench::new("burst");
        let film = Film::objects(4, 160_000);
        let whole = bench.whole(&film, Pairs::default());
        // A chunk is a quarter of a block of each of the four elements: a
        // block of work. Twice what comes, but a quarter of it from the
        // fifth block of film to the eighth.
        let blocks = |n: usize| if (20..32).contains(&n) { 0.25 } else { 2.0 };
        let (made, followed, said) = follow_gated(
            &bench,
            &film,
            Pairs::default(),
            Vec::new(),
            STEP / 4,
            Duration::ZERO,
            true,
            Some(&blocks),
        );
        let lines = said.0.lock().unwrap().clone();
        let all = lines.join("\n");
        assert_eq!(
            lines_in(&lines, ": behind: 13 is 2 blocks behind").len(),
            1,
            "{all}"
        );
        assert!(
            !lines_in(&lines, "not catching up, 1 round of 2").is_empty(),
            "{all}"
        );
        assert!(
            !lines_in(&lines, "catching up, all kept").is_empty(),
            "{all}"
        );
        assert!(!lines_in(&lines, ": caught up").is_empty(), "{all}");
        assert!(lines_in(&lines, "let go").is_empty(), "{all}");
        assert!(said.any("4 of the original's 4 elements followed"), "{all}");
        ends_on_what_was_made(&said, made, 4);
        same(&followed, &whole);
    }

    /// **A machine that does not keep up lets one go after two rounds.**
    /// Fast enough to take every element at first, it then separates three
    /// fifths of what comes: the last two are let go, one at a time — the
    /// first after two rounds behind in a row, the next after one more —
    /// not one every block made.
    #[test]
    fn a_machine_that_falls_behind_lets_the_last_elements_go_a_round_apart() {
        let bench = Bench::new("behind");
        let film = Film::objects(4, 160_000);
        let whole = bench.whole(&film, Pairs::default());
        let blocks = |n: usize| if n < 9 { 3.0 } else { 0.6 };
        let (made, followed, said) = follow_gated(
            &bench,
            &film,
            Pairs::default(),
            Vec::new(),
            STEP / 4,
            Duration::ZERO,
            true,
            Some(&blocks),
        );
        let lines = said.0.lock().unwrap().clone();
        let all = lines.join("\n");
        let let_go: Vec<String> = lines_in(&lines, "does not keep up with the extraction")
            .into_iter()
            .cloned()
            .collect();
        assert_eq!(let_go.len(), 2, "{all}");
        assert!(let_go[0].contains("  13 let go"), "{all}");
        assert!(let_go[1].contains("  12 let go"), "{all}");
        // Between being behind and the first let go, two rounds of as many
        // blocks as elements followed; then one round for the next.
        let mut since = lines.iter().position(|l| l.contains(": behind: ")).unwrap();
        for (gone, rounds) in let_go.iter().zip([STUCK * 4, 3]) {
            let at = lines.iter().position(|l| l == gone).unwrap();
            let blocks = lines[since..at]
                .iter()
                .filter(|l| l.contains(" block ") && l.contains(" of film written"))
                .count();
            assert!(blocks >= rounds, "{blocks} blocks\n{all}");
            since = at;
        }
        assert!(
            said.any("not catching up, 1 rounds in a row: 12 let go"),
            "{all}"
        );
        // What was made of an element let go is no longer counted.
        ends_on_what_was_made(&said, made, 4);
        same(&followed, &whole);
    }

    /// **Half made, an element is kept.** The machine keeps up through the
    /// first three fifths of the film, then falls behind for good: what is
    /// left of every element is finished after the extraction, rather than
    /// made again whole.
    #[test]
    fn an_element_half_made_is_kept_however_far_behind() {
        let bench = Bench::new("half");
        let film = Film::objects(4, 160_000);
        let whole = bench.whole(&film, Pairs::default());
        let blocks = |n: usize| if n < 48 { 2.0 } else { 0.4 };
        let (made, followed, said) = follow_gated(
            &bench,
            &film,
            Pairs::default(),
            Vec::new(),
            STEP / 4,
            Duration::ZERO,
            true,
            Some(&blocks),
        );
        let all = said.lines();
        assert!(
            said.any("rounds in a row, but 13 is half made: kept"),
            "{all}"
        );
        assert!(!said.any("let go"), "{all}");
        assert!(made < 3.5, "{made}: it did not fall behind\n{all}");
        ends_on_what_was_made(&said, made, 4);
        same(&followed, &whole);
    }

    /// **The trace says what each request took.** Every block on this
    /// machine, every element taken, and the end of the decode.
    #[test]
    fn the_trace_says_every_block_and_what_it_took() {
        let bench = Bench::new("trace");
        let film = Film::objects(3, 60_000);
        let (_, _, said) = follow(
            &bench,
            &film,
            Pairs::default(),
            Vec::new(),
            2_000,
            Duration::from_millis(2),
            true,
        );
        let lines = said.0.lock().unwrap().clone();
        let all = lines.join("\n");
        for element in ["10", "11", "12"] {
            assert!(
                said.any(&format!("  follow, this machine: takes {element}, ")),
                "{all}"
            );
            assert!(
                said.any(&format!("  follow, this machine: {element} block 0 in ")),
                "{all}"
            );
        }
        assert!(said.any(" (worker "), "{all}");
        assert!(said.any(", fold "), "{all}");
        // What it waited for while the film came: the first block of each.
        assert!(said.any("  follow, this machine: idle "), "{all}");
        assert!(said.any(", waiting for 10 block 0, which needs "), "{all}");
        assert!(
            said.any("this machine: the decode is over: 3 followed, "),
            "{all}"
        );
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

    /// A machine held to `per_chunk(n)` blocks for the `n`-th `chunk` of
    /// frames the extraction writes follows the original's four elements,
    /// and a 5.1 dub when `dubbed`: what it said, after checking that
    /// everything came out as whole.
    fn held_to(
        name: &str,
        chunk: u64,
        per_chunk: &dyn Fn(usize) -> f64,
        dubbed: bool,
    ) -> Arc<Said> {
        let bench = Bench::new(name);
        let mut film = Film::objects(4, 160_000);
        if dubbed {
            film = film.dubbed();
        }
        let whole = bench.whole(&film, Pairs::default());
        let ran = follow_with(
            &bench,
            &film,
            Pairs::default(),
            [Vec::new(), Vec::new()],
            chunk,
            Duration::ZERO,
            true,
            Some(per_chunk),
        );
        same(&ran.outputs, &whole);
        ran.said
    }

    /// What the follower said of the original, at the end of the
    /// extraction, and what of it it let go.
    fn the_original_s(said: &Said) -> (Vec<String>, Vec<String>) {
        let let_go = lines_with(said, " let go: ")
            .into_iter()
            .filter(|line| {
                !DUB.iter()
                    .any(|name| line.contains(&format!(" {name} let go")))
            })
            .collect();
        (
            lines_with(said, "of the original's 4 elements followed"),
            let_go,
        )
    }

    /// **The dub's line outlives the original's.** Eight elements and a
    /// machine slow through the first two blocks of film: the original's
    /// line closes with elements still in it. The machine is then far
    /// faster than the original needs, and the dub takes its channels in
    /// those gaps, though the film is past the original's two blocks.
    #[test]
    fn the_dub_takes_its_channels_after_the_original_s_line_closed() {
        let bench = Bench::new("dub-late");
        let film = Film::objects(8, 160_000).dubbed();
        let whole = bench.whole(&film, Pairs::default());
        let per_chunk = |n: usize| if n < 8 { 1.0 } else { 6.0 };
        let ran = follow_with(
            &bench,
            &film,
            Pairs::default(),
            [Vec::new(), Vec::new()],
            STEP / 4,
            Duration::ZERO,
            true,
            Some(&per_chunk),
        );
        let said = &ran.said;
        assert!(
            said.any("  follow, this machine: takes no more: "),
            "{}",
            said.lines()
        );
        assert!(
            said.any("  follow, this machine: takes the dub's "),
            "{}",
            said.lines()
        );
        assert!(ran.made[1] > 1.0, "{:?}\n{}", ran.made, said.lines());
        same(&ran.outputs, &whole);
    }

    /// **The dub has the gaps the original leaves.** A machine that makes
    /// eight blocks while the extraction writes one keeps the original's
    /// four elements and follows some of the dub in the time left; nothing
    /// is let go.
    #[test]
    fn the_dub_has_the_gaps_the_original_leaves() {
        let said = held_to("dub-gaps", STEP / 4, &|_| 2.0, true);
        assert!(
            said.any("during the extraction: 4 of the original's 4 elements followed"),
            "{}",
            said.lines()
        );
        assert!(
            lines_with(&said, " let go: ").is_empty(),
            "{}",
            said.lines()
        );
        assert!(
            lines_with(&said, "of the dub's 5 channels followed").len() == 1,
            "{}",
            said.lines()
        );
    }

    /// **The dub never holds the original back.** The extraction does not
    /// write at one rate: where the video is lighter the film comes faster,
    /// and a machine that keeps up with the original with a fifth to spare
    /// keeps up with none of it there. Seen on a film: the dub, taken at the
    /// start and served whenever the original had nothing ready, left the
    /// original no slack for those stretches, and both machines let one
    /// element of it go after another. The original must fare the same with
    /// the dub beside it as alone.
    #[test]
    fn the_dub_never_holds_the_original_back() {
        // About five blocks for every one of film, three in a stretch of
        // two blocks of film that come twice as fast.
        let bursty = |n: usize| if (32..48).contains(&n) { 0.3 } else { 0.6 };
        let alone = held_to("dub-tight-alone", STEP / 8, &bursty, false);
        let beside = held_to("dub-tight", STEP / 8, &bursty, true);
        let (followed, let_go) = the_original_s(&alone);
        assert_eq!(followed.len(), 1, "{}", alone.lines());
        assert_eq!(
            the_original_s(&beside),
            (followed, let_go),
            "alone:\n{}\n\nbeside the dub:\n{}",
            alone.lines(),
            beside.lines()
        );
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
        // The pipeline does not hand the follower the dub in pairs.
        let mut config = bench.config.clone();
        config.inference.engine = Engine::Pymss;
        assert!(follows(&config) && !follows_dub(&config, pairs));
        assert!(follows_dub(&config, Pairs::default()));
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
        // And handed it anyway, the follower leaves it to its step.
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
        // A remote's blocks are traced with its transfers.
        let traced = said.0.lock().unwrap().iter().any(|l| {
            l.contains(&format!("  follow, {far}: 11 block 0 in "))
                && l.contains(" (send ")
                && l.contains(", back ")
        });
        assert!(traced, "{}", said.lines());
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

    /// The original of `film`, extracted, then separated with the bench's
    /// remotes beside this machine, whose worker (the mock for `here`)
    /// starts once `until` holds of what the run said, a minute at most:
    /// what each machine makes does not depend on how fast it is. `watch`
    /// sees what the run said as it is said.
    #[cfg(unix)]
    fn beside_remotes(
        bench: &Bench,
        film: &Film,
        here: &[(&str, String)],
        until: impl Fn(&Said) -> bool + Sync,
        watch: impl Fn(&Said) + Sync,
    ) -> (Outputs, Arc<Said>) {
        let job = bench.job("beside", Pairs::default());
        film.extracted(&job.extract_dir());
        let said = Arc::new(Said::default());
        let record = said.progress();
        let progress = |event: Event| {
            record(event);
            watch(&said);
        };
        let start = |stop: Stop| {
            let deadline = std::time::Instant::now() + Duration::from_secs(60);
            while !until(&said) && !stop.raised() && std::time::Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(2));
            }
            Worker::start_command(mock(here), Some(stop), &quiet)
        };
        bench.separate_started(&job, None, &Beside::default(), &start, &progress);
        (Bench::outputs(&job), said)
    }

    /// The passes the tests' remote made, by what it said.
    #[cfg(unix)]
    fn made_there(said: &Said) -> usize {
        lines_with(said, &format!("[{}] ", remote::SHELL))
            .iter()
            .filter(|line| line.contains(" done in "))
            .count()
    }

    /// The blocks the mock as a remote was asked for, in order: the
    /// element, the block, and the elements whose inputs were there.
    #[cfg(unix)]
    fn asked(log: &str) -> Vec<(String, u64, Vec<String>)> {
        log.lines()
            .filter_map(|line| {
                let mut words = line.strip_prefix("ask ")?.split(' ');
                let base = words.next()?.to_string();
                let k = words.next()?.parse().ok()?;
                let sent = match (words.next(), words.next()) {
                    (Some("sent"), Some(list)) => list
                        .split(',')
                        .filter(|b| !b.is_empty())
                        .map(str::to_string)
                        .collect(),
                    _ => Vec::new(),
                };
                Some((base, k, sent))
            })
            .collect()
    }

    /// **A remote's whole passes, a block at a time.** After the extraction
    /// the remote makes three of the four elements, and its GPU never waits
    /// on a transfer. Each block's stems are fetched while it makes the
    /// next, an element's last ones while it makes the next element's first
    /// block: the mock holds them back until it is asked for that one, and
    /// answers for it only once they went. The input of each element after
    /// its first is there before that element's first block is asked for:
    /// sent while the one before was made. All comes out as separated whole
    /// here, and nothing is left there.
    #[cfg(unix)]
    #[test]
    fn a_remote_fetches_each_block_while_it_makes_the_next_and_has_its_next_input_first() {
        let mut bench = Bench::new("far-blocks");
        let film = Film::objects(4, 60_000);
        let whole = bench.whole(&film, Pairs::default());
        let log = bench.root.join("far.log");
        bench.config.remotes = vec![far_mock(
            &bench,
            &[
                (MOCK_HOLD, "1".to_string()),
                (MOCK_LOG, log.to_string_lossy().into_owned()),
            ],
        )];
        // This machine starts once the remote made all but the element it
        // took first.
        let (outputs, said) =
            beside_remotes(&bench, &film, &[], |said| made_there(said) >= 3, |_| {});
        same(&outputs, &whole);
        assert_eq!(made_there(&said), 3, "{}", said.lines());
        assert!(!said.any("instead"), "{}", said.lines());
        let log = fs::read_to_string(&log).unwrap();
        let asked = asked(&log);
        let bases: Vec<&String> = asked
            .iter()
            .filter(|(_, k, _)| *k == 0)
            .map(|(base, ..)| base)
            .collect();
        assert_eq!(bases.len(), 3, "{log}");
        let blocks = Grid {
            step: STEP,
            ahead: AHEAD,
        }
        .blocks(film.frames);
        assert_eq!(blocks, 8);
        assert_eq!(asked.len() as u64, 3 * blocks, "{log}");
        for base in &bases {
            for k in 1..blocks {
                assert!(
                    log.contains(&format!("fetched {base} {} during {base} {k}\n", k - 1)),
                    "block {} of {base} was not fetched while {k} was made:\n{log}",
                    k - 1
                );
            }
        }
        assert!(!log.contains("not fetched"), "{log}");
        for pair in bases.windows(2) {
            let (before, next) = (pair[0], pair[1]);
            // Its GPU went on to the next at once.
            assert!(
                log.contains(&format!(
                    "fetched {before} {} during {next} 0\n",
                    blocks - 1
                )),
                "the last block of {before} was not fetched while {next} began:\n{log}"
            );
            let first = asked
                .iter()
                .position(|(base, k, _)| base == next && *k == 0)
                .unwrap();
            assert!(
                asked[..first]
                    .iter()
                    .any(|(base, _, sent)| base == before && sent.contains(next)),
                "{next} was not sent while {before} was made:\n{log}"
            );
        }
        assert_eq!(left_there(&bench, "in"), Vec::<String>::new(), "{log}");
        assert_eq!(left_there(&bench, "out"), Vec::<String>::new(), "{log}");
    }

    /// **A remote's pairs, and the centre made twice.** The dub in pairs,
    /// its centre beside its left and its right: the remote makes two of
    /// the three passes, a block at a time. Each side of a pair's stems is
    /// a channel's, and the centre is the average of two estimates, kept
    /// from passes made on both machines or on the remote alone: every
    /// voice is the one this machine writes alone.
    #[cfg(unix)]
    #[test]
    fn a_remote_s_pairs_and_the_centre_made_twice_come_out_as_here() {
        let mut bench = Bench::new("far-pairs");
        let names = ["L", "R", "C", "LFE", "Lss", "Rss"];
        let channels: Vec<String> = names.iter().map(|n| n.to_string()).collect();
        let dub = |bench: &Bench, job: &str, start: super::super::Start, progress: Progress| {
            let job = bench.job(job, Pairs::ALL);
            let dir = job.extract_dir();
            fs::create_dir_all(&dir).unwrap();
            for (i, name) in names.iter().enumerate() {
                let mut bytes = Film::wav_header(60_000);
                bytes.extend(Film::bytes(40 + i as u32, 0, 60_000));
                fs::write(dir.join(format!("{}{name}.wav", extract::DUB)), bytes).unwrap();
            }
            super::super::dub_with(
                &bench.config,
                &job,
                &bench.model,
                &channels,
                Pairs::ALL,
                None,
                Some(start),
                progress,
            )
            .unwrap();
            Bench::outputs(&job)
        };
        let now = |stop: Stop| Worker::start_command(mock(&[]), Some(stop), &quiet);
        let alone = dub(&bench, "here", &now, &quiet);
        assert!(alone.iter().any(|(name, _)| name == "voix_C.wav"));
        bench.config.remotes = vec![far_mock(&bench, &[])];
        let said = Arc::new(Said::default());
        let progress = said.progress();
        let late = |stop: Stop| {
            let deadline = std::time::Instant::now() + Duration::from_secs(60);
            while made_there(&said) < 2 && !stop.raised() && std::time::Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(2));
            }
            Worker::start_command(mock(&[]), Some(stop), &quiet)
        };
        let beside = dub(&bench, "beside", &late, &progress);
        same(&beside, &alone);
        assert_eq!(made_there(&said), 2, "{}", said.lines());
        assert!(!said.any("instead"), "{}", said.lines());
        assert_eq!(left_there(&bench, "in"), Vec::<String>::new());
        assert_eq!(left_there(&bench, "out"), Vec::<String>::new());
    }

    /// **A remote that refuses a block of a whole pass.** Its worker lives:
    /// what it holds of the element goes, and it makes the element whole in
    /// one request, as it did before — which is where an element out of
    /// device memory goes through MSST. Nothing counts as a failure, and
    /// nothing is left there.
    #[cfg(unix)]
    #[test]
    fn a_remote_that_refuses_a_block_makes_the_element_whole_instead() {
        let mut bench = Bench::new("far-refuses");
        let film = Film::objects(4, 60_000);
        let whole = bench.whole(&film, Pairs::default());
        bench.config.remotes = vec![far_mock(&bench, &[(MOCK_FAIL, "*:3".to_string())])];
        let (outputs, said) =
            beside_remotes(&bench, &film, &[], |said| made_there(said) >= 3, |_| {});
        same(&outputs, &whole);
        assert_eq!(made_there(&said), 3, "{}", said.lines());
        assert_eq!(
            lines_with(&said, "the mock fails here; separated whole there instead").len(),
            3,
            "{}",
            said.lines()
        );
        assert!(!said.any("made here instead"), "{}", said.lines());
        assert_eq!(left_there(&bench, "in"), Vec::<String>::new());
        assert_eq!(left_there(&bench, "out"), Vec::<String>::new());
    }

    /// **A remote whose worker dies under a block of a whole pass**, and
    /// again under the same block once started again: the element is made
    /// here, and the remote is left out of the rest of the step.
    #[cfg(unix)]
    #[test]
    fn a_remote_that_dies_under_a_block_twice_is_left_out() {
        let mut bench = Bench::new("far-dies-whole");
        let film = Film::objects(4, 60_000);
        let whole = bench.whole(&film, Pairs::default());
        bench.config.remotes = vec![far_mock(&bench, &[(MOCK_DIE, "*:3".to_string())])];
        let left_out = format!("{}: left out for the rest of this step", remote::SHELL);
        let (outputs, said) =
            beside_remotes(&bench, &film, &[], |said| said.any(&left_out), |_| {});
        same(&outputs, &whole);
        assert!(said.any(&left_out), "{}", said.lines());
        assert_eq!(
            lines_with(&said, "failed, made here instead").len(),
            2,
            "{}",
            said.lines()
        );
        assert_eq!(made_there(&said), 0, "{}", said.lines());
    }

    /// **A remote whose elements' last stems never come back.** Each element
    /// it makes is claimed and set aside, its GPU on to the next, and the
    /// fetch of its last block then fails: a failure of the remote's, as a
    /// pass's, which the next element made meanwhile does not undo — only
    /// outputs written do. Two in a row and it is left out of the rest of
    /// the step; the element it had set aside by then fails too. Every
    /// element is made here, as this machine makes it alone.
    #[cfg(unix)]
    #[test]
    fn a_remote_whose_last_stems_fail_twice_in_a_row_is_left_out() {
        let mut bench = Bench::new("far-tails-fail");
        let film = Film::objects(4, 60_000);
        let whole = bench.whole(&film, Pairs::default());
        bench.config.remotes = vec![far_mock(&bench, &[(MOCK_LOSE_LAST, "1".to_string())])];
        let left_out = format!("{}: left out for the rest of this step", remote::SHELL);
        // This machine starts once the remote is left out.
        let (outputs, said) =
            beside_remotes(&bench, &film, &[], |said| said.any(&left_out), |_| {});
        same(&outputs, &whole);
        assert!(said.any(&left_out), "{}", said.lines());
        assert_eq!(
            lines_with(&said, "failed, made here instead").len(),
            3,
            "{}",
            said.lines()
        );
        assert_eq!(made_there(&said), 0, "{}", said.lines());
    }

    /// **A remote that hangs on a block of a whole pass.** This machine,
    /// with nothing else left, makes that element again beside it and keeps
    /// its own; the remote's worker is killed under the block, its element
    /// dropped, rather than waited for.
    #[cfg(unix)]
    #[test]
    fn a_remote_that_hangs_on_a_block_is_overtaken_here() {
        let mut bench = Bench::new("far-hangs-whole");
        let film = Film::objects(4, 60_000);
        let whole = bench.whole(&film, Pairs::default());
        bench.config.remotes = vec![far_mock(&bench, &[(MOCK_HANG, "*:2".to_string())])];
        let far = remote::SHELL;
        let started = std::time::Instant::now();
        // This machine starts once the remote is on an element.
        let (outputs, said) = beside_remotes(
            &bench,
            &film,
            &[],
            |said| said.any(&format!("[{far}] element ")),
            |_| {},
        );
        same(&outputs, &whole);
        assert!(
            said.any(&format!(
                "again, beside {far}: the first to finish keeps it"
            )),
            "{}",
            said.lines()
        );
        assert!(
            lines_with(
                &said,
                "made first by another machine; this one's is dropped"
            )
            .iter()
            .any(|line| line.contains(&format!("[{far}] "))),
            "{}",
            said.lines()
        );
        assert_eq!(made_there(&said), 0, "{}", said.lines());
        // Not the hour a worker is given between two words.
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "{:?}",
            started.elapsed()
        );
    }

    /// **The input sent ahead of an element another machine took.** The
    /// remote sends the next in line ahead while it makes its first; this
    /// machine takes that one first (the remote's last block waits until it
    /// has, and this machine's next separation until the remote is on
    /// another element). The remote takes the one after, and what it was
    /// sent of the other goes from there before it says so.
    #[cfg(unix)]
    #[test]
    fn an_input_sent_ahead_of_an_element_taken_here_goes_from_the_remote() {
        let mut bench = Bench::new("far-ahead");
        let film = Film::objects(4, 60_000);
        let whole = bench.whole(&film, Pairs::default());
        let log = bench.root.join("far.log");
        let taken_here = bench.root.join("taken-here");
        let on_another = bench.root.join("on-another");
        let path = |p: &Path| p.to_string_lossy().into_owned();
        bench.config.remotes = vec![far_mock(
            &bench,
            &[(MOCK_LOG, path(&log)), (MOCK_WAIT_FOR, path(&taken_here))],
        )];
        let far = remote::SHELL;
        let sent_ahead = || {
            asked(&fs::read_to_string(&log).unwrap_or_default())
                .iter()
                .any(|(_, _, sent)| sent.len() > 1)
        };
        let (outputs, said) = beside_remotes(
            &bench,
            &film,
            &[(MOCK_SEP_WAIT, path(&on_another))],
            |_| sent_ahead(),
            |said| {
                if lines_with(said, "[local] element ").len() >= 2 {
                    let _ = fs::write(&taken_here, b"");
                }
                if lines_with(said, &format!("[{far}] element ")).len() >= 2 {
                    let _ = fs::write(&on_another, b"");
                }
            },
        );
        same(&outputs, &whole);
        let log = fs::read_to_string(&log).unwrap();
        let asked = asked(&log);
        let first = asked[0].0.clone();
        let ahead = asked
            .iter()
            .find_map(|(_, _, sent)| sent.iter().find(|base| **base != first))
            .unwrap_or_else(|| panic!("nothing was sent ahead:\n{log}"))
            .clone();
        // Taken here, it was never asked for there, and its input went.
        assert!(
            lines_with(&said, "[local] element ")
                .iter()
                .any(|line| line.ends_with(&format!("({ahead})"))),
            "{}",
            said.lines()
        );
        assert!(!asked.iter().any(|(base, ..)| *base == ahead), "{log}");
        assert!(!left_there(&bench, "in").contains(&ahead), "{log}");
        // The remote went on to another one, sent nothing of it ahead.
        let then = lines_with(&said, &format!("[{far}] element "));
        assert_eq!(then.len(), 2, "{}", said.lines());
        assert!(
            !then
                .iter()
                .any(|line| line.ends_with(&format!("({ahead})"))),
            "{}",
            said.lines()
        );
        assert!(
            asked
                .iter()
                .filter(|(base, ..)| *base != first)
                .all(|(_, _, sent)| !sent.contains(&ahead)),
            "{log}"
        );
    }
}
