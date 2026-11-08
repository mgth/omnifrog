//! The original separated while the extraction is still writing it.
//!
//! The extraction reads the whole container and is bound by the disk: a
//! quarter of an hour for a film, during which the GPU has nothing to do.
//! Every waveform grows in step with the others, so no element is whole
//! before the very end; but the worker cuts an element into blocks anyway
//! (`block_seconds`), and a block only needs the frames it covers and a
//! window past them. So the follower hands this machine's worker one block
//! at a time, as the decoder writes them, and keeps what comes back.
//!
//! **As many as it keeps up with.** It starts on every element and serves
//! them in order, the first one with a block ready first. The GPU is faster
//! than the disk, but not thirteen times: the last elements fall behind, and
//! one that is [`BEHIND`] blocks behind is let go — the next one, if the
//! others still do not keep up, once another block is made. What is let go
//! is separated whole after the extraction, by any machine, like the
//! elements nobody followed. Only this machine follows: a remote would need
//! every block carried there, and is handed whole elements as before.
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
//! **Handed to the next step.** When the decode ends, what each element
//! still needs is its last block or two, and the fade the worker keeps for
//! it: the worker and what it followed go to the separation of the original
//! ([`Followed`]), which finishes them first while the remotes start on the
//! rest. The model is loaded once for both steps.
//!
//! **Nothing half-made survives.** A decode that fails or is stopped kills
//! the worker at once and drops everything (the extraction runs again from
//! the beginning anyway); a run that ends before the separation does too.
//! Nothing a follower writes is under an output's name until it is whole.
//!
//! Not followed: the original under MSST, which separates whole elements
//! only, or on a CPU, which the decoder needs; the original in pairs when it
//! has objects, whose pairs are drawn from where each one is over the whole
//! film; and the dub, whose model is another one.

use super::{Form, Plan, Stage, Start, Stem, Unit, discard, elements_of, pass_name, plan_original};
use crate::config::{Config, Engine, Pairs};
use crate::extract;
use crate::job::Job;
use crate::master::MasterSet;
use crate::models::{self, Model};
use crate::progress::{Event, Progress};
use crate::run::Stop;
use crate::worker::{Ask, Grid, REST_SUFFIX, VOICE_SUFFIX, Worker};
use crate::{Error, Result, wav};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, SystemTime};

/// Where a follower keeps what it grows, in the separation's directory.
pub(super) const DIR: &str = "_follow_element";
/// Blocks a followed element may be behind the extraction before the last
/// of the followed is let go.
const BEHIND: u64 = 2;
/// How much older than the follower a master set's header may look and
/// still be this extraction's: some file systems keep coarse times.
const FRESH: Duration = Duration::from_secs(2);

/// Whether this configuration separates the original during the extraction:
/// under pymss, which cuts an element into blocks.
pub fn follows(config: &Config) -> bool {
    config.inference.engine == Engine::Pymss
}

/// Follows the original's elements as the extraction writes them: one
/// thread runs [`Follower::follow`] while the extraction runs, and the
/// extraction calls [`Follower::decoded`] once the decoder is done.
pub struct Follower<'a> {
    config: &'a Config,
    job: &'a Job,
    model: &'a Model,
    pairs: Pairs,
    /// The tests' mock worker; the inference otherwise.
    start: Option<Start<'a>>,
    /// How often the decoder's files are looked at.
    poll: Duration,
    /// Raised, the worker is killed wherever it is.
    stop: Stop,
    end: Mutex<End>,
    changed: Condvar,
    /// The times it found nothing to do: the tests' extraction keeps in step
    /// with it.
    #[cfg(test)]
    waits: std::sync::atomic::AtomicU64,
}

#[derive(Default)]
struct End {
    /// The decode is over, and whether it succeeded.
    decoded: Option<bool>,
    /// The follower makes no more requests.
    quiet: bool,
}

impl<'a> Follower<'a> {
    pub fn new(config: &'a Config, job: &'a Job, model: &'a Model, pairs: Pairs) -> Self {
        Self {
            config,
            job,
            model,
            pairs,
            start: None,
            poll: extract::POLL,
            stop: Stop::new(),
            end: Mutex::new(End::default()),
            changed: Condvar::new(),
            #[cfg(test)]
            waits: Default::default(),
        }
    }

    fn end(&self) -> MutexGuard<'_, End> {
        self.end.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn decode_ended(&self) -> Option<bool> {
        self.end().decoded
    }

    /// Nothing to do: up to a poll, or until the decode ends.
    fn wait(&self) {
        #[cfg(test)]
        self.waits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
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
    /// makes no more requests — after the block in flight, which is worth
    /// having, or at once when the decode failed: its worker is killed and
    /// nothing it made is kept. Called again, it only waits.
    pub fn decoded(&self, ok: bool) {
        let mut end = self.end();
        if end.decoded.is_none() {
            end.decoded = Some(ok);
            if !ok {
                self.stop.raise();
            }
            self.changed.notify_all();
        }
        while !end.quiet {
            end = self.changed.wait(end).unwrap_or_else(|p| p.into_inner());
        }
    }

    /// Follow the original until the decode ends. What was followed, with
    /// the worker that followed it, when the decode succeeded; `None` when
    /// it failed, or when nothing could be followed.
    pub fn follow(&self, progress: Progress) -> Option<Followed> {
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
            Ok(followed) if self.decode_ended() == Some(true) => followed,
            Ok(_) => None,
            Err(why) => {
                if self.decode_ended() != Some(false) {
                    progress(Event::message(format!(
                        "the original is separated after the extraction: {why}"
                    )));
                }
                None
            }
        }
    }

    fn run(&self, progress: Progress) -> Result<Option<Followed>> {
        let raw = self.job.extract_dir();
        let dir = self.job.separate_dir().join(DIR);
        // What a run that stopped left.
        discard(&dir)?;
        let since = SystemTime::now()
            .checked_sub(FRESH)
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let header = raw.join(extract::MASTER).with_extension("atmos");
        // The decoder writes the header as soon as it knows the elements:
        // seconds into the film. One left by an earlier run is not it.
        let master = loop {
            if self.decode_ended().is_some() {
                return Ok(None);
            }
            let fresh = fs::metadata(&header)
                .and_then(|m| m.modified())
                .is_ok_and(|at| at >= since);
            if fresh && let Ok(master) = MasterSet::read(&header) {
                break master;
            }
            self.wait();
        };
        let elements = elements_of(&master);
        let units: Vec<Unit> = elements
            .iter()
            .map(|id| Unit::of(self.job, Stage::Original, &id.to_string()))
            .collect();
        let Some(plan) = plan_original(&master, &elements, &units, self.pairs, None) else {
            progress(Event::message(
                "the original's elements are paired by where each one is over the whole film, \
                 known once it is read: they are separated after the extraction",
            ));
            return Ok(None);
        };
        // A previous run's stems for these parameters are the separation's
        // to check, at their length.
        let wanted: Vec<usize> = (0..plan.passes.len())
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
            return Ok(None);
        }

        let worker = match self.start {
            Some(start) => start(self.stop.clone())?,
            None => {
                let local = models::fetch(self.model, self.config, progress)?;
                Worker::start_under(
                    self.config,
                    self.model,
                    &local,
                    progress,
                    Some(self.stop.clone()),
                )?
            }
        };
        let mut followed = Followed {
            worker: Some(worker),
            streams: Vec::new(),
            plan: plan.clone(),
            channels: master.channels.iter().map(|c| (c.index, c.id)).collect(),
            dir: dir.clone(),
        };
        let grid = match followed.worker_mut()?.grid(self.model.rate) {
            Ok(grid) => grid,
            // Loaded all the same: the separation gets it as it is.
            Err(why) => {
                progress(Event::message(format!(
                    "the original is separated after the extraction: {why}"
                )));
                while self.decode_ended().is_none() {
                    self.wait();
                }
                return Ok(Some(followed));
            }
        };
        let names: Vec<String> = plan.passes.iter().map(|p| pass_name(p, &units)).collect();
        for &pass in &wanted {
            followed.streams.push(Stream::new(
                pass,
                &names[pass],
                &plan,
                &units,
                &master,
                &dir,
                grid,
                self.model.rate,
            )?);
        }
        progress(Event::message(format!(
            "separating the original as the extraction writes it, on this machine: as many of \
             its {} {} as it keeps up with",
            wanted.len(),
            if plan.passes.len() < units.len() {
                "pairs"
            } else {
                "elements"
            }
        )));

        // One block at a time, the first element with one ready first; one
        // that falls behind lets the last of the followed go, once a block
        // was made since the last went.
        let mut may_let_go = true;
        while self.decode_ended().is_none() && !followed.streams.is_empty() {
            let held: Vec<u64> = followed.streams.iter().map(Stream::held).collect();
            let behind = followed
                .streams
                .iter()
                .zip(&held)
                .any(|(stream, &held)| grid.ready(held) >= stream.blocks + BEHIND);
            if behind && may_let_go {
                let stream = followed.streams.pop().expect("not empty");
                progress(Event::message(format!(
                    "  {} let go: this machine does not keep up with the extraction on that many; \
                     it is separated after it",
                    stream.base
                )));
                followed.let_go(stream);
                may_let_go = false;
                continue;
            }
            let ready = (0..followed.streams.len())
                .find(|&i| held[i] >= grid.ready_at(followed.streams[i].blocks));
            let Some(i) = ready else {
                self.wait();
                continue;
            };
            let (worker, stream) = followed.split(i)?;
            let inputs = stream.raw.clone();
            match stream.step(worker, &inputs, held[i], false) {
                Ok(_) => may_let_go = true,
                // The decode failed, or the worker is gone: nothing is kept.
                Err(why) if self.stop.raised() || !worker.is_alive() => return Err(why),
                Err(why) => {
                    let stream = followed.streams.remove(i);
                    progress(Event::message(format!(
                        "  {}: {why}; it is separated after the extraction",
                        stream.base
                    )));
                    followed.let_go(stream);
                }
            }
        }
        // Everything let go, the worker waits all the same: the next step
        // gets it loaded.
        while self.decode_ended().is_none() {
            self.wait();
        }
        if self.decode_ended() != Some(true) {
            return Ok(None);
        }
        // The decoder is done: every file is as long as it will be.
        let (begun, idle): (Vec<Stream>, Vec<Stream>) = std::mem::take(&mut followed.streams)
            .into_iter()
            .partition(|stream| stream.blocks > 0);
        for stream in idle {
            followed.let_go(stream);
        }
        followed.streams = begun;
        for stream in &mut followed.streams {
            stream.length = stream.held();
        }
        if !followed.streams.is_empty() {
            let (made, length): (u64, u64) = followed
                .streams
                .iter()
                .fold((0, 0), |(m, l), s| (m + s.written, l + s.length));
            progress(Event::message(format!(
                "during the extraction: {} of the original's {} {} followed, {:.0} % of them \
                 separated; the rest of them is made first in the next step",
                followed.streams.len(),
                plan.passes.len(),
                if plan.passes.len() < units.len() {
                    "pairs"
                } else {
                    "elements"
                },
                100.0 * made as f64 / length.max(1) as f64,
            )));
        }
        Ok(Some(followed))
    }

    /// [`Follower::new`], this machine's worker started by `start` and the
    /// files looked at every `poll`: the tests'.
    #[cfg(test)]
    fn with_mock(
        config: &'a Config,
        job: &'a Job,
        model: &'a Model,
        pairs: Pairs,
        start: Start<'a>,
        poll: Duration,
    ) -> Self {
        Self {
            start: Some(start),
            poll,
            ..Self::new(config, job, model, pairs)
        }
    }

    #[cfg(test)]
    fn waits(&self) -> u64 {
        self.waits.load(std::sync::atomic::Ordering::SeqCst)
    }
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

/// What was followed during the extraction, and the worker that followed
/// it: for the separation of the original to finish. Dropped, its worker
/// lets the device go and what it grew goes from the disk.
pub struct Followed {
    worker: Option<Worker>,
    streams: Vec<Stream>,
    /// The passes it was cut into.
    plan: Plan,
    /// The master set's elements as it read them: the decoder's rank, the
    /// identifier.
    channels: Vec<(usize, u32)>,
    dir: PathBuf,
}

impl Followed {
    fn worker_mut(&mut self) -> Result<&mut Worker> {
        self.worker
            .as_mut()
            .ok_or_else(|| Error::msg("the worker that followed the original is gone"))
    }

    fn split(&mut self, stream: usize) -> Result<(&mut Worker, &mut Stream)> {
        let worker = self
            .worker
            .as_mut()
            .ok_or_else(|| Error::msg("the worker that followed the original is gone"))?;
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

    /// The passes being followed.
    pub(super) fn passes(&self) -> Vec<usize> {
        self.streams.iter().map(|stream| stream.pass).collect()
    }

    /// How much of them is made, in passes.
    pub(super) fn made(&self) -> f64 {
        self.streams.iter().map(Stream::made).sum()
    }

    /// Itself, if what it followed is what the separation is about to make:
    /// the same elements, from the same waveforms, in the same passes. Its
    /// worker is kept either way.
    pub(super) fn fits(
        mut self,
        master: &MasterSet,
        plan: &Plan,
        progress: Progress,
    ) -> Option<Self> {
        let channels: Vec<(usize, u32)> = master.channels.iter().map(|c| (c.index, c.id)).collect();
        if !self.streams.is_empty() && (channels != self.channels || *plan != self.plan) {
            progress(Event::message(
                "the master set is not the one the extraction was followed on: what was \
                 separated during it is dropped",
            ));
            for stream in std::mem::take(&mut self.streams) {
                self.let_go(stream);
            }
        }
        Some(self)
    }

    pub(super) fn take_worker(&mut self) -> Option<Worker> {
        self.worker.take()
    }

    pub(super) fn streams(&mut self) -> &mut [Stream] {
        &mut self.streams
    }
}

impl Drop for Followed {
    fn drop(&mut self) {
        // The device first, then the files: closed before they go, which
        // Windows asks.
        drop(self.worker.take());
        for stream in self.streams.drain(..) {
            stream.remove();
        }
        let _ = discard(&self.dir);
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
    #[allow(clippy::too_many_arguments)]
    fn new(
        pass: usize,
        base: &str,
        plan: &Plan,
        units: &[Unit],
        master: &MasterSet,
        dir: &Path,
        grid: Grid,
        rate: u32,
    ) -> Result<Self> {
        let step = plan.passes[pass];
        let sides: Vec<usize> = std::iter::once(step.left).chain(step.right).collect();
        let raw_dir = master.header.parent().unwrap_or(Path::new("."));
        let raw = sides
            .iter()
            .map(|&u| {
                let id: u32 = units[u]
                    .name
                    .parse()
                    .map_err(|_| Error::msg(format!("{}: not an element", units[u].name)))?;
                let channel =
                    master.channels.iter().find(|c| c.id == id).ok_or_else(|| {
                        Error::msg(format!("element {id}: not in the master set"))
                    })?;
                Ok(raw_dir.join(format!("{}{}.wav", extract::RAW, channel.index)))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut outputs = Vec::new();
        for (u, side) in serving(plan, pass) {
            for output in &units[u].outputs {
                let growing = dir.join(output.path.file_name().unwrap_or_default());
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
            base: base.to_string(),
            sides,
            raw,
            stems: dir.join("stems"),
            grid,
            rate,
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
    pub(super) fn finish(
        &mut self,
        worker: &mut Worker,
        units: &[Unit],
        frames: &[u64],
        mastered_at: u32,
        mut made: impl FnMut(f64),
    ) -> Result<()> {
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
        for grown in &mut self.outputs {
            grown.settle(frames[grown.unit], mastered_at)?;
        }
        Ok(())
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
    use crate::job::Job;
    use crate::models::Catalogue;
    use crate::progress::quiet;
    use std::collections::HashMap;
    use std::io::{BufRead, BufWriter, Read, Seek, SeekFrom, Write};
    use std::process::Command;
    use std::sync::Arc;

    /// Set in the child's environment: it is the mock, not the tests.
    const MOCK: &str = "OMNIFROG_MOCK_WORKER";
    /// Milliseconds the mock spends on a block, as a GPU would.
    const MOCK_BLOCK_MS: &str = "OMNIFROG_MOCK_BLOCK_MS";
    /// `<base>:<block>`: the block the mock fails on.
    const MOCK_FAIL: &str = "OMNIFROG_MOCK_FAIL";
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
        let fail = std::env::var(MOCK_FAIL).ok();
        let say = |line: &str| {
            let mut out = std::io::stdout().lock();
            let _ = writeln!(out, "{line}");
            let _ = out.flush();
        };
        // The test harness has printed « test …mock_worker ... » with no end
        // of line: this one ends it, so that `PRET` is a line of its own.
        say("");
        say("PRET");
        let mut streams: HashMap<String, (u64, Option<Stems>)> = HashMap::new();
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            let fields: Vec<&str> = line.split('\t').collect();
            let answer = match fields[0] {
                "QUIT" => break,
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
                    if fail.as_deref() == Some(&format!("{base}:{k}")) {
                        return Err("RuntimeError: the mock fails here".to_string());
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
    /// four dead bits — the original mastered at 20.
    struct Film {
        elements: Vec<(usize, u32, Option<&'static str>)>,
        frames: u64,
    }

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

        /// The extraction done: `vo_<id>.wav` whole, and the header.
        fn extracted(&self, dir: &Path) {
            fs::create_dir_all(dir).unwrap();
            for (_, id, _) in &self.elements {
                let mut bytes = Self::wav_header(self.frames);
                bytes.extend(Self::bytes(*id, 0, self.frames));
                fs::write(dir.join(format!("{}{id}.wav", extract::ORIGINAL)), bytes).unwrap();
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
            let end = until.unwrap_or(self.frames);
            let mut at = 0;
            while at < end {
                let to = (at + chunk).min(end);
                for (file, (_, id, _)) in files.iter_mut().zip(&self.elements) {
                    file.write_all(&Self::bytes(*id, at, to)).unwrap();
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
                // harletty's finish: the sizes, once everything is written.
                for file in &mut files {
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
            Self { elements, frames }
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
                Some(&start),
                progress,
            )
            .unwrap()
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

        /// The reference: the film extracted, then separated whole.
        fn whole(&self, film: &Film, pairs: Pairs) -> Outputs {
            let job = self.job("whole", pairs);
            film.extracted(&job.extract_dir());
            self.separate(&job, None, &[], &quiet);
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
    struct Said(Mutex<Vec<String>>);

    impl Said {
        fn progress(self: &Arc<Self>) -> impl Fn(Event) + Sync + use<> {
            let said = Arc::clone(self);
            move |event| {
                if let Event::Message { text } = event {
                    said.0.lock().unwrap().push(text);
                }
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

    /// Follow `film` as it grows, then separate it: what was followed, the
    /// outputs, and what the run said. `in_step`, the extraction writes a
    /// chunk once the follower made every block the last one let it make, so
    /// that what is followed does not depend on how fast the machine is;
    /// otherwise it writes one every `pause`.
    fn follow(
        bench: &Bench,
        film: &Film,
        pairs: Pairs,
        env: Vec<(&'static str, String)>,
        chunk: u64,
        pause: Duration,
        in_step: bool,
    ) -> (f64, Outputs, Arc<Said>) {
        let job = bench.job("followed", pairs);
        let said = Arc::new(Said::default());
        let progress = said.progress();
        let start = |stop: Stop| Worker::start_command(mock(&env), Some(stop), &quiet);
        let follower = Follower::with_mock(
            &bench.config,
            &job,
            &bench.model,
            pairs,
            &start,
            Duration::from_millis(2),
        );
        let dir = job.extract_dir();
        let followed = std::thread::scope(|scope| {
            let following = scope.spawn(|| follower.follow(&progress));
            let begun =
                || said.any("as the extraction writes it") || said.any("known once it is read");
            // Caught up once it found nothing to do twice since the chunk was
            // written: the second time, it had seen the chunk. A follower that
            // never is gets a minute, then the extraction runs on.
            let late = std::cell::Cell::new(false);
            let caught_up = || {
                let seen = follower.waits();
                let deadline = std::time::Instant::now() + Duration::from_secs(60);
                while !late.get() && follower.waits() < seen + 2 && !following.is_finished() {
                    late.set(std::time::Instant::now() >= deadline);
                    std::thread::sleep(Duration::from_millis(1));
                }
            };
            film.grow(
                &dir,
                chunk,
                pause,
                None,
                &begun,
                in_step.then_some(&caught_up as &dyn Fn()),
            );
            follower.decoded(true);
            following.join().unwrap()
        });
        film.rename(&dir);
        let made = followed.as_ref().map_or(0.0, Followed::made);
        bench.separate(&job, followed, &env, &progress);
        assert!(
            !job.separate_dir().join(DIR).exists(),
            "the follower's files are left"
        );
        (made, Bench::outputs(&job), said)
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
        let (_, followed, said) = follow(
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
        let film = Film::objects(3, 60_000);
        let job = bench.job("failed", Pairs::default());
        let env = vec![(MOCK_BLOCK_MS, "5".to_string())];
        let start = |stop: Stop| Worker::start_command(mock(&env), Some(stop), &quiet);
        let follower = Follower::with_mock(
            &bench.config,
            &job,
            &bench.model,
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
        assert!(followed.is_none());
        assert!(!job.separate_dir().join(DIR).exists());
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
}
