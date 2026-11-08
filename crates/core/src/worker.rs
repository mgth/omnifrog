//! The inference worker: `separer_msst.py`, driven from Rust.
//!
//! The script is compiled into the binary and written where the venv can
//! find it, so a bundle is one executable plus the tools. It is started
//! once per step in its persistent mode and fed one element at a time:
//!
//! ```text
//! PRET                                  torch, the device and the checkpoint are up
//! SEP\t<out dir>\t<wav>[\t<wav>…]       ← a request, on standard input
//! GLOB <0..1>                           how far through the request it is
//! OK <base>                             one input done
//! FIN                                   the request is done
//! ERR <why>                             the request failed; the worker lives on
//! QUIT                                  ← shut down, releasing the device
//! ```
//!
//! Under pymss, an element can also be separated while the extraction is
//! still writing it ([`crate::separate`]'s follower), one block at a time:
//!
//! ```text
//! GRID\t<rate>                          ← how an input labelled <rate> is cut
//! GRID <step> <ahead>                   block k can be made at (k+1)·step + ahead frames
//! BLK\t<out>\t<base>\t<rate>\t<frames>\t<whole>\t<wav>[\t<wav>]
//!                                       ← the next block of <base>, its inputs
//!                                         holding <frames> (whole: and no more)
//! BLK <base> <from> <to>                <out>/<base>.voix.wav and .sansvoix.wav
//!                                         hold frames [from, to) of the stems
//! OK <base>                             that was the last block
//! DROP\t<base>                          ← forget it
//! ```
//!
//! Each of them ends with `FIN` or `ERR` like a separation.
//!
//! Under pymss, an element that runs out of device memory goes through MSST
//! alone, loaded then and kept: the same stems and `OK` come out, and the
//! worker says so on standard error, which is passed on to the log.
//!
//! Why persistent: a film is a dozen elements and as many channels, each
//! separated on its own so that the temporaries stay small. A process per
//! element would re-import torch and re-map gigabytes of ROCm or CUDA
//! libraries every time, which freezes the machine when memory is tight,
//! and reload the checkpoint on top.
//!
//! One worker runs at a time on this machine: [`Lock`] is taken in the
//! working directory before the first one starts. Two separations sharing a
//! GPU is how a driver panic is arranged. Nor does one start under a ROCm
//! override the card cannot run (see [`crate::rocm`] and [`command`]),
//! which is how a GPU reset is.
//!
//! A worker can also run on another machine, through `ssh` and that
//! machine's omnifrog ([`Worker::start_remote`], [`crate::remote`]): the
//! same words, two more before `PRET` (`BUSY`, `DIR`), and the files carried
//! there and back around each request.

use crate::config::{Config, Engine, Inference, Remote};
use crate::models::{Local, Model};
use crate::progress::{Event, Progress};
use crate::remote::{self, Ssh};
use crate::run::Stop;
use crate::{Error, Result, wav};
use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// The worker's source.
pub const SOURCE: &str = include_str!("../../../py/separer_msst.py");
pub const FILE_NAME: &str = "separer_msst.py";

/// The suffixes the worker writes beside each input.
pub const VOICE_SUFFIX: &str = ".voix.wav";
pub const REST_SUFFIX: &str = ".sansvoix.wav";

/// How long the loading of a checkpoint may take before it is called dead.
const READY_TIMEOUT: Duration = Duration::from_secs(20 * 60);
/// How long a request may go without a word from the worker.
const QUIET_TIMEOUT: Duration = Duration::from_secs(60 * 60);
/// How long `QUIT` is given before the process is killed.
const QUIT_TIMEOUT: Duration = Duration::from_secs(10);
/// How often a worker that can be stopped looks at its [`Stop`] while it
/// waits for a word.
const TICK: Duration = Duration::from_millis(100);
/// What a worker told to stop says it was.
const STOPPED: &str = "stopped: the pass was made elsewhere first, or the step is over";
/// Lines of standard error kept for a failure report.
const TAIL: usize = 60;
/// How the worker's standard error starts the line saying an element ran
/// out of device memory under pymss and went through MSST instead.
const FALLBACK: &str = "pymss : VRAM insuffisante";

/// Where the worker is on disk, writing it when absent or out of date.
pub fn ensure() -> Result<PathBuf> {
    let dir = Config::data_dir().join("worker");
    fs::create_dir_all(&dir).map_err(|e| Error::io(&dir, e))?;
    let path = dir.join(FILE_NAME);
    let current = fs::read_to_string(&path).ok();
    if current.as_deref() != Some(SOURCE) {
        fs::write(&path, SOURCE).map_err(|e| Error::io(&path, e))?;
    }
    Ok(path)
}

/// What the worker said, one line at a time.
enum Word {
    Ready,
    /// How far through the request, from 0 to 1.
    Fraction(f64),
    /// One input is done. The worker names it, but the caller sent the
    /// batch and already knows what it holds.
    Done,
    /// The request is done.
    Finished,
    /// The request failed, with the reason; the worker is still alive.
    Failed(String),
    /// A line of standard error worth the log: an element pymss ran out of
    /// device memory on, separated through MSST instead, or what a remote's
    /// omnifrog said (a model it fetched).
    Note(String),
    /// A remote's GPU is someone else's for now; nothing was loaded.
    Busy(String),
    /// Where a remote wants the inputs, and leaves the stems.
    Dir(String),
    /// How the worker cuts a followed element into blocks.
    Grid(Grid),
    /// The frames of a followed element one block's stems hold.
    Block {
        from: u64,
        to: u64,
    },
    /// Anything else the worker or MSST printed.
    Other(String),
}

/// How the worker cuts an element it follows: blocks of `step` frames, and
/// block k — unless it is the last — can be made once the input holds
/// `(k + 1) · step + ahead` frames. The last runs from `k · step` to the end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub step: u64,
    pub ahead: u64,
}

impl Grid {
    /// What the input has to hold before block `k` can be asked for, unless
    /// it is the last.
    pub fn ready_at(self, k: u64) -> u64 {
        (k + 1) * self.step + self.ahead
    }

    /// How many blocks an input holding `held` frames can be cut into
    /// before its last one: those that can be made now.
    pub fn ready(self, held: u64) -> u64 {
        held.saturating_sub(self.ahead) / self.step.max(1)
    }

    /// How many blocks an element of `frames` frames is cut into: the plan
    /// the worker follows, said here only to count.
    pub fn blocks(self, frames: u64) -> u64 {
        self.ready(frames) + 1
    }
}

/// One block of a followed element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    /// The frames of the element its stems hold: `from == to` when the
    /// input did not hold enough for one.
    pub from: u64,
    pub to: u64,
    /// The element is done.
    pub last: bool,
}

/// What a block is asked of.
#[derive(Debug, Clone, Copy)]
pub struct Ask<'a> {
    /// The element's name: the stems are `<base>.voix.wav` and
    /// `<base>.sansvoix.wav` in `out_dir`.
    pub base: &'a str,
    pub out_dir: &'a Path,
    /// The rate the model is handed its samples at (the no-resample trick).
    pub rate: u32,
    /// Frames every input holds.
    pub held: u64,
    /// The inputs are whole: `held` is their length.
    pub whole: bool,
    /// One mono input, or two side by side.
    pub inputs: &'a [PathBuf],
}

/// A loaded model, fed one element at a time.
pub struct Worker {
    child: Child,
    stdin: Option<ChildStdin>,
    words: Receiver<Word>,
    errors: Arc<Mutex<VecDeque<String>>>,
    pumps: Vec<JoinHandle<()>>,
    /// The most device memory torch reserved during a request, in GiB.
    peak: Option<f64>,
    /// What standard error said during the requests that is worth the log,
    /// until [`Worker::take_notes`].
    notes: Vec<String>,
    /// Held for as long as the worker is: one separation at a time. A
    /// remote's is taken on its own machine.
    _lock: Option<Lock>,
    /// What the messages call the process.
    program: String,
    /// Set for a worker on another machine.
    remote: Option<Far>,
    /// Raised, the worker is killed wherever it is: another device made its
    /// pass first, or the step is over.
    stop: Option<Stop>,
    /// The process was killed rather than left to end. Its streams may then
    /// stay open for a while — held by what it started, or by ssh's master
    /// connection — so the threads reading them are not waited for.
    killed: bool,
}

/// Why no word came.
enum Silence {
    /// Nothing for the time allowed.
    Quiet,
    /// The worker's output closed: it is gone.
    Gone,
    /// Its [`Stop`] was raised.
    Stopped,
}

/// A worker on another machine: how to reach it, and the directory there
/// the inputs go to and the stems come out of.
struct Far {
    ssh: Ssh,
    host: String,
    dir: Option<String>,
    /// The inputs of the elements it follows, as far as they were sent.
    sent: Vec<Sent>,
}

/// An input of a followed element, carried to the remote a block at a time.
struct Sent {
    /// Its path there.
    path: String,
    /// The bytes of it there.
    bytes: u64,
    /// Where its samples start: a header written again otherwise is sent
    /// again from the beginning.
    data_offset: u64,
}

impl Far {
    fn dir(&self) -> &str {
        self.dir.as_deref().unwrap_or_default()
    }

    /// Where the inputs of the followed element `base` are there. Named by
    /// their rank, not their name, which the extraction changes once it is
    /// done: the same file goes on growing there.
    fn inputs_of(&self, base: &str) -> String {
        format!("{}/in/{base}", self.dir())
    }

    /// Carry there what it does not have yet of `ask`'s inputs — their
    /// header, and up to the frames they are said to hold — and return
    /// their paths there.
    fn carry(&mut self, ask: &Ask) -> Result<Vec<String>> {
        let dir = self.inputs_of(ask.base);
        let mut there = Vec::with_capacity(ask.inputs.len());
        for (rank, input) in ask.inputs.iter().enumerate() {
            let to = format!("{dir}/{rank}.wav");
            let format = wav::read_header(input)?;
            let want = format.data_offset + ask.held * format.bytes_per_frame();
            // Taken out while it is sent: a transfer cut short leaves a file
            // there of an unknown length, sent again whole next time.
            let had = self
                .sent
                .iter()
                .position(|sent| sent.path == to)
                .map(|at| self.sent.swap_remove(at))
                .filter(|sent| sent.data_offset == format.data_offset)
                .map_or(0, |sent| sent.bytes);
            if want > had {
                self.ssh.send(input, had, want - had, &to)?;
            }
            self.sent.push(Sent {
                path: to.clone(),
                bytes: want.max(had),
                data_offset: format.data_offset,
            });
            there.push(to);
        }
        Ok(there)
    }

    /// The inputs of `base` go from there.
    fn forget(&mut self, base: &str) -> Result<()> {
        let dir = self.inputs_of(base);
        let inside = format!("{dir}/");
        self.sent.retain(|sent| !sent.path.starts_with(&inside));
        self.ssh
            .run(
                &format!("rm -rf -- {}", remote::quote(&dir)),
                remote::QUICK_TIMEOUT,
            )
            .map(|_| ())
    }
}

/// How a worker on another machine started.
pub enum Started {
    Ready(Box<Worker>),
    /// Its GPU is someone else's for now, or another run's: the reason.
    Busy(String),
}

/// The command that starts the worker on `model`, in its persistent mode,
/// with `inference`'s settings; its streams are the caller's to set.
///
/// With `idle_below_gib` (a remote's worker), the worker first reads how much
/// of its device's memory other programs hold, and answers `BUSY <why>` and
/// exits, having loaded nothing, when that is as much or more.
///
/// Refused under a ROCm override the card cannot run, the first copy to
/// which resets the GPU and the desktop with it: here, so that a run on
/// this machine and another machine's pass ([`crate::remote::serve`]) are
/// both refused before Python starts.
pub fn command(
    config: &Config,
    model: &Model,
    local: &Local,
    inference: Inference,
    idle_below_gib: Option<f64>,
) -> Result<Command> {
    if let Some(why) = crate::rocm::check().and_then(|report| report.broken) {
        return Err(Error::msg(format!("the inference is not started: {why}")));
    }
    let script = ensure()?;
    let python = config.python();
    if !python.is_file() {
        return Err(Error::msg(format!(
            "{}: no Python for the inference; run `omnifrog setup`",
            python.display()
        )));
    }
    let msst = config.msst_dir();
    if !msst.join("inference.py").is_file() {
        return Err(Error::msg(format!(
            "{}: no MSST checkout; run `omnifrog setup`",
            msst.display()
        )));
    }
    let mut command = crate::run::command(&python);
    command
        .arg(&script)
        .arg(&msst)
        .arg(&local.config)
        .arg(&local.checkpoint)
        .arg(&model.model_type)
        .arg(model.bigshifts.to_string())
        .arg(if model.tta { "1" } else { "0" })
        .arg(match &local.module {
            Some(module) => module.as_os_str(),
            None => "-".as_ref(),
        });
    if model.overlap > 0 {
        // Optional, between the module and the mode: the worker applies
        // it as it loads the config, which stays as downloaded.
        command.arg(format!("overlap={}", model.overlap));
    }
    if inference.engine == Engine::Pymss {
        // pymss on the same files, the windows overlapping by this share;
        // the worker falls back to MSST when it cannot.
        command.arg(format!("pymss={}", inference.overlap.min(90)));
        command.arg(format!("bloc={}", inference.block_seconds.max(1)));
    }
    // The configuration's batch, else the catalogue's, else the author's.
    let batch = match inference.batch_size {
        0 => model.batch,
        asked => asked,
    };
    if batch > 0 {
        command.arg(format!("batch={batch}"));
    }
    if let Some(idle) = idle_below_gib.filter(|&idle| idle > 0.0) {
        command.arg(format!("idle={idle}"));
    }
    command
        .arg("--serveur")
        // Python decodes a pipe in the code page on Windows, which
        // would mangle any path with an accent in it.
        .env("PYTHONUTF8", "1")
        .env("PYTHONUNBUFFERED", "1");
    Ok(command)
}

impl Worker {
    /// Start a worker on `model` and wait until the checkpoint is on the
    /// device. The model's files are expected to be present; `separate`
    /// fetches them for the caller that has not.
    pub fn start(
        config: &Config,
        model: &Model,
        local: &Local,
        progress: Progress,
    ) -> Result<Self> {
        Self::start_under(config, model, local, progress, None)
    }

    /// [`Worker::start`], the worker killed whenever `stop` is raised —
    /// while it loads, or under a request.
    pub fn start_under(
        config: &Config,
        model: &Model,
        local: &Local,
        progress: Progress,
        stop: Option<Stop>,
    ) -> Result<Self> {
        // What is off without endangering the card; an override it cannot
        // run is refused by `command`.
        if let Some(report) = crate::rocm::check() {
            for warning in report.warnings {
                progress(Event::message(format!("warning: {warning}")));
            }
        }
        let lock = Lock::take(&config.work_dir())?;
        let command = command(config, model, local, config.inference, None)?;
        progress(Event::message(format!(
            "loading {} on the inference device",
            model.name
        )));
        let program = config.python().display().to_string();
        let mut worker = Self::spawn(command, program, Some(lock), None, stop)?;
        match worker.wait_ready(progress)? {
            None => Ok(worker),
            Some(why) => Err(Error::msg(why)),
        }
    }

    /// Start a worker on `model` on another machine, with this machine's
    /// `inference` settings, so that a stem comes out the same wherever it
    /// was made. The model is fetched there if it has to be. The worker is
    /// stopped by `ssh`'s [`Stop`], as its transfers are.
    pub fn start_remote(
        remote: &Remote,
        ssh: &Ssh,
        model: &Model,
        inference: Inference,
        progress: Progress,
    ) -> Result<Started> {
        let json = |value: std::result::Result<String, serde_json::Error>| {
            value.map_err(|e| Error::msg(format!("the request for {}: {e}", remote.host)))
        };
        let line = format!(
            "{} remote-worker {} {} {}",
            remote.command,
            remote::quote(&json(serde_json::to_string(model))?),
            remote::quote(&json(serde_json::to_string(&inference))?),
            remote.idle_below_gib
        );
        progress(Event::message(format!(
            "loading {} on {}",
            model.name, remote.host
        )));
        let far = Far {
            ssh: ssh.clone(),
            host: remote.host.clone(),
            dir: None,
            sent: Vec::new(),
        };
        let program = format!("omnifrog on {}", remote.host);
        let stop = Some(ssh.stop().clone());
        let mut worker = Self::spawn(ssh.command(&line), program, None, Some(far), stop)?;
        match worker.wait_ready(progress)? {
            Some(why) => Ok(Started::Busy(why)),
            None if worker.remote.as_ref().is_some_and(|far| far.dir.is_some()) => {
                Ok(Started::Ready(Box::new(worker)))
            }
            None => Err(worker.died("the remote did not say where its files go")),
        }
    }

    /// The process, its streams read by two threads; not ready yet.
    fn spawn(
        mut command: Command,
        program: String,
        lock: Option<Lock>,
        remote: Option<Far>,
        stop: Option<Stop>,
    ) -> Result<Self> {
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn().map_err(|e| Error::Command {
            program: program.clone(),
            status: "could not start".into(),
            detail: e.to_string(),
        })?;
        let (sender, words) = channel();
        let errors = Arc::new(Mutex::new(VecDeque::with_capacity(TAIL)));
        let mut pumps = Vec::new();
        let stdout = child.stdout.take().expect("stdout was piped");
        let notes = sender.clone();
        pumps.push(std::thread::spawn(move || {
            for line in BufReader::new(stdout)
                .lines()
                .map_while(std::result::Result::ok)
            {
                if sender.send(word(&line)).is_err() {
                    break;
                }
            }
        }));
        let stderr = child.stderr.take().expect("stderr was piped");
        let tail = Arc::clone(&errors);
        pumps.push(std::thread::spawn(move || {
            for line in BufReader::new(stderr)
                .lines()
                .map_while(std::result::Result::ok)
            {
                let line = line.trim_end().to_string();
                if line.is_empty() {
                    continue;
                }
                if line.starts_with(FALLBACK) {
                    let _ = notes.send(Word::Note(line.clone()));
                } else if let Some(said) = line.strip_prefix(remote::SAYS) {
                    let _ = notes.send(Word::Note(said.trim().to_string()));
                }
                let mut tail = tail.lock().expect("the tail is never poisoned");
                if tail.len() == TAIL {
                    tail.pop_front();
                }
                tail.push_back(line);
            }
        }));
        let stdin = child.stdin.take();
        Ok(Self {
            child,
            stdin,
            words,
            errors,
            pumps,
            peak: None,
            notes: Vec::new(),
            _lock: lock,
            program,
            remote,
            stop,
            killed: false,
        })
    }

    /// Whether this worker was told to stop.
    pub fn stopped(&self) -> bool {
        self.stop.as_ref().is_some_and(Stop::raised)
    }

    /// The next word, waiting up to `limit` for it, and no longer than the
    /// next look at the stop.
    fn next(&self, limit: Duration) -> std::result::Result<Word, Silence> {
        let Some(stop) = &self.stop else {
            return self.words.recv_timeout(limit).map_err(|e| match e {
                RecvTimeoutError::Timeout => Silence::Quiet,
                RecvTimeoutError::Disconnected => Silence::Gone,
            });
        };
        let start = std::time::Instant::now();
        loop {
            if stop.raised() {
                return Err(Silence::Stopped);
            }
            let left = limit.saturating_sub(start.elapsed());
            if left.is_zero() {
                return Err(Silence::Quiet);
            }
            match self.words.recv_timeout(left.min(TICK)) {
                Ok(word) => return Ok(word),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return Err(Silence::Gone),
            }
        }
    }

    /// Until the checkpoint is on the device; `Some` is a remote that
    /// declined, with its reason.
    fn wait_ready(&mut self, progress: Progress) -> Result<Option<String>> {
        loop {
            match self.next(READY_TIMEOUT) {
                Ok(Word::Ready) => return Ok(None),
                Ok(Word::Busy(why)) => return Ok(Some(why)),
                Ok(Word::Dir(dir)) => {
                    if let Some(far) = &mut self.remote {
                        far.dir = Some(dir);
                    }
                }
                Ok(Word::Note(said)) => progress(Event::message(format!("  {said}"))),
                // MSST prints the device it settled on, which is the one
                // thing worth seeing before an hour of inference.
                Ok(Word::Other(line)) => {
                    // The device, and a catalogue overlap or a configuration's
                    // batch replacing the published one.
                    if line.starts_with("Using device")
                        || line.starts_with("num_overlap")
                        || line.starts_with("batch_size")
                    {
                        progress(Event::message(format!("  {}", line.trim())));
                    }
                    // With pymss asked: which engine really runs, and why not
                    // pymss when it falls back (the detail is on standard error).
                    if let Some(engine) = line.strip_prefix("MOTEUR ") {
                        progress(Event::message(format!("  engine: {}", engine.trim())));
                    }
                }
                Ok(_) => {}
                Err(Silence::Quiet) => {
                    return Err(self.died("the model did not load within twenty minutes"));
                }
                Err(Silence::Gone) => {
                    return Err(self.died("the worker stopped before the model was loaded"));
                }
                Err(Silence::Stopped) => return Err(self.died(STOPPED)),
            }
        }
    }

    /// Separate `inputs` into `out_dir`, where each one leaves
    /// `<name>.voix.wav` and `<name>.sansvoix.wav`. `progress` is called
    /// with the fraction of the request done.
    pub fn separate(
        &mut self,
        inputs: &[PathBuf],
        out_dir: &Path,
        mut fraction: impl FnMut(f64),
    ) -> Result<Vec<Stems>> {
        // Told to stop before it began: nothing to kill, the worker can
        // serve the next request.
        if self.stopped() {
            return Err(Error::msg(STOPPED));
        }
        fs::create_dir_all(out_dir).map_err(|e| Error::io(out_dir, e))?;
        // Here, the paths as they are; there, each input is carried to the
        // remote's directory first.
        let (request_dir, request_inputs) = match &self.remote {
            None => (
                out_dir.to_string_lossy().into_owned(),
                inputs
                    .iter()
                    .map(|input| input.to_string_lossy().into_owned())
                    .collect::<Vec<_>>(),
            ),
            Some(far) => {
                let dir = far.dir.as_deref().unwrap_or_default();
                let mut sent = Vec::with_capacity(inputs.len());
                for input in inputs {
                    let name = input
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    let to = format!("{dir}/in/{name}");
                    far.ssh.upload(input, &to)?;
                    sent.push(to);
                }
                (format!("{dir}/out"), sent)
            }
        };
        let mut request = String::from("SEP\t");
        request.push_str(&request_dir);
        for input in &request_inputs {
            request.push('\t');
            request.push_str(input);
        }
        self.request(&request, |word| {
            if let Word::Fraction(f) = word {
                fraction(f.clamp(0.0, 1.0));
            }
        })?;
        let bases: Vec<String> = inputs
            .iter()
            .map(|input| {
                input
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default()
            })
            .collect();
        if let Some(far) = &self.remote {
            // The stems come back, then everything of the request goes
            // from the remote's disk: the next one starts on an empty one.
            let dir = far.dir.as_deref().unwrap_or_default();
            let mut gone = request_inputs.clone();
            for base in &bases {
                for suffix in [VOICE_SUFFIX, REST_SUFFIX] {
                    let there = format!("{dir}/out/{base}{suffix}");
                    far.ssh
                        .download(&there, &out_dir.join(format!("{base}{suffix}")))?;
                    gone.push(there);
                }
            }
            let quoted: Vec<String> = gone.iter().map(|path| remote::quote(path)).collect();
            far.ssh.run(
                &format!("rm -f -- {}", quoted.join(" ")),
                remote::QUICK_TIMEOUT,
            )?;
        }
        bases
            .iter()
            .map(|base| {
                let stems = Stems {
                    voice: out_dir.join(format!("{base}{VOICE_SUFFIX}")),
                    rest: out_dir.join(format!("{base}{REST_SUFFIX}")),
                };
                if !stems.voice.is_file() || !stems.rest.is_file() {
                    return Err(Error::msg(format!(
                        "the separation of `{base}` left no stems in {}",
                        out_dir.display()
                    )));
                }
                Ok(stems)
            })
            .collect()
    }

    /// Send one request and read the worker's words until its `FIN`. `on`
    /// is handed every word about the request itself (fractions, `OK`,
    /// `GRID`, `BLK`); the device's peak and the notes are kept here. The
    /// worker is given an hour between two words, and killed when its stop
    /// is raised, whatever the request.
    fn request(&mut self, line: &str, mut on: impl FnMut(Word)) -> Result<()> {
        // Told to stop before it began: nothing to kill, the worker can
        // serve the next request.
        if self.stopped() {
            return Err(Error::msg(STOPPED));
        }
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| Error::msg("the worker's input is closed"))?;
        if stdin.write_all(line.as_bytes()).is_err()
            || stdin.write_all(b"\n").is_err()
            || stdin.flush().is_err()
        {
            return Err(self.died("the worker stopped before the request was sent"));
        }
        loop {
            match self.next(QUIET_TIMEOUT) {
                Ok(Word::Finished) => return Ok(()),
                Ok(Word::Other(line)) => {
                    if let Some(peak) = line
                        .strip_prefix("VRAM ")
                        .and_then(|v| v.trim().parse::<f64>().ok())
                    {
                        self.peak = Some(self.peak.map_or(peak, |p| p.max(peak)));
                    }
                }
                Ok(Word::Note(line)) => self.notes.push(line),
                Ok(Word::Failed(why)) => {
                    return Err(Error::msg(format!("the separation failed: {why}")));
                }
                Ok(word) => on(word),
                Err(Silence::Quiet) => {
                    return Err(self.died("the worker went an hour without a word"));
                }
                Err(Silence::Gone) => {
                    return Err(self.died("the worker stopped during the separation"));
                }
                Err(Silence::Stopped) => return Err(self.died(STOPPED)),
            }
        }
    }

    /// How this worker cuts an element it follows, at `rate`. An error is
    /// a worker that cannot follow one: MSST, which separates whole
    /// elements only, or a worker on a CPU, where following would take the
    /// cores the extraction decodes on.
    pub fn grid(&mut self, rate: u32) -> Result<Grid> {
        let mut grid = None;
        self.request(&format!("GRID\t{rate}"), |word| {
            if let Word::Grid(said) = word {
                grid = Some(said);
            }
        })?;
        grid.filter(|grid| grid.step > 0)
            .ok_or_else(|| Error::msg("the worker did not say how it cuts an element"))
    }

    /// The next block of a followed element, if its inputs hold enough for
    /// one; `fraction` is how far into the block the worker is. A remote is
    /// sent first what its inputs gained since the last block, and its
    /// stems come back into `out_dir`.
    pub fn block(&mut self, ask: &Ask, mut fraction: impl FnMut(f64)) -> Result<Block> {
        fs::create_dir_all(ask.out_dir).map_err(|e| Error::io(ask.out_dir, e))?;
        let (out_dir, inputs) = match &mut self.remote {
            None => (
                ask.out_dir.to_string_lossy().into_owned(),
                ask.inputs
                    .iter()
                    .map(|input| input.to_string_lossy().into_owned())
                    .collect::<Vec<_>>(),
            ),
            Some(far) => (format!("{}/out", far.dir()), far.carry(ask)?),
        };
        let mut request = format!(
            "BLK\t{out_dir}\t{}\t{}\t{}\t{}",
            ask.base,
            ask.rate,
            ask.held,
            u8::from(ask.whole)
        );
        for input in &inputs {
            request.push('\t');
            request.push_str(input);
        }
        let (mut span, mut last) = (None, false);
        self.request(&request, |word| match word {
            Word::Fraction(f) => fraction(f.clamp(0.0, 1.0)),
            Word::Block { from, to } => span = Some((from, to)),
            Word::Done => last = true,
            _ => {}
        })?;
        let (from, to) = span.ok_or_else(|| {
            Error::msg(format!("the worker said nothing of {}'s block", ask.base))
        })?;
        if to < from || to > ask.held || (last && to != ask.held) {
            return Err(Error::msg(format!(
                "{}: a block of frames {from} to {to}, out of inputs holding {}",
                ask.base, ask.held
            )));
        }
        if let Some(far) = &mut self.remote {
            // The block's stems come back, and go from there; with the last
            // one, the inputs too.
            if to > from {
                let mut gone = Vec::with_capacity(2);
                for suffix in [VOICE_SUFFIX, REST_SUFFIX] {
                    let there = format!("{out_dir}/{}{suffix}", ask.base);
                    far.ssh
                        .download(&there, &ask.out_dir.join(format!("{}{suffix}", ask.base)))?;
                    gone.push(remote::quote(&there));
                }
                far.ssh.run(
                    &format!("rm -f -- {}", gone.join(" ")),
                    remote::QUICK_TIMEOUT,
                )?;
            }
            if last {
                far.forget(ask.base)?;
            }
        }
        Ok(Block { from, to, last })
    }

    /// Forget a followed element: what the worker kept of it goes, and on a
    /// remote what it was sent of it.
    pub fn forget(&mut self, base: &str) -> Result<()> {
        let dropped = self.request(&format!("DROP\t{base}"), |_| {});
        if let Some(far) = &mut self.remote {
            far.forget(base)?;
        }
        dropped
    }

    /// From now on, `stop` is what kills this worker, and a remote's
    /// transfers: a worker handed from one step to the next answers to the
    /// step it serves.
    pub fn set_stop(&mut self, stop: Option<Stop>) {
        if let (Some(far), Some(stop)) = (&mut self.remote, &stop) {
            far.ssh = far.ssh.clone().stopped_by(stop.clone());
        }
        self.stop = stop;
    }

    /// A worker started by `command`, which speaks the protocol: the tests'
    /// stand-in for the inference.
    #[cfg(test)]
    pub(crate) fn start_command(
        command: Command,
        stop: Option<Stop>,
        progress: Progress,
    ) -> Result<Self> {
        let mut worker = Self::spawn(command, "a mock worker".into(), None, None, stop)?;
        match worker.wait_ready(progress)? {
            None => Ok(worker),
            Some(why) => Err(Error::msg(why)),
        }
    }

    /// What standard error said during the requests so far that belongs in
    /// the log, as the worker wrote it; emptied by the call. Standard error
    /// is a pipe of its own, so a line written just before `FIN` can come
    /// with the next request.
    pub fn take_notes(&mut self) -> Vec<String> {
        std::mem::take(&mut self.notes)
    }

    /// Whether the worker's process is still running.
    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// The most device memory torch reserved during any request so far, in
    /// GiB; `None` on a CPU or before the first request.
    pub fn device_peak(&self) -> Option<f64> {
        self.peak
    }

    /// The machine a remote worker runs on; `None` here.
    pub fn host(&self) -> Option<&str> {
        self.remote.as_ref().map(|far| far.host.as_str())
    }

    /// The error a dead worker leaves, with the last it said.
    fn died(&mut self, what: &str) -> Error {
        // The input closed too: over there, the end of it is what tells
        // the remote's omnifrog to let its worker go.
        self.stdin = None;
        let _ = self.child.kill();
        self.killed = true;
        let status = self.child.wait().ok();
        let tail = self
            .errors
            .lock()
            .expect("the tail is never poisoned")
            .iter()
            .rev()
            .take(6)
            .rev()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        Error::Command {
            program: self.program.clone(),
            status: status
                .map(|s| s.to_string())
                .unwrap_or_else(|| "no status".into()),
            detail: if tail.is_empty() {
                what.to_string()
            } else {
                format!("{what}\n{tail}")
            },
        }
    }
}

impl Drop for Worker {
    /// `QUIT` first, which lets torch release the device, then the axe.
    fn drop(&mut self) {
        if let Some(mut stdin) = self.stdin.take() {
            let _ = stdin.write_all(b"QUIT\n");
            let _ = stdin.flush();
        }
        let deadline = std::time::Instant::now() + QUIT_TIMEOUT;
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if std::time::Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                _ => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    self.killed = true;
                    break;
                }
            }
        }
        // Killed, the streams close when whatever still holds them lets
        // go; the threads end then, on their own.
        if !self.killed {
            for pump in self.pumps.drain(..) {
                let _ = pump.join();
            }
        }
    }
}

/// The two files a separated input leaves, at the model's rate.
#[derive(Debug, Clone, PartialEq)]
pub struct Stems {
    /// The dialogue.
    pub voice: PathBuf,
    /// Everything else, summed: music and effects.
    pub rest: PathBuf,
}

fn word(line: &str) -> Word {
    let line = line.trim_end();
    if line == "PRET" {
        Word::Ready
    } else if line == "FIN" {
        Word::Finished
    } else if let Some(why) = line.strip_prefix("ERR ") {
        Word::Failed(why.to_string())
    } else if let Some(why) = line.strip_prefix("BUSY ") {
        Word::Busy(why.to_string())
    } else if let Some(dir) = line.strip_prefix("DIR ") {
        Word::Dir(dir.to_string())
    } else if line.starts_with("OK ") {
        Word::Done
    } else if let Some(grid) = line.strip_prefix("GRID ").and_then(|rest| {
        let (step, ahead) = rest.trim().split_once(' ')?;
        Some(Grid {
            step: step.parse().ok()?,
            ahead: ahead.trim().parse().ok()?,
        })
    }) {
        Word::Grid(grid)
    } else if let Some((from, to)) = line.strip_prefix("BLK ").and_then(|rest| {
        // The base comes first and may hold anything; the two numbers end
        // the line.
        let mut fields = rest.rsplitn(3, ' ');
        let to = fields.next()?.parse().ok()?;
        let from = fields.next()?.parse().ok()?;
        fields.next()?;
        Some((from, to))
    }) {
        Word::Block { from, to }
    } else if let Some(value) = line.strip_prefix("GLOB ") {
        match value.trim().parse() {
            Ok(fraction) => Word::Fraction(fraction),
            Err(_) => Word::Other(line.to_string()),
        }
    } else {
        Word::Other(line.to_string())
    }
}

/// One separation at a time on this machine.
///
/// Two of them sharing a GPU is how the driver is made to panic and the
/// session to freeze — seen on this hardware — so the second run is told
/// to wait rather than being allowed to try.
///
/// The lock is the operating system's, on a file in the working directory:
/// `flock` on Unix, `LockFileEx` on Windows (see [`lock`]). The system drops it when the
/// process that holds it ends, however it ends, so a run that crashed never
/// leaves the next one locked out — which a file that merely exists would,
/// and which no check of a process id can rule out once ids are reused. The
/// file itself stays; it holds the id of the last run that took it, for the
/// message a second run gets.
pub struct Lock {
    _file: fs::File,
}

impl Lock {
    pub fn take(work_dir: &Path) -> Result<Self> {
        fs::create_dir_all(work_dir).map_err(|e| Error::io(work_dir, e))?;
        let path = work_dir.join(".separating.lock");
        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| Error::io(&path, e))?;
        match lock(&file) {
            Ok(()) => {
                let _ = file.set_len(0);
                let _ = writeln!(file, "{}", std::process::id());
                let _ = file.flush();
                Ok(Self { _file: file })
            }
            Err(fs::TryLockError::WouldBlock) => {
                let holder = fs::read_to_string(&path).unwrap_or_default();
                Err(Error::msg(format!(
                    "another run is separating{}; wait for it to finish",
                    match holder.trim() {
                        "" => String::new(),
                        pid => format!(" (process {pid})"),
                    }
                )))
            }
            Err(fs::TryLockError::Error(e)) => Err(Error::io(&path, e)),
        }
    }
}

/// The lock on `file`, taken if it is free: the standard library's, on
/// the whole file.
#[cfg(not(windows))]
fn lock(file: &fs::File) -> std::result::Result<(), fs::TryLockError> {
    file.try_lock()
}

/// On Windows a locked region is one no other handle may read, so a lock
/// on the whole file would keep the id of the run holding it from the run
/// it refuses. The lock is on a single byte far past the end of the file
/// instead; the standard library's, on everything, still conflicts with it.
#[cfg(windows)]
fn lock(file: &fs::File) -> std::result::Result<(), fs::TryLockError> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{ERROR_IO_PENDING, ERROR_LOCK_VIOLATION};
    use windows_sys::Win32::Storage::FileSystem::{
        LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY, LockFileEx,
    };
    use windows_sys::Win32::System::IO::OVERLAPPED;

    // SAFETY: a plain Win32 call on the file's own handle, with a zeroed
    // OVERLAPPED that carries only the offset of the byte: 4 GiB in.
    let locked = unsafe {
        let mut overlapped: OVERLAPPED = std::mem::zeroed();
        overlapped.Anonymous.Anonymous.OffsetHigh = 1;
        LockFileEx(
            file.as_raw_handle(),
            LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
            0,
            1,
            0,
            &mut overlapped,
        )
    };
    if locked != 0 {
        return Ok(());
    }
    let e = std::io::Error::last_os_error();
    match e.raw_os_error() {
        Some(code) if code == ERROR_LOCK_VIOLATION as i32 || code == ERROR_IO_PENDING as i32 => {
            Err(fs::TryLockError::WouldBlock)
        }
        _ => Err(fs::TryLockError::Error(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_protocol_is_read_one_line_at_a_time() {
        assert!(matches!(word("PRET"), Word::Ready));
        assert!(matches!(word("FIN"), Word::Finished));
        assert!(matches!(word("OK vo_10"), Word::Done));
        assert!(
            matches!(word("ERR RuntimeError: out of memory"), Word::Failed(why) if why.contains("out of memory"))
        );
        match word("GLOB 0.4213") {
            Word::Fraction(f) => assert!((f - 0.4213).abs() < 1e-9),
            _ => panic!("GLOB is a fraction"),
        }
    }

    #[test]
    fn a_followed_element_s_words_are_read_too() {
        assert!(matches!(
            word("GRID 26460000 352801"),
            Word::Grid(Grid {
                step: 26_460_000,
                ahead: 352_801
            })
        ));
        assert!(matches!(
            word("BLK 12+11 26460000 52920000"),
            Word::Block {
                from: 26_460_000,
                to: 52_920_000
            }
        ));
        // Nothing ready: an empty block.
        assert!(matches!(word("BLK 10 0 0"), Word::Block { from: 0, to: 0 }));
        assert!(matches!(word("BLK 10 0"), Word::Other(_)));
        assert!(matches!(word("GRID soon"), Word::Other(_)));
        // The worker's own source answers what the Rust side asks.
        for said in [
            "\"GRID {bloc} {avance}\"",
            "\"BLK {base} {deb} {deb + len(voix)}\"",
            "champs[0] == \"DROP\"",
            "champs[0] == \"BLK\" and len(champs) >= 7",
        ] {
            assert!(SOURCE.contains(said), "the worker no longer has {said}");
        }
    }

    #[test]
    fn a_grid_counts_the_blocks_the_worker_cuts() {
        let grid = Grid {
            step: 100,
            ahead: 11,
        };
        // Short of one step and its lookahead: one block, made at the end.
        assert_eq!(grid.blocks(1), 1);
        assert_eq!(grid.blocks(110), 1);
        assert_eq!(grid.ready_at(0), 111);
        assert_eq!(grid.blocks(111), 2);
        assert_eq!(grid.blocks(210), 2);
        assert_eq!(grid.blocks(211), 3);
    }

    #[test]
    fn anything_else_the_worker_prints_is_passed_through() {
        assert!(matches!(word("Using device:  cuda:0"), Word::Other(_)));
        // A malformed fraction is chatter, not a fraction.
        assert!(matches!(word("GLOB soon"), Word::Other(_)));
    }

    #[test]
    fn the_fallback_note_is_the_line_the_worker_writes() {
        // The worker's French is matched as written: a change on either side
        // would silently stop the log saying an element went through MSST.
        assert!(
            SOURCE.contains(&format!("sys.stderr.write(f\"{FALLBACK} pour ")),
            "the worker no longer writes `{FALLBACK}`"
        );
    }

    #[test]
    fn a_lock_is_one_at_a_time_and_released_on_the_way_out() {
        let dir = std::env::temp_dir().join(format!("omnifrog-lock-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        {
            let _held = Lock::take(&dir).unwrap();
            let refused = Lock::take(&dir).err().unwrap().to_string();
            assert!(refused.contains("another run is separating"), "{refused}");
            assert!(
                refused.contains(&std::process::id().to_string()),
                "{refused}"
            );
        }
        // Released: the next run gets it, although the file is still there.
        assert!(dir.join(".separating.lock").is_file());
        let _next = Lock::take(&dir).unwrap();
        drop(_next);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_lock_file_nobody_holds_does_not_lock_anybody_out() {
        // What a run that crashed leaves: the file, with its id, and no lock.
        let dir = std::env::temp_dir().join(format!("omnifrog-stale-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(".separating.lock"), "4294967294\n").unwrap();
        let taken = Lock::take(&dir);
        assert!(taken.is_ok());
        drop(taken);
        let _ = fs::remove_dir_all(&dir);
    }
}
