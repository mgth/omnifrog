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
//! GPU is how a driver panic is arranged.
//!
//! A worker can also run on another machine, through `ssh` and that
//! machine's omnifrog ([`Worker::start_remote`], [`crate::remote`]): the
//! same words, two more before `PRET` (`BUSY`, `DIR`), and the files carried
//! there and back around each request.

use crate::config::{Config, Engine, Inference, Remote};
use crate::models::{Local, Model};
use crate::progress::{Event, Progress};
use crate::remote::{self, Ssh};
use crate::{Error, Result};
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
    /// Anything else the worker or MSST printed.
    Other(String),
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
}

/// A worker on another machine: how to reach it, and the directory there
/// the inputs go to and the stems come out of.
struct Far {
    ssh: Ssh,
    host: String,
    dir: Option<String>,
}

/// How a worker on another machine started.
pub enum Started {
    Ready(Box<Worker>),
    /// Its GPU is someone else's for now, or another run's: the reason.
    Busy(String),
}

/// The command that starts the worker on `model`, in its persistent mode,
/// with `inference`'s settings; its streams are the caller's to set.
pub fn command(
    config: &Config,
    model: &Model,
    local: &Local,
    inference: Inference,
) -> Result<Command> {
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
        let lock = Lock::take(&config.work_dir())?;
        let command = command(config, model, local, config.inference)?;
        progress(Event::message(format!(
            "loading {} on the inference device",
            model.name
        )));
        let program = config.python().display().to_string();
        let mut worker = Self::spawn(command, program, Some(lock), None)?;
        match worker.wait_ready(progress)? {
            None => Ok(worker),
            Some(why) => Err(Error::msg(why)),
        }
    }

    /// Start a worker on `model` on another machine, with this machine's
    /// `inference` settings, so that a stem comes out the same wherever it
    /// was made. The model is fetched there if it has to be.
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
        };
        let program = format!("omnifrog on {}", remote.host);
        let mut worker = Self::spawn(ssh.command(&line), program, None, Some(far))?;
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
        })
    }

    /// Until the checkpoint is on the device; `Some` is a remote that
    /// declined, with its reason.
    fn wait_ready(&mut self, progress: Progress) -> Result<Option<String>> {
        loop {
            match self.words.recv_timeout(READY_TIMEOUT) {
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
                Err(RecvTimeoutError::Timeout) => {
                    return Err(self.died("the model did not load within twenty minutes"));
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(self.died("the worker stopped before the model was loaded"));
                }
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
        request.push('\n');
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| Error::msg("the worker's input is closed"))?;
        if stdin.write_all(request.as_bytes()).is_err() || stdin.flush().is_err() {
            return Err(self.died("the worker stopped before the request was sent"));
        }
        loop {
            match self.words.recv_timeout(QUIET_TIMEOUT) {
                Ok(Word::Fraction(f)) => fraction(f.clamp(0.0, 1.0)),
                Ok(Word::Finished) => break,
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
                Ok(_) => {}
                Err(RecvTimeoutError::Timeout) => {
                    return Err(self.died("the worker went an hour without a word"));
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(self.died("the worker stopped during the separation"));
                }
            }
        }
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
            far.ssh.run(&format!("rm -f -- {}", quoted.join(" ")))?;
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
        let _ = self.child.kill();
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
                    break;
                }
            }
        }
        for pump in self.pumps.drain(..) {
            let _ = pump.join();
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
