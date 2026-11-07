//! Separating on another machine's GPU, over SSH.
//!
//! A remote is a machine with omnifrog installed and set up there (`omnifrog
//! setup` on it), that `ssh` reaches with a key and no question
//! ([`crate::config::Remote`]). Two commands of that machine's omnifrog are
//! all this side runs there, both hidden from its help:
//!
//! - `remote-check`, once a step: its version, the device torch would run
//!   on, and how much of that device's memory is in use ([`probe`]). A
//!   remote of another version, or without a GPU torch can use, is left
//!   out.
//! - `remote-worker '<model>' '<inference>' <idle below GiB>`, once a step
//!   ([`serve`]): the worker of [`crate::worker`], its model loaded once
//!   and fed one pass after another as this machine's is, its protocol
//!   untouched, with two words of its own before `PRET`:
//!
//! ```text
//! BUSY <why>     another program holds its GPU, or another run does: nothing was loaded
//! DIR <path>     where the inputs go and the stems come out, on that machine
//! ```
//!
//! The inputs go there and the stems come back through `ssh … cat`, over
//! the connection the first command opened and kept (on Unix): a pass of a
//! two-hour film is about 1.5 GB out and 5 GB back, seconds on a 10 Gb/s
//! link against minutes of inference.
//!
//! **A GPU in use is left alone.** The worker starts only if the other
//! programs there hold less than `idle_below_gib` of the memory of the
//! device torch computes on — that card's own counters, read before it
//! loads anything — and then keeps the card for the rest of the step,
//! without loading the checkpoint again for every pass. A machine whose GPU
//! serves other work is meant to be given to omnifrog alone while a step
//! runs. Nothing there is ever unloaded or stopped. A
//! remote that is busy is asked again a minute later; one that fails twice
//! in a row is left out for the rest of the step. This machine's GPU does
//! whatever the remotes do not, so a step never waits on one.
//!
//! **Nothing here waits on a remote without a limit.** `remote-check` has
//! two minutes; a transfer is cut when no byte moved for one; every command
//! under way is killed when the step ends or fails, or when this machine
//! made the pass first (see `separate`). Over there, a worker whose
//! connection went is killed ten seconds later, so the lock and the GPU are
//! free again.

use crate::config::{Config, Inference};
use crate::models::{self, Model};
use crate::progress::Event;
use crate::run::Stop;
use crate::worker::{self, Lock};
use crate::{Error, Result};
use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

/// How a line of a remote's standard error starts when it is meant for the
/// log rather than for an error report.
pub const SAYS: &str = "omnifrog: ";

/// What the two sides must agree on: the worker's script and these words.
/// Bumped with any change to `remote-worker`'s arguments or words.
const PROTOCOL: u32 = 1;

/// The directory of a remote's work directory a pass's files live in.
const DIR: &str = "remote";

/// `text` as one word of a POSIX shell, whatever it holds.
pub fn quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

/// What both sides compute and compare: a remote whose worker speaks
/// another version of the protocol is left out.
pub fn protocol() -> String {
    // FNV-1a: stable across builds and platforms, unlike the std hasher.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in worker::SOURCE.bytes().chain(PROTOCOL.to_le_bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// How long `remote-check` may take over there: torch imported cold, from a
/// slow disk. Its own Python is given less ([`CHECK_PYTHON`]), so that the
/// remote can still say what hung.
const CHECK_TIMEOUT: Duration = Duration::from_secs(120);
/// How long the Python of `remote-check` may take to name its device.
const CHECK_PYTHON: Duration = Duration::from_secs(90);
/// How long a small command there (`rm`) may take.
pub const QUICK_TIMEOUT: Duration = Duration::from_secs(30);
/// How long a transfer may go without a byte moving before it is called
/// stalled. ssh's own keepalives only notice a link that died.
const STALL: Duration = Duration::from_secs(60);
/// How long the worker over there is given to follow `QUIT`, or to notice
/// it is alone, once this machine has stopped talking to it.
const QUIT_GRACE: Duration = Duration::from_secs(10);

/// The way to one machine: `ssh` with the options every call needs.
#[derive(Debug, Clone)]
pub struct Ssh {
    host: String,
    /// On Unix, the socket the first connection leaves open for the next
    /// ones (`ControlMaster`), so a transfer does not pay a handshake.
    control: Option<PathBuf>,
    /// Raised, every command under way through this one is killed: the step
    /// is over, or another device made the pass first.
    stop: Stop,
}

impl Ssh {
    pub fn new(host: &str) -> Self {
        let control = cfg!(unix).then(|| {
            dirs::runtime_dir()
                .unwrap_or_else(std::env::temp_dir)
                .join("omnifrog-ssh-%C")
        });
        Self {
            host: host.to_string(),
            control,
            stop: Stop::new(),
        }
    }

    /// The same, its commands killed when `stop` is raised.
    pub fn stopped_by(mut self, stop: Stop) -> Self {
        self.stop = stop;
        self
    }

    pub fn stop(&self) -> &Stop {
        &self.stop
    }

    /// `ssh` running `line` in the remote's shell; the streams are the
    /// caller's to set.
    pub fn command(&self, line: &str) -> Command {
        let mut command = crate::run::command("ssh");
        // No terminal, never a question: a key or nothing. A link that dies
        // is noticed within a minute rather than at the next request.
        command.arg("-T").args([
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=4",
        ]);
        if let Some(control) = &self.control {
            command
                .arg("-o")
                .arg("ControlMaster=auto")
                .arg("-o")
                .arg(format!("ControlPath={}", control.display()))
                .arg("-o")
                .arg("ControlPersist=120");
        }
        command.arg("--").arg(&self.host).arg(line);
        command
    }

    /// Run `line` there, for at most `within`, and return what it printed.
    pub fn run(&self, line: &str, within: Duration) -> Result<String> {
        let mut child = self
            .command(line)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| self.failed("could not start", e.to_string()))?;
        let out = read_all(child.stdout.take());
        let err = read_all(child.stderr.take());
        // Killed, its streams may stay open a while longer (held by ssh's
        // master connection): what it said is not waited for.
        let status = self
            .watch(&mut child, Some(within), None)
            .map_err(|cut| self.failed(cut, String::new()))?;
        let (out, err) = (joined(out), joined(err));
        if !status.success() {
            return Err(self.failed(&status.to_string(), err.trim().to_string()));
        }
        Ok(out)
    }

    /// Copy `from` to the path `to` on that machine.
    pub fn upload(&self, from: &Path, to: &str) -> Result<()> {
        let mut file = fs::File::open(from).map_err(|e| Error::io(from, e))?;
        let mut child = self
            .command(&format!("cat > {}", quote(to)))
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| self.failed("could not start", e.to_string()))?;
        let err = read_all(child.stderr.take());
        let sent = Arc::new(AtomicU64::new(0));
        let feed = child.stdin.take().map(|mut stdin| {
            let sent = Arc::clone(&sent);
            // The pipe closes when the copy ends, which is the end of the
            // file for `cat` over there; a killed ssh ends the copy too.
            std::thread::spawn(move || copy_counting(&mut file, &mut stdin, &sent))
        });
        let moved = || sent.load(Ordering::Relaxed);
        let what = |detail: &str| format!("sending {}: {}", from.display(), detail.trim());
        // Killed, nothing more is waited for (see `run`).
        let status = self
            .watch(&mut child, None, Some(&moved))
            .map_err(|cut| self.failed(cut, what("")))?;
        if let Some(feed) = feed {
            let _ = feed.join();
        }
        let err = joined(err);
        if !status.success() {
            return Err(self.failed(&status.to_string(), what(&err)));
        }
        Ok(())
    }

    /// Copy the path `from` on that machine to `to`, aside and then
    /// renamed: a file under its real name is always whole.
    pub fn download(&self, from: &str, to: &Path) -> Result<()> {
        let part = to.with_extension("wav.part");
        let mut file = fs::File::create(&part).map_err(|e| Error::io(&part, e))?;
        let mut child = self
            .command(&format!("cat {}", quote(from)))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| self.failed("could not start", e.to_string()))?;
        let err = read_all(child.stderr.take());
        let got = Arc::new(AtomicU64::new(0));
        let drain = child.stdout.take().map(|mut stdout| {
            let got = Arc::clone(&got);
            std::thread::spawn(move || copy_counting(&mut stdout, &mut file, &got))
        });
        let moved = || got.load(Ordering::Relaxed);
        let what = |detail: &str| format!("fetching {from}: {}", detail.trim());
        let outcome = match self.watch(&mut child, None, Some(&moved)) {
            // Killed, nothing more is waited for (see `run`).
            Err(cut) => Err(self.failed(cut, what(""))),
            Ok(status) => {
                let copied = drain.map(|drain| drain.join());
                let err = joined(err);
                if !status.success() {
                    Err(self.failed(&status.to_string(), what(&err)))
                } else {
                    match copied {
                        Some(Ok(Err(e))) => Err(Error::io(&part, e)),
                        _ => Ok(()),
                    }
                }
            }
        };
        if let Err(e) = outcome {
            let _ = fs::remove_file(&part);
            return Err(e);
        }
        fs::rename(&part, to).map_err(|e| Error::io(to, e))
    }

    /// [`crate::run::watch`] under this connection's stop: the error is the
    /// reason it was killed.
    fn watch(
        &self,
        child: &mut Child,
        within: Option<Duration>,
        moved: Option<&dyn Fn() -> u64>,
    ) -> std::result::Result<ExitStatus, &'static str> {
        match crate::run::watch(child, Some(&self.stop), within, moved.map(|m| (m, STALL))) {
            Ok(Ok(status)) => Ok(status),
            Ok(Err(cut)) => Err(cut.describe()),
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                Err("could not wait")
            }
        }
    }

    fn failed(&self, status: &str, detail: String) -> Error {
        Error::Command {
            program: format!("ssh {}", self.host),
            status: status.to_string(),
            detail,
        }
    }
}

/// What a remote said about itself.
#[derive(Debug, Clone, PartialEq)]
pub struct Probe {
    /// Its omnifrog's version.
    pub version: String,
    /// Its [`protocol`].
    pub protocol: String,
    /// The device torch runs on there: a GPU's name, or `cpu`.
    pub device: String,
    /// Memory in use and in all on its GPU, in GiB, when it can be read.
    pub vram: Option<(f64, f64)>,
    /// What is missing there for a separation to run.
    pub missing: Vec<String>,
}

impl Probe {
    /// Why this machine leaves the remote out; `None` = it takes passes.
    pub fn unfit(&self) -> Option<String> {
        if let Some(missing) = self.missing.first() {
            return Some(missing.clone());
        }
        if self.protocol != protocol() {
            return Some(format!(
                "its omnifrog is {} and speaks another protocol than this {}: install the same \
                 version there",
                self.version,
                env!("CARGO_PKG_VERSION")
            ));
        }
        if self.device == "cpu" || self.device.is_empty() {
            return Some("torch sees no GPU there".to_string());
        }
        None
    }

    /// One line for a progress message.
    pub fn describe(&self) -> String {
        match self.vram {
            Some((used, total)) => {
                format!("{} · {used:.1} of {total:.0} GiB in use", self.device)
            }
            None => self.device.clone(),
        }
    }

    fn parse(text: &str) -> Self {
        let mut probe = Probe {
            version: String::new(),
            protocol: String::new(),
            device: String::new(),
            vram: None,
            missing: Vec::new(),
        };
        for line in text.lines() {
            let line = line.trim_end();
            if let Some(rest) = line.strip_prefix("OMNIFROG ") {
                let mut words = rest.split_whitespace();
                probe.version = words.next().unwrap_or_default().to_string();
                probe.protocol = words.next().unwrap_or_default().to_string();
            } else if let Some(device) = line.strip_prefix("DEVICE ") {
                probe.device = device.trim().to_string();
            } else if let Some(rest) = line.strip_prefix("VRAM ") {
                let mut words = rest.split_whitespace().map(str::parse::<f64>);
                if let (Some(Ok(used)), Some(Ok(total))) = (words.next(), words.next()) {
                    probe.vram = Some((used, total));
                }
            } else if let Some(why) = line.strip_prefix("MISSING ") {
                probe.missing.push(why.trim().to_string());
            }
        }
        if probe.version.is_empty() {
            probe.missing.insert(
                0,
                "no omnifrog answered there (`command` in `remotes`)".to_string(),
            );
        }
        probe
    }
}

/// Ask `remote` what it is: `remote-check` over SSH.
pub fn probe(remote: &crate::config::Remote, ssh: &Ssh) -> Result<Probe> {
    let text = ssh.run(&format!("{} remote-check", remote.command), CHECK_TIMEOUT)?;
    Ok(Probe::parse(&text))
}

/// The remote's side of [`probe`]: what `remote-check` prints.
pub fn check(config: &Config) -> Vec<String> {
    let mut lines = vec![format!(
        "OMNIFROG {} {}",
        env!("CARGO_PKG_VERSION"),
        protocol()
    )];
    let python = config.python();
    if !python.is_file() {
        lines.push(format!(
            "MISSING no Python for the inference at {}: run `omnifrog setup` there",
            python.display()
        ));
    } else {
        // The worker names the device it would run on, and reads that very
        // card's memory (`DEVICE`, `VRAM`): what a pass would find.
        match device_lines(&python) {
            Ok(said) => lines.extend(said),
            Err(why) => lines.push(format!(
                "MISSING the worker does not start in {} ({why}): run `omnifrog setup` there",
                python.display()
            )),
        }
    }
    if !config.msst_dir().join("inference.py").is_file() {
        lines.push(format!(
            "MISSING no MSST checkout at {}: run `omnifrog setup` there",
            config.msst_dir().display()
        ));
    }
    lines
}

/// What `separer_msst.py --device` prints: its `DEVICE` and `VRAM` lines.
fn device_lines(python: &Path) -> std::result::Result<Vec<String>, String> {
    let script = worker::ensure().map_err(|e| e.to_string())?;
    let mut child = crate::run::command(python)
        .arg(&script)
        .arg("--device")
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| crate::run::start_error(&e))?;
    let out = read_all(child.stdout.take());
    let ended = crate::run::watch(&mut child, None, Some(CHECK_PYTHON), None);
    let out = joined(out);
    match ended {
        Ok(Ok(status)) if status.success() => Ok(out
            .lines()
            .filter(|line| line.starts_with("DEVICE ") || line.starts_with("VRAM "))
            .map(str::to_string)
            .collect()),
        Ok(Ok(status)) => Err(status.to_string()),
        Ok(Err(_)) => Err(format!(
            "torch named no device within {} s",
            CHECK_PYTHON.as_secs()
        )),
        Err(e) => Err(e.to_string()),
    }
}

/// The remote's side of a pass: decline if another run has the GPU, else
/// fetch the model, say where the files go and become the worker, on this
/// process's own standard streams. The worker itself declines when other
/// programs hold its device (`idle=`, see [`worker::command`]). Returns the
/// worker's exit code.
pub fn serve(
    config: &Config,
    model: &Model,
    inference: Inference,
    idle_below_gib: f64,
) -> Result<i32> {
    let say = |line: &str| {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{line}");
        let _ = out.flush();
    };
    // A run on this machine, or another machine's pass, has the GPU.
    let lock = match Lock::take(&config.work_dir()) {
        Ok(lock) => lock,
        Err(why) => {
            say(&format!("BUSY {why}"));
            return Ok(0);
        }
    };
    let local = models::fetch(model, config, &|event| {
        if let Event::Message { text } = event {
            eprintln!("{SAYS}{text}");
        }
    })?;
    // One pass at a time here, under the lock: whatever is in it was left
    // by a pass that did not end.
    let dir = config.work_dir().join(DIR);
    if dir.exists() {
        fs::remove_dir_all(&dir).map_err(|e| Error::io(&dir, e))?;
    }
    for sub in ["in", "out"] {
        let path = dir.join(sub);
        fs::create_dir_all(&path).map_err(|e| Error::io(&path, e))?;
    }
    let mut command = worker::command(config, model, &local, inference, Some(idle_below_gib))?;
    say(&format!("DIR {}", dir.display()));
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| Error::Command {
            program: config.python().display().to_string(),
            status: "could not start".into(),
            detail: e.to_string(),
        })?;
    let status = forward_until_alone(&mut child);
    let _ = fs::remove_dir_all(&dir);
    drop(lock);
    Ok(status.and_then(|status| status.code()).unwrap_or(1))
}

/// Pass this process's standard input on to the worker's, line by line,
/// and wait for the worker. When the input ends — `QUIT` was sent and the
/// other machine went, or the other machine stopped this pass, or the link
/// died — the worker has [`QUIT_GRACE`] to end on its own before it is
/// killed: a pass nobody waits for any more must not keep the GPU, or the
/// lock that turns every next pass away.
fn forward_until_alone(child: &mut Child) -> Option<ExitStatus> {
    let alone = Stop::new();
    if let Some(mut to) = child.stdin.take() {
        let alone = alone.clone();
        // Not joined: it may sit on a read of a terminal that never ends,
        // and the process exits right after this.
        std::thread::spawn(move || {
            let mut input = std::io::stdin().lock();
            let mut line = Vec::new();
            loop {
                line.clear();
                match input.read_until(b'\n', &mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if to.write_all(&line).is_err() || to.flush().is_err() {
                            break;
                        }
                    }
                }
            }
            drop(to);
            alone.raise();
        });
    }
    let mut since: Option<std::time::Instant> = None;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) => {}
            Err(_) => break,
        }
        if alone.raised() {
            let since = *since.get_or_insert_with(std::time::Instant::now);
            if since.elapsed() >= QUIT_GRACE {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let _ = child.kill();
    child.wait().ok()
}

/// A stream read to its end on a thread of its own: a pipe nobody reads
/// fills up and blocks the program writing to it.
fn read_all(stream: Option<impl Read + Send + 'static>) -> Option<JoinHandle<String>> {
    stream.map(|mut stream| {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = stream.read_to_end(&mut bytes);
            String::from_utf8_lossy(&bytes).into_owned()
        })
    })
}

fn joined(reader: Option<JoinHandle<String>>) -> String {
    reader
        .and_then(|reader| reader.join().ok())
        .unwrap_or_default()
}

/// `io::copy`, keeping `count` up to date as it goes.
fn copy_counting(from: &mut impl Read, to: &mut impl Write, count: &AtomicU64) -> io::Result<u64> {
    let mut buffer = vec![0u8; 1 << 20];
    let mut total = 0;
    loop {
        let n = match from.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        to.write_all(&buffer[..n])?;
        total += n as u64;
        count.store(total, Ordering::Relaxed);
    }
    to.flush()?;
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quoted_word_survives_any_quote_inside_it() {
        assert_eq!(quote("plain"), "'plain'");
        assert_eq!(quote("it's"), "'it'\\''s'");
        assert_eq!(quote(r#"{"a":"b c"}"#), r#"'{"a":"b c"}'"#);
    }

    #[test]
    fn the_protocol_is_the_same_on_both_sides_of_one_build() {
        assert_eq!(protocol(), protocol());
        assert_eq!(protocol().len(), 16);
    }

    #[test]
    fn a_probe_reads_what_remote_check_prints() {
        let probe = Probe::parse(&format!(
            "OMNIFROG 0.1.0 {}\nDEVICE AMD Radeon RX 7900 XTX\nVRAM 0.42 23.98\n",
            protocol()
        ));
        assert_eq!(probe.version, "0.1.0");
        assert_eq!(probe.device, "AMD Radeon RX 7900 XTX");
        assert_eq!(probe.vram, Some((0.42, 23.98)));
        assert_eq!(probe.unfit(), None);
        assert_eq!(
            probe.describe(),
            "AMD Radeon RX 7900 XTX · 0.4 of 24 GiB in use"
        );
    }

    #[test]
    fn a_remote_without_a_gpu_or_of_another_version_is_left_out() {
        let cpu = Probe::parse(&format!("OMNIFROG 0.1.0 {}\nDEVICE cpu\n", protocol()));
        assert!(cpu.unfit().unwrap().contains("no GPU"));
        let other = Probe::parse("OMNIFROG 0.0.9 0123456789abcdef\nDEVICE Some GPU\n");
        assert!(other.unfit().unwrap().contains("another protocol"));
        let missing = Probe::parse(&format!(
            "OMNIFROG 0.1.0 {}\nMISSING no MSST checkout\n",
            protocol()
        ));
        assert_eq!(missing.unfit().unwrap(), "no MSST checkout");
        let nothing = Probe::parse("bash: omnifrog: command not found\n");
        assert!(nothing.unfit().unwrap().contains("no omnifrog answered"));
    }

    #[test]
    fn a_copy_counts_what_it_moved_as_it_goes() {
        let data = vec![7u8; (1 << 20) * 3 + 5];
        let count = AtomicU64::new(0);
        let mut out = Vec::new();
        let copied = copy_counting(&mut data.as_slice(), &mut out, &count).unwrap();
        assert_eq!(copied, data.len() as u64);
        assert_eq!(count.load(Ordering::Relaxed), data.len() as u64);
        assert_eq!(out, data);
    }

    #[test]
    fn the_worker_reads_the_memory_of_the_card_it_computes_on() {
        // Found by its device's PCI address, not as the largest card: an
        // 8 GiB card in use beside an idle 24 GiB one is busy.
        assert!(worker::SOURCE.contains("pci_bus_id"));
        assert!(worker::SOURCE.contains("--device"));
        assert!(worker::SOURCE.contains("\"idle=\""));
        assert!(
            worker::SOURCE.contains("BUSY another program holds"),
            "the worker no longer says BUSY the way this side reads it"
        );
    }
}
