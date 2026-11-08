//! Separating on another machine's GPU, over SSH.
//!
//! A remote is a machine with omnifrog installed and set up there (`omnifrog
//! setup` on it), that `ssh` reaches with a key and no question
//! ([`crate::config::Remote`]). Two commands of that machine's omnifrog are
//! all this side runs there, both hidden from its help:
//!
//! - `remote-check`, once a step: its version, the device torch would run
//!   on, and how much of that device's memory is in use ([`probe`]). A
//!   remote of another version, without a GPU torch can use, or under a
//!   ROCm override its card cannot run ([`crate::rocm`]), is left out.
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
//! link against minutes of inference. An element the remote follows during
//! the extraction goes there a piece at a time — what its waveform gained
//! since the last block, appended (`cat >>`) — and each block's stems come
//! back as it is made.
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
use std::io::{self, BufRead, Read, Seek, SeekFrom, Write};
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

/// The host the tests' remote answers to: its commands run in this
/// machine's shell.
#[cfg(test)]
pub(crate) const SHELL: &str = "shell.invalid";

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
    /// The agent's socket, given to ssh when this process has no
    /// `SSH_AUTH_SOCK` (see [`agent`]).
    agent: Option<PathBuf>,
}

impl Ssh {
    pub fn new(host: &str) -> Self {
        let control = control_path([
            dirs::runtime_dir(),
            dirs::home_dir().map(|home| home.join(".ssh")),
        ]);
        let agent = match agent() {
            Agent::Found(socket) => Some(socket),
            Agent::Known | Agent::None => None,
        };
        Self {
            host: host.to_string(),
            control,
            stop: Stop::new(),
            agent,
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
        // The tests' remote: this machine's shell, standing in for ssh.
        #[cfg(test)]
        if self.host == SHELL {
            let mut command = crate::run::command("sh");
            command.arg("-c").arg(line);
            return command;
        }
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
        if let Some(agent) = &self.agent {
            command.env("SSH_AUTH_SOCK", agent);
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
        let file = fs::File::open(from).map_err(|e| Error::io(from, e))?;
        self.feed(from, file, &format!("cat > {}", quote(to)))
            .map(|_| ())
    }

    /// Copy `length` bytes of `from`, from byte `offset`, to the path `to`
    /// on that machine: a new file there, its directory made, when `offset`
    /// is 0, and added to the end of it otherwise. A file still growing here
    /// goes there a piece at a time.
    pub fn send(&self, from: &Path, offset: u64, length: u64, to: &str) -> Result<()> {
        let mut file = fs::File::open(from).map_err(|e| Error::io(from, e))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| Error::io(from, e))?;
        let line = if offset == 0 {
            let dir = to.rsplit_once('/').map_or(".", |(dir, _)| dir);
            format!("mkdir -p -- {} && cat > {}", quote(dir), quote(to))
        } else {
            format!("cat >> {}", quote(to))
        };
        let sent = self.feed(from, file.take(length), &line)?;
        if sent != length {
            return Err(self.failed(
                "cut short",
                format!(
                    "sending {}: {sent} of its {length} bytes from {offset}",
                    from.display()
                ),
            ));
        }
        Ok(())
    }

    /// Run `line` there, its input `source` (read from `from`); returns the
    /// bytes it was given.
    fn feed(&self, from: &Path, mut source: impl Read + Send + 'static, line: &str) -> Result<u64> {
        let mut child = self
            .command(line)
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
            std::thread::spawn(move || copy_counting(&mut source, &mut stdin, &sent))
        });
        let moved = || sent.load(Ordering::Relaxed);
        let what = |detail: &str| format!("sending {}: {}", from.display(), detail.trim());
        // Killed, nothing more is waited for (see `run`).
        let status = self
            .watch(&mut child, None, Some(&moved))
            .map_err(|cut| self.failed(cut, what("")))?;
        let copied = feed.map(|feed| feed.join());
        let err = joined(err);
        if !status.success() {
            return Err(self.failed(&status.to_string(), what(&err)));
        }
        match copied {
            Some(Ok(Err(e))) => Err(Error::io(from, e)),
            _ => Ok(moved()),
        }
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

/// The room a Unix socket's path has: 104 bytes on macOS, 108 on Linux.
const SUN_PATH: usize = 104;

/// Where the connection to a remote is kept open for the next commands, on
/// Unix: the first of `dirs` that is there and short enough. `%C` becomes
/// 40 characters, and ssh adds 17 more while it sets the socket up; a path
/// past [`SUN_PATH`] fails every command (« ControlPath too long ») — the
/// temporary directory of macOS is that long. None fits: no sharing, a
/// handshake per command.
fn control_path(dirs: [Option<PathBuf>; 2]) -> Option<PathBuf> {
    if !cfg!(unix) {
        return None;
    }
    dirs.into_iter()
        .flatten()
        .filter(|dir| dir.is_dir())
        .map(|dir| dir.join("omnifrog-ssh-%C"))
        .find(|path| path.as_os_str().len() - "%C".len() + 40 + 17 < SUN_PATH)
}

/// Where agents listen when nothing says so, under `$XDG_RUNTIME_DIR`:
/// OpenSSH's systemd user unit (`ssh-agent.socket`), then GNOME's — gcr's,
/// and the older gnome-keyring's.
const AGENT_SOCKETS: [&str; 3] = ["ssh-agent.socket", "gcr/ssh", "keyring/ssh"];

/// Where ssh will find the agent holding the keys, as this process sees it.
#[derive(Debug, Clone, PartialEq)]
pub enum Agent {
    /// ssh knows: `SSH_AUTH_SOCK` is set, or, on Windows, the system's own
    /// agent service is what ssh asks.
    Known,
    /// `SSH_AUTH_SOCK` is not set here, and this socket, where agents
    /// usually listen, is there: it is given to ssh.
    Found(PathBuf),
    /// Neither: ssh can only use a key without a passphrase.
    None,
}

/// Where the agent is. A program started from a desktop's menu has no
/// `SSH_AUTH_SOCK` when only a shell's startup file exports it; ssh then
/// cannot use a key with a passphrase, even one loaded in the agent, and
/// every remote answers « Permission denied ».
pub fn agent() -> Agent {
    agent_among(std::env::var_os("SSH_AUTH_SOCK"), dirs::runtime_dir())
}

fn agent_among(set: Option<std::ffi::OsString>, runtime: Option<PathBuf>) -> Agent {
    if set.is_some_and(|socket| !socket.is_empty()) || !cfg!(unix) {
        return Agent::Known;
    }
    runtime
        .and_then(|dir| {
            AGENT_SOCKETS
                .iter()
                .map(|socket| dir.join(socket))
                .find(|socket| answers(socket))
        })
        .map_or(Agent::None, Agent::Found)
}

/// Something listens at `socket` and takes a connection at once. An agent
/// that died leaves its socket behind, and a connection to it is refused;
/// one whose queue is full would make a plain connection wait for as long
/// as it stays full. The connection is made without waiting, so neither
/// holds anything up: the next socket is tried. A socket systemd holds for
/// an agent not started yet starts it.
#[cfg(unix)]
fn answers(socket: &Path) -> bool {
    use socket2::{Domain, SockAddr, Socket, Type};
    let Ok(address) = SockAddr::unix(socket) else {
        return false;
    };
    Socket::new(Domain::UNIX, Type::STREAM, None)
        .and_then(|stream| {
            stream.set_nonblocking(true)?;
            stream.connect(&address)
        })
        .is_ok()
}

#[cfg(not(unix))]
fn answers(_: &Path) -> bool {
    false
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

/// What can be said of a remote before a step: `omnifrog remotes`, and the
/// window's setup tab.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// ssh got no answer: the machine is off, or the key is not loaded.
    Unreachable(String),
    /// It answered, and cannot take passes: why.
    LeftOut(String),
    /// It can take passes. `busy`: other programs hold its GPU now, so its
    /// worker would wait for them.
    Ready { probe: Probe, busy: bool },
}

impl Verdict {
    /// Ask `remote`, over SSH: up to two minutes for one that hangs.
    pub fn of(remote: &crate::config::Remote) -> Self {
        Self::judge(remote, probe(remote, &Ssh::new(&remote.host)))
    }

    fn judge(remote: &crate::config::Remote, probed: Result<Probe>) -> Self {
        match probed {
            Err(why) => Verdict::Unreachable(why.to_string()),
            Ok(probe) => match probe.unfit() {
                Some(why) => Verdict::LeftOut(why),
                None => {
                    let busy = probe.vram.is_some_and(|(used, _)| {
                        remote.idle_below_gib > 0.0 && used >= remote.idle_below_gib
                    });
                    Verdict::Ready { probe, busy }
                }
            },
        }
    }

    /// One line, after the host.
    pub fn describe(&self) -> String {
        match self {
            Verdict::Unreachable(why) => format!("unreachable: {why}"),
            Verdict::LeftOut(why) => format!("left out: {why}"),
            Verdict::Ready { probe, busy } => format!(
                "ready: {} (omnifrog {}){}",
                probe.describe(),
                probe.version,
                if *busy {
                    " — busy now: its worker waits for the other programs there"
                } else {
                    ""
                }
            ),
        }
    }
}

/// The remote's side of [`probe`]: what `remote-check` prints.
pub fn check(config: &Config) -> Vec<String> {
    check_under(config, crate::rocm::check())
}

/// [`check`], with what [`crate::rocm::check`] said of this machine.
fn check_under(config: &Config, rocm: Option<crate::rocm::Report>) -> Vec<String> {
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
    } else if let Some(why) = rocm.and_then(|report| report.broken) {
        // Naming the device has torch open the card, under an override it
        // cannot run: say why instead, and leave Python alone.
        lines.push(format!("MISSING {why}"));
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
    fn a_verdict_says_why_a_remote_cannot_take_passes_or_how_busy_it_is() {
        let remote = crate::config::Remote::new("user@gpu-box");
        let ready = |vram: &str| {
            Verdict::judge(
                &remote,
                Ok(Probe::parse(&format!(
                    "OMNIFROG 0.1.0 {}\nDEVICE AMD Radeon RX 7900 XTX\n{vram}",
                    protocol()
                ))),
            )
        };
        assert!(matches!(
            ready("VRAM 0.03 23.98\n"),
            Verdict::Ready { busy: false, .. }
        ));
        let busy = ready("VRAM 20.40 23.98\n");
        assert!(matches!(busy, Verdict::Ready { busy: true, .. }));
        assert!(busy.describe().contains("busy now"));
        // Nothing read: not called busy.
        assert!(matches!(ready(""), Verdict::Ready { busy: false, .. }));
        let cpu = Verdict::judge(
            &remote,
            Ok(Probe::parse(&format!(
                "OMNIFROG 0.1.0 {}\nDEVICE cpu\n",
                protocol()
            ))),
        );
        assert!(cpu.describe().starts_with("left out: torch sees no GPU"));
        let off = Verdict::judge(&remote, Err(Error::msg("connection timed out")));
        assert_eq!(off.describe(), "unreachable: connection timed out");
    }

    #[cfg(unix)]
    #[test]
    fn the_shared_connection_lives_where_its_path_fits_a_socket() {
        let short = std::env::temp_dir().join(format!("of-{}", std::process::id()));
        let long = short.join("a-directory-whose-name-makes-the-socket-path-too-long-for-ssh");
        fs::create_dir_all(&long).unwrap();
        let missing = short.join("missing");
        assert_eq!(
            control_path([Some(long.clone()), Some(short.clone())]),
            Some(short.join("omnifrog-ssh-%C"))
        );
        assert_eq!(control_path([Some(missing), None]), None);
        assert_eq!(control_path([Some(long), None]), None);
        fs::remove_dir_all(&short).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn without_ssh_auth_sock_the_usual_agent_socket_is_found() {
        use std::os::unix::net::UnixListener;
        let dir = std::env::temp_dir().join(format!("omnifrog-agent-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("gcr")).unwrap();
        let among = |set: Option<&str>| agent_among(set.map(Into::into), Some(dir.clone()));
        let first = dir.join("ssh-agent.socket");
        let second = dir.join("gcr/ssh");
        assert_eq!(among(None), Agent::None);
        let _gcr = UnixListener::bind(&second).unwrap();
        assert_eq!(among(None), Agent::Found(second.clone()));
        // An agent that died left its socket: nobody answers there, the
        // next one is taken.
        drop(UnixListener::bind(&first).unwrap());
        assert!(first.exists());
        assert_eq!(among(None), Agent::Found(second.clone()));
        // A file of that name is no agent either.
        fs::remove_file(&first).unwrap();
        fs::write(&first, b"").unwrap();
        assert_eq!(among(None), Agent::Found(second.clone()));
        // One whose queue is full is passed over at once, not waited on.
        fs::remove_file(&first).unwrap();
        {
            use socket2::{Domain, SockAddr, Socket, Type};
            let full = Socket::new(Domain::UNIX, Type::STREAM, None).unwrap();
            full.bind(&SockAddr::unix(&first).unwrap()).unwrap();
            full.listen(1).unwrap();
            let mut waiting = Vec::new();
            while answers(&first) {
                let client = Socket::new(Domain::UNIX, Type::STREAM, None).unwrap();
                client.set_nonblocking(true).unwrap();
                client.connect(&SockAddr::unix(&first).unwrap()).unwrap();
                waiting.push(client);
                assert!(waiting.len() < 64, "the queue never filled");
            }
            let started = std::time::Instant::now();
            assert_eq!(among(None), Agent::Found(second.clone()));
            assert!(started.elapsed() < Duration::from_secs(1));
        }
        // OpenSSH's own unit first, when it answers.
        fs::remove_file(&first).unwrap();
        let _openssh = UnixListener::bind(&first).unwrap();
        assert_eq!(among(None), Agent::Found(first.clone()));
        // Whatever the environment says is what ssh uses.
        assert_eq!(among(Some("/elsewhere/agent")), Agent::Known);
        assert_eq!(among(Some("")), Agent::Found(first.clone()));
        assert_eq!(agent_among(None, None), Agent::None);
        fs::remove_dir_all(&dir).unwrap();
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

    #[test]
    fn under_an_override_the_card_cannot_run_the_check_leaves_python_alone() {
        let dir = std::env::temp_dir().join(format!("omnifrog-remote-rocm-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // Not a program: had the check started it to name the device, it
        // would say that the worker does not start.
        let python = dir.join("python");
        fs::write(&python, "not a program\n").unwrap();
        let config = Config {
            python: Some(python),
            msst_dir: Some(dir.clone()),
            ..Config::default()
        };
        let why = "HSA_OVERRIDE_GFX_VERSION=11.0.0 tells ROCm the card the separation runs on is \
                   a gfx1100, but it is a gfx1201, of another generation";
        let broken = crate::rocm::Report {
            summary: "gfx1201".to_string(),
            broken: Some(why.to_string()),
            warnings: Vec::new(),
        };
        let lines = check_under(&config, Some(broken));
        assert!(
            !lines.iter().any(|line| line.contains("does not start")),
            "{lines:?}"
        );
        assert!(
            !lines.iter().any(|line| line.starts_with("DEVICE ")),
            "{lines:?}"
        );
        // This side leaves the remote out, and says why.
        let probe = Probe::parse(&lines.join("\n"));
        assert_eq!(probe.unfit().as_deref(), Some(why));
        let _ = fs::remove_dir_all(&dir);
    }
}
