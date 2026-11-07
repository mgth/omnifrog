//! Running an external program and reading its output line by line.
//!
//! Every tool this crate drives — ffmpeg, harletty, mkvmerge, pip, the
//! inference worker — talks in lines on its standard streams, and every
//! caller wants the same two things: each line as it arrives, and a clean
//! error carrying the last of them when the program fails.

use crate::{Error, Result};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// A flag one thread raises to have the programs another one runs under it
/// stopped: whoever waits for such a program looks at it as it waits
/// ([`watch`]). Clones share the flag.
#[derive(Debug, Clone, Default)]
pub struct Stop(Arc<AtomicBool>);

impl Stop {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn raise(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn lower(&self) {
        self.0.store(false, Ordering::SeqCst);
    }

    pub fn raised(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Why [`watch`] killed a program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cut {
    /// Its [`Stop`] was raised.
    Stopped,
    /// It ran past its time.
    TimedOut,
    /// What it moves did not move for the time it is allowed to stall.
    Stalled,
}

impl Cut {
    pub fn describe(self) -> &'static str {
        match self {
            Cut::Stopped => "stopped",
            Cut::TimedOut => "timed out",
            Cut::Stalled => "stalled",
        }
    }
}

/// How often a program under [`watch`] is looked at.
const TICK: Duration = Duration::from_millis(100);

/// Wait for `child`, killing and reaping it when `stop` is raised, when
/// `within` has passed, or when `moved` — a count of what it moved so far —
/// stays the same for its duration. `Ok(Err(cut))` is a program killed for
/// that reason.
pub fn watch(
    child: &mut Child,
    stop: Option<&Stop>,
    within: Option<Duration>,
    moved: Option<(&dyn Fn() -> u64, Duration)>,
) -> std::io::Result<std::result::Result<ExitStatus, Cut>> {
    let start = Instant::now();
    let mut last = moved.map(|(count, _)| count());
    let mut since = start;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Ok(status));
        }
        let cut = if stop.is_some_and(Stop::raised) {
            Some(Cut::Stopped)
        } else if within.is_some_and(|limit| start.elapsed() >= limit) {
            Some(Cut::TimedOut)
        } else if let Some((count, quiet)) = moved {
            let now = count();
            if last != Some(now) {
                last = Some(now);
                since = Instant::now();
                None
            } else {
                (since.elapsed() >= quiet).then_some(Cut::Stalled)
            }
        } else {
            None
        };
        if let Some(cut) = cut {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(Err(cut));
        }
        thread::sleep(TICK);
    }
}

/// A command for `program`, as every program this crate starts is made.
///
/// On Windows a console program started by one without a console — the
/// egui client, or its worker — opens a window of its own; every tool here
/// is read through pipes, so none of them needs one.
pub fn command(program: impl AsRef<std::ffi::OsStr>) -> Command {
    #[allow(unused_mut)]
    let mut command = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
}

/// Why a program could not be started, in words for the person who chose
/// the file, with the system's own code.
///
/// Windows says only "%1 is not a valid Win32 application" when the file
/// is a library, an archive or a program built for another system.
pub fn start_error(e: &std::io::Error) -> String {
    let what = match e.raw_os_error() {
        // ERROR_BAD_EXE_FORMAT, or ERROR_EXE_MACHINE_TYPE_MISMATCH, which
        // Windows also answers for a file that is not a program at all.
        Some(193 | 216) if cfg!(windows) => {
            "not a Windows program for this machine — a library, an archive, or a program for \
             another system or processor"
        }
        // ENOEXEC
        Some(8) if cfg!(unix) => "not a program for this system",
        _ if e.kind() == std::io::ErrorKind::PermissionDenied => {
            "not allowed to run it — is it a program, and executable?"
        }
        _ => return e.to_string(),
    };
    match e.raw_os_error() {
        Some(code) => format!("{what} (os error {code})"),
        None => what.to_string(),
    }
}

/// How many trailing lines an error report keeps.
const TAIL: usize = 40;

/// Which stream a line came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Stdout,
    Stderr,
}

/// Run `command`, feeding every line of both streams to `on_line` as it
/// arrives, and fail with the last lines when the status is not success.
pub fn lines(command: Command, mut on_line: impl FnMut(Stream, &str)) -> Result<()> {
    lines_until(command, |stream, line| {
        on_line(stream, line);
        true
    })
    .map(|_| ())
}

/// [`lines`], for a caller that may have to stop the program: `on_line`
/// returns `false` to have it killed. Returns `Ok(false)` when it was
/// stopped that way — a program told to stop has no status worth failing
/// on — and `Ok(true)` when it ran to a successful end.
///
/// For a program that can hang rather than fail: Dee under Wine, after an
/// access violation, stops making progress and never exits, and the only
/// sign of it is the line it printed on the way.
pub fn lines_until(command: Command, on_line: impl FnMut(Stream, &str) -> bool) -> Result<bool> {
    let ran = drive(command, on_line)?;
    if ran.stopped {
        Ok(false)
    } else if ran.status.success() {
        Ok(true)
    } else {
        Err(Error::Command {
            program: ran.program,
            status: ran.status.to_string(),
            detail: ran.tail.join("\n"),
        })
    }
}

/// [`lines`], for a program whose exit code means more than pass or fail:
/// mkvmerge exits 1 when it only warned and the file is written. Returns the
/// status and the last lines, and leaves the reading of them to the caller.
pub fn lines_status(
    command: Command,
    mut on_line: impl FnMut(Stream, &str),
) -> Result<(ExitStatus, Vec<String>)> {
    let ran = drive(command, |stream, line| {
        on_line(stream, line);
        true
    })?;
    Ok((ran.status, ran.tail))
}

/// How a program ended.
struct Ran {
    program: String,
    status: ExitStatus,
    /// `on_line` asked for it to be killed.
    stopped: bool,
    /// Its last [`TAIL`] lines, both streams together.
    tail: Vec<String>,
}

fn drive(mut command: Command, mut on_line: impl FnMut(Stream, &str) -> bool) -> Result<Ran> {
    let program = command.get_program().to_string_lossy().into_owned();
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().map_err(|e| Error::Command {
        program: program.clone(),
        status: "could not start".into(),
        detail: start_error(&e),
    })?;
    let (tx, rx) = mpsc::channel::<(Stream, String)>();
    let mut pumps = Vec::new();
    if let Some(out) = child.stdout.take() {
        let tx = tx.clone();
        pumps.push(thread::spawn(move || pump(out, Stream::Stdout, tx)));
    }
    if let Some(err) = child.stderr.take() {
        let tx = tx.clone();
        pumps.push(thread::spawn(move || pump(err, Stream::Stderr, tx)));
    }
    drop(tx);
    let mut tail = VecDeque::with_capacity(TAIL);
    let mut stopped = false;
    for (stream, line) in rx {
        if !stopped && !on_line(stream, &line) {
            stopped = true;
            // The pumps end when the pipes close, which is when it dies.
            let _ = child.kill();
        }
        if tail.len() == TAIL {
            tail.pop_front();
        }
        tail.push_back(line);
    }
    for pump in pumps {
        let _ = pump.join();
    }
    let status = child.wait().map_err(|e| Error::Command {
        program: program.clone(),
        status: "could not wait".into(),
        detail: e.to_string(),
    })?;
    Ok(Ran {
        program,
        status,
        stopped,
        tail: tail.into(),
    })
}

/// Run `command` and return its standard output, trimmed, failing with its
/// standard error when the status is not success.
pub fn output(mut command: Command) -> Result<String> {
    let program = command.get_program().to_string_lossy().into_owned();
    let out = command
        .stdin(Stdio::null())
        .output()
        .map_err(|e| Error::Command {
            program: program.clone(),
            status: "could not start".into(),
            detail: start_error(&e),
        })?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        Err(Error::Command {
            program,
            status: out.status.to_string(),
            detail: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        })
    }
}

/// Read one of a child's streams to its end on a thread of its own,
/// keeping the last [`TAIL`] lines.
///
/// Every process this crate drives has to have its standard error drained:
/// a pipe nobody reads fills up, and the child then blocks forever on its
/// next warning. Keeping only the tail bounds what a very chatty one costs
/// — the raw TrueHD muxer writes a line per packet, about 1.4 million of
/// them over a film — while keeping the line that says what went wrong.
pub fn tail(reader: impl std::io::Read + Send + 'static) -> thread::JoinHandle<Vec<String>> {
    thread::spawn(move || {
        let mut lines = VecDeque::with_capacity(TAIL);
        for line in BufReader::new(reader).split(b'\n') {
            let Ok(mut bytes) = line else { break };
            if bytes.last() == Some(&b'\r') {
                bytes.pop();
            }
            let text = String::from_utf8_lossy(&bytes);
            let text = text.rsplit('\r').next().unwrap_or("").trim().to_string();
            if text.is_empty() {
                continue;
            }
            if lines.len() == TAIL {
                lines.pop_front();
            }
            lines.push_back(text);
        }
        lines.into()
    })
}

fn pump(reader: impl std::io::Read, stream: Stream, tx: mpsc::Sender<(Stream, String)>) {
    let reader = BufReader::new(reader);
    for line in reader.split(b'\n') {
        let Ok(mut bytes) = line else { break };
        if bytes.last() == Some(&b'\r') {
            bytes.pop();
        }
        // A progress bar rewrites its line with `\r`; keep its last state.
        let text = String::from_utf8_lossy(&bytes);
        let text = text.rsplit('\r').next().unwrap_or("").to_string();
        if tx.send((stream, text)).is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A program that runs for half a minute unless killed.
    fn long() -> Child {
        let mut command = if cfg!(windows) {
            let mut ping = super::command("ping");
            ping.args(["-n", "30", "127.0.0.1"]);
            ping
        } else {
            let mut sleep = super::command("sleep");
            sleep.arg("30");
            sleep
        };
        command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap()
    }

    #[test]
    fn a_program_that_ends_is_its_status() {
        let mut command = if cfg!(windows) {
            let mut cmd = super::command("cmd");
            cmd.args(["/C", "exit 0"]);
            cmd
        } else {
            super::command("true")
        };
        let mut child = command.spawn().unwrap();
        let ended = watch(&mut child, None, Some(Duration::from_secs(20)), None).unwrap();
        assert!(ended.unwrap().success());
    }

    #[test]
    fn a_program_past_its_time_is_killed() {
        let started = Instant::now();
        let mut child = long();
        let ended = watch(&mut child, None, Some(Duration::from_millis(300)), None).unwrap();
        assert_eq!(ended, Err(Cut::TimedOut));
        assert!(started.elapsed() < Duration::from_secs(10));
        // Reaped: nothing left to wait for.
        assert!(child.try_wait().unwrap().is_some());
    }

    #[test]
    fn a_raised_stop_kills_it_from_another_thread() {
        let stop = Stop::new();
        let mut child = long();
        let started = Instant::now();
        let ended = thread::scope(|scope| {
            scope.spawn(|| {
                thread::sleep(Duration::from_millis(200));
                stop.raise();
            });
            watch(&mut child, Some(&stop), None, None).unwrap()
        });
        assert_eq!(ended, Err(Cut::Stopped));
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn a_program_that_moves_nothing_is_stalled_and_one_that_moves_is_not() {
        let mut child = long();
        let nothing = || 0;
        let ended = watch(
            &mut child,
            None,
            None,
            Some((&nothing, Duration::from_millis(300))),
        )
        .unwrap();
        assert_eq!(ended, Err(Cut::Stalled));

        // Moving all the time, it runs until its time is up instead.
        let mut child = long();
        let counter = std::sync::atomic::AtomicU64::new(0);
        let moving = || counter.fetch_add(1, Ordering::Relaxed);
        let ended = watch(
            &mut child,
            None,
            Some(Duration::from_millis(800)),
            Some((&moving, Duration::from_millis(300))),
        )
        .unwrap();
        assert_eq!(ended, Err(Cut::TimedOut));
    }
}
