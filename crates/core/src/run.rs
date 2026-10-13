//! Running an external program and reading its output line by line.
//!
//! Every tool this crate drives — ffmpeg, harletty, mkvmerge, pip, the
//! inference worker — talks in lines on its standard streams, and every
//! caller wants the same two things: each line as it arrives, and a clean
//! error carrying the last of them when the program fails.

use crate::{Error, Result};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;

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
        detail: e.to_string(),
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
            detail: e.to_string(),
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
