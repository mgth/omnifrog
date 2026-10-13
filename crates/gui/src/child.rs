//! The window's side of a worker: start it, read its replies, stop it.

use crate::task::{Reply, Request, WORKER_FLAG};
use omnifrog_core::progress::Event;
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};

/// A worker at work.
pub struct Running {
    /// Taken by [`Running::stop`], which hands the worker to a thread that
    /// waits on it — see [`reap`].
    child: Option<Child>,
    /// Held open: the worker takes its closing for the window's death.
    stdin: ChildStdin,
    replies: Receiver<Reply>,
    /// Set once the worker has said how the work ended: it is then exiting
    /// on its own, and is only waited on.
    ended: Arc<AtomicBool>,
    #[cfg(windows)]
    job: job::Job,
}

impl Running {
    /// Start a copy of this program on `request`; `wake` is called whenever
    /// a reply is waiting.
    pub fn start(
        request: &Request,
        wake: impl Fn() + Send + Clone + 'static,
    ) -> std::io::Result<Self> {
        let mut command = Command::new(program()?);
        command.arg(WORKER_FLAG);
        let mut running = Self::spawn(command, wake)?;
        // Once `running` holds the worker: should the write fail, it is
        // dropped, so stopped and reaped like any other.
        let line = serde_json::to_string(request).map_err(std::io::Error::other)?;
        writeln!(running.stdin, "{line}")?;
        running.stdin.flush()?;
        Ok(running)
    }

    /// Start `command` as a worker — piped, leading a process group of its
    /// own (held in a job object on Windows) — and read what it says.
    fn spawn(
        mut command: Command,
        wake: impl Fn() + Send + Clone + 'static,
    ) -> std::io::Result<Self> {
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = command
            .spawn()
            .map_err(|e| could_not_start(std::env::current_exe().ok().as_deref(), e))?;
        // In the job before it reads its request, so before it starts anything.
        #[cfg(windows)]
        let job = job::Job::holding(&child)?;
        let stdin = child.stdin.take().expect("piped");

        let (tx, replies) = channel();
        let ended = Arc::new(AtomicBool::new(false));
        let stdout = child.stdout.take().expect("piped");
        let stderr = child.stderr.take().expect("piped");
        {
            let tx = tx.clone();
            let wake = wake.clone();
            let ended = Arc::clone(&ended);
            std::thread::spawn(move || read_replies(stdout, tx, wake, &ended));
        }
        std::thread::spawn(move || read_stderr(stderr, tx, wake));
        Ok(Self {
            child: Some(child),
            stdin,
            replies,
            ended,
            #[cfg(windows)]
            job,
        })
    }

    /// Every reply that came in since the last call.
    pub fn poll(&self) -> Vec<Reply> {
        self.replies.try_iter().collect()
    }

    /// Stop the worker and everything it started. The replies it has still
    /// to give — its output ending, at the least — come in as before.
    pub fn stop(&mut self) {
        let Some(child) = self.child.take() else {
            return;
        };
        #[cfg(unix)]
        {
            let group = child.id() as libc::pid_t;
            // SAFETY: signals the process group the worker leads, which
            // holds only it and what it started.
            unsafe {
                libc::kill(-group, libc::SIGTERM);
            }
            // pip, ffmpeg and the inference worker all end on SIGTERM; one
            // that does not is not left to hold the GPU.
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_secs(5));
                unsafe {
                    libc::kill(-group, libc::SIGKILL);
                }
            });
        }
        #[cfg(windows)]
        self.job.terminate();
        reap(child);
    }
}

impl Drop for Running {
    /// A worker that said how the work ended is exiting on its own, with
    /// nothing left under it: it is only waited on. One whose output ended
    /// without a word is reaped here and now if it has ended, and stopped
    /// otherwise — as is one let go of while still at work.
    fn drop(&mut self) {
        if self.ended.load(Ordering::Acquire) {
            if let Some(child) = self.child.take() {
                reap(child);
            }
            return;
        }
        if let Some(child) = &mut self.child
            && matches!(child.try_wait(), Ok(Some(_)))
        {
            return;
        }
        self.stop();
    }
}

/// Wait on a worker off the window's thread, so it does not stay a zombie
/// beside the window once it has ended. It has, or is about to: its last
/// reply comes just before it exits, and a stopped one ends on the signal
/// or, at the latest, on the kill five seconds after.
fn reap(mut child: Child) {
    std::thread::spawn(move || {
        let _ = child.wait();
    });
}

/// The program a worker is: this one. On Linux, `/proc/self/exe` rather
/// than the path it resolves to, so the worker is the build the window
/// runs even once `cargo build` has replaced the file under it — the
/// resolved path then ends in « (deleted) » and starts nothing.
fn program() -> std::io::Result<PathBuf> {
    if cfg!(target_os = "linux") {
        Ok(PathBuf::from("/proc/self/exe"))
    } else {
        std::env::current_exe()
    }
}

/// A failure to start the worker, worded with the program it was and —
/// when that file is gone — why: the window outlived its own build.
fn could_not_start(exe: Option<&Path>, e: std::io::Error) -> std::io::Error {
    let shown = exe.map_or_else(|| "this program".to_string(), |p| p.display().to_string());
    let gone = e.kind() == ErrorKind::NotFound && exe.is_none_or(|p| !p.is_file());
    let message = if gone {
        format!(
            "{shown}: {e}; the program was replaced or removed since this window \
             started (rebuilt?), start omnifrog again"
        )
    } else {
        format!("{shown}: {e}")
    };
    std::io::Error::new(e.kind(), message)
}

/// The worker's replies, then — when its output ends without one that
/// says how the work ended — a failure that says so: it was killed, or
/// the machine ran out of memory under it. `ended` is set, before the
/// reply is passed on, when the worker says how the work ended.
fn read_replies(
    stdout: impl std::io::Read,
    tx: Sender<Reply>,
    wake: impl Fn(),
    ended: &AtomicBool,
) {
    for line in BufReader::new(stdout).lines() {
        let Ok(line) = line else { break };
        let reply = serde_json::from_str::<Reply>(&line).unwrap_or_else(|_| Reply::Progress {
            event: Event::message(line),
        });
        if !matches!(reply, Reply::Progress { .. }) {
            ended.store(true, Ordering::Release);
        }
        if tx.send(reply).is_err() {
            return;
        }
        wake();
    }
    if !ended.load(Ordering::Relaxed) {
        let _ = tx.send(Reply::Failed {
            error: "the worker stopped without saying how the work ended".into(),
        });
        wake();
    }
}

/// What the worker prints outside its replies: a panic's message, a
/// library's warning. Shown in the log as it is.
fn read_stderr(stderr: impl std::io::Read, tx: Sender<Reply>, wake: impl Fn()) {
    for line in BufReader::new(stderr).lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        if tx
            .send(Reply::Progress {
                event: Event::message(line),
            })
            .is_err()
        {
            break;
        }
        wake();
    }
}

#[cfg(windows)]
mod job {
    //! A job object that kills everything in it when it is terminated or
    //! when its last handle — the window's — closes.

    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject, TerminateJobObject,
    };

    pub struct Job(HANDLE);

    // SAFETY: a job handle is a kernel handle, usable from any thread.
    unsafe impl Send for Job {}

    impl Job {
        pub fn holding(child: &std::process::Child) -> std::io::Result<Self> {
            // SAFETY: plain Win32 calls on a handle this function owns and
            // on the child's, which outlives the call.
            unsafe {
                let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                if handle.is_null() {
                    return Err(std::io::Error::last_os_error());
                }
                let job = Job(handle);
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                if SetInformationJobObject(
                    handle,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const _,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                ) == 0
                {
                    return Err(std::io::Error::last_os_error());
                }
                if AssignProcessToJobObject(handle, child.as_raw_handle() as HANDLE) == 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(job)
            }
        }

        pub fn terminate(&self) {
            // SAFETY: the handle is open until drop.
            unsafe {
                TerminateJobObject(self.0, 1);
            }
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            // SAFETY: closed once; with KILL_ON_JOB_CLOSE this ends the
            // worker if it is still running.
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gone_program_is_named_and_explained() {
        let exe = PathBuf::from("/builds/omnifrog-gui (deleted)");
        let e = std::io::Error::from(ErrorKind::NotFound);
        let message = could_not_start(Some(&exe), e).to_string();
        assert!(
            message.starts_with("/builds/omnifrog-gui (deleted): "),
            "{message}"
        );
        assert!(message.contains("replaced or removed"), "{message}");
        assert!(message.contains("start omnifrog again"), "{message}");
    }

    #[test]
    fn a_program_still_there_is_only_named() {
        let exe = std::env::current_exe().unwrap();
        let e = std::io::Error::from(ErrorKind::PermissionDenied);
        let message = could_not_start(Some(&exe), e).to_string();
        assert!(message.starts_with(&exe.display().to_string()), "{message}");
        assert!(!message.contains("replaced"), "{message}");
        let unknown = could_not_start(None, std::io::Error::from(ErrorKind::NotFound)).to_string();
        assert!(unknown.starts_with("this program: "), "{unknown}");
        assert!(unknown.contains("replaced or removed"), "{unknown}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_worker_program_starts() {
        // This test binary, asked for its own help: the path itself is what
        // is under test, not what the program says.
        let status = Command::new(program().unwrap())
            .arg("--help")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(status.success());
    }

    /// Whether `pid` is gone: a worker that ended but was never waited on
    /// is still there, as a zombie.
    #[cfg(unix)]
    fn reaped(pid: u32) -> bool {
        // SAFETY: signal 0 sends nothing; it only asks whether the process
        // exists.
        let gone = unsafe { libc::kill(pid as libc::pid_t, 0) } == -1;
        gone && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
    }

    /// Whether `what` holds within ten seconds.
    #[cfg(unix)]
    fn soon(mut what: impl FnMut() -> bool) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !what() {
            if std::time::Instant::now() > deadline {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        true
    }

    #[cfg(unix)]
    #[test]
    fn a_worker_that_ended_is_reaped_once_let_go() {
        // This test binary, asked for its help: a worker whose output ends
        // at once, without a reply that says how the work ended.
        let mut command = Command::new(program().unwrap());
        command.arg("--help");
        let running = Running::spawn(command, || {}).unwrap();
        let pid = running.child.as_ref().unwrap().id();
        assert!(
            soon(|| running
                .poll()
                .iter()
                .any(|reply| matches!(reply, Reply::Failed { .. }))),
            "the worker's output did not end"
        );
        // Ended, or about to: the window lets it go the moment it reads
        // the reply, the way `Run::finish` does.
        drop(running);
        assert!(soon(|| reaped(pid)), "worker {pid} is still there");
    }

    #[cfg(unix)]
    #[test]
    fn a_worker_that_said_how_it_ended_is_left_to_exit_on_its_own() {
        // A worker that replies, then takes its time to exit: left alone,
        // it gets to leave its mark; stopped, it would not.
        let dir = std::env::temp_dir().join(format!("omnifrog-child-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mark = dir.join("exited-on-its-own");
        let done = serde_json::to_string(&Reply::Done { output: None }).unwrap();
        let mut command = Command::new("sh");
        command.arg("-c").arg(format!(
            "printf '%s\n' '{done}'; sleep 0.3; : > '{}'",
            mark.display()
        ));
        let running = Running::spawn(command, || {}).unwrap();
        let pid = running.child.as_ref().unwrap().id();
        assert!(
            soon(|| running
                .poll()
                .iter()
                .any(|reply| matches!(reply, Reply::Done { .. }))),
            "the worker did not say it was done"
        );
        drop(running);
        assert!(
            soon(|| mark.is_file()),
            "the worker was not left to exit on its own"
        );
        assert!(soon(|| reaped(pid)), "worker {pid} is still there");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn a_worker_stopped_at_work_is_reaped() {
        let mut command = Command::new("sleep");
        command.arg("30");
        let mut running = Running::spawn(command, || {}).unwrap();
        let pid = running.child.as_ref().unwrap().id();
        assert!(!reaped(pid));
        running.stop();
        assert!(soon(|| reaped(pid)), "worker {pid} is still there");
        // Stopped already: dropping it has nothing left to do.
        running.stop();
        drop(running);
    }

    #[cfg(unix)]
    #[test]
    fn a_worker_let_go_at_work_is_stopped_and_reaped() {
        let mut command = Command::new("sleep");
        command.arg("30");
        let running = Running::spawn(command, || {}).unwrap();
        let pid = running.child.as_ref().unwrap().id();
        drop(running);
        assert!(soon(|| reaped(pid)), "worker {pid} is still there");
    }
}
