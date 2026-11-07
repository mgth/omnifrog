//! A run's own log: every line it reported, timed, in a file.
//!
//! What a run said is otherwise only on a terminal, or in a window that is
//! closed by the time someone asks why a film came out as it did. The log
//! lives in the job's directory while the run goes, so a run that stops
//! halfway leaves it beside the steps it finished, and a resumed run adds
//! to it; once the track is written, it is copied beside it under the same
//! name (`….log`), after the temporaries are gone, and the job's copy
//! goes last; a cleanup that fails leaves it, with why.
//!
//! Lines are kept in memory until the job's directory is known — the probe
//! of the film comes first — then written as they come, so a crash loses
//! nothing. Writing it never fails a run: a log that cannot be written is
//! a log that is not there.

use crate::progress::Event;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// The name of the log in the job's directory.
pub const FILE_NAME: &str = "omnifrog.log";

pub struct RunLog {
    state: Mutex<State>,
}

struct State {
    start: Instant,
    /// When the run started, for the line that opens it.
    started: SystemTime,
    /// Lines written before there was a file to write them to.
    pending: Vec<String>,
    file: Option<fs::File>,
    path: Option<PathBuf>,
}

impl Default for RunLog {
    fn default() -> Self {
        Self::new()
    }
}

impl RunLog {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State {
                start: Instant::now(),
                started: SystemTime::now(),
                pending: Vec::new(),
                file: None,
                path: None,
            }),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Keep what `event` says: a step, or a message. How far a step is
    /// would be thousands of lines, and says nothing a message does not.
    pub fn record(&self, event: &Event) {
        let text = match event {
            Event::Step {
                number,
                total,
                label,
            } => format!("── {number}/{total} {label}"),
            Event::Message { text } => text.clone(),
            Event::Fraction { .. } => return,
        };
        self.note(&text);
    }

    /// One line of the run's own, as a message would be.
    pub fn note(&self, text: &str) {
        let mut state = self.state();
        let line = timed(state.start.elapsed().as_secs(), text);
        match state.file.as_mut() {
            Some(file) => {
                let _ = file.write_all(line.as_bytes());
                let _ = file.flush();
            }
            None => state.pending.push(line),
        }
    }

    /// Write the log at `path` from now on, after what was said so far; an
    /// existing one (a run resumed) is added to.
    pub fn open(&self, path: &Path) {
        let mut state = self.state();
        if state.file.is_some() {
            return;
        }
        let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(path) else {
            return;
        };
        let mut text = format!("run started {} UTC\n", utc(state.started));
        for line in state.pending.drain(..) {
            text.push_str(&line);
        }
        let _ = file.write_all(text.as_bytes());
        let _ = file.flush();
        state.file = Some(file);
        state.path = Some(path.to_path_buf());
    }

    /// Where the log is written, once it is.
    pub fn path(&self) -> Option<PathBuf> {
        self.state().path.clone()
    }
}

/// `text` after the time since the run started; the lines of a message of
/// several are indented under the first.
fn timed(seconds: u64, text: &str) -> String {
    let stamp = format!(
        "[+{:02}:{:02}:{:02}] ",
        seconds / 3600,
        seconds / 60 % 60,
        seconds % 60
    );
    let mut lines = text.lines();
    let mut out = format!("{stamp}{}\n", lines.next().unwrap_or_default());
    for line in lines {
        out.push_str(&" ".repeat(stamp.len()));
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// `when` as `YYYY-MM-DD HH:MM:SS`, in UTC: the date of the proleptic
/// Gregorian calendar from the days since 1970 (Howard Hinnant's
/// `civil_from_days`).
fn utc(when: SystemTime) -> String {
    let seconds = when
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (days, rest) = (seconds / 86_400, seconds % 86_400);
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02}",
        rest / 3600,
        rest / 60 % 60,
        rest % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn a_date_is_the_calendar_s() {
        let at = |seconds: u64| utc(UNIX_EPOCH + Duration::from_secs(seconds));
        assert_eq!(at(0), "1970-01-01 00:00:00");
        assert_eq!(at(951_782_400), "2000-02-29 00:00:00");
        assert_eq!(at(1_762_508_402), "2025-11-07 09:40:02");
    }

    #[test]
    fn a_line_is_timed_and_its_continuations_indented() {
        assert_eq!(timed(3_725, "one"), "[+01:02:05] one\n");
        assert_eq!(
            timed(5, "first\nsecond"),
            "[+00:00:05] first\n            second\n"
        );
    }

    #[test]
    fn what_was_said_before_the_file_is_written_first_and_a_resumed_run_adds() {
        let dir = std::env::temp_dir().join(format!("omnifrog-runlog-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(FILE_NAME);

        let log = RunLog::new();
        log.record(&Event::message("probing the film"));
        log.record(&Event::ratio("extraction", 0.5));
        assert_eq!(log.path(), None);
        log.open(&path);
        log.record(&Event::Step {
            number: 1,
            total: 6,
            label: "extraction".into(),
        });
        log.note("failed: the disk is full");
        let text = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("run started ") && lines[0].ends_with(" UTC"));
        assert!(lines[1].ends_with("] probing the film"));
        assert!(lines[2].ends_with("] ── 1/6 extraction"));
        assert!(lines[3].ends_with("] failed: the disk is full"));
        assert_eq!(lines.len(), 4, "a fraction leaves no line");
        assert_eq!(log.path(), Some(path.clone()));

        // A second run on the same job adds to it.
        let again = RunLog::new();
        again.open(&path);
        again.record(&Event::message("resumed"));
        let text = fs::read_to_string(&path).unwrap();
        assert_eq!(text.matches("run started ").count(), 2);
        assert!(text.trim_end().ends_with("] resumed"));
        fs::remove_dir_all(&dir).unwrap();
    }
}
