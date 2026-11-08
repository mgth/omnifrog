//! How a long step tells its caller where it is.
//!
//! One enum, one callback. The command line turns events into lines on
//! standard error (or JSON, one object per line); an egui client would turn
//! them into a bar and a log. Nothing in the core prints.

/// What a fraction counts, so a consumer can word it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Unit {
    /// Bytes of a download or of a file being written.
    Bytes,
    /// Things: elements separated, channels extracted.
    Items,
    /// A bare ratio, worth showing as a percentage and nothing else.
    Ratio,
}

/// One thing that happened during a step. Serialised, it is one JSON
/// object per line — what `--json` prints and what a front end reads.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "event", rename_all = "lowercase")]
pub enum Event {
    /// A run's step starts: « 2/6 dialogue out of the original ». Every
    /// event after it belongs to that step, until the next one.
    Step {
        number: u32,
        total: u32,
        label: String,
    },
    /// A line worth showing: what is starting, what was decided, what a
    /// subprocess said.
    Message { text: String },
    /// A line for the run's log alone: what a machine following the
    /// extraction asked of its worker, how long that took, and why it
    /// waited. Hundreds in a film, worth reading only when a run did not go
    /// as it should; a front end leaves them out.
    Trace { text: String },
    /// Progress of one unit of work; `total` is `None` when the server or
    /// the tool did not say.
    Fraction {
        what: String,
        done: u64,
        total: Option<u64>,
        unit: Unit,
    },
    /// A step ended, after `seconds` of work: measured where the work is
    /// done, so that it is right whenever a front end gets to read it. A
    /// resumed run first says it of every step an earlier run did, with
    /// `earlier` and the time the job's manifest recorded.
    Took {
        number: u32,
        total: u32,
        label: String,
        seconds: f64,
        earlier: bool,
    },
}

impl Event {
    /// A fraction in bytes.
    pub fn bytes(what: impl Into<String>, done: u64, total: Option<u64>) -> Self {
        Self::Fraction {
            what: what.into(),
            done,
            total,
            unit: Unit::Bytes,
        }
    }

    /// A fraction in things.
    pub fn items(what: impl Into<String>, done: u64, total: u64) -> Self {
        Self::Fraction {
            what: what.into(),
            done,
            total: Some(total),
            unit: Unit::Items,
        }
    }

    /// A fraction of the whole, from 0 to 1.
    pub fn ratio(what: impl Into<String>, done: f64) -> Self {
        Self::Fraction {
            what: what.into(),
            done: (done.clamp(0.0, 1.0) * 10_000.0) as u64,
            total: Some(10_000),
            unit: Unit::Ratio,
        }
    }

    pub fn message(text: impl Into<String>) -> Self {
        Self::Message { text: text.into() }
    }

    pub fn trace(text: impl Into<String>) -> Self {
        Self::Trace { text: text.into() }
    }

    /// The line a [`Event::Took`] is told as: « 2/6 dialogue out of the
    /// original took 12:54 », « … took 14:53, in an earlier run ». `None`
    /// for any other event.
    pub fn took_line(&self) -> Option<String> {
        let Self::Took {
            number,
            total,
            label,
            seconds,
            earlier,
        } = self
        else {
            return None;
        };
        Some(format!(
            "{number}/{total} {label} took {}{}",
            clock(seconds.max(0.0) as u64),
            if *earlier { ", in an earlier run" } else { "" }
        ))
    }
}

/// « 1:02:03 », « 4:05 »: how long something took, the way a player shows
/// where it is.
pub fn clock(seconds: u64) -> String {
    if seconds >= 3600 {
        format!(
            "{}:{:02}:{:02}",
            seconds / 3600,
            seconds / 60 % 60,
            seconds % 60
        )
    } else {
        format!("{}:{:02}", seconds / 60, seconds % 60)
    }
}

/// The callback a step reports through.
pub type Progress<'a> = &'a (dyn Fn(Event) + Sync);

/// A callback that discards everything, for callers that do not care.
pub fn quiet(_: Event) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_step_s_time_is_told_as_a_player_shows_it() {
        assert_eq!(clock(0), "0:00");
        assert_eq!(clock(893), "14:53");
        assert_eq!(clock(28_364), "7:52:44");
        let took = |seconds, earlier| Event::Took {
            number: 5,
            total: 6,
            label: "encode".into(),
            seconds,
            earlier,
        };
        assert_eq!(
            took(170.6, false).took_line().unwrap(),
            "5/6 encode took 2:50"
        );
        assert_eq!(
            took(893.37, true).took_line().unwrap(),
            "5/6 encode took 14:53, in an earlier run"
        );
        assert_eq!(Event::message("hello").took_line(), None);
    }

    /// Serialised as the window reads it from its worker.
    #[test]
    fn a_step_s_time_goes_through_the_worker_s_line() {
        let event = Event::Took {
            number: 2,
            total: 6,
            label: "dialogue out of the original".into(),
            seconds: 773.5,
            earlier: false,
        };
        let line = serde_json::to_string(&event).unwrap();
        assert!(line.contains(r#""event":"took""#), "{line}");
        assert_eq!(serde_json::from_str::<Event>(&line).unwrap(), event);
    }
}
