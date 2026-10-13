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
    /// Progress of one unit of work; `total` is `None` when the server or
    /// the tool did not say.
    Fraction {
        what: String,
        done: u64,
        total: Option<u64>,
        unit: Unit,
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
}

/// The callback a step reports through.
pub type Progress<'a> = &'a (dyn Fn(Event) + Sync);

/// A callback that discards everything, for callers that do not care.
pub fn quiet(_: Event) {}
