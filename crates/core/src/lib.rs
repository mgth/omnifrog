//! UI-agnostic core of omnifrog.
//!
//! Everything here runs without a window and without printing: the
//! configuration, the separation-model catalogue, where the external tools
//! are, the Python environment the inference runs in, and the dubbing pipeline
//! itself. Long steps report through
//! [`progress::Event`]; the command line in `crates/cli` is one consumer, the
//! window in `crates/gui` is another.
//!
//! The decoders and encoders (harletty, harlettizer, Dee, ffmpeg, mkvmerge)
//! are binaries this crate runs, never crates it links: the workspace builds on its
//! own, on Windows as on Linux, and its licence does not depend on theirs.
//! Python is inference only — one venv, one worker script — and every bit of
//! DSP and orchestration is Rust.

pub mod bench;
pub mod caf;
pub mod config;
pub mod depth;
pub mod doctor;
pub mod download;
pub mod duplicates;
pub mod encode;
pub mod extract;
pub mod job;
pub mod loudness;
pub mod master;
pub mod models;
pub mod pipeline;
pub mod probe;
pub mod progress;
pub mod remux;
pub mod run;
pub mod separate;
pub mod setup;
pub mod tags;
pub mod text;
pub mod tools;
pub mod tracks;
pub mod voices;
pub mod wav;
pub mod worker;

use std::path::PathBuf;

/// The program's name, as it appears in configuration paths and in the
/// `OMNIFROG_TOOL` tag of the tracks it writes.
pub const NAME: &str = "omnifrog";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Message(String),
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}: {source}")]
    Yaml {
        path: PathBuf,
        #[source]
        source: serde_yaml_ng::Error,
    },
    #[error("downloading {url}: {detail}")]
    Download { url: String, detail: String },
    #[error("`{tool}` is not configured, not beside the program and not on PATH")]
    ToolMissing { tool: String },
    #[error("{program} failed ({status}): {detail}")]
    Command {
        program: String,
        status: String,
        detail: String,
    },
    #[error("no model `{key}` in the catalogue")]
    UnknownModel { key: String },
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub fn msg(text: impl Into<String>) -> Self {
        Self::Message(text.into())
    }
}
