//! What the window remembers of this machine from one run to the next.
//!
//! Kept beside the configuration, not in the bench: a set may be read from
//! two machines, and their outputs are not the same.

use crate::output::Choice;
use omnifrog_core::config::Config;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prefs {
    /// The audio output chosen; the machine's default when there is none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Choice>,
}

impl Prefs {
    pub fn path() -> PathBuf {
        Config::path().with_file_name("bench-window.json")
    }

    /// What the file holds, or nothing chosen when it is not there or does
    /// not read: a preference is not worth refusing to start.
    pub fn load(path: &Path) -> Self {
        fs::read(path)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
    }
}
