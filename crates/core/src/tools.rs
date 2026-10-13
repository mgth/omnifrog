//! Where the external programs are.
//!
//! Three places, in order: the path the configuration gives, a `tools`
//! directory beside this executable (how a bundle ships them), then `PATH`.
//! Nothing is downloaded: a decoder is a thing to install, not to fetch
//! behind the user's back.

use crate::config::{Config, expand};
use crate::{Error, Result, run};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tool {
    Ffmpeg,
    Ffprobe,
    Mkvmerge,
    Harletty,
    Harlettizer,
    Dee,
    Wine,
}

impl Tool {
    pub const ALL: [Tool; 7] = [
        Tool::Ffmpeg,
        Tool::Ffprobe,
        Tool::Mkvmerge,
        Tool::Harletty,
        Tool::Harlettizer,
        Tool::Dee,
        Tool::Wine,
    ];

    /// The tools this platform uses: all of them, but Wine only where a
    /// Windows `dee.exe` needs it — not on Windows itself.
    pub fn here() -> impl Iterator<Item = Tool> {
        Self::ALL
            .into_iter()
            .filter(|tool| !(cfg!(windows) && *tool == Tool::Wine))
    }

    /// The program's name on disk, without the platform's extension.
    pub fn name(self) -> &'static str {
        match self {
            Tool::Ffmpeg => "ffmpeg",
            Tool::Ffprobe => "ffprobe",
            Tool::Mkvmerge => "mkvmerge",
            Tool::Harletty => "harletty",
            Tool::Harlettizer => "harlettizer",
            Tool::Dee => "dee",
            Tool::Wine => "wine",
        }
    }

    /// What it is for, one line.
    pub fn purpose(self) -> &'static str {
        match self {
            Tool::Ffmpeg => "demuxes the tracks and splits the dubbed track into channels",
            Tool::Ffprobe => "lists the tracks of the source",
            Tool::Mkvmerge => "writes the .mka / .mkv",
            Tool::Harletty => {
                "decodes the original (TrueHD Atmos, E-AC-3 JOC, DTS:X, Auro-3D) into a master set"
            }
            Tool::Harlettizer => {
                "encodes the re-voiced master as TrueHD Atmos (encoder: hz) or as IAMF (encoder: iamf or iamf-lit)"
            }
            Tool::Dee => "an encoder you supply (encoder: dee)",
            Tool::Wine => "runs dee.exe on Linux",
        }
    }

    fn configured(self, config: &Config) -> Option<&PathBuf> {
        let t = &config.tools;
        match self {
            Tool::Ffmpeg => t.ffmpeg.as_ref(),
            Tool::Ffprobe => t.ffprobe.as_ref(),
            Tool::Mkvmerge => t.mkvmerge.as_ref(),
            Tool::Harletty => t.harletty.as_ref(),
            Tool::Harlettizer => t.harlettizer.as_ref(),
            Tool::Dee => t.dee.as_ref(),
            Tool::Wine => t.wine.as_ref(),
        }
    }

    /// The argument that makes the program print its version, if it has one.
    fn version_flag(self) -> Option<&'static str> {
        match self {
            Tool::Ffmpeg | Tool::Ffprobe | Tool::Wine => Some("-version"),
            Tool::Mkvmerge | Tool::Harletty | Tool::Harlettizer => Some("--version"),
            Tool::Dee => None,
        }
    }
}

/// The program, wherever it is; `None` when it is nowhere.
pub fn find(config: &Config, tool: Tool) -> Option<PathBuf> {
    if let Some(path) = tool.configured(config) {
        let path = expand(path);
        return path.is_file().then_some(path);
    }
    if let Some(beside) = beside_executable(tool) {
        return Some(beside);
    }
    on_path(tool.name())
}

pub fn require(config: &Config, tool: Tool) -> Result<PathBuf> {
    find(config, tool).ok_or_else(|| Error::ToolMissing {
        tool: tool.name().to_string(),
    })
}

/// `<exe dir>/tools/<name>` or `<exe dir>/<name>`, with the platform's
/// executable extension.
fn beside_executable(tool: Tool) -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    for candidate in [dir.join("tools").join(tool.name()), dir.join(tool.name())] {
        for name in with_extensions(&candidate) {
            if name.is_file() {
                return Some(name);
            }
        }
    }
    None
}

/// The first `name` on `PATH` that is a file.
pub fn on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for candidate in with_extensions(&dir.join(name)) {
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn with_extensions(base: &Path) -> Vec<PathBuf> {
    if cfg!(windows) {
        let mut v = vec![
            base.with_extension("exe"),
            base.with_extension("cmd"),
            base.with_extension("bat"),
        ];
        v.push(base.to_path_buf());
        v
    } else {
        vec![base.to_path_buf()]
    }
}

/// The first line the program prints for its version, if it prints one.
pub fn version(path: &Path, tool: Tool) -> Option<String> {
    let flag = tool.version_flag()?;
    let mut command = run::command(path);
    command.arg(flag);
    let text = run::output(command).ok()?;
    text.lines().next().map(|l| l.trim().to_string())
}

/// Whether a Dee path is a Windows executable, which Linux runs through Wine.
pub fn needs_wine(dee: &Path) -> bool {
    !cfg!(windows)
        && dee
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("exe"))
}
