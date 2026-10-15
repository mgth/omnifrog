//! Where the external programs are.
//!
//! Three places, in order: the path the configuration gives, a `tools`
//! directory beside this executable (how a bundle ships them), then `PATH`.
//! Nothing is downloaded: a decoder is a thing to install, not to fetch
//! behind the user's back.

use crate::config::{Config, expand};
use crate::{Error, Result, run};
use std::path::{Path, PathBuf};
use std::process::Stdio;

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

    /// Which file to choose, for a tool whose release ships other files
    /// beside it that are easy to take for it.
    pub fn which_file(self) -> Option<String> {
        match self {
            Tool::Harletty => Some(format!(
                "harletty{} is in the harletty-cli archive of a harletty-bridge release; the \
                 harletty-bridge archive holds only libraries",
                std::env::consts::EXE_SUFFIX
            )),
            _ => None,
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

/// The names `base` may have on disk. On Windows only those with a program's
/// extension: a file without one there is most likely the Linux build of
/// the tool, unpacked from the wrong archive, which Windows cannot start.
fn with_extensions(base: &Path) -> Vec<PathBuf> {
    if cfg!(windows) {
        vec![
            base.with_extension("exe"),
            base.with_extension("cmd"),
            base.with_extension("bat"),
        ]
    } else {
        vec![base.to_path_buf()]
    }
}

/// Why a file is plainly not a program, from its name alone: a library or
/// an archive, which a file picker lets anyone choose.
pub fn not_a_program(path: &Path) -> Option<&'static str> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    match extension.as_str() {
        "dll" | "so" | "dylib" => Some("a library, not a program"),
        "zip" | "7z" | "rar" | "tar" | "gz" | "tgz" | "xz" | "bz2" | "zst" => {
            Some("an archive, not a program: unpack it and choose the program inside")
        }
        _ => None,
    }
}

/// What starting a program to ask its version found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Version {
    /// It started: the first line it printed, empty when it printed none or
    /// failed.
    Started(String),
    /// It has no version flag, so it was not started.
    NotAsked,
    /// It could not be started at all, and why.
    CannotStart(String),
}

/// Start the program to ask its version.
pub fn version(path: &Path, tool: Tool) -> Version {
    let Some(flag) = tool.version_flag() else {
        return Version::NotAsked;
    };
    let mut command = run::command(path);
    command.arg(flag).stdin(Stdio::null());
    match command.output() {
        Err(e) => Version::CannotStart(run::start_error(&e)),
        Ok(out) if out.status.success() => Version::Started(
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .unwrap_or_default()
                .to_string(),
        ),
        Ok(_) => Version::Started(String::new()),
    }
}

/// Whether a Dee path is a Windows executable, which Linux runs through Wine.
pub fn needs_wine(dee: &Path) -> bool {
    !cfg!(windows)
        && dee
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("exe"))
}
