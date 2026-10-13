//! The last step: the new track into a file someone plays.
//!
//! Two shapes, both written by mkvmerge, which copies and never re-encodes:
//!
//! - an **`.mka`** of the new track alone — French, `Atmos Français`
//!   (`IAMF Français` for IAMF, `IAMF Français (voix sur le lit)` with the
//!   voices on the bed), default, carrying its tags. Named `<film>.Omnifrog-<codes>.mka` and put
//!   beside the film, it is what a player loads beside the video;
//! - a full **`.mkv`** — the source's video, the new track (default), the
//!   original (`<format> Original`, in its language), the dub as it was
//!   (`Français (<codec> <layout>)`), and the source's subtitles, chapters
//!   and attachments; every other audio track of the source is left out.
//!
//! **When the track starts.** The new track was decoded from the original's
//! raw stream, which starts at zero, but the original itself may not: a
//! disc remux can start its audio two seconds after its picture. So the new
//! track is shifted by the original's own start — in the `.mka` as in the
//! `.mkv` — and plays against the picture exactly where the original does.
//!
//! **What goes in.** A TrueHD encoder writes a raw stream; harlettizer
//! writes IAMF already in an `.mka` of one track (`A_IAMF`), since IAMF has
//! no raw form a muxer reads. mkvmerge takes either the same way — the
//! first and only track of the file, numbered 0 — and copies an IAMF track's
//! descriptors, codec delay and pre-roll as they are; the name, language,
//! tags and shift given here replace what the encoder's file said.
//!
//! The file is written under a hidden temporary name beside its final one
//! and renamed when mkvmerge is done, so a file under its real name is whole:
//! a stopped remux does not leave half a track for a library to find.

use crate::config::{Config, Encoder};
use crate::job::TrackRef;
use crate::progress::{Event, Progress};
use crate::run::{self, Stream};
use crate::tools::{self, Tool};
use crate::tracks;
use crate::{Error, Result};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The name the new track is given: what it is, then its language.
pub fn track_name(encoder: Encoder) -> &'static str {
    match encoder {
        Encoder::Hz | Encoder::Dee => "Atmos Français",
        Encoder::Iamf => "IAMF Français",
        Encoder::IamfBed => "IAMF Français (voix sur le lit)",
    }
}

/// 0.1 ms. A TrueHD access unit lasts 1/1200 s, which a millisecond clock
/// rounds by up to half a millisecond each time; a tenth of one keeps its
/// timestamps where they belong, and still spans any film.
const TIMESTAMP_SCALE: &str = "100000";

/// Write the new track alone, as an `.mka`.
pub fn mka(
    config: &Config,
    stream: &Path,
    name: &str,
    tags: &Path,
    start: f64,
    output: &Path,
    progress: Progress,
) -> Result<()> {
    let mkvmerge = tools::require(config, Tool::Mkvmerge)?;
    let temporary = temporary(output)?;
    let mut command = crate::run::command(mkvmerge);
    command.args(["--gui-mode", "-o"]).arg(&temporary).args([
        "--timestamp-scale",
        TIMESTAMP_SCALE,
        "--no-chapters",
    ]);
    new_track(&mut command, stream, name, tags, start);
    merge(command, &temporary, output, progress)
}

/// Write the full film: the source's picture, subtitles, chapters and
/// attachments, the new track, the original, and the dub as it was.
#[allow(clippy::too_many_arguments)]
pub fn mkv(
    config: &Config,
    source: &Path,
    original: &TrackRef,
    dub: &TrackRef,
    stream: &Path,
    name: &str,
    tags: &Path,
    output: &Path,
    progress: Progress,
) -> Result<()> {
    let mkvmerge = tools::require(config, Tool::Mkvmerge)?;
    let ids = identify(&mkvmerge, source)?;
    let original_id = ids.audio(original.index)?;
    let dub_id = ids.audio(dub.index)?;
    let temporary = temporary(output)?;

    let mut command = crate::run::command(&mkvmerge);
    command
        .args(["--gui-mode", "-o"])
        .arg(&temporary)
        .args(["--timestamp-scale", TIMESTAMP_SCALE])
        .args(["--audio-tracks", &format!("{original_id},{dub_id}")])
        .args([
            "--track-name",
            &format!("{original_id}:{}", original_name(original)),
        ])
        .args([
            "--language",
            &format!(
                "{original_id}:{}",
                tracks::iso3(original.language.as_deref())
            ),
        ])
        .args(["--default-track-flag", &format!("{original_id}:no")])
        .args(["--track-name", &format!("{dub_id}:{}", dub_name(dub))])
        .args(["--language", &format!("{dub_id}:fre")])
        .args(["--default-track-flag", &format!("{dub_id}:no")])
        .arg(source);
    new_track(&mut command, stream, name, tags, original.start);
    // The picture first, then the new track, the original and the dub; the
    // subtitles keep their own order after them.
    let mut order: Vec<String> = ids.video.iter().map(|id| format!("0:{id}")).collect();
    order.push("1:0".into());
    order.push(format!("0:{original_id}"));
    order.push(format!("0:{dub_id}"));
    command.args(["--track-order", &order.join(",")]);
    merge(command, &temporary, output, progress)
}

/// The options of the new track, then the file it is read from.
fn new_track(command: &mut Command, stream: &Path, name: &str, tags: &Path, start: f64) {
    let mut tag_option = std::ffi::OsString::from("0:");
    tag_option.push(tags);
    command
        .arg("--tags")
        .arg(tag_option)
        .args(["--language", "0:fre"])
        .args(["--track-name", &format!("0:{name}")])
        .args(["--default-track-flag", "0:yes"]);
    let shift = (start * 1000.0).round() as i64;
    if shift != 0 {
        command.args(["--sync", &format!("0:{shift}")]);
    }
    // Only our tags on the new track: none of the stream's own, global or
    // per track. Neither encoder writes any today (harlettizer's Matroska
    // only names itself as muxing and writing application), so this is a
    // guard; it applies to this input alone, never to the source film's.
    command.args(["--no-global-tags", "--no-track-tags"]);
    command.arg(stream);
}

/// `<format> Original`: what the original's scene is, not a brand it may
/// not be. A DTS:X original is not an Atmos one.
fn original_name(track: &TrackRef) -> String {
    let label = track.spatial.as_deref().unwrap_or("");
    let format = if label.starts_with("DTS:X") {
        "DTS:X"
    } else if label.starts_with("Auro") {
        "Auro-3D"
    } else {
        "Atmos"
    };
    format!("{format} Original")
}

/// `Français (DTS 5.1)`: the dub as the source carried it.
fn dub_name(track: &TrackRef) -> String {
    let layout = match track.channels {
        0 => "?".to_string(),
        1 => "1.0".to_string(),
        2 => "2.0".to_string(),
        n => format!("{}.1", n - 1),
    };
    format!("Français ({} {layout})", track.codec)
}

/// Run mkvmerge and move its file into place. It exits 0 when all went
/// well, 1 when it only warned — the file is written — and 2 on an error.
fn merge(command: Command, temporary: &Path, output: &Path, progress: Progress) -> Result<()> {
    let mut warnings = Vec::new();
    let (status, tail) = run::lines_status(command, |stream, line| {
        if let Some(percent) = line
            .strip_prefix("#GUI#progress ")
            .and_then(|p| p.trim().trim_end_matches('%').parse::<f64>().ok())
        {
            progress(Event::ratio("remuxing", percent / 100.0));
        } else if stream == Stream::Stdout && line.starts_with("#GUI#warning") {
            warnings.push(line.trim_start_matches("#GUI#warning").trim().to_string());
        }
    })?;
    match status.code() {
        Some(0) => {}
        Some(1) => {
            for warning in warnings {
                progress(Event::message(format!("mkvmerge: {warning}")));
            }
        }
        _ => {
            let _ = fs::remove_file(temporary);
            let why = tail
                .iter()
                .rev()
                .find(|l| l.starts_with("#GUI#error") || l.starts_with("Error"))
                .or(tail.last())
                .map(|l| l.trim_start_matches("#GUI#error").trim().to_string())
                .unwrap_or_else(|| "no detail".into());
            return Err(Error::Command {
                program: "mkvmerge".into(),
                status: status.to_string(),
                detail: why,
            });
        }
    }
    fs::rename(temporary, output).map_err(|e| Error::io(output, e))
}

/// `.<name>.part` beside the final file: hidden, and with an extension no
/// library scanner takes for a track.
fn temporary(output: &Path) -> Result<PathBuf> {
    let dir = output
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    let name = output
        .file_name()
        .ok_or_else(|| Error::msg(format!("{}: not a file name", output.display())))?;
    let mut hidden = std::ffi::OsString::from(".");
    hidden.push(name);
    hidden.push(".part");
    let path = dir.join(hidden);
    let _ = fs::remove_file(&path);
    Ok(path)
}

/// The source's track numbers as mkvmerge counts them.
struct Ids {
    video: Vec<u64>,
    audio: Vec<u64>,
}

impl Ids {
    /// The mkvmerge number of the `index`-th audio track, from 1.
    fn audio(&self, index: u32) -> Result<u64> {
        self.audio.get(index as usize - 1).copied().ok_or_else(|| {
            Error::msg(format!(
                "mkvmerge sees no audio track {index} in the source"
            ))
        })
    }
}

#[derive(Deserialize)]
struct Identified {
    #[serde(default)]
    tracks: Vec<IdentifiedTrack>,
}

#[derive(Deserialize)]
struct IdentifiedTrack {
    id: u64,
    #[serde(rename = "type")]
    kind: String,
}

fn identify(mkvmerge: &Path, source: &Path) -> Result<Ids> {
    let mut command = crate::run::command(mkvmerge);
    command.arg("-J").arg(source);
    let json = run::output(command)?;
    let identified: Identified = serde_json::from_str(&json)
        .map_err(|e| Error::msg(format!("mkvmerge -J: unreadable ({e})")))?;
    let of = |kind: &str| {
        identified
            .tracks
            .iter()
            .filter(|t| t.kind == kind)
            .map(|t| t.id)
            .collect::<Vec<_>>()
    };
    Ok(Ids {
        video: of("video"),
        audio: of("audio"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(codec: &str, channels: u32, spatial: Option<&str>) -> TrackRef {
        TrackRef {
            index: 1,
            stream: 1,
            codec: codec.into(),
            channels,
            language: Some("eng".into()),
            title: None,
            spatial: spatial.map(str::to_string),
            start: 0.0,
        }
    }

    #[test]
    fn the_original_is_named_after_what_it_is() {
        assert_eq!(
            original_name(&track("TrueHD", 8, Some("TrueHD"))),
            "Atmos Original"
        );
        assert_eq!(
            original_name(&track("DTS", 8, Some("DTS:X-7.1.4+5"))),
            "DTS:X Original"
        );
        assert_eq!(
            original_name(&track("DTS", 8, Some("Auro-3D-13.1"))),
            "Auro-3D Original"
        );
        assert_eq!(dub_name(&track("DTS", 6, None)), "Français (DTS 5.1)");
    }

    #[test]
    fn the_new_track_is_shifted_by_the_originals_start() {
        let mut command = crate::run::command("mkvmerge");
        new_track(
            &mut command,
            Path::new("vf.thd"),
            track_name(Encoder::Hz),
            Path::new("tags.xml"),
            2.0,
        );
        let args: Vec<String> = command
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        let sync = args.iter().position(|a| a == "--sync").unwrap();
        assert_eq!(args[sync + 1], "0:2000");
        assert_eq!(args.last().unwrap(), "vf.thd");
        // The stream's own tags are dropped, just before the stream itself.
        let n = args.len();
        assert_eq!(args[n - 3..n - 1], ["--no-global-tags", "--no-track-tags"]);
        // No shift for a track that starts with the picture.
        let mut command = crate::run::command("mkvmerge");
        new_track(
            &mut command,
            Path::new("vf.thd"),
            track_name(Encoder::Hz),
            Path::new("tags.xml"),
            0.0,
        );
        assert!(!command.get_args().any(|a| a == "--sync"));
    }

    #[test]
    fn an_iamf_track_is_named_for_what_it_is() {
        let mut command = crate::run::command("mkvmerge");
        new_track(
            &mut command,
            Path::new("vf.mka"),
            track_name(Encoder::Iamf),
            Path::new("tags.xml"),
            0.084,
        );
        let args: Vec<String> = command
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        let name = args.iter().position(|a| a == "--track-name").unwrap();
        assert_eq!(args[name + 1], "0:IAMF Français");
        let sync = args.iter().position(|a| a == "--sync").unwrap();
        assert_eq!(args[sync + 1], "0:84");
        assert_eq!(args.last().unwrap(), "vf.mka");
        assert_eq!(track_name(Encoder::Dee), "Atmos Français");
        assert_eq!(
            track_name(Encoder::IamfBed),
            "IAMF Français (voix sur le lit)"
        );
    }

    #[test]
    fn a_remux_in_progress_is_hidden_from_a_library() {
        // A file where the directory should be: making it is the only
        // thing that fails, on every system; the name is what matters.
        let dir = std::env::temp_dir();
        let taken = dir.join(format!("omnifrog-remux-{}", std::process::id()));
        fs::write(&taken, b"").unwrap();
        let path = temporary(&taken.join("F.Omnifrog-x.mka"));
        assert!(path.is_err());
        let _ = fs::remove_file(&taken);
        let path = temporary(&dir.join("F.Omnifrog-x.mka")).unwrap();
        assert_eq!(path, dir.join(".F.Omnifrog-x.mka.part"));
    }
}
