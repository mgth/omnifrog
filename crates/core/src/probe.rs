//! What a file holds: its audio tracks, and what each one really carries.
//!
//! Two readers, because neither knows the whole answer. `ffprobe` lists the
//! streams — codec, channels, language, title, disposition — which is all a
//! container states. What a stream carries *beyond* its bed it does not
//! state: a DTS:X profile presents itself as ordinary DTS-HD MA, Auro-3D
//! lives in the low bits of the PCM, and a TrueHD track says nothing about
//! its objects. Only the decoder can tell, so the tracks that could carry a
//! scene are handed a few seconds of themselves through
//! `harletty info --json`.
//!
//! MediaInfo could answer all of this too. `ffprobe` is already required
//! for the extraction, its disposition flags name an audio description or a
//! commentary better than MediaInfo's service kind, and the decoder settles
//! the rest: there is no MediaInfo here.

use crate::config::Config;
use crate::progress::{Event, Progress};
use crate::tools::{self, Tool};
use crate::{Error, Result, run};
use serde::Deserialize;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Stdio;

/// Seconds of audio the decoder is asked for. Past the two seconds a plain
/// DTS track takes to be called plain, and the two seconds of E-AC-3 a JOC
/// sniff reads.
pub const PROBE_SECONDS: f64 = 4.0;

/// The report shape [`Report`] understands (`harletty info --json`).
pub const SCHEMA: u32 = 1;

/// A compressed audio codec, as far as this program cares.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Codec {
    TrueHd,
    Eac3,
    Ac3,
    Dts,
    Lossless(String),
    Other(String),
}

impl Codec {
    fn from_ffprobe(name: &str) -> Self {
        match name {
            "truehd" | "mlp" => Codec::TrueHd,
            "eac3" => Codec::Eac3,
            "ac3" => Codec::Ac3,
            "dts" => Codec::Dts,
            "flac" => Codec::Lossless(name.to_string()),
            n if n.starts_with("pcm_") => Codec::Lossless(n.to_string()),
            n => Codec::Other(n.to_string()),
        }
    }

    /// The raw muxer ffmpeg writes this codec into, and the `--codec` the
    /// decoder reads it as; `None` for a codec the decoder does not read.
    pub fn raw_format(&self) -> Option<&'static str> {
        match self {
            Codec::TrueHd => Some("truehd"),
            Codec::Eac3 => Some("eac3"),
            Codec::Ac3 => Some("ac3"),
            Codec::Dts => Some("dts"),
            _ => None,
        }
    }

    /// Whether a scene could hide in it: only these three are ever asked.
    fn could_be_spatial(&self) -> bool {
        matches!(self, Codec::TrueHd | Codec::Eac3 | Codec::Dts)
    }

    pub fn name(&self) -> &str {
        match self {
            Codec::TrueHd => "TrueHD",
            Codec::Eac3 => "E-AC-3",
            Codec::Ac3 => "AC-3",
            Codec::Dts => "DTS",
            Codec::Lossless(n) | Codec::Other(n) => n,
        }
    }
}

/// What a stream carries beyond its bed, as the decoder labels it.
#[derive(Debug, Clone, PartialEq)]
pub struct Spatial {
    /// The `sourceCodec` the decoder writes into the master set:
    /// `TrueHD`, `EAC3-JOC`, `DTS:X-7.1.4+5`, `Auro-3D-13.1`…
    pub label: String,
    /// `atmos`, `joc`, `dtsx` or `auro`.
    pub kind: String,
    /// Waveforms presented as objects, when the presentation states them.
    pub objects: Option<u32>,
    /// Waveforms at a fixed position, when the presentation states them.
    pub fixed: Option<u32>,
    pub presentation: Option<String>,
    pub experimental: bool,
}

impl Spatial {
    /// The line a listing shows: the label, and the counts when known.
    pub fn summary(&self) -> String {
        match (self.objects, self.fixed) {
            (Some(o), Some(f)) => format!("{} ({o} objects, {f} fixed)", self.label),
            (Some(o), None) => format!("{} ({o} objects)", self.label),
            _ => self.label.clone(),
        }
    }
}

/// One audio track of a file.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioTrack {
    /// Rank among the audio tracks, from 1: what `--vo` and `--vf` name,
    /// and what `ffmpeg -map 0:a:N` counts from 0.
    pub index: u32,
    /// Rank among every stream of the container, what mkvmerge calls the
    /// track number.
    pub stream: u32,
    pub codec: Codec,
    /// ffprobe's profile line (`DTS-HD MA`…).
    pub profile: Option<String>,
    pub channels: u32,
    pub sample_rate: u32,
    /// Bits per second, from the stream or from Matroska's `BPS` tag; 0
    /// when neither says.
    pub bit_rate: u64,
    pub language: Option<String>,
    pub title: Option<String>,
    pub default: bool,
    /// When the track's first sample plays, in seconds on the container's
    /// clock. Usually 0; a disc remux can start its audio two seconds after
    /// its picture, and a track decoded from its raw stream loses that.
    pub start: f64,
    /// Not the programme: an audio description, a commentary, a
    /// hearing-impaired or music-and-effects track. Never dubbed onto.
    pub side: bool,
    /// What the decoder found, `None` for a plain track and for one that
    /// was not asked.
    pub spatial: Option<Spatial>,
    /// Why the decoder could not be asked, when it could not.
    pub probe_error: Option<String>,
}

impl AudioTrack {
    /// Whether this track carries a scene a master set can be decoded from.
    pub fn is_spatial(&self) -> bool {
        self.spatial.is_some()
    }

    /// Channels as they are usually written: 6 is 5.1, 8 is 7.1.
    pub fn layout(&self) -> String {
        match self.channels {
            0 => "?".to_string(),
            1 => "1.0".to_string(),
            2 => "2.0".to_string(),
            n if n >= 3 => format!("{}.1", n - 1),
            n => format!("{n} ch"),
        }
    }

    /// The line `probe` prints.
    pub fn summary(&self) -> String {
        let mut text = format!("{:<8} {:<5}", self.codec.name(), self.layout());
        if self.bit_rate > 0 {
            text += &format!(" {:>6} kb/s", self.bit_rate / 1000);
        } else {
            text += "            ";
        }
        text += &format!("  {:<4}", self.language.as_deref().unwrap_or("und"));
        if let Some(spatial) = &self.spatial {
            text += &format!("  {}", spatial.summary());
        }
        if let Some(title) = &self.title {
            text += &format!("  « {title} »");
        }
        if self.side {
            text += "  [side track]";
        }
        text
    }
}

/// A file, as far as this program reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct Media {
    pub path: PathBuf,
    /// Seconds; 0 when the container does not say.
    pub duration: f64,
    pub title: Option<String>,
    pub audio: Vec<AudioTrack>,
    /// How many subtitle streams the file has, which the full `.mkv` keeps.
    pub subtitles: usize,
    pub video: usize,
}

impl Media {
    /// The track `--vo N` or `--vf N` names, counting audio tracks from 1.
    pub fn track(&self, index: u32) -> Result<&AudioTrack> {
        self.audio.iter().find(|t| t.index == index).ok_or_else(|| {
            Error::msg(format!(
                "no audio track {index} in {} ({} audio tracks)",
                self.path.display(),
                self.audio.len()
            ))
        })
    }

    /// The film's name, as every output is named after it.
    pub fn stem(&self) -> String {
        self.path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "film".to_string())
    }
}

/// Everything about `path`: `ffprobe` for the streams, then the decoder for
/// the tracks that could carry a scene. `progress` receives one message per
/// track asked.
pub fn media(config: &Config, path: &Path, progress: Progress) -> Result<Media> {
    let mut media = streams(config, path)?;
    let decoder = match tools::find(config, Tool::Harletty) {
        Some(decoder) => decoder,
        None => {
            progress(Event::message(
                "harletty is not installed: no track can be told plain from spatial",
            ));
            return Ok(media);
        }
    };
    let ffmpeg = tools::require(config, Tool::Ffmpeg)?;
    for track in &mut media.audio {
        if !track.codec.could_be_spatial() || track.channels < 6 {
            continue;
        }
        progress(Event::message(format!(
            "asking the decoder about track {} ({}, {})",
            track.index,
            track.codec.name(),
            track.layout()
        )));
        match report(&ffmpeg, &decoder, path, track) {
            Ok(report) => {
                if let Some(channels) = report.channels {
                    track.channels = channels;
                }
                if let Some(rate) = report.sample_rate {
                    track.sample_rate = rate;
                }
                track.spatial = report.spatial.map(|s| Spatial {
                    label: s.label,
                    kind: s.kind,
                    objects: s.objects,
                    fixed: s.fixed,
                    presentation: s.presentation,
                    experimental: s.experimental,
                });
                track.probe_error = report.error;
            }
            Err(e) => {
                progress(Event::message(format!(
                    "  track {}: {e} — taken as what the container says",
                    track.index
                )));
                track.probe_error = Some(e.to_string());
            }
        }
    }
    Ok(media)
}

/// `ffprobe` alone: the streams as the container states them.
pub fn streams(config: &Config, path: &Path) -> Result<Media> {
    if !path.is_file() {
        return Err(Error::msg(format!("{}: no such file", path.display())));
    }
    let ffprobe = tools::require(config, Tool::Ffprobe)?;
    let mut command = run::command(ffprobe);
    command
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
        ])
        .arg(path);
    let json = run::output(command)?;
    let probe: FfProbe = serde_json::from_str(&json).map_err(|e| {
        Error::msg(format!(
            "{}: ffprobe output unreadable: {e}",
            path.display()
        ))
    })?;
    let mut audio = Vec::new();
    let mut subtitles = 0;
    let mut video = 0;
    for stream in &probe.streams {
        match stream.codec_type.as_str() {
            "video" => video += 1,
            "subtitle" => subtitles += 1,
            "audio" => {
                let title = stream.tags.get("title").cloned();
                audio.push(AudioTrack {
                    index: audio.len() as u32 + 1,
                    stream: stream.index,
                    codec: Codec::from_ffprobe(&stream.codec_name),
                    profile: stream.profile.clone(),
                    channels: stream.channels.unwrap_or(0),
                    sample_rate: stream
                        .sample_rate
                        .as_deref()
                        .and_then(|r| r.parse().ok())
                        .unwrap_or(0),
                    bit_rate: stream
                        .bit_rate
                        .as_deref()
                        .and_then(|r| r.parse().ok())
                        .or_else(|| stream.tags.get("BPS").and_then(|b| b.parse().ok()))
                        .unwrap_or(0),
                    language: stream.tags.get("language").cloned(),
                    default: stream.disposition.default == 1,
                    start: stream
                        .start_time
                        .as_deref()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0),
                    side: stream.disposition.is_side() || side_title(title.as_deref()),
                    title,
                    spatial: None,
                    probe_error: None,
                });
            }
            _ => {}
        }
    }
    Ok(Media {
        path: path.to_path_buf(),
        duration: probe
            .format
            .duration
            .as_deref()
            .and_then(|d| d.parse().ok())
            .unwrap_or(0.0),
        title: probe.format.tags.get("title").cloned(),
        audio,
        subtitles,
        video,
    })
}

/// A title that names an audio description, a commentary, a
/// hearing-impaired mix or a music-and-effects stem — whatever the
/// disposition flags say, which many files leave at zero.
///
/// The initialisms are matched in capitals only, as whole words: `AD` in
/// « VFF AAC 2.0 (AD) » is an audio description, the `Ad` of `Ad Libitum` and
/// the `ad` of `ad hoc` are not. The words are matched folded, because a
/// muxer writes `Audiodescription`, `AUDIO DESC.` or `malentendants` as it
/// pleases.
fn side_title(title: Option<&str>) -> bool {
    const CAPITALS: [&str; 5] = ["AD", "SDH", "HI", "VI", "CC"];
    const WORDS: [&str; 7] = [
        "audiodesc",
        "audio desc",
        "malentendant",
        "malvoyant",
        "commentai",
        "commentar",
        "descriptive audio",
    ];
    let Some(title) = title else { return false };
    if CAPITALS
        .iter()
        .any(|word| crate::text::has_exact_word(title, word))
    {
        return true;
    }
    let folded = crate::text::fold(title);
    WORDS.iter().any(|needle| folded.contains(needle))
        || crate::text::has_word(&crate::text::words(title), "m&e")
}

/// `ffmpeg … | harletty info --json` over the head of one track.
fn report(ffmpeg: &Path, decoder: &Path, source: &Path, track: &AudioTrack) -> Result<Report> {
    let format = track.codec.raw_format().ok_or_else(|| {
        Error::msg(format!(
            "{} is not a codec harletty reads",
            track.codec.name()
        ))
    })?;
    let mut demux = run::command(ffmpeg)
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(source)
        .args([
            "-map",
            &format!("0:a:{}", track.index - 1),
            "-c:a",
            "copy",
            "-f",
            format,
            "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| Error::Command {
            program: "ffmpeg".into(),
            status: "could not start".into(),
            detail: e.to_string(),
        })?;
    let stream = demux.stdout.take().expect("stdout was piped");
    let child = run::command(decoder)
        .args(["--loglevel", "off", "info", "--json", "--codec", format])
        .args(["--max-seconds", &PROBE_SECONDS.to_string()])
        .arg("-")
        .stdin(Stdio::from(stream))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let result = match child {
        Ok(child) => finish(child),
        Err(e) => Err(Error::Command {
            program: "harletty".into(),
            status: "could not start".into(),
            detail: e.to_string(),
        }),
    };
    // The decoder stops at its bound; ffmpeg would otherwise sit on a full
    // pipe for the rest of the film.
    let _ = demux.kill();
    let _ = demux.wait();
    result
}

fn finish(mut child: std::process::Child) -> Result<Report> {
    let mut stdout = String::new();
    let mut stderr = String::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_string(&mut stdout);
    }
    if let Some(mut err) = child.stderr.take() {
        let _ = err.read_to_string(&mut stderr);
    }
    let status = child.wait().map_err(|e| Error::Command {
        program: "harletty".into(),
        status: "could not wait".into(),
        detail: e.to_string(),
    })?;
    if !status.success() {
        return Err(Error::Command {
            program: "harletty".into(),
            status: status.to_string(),
            detail: stderr.trim().lines().last().unwrap_or("").to_string(),
        });
    }
    parse(&stdout)
}

/// The report, from the last non-empty line the decoder printed.
fn parse(text: &str) -> Result<Report> {
    let line = text
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| Error::msg("harletty printed no report"))?;
    let report: Report = serde_json::from_str(line)
        .map_err(|e| Error::msg(format!("harletty report unreadable: {e}")))?;
    if report.schema != SCHEMA {
        return Err(Error::msg(format!(
            "harletty report schema {}, this version reads {SCHEMA}",
            report.schema
        )));
    }
    Ok(report)
}

/// One stream, as `harletty info --json` reports it. Unknown fields are
/// ignored and absent blocks read as `None`, so a newer decoder of the same
/// schema still reads. See the bridge's `docs/harletty-cli.md`,
/// "Machine-readable info".
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
struct Report {
    schema: u32,
    #[serde(default)]
    codec: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    channels: Option<u32>,
    #[serde(default)]
    sample_rate: Option<u32>,
    #[serde(default)]
    spatial: Option<ReportSpatial>,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
struct ReportSpatial {
    label: String,
    #[serde(default)]
    kind: String,
    #[serde(default)]
    objects: Option<u32>,
    #[serde(default)]
    fixed: Option<u32>,
    #[serde(default)]
    presentation: Option<String>,
    #[serde(default)]
    experimental: bool,
}

#[derive(Debug, Default, Deserialize)]
struct FfProbe {
    #[serde(default)]
    streams: Vec<FfStream>,
    #[serde(default)]
    format: FfFormat,
}

#[derive(Debug, Default, Deserialize)]
struct FfFormat {
    #[serde(default)]
    duration: Option<String>,
    #[serde(default)]
    tags: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct FfStream {
    index: u32,
    codec_type: String,
    #[serde(default)]
    codec_name: String,
    #[serde(default)]
    profile: Option<String>,
    #[serde(default)]
    channels: Option<u32>,
    #[serde(default)]
    sample_rate: Option<String>,
    #[serde(default)]
    bit_rate: Option<String>,
    #[serde(default)]
    start_time: Option<String>,
    #[serde(default)]
    disposition: Disposition,
    #[serde(default)]
    tags: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Default, Deserialize)]
struct Disposition {
    #[serde(default)]
    default: u8,
    #[serde(default)]
    comment: u8,
    #[serde(default)]
    hearing_impaired: u8,
    #[serde(default)]
    visual_impaired: u8,
    #[serde(default)]
    descriptions: u8,
    #[serde(default)]
    karaoke: u8,
    /// A music-and-effects stem: the programme without its dialogue.
    #[serde(default)]
    clean_effects: u8,
}

impl Disposition {
    fn is_side(&self) -> bool {
        self.comment
            | self.hearing_impaired
            | self.visual_impaired
            | self.descriptions
            | self.karaoke
            | self.clean_effects
            != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_of_a_later_decoder_still_reads() {
        let report = parse(concat!(
            r#"{"schema":1,"harletty":"0.8.0","taxonomy":3,"codec":"DTS-HD MA","channels":8,"#,
            r#""sample_rate":48000,"spatial":{"label":"DTS:X-7.1.4+5","kind":"dtsx","objects":5,"#,
            r#""fixed":4,"experimental":true,"presentation":"ObjectsD4"},"later":{"x":1}}"#
        ))
        .unwrap();
        assert_eq!(report.channels, Some(8));
        let spatial = report.spatial.unwrap();
        assert_eq!(spatial.label, "DTS:X-7.1.4+5");
        assert_eq!(spatial.objects, Some(5));
    }

    #[test]
    fn a_plain_track_reports_no_scene() {
        let report =
            parse(r#"{"schema":1,"taxonomy":3,"codec":"DTS-HD MA","channels":6,"spatial":null}"#)
                .unwrap();
        assert!(report.spatial.is_none());
    }

    #[test]
    fn another_schema_is_refused_rather_than_guessed() {
        assert!(parse(r#"{"schema":2,"taxonomy":9}"#).is_err());
    }

    #[test]
    fn a_side_track_is_named_by_its_title_or_its_flags() {
        assert!(side_title(Some("VFF AAC 2.0 (AD)")));
        assert!(side_title(Some("Audiodescription")));
        assert!(side_title(Some("AUDIO DESC. (fr)")));
        assert!(side_title(Some("Commentary by the director")));
        assert!(side_title(Some("Version malentendants")));
        // A film's title is not an audio description, and neither is a plain dub.
        assert!(!side_title(Some("Ad Libitum VFF")));
        assert!(!side_title(Some("French (VFF)")));
        assert!(!side_title(None));
    }
}
