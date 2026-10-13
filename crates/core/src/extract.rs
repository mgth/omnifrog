//! One pass over the source: the original decoded, the dub split.
//!
//! `ffmpeg` demuxes the original's elementary stream to its standard
//! output and `harletty decode` reads it there, writing the master set's
//! header, its metadata and one mono 24-bit WAV per waveform — no
//! intermediate bitstream, no interleaved CAF, nothing to de-interleave
//! afterwards.
//!
//! The dub's channels come out of the *same* ffmpeg: a `channelsplit` into
//! one WAV per channel, as extra outputs of the process that is already
//! reading the file. The matroska demuxer reads the whole container — video
//! included — even for a single track, so a second pass for the dub costs
//! another 50 to 80 GB of reading (two to seven minutes a film) for fifteen
//! seconds of AC-3 decoding. Riding along, it costs nothing.
//!
//! The decoder writes its channels as `canal_<n>.wav`, n from 0 in the
//! master set's own order. Once the header is readable those are renamed
//! `vo_<id>.wav` after the identifier each one carries, which is how every
//! later step names an element. The dub's channels are `vf_<name>.wav`,
//! named as the bed of a master set names its channels (`L`, `C`, `LFE`,
//! `Lss`, `Lrs`…), so the master step can match a dub channel to a position
//! without a table in between.

use crate::config::Config;
use crate::master::MasterSet;
use crate::probe::{AudioTrack, Codec};
use crate::progress::{Event, Progress};
use crate::tools::{self, Tool};
use crate::{Error, Result, run, wav};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

/// Prefix of the original's waveforms once named after their identifier.
pub const ORIGINAL: &str = "vo_";
/// Prefix the decoder writes its channels under, before they are renamed.
pub const RAW: &str = "canal_";
/// Prefix of the dub's channels.
pub const DUB: &str = "vf_";
/// Prefix of the dialogue lifted out of one of the dub's channels.
pub const DUB_VOICE: &str = "voix_";
/// Prefix of one of the original's elements with its dialogue removed.
pub const MUSIC_AND_EFFECTS: &str = "sons_";
/// Prefix of the dialogue lifted out of one of the original's elements.
pub const ORIGINAL_VOICE: &str = "voixvo_";
/// The master set's file name inside the job's extraction directory.
pub const MASTER: &str = "vo";

/// Every master set is at this rate, whatever the source was.
pub const RATE: u64 = 48_000;
/// Waveforms are written 24-bit.
pub const BYTES_PER_SAMPLE: u64 = 3;

/// How often the extraction looks at what has been written.
const POLL: Duration = Duration::from_millis(400);

/// What one pass produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Extracted {
    pub master: MasterSet,
    /// The original's waveforms, `vo_<id>.wav`, in the master set's order.
    pub elements: Vec<PathBuf>,
    /// The dub's channels, `vf_<name>.wav`, in the track's own order.
    pub dub: Vec<(String, PathBuf)>,
    /// Frames every file holds, at 48 kHz.
    pub frames: u64,
    /// For a run from a start point: the keyframe both tracks were cut at,
    /// in seconds on the film's clock.
    pub begin: Option<f64>,
}

/// How far before the keyframe the input is sought: the demuxer lands on
/// some earlier keyframe of its own choosing, and what is read between
/// there and the cut is thrown away.
const PREROLL: f64 = 5.0;

/// What one pass has to read, and from where.
#[derive(Debug, Clone, Copy)]
pub struct Plan<'a> {
    pub source: &'a Path,
    /// The track carrying the scene.
    pub original: &'a AudioTrack,
    /// The track the dialogue is taken from.
    pub dub: &'a AudioTrack,
    /// Where the master set and the channels are written.
    pub dir: &'a Path,
    /// The source's duration in seconds, 0 when the container did not say.
    pub duration: f64,
    /// Read only this many seconds, for a trial run.
    pub seconds: Option<f64>,
    /// Read from this point of the film rather than from its beginning.
    pub start: Option<f64>,
}

/// Decode the original into `plan.dir` and split the dub beside it, in one
/// pass over the source.
pub fn run(config: &Config, plan: &Plan, progress: Progress) -> Result<Extracted> {
    let Plan {
        source,
        original,
        dub,
        dir,
        duration,
        seconds,
        start,
    } = *plan;
    let ffmpeg = tools::require(config, Tool::Ffmpeg)?;
    // From a start point, both tracks are cut at the original's first
    // keyframe at or after it: a stream copied from anywhere else starts
    // wherever the demuxer's seek happened to land. Measured on a
    // TrueHD Atmos film: cut so, the original decodes bit for bit as it does
    // from the beginning, and the dub lines up to the sample.
    let begin = match start {
        Some(start) => {
            let from = start.max(original.start).max(dub.start);
            let at = keyframe(config, source, original, from)?;
            progress(Event::message(format!(
                "from {} (the original's keyframe at or after {})",
                clock(at),
                clock(start)
            )));
            Some(at)
        }
        None => None,
    };
    let decoder = tools::require(config, Tool::Harletty)?;
    let format = original.codec.raw_format().ok_or_else(|| {
        Error::msg(format!(
            "track {} is {}, which harletty does not decode",
            original.index,
            original.codec.name()
        ))
    })?;
    let names = layout(dub.channels)?;
    fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    // A run stopped halfway leaves files that would otherwise be taken for
    // this run's.
    clear(dir, RAW)?;
    clear(dir, DUB)?;

    let dub_paths: Vec<(String, PathBuf)> = names
        .iter()
        .map(|name| ((*name).to_string(), dir.join(format!("{DUB}{name}.wav"))))
        .collect();
    let raw_prefix = dir.join(RAW.trim_end_matches('_'));
    let master_prefix = dir.join(MASTER);

    let mut demux = run::command(&ffmpeg);
    demux.args(["-nostdin", "-loglevel", "error", "-y"]);
    // Seeking the input lands near the cut, fast; each output is then cut
    // exactly, on the film's own timestamps (`-copyts`).
    let sought = begin.map(|at| (at - PREROLL).max(0.0));
    if let Some(sought) = sought {
        demux.args(["-copyts", "-ss", &format!("{sought}")]);
    }
    if let Some(seconds) = seconds {
        // Bounding the *input* stops every output at once; bounding an
        // output would only stop that one. It counts from where the input
        // was sought.
        let read = seconds + begin.zip(sought).map_or(0.0, |(at, from)| at - from);
        demux.args(["-t", &format!("{read}")]);
    }
    demux.arg("-i").arg(source);
    // The dub first, as file outputs: one 24-bit WAV per channel, which
    // also keeps every one of them under the 4 GB a WAV can address.
    demux.arg("-filter_complex").arg(filter(dub, &names));
    for (i, (_, path)) in dub_paths.iter().enumerate() {
        demux.args(["-map", &format!("[a{i}]")]);
        if let Some(at) = begin {
            // Decoded, so cut to the sample.
            demux.args(["-ss", &format!("{at}")]);
        }
        demux.args(["-c:a", "pcm_s24le"]).arg(path);
    }
    // The original last, raw, to the pipe: `-c`/`-f` only bind to the
    // output that follows them.
    demux.args(["-map", &format!("0:a:{}", original.index - 1)]);
    if let Some(at) = begin {
        // A copy keeps nothing before its first keyframe at or after the
        // cut: half a millisecond early, so that keyframe is it whatever the
        // rounding of a timestamp.
        demux.args(["-ss", &format!("{}", at - 0.0005)]);
    }
    demux.args(["-c", "copy", "-f", format, "-"]);

    let mut decode = run::command(&decoder);
    decode.args(["decode", "--no-estimate-progress", "--codec", format]);
    if original.codec == Codec::Dts {
        // DTS:X and Auro-3D: the corner heights and the wides leave the bed
        // for static objects at their speaker positions, which is the bed
        // an Atmos encoder accepts.
        decode.arg("--bed-conform");
    } else {
        // TrueHD and E-AC-3: presentation 3 is the one carrying the objects.
        decode.args(["--presentation", "3"]);
    }
    decode
        .arg("--mono-prefix")
        .arg(&raw_prefix)
        .arg("--output-path")
        .arg(&master_prefix)
        .arg("-");

    // Every channel is written in step, so the bytes one of them will hold
    // is the whole decode's yardstick, whatever the master set turns out to
    // carry.
    let left = duration - begin.unwrap_or(0.0);
    let wanted = match (left, seconds) {
        (d, Some(s)) if d > 0.0 => s.min(d),
        (_, Some(s)) => s,
        (d, None) if d > 0.0 => d,
        _ => 0.0,
    };
    let expected = (wanted > 0.0).then(|| (wanted * RATE as f64) as u64 * BYTES_PER_SAMPLE);
    pipe(demux, decode, dir, expected, progress)?;

    let header = master_prefix.with_extension("atmos");
    let master =
        MasterSet::read(&header).or_else(|first| MasterSet::find(dir).map_err(|_| first))?;
    // `--mono-prefix` does not write the interleaved file the header names,
    // but a previous run with another decoder may have left one: half the
    // job's disk for something nothing reads again.
    let _ = fs::remove_file(&master.audio);
    clear(dir, ORIGINAL)?;
    let elements = rename(&master, dir)?;

    let mut frames = u64::MAX;
    for path in elements.iter().chain(dub_paths.iter().map(|(_, p)| p)) {
        let format = wav::read_header(path)?;
        frames = frames.min(format.frames);
    }
    if frames == u64::MAX {
        frames = 0;
    }
    progress(Event::message(format!(
        "extracted {} elements ({} bed, {} objects) and {} dub channels, {:.1} s",
        master.channels.len(),
        master.bed().count(),
        master.objects().count(),
        dub_paths.len(),
        frames as f64 / RATE as f64
    )));
    Ok(Extracted {
        master,
        elements,
        dub: dub_paths,
        frames,
        begin,
    })
}

/// The time of the original's first keyframe at or after `from`, in seconds
/// on the film's clock: where a copy of its stream can start and decode
/// whole. Every packet of an AC-3 or DTS track is one; TrueHD has one about
/// every tenth of a second.
fn keyframe(config: &Config, source: &Path, original: &AudioTrack, from: f64) -> Result<f64> {
    let ffprobe = tools::require(config, Tool::Ffprobe)?;
    let out = run::command(&ffprobe)
        .args(["-v", "error", "-select_streams"])
        .arg(format!("a:{}", original.index - 1))
        .arg("-read_intervals")
        .arg(format!("{from}%+2"))
        .args(["-show_entries", "packet=pts_time,flags", "-of", "csv=p=0"])
        .arg(source)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| Error::io(&ffprobe, e))?;
    first_keyframe(&String::from_utf8_lossy(&out.stdout), from).ok_or_else(|| {
        Error::msg(format!(
            "no keyframe of track {} within two seconds of {}: past the end of the film?",
            original.index,
            clock(from)
        ))
    })
}

/// The first `pts_time,flags` line flagged a keyframe at or after `from`.
fn first_keyframe(packets: &str, from: f64) -> Option<f64> {
    packets.lines().find_map(|line| {
        let (pts, flags) = line.split_once(',')?;
        let pts: f64 = pts.trim().parse().ok()?;
        (flags.starts_with('K') && pts >= from).then_some(pts)
    })
}

/// Seconds as `h:mm:ss.mmm`.
pub fn clock(seconds: f64) -> String {
    let ms = (seconds.max(0.0) * 1000.0).round() as u64;
    format!(
        "{}:{:02}:{:02}.{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}

/// Whether everything an extraction recorded is still on disk and whole:
/// the master set, one waveform per element, one file per channel of the
/// dub, all of the same length.
pub fn complete(dir: &Path, elements: &[u32], dub_channels: &[String]) -> bool {
    if MasterSet::read(&dir.join(MASTER).with_extension("atmos")).is_err() {
        return false;
    }
    let mut frames = 0;
    for path in elements
        .iter()
        .map(|id| dir.join(format!("{ORIGINAL}{id}.wav")))
        .chain(
            dub_channels
                .iter()
                .map(|name| dir.join(format!("{DUB}{name}.wav"))),
        )
    {
        let held = crate::wav::frames(&path);
        if held == 0 {
            return false;
        }
        // Every file of one extraction is the same length but for the two
        // seconds a bounded read can leave between the two tracks.
        frames = frames.max(held);
    }
    frames > 0
}

/// The paths the dub's channels would be written to, without writing
/// anything: what a resume checks before deciding the step is done.
pub fn dub_paths(dir: &Path, channels: u32) -> Result<Vec<(String, PathBuf)>> {
    Ok(layout(channels)?
        .iter()
        .map(|name| ((*name).to_string(), dir.join(format!("{DUB}{name}.wav"))))
        .collect())
}

/// `ffmpeg … | harletty decode …`, watching what lands on disk.
///
/// ffmpeg's standard output *is* the decoder's standard input, so the two
/// run at the speed of the slower one with no copying in between. Both are
/// waited on: ffmpeg is also writing the dub's channels, which have to be
/// whole before anything reads them. A decoder that dies early would leave
/// ffmpeg blocked on a full pipe, so it is killed as soon as that happens.
fn pipe(
    mut demux: Command,
    mut decode: Command,
    dir: &Path,
    expected: Option<u64>,
    progress: Progress,
) -> Result<()> {
    let mut demux = demux
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| Error::Command {
            program: "ffmpeg".into(),
            status: "could not start".into(),
            detail: e.to_string(),
        })?;
    let stream = demux.stdout.take().expect("stdout was piped");
    let decode = decode
        .stdin(Stdio::from(stream))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut decode = match decode {
        Ok(child) => child,
        Err(e) => {
            // ffmpeg would otherwise write into a pipe nobody reads.
            let _ = demux.kill();
            let _ = demux.wait();
            return Err(Error::Command {
                program: "harletty".into(),
                status: "could not start".into(),
                detail: e.to_string(),
            });
        }
    };
    let demux_err = run::tail(demux.stderr.take().expect("stderr was piped"));
    let decode_err = run::tail(decode.stderr.take().expect("stderr was piped"));
    let decode_out = run::tail(decode.stdout.take().expect("stdout was piped"));

    let first_channel = dir.join(format!("{RAW}0.wav"));
    let (mut demux_status, mut decode_status) = (None, None);
    loop {
        if demux_status.is_none() {
            demux_status = demux.try_wait().unwrap_or(None);
        }
        if decode_status.is_none() {
            decode_status = decode.try_wait().unwrap_or(None);
            if decode_status.is_some_and(|s| !s.success()) && demux_status.is_none() {
                let _ = demux.kill();
            }
        }
        if demux_status.is_some() && decode_status.is_some() {
            break;
        }
        let done = fs::metadata(&first_channel).map(|m| m.len()).unwrap_or(0);
        progress(Event::bytes("decoding", done, expected));
        std::thread::sleep(POLL);
    }
    let demux_lines = demux_err.join().unwrap_or_default();
    let decode_lines = decode_err.join().unwrap_or_default();
    let _ = decode_out.join();
    // The decoder first: when both failed it is nearly always the cause,
    // and ffmpeg's failure is the broken pipe that followed.
    for (program, status, lines) in [
        ("harletty", decode_status, &decode_lines),
        ("ffmpeg", demux_status, &demux_lines),
    ] {
        let status = status.expect("both children have been waited on");
        if !status.success() {
            return Err(Error::Command {
                program: program.into(),
                status: status.to_string(),
                detail: last_error(lines),
            });
        }
    }
    Ok(())
}

/// `canal_<n>.wav` renamed `vo_<id>.wav` after the master set's own order:
/// one move each, nothing copied.
fn rename(master: &MasterSet, dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::with_capacity(master.channels.len());
    for channel in &master.channels {
        let raw = dir.join(format!("{RAW}{}.wav", channel.index));
        if !raw.is_file() {
            return Err(Error::msg(format!(
                "{}: the decoder wrote no channel {} of the {} its master set declares",
                dir.display(),
                channel.index,
                master.channels.len()
            )));
        }
        let path = dir.join(format!("{ORIGINAL}{}.wav", channel.id));
        fs::rename(&raw, &path).map_err(|e| Error::io(&path, e))?;
        paths.push(path);
    }
    let extra = dir.join(format!("{RAW}{}.wav", master.channels.len()));
    if extra.is_file() {
        return Err(Error::msg(format!(
            "{}: the decoder wrote more channels than the {} its master set declares",
            dir.display(),
            master.channels.len()
        )));
    }
    Ok(paths)
}

/// The `channelsplit` that turns the dub into one stream per channel.
fn filter(dub: &AudioTrack, names: &[&str]) -> String {
    let labels: String = (0..names.len()).map(|i| format!("[a{i}]")).collect();
    format!(
        "[0:a:{}]channelsplit=channel_layout={}{labels}",
        dub.index - 1,
        ffmpeg_layout(dub.channels)
    )
}

/// What ffmpeg is told the dub's layout is. The names are only used to
/// interpret the channel order, which is the container's either way, so an
/// unnamed count does as well for anything unusual.
fn ffmpeg_layout(channels: u32) -> String {
    match channels {
        1 => "mono".to_string(),
        2 => "stereo".to_string(),
        6 => "5.1".to_string(),
        8 => "7.1".to_string(),
        n => format!("{n}c"),
    }
}

/// The name each channel of a dub is written under, in the container's
/// order, as a master set's bed names its channels. ffmpeg's 5.1 is
/// FL FR FC LFE BL BR and its 7.1 is FL FR FC LFE BL BR SL SR, so the 7.1
/// pair that comes first is the rear one.
fn layout(channels: u32) -> Result<Vec<&'static str>> {
    Ok(match channels {
        0 => {
            return Err(Error::msg(
                "the dub declares no channel; ffmpeg cannot split it",
            ));
        }
        1 => vec!["C"],
        2 => vec!["L", "R"],
        6 => vec!["L", "R", "C", "LFE", "Lss", "Rss"],
        8 => vec!["L", "R", "C", "LFE", "Lrs", "Rrs", "Lss", "Rss"],
        n => {
            return Err(Error::msg(format!(
                "a dub of {n} channels has no known layout; use --vf to pick a 2.0, 5.1 or 7.1 track"
            )));
        }
    })
}

/// Delete what a previous run left under a prefix, so nothing of it is
/// taken for this run's.
fn clear(dir: &Path, prefix: &str) -> Result<()> {
    for entry in fs::read_dir(dir).map_err(|e| Error::io(dir, e))? {
        let path = entry.map_err(|e| Error::io(dir, e))?.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if name.starts_with(prefix) && name.ends_with(".wav") {
            fs::remove_file(&path).map_err(|e| Error::io(&path, e))?;
        }
    }
    Ok(())
}

/// The line of a failing run worth showing. ffmpeg prints a banner and its
/// warnings before the one line that says what went wrong, and running out
/// of disk is what goes wrong most.
fn last_error(lines: &[String]) -> String {
    const FIRST: [&str; 7] = [
        "No space left",
        "Permission denied",
        "not found",
        "Invalid",
        "Could not",
        "Error",
        "failed",
    ];
    for needle in FIRST {
        if let Some(line) = lines.iter().find(|l| l.contains(needle)) {
            return line.clone();
        }
    }
    lines
        .last()
        .cloned()
        .unwrap_or_else(|| "(no detail)".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cut_is_the_first_keyframe_at_or_after_the_start() {
        let packets =
            "59.939000,K__\n59.940000,___\n60.000000,___\nN/A,K__\n60.046000,K__\n60.144000,K__\n";
        assert_eq!(first_keyframe(packets, 60.0), Some(60.046));
        assert_eq!(first_keyframe(packets, 59.939), Some(59.939));
        assert_eq!(first_keyframe(packets, 61.0), None);
        assert_eq!(first_keyframe("", 0.0), None);
    }

    #[test]
    fn a_time_reads_as_a_clock() {
        assert_eq!(clock(0.0), "0:00:00.000");
        assert_eq!(clock(779.0), "0:12:59.000");
        assert_eq!(clock(3_725.046), "1:02:05.046");
    }
    use crate::probe::Codec;

    fn track(index: u32, channels: u32) -> AudioTrack {
        AudioTrack {
            index,
            stream: index,
            codec: Codec::Eac3,
            profile: None,
            channels,
            sample_rate: 48000,
            bit_rate: 640_000,
            language: Some("fre".into()),
            title: None,
            default: false,
            start: 0.0,
            side: false,
            spatial: None,
            probe_error: None,
        }
    }

    #[test]
    fn a_five_one_dub_splits_into_the_bed_names() {
        assert_eq!(layout(6).unwrap(), ["L", "R", "C", "LFE", "Lss", "Rss"]);
        // ffmpeg's 7.1 puts the rear pair before the side pair.
        assert_eq!(
            layout(8).unwrap(),
            ["L", "R", "C", "LFE", "Lrs", "Rrs", "Lss", "Rss"]
        );
    }

    #[test]
    fn a_layout_nobody_knows_says_so_instead_of_writing_ch0() {
        assert!(layout(0).is_err());
        assert!(layout(7).is_err());
    }

    #[test]
    fn every_output_of_the_split_is_labelled() {
        let filter = filter(&track(4, 6), &layout(6).unwrap());
        assert_eq!(
            filter,
            "[0:a:3]channelsplit=channel_layout=5.1[a0][a1][a2][a3][a4][a5]"
        );
    }

    #[test]
    fn the_dub_paths_are_the_same_before_and_after_the_run() {
        let dir = Path::new("/job/1.extract");
        let paths = dub_paths(dir, 6).unwrap();
        assert_eq!(paths[3].0, "LFE");
        assert_eq!(paths[3].1, dir.join("vf_LFE.wav"));
    }

    #[test]
    fn a_full_disk_is_reported_rather_than_the_last_warning() {
        let lines = [
            "[matroska @ 0x1] Warning: something".to_string(),
            "av_interleaved_write_frame(): No space left on device".to_string(),
            "Conversion failed!".to_string(),
        ];
        assert!(last_error(&lines).contains("No space left"));
        assert_eq!(last_error(&[]), "(no detail)");
    }
}
