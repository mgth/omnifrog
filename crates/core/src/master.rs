//! The master set the decoder writes: reading the `.atmos` header.
//!
//! An Atmos master set is three files — `<name>.atmos` (the header, a
//! small YAML), `<name>.atmos.metadata` (what moves where, per block) and
//! `<name>.atmos.audio` (one interleaved CAF of every waveform). The header
//! is what says how many waveforms there are, which of them are the bed and
//! which are objects, and what each one's identifier is.
//!
//! The header is read far enough to name the channels: the decoder writes
//! them as `<prefix>_<n>.wav`, n from 0 in the header's own order, and this
//! is what turns that rank back into the identifier the rest of the
//! pipeline uses (`vo_10.wav` is object 10, `vo_3.wav` is the LFE).
//!
//! The metadata is read far enough to know where each object *is* over
//! time, which the loudness of the original's dialogue needs: BS.1770
//! weights a surround channel above a front one, and an object is not a
//! channel — it moves, so its weight moves with it.
//!
//! And a master set is written here: the original's, with the dub's voices
//! added to it as static objects at the end. The original's header and
//! metadata are copied line for line — every element keeps exactly the
//! metadata it arrived with, which is what an overlay encode relies on —
//! and the voices' objects are appended. The audio is not written here; it
//! is interleaved by [`crate::caf`] when an encoder needs it.

use crate::job::Job;
use crate::progress::{Event, Progress};
use crate::voices::VoiceObject;
use crate::{Error, Result, duplicates, extract, wav};
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// One waveform of the master set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Channel {
    /// Rank in the interleaved file, from 0: the `n` of `<prefix>_<n>.wav`.
    pub index: usize,
    /// The DAMF identifier: 0 to 9 for the bed, 10 and up for objects.
    pub id: u32,
    /// `L`, `LFE`, `Lss`… for the bed; `obj<id>` for an object.
    pub name: String,
    pub object: bool,
}

impl Channel {
    /// The one channel that is never separated and never re-voiced.
    pub fn is_lfe(&self) -> bool {
        !self.object && self.name.eq_ignore_ascii_case("LFE")
    }
}

/// A master set, as its header states it.
#[derive(Debug, Clone, PartialEq)]
pub struct MasterSet {
    pub header: PathBuf,
    pub metadata: PathBuf,
    /// The interleaved CAF the header names. `--mono-prefix` does not write
    /// it, so this path usually does not exist.
    pub audio: PathBuf,
    /// What the stream was, as the decoder labelled it: `TrueHD`,
    /// `EAC3-JOC`, `DTS:X-7.1.4+5`, `Auro-3D-13.1`…
    pub source_codec: String,
    pub fps: String,
    pub channels: Vec<Channel>,
}

impl MasterSet {
    pub fn objects(&self) -> impl Iterator<Item = &Channel> {
        self.channels.iter().filter(|c| c.object)
    }

    pub fn bed(&self) -> impl Iterator<Item = &Channel> {
        self.channels.iter().filter(|c| !c.object)
    }

    /// Read the header of the master set at `path`.
    pub fn read(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
        let dir = path.parent().unwrap_or(Path::new("."));
        Self::parse(&text, path, dir)
    }

    /// The only `.atmos` in a directory, for a decoder that named it
    /// itself.
    pub fn find(dir: &Path) -> Result<Self> {
        let mut found = Vec::new();
        for entry in fs::read_dir(dir).map_err(|e| Error::io(dir, e))? {
            let path = entry.map_err(|e| Error::io(dir, e))?.path();
            if path.extension().is_some_and(|e| e == "atmos") {
                found.push(path);
            }
        }
        match found.len() {
            1 => Self::read(&found[0]),
            0 => Err(Error::msg(format!(
                "{}: no .atmos master set",
                dir.display()
            ))),
            n => Err(Error::msg(format!(
                "{}: {n} .atmos master sets, expected one",
                dir.display()
            ))),
        }
    }

    /// The header is a small, flat YAML the decoder writes itself: one
    /// presentation, a list of bed channels with their identifier, then a
    /// list of object identifiers. Reading it line by line keeps the crate
    /// free of a YAML shape it would have to follow; an unexpected file
    /// fails on the channel count rather than being half-read.
    fn parse(text: &str, header: &Path, dir: &Path) -> Result<Self> {
        let (mut metadata, mut audio, mut source_codec, mut fps) =
            (String::new(), String::new(), String::new(), String::new());
        let mut bed: Vec<(u32, String)> = Vec::new();
        let mut objects: Vec<u32> = Vec::new();
        let mut section = Section::None;
        let mut channel_name = String::from("LFE");
        for line in text.lines() {
            let line = line.trim();
            if let Some(v) = after(line, "metadata:") {
                metadata = v.to_string();
            } else if let Some(v) = after(line, "audio:") {
                audio = v.to_string();
            } else if let Some(v) = after(line, "sourceCodec:") {
                source_codec = v.to_string();
            } else if let Some(v) = after(line, "fps:") {
                fps = v.to_string();
            } else if line.starts_with("bedInstances:") {
                section = Section::Bed;
            } else if line.starts_with("objects:") {
                section = Section::Objects;
            } else if section == Section::Bed {
                // `- channels:` is the list header, `- channel: Lss` a name:
                // the trailing colon tells them apart.
                if let Some(v) = after(line, "channel:") {
                    channel_name = v.to_string();
                } else if let Some(id) = after(line, "ID:").and_then(|v| v.parse().ok()) {
                    bed.push((id, std::mem::take(&mut channel_name)));
                }
            } else if section == Section::Objects
                && let Some(id) = after(line, "ID:").and_then(|v| v.parse().ok())
            {
                objects.push(id);
            }
        }
        if bed.is_empty() && objects.is_empty() {
            return Err(Error::msg(format!(
                "{}: no bed and no object; is it a master set header?",
                header.display()
            )));
        }
        let mut channels = Vec::with_capacity(bed.len() + objects.len());
        for (id, name) in bed {
            channels.push(Channel {
                index: channels.len(),
                id,
                name,
                object: false,
            });
        }
        for id in objects {
            channels.push(Channel {
                index: channels.len(),
                id,
                name: format!("obj{id}"),
                object: true,
            });
        }
        Ok(Self {
            header: header.to_path_buf(),
            metadata: dir.join(&metadata),
            audio: dir.join(&audio),
            source_codec,
            fps,
            channels,
        })
    }
}

/// Where one object is, at one instant.
pub type Keyframe = (u64, [f64; 3]);

/// Where every object of a master set is over time.
///
/// The metadata file states one event per change, each naming an object,
/// the sample it takes effect at, and — when the change is a move — the
/// position it moves to. A film's file is tens of megabytes of that, so
/// only the positions are kept: an event that changes a gain or a zone is
/// read and dropped.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Trajectories {
    pub sample_rate: f64,
    by_id: BTreeMap<u32, Vec<Keyframe>>,
}

impl Trajectories {
    pub fn read(path: &Path) -> Result<Self> {
        let file = fs::File::open(path).map_err(|e| Error::io(path, e))?;
        Self::parse(BufReader::new(file)).map_err(|e| match e {
            Error::Message(text) => Error::msg(format!("{}: {text}", path.display())),
            e => e,
        })
    }

    /// Where object `id` is over time, oldest first; empty for a bed
    /// channel, which states no position and does not move.
    pub fn of(&self, id: u32) -> &[Keyframe] {
        self.by_id.get(&id).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.by_id.keys().copied()
    }

    /// Read the events line by line, keeping only the positions. The file
    /// is never held: a film's is tens of megabytes of text and this runs
    /// while a decode may still be going.
    fn parse(source: impl BufRead) -> Result<Self> {
        let mut trajectories = Self {
            sample_rate: 48000.0,
            by_id: BTreeMap::new(),
        };
        let (mut id, mut sample, mut position) = (None, 0u64, None);
        let mut saw_events = false;
        for line in source.lines() {
            let line = line.map_err(|e| Error::msg(format!("reading the metadata: {e}")))?;
            let line = line.trim();
            if let Some(v) = after(line, "sampleRate:") {
                trajectories.sample_rate = v.parse().unwrap_or(48000.0);
            } else if line.starts_with("events:") {
                saw_events = true;
            } else if line.starts_with("- ID:") {
                trajectories.push(id.take(), sample, position.take());
                // `after` has already stepped over the list dash.
                id = after(line, "ID:").and_then(|v| v.parse().ok());
                sample = 0;
            } else if let Some(v) = after(line, "samplePos:") {
                // A sample position may be written with a decimal point.
                sample = v.split('.').next().unwrap_or("0").parse().unwrap_or(0);
            } else if let Some(v) = after(line, "pos:") {
                position = parse_position(v);
            }
        }
        trajectories.push(id, sample, position);
        if !saw_events {
            return Err(Error::msg(
                "no `events:` section; is it a master set's metadata?",
            ));
        }
        for keyframes in trajectories.by_id.values_mut() {
            keyframes.sort_by_key(|(sample, _)| *sample);
        }
        Ok(trajectories)
    }

    fn push(&mut self, id: Option<u32>, sample: u64, position: Option<[f64; 3]>) {
        if let (Some(id), Some(position)) = (id, position) {
            self.by_id.entry(id).or_default().push((sample, position));
        }
    }
}

/// `[x, y, z]` — left to right, back to screen, floor to ceiling.
fn parse_position(text: &str) -> Option<[f64; 3]> {
    let inside = text.trim().strip_prefix('[')?.strip_suffix(']')?;
    let mut axes = [0.0; 3];
    let mut count = 0;
    for (into, value) in axes.iter_mut().zip(inside.split(',')) {
        *into = value.trim().parse().ok()?;
        count += 1;
    }
    (count == 3).then_some(axes)
}

/// The re-voiced master set, as the encoders take it.
#[derive(Debug, Clone, PartialEq)]
pub struct Revoiced {
    /// `3.master/vf.atmos`.
    pub header: PathBuf,
    /// Every waveform, in the master set's order: the original's elements
    /// without their dialogue (the LFE as it was), then the voices.
    pub inputs: Vec<PathBuf>,
    /// How many of the trailing waveforms are voices: `--overlay K`.
    pub voices: usize,
    pub frames: u64,
    /// What the voices were brought up or down by, before each one's
    /// share, in dB.
    pub gain_db: f64,
    /// Samples of the levelled voices that did not fit 24 bits.
    pub clipped: u64,
    /// What the duplicates' cancellation did, when it ran.
    pub duplicates: Option<crate::duplicates::Report>,
}

/// The name of the re-voiced master set inside `3.master/`.
pub const REVOICED: &str = "vf";

/// Build the re-voiced master set: the original's elements with their
/// dialogue taken out, and after them the dub's voices, each levelled and
/// placed as a static object.
///
/// `gain_db` is the levelling — the original's dialogue loudness minus the
/// dub's, plus whatever the run asked for on top — and `dub_channels` how
/// many channels the dub's dialogue was taken from.
///
/// The level of a false multichannel needs one correction. Its voice is the
/// centre alone, placed three times at shares of 1, ½ and ½; BS.1770 sums
/// energies, so the loudness measured over N identical channels is N times
/// the centre's, and three objects at those shares deliver 1.5 times it.
/// The base gain is scaled by √(N / 1.5) so the placed voice measures what
/// the dub's channels measured together.
///
/// With `cancel_duplicates`, the voices are written by
/// [`crate::duplicates::cancel`]: moved onto the original's clock and rid of
/// the original's music and effects they still carry.
pub fn revoice(
    job: &Job,
    voices: &[VoiceObject],
    identical: bool,
    dub_channels: usize,
    gain_db: f64,
    cancel_duplicates: bool,
    progress: Progress,
) -> Result<Revoiced> {
    let original = MasterSet::find(&job.extract_dir())?;
    let into = job.master_dir();
    fs::create_dir_all(&into).map_err(|e| Error::io(&into, e))?;
    if voices.is_empty() {
        return Err(Error::msg(
            "no voice to place: the dub's separation left none",
        ));
    }

    // The master set's length is the original's, whatever the dub's is: a
    // dub a minute short leaves a minute without French, not a film a
    // minute short.
    let first = job.extract_dir().join(format!(
        "{}{}.wav",
        extract::ORIGINAL,
        original.channels[0].id
    ));
    let frames = wav::frames(&first);
    if frames == 0 {
        return Err(Error::msg(format!(
            "{}: not there; the extraction has to run again",
            first.display()
        )));
    }

    let mut inputs = Vec::with_capacity(original.channels.len() + voices.len());
    for channel in &original.channels {
        let path = if channel.is_lfe() {
            // The one channel with no dialogue in it, kept as it came.
            job.extract_dir()
                .join(format!("{}{}.wav", extract::ORIGINAL, channel.id))
        } else {
            job.separate_dir()
                .join(format!("{}{}.wav", extract::MUSIC_AND_EFFECTS, channel.id))
        };
        if !wav::holds(&path, frames) {
            return Err(Error::msg(format!(
                "{}: missing or not the original's length; the separation has to run again",
                path.display()
            )));
        }
        inputs.push(path);
    }

    let base = 10f64.powf(gain_db / 20.0)
        * if identical {
            (dub_channels.max(1) as f64 / 1.5).sqrt()
        } else {
            1.0
        };
    let first_id = original.channels.iter().map(|c| c.id).max().unwrap_or(9) + 1;
    let voice_file = |channel: &str| {
        job.separate_dir()
            .join(format!("{}{channel}.wav", extract::DUB_VOICE))
    };
    let mut added = Vec::with_capacity(voices.len());
    let mut outputs = Vec::with_capacity(voices.len());
    for (k, voice) in voices.iter().enumerate() {
        let id = first_id + k as u32;
        let path = into.join(format!("voice_{id}.wav"));
        let gain = base * voice.weight;
        progress(Event::message(format!(
            "voice {id}: {} at {} × {gain:.3} ({:+.2} dB)",
            voice.channel,
            position_text(&voice.position),
            20.0 * gain.log10()
        )));
        outputs.push(duplicates::Output {
            channel: voice.channel.clone(),
            path: path.clone(),
            gain: gain as f32,
        });
        inputs.push(path);
        added.push(Added {
            id,
            position: voice.position,
        });
    }
    let mut clipped = 0;
    let mut report = None;
    if cancel_duplicates {
        // The map's references: the original's elements and the dub's
        // channels as decoded, their LFE aside; what is cancelled: the
        // original's elements as the new mix carries them.
        let elements: Vec<&Channel> = original.channels.iter().filter(|c| !c.is_lfe()).collect();
        let decoded: Vec<PathBuf> = elements
            .iter()
            .map(|c| {
                job.extract_dir()
                    .join(format!("{}{}.wav", extract::ORIGINAL, c.id))
            })
            .collect();
        let effects: Vec<PathBuf> = elements
            .iter()
            .map(|c| {
                job.separate_dir()
                    .join(format!("{}{}.wav", extract::MUSIC_AND_EFFECTS, c.id))
            })
            .collect();
        let separated = job
            .manifest
            .separate_dub
            .as_ref()
            .map(|state| state.elements.clone())
            .unwrap_or_default();
        let dub: Vec<PathBuf> = separated
            .iter()
            .filter(|name| !name.eq_ignore_ascii_case("LFE"))
            .map(|name| {
                job.extract_dir()
                    .join(format!("{}{name}.wav", extract::DUB))
            })
            .collect();
        let mut channels: Vec<(String, PathBuf)> = Vec::new();
        for voice in voices {
            if !channels.iter().any(|(c, _)| *c == voice.channel) {
                channels.push((voice.channel.clone(), voice_file(&voice.channel)));
            }
        }
        // Both tracks' first samples are index 0 of their files: the dub's
        // voices are late by the original's start minus the dub's — none
        // when both were cut at the same keyframe.
        let offset = ((job.manifest.original_begin() - job.manifest.dub_begin())
            * crate::duplicates::RATE as f64)
            .round() as i64;
        let (done, over) = duplicates::cancel(
            &duplicates::Plan {
                original: &decoded,
                dub: &dub,
                effects: &effects,
                voices: &channels,
                outputs: &outputs,
                frames,
                offset,
            },
            progress,
        )?;
        progress(Event::message(format!("duplicates: {}", done.text())));
        clipped = over;
        report = Some(done);
    } else {
        for output in &outputs {
            let temporary = output.path.with_extension("wav.part");
            clipped += wav::copy_mono(
                &voice_file(&output.channel),
                &temporary,
                frames,
                crate::separate::RATE,
                output.gain,
                wav::Sample::Pcm(24),
            )?;
            fs::rename(&temporary, &output.path).map_err(|e| Error::io(&output.path, e))?;
        }
    }
    if clipped > 0 {
        progress(Event::message(format!(
            "{clipped} samples of the levelled voices were clipped to fit 24 bits"
        )));
    }

    let header = into.join(REVOICED).with_extension("atmos");
    let audio = format!("{REVOICED}.atmos.audio");
    let metadata = format!("{REVOICED}.atmos.metadata");
    write_metadata(&original.metadata, &into.join(&metadata), &added)?;
    let ids: Vec<u32> = added.iter().map(|a| a.id).collect();
    write_header(&original, &header, &audio, &metadata, &ids)?;
    Ok(Revoiced {
        header,
        inputs,
        voices: voices.len(),
        frames,
        gain_db: 20.0 * base.log10(),
        clipped,
        duplicates: report,
    })
}

/// A static object added to a master set: a voice, at a place.
#[derive(Debug, Clone, PartialEq)]
pub struct Added {
    pub id: u32,
    pub position: [f64; 3],
}

/// Copy `source`, a master set's metadata, to `destination` with `added`
/// inserted as static objects: one event each, at sample 0, stating the
/// whole state of the object.
///
/// They go after the source's own events at sample 0 and before its first
/// later one, so the file stays in time order. The source is read and
/// written a line at a time — a film's is tens of megabytes — and its lines
/// are copied as they are: an element's metadata is the original's, event
/// for event.
pub fn write_metadata(source: &Path, destination: &Path, added: &[Added]) -> Result<()> {
    use std::io::Write;
    let input = fs::File::open(source).map_err(|e| Error::io(source, e))?;
    let output = fs::File::create(destination).map_err(|e| Error::io(destination, e))?;
    let mut out = std::io::BufWriter::new(output);
    let write = |out: &mut std::io::BufWriter<fs::File>, line: &str| {
        out.write_all(line.as_bytes())
            .and_then(|_| out.write_all(b"\n"))
            .map_err(|e| Error::io(destination, e))
    };

    let mut in_events = false;
    let mut inserted = added.is_empty();
    // The event being read, held until its sample position is known.
    let mut event: Vec<String> = Vec::new();
    let mut event_sample: Option<u64> = None;
    let flush = |out: &mut std::io::BufWriter<fs::File>,
                 event: &mut Vec<String>,
                 sample: Option<u64>,
                 inserted: &mut bool|
     -> Result<()> {
        if event.is_empty() {
            return Ok(());
        }
        if !*inserted && sample.unwrap_or(0) > 0 {
            for object in added {
                for line in static_object(object) {
                    write(out, &line)?;
                }
            }
            *inserted = true;
        }
        for line in event.drain(..) {
            write(out, &line)?;
        }
        Ok(())
    };
    for line in BufReader::new(input).lines() {
        let line = line.map_err(|e| Error::io(source, e))?;
        let trimmed = line.trim();
        if !in_events {
            write(&mut out, &line)?;
            if trimmed.starts_with("events:") {
                in_events = true;
            }
            continue;
        }
        if trimmed.starts_with("- ID:") {
            flush(&mut out, &mut event, event_sample, &mut inserted)?;
            event_sample = None;
        } else if let Some(v) = after(trimmed, "samplePos:") {
            event_sample = v.split('.').next().and_then(|v| v.parse().ok());
        }
        event.push(line);
    }
    flush(&mut out, &mut event, event_sample, &mut inserted)?;
    if !in_events {
        return Err(Error::msg(format!(
            "{}: no `events:` section; is it a master set's metadata?",
            source.display()
        )));
    }
    if !inserted {
        for object in added {
            for line in static_object(object) {
                write(&mut out, &line)?;
            }
        }
    }
    out.flush().map_err(|e| Error::io(destination, e))
}

/// One static object's event: its whole state at sample 0, in the field
/// order a master set's own events use.
fn static_object(object: &Added) -> Vec<String> {
    vec![
        format!("  - ID: {}", object.id),
        "    samplePos: 0".to_string(),
        "    active: true".to_string(),
        format!("    pos: {}", position_text(&object.position)),
        "    snap: false".to_string(),
        "    elevation: true".to_string(),
        "    zones: all".to_string(),
        "    size: 0.0".to_string(),
        "    importance: 1.0".to_string(),
        "    gain: 0".to_string(),
        "    rampLength: 608".to_string(),
        "    trimBypass: false".to_string(),
        "    dialog: -1".to_string(),
        "    music: -1".to_string(),
        "    screenFactor: 0.0".to_string(),
        "    depthFactor: 0.25".to_string(),
        "    headTrackMode: undefined".to_string(),
        "    binauralRenderMode: undefined".to_string(),
    ]
}

/// `[0, 1, 0]`, `[-1, 0.5, 0]`: whole numbers without a decimal point, as
/// a master set writes them.
fn position_text(position: &[f64; 3]) -> String {
    let axis = |v: f64| {
        if v == v.trunc() {
            format!("{}", v as i64)
        } else {
            format!("{v}")
        }
    };
    format!(
        "[{}, {}, {}]",
        axis(position[0]),
        axis(position[1]),
        axis(position[2])
    )
}

/// Write the header of a master set derived from `source`: the same
/// presentation, pointing at `audio` and `metadata`, with the objects of
/// `added` after the source's own.
///
/// Two things are not carried over, because they would no longer be true.
/// `sourceCodec` says what stream a master was decoded from, and a
/// re-voiced one was not decoded from anything — its elements came out of
/// one stream and its voices out of another; the manifest keeps what the
/// original was. And its creation tool is this program, not the decoder. An
/// encoder reads neither, and one of them (harlettizer, reading the labels
/// against a closed list) refuses a label it has not heard of.
pub fn write_header(
    source: &MasterSet,
    destination: &Path,
    audio: &str,
    metadata: &str,
    added: &[u32],
) -> Result<()> {
    let text = fs::read_to_string(&source.header).map_err(|e| Error::io(&source.header, e))?;
    let mut out = String::with_capacity(text.len() + added.len() * 16);
    let mut object_indent: Option<String> = None;
    let mut has_objects = false;
    for line in text.lines() {
        let trimmed = line.trim_start();
        let indent = &line[..line.len() - trimmed.len()];
        if trimmed.starts_with("metadata:") {
            out.push_str(&format!("{indent}metadata: {metadata}\n"));
            continue;
        }
        if trimmed.starts_with("audio:") {
            out.push_str(&format!("{indent}audio: {audio}\n"));
            continue;
        }
        if trimmed.starts_with("sourceCodec:") {
            continue;
        }
        if trimmed.starts_with("creationTool:") {
            out.push_str(&format!("{indent}creationTool: {}\n", crate::NAME));
            continue;
        }
        if trimmed.starts_with("creationToolVersion:") {
            out.push_str(&format!(
                "{indent}creationToolVersion: {}\n",
                crate::VERSION
            ));
            continue;
        }
        if trimmed.starts_with("objects:") {
            has_objects = true;
        } else if has_objects && trimmed.starts_with("- ID:") {
            object_indent = Some(indent.to_string());
        }
        out.push_str(line);
        out.push('\n');
    }
    if !added.is_empty() && !has_objects {
        // A master of bed channels alone: the objects need a list to go in.
        out.push_str("    objects:\n");
    }
    let indent = object_indent.unwrap_or_else(|| "      ".to_string());
    for id in added {
        out.push_str(&format!("{indent}- ID: {id}\n"));
    }
    fs::write(destination, out).map_err(|e| Error::io(destination, e))
}

#[derive(PartialEq, Eq)]
enum Section {
    None,
    Bed,
    Objects,
}

/// The value of `key: value`, with the list dash and the quotes off.
fn after<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let rest = line.strip_prefix("- ").unwrap_or(line).trim_start();
    let rest = rest.strip_prefix(key)?;
    Some(rest.trim().trim_matches(['"', '\'']))
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = r#"version: 0.5.1
presentations:
  - type: home
    simplified: false
    metadata: vo.atmos.metadata
    audio: vo.atmos.audio
    offset: 0.0
    fps: 24
    scBedConfiguration: [0, 1, 2, 3, 4, 5, 6, 7]
    creationTool: harletty
    creationToolVersion: 0.8.0
    sourceCodec: DTS:X-7.1.4+5
    bedInstances:
      - channels:
          - channel: L
            ID: 0
          - channel: R
            ID: 1
          - channel: C
            ID: 2
          - channel: LFE
            ID: 3
          - channel: Lss
            ID: 4
          - channel: Rss
            ID: 5
          - channel: Lrs
            ID: 6
          - channel: Rrs
            ID: 7
    objects:
      - ID: 10
      - ID: 11
"#;

    #[test]
    fn the_bed_comes_first_and_keeps_the_header_order() {
        let set = MasterSet::parse(HEADER, Path::new("/job/vo.atmos"), Path::new("/job")).unwrap();
        assert_eq!(set.source_codec, "DTS:X-7.1.4+5");
        assert_eq!(set.metadata, Path::new("/job/vo.atmos.metadata"));
        assert_eq!(set.channels.len(), 10);
        assert_eq!(set.bed().count(), 8);
        assert_eq!(set.objects().count(), 2);
        assert_eq!(set.channels[3].name, "LFE");
        assert!(set.channels[3].is_lfe());
        assert_eq!(set.channels[8].index, 8);
        assert_eq!(set.channels[8].id, 10);
        assert_eq!(set.channels[8].name, "obj10");
        assert!(set.channels[8].object);
    }

    #[test]
    fn the_list_header_is_not_a_channel_name() {
        let set = MasterSet::parse(HEADER, Path::new("/job/vo.atmos"), Path::new("/job")).unwrap();
        let names: Vec<&str> = set.bed().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["L", "R", "C", "LFE", "Lss", "Rss", "Lrs", "Rrs"]);
    }

    #[test]
    fn a_file_that_is_not_a_header_fails_rather_than_reading_empty() {
        let err = MasterSet::parse("hello: world\n", Path::new("/x.atmos"), Path::new("/"));
        assert!(err.is_err());
    }

    const METADATA: &str = r#"sampleRate: 48000
events:
  - ID: 3
    samplePos: 21
    active: true
    gain: 0
  - ID: 10
    samplePos: 21
    active: true
    pos: [-1, 1, 0]
    size: 0.0
  - ID: 11
    samplePos: 21
    active: true
    pos: [0.5, -0.25, 0.75]
  - ID: 10
    samplePos: 48021
    pos: [-0.5, 0.25, 0]
  - ID: 10
    samplePos: 96021
    gain: -3
"#;

    #[test]
    fn an_object_keeps_the_positions_it_moves_through() {
        let t = Trajectories::parse(METADATA.as_bytes()).unwrap();
        assert_eq!(t.sample_rate, 48000.0);
        assert_eq!(
            t.of(10),
            [(21, [-1.0, 1.0, 0.0]), (48021, [-0.5, 0.25, 0.0])]
        );
        assert_eq!(t.of(11), [(21, [0.5, -0.25, 0.75])]);
        // A bed channel states no position, and an event that changes
        // something else is not a move.
        assert!(t.of(3).is_empty());
        assert_eq!(t.ids().collect::<Vec<_>>(), [10, 11]);
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("omnifrog-master-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_voice_object_goes_in_before_the_first_later_event() {
        let dir = scratch("meta");
        let (source, destination) = (dir.join("vo.atmos.metadata"), dir.join("vf.atmos.metadata"));
        fs::write(&source, METADATA).unwrap();
        write_metadata(
            &source,
            &destination,
            &[Added {
                id: 12,
                position: [-1.0, 1.0, 0.0],
            }],
        )
        .unwrap();
        let written = fs::read_to_string(&destination).unwrap();
        // The source's first events are at sample 21, so the voice, at 0,
        // opens the list and the file stays in time order.
        let first_event = written.find("  - ID:").unwrap();
        assert!(written[first_event..].starts_with("  - ID: 12\n    samplePos: 0\n"));
        assert!(written.contains("    pos: [-1, 1, 0]\n"));
        // Every line of the source is there, in its order.
        let rest: Vec<&str> = written.lines().filter(|l| !l.is_empty()).collect();
        let original: Vec<&str> = METADATA.lines().collect();
        let mut at = 0;
        for line in &rest {
            if at < original.len() && *line == original[at] {
                at += 1;
            }
        }
        assert_eq!(at, original.len(), "{written}");
        // And it reads back with the voice where it was put.
        let t = Trajectories::read(&destination).unwrap();
        assert_eq!(t.of(12), [(0, [-1.0, 1.0, 0.0])]);
        assert_eq!(t.of(10).len(), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn voices_follow_the_events_already_at_sample_zero() {
        let dir = scratch("zero");
        let (source, destination) = (dir.join("a.metadata"), dir.join("b.metadata"));
        fs::write(
            &source,
            "sampleRate: 48000\nevents:\n  - ID: 10\n    samplePos: 0\n    pos: [0, 1, 0]\n  - ID: 10\n    samplePos: 480\n    pos: [1, 1, 0]\n",
        )
        .unwrap();
        write_metadata(
            &source,
            &destination,
            &[Added {
                id: 11,
                position: [0.0, 1.0, 0.0],
            }],
        )
        .unwrap();
        let written = fs::read_to_string(&destination).unwrap();
        let ids: Vec<&str> = written
            .lines()
            .filter(|l| l.trim().starts_with("- ID:"))
            .collect();
        assert_eq!(ids, ["  - ID: 10", "  - ID: 11", "  - ID: 10"]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_derived_header_points_at_its_own_files_and_lists_the_voices_last() {
        let dir = scratch("header");
        let header = dir.join("vo.atmos");
        fs::write(&header, HEADER).unwrap();
        let set = MasterSet::read(&header).unwrap();
        let derived = dir.join("vf.atmos");
        write_header(
            &set,
            &derived,
            "vf.atmos.audio",
            "vf.atmos.metadata",
            &[19, 20],
        )
        .unwrap();
        let again = MasterSet::read(&derived).unwrap();
        assert_eq!(again.audio, dir.join("vf.atmos.audio"));
        assert_eq!(again.metadata, dir.join("vf.atmos.metadata"));
        // Not decoded from anything, and not made by the decoder.
        assert_eq!(again.source_codec, "");
        let text = fs::read_to_string(&derived).unwrap();
        assert!(text.contains("    creationTool: omnifrog\n"), "{text}");
        assert!(!text.contains("harletty"), "{text}");
        assert_eq!(again.channels.len(), set.channels.len() + 2);
        assert_eq!(again.channels[..set.channels.len()], set.channels[..]);
        let last = again.channels.last().unwrap();
        assert!(last.object);
        assert_eq!(last.id, 20);
        assert!(text.contains("      - ID: 19\n"), "{text}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_position_is_written_as_a_master_set_writes_one() {
        assert_eq!(position_text(&[0.0, 1.0, 0.0]), "[0, 1, 0]");
        assert_eq!(position_text(&[-1.0, 0.5, 0.0]), "[-1, 0.5, 0]");
    }

    #[test]
    fn something_that_is_not_a_metadata_file_is_refused() {
        assert!(Trajectories::parse("sampleRate: 48000\n".as_bytes()).is_err());
    }
}
