//! A job: its directory, its manifest, and where a second run picks up.
//!
//! One film is one job, and one job is one directory named after the film —
//! in ASCII, because Dee cannot open a path with an accent in it. Beside
//! the working directories sits `manifest.json`: every parameter the run
//! was given, and how far it got. It is what makes the pipeline resumable
//! (a film is hours of separation, and nothing is thrown away for a
//! parameter that did not change) and what the track's `OMNIFROG_MANIFEST`
//! tag carries at the end.
//!
//! The rule for picking up is one comparison per parameter: the earliest
//! step a change invalidates is the step the run restarts at. Changing the
//! model that takes the dialogue out of the original does not re-extract
//! the film; changing a track does, because everything downstream came out
//! of that track.

use crate::config::{Config, DEFAULT_IAMF_BITRATE, Encoder, IamfCodec, Inference, Pairs};
use crate::models::Model;
use crate::probe::{AudioTrack, Media};
use crate::text;
use crate::{Error, NAME, Result, VERSION};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// The steps of a run, in order. A run reports « 2/6 » against this list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Step {
    /// One pass: the original decoded into a master set, the dub split.
    Extract,
    /// The dialogue taken out of every element of the original.
    SeparateOriginal,
    /// The dialogue taken out of every channel of the dub.
    SeparateDub,
    /// The dub's voices levelled, placed, and written as a master set.
    Master,
    /// TrueHD Atmos, by harlettizer or by Dee, or IAMF by harlettizer.
    Encode,
    /// The `.mka` or the full `.mkv`, with the tags.
    Remux,
}

impl Step {
    pub const ALL: [Step; 6] = [
        Step::Extract,
        Step::SeparateOriginal,
        Step::SeparateDub,
        Step::Master,
        Step::Encode,
        Step::Remux,
    ];

    /// Rank in a run, from 1.
    pub fn number(self) -> u32 {
        Self::ALL.iter().position(|s| *s == self).unwrap_or(0) as u32 + 1
    }

    pub fn label(self) -> &'static str {
        match self {
            Step::Extract => "extraction",
            Step::SeparateOriginal => "dialogue out of the original",
            Step::SeparateDub => "dialogue out of the dub",
            Step::Master => "master",
            Step::Encode => "encode",
            Step::Remux => "remux",
        }
    }
}

/// One of the source's tracks, as the manifest remembers it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackRef {
    /// Rank among the audio tracks, from 1: what `--vo` / `--vf` name.
    pub index: u32,
    pub stream: u32,
    pub codec: String,
    pub channels: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// What the decoder called the scene, for the original.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spatial: Option<String>,
    /// When its first sample plays, in seconds: what the new track is
    /// shifted by, since it was decoded from a raw stream that starts at 0.
    #[serde(default)]
    pub start: f64,
}

impl TrackRef {
    pub fn of(track: &AudioTrack) -> Self {
        Self {
            index: track.index,
            stream: track.stream,
            codec: track.codec.name().to_string(),
            channels: track.channels,
            language: track.language.clone(),
            title: track.title.clone(),
            spatial: track.spatial.as_ref().map(|s| s.label.clone()),
            start: track.start,
        }
    }
}

/// The two models a run used, by key and by the code that goes in the
/// output's name. The checkpoint is recorded too: two catalogue entries can
/// share one, and the file name alone would not tell them apart.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelRef {
    pub key: String,
    pub code: String,
    pub checkpoint: String,
}

impl ModelRef {
    pub fn of(model: &Model) -> Self {
        Self {
            key: model.key.clone(),
            code: model.code.clone(),
            checkpoint: model.checkpoint.file.clone(),
        }
    }
}

/// What one of the two separations left behind, and what it measured.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeparateState {
    /// What the stems on disk were made for: a change throws them away.
    pub signature: String,
    /// The elements separated, by identifier or by channel name.
    pub elements: Vec<String>,
    /// BS.1770 loudness of the mix of voices, in LKFS; 0 = not measured,
    /// since a valid one is negative.
    pub lkfs: f64,
    /// Only for the dub: every channel is a copy of the centre, so one
    /// voice is placed rather than one per channel.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identical: Option<bool>,
}

/// What the re-voiced master set is made of.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MasterState {
    /// How many of the trailing waveforms are voices: `--overlay K`.
    pub voices: usize,
    /// What the voices were brought by before each one's share, in dB.
    pub gain_db: f64,
    pub frames: u64,
    /// Every waveform in the master set's order, relative to the job's
    /// directory so the job can move.
    pub inputs: Vec<PathBuf>,
    /// Samples of the levelled voices clipped to fit 24 bits.
    #[serde(default)]
    pub clipped: u64,
    /// What the duplicates' cancellation did, when it ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duplicates: Option<crate::duplicates::Report>,
}

/// What the encoder wrote.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncodeState {
    pub encoder: Encoder,
    /// Relative to the job's directory: a raw TrueHD stream, or an `.mka`
    /// of one IAMF track.
    pub stream: PathBuf,
    /// The line worth reading without opening the encoder's log.
    pub summary: String,
    /// What harlettizer rounded the mixed elements to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<u32>,
}

/// What the extraction left on disk, so a later run can tell whether it is
/// still there and still whole.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractState {
    /// What the decoder called the original's scene.
    pub source_codec: String,
    /// The identifiers of the master set's waveforms, in its own order.
    pub elements: Vec<u32>,
    pub objects: usize,
    /// The names the dub's channels were written under.
    pub dub_channels: Vec<String>,
    /// Frames every file holds, at 48 kHz.
    pub frames: u64,
    /// Where both tracks' files begin on the film's clock, in seconds, for
    /// a run from a start point: the original's keyframe it was cut at.
    /// Absent, each track's file begins where the track does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub begin: Option<f64>,
}

impl ExtractState {
    pub fn elements(&self) -> &[u32] {
        &self.elements
    }
}

fn default_iamf_bitrate() -> u32 {
    DEFAULT_IAMF_BITRATE
}

/// Every parameter of a run, and how far it got.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// The program and version that wrote it; the `OMNIFROG_TOOL` tag.
    pub tool: String,
    pub source: PathBuf,
    /// Size of the source when the job was made: a file replaced under the
    /// same name is not the same film.
    pub source_bytes: u64,
    pub duration: f64,
    pub original: TrackRef,
    pub dub: TrackRef,
    /// Only this many seconds were read, for a trial run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seconds: Option<f64>,
    /// Read from this point of the film, in seconds, rather than from its
    /// beginning: a trial of a scene. The run starts at the original's first
    /// keyframe at or after it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<f64>,
    pub model_original: ModelRef,
    pub model_dub: ModelRef,
    /// Which separations hand the models two elements at once. Absent from
    /// a manifest written before it existed, which ran every element alone.
    #[serde(default)]
    pub pairs: Pairs,
    /// What was asked to run the models; absent from a manifest written
    /// before it existed, which ran MSST. The worker may still have fallen
    /// back to MSST for a model pymss cannot load, as its log says.
    #[serde(default = "Inference::msst")]
    pub inference: Inference,
    pub encoder: Encoder,
    pub hz_opts: String,
    /// The options of `harlettizer iamf`; absent from a manifest written
    /// before IAMF could be asked for, which used none.
    #[serde(default)]
    pub iamf_opts: String,
    /// The IAMF codec; absent from a manifest written before it could be
    /// chosen, which left harlettizer's FLAC.
    #[serde(default)]
    pub iamf_codec: IamfCodec,
    /// Opus only, kilobits a second for each channel.
    #[serde(default = "default_iamf_bitrate")]
    pub iamf_bitrate: u32,
    /// FLAC only: the depth chosen by hand; absent, the original's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iamf_bits: Option<u32>,
    pub gain_db: f64,
    /// The dub's voices put on the original's clock and rid of the
    /// original's music and effects. Absent from a manifest written before
    /// it existed, which did neither.
    #[serde(default)]
    pub cancel_duplicates: bool,
    /// A full `.mkv` rather than an `.mka` of the new track alone.
    pub mkv: bool,
    /// Where the track is written; `None` is beside the source, where
    /// a player finds an `.mka` to load with the film.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_dir: Option<PathBuf>,
    /// How many steps of [`Step::ALL`] are done.
    pub done: u32,
    /// Seconds each finished step took, by its name.
    #[serde(default)]
    pub durations: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extract: Option<ExtractState>,
    /// The dialogue taken out of the original's elements.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub separate_original: Option<SeparateState>,
    /// The dialogue taken out of the dub's channels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub separate_dub: Option<SeparateState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub master: Option<MasterState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encode: Option<EncodeState>,
    /// The file the remux wrote.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<PathBuf>,
    /// Keys a later version wrote and this one does not know, kept so a
    /// job is not downgraded by being read.
    #[serde(flatten, default)]
    pub extra: BTreeMap<String, Value>,
}

impl Manifest {
    /// The manifest of a fresh run.
    pub fn new(
        media: &Media,
        original: &AudioTrack,
        dub: &AudioTrack,
        models: (&Model, &Model),
        config: &Config,
    ) -> Self {
        Self {
            tool: format!("{NAME} {VERSION}"),
            source: media.path.clone(),
            source_bytes: fs::metadata(&media.path).map(|m| m.len()).unwrap_or(0),
            duration: media.duration,
            original: TrackRef::of(original),
            dub: TrackRef::of(dub),
            seconds: None,
            start: None,
            model_original: ModelRef::of(models.0),
            model_dub: ModelRef::of(models.1),
            pairs: config.pairs.effective(),
            inference: config.inference,
            encoder: config.encoder,
            hz_opts: config.hz_opts.clone(),
            iamf_opts: config.iamf_opts.clone(),
            iamf_codec: config.iamf_codec,
            iamf_bitrate: config.iamf_bitrate,
            iamf_bits: config.iamf_bits,
            gain_db: 0.0,
            cancel_duplicates: config.cancel_duplicates,
            mkv: false,
            done: 0,
            durations: BTreeMap::new(),
            extract: None,
            separate_original: None,
            separate_dub: None,
            master: None,
            encode: None,
            out_dir: None,
            output: None,
            extra: BTreeMap::new(),
        }
    }

    /// The options the encoder of this run is handed: `hz_opts` for
    /// harlettizer's TrueHD, the codec then `iamf_opts` for its IAMF in
    /// either mode, none for Dee.
    pub fn encoder_options(&self) -> String {
        match self.encoder {
            Encoder::Hz => self.hz_opts.clone(),
            Encoder::Iamf | Encoder::IamfBed => {
                self.iamf_codec
                    .options(self.iamf_bitrate, self.iamf_bits, &self.iamf_opts)
            }
            Encoder::Dee => String::new(),
        }
    }

    /// When the original's first extracted sample plays on the film's clock:
    /// the keyframe a run from a start point was cut at, else the track's
    /// own start. What the new track is shifted by.
    pub fn original_begin(&self) -> f64 {
        self.extract
            .as_ref()
            .and_then(|e| e.begin)
            .unwrap_or(self.original.start)
    }

    /// The same for the dub: cut at the same instant as the original, or
    /// from its own start.
    pub fn dub_begin(&self) -> f64 {
        self.extract
            .as_ref()
            .and_then(|e| e.begin)
            .unwrap_or(self.dub.start)
    }

    /// The gain the dub's voices are brought to the original's dialogue by,
    /// with whatever the run asked for on top. `None` when either loudness
    /// is missing: the levelling is then left alone rather than guessed.
    pub fn voice_gain_db(&self) -> Option<f64> {
        let original = self.separate_original.as_ref()?.lkfs;
        let dub = self.separate_dub.as_ref()?.lkfs;
        (original < 0.0 && dub < 0.0).then_some(original - dub + self.gain_db)
    }

    /// What a previous run measured, kept when its parameters still hold:
    /// everything that is a result rather than an input. One place to
    /// extend whenever a step learns to record something.
    pub fn carry_over(&mut self, previous: &Manifest) {
        self.durations = previous.durations.clone();
        self.extract = previous.extract.clone();
        self.separate_original = previous.separate_original.clone();
        self.separate_dub = previous.separate_dub.clone();
        self.master = previous.master.clone();
        self.encode = previous.encode.clone();
        self.output = previous.output.clone();
        self.done = previous.done;
        for (key, value) in &previous.extra {
            self.extra.entry(key.clone()).or_insert(value.clone());
        }
    }

    /// The first step this run has to do, given what a previous one left;
    /// `None` when there is nothing left to do.
    ///
    /// Two things decide it, and the earlier one wins: how far the previous
    /// run got, and the first step a changed parameter invalidates. So
    /// asking for another encoder on a job that only reached the extraction
    /// carries on at the separation, not at the encoding.
    pub fn resume_at(&self, previous: Option<&Manifest>) -> Option<Step> {
        let Some(previous) = previous else {
            return Some(Step::Extract);
        };
        let next = Step::ALL.get(previous.done as usize).copied();
        match (self.invalidated(previous), next) {
            (Some(changed), Some(next)) => Some(changed.min(next)),
            (Some(changed), None) => Some(changed),
            (None, next) => next,
        }
    }

    /// The first step a change of parameter makes stale, `None` when every
    /// parameter is the one the previous run used.
    ///
    /// A parameter belongs to exactly one step: change it and that step,
    /// with everything after it, is done again. Everything before it is
    /// untouched — which is the whole point, since separating a film is
    /// hours and re-extracting it is tens of gigabytes.
    fn invalidated(&self, previous: &Manifest) -> Option<Step> {
        // What was decoded, and out of what.
        if previous.source != self.source
            || previous.source_bytes != self.source_bytes
            || previous.original.index != self.original.index
            || previous.dub.index != self.dub.index
            || previous.seconds != self.seconds
            || previous.start != self.start
        {
            return Some(Step::Extract);
        }
        if previous.model_original != self.model_original
            || previous.pairs.original != self.pairs.original
        {
            return Some(Step::SeparateOriginal);
        }
        if previous.model_dub != self.model_dub
            || previous.pairs.dub != self.pairs.dub
            || previous.pairs.centre_twice != self.pairs.centre_twice
        {
            return Some(Step::SeparateDub);
        }
        if previous.gain_db != self.gain_db || previous.cancel_duplicates != self.cancel_duplicates
        {
            return Some(Step::Master);
        }
        // Only the options of the encoder in use: another encoder's are
        // not passed to anything, and changing them redoes nothing.
        if previous.encoder != self.encoder || previous.encoder_options() != self.encoder_options()
        {
            return Some(Step::Encode);
        }
        if previous.mkv != self.mkv || previous.out_dir != self.out_dir {
            return Some(Step::Remux);
        }
        None
    }

    /// The codes of the two models, joined as the output's name joins them.
    pub fn model_codes(&self) -> String {
        [&self.model_original.code, &self.model_dub.code]
            .into_iter()
            .filter(|code| !code.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join("-")
    }
}

/// A job on disk.
#[derive(Debug, Clone)]
pub struct Job {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

/// The longest film name a job directory carries on Windows.
///
/// Python there cannot open a path past 260 characters unless long paths
/// were switched on in the registry, and the worker writes about sixty
/// below the job's directory (`2.separate\_stems_element\_msst_out\…`).
/// Release names run past a hundred characters, so on Windows the film part
/// of the directory is cut here; Rust itself, and so everything else in the
/// pipeline, has no such limit.
const WINDOWS_NAME: usize = 48;

/// `name` cut to `most` characters, with a hash of the whole of it where it
/// was cut, so two long names that start alike stay two directories. The
/// hash is FNV-1a, spelled out rather than taken from the standard library,
/// whose hasher may change between releases: a job has to find its
/// directory again after an upgrade.
fn short(name: &str, most: usize) -> String {
    if name.chars().count() <= most {
        return name.to_string();
    }
    let mut hash: u32 = 0x811c_9dc5;
    for byte in name.bytes() {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    let head: String = name.chars().take(most - 9).collect();
    format!("{head}~{hash:08x}")
}

impl Job {
    /// The job of a manifest: one directory per film under the working
    /// directory, `<film>.omnifrog`.
    ///
    /// The suffix is what keeps it this program's own. Another tool may
    /// name its jobs after the film alone, and a working directory shared
    /// with it would otherwise hand this program that tool's job — its
    /// manifest to overwrite and, at the end of a run, its directory to
    /// clean.
    pub fn new(config: &Config, manifest: Manifest) -> Self {
        let stem = manifest
            .source
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "film".to_string());
        let name = if cfg!(windows) {
            short(&text::ascii(&stem), WINDOWS_NAME)
        } else {
            text::ascii(&stem)
        };
        Self {
            dir: config.work_dir().join(format!("{name}.{}", crate::NAME)),
            manifest,
        }
    }

    /// The master set of the original, its waveforms, and the dub's
    /// channels.
    pub fn extract_dir(&self) -> PathBuf {
        self.dir.join("1.extract")
    }

    /// The stems the separation writes.
    pub fn separate_dir(&self) -> PathBuf {
        self.dir.join("2.separate")
    }

    /// The re-voiced master set, the encoded stream and the tags.
    pub fn master_dir(&self) -> PathBuf {
        self.dir.join("3.master")
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.dir.join("manifest.json")
    }

    /// The film's name, which every output is named after.
    pub fn stem(&self) -> String {
        self.manifest
            .source
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "film".to_string())
    }

    /// `<film>.Omnifrog-<code>-<code>-<encoder>.mka`.
    ///
    /// One dotted segment after the film's name, never two: a tool that
    /// pairs the track with its film takes everything before the *last*
    /// dot of the name, and a film's own name is full of dots.
    /// `Some.Film.Omnifrog.uhav.mka` would make it believe in a film called
    /// `Some.Film.Omnifrog`.
    pub fn output_name(&self) -> String {
        // A lossy stream is named for its bitrate, so it lands beside the
        // lossless one of the same models rather than over it — by the
        // codec it is really coded with, free options included.
        let codec = if self.manifest.encoder.is_iamf() {
            self.manifest
                .iamf_codec
                .name_part(self.manifest.iamf_bitrate, &self.manifest.iamf_opts)
        } else {
            String::new()
        };
        let label = [
            "Omnifrog",
            &self.manifest.model_codes(),
            self.manifest.encoder.code(),
            &codec,
        ]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");
        let extension = if self.manifest.mkv { "mkv" } else { "mka" };
        format!("{}.{label}.{extension}", self.stem())
    }

    /// Where the track goes: the directory asked for, else beside the
    /// source — which is where a player looks for a film's `.mka`.
    pub fn output(&self) -> PathBuf {
        let dir = match &self.manifest.out_dir {
            Some(dir) => dir.clone(),
            None => self
                .manifest
                .source
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_default(),
        };
        dir.join(self.output_name())
    }

    /// Remove what this program put in the job's directory, and the
    /// directory once it is empty. Only its own entries, by name: whatever
    /// else is in there is not this program's to delete.
    pub fn clean(&self) -> Result<()> {
        for dir in [self.extract_dir(), self.separate_dir(), self.master_dir()] {
            if dir.is_dir() {
                fs::remove_dir_all(&dir).map_err(|e| Error::io(&dir, e))?;
            }
        }
        for file in [
            self.manifest_path(),
            self.manifest_path().with_extension("json.tmp"),
        ] {
            if file.is_file() {
                fs::remove_file(&file).map_err(|e| Error::io(&file, e))?;
            }
        }
        // Not `remove_dir_all`: a directory with anything left in it stays.
        let _ = fs::remove_dir(&self.dir);
        Ok(())
    }

    /// The manifest a previous run left, or `None`. An unreadable one is
    /// moved aside and reported rather than silently overwritten.
    pub fn previous(&self) -> (Option<Manifest>, Option<String>) {
        let path = self.manifest_path();
        if !path.is_file() {
            return (None, None);
        }
        match fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|text| serde_json::from_str::<Manifest>(&text).map_err(|e| e.to_string()))
        {
            Ok(manifest) => (Some(manifest), None),
            Err(detail) => {
                let aside = path.with_extension("json.unreadable");
                let _ = fs::rename(&path, &aside);
                (
                    None,
                    Some(format!(
                        "{}: {detail} — moved to {}, the job starts over",
                        path.display(),
                        aside.display()
                    )),
                )
            }
        }
    }

    /// Write the manifest: a temporary file and a rename, so an
    /// interruption never leaves half of one.
    pub fn save(&self) -> Result<()> {
        fs::create_dir_all(&self.dir).map_err(|e| Error::io(&self.dir, e))?;
        let path = self.manifest_path();
        let temporary = path.with_extension("json.tmp");
        let text = serde_json::to_string_pretty(&self.manifest)
            .map_err(|e| Error::msg(format!("writing the manifest: {e}")))?;
        fs::write(&temporary, text).map_err(|e| Error::io(&temporary, e))?;
        fs::rename(&temporary, &path).map_err(|e| Error::io(&path, e))
    }

    /// Note that a step is done and how long it took, and write the
    /// manifest.
    pub fn finished(&mut self, step: Step, seconds: f64) -> Result<()> {
        self.manifest.done = self.manifest.done.max(step.number());
        self.manifest
            .durations
            .insert(format!("{}", step.number()), seconds);
        self.save()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A finished job's manifest, for the tests of any module.
    pub(crate) fn manifest() -> Manifest {
        Manifest {
            tool: "omnifrog 0.1.0".into(),
            source: "/films/Some.Film.mkv".into(),
            source_bytes: 27_396_390_575,
            duration: 6281.417,
            original: TrackRef {
                index: 1,
                stream: 1,
                codec: "DTS".into(),
                channels: 8,
                language: Some("chi".into()),
                title: None,
                spatial: Some("DTS:X-7.1.4+5".into()),
                start: 2.0,
            },
            dub: TrackRef {
                index: 4,
                stream: 4,
                codec: "DTS".into(),
                channels: 6,
                language: Some("fre".into()),
                title: None,
                spatial: None,
                start: 2.0,
            },
            seconds: None,
            start: None,
            model_original: ModelRef {
                key: "uhav".into(),
                code: "uhav".into(),
                checkpoint: "bs_roformer_voc_hyperacev2.ckpt".into(),
            },
            model_dub: ModelRef {
                key: "ubb7".into(),
                code: "ubb7".into(),
                checkpoint: "big_beta7.ckpt".into(),
            },
            pairs: Pairs::default(),
            inference: Inference::default(),
            encoder: Encoder::Hz,
            hz_opts: "--presentations".into(),
            iamf_opts: String::new(),
            iamf_codec: IamfCodec::Flac,
            iamf_bitrate: DEFAULT_IAMF_BITRATE,
            iamf_bits: None,
            gain_db: 0.0,
            cancel_duplicates: true,
            mkv: false,
            done: 6,
            durations: BTreeMap::new(),
            extract: None,
            separate_original: None,
            separate_dub: None,
            master: None,
            encode: None,
            out_dir: None,
            output: None,
            extra: BTreeMap::new(),
        }
    }

    fn separated(lkfs: f64) -> SeparateState {
        SeparateState {
            signature: "sig".into(),
            elements: vec!["10".into()],
            lkfs,
            identical: None,
        }
    }

    #[test]
    fn the_levelling_gain_is_the_difference_of_the_two_loudnesses() {
        let mut m = manifest();
        m.separate_original = Some(separated(-27.0));
        m.separate_dub = Some(separated(-24.0));
        assert_eq!(m.voice_gain_db(), Some(-3.0));
        // What the run asked for on top.
        m.gain_db = 1.5;
        assert_eq!(m.voice_gain_db(), Some(-1.5));
    }

    #[test]
    fn an_unmeasured_loudness_leaves_the_levelling_alone() {
        let mut m = manifest();
        m.separate_original = Some(separated(0.0));
        m.separate_dub = Some(separated(-24.0));
        assert_eq!(m.voice_gain_db(), None);
        m.separate_dub = None;
        assert_eq!(m.voice_gain_db(), None);
    }

    #[test]
    fn a_finished_job_with_the_same_parameters_has_nothing_left_to_do() {
        assert_eq!(manifest().resume_at(Some(&manifest())), None);
    }

    #[test]
    fn switching_the_duplicates_redoes_the_master_set() {
        let mut off = manifest();
        off.cancel_duplicates = false;
        assert_eq!(manifest().resume_at(Some(&off)), Some(Step::Master));
        // A job from before the switch cancelled nothing.
        let json = serde_json::to_string(&manifest()).unwrap();
        let old: Manifest =
            serde_json::from_str(&json.replace("\"cancel_duplicates\":true,", "")).unwrap();
        assert!(!old.cancel_duplicates);
        assert_eq!(manifest().resume_at(Some(&old)), Some(Step::Master));
    }

    #[test]
    fn an_unfinished_job_carries_on_where_it_stopped() {
        let mut half = manifest();
        half.done = 1;
        assert_eq!(
            manifest().resume_at(Some(&half)),
            Some(Step::SeparateOriginal)
        );
    }

    #[test]
    fn a_first_run_starts_at_the_beginning() {
        assert_eq!(manifest().resume_at(None), Some(Step::Extract));
    }

    #[test]
    fn a_model_change_does_not_re_extract_the_film() {
        let previous = manifest();
        let mut current = manifest();
        current.model_dub.key = "ulin".into();
        current.model_dub.checkpoint = "unwa_large.ckpt".into();
        assert_eq!(current.resume_at(Some(&previous)), Some(Step::SeparateDub));
        let mut current = manifest();
        current.model_original.checkpoint = "other.ckpt".into();
        assert_eq!(
            current.resume_at(Some(&previous)),
            Some(Step::SeparateOriginal)
        );
    }

    #[test]
    fn another_track_or_another_file_re_extracts() {
        let previous = manifest();
        for change in [
            |m: &mut Manifest| m.dub.index = 3,
            |m: &mut Manifest| m.original.index = 2,
            |m: &mut Manifest| m.source = "/films/Other.mkv".into(),
            |m: &mut Manifest| m.source_bytes += 1,
            |m: &mut Manifest| m.seconds = Some(120.0),
        ] {
            let mut current = manifest();
            change(&mut current);
            assert_eq!(current.resume_at(Some(&previous)), Some(Step::Extract));
        }
    }

    #[test]
    fn the_pairs_send_back_only_the_separation_they_change() {
        let previous = manifest();
        let mut current = manifest();
        current.pairs.original = true;
        assert_eq!(
            current.resume_at(Some(&previous)),
            Some(Step::SeparateOriginal)
        );
        let mut current = manifest();
        current.pairs.dub = true;
        assert_eq!(current.resume_at(Some(&previous)), Some(Step::SeparateDub));
        let mut current = manifest();
        current.pairs = Pairs::ALL;
        let mut before = manifest();
        before.pairs = Pairs {
            centre_twice: false,
            ..Pairs::ALL
        };
        assert_eq!(current.resume_at(Some(&before)), Some(Step::SeparateDub));
    }

    #[test]
    fn a_manifest_from_before_the_pairs_ran_every_element_alone() {
        let mut value = serde_json::to_value(manifest()).unwrap();
        value.as_object_mut().unwrap().remove("pairs");
        let read: Manifest = serde_json::from_value(value).unwrap();
        assert_eq!(read.pairs, Pairs::default());
        assert_eq!(read.resume_at(Some(&manifest())), None);
    }

    #[test]
    fn the_encoder_and_the_container_are_the_last_two_steps() {
        let previous = manifest();
        let mut current = manifest();
        current.encoder = Encoder::Dee;
        assert_eq!(current.resume_at(Some(&previous)), Some(Step::Encode));
        let mut current = manifest();
        current.hz_opts = String::new();
        assert_eq!(current.resume_at(Some(&previous)), Some(Step::Encode));
        let mut current = manifest();
        current.mkv = true;
        assert_eq!(current.resume_at(Some(&previous)), Some(Step::Remux));
        let mut current = manifest();
        current.out_dir = Some("/elsewhere".into());
        assert_eq!(current.resume_at(Some(&previous)), Some(Step::Remux));
    }

    #[test]
    fn only_the_options_of_the_encoder_in_use_redo_the_encode() {
        let mut previous = manifest();
        previous.encoder = Encoder::Iamf;
        let mut current = previous.clone();
        current.iamf_opts = "--codec opus".into();
        assert_eq!(current.resume_at(Some(&previous)), Some(Step::Encode));
        // harlettizer's TrueHD options are nothing to an IAMF encode...
        let mut current = previous.clone();
        current.hz_opts = "--presentations --overlay-cost 0".into();
        assert_eq!(current.resume_at(Some(&previous)), None);
        // ...nor IAMF's to a TrueHD one, nor either to Dee.
        let mut current = manifest();
        current.iamf_opts = "--objects".into();
        assert_eq!(current.resume_at(Some(&manifest())), None);
        let mut dee = manifest();
        dee.encoder = Encoder::Dee;
        let mut current = dee.clone();
        current.hz_opts = String::new();
        current.iamf_opts = "--objects".into();
        assert_eq!(current.resume_at(Some(&dee)), None);
        // Going from TrueHD to IAMF is an encode again, the separation kept.
        let mut current = manifest();
        current.encoder = Encoder::Iamf;
        assert_eq!(current.resume_at(Some(&manifest())), Some(Step::Encode));
    }

    #[test]
    fn changing_the_iamf_mode_redoes_the_encode() {
        let mut previous = manifest();
        previous.encoder = Encoder::Iamf;
        previous.iamf_opts = "--codec opus".into();
        let mut current = previous.clone();
        current.encoder = Encoder::IamfBed;
        assert_eq!(current.resume_at(Some(&previous)), Some(Step::Encode));
        assert_eq!(previous.resume_at(Some(&current)), Some(Step::Encode));
        // Both modes are handed the same options, and those alone.
        assert_eq!(current.encoder_options(), "--codec opus");
        let mut bed = current.clone();
        bed.iamf_opts = "--codec flac".into();
        assert_eq!(bed.resume_at(Some(&current)), Some(Step::Encode));
        let mut bed = current.clone();
        bed.hz_opts = "--overlay-cost 0".into();
        assert_eq!(bed.resume_at(Some(&current)), None);
    }

    #[test]
    fn the_iamf_codec_is_an_encode_option() {
        let mut previous = manifest();
        previous.encoder = Encoder::Iamf;
        assert_eq!(previous.encoder_options(), "");
        let mut current = previous.clone();
        current.iamf_codec = IamfCodec::Opus;
        assert_eq!(current.encoder_options(), "--codec opus --bitrate 64");
        assert_eq!(current.resume_at(Some(&previous)), Some(Step::Encode));
        let mut lower = current.clone();
        lower.iamf_bitrate = 32;
        assert_eq!(lower.resume_at(Some(&current)), Some(Step::Encode));
        // The bitrate means nothing to FLAC, nor the codec to TrueHD.
        let mut flac = previous.clone();
        flac.iamf_bitrate = 32;
        assert_eq!(flac.resume_at(Some(&previous)), None);
        let mut truehd = manifest();
        truehd.iamf_codec = IamfCodec::Opus;
        assert_eq!(truehd.resume_at(Some(&manifest())), None);
    }

    #[test]
    fn a_manifest_from_before_the_iamf_codec_was_flac() {
        let mut value = serde_json::to_value(manifest()).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("iamf_codec");
        object.remove("iamf_bitrate");
        let read: Manifest = serde_json::from_value(value).unwrap();
        assert_eq!(read.iamf_codec, IamfCodec::Flac);
        assert_eq!(read.iamf_bitrate, DEFAULT_IAMF_BITRATE);
        assert_eq!(read.resume_at(Some(&manifest())), None);
    }

    #[test]
    fn a_start_point_is_an_extraction_again() {
        let previous = manifest();
        let mut current = manifest();
        current.start = Some(779.0);
        assert_eq!(current.resume_at(Some(&previous)), Some(Step::Extract));
        let mut later = current.clone();
        later.start = Some(800.0);
        assert_eq!(later.resume_at(Some(&current)), Some(Step::Extract));
        assert_eq!(current.resume_at(Some(&current.clone())), None);
        // A manifest from before it read from the beginning.
        let mut value = serde_json::to_value(manifest()).unwrap();
        value.as_object_mut().unwrap().remove("start");
        let read: Manifest = serde_json::from_value(value).unwrap();
        assert_eq!(read.start, None);
        assert_eq!(read.resume_at(Some(&manifest())), None);
    }

    #[test]
    fn both_tracks_begin_at_the_cut_of_a_run_from_a_start_point() {
        let mut m = manifest();
        m.original.start = 0.084;
        m.dub.start = 0.094;
        m.extract = Some(ExtractState {
            source_codec: "TrueHD".into(),
            elements: vec![1],
            objects: 0,
            dub_channels: vec!["L".into()],
            frames: 48_000,
            begin: None,
        });
        assert_eq!(m.original_begin(), 0.084);
        assert_eq!(m.dub_begin(), 0.094);
        m.extract.as_mut().unwrap().begin = Some(60.046);
        assert_eq!(m.original_begin(), 60.046);
        assert_eq!(m.dub_begin(), 60.046);
    }

    #[test]
    fn a_hand_chosen_flac_depth_redoes_the_encode_and_the_sources_does_not_show() {
        let mut previous = manifest();
        previous.encoder = Encoder::Iamf;
        let mut current = previous.clone();
        current.iamf_bits = Some(16);
        assert_eq!(current.encoder_options(), "--bits 16");
        assert_eq!(current.resume_at(Some(&previous)), Some(Step::Encode));
        // Opus has no depth: a depth chosen for FLAC is nothing to it.
        let mut opus = previous.clone();
        opus.iamf_codec = IamfCodec::Opus;
        let mut opus16 = opus.clone();
        opus16.iamf_bits = Some(16);
        assert_eq!(opus16.resume_at(Some(&opus)), None);
    }

    #[test]
    fn a_manifest_from_before_iamf_used_no_iamf_options() {
        let mut value = serde_json::to_value(manifest()).unwrap();
        value.as_object_mut().unwrap().remove("iamf_opts");
        let read: Manifest = serde_json::from_value(value).unwrap();
        assert_eq!(read.iamf_opts, "");
        assert_eq!(read.resume_at(Some(&manifest())), None);
    }

    #[test]
    fn cleaning_a_job_leaves_what_is_not_its_own() {
        let work = std::env::temp_dir().join(format!("omnifrog-clean-{}", std::process::id()));
        let _ = fs::remove_dir_all(&work);
        let config = Config {
            work_dir: Some(work.clone()),
            ..Config::default()
        };
        let job = Job::new(&config, manifest());
        fs::create_dir_all(job.separate_dir()).unwrap();
        fs::write(job.separate_dir().join("sons_10.wav"), b"x").unwrap();
        job.save().unwrap();
        fs::write(job.dir.join("notes.txt"), b"mine").unwrap();
        job.clean().unwrap();
        assert!(!job.separate_dir().exists());
        assert!(!job.manifest_path().exists());
        // Somebody else's file keeps the directory.
        assert!(job.dir.join("notes.txt").is_file());
        fs::remove_file(job.dir.join("notes.txt")).unwrap();
        job.clean().unwrap();
        assert!(!job.dir.exists());
        let _ = fs::remove_dir_all(&work);
    }

    #[test]
    fn a_change_late_in_the_run_does_not_skip_what_was_never_done() {
        // The job only got as far as the extraction; asking for another
        // encoder carries on at the separation, not at the encoding.
        let mut previous = manifest();
        previous.done = 1;
        let mut current = manifest();
        current.encoder = Encoder::Dee;
        assert_eq!(
            current.resume_at(Some(&previous)),
            Some(Step::SeparateOriginal)
        );
    }

    #[test]
    fn the_output_name_holds_one_dotted_segment_after_the_film() {
        let config = Config {
            work_dir: Some("/work".into()),
            ..Config::default()
        };
        let job = Job::new(&config, manifest());
        assert_eq!(job.dir, Path::new("/work/Some.Film.omnifrog"));
        assert_eq!(job.output_name(), "Some.Film.Omnifrog-uhav-ubb7-hz.mka");
        // Beside the source unless told otherwise.
        assert_eq!(
            job.output(),
            Path::new("/films/Some.Film.Omnifrog-uhav-ubb7-hz.mka")
        );
        assert_eq!(
            job.output_name().rsplit_once('.').unwrap().0,
            "Some.Film.Omnifrog-uhav-ubb7-hz"
        );
        // IAMF says so where the encoder goes, in an .mka as in an .mkv.
        let mut m = manifest();
        m.encoder = Encoder::Iamf;
        let job = Job::new(&config, m);
        assert_eq!(job.output_name(), "Some.Film.Omnifrog-uhav-ubb7-iamf.mka");
        let mut m = manifest();
        m.encoder = Encoder::Iamf;
        m.mkv = true;
        let job = Job::new(&config, m);
        assert_eq!(job.output_name(), "Some.Film.Omnifrog-uhav-ubb7-iamf.mkv");
        // With the voices on the bed, beside the overlay rather than over it.
        let mut m = manifest();
        m.encoder = Encoder::IamfBed;
        let job = Job::new(&config, m);
        assert_eq!(
            job.output_name(),
            "Some.Film.Omnifrog-uhav-ubb7-iamf-lit.mka"
        );
        assert_eq!(
            job.output_name().rsplit_once('.').unwrap().0,
            "Some.Film.Omnifrog-uhav-ubb7-iamf-lit"
        );
        // Opus is named for its bitrate, beside the FLAC stream; a TrueHD
        // track ignores an IAMF codec it was never handed.
        let mut m = manifest();
        m.encoder = Encoder::Iamf;
        m.iamf_codec = IamfCodec::Opus;
        m.iamf_bitrate = 48;
        let job = Job::new(&config, m);
        assert_eq!(
            job.output_name(),
            "Some.Film.Omnifrog-uhav-ubb7-iamf-opus48.mka"
        );
        let mut m = manifest();
        m.iamf_codec = IamfCodec::Opus;
        let job = Job::new(&config, m);
        assert_eq!(job.output_name(), "Some.Film.Omnifrog-uhav-ubb7-hz.mka");
    }

    /// Free options that name the codec take the place of the structured
    /// choice at the encode; the name follows what is really coded, or one
    /// export is written over another.
    #[test]
    fn the_output_is_named_for_the_codec_the_free_options_ask_for() {
        let config = Config::default();
        let name = |codec: IamfCodec, bitrate: u32, free: &str| {
            let mut m = manifest();
            m.encoder = Encoder::Iamf;
            m.iamf_codec = codec;
            m.iamf_bitrate = bitrate;
            m.iamf_opts = free.into();
            let name = Job::new(&config, m).output_name();
            // What comes after the encoder's code, without the extension.
            let label = name.rsplit_once('.').unwrap().0;
            label.rsplit_once("-iamf").unwrap().1.to_string()
        };
        // FLAC configured, Opus typed by hand: not the FLAC export's name.
        assert_eq!(name(IamfCodec::Flac, 64, ""), "");
        assert_eq!(
            name(IamfCodec::Flac, 64, "--codec opus --bitrate 48"),
            "-opus48"
        );
        assert_eq!(
            name(IamfCodec::Flac, 64, "--codec=opus --bitrate=48"),
            "-opus48"
        );
        // Opus without a bitrate reaching the encoder: the encoder's own.
        assert_eq!(name(IamfCodec::Flac, 64, "--codec opus"), "-opus96");
        assert_eq!(
            name(IamfCodec::Flac, 64, "--objects 4 --codec LPCM"),
            "-lpcm"
        );
        // The other way round: Opus configured, FLAC typed.
        assert_eq!(name(IamfCodec::Opus, 48, ""), "-opus48");
        assert_eq!(name(IamfCodec::Opus, 48, "--codec flac"), "");
        // A depth typed by hand pushes the structured choice aside whole:
        // the encoder is left with its default codec, FLAC.
        assert_eq!(name(IamfCodec::Opus, 48, "--bits 16"), "");
        // Quoted, an option is the same option: the name reads the words
        // the encoder is handed, not the text as typed.
        assert_eq!(
            name(IamfCodec::Flac, 64, "--codec opus --bitrate '48'"),
            "-opus48"
        );
        assert_eq!(
            name(IamfCodec::Flac, 64, "--codec opus --bitrate '64'"),
            "-opus64"
        );
        assert_eq!(
            name(IamfCodec::Flac, 64, r#""--codec" "opus" "--bitrate=32""#),
            "-opus32"
        );
        for free in ["--codec opus --bitrate '48'", r#""--codec" "opus""#] {
            let handed = crate::encode::split(free).unwrap().join(" ");
            assert!(crate::config::names_codec(free), "{free}");
            assert_eq!(
                name(IamfCodec::Flac, 64, free),
                name(IamfCodec::Flac, 64, &handed),
                "{free}"
            );
        }
        // Options that say nothing of the codec leave the choice alone.
        assert_eq!(name(IamfCodec::Opus, 48, "--objects 4"), "-opus48");
        // And the name agrees with the options handed to the encoder.
        for (codec, free) in [
            (IamfCodec::Flac, "--codec opus --bitrate 48"),
            (IamfCodec::Opus, "--bits 16"),
        ] {
            let options = codec.options(48, None, free);
            assert_eq!(options, free, "the structured options were dropped");
        }
    }

    #[test]
    fn a_key_a_later_version_wrote_survives_being_read_here() {
        let mut json = serde_json::to_value(manifest()).unwrap();
        json.as_object_mut()
            .unwrap()
            .insert("listened".into(), Value::Bool(true));
        let text = serde_json::to_string(&json).unwrap();
        let read: Manifest = serde_json::from_str(&text).unwrap();
        assert_eq!(read.extra.get("listened"), Some(&Value::Bool(true)));
        let back = serde_json::to_string(&read).unwrap();
        assert!(back.contains("\"listened\":true"), "{back}");
    }

    #[test]
    fn a_job_directory_is_ascii_whatever_the_film_is_called() {
        let config = Config {
            work_dir: Some("/work".into()),
            ..Config::default()
        };
        let mut m = manifest();
        m.source = "/films/Film.Accentué.mkv".into();
        let job = Job::new(&config, m);
        assert_eq!(job.dir, Path::new("/work/Film.Accentue.omnifrog"));
    }

    #[test]
    fn a_long_film_name_is_cut_with_a_hash_of_the_rest() {
        let long = "Some.Long.Film.Name.Extended.Cut.2160p.UHD.BLURAY.REMUX.HDR.HEVC.MULTI.VFI.TrueHD.Atmos-GROUP";
        let cut = short(long, WINDOWS_NAME);
        assert_eq!(cut.chars().count(), WINDOWS_NAME);
        assert!(
            cut.starts_with("Some.Long.Film.Name.Extended.Cut.2160p.~"),
            "{cut}"
        );
        // The same name gives the same directory, run after run...
        assert_eq!(cut, short(long, WINDOWS_NAME));
        // ...and two names that differ only past the cut give two.
        let other = long.replace("GROUP", "OTHER");
        assert_ne!(cut, short(&other, WINDOWS_NAME));
        // A short name is left alone.
        assert_eq!(short("Some.Film", WINDOWS_NAME), "Some.Film");
    }

    #[test]
    fn every_step_is_numbered_once() {
        for (i, step) in Step::ALL.iter().enumerate() {
            assert_eq!(step.number(), i as u32 + 1);
        }
    }
}
