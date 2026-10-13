//! The user's configuration: one YAML file, every key optional.
//!
//! Only what this machine decides lives here — where the work goes, which
//! Python runs the inference, where the tools are, which two models to use.
//! What a *programme* decides (its tracks, its range) is a command-line
//! argument, and what the output must carry is written into the track.
//!
//! Keys this version does not know are kept and written back, so a file
//! edited for a newer version survives an older one.

use crate::{Error, NAME, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Environment variable naming the configuration file, for a second profile
/// or a test run.
pub const ENV_CONFIG: &str = "OMNIFROG_CONFIG";

/// Environment variable naming the data directory (venv, MSST, models,
/// worker, work), for the same reasons.
pub const ENV_DATA: &str = "OMNIFROG_DATA";

/// The harlettizer options a fresh configuration passes: real 2.0/5.1/7.1
/// folds for a decoder that stops before the objects, as a delivered stream
/// carries and as Dee always writes.
pub const DEFAULT_HZ_OPTS: &str = "--presentations";

/// Kilobits a second for each channel of an Opus IAMF stream, harlettizer's
/// own default: a 7.1.4 bed at about 720 kbit/s.
pub const DEFAULT_IAMF_BITRATE: u32 = 64;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    /// Where the jobs' temporaries go: 30 to 45 GB per film. ASCII only when
    /// Dee encodes, which cannot open a path with an accent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub work_dir: Option<PathBuf>,
    /// Checkpoints, their configs and their replacement modules.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models_dir: Option<PathBuf>,
    /// The Python of the venv that has torch and the MSST dependencies;
    /// `setup` fills it in.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub python: Option<PathBuf>,
    /// The MSST checkout (`inference.py` at its root); `setup` fills it in.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub msst_dir: Option<PathBuf>,
    pub tools: Tools,
    pub models: Selection,
    /// How the elements are handed to the models, which are stereo: two by
    /// two rather than each one alone on both sides.
    pub pairs: Pairs,
    /// What runs the models.
    pub inference: Inference,
    pub encoder: Encoder,
    /// Options passed verbatim to `harlettizer encode`, after the ones this
    /// program sets itself (`--overlay`, `--fold-depth`, `--progress`, `--out`).
    pub hz_opts: String,
    /// Options passed verbatim to `harlettizer iamf` (`encoder: iamf` or
    /// `iamf-lit`), after the ones this program sets itself (`--objects`,
    /// `--overlay` or `--voices-to-bed`, `--out`, `--progress`,
    /// `--mono-prefix`, the codec below). Empty is harlettizer's own
    /// choice for everything else.
    pub iamf_opts: String,
    /// What the IAMF stream is coded with: FLAC, lossless, or Opus, lossy.
    pub iamf_codec: IamfCodec,
    /// Opus only: kilobits a second for each channel; a coupled pair takes
    /// twice it and the LFE a quarter.
    pub iamf_bitrate: u32,
    /// FLAC only: 16 or 24 bits a sample; absent, the original's own depth
    /// decides — 16 for a source mastered at 16 or fewer, 24 otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iamf_bits: Option<u32>,
    /// Cancel the duplicates between the original and the dub: the dub's
    /// voices moved onto the original's clock, and the original's music and
    /// effects they still carry taken out (see [`crate::duplicates`]).
    pub cancel_duplicates: bool,
    /// Keep the job directory after a successful run.
    pub keep_work: bool,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_yaml_ng::Value>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            work_dir: None,
            models_dir: None,
            python: None,
            msst_dir: None,
            tools: Tools::default(),
            models: Selection::default(),
            pairs: Pairs::default(),
            inference: Inference::default(),
            encoder: Encoder::default(),
            hz_opts: DEFAULT_HZ_OPTS.to_string(),
            iamf_opts: String::new(),
            iamf_codec: IamfCodec::default(),
            iamf_bitrate: DEFAULT_IAMF_BITRATE,
            iamf_bits: None,
            cancel_duplicates: true,
            keep_work: false,
            extra: BTreeMap::new(),
        }
    }
}

/// Explicit paths to the external programs; an absent one is looked for
/// beside the executable, then on `PATH`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Tools {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ffmpeg: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ffprobe: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mkvmerge: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub harletty: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub harlettizer: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dee: Option<PathBuf>,
    /// Linux only: the Wine that runs a `dee.exe`; empty = the system's.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wine: Option<PathBuf>,
}

/// Which separations hand the models two elements at once.
///
/// Our elements are mono and the models are stereo: alone, an element is
/// written on both sides and the model sees a perfectly centred mix. Two
/// elements side by side is half the passes, and where they carry the same
/// voice — a panner spreads a line over neighbouring objects — the model
/// sees what it was trained on, a voice panned in a stereo mix, and leaves
/// less of it in the music and effects. Measured: a whole film came out in
/// 1 h 38 instead of 4 h 38 with every switch on.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(default)]
pub struct Pairs {
    /// The original's elements two by two: spatial neighbours, starting at
    /// the most central one, which the dialogue is usually around.
    pub original: bool,
    /// The dub's channels left with right (L+R, Lss+Rss, Lrs+Rrs), the
    /// centre alone.
    pub dub: bool,
    /// With `dub`: the centre separated twice, beside L and beside R, and
    /// the two averaged. L and R come out of the same two passes, so it
    /// costs no pass.
    pub centre_twice: bool,
}

impl Pairs {
    /// Every switch on.
    pub const ALL: Pairs = Pairs {
        original: true,
        dub: true,
        centre_twice: true,
    };

    /// What the switches really do: the centre is only separated twice when
    /// the dub's channels are paired at all.
    pub fn effective(self) -> Self {
        Self {
            centre_twice: self.centre_twice && self.dub,
            ..self
        }
    }

    /// Parse the command line's list: `original`, `dub`, `centre`, `all`
    /// or `none`, separated by commas.
    pub fn parse(list: &str) -> std::result::Result<Self, String> {
        let mut pairs = Pairs::default();
        for word in list.split(',').map(str::trim).filter(|w| !w.is_empty()) {
            match word {
                "original" | "vo" => pairs.original = true,
                "dub" | "vf" => pairs.dub = true,
                "centre" | "center" => {
                    pairs.dub = true;
                    pairs.centre_twice = true;
                }
                "all" => pairs = Pairs::ALL,
                "none" => pairs = Pairs::default(),
                other => {
                    return Err(format!("{other}: one of original, dub, centre, all, none"));
                }
            }
        }
        Ok(pairs)
    }

    /// What a manifest, a tag and a progress line call it.
    pub fn describe(self) -> String {
        let self_ = self.effective();
        let mut words = Vec::new();
        if self_.original {
            words.push("original");
        }
        if self_.dub {
            words.push(if self_.centre_twice {
                "dub+centre"
            } else {
                "dub"
            });
        }
        if words.is_empty() {
            "none".to_string()
        } else {
            words.join(",")
        }
    }
}

/// pymss's block when the configuration does not say: five minutes, 2.7 GiB
/// reserved by torch for Fullness V3 or Kim FT2 at 25 %, against 2.9 GiB for
/// ten, in the same time and an output 61 dB below the signal of the one at
/// ten (`bench-vram`, 7900 XTX, equal blocks): see the README.
pub const DEFAULT_BLOCK_SECONDS: u32 = 300;

/// What runs the models: MSST's own inference, or pymss on the very same
/// checkpoints and configs.
///
/// pymss keeps a whole element on the device and lets the windows overlap
/// by a chosen share. Measured on five minutes of a film (7900 XTX,
/// warm): HyperACE v2 Voc 25.9 s under MSST, 16.3 s at MSST's step, 10.7 s
/// at 25 %, 8.3 s at 5 %; Big Beta 7 22.0, 12.2, 8.4, 6.5 s. At MSST's step
/// the output is MSST's within 50 dB; at 25 % or 5 % it moves (about
/// -30 dB), yet against a ground truth — clean lines mixed into an
/// element with no speech — voice SI-SDR and M&E SDR stay within 0.04 dB.
/// A whole 1 h 57 element took 277 s, the device at most 72 % full.
///
/// The worker falls back to MSST, and says so, when pymss is not in the
/// venv, cannot load the model (unwa's Large Inst) or the model asks for
/// shifted passes. Like a catalogue overlap, the engine is not part of a
/// separation's signature: stems already made are kept.
///
/// pymss at 25 % is the default: three times MSST's speed for the same
/// score, and Kim FT2 Bleedless, published at MSST's slowest step (eight
/// passes), would otherwise be the longest part of a run.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct Inference {
    pub engine: Engine,
    /// With pymss: the percentage of a window its neighbour computes again;
    /// 0 = MSST's step (the catalogue's `overlap`, else the config's).
    pub overlap: u32,
    /// With pymss: the seconds of an element handed over at once. pymss
    /// keeps a whole block on the device, so this is what its memory grows
    /// with; outside the fades between blocks the output does not depend on
    /// it, to rounding. It is never taken shorter than four of the model's
    /// windows, and an element is cut into equal blocks of at most that plus
    /// one window (the fade), the last one no longer than the others.
    pub block_seconds: u32,
    /// Windows computed at once, replacing the model's published
    /// `inference.batch_size`; 0 = the model's. Memory grows with it, and a
    /// large card goes faster.
    pub batch_size: u32,
}

impl Default for Inference {
    fn default() -> Self {
        Self {
            engine: Engine::Pymss,
            overlap: 25,
            block_seconds: DEFAULT_BLOCK_SECONDS,
            batch_size: 0,
        }
    }
}

impl Inference {
    /// What ran before there was a choice: MSST.
    pub fn msst() -> Self {
        Self {
            engine: Engine::Msst,
            ..Self::default()
        }
    }

    /// What a manifest, a tag and a progress line call it.
    pub fn describe(self) -> String {
        let mut text = match self.engine {
            Engine::Msst => "msst".to_string(),
            Engine::Pymss if self.overlap == 0 => "pymss, MSST's step".to_string(),
            Engine::Pymss => format!("pymss, {} % overlap", self.overlap.min(90)),
        };
        if self.engine == Engine::Pymss && self.block_seconds != DEFAULT_BLOCK_SECONDS {
            text.push_str(&format!(", blocks of {} s", self.block_seconds));
        }
        if self.batch_size > 0 {
            text.push_str(&format!(", batches of {}", self.batch_size));
        }
        text
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    Msst,
    #[default]
    Pymss,
}

/// The two models a run uses, by catalogue key: Gabox's Fullness V3 for the
/// original, which keeps the most of the music and effects, and Kim FT2
/// Bleedless for the dub, which lets the least of them into the voices.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Selection {
    /// Removes the dialogue from the original's elements.
    pub vo: String,
    /// Extracts the dialogue from the dubbed track's channels.
    pub vf: String,
}

impl Default for Selection {
    fn default() -> Self {
        Self {
            vo: "gfu3".to_string(),
            vf: "kf2b".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Encoder {
    /// harlettizer, the open encoder: the original's elements kept as they
    /// are and the dubbed voices panned onto them.
    #[default]
    Hz,
    /// Dee, an encoder the user supplies, which re-clusters the whole scene.
    Dee,
    /// harlettizer's IAMF, in Matroska: the open format browsers and
    /// televisions decode. The master's elements carried as IAMF v2.0
    /// objects, as they are, and the voices panned onto them exactly as the
    /// TrueHD overlay pans them: the same objects, the same projection.
    Iamf,
    /// harlettizer's IAMF, in Matroska, the voices put on the fixed bed
    /// rather than panned onto the objects; the master's other objects stay
    /// independent.
    #[serde(rename = "iamf-lit", alias = "iamf-bed")]
    IamfBed,
}

impl Encoder {
    /// The short code in the output name and the manifest.
    pub fn code(self) -> &'static str {
        match self {
            Self::Hz => "hz",
            Self::Dee => "dee",
            Self::Iamf => "iamf",
            Self::IamfBed => "iamf-lit",
        }
    }

    /// What the encoder writes, as a person names it.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Hz => "TrueHD Atmos by harlettizer",
            Self::Dee => "TrueHD Atmos by Dee",
            Self::Iamf => "IAMF by harlettizer, the voices panned onto the objects",
            Self::IamfBed => "IAMF by harlettizer, the voices on the bed",
        }
    }

    /// Whether it writes IAMF, in either mode: harlettizer's `iamf`, handed
    /// `iamf_opts`.
    pub fn is_iamf(self) -> bool {
        matches!(self, Self::Iamf | Self::IamfBed)
    }
}

/// The codec of an IAMF stream.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum IamfCodec {
    /// Lossless, about half the size of the samples: harlettizer's own
    /// default, and what passes no option at all.
    #[default]
    Flac,
    /// Lossy, at a chosen bitrate: what a browser or a streaming service is
    /// sent. Needs a harlettizer built with its `opus` feature.
    Opus,
}

impl IamfCodec {
    pub fn code(self) -> &'static str {
        match self {
            Self::Flac => "flac",
            Self::Opus => "opus",
        }
    }

    /// The options of `harlettizer iamf` that ask for this codec, put before
    /// the free ones: `--codec opus --bitrate` for Opus, `--bits` for FLAC at
    /// a depth chosen by hand, nothing for FLAC at the original's depth —
    /// that one is measured at the encode. None either when the free options
    /// already name the codec, the bitrate or the depth: an option given
    /// twice is an error, and what was typed by hand is the more deliberate
    /// choice.
    pub fn options(self, bitrate: u32, bits: Option<u32>, free: &str) -> String {
        let structured = match self {
            _ if names_codec(free) => String::new(),
            Self::Opus => format!("--codec opus --bitrate {bitrate}"),
            Self::Flac => bits.map(|b| format!("--bits {b}")).unwrap_or_default(),
        };
        [structured.as_str(), free.trim()]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// What the output's name says of the codec: nothing for a lossless
    /// FLAC, `opus<bitrate>` for Opus, the codec's own name otherwise.
    ///
    /// From what the stream is *really* coded with. Free options that name
    /// the codec, the bitrate or the depth take the place of the structured
    /// choice in [`IamfCodec::options`]; the name has to follow them, or an
    /// Opus export asked for by hand over a FLAC configuration would be
    /// named as the FLAC one and written over it. What the free options do
    /// not say is then the encoder's own default — FLAC, and 96 kbit/s a
    /// channel for Opus — not the structured value they pushed aside.
    pub fn name_part(self, bitrate: u32, free: &str) -> String {
        let (codec, bitrate) = if names_codec(free) {
            (
                free_value(free, "--codec")
                    .unwrap_or_else(|| "flac".to_string())
                    .to_ascii_lowercase(),
                free_value(free, "--bitrate")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(ENCODER_OPUS_BITRATE),
            )
        } else {
            (self.code().to_string(), bitrate)
        };
        match codec.as_str() {
            "flac" => String::new(),
            "opus" => format!("opus{bitrate}"),
            _ => codec,
        }
    }
}

/// What `harlettizer iamf` codes Opus at when no `--bitrate` reaches it, in
/// kbit/s a channel.
const ENCODER_OPUS_BITRATE: u32 = 96;

/// The value free options give a flag, written `--flag value` or
/// `--flag=value`; the last one, as the encoder would read it.
fn free_value(free: &str, flag: &str) -> Option<String> {
    let mut words = words(free).into_iter();
    let mut found = None;
    while let Some(word) = words.next() {
        if word == flag {
            found = words.next().or(found);
        } else if let Some(value) = word.strip_prefix(flag).and_then(|w| w.strip_prefix('=')) {
            found = Some(value.to_string());
        }
    }
    found
}

/// Free options as the words the encoder is handed: split the very way the
/// encode splits them, quotes taken off. Read any other way, `--bitrate '48'`
/// would be a bitrate for the encoder and not for whoever names the output.
/// Options the encode cannot split (an unbalanced quote) stop it before
/// anything is named; they are read on whitespace here, for want of better.
fn words(free: &str) -> Vec<String> {
    crate::encode::split(free)
        .unwrap_or_else(|_| free.split_whitespace().map(str::to_string).collect())
}

/// Whether free `harlettizer iamf` options say what to code with: the
/// codec, its bitrate or its depth.
pub fn names_codec(free: &str) -> bool {
    words(free).iter().any(|word| {
        ["--codec", "--bitrate", "--bits"]
            .iter()
            .any(|flag| word == *flag || word.starts_with(&format!("{flag}=")))
    })
}

/// The depths a lossless IAMF stream is offered at: what FLAC decoders read.
pub const IAMF_BITS: [u32; 2] = [16, 24];

/// The depth a lossless IAMF stream is coded at, from the original's: 16 for
/// a master at 16 bits or fewer, 24 for anything finer — FLAC drops the
/// bits no sample sets, so 24 costs a 20-bit master nothing.
pub fn iamf_bits_for(original: u32) -> u32 {
    if original <= 16 { 16 } else { 24 }
}

impl Config {
    /// Where the configuration file is: `$OMNIFROG_CONFIG`, else the
    /// platform's configuration directory.
    pub fn path() -> PathBuf {
        if let Some(p) = std::env::var_os(ENV_CONFIG) {
            return PathBuf::from(p);
        }
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(NAME)
            .join("config.yaml")
    }

    /// The user's own additions to the model catalogue, beside the config.
    pub fn user_catalogue_path() -> PathBuf {
        Self::path().with_file_name("models.yaml")
    }

    /// Load the file, or the defaults when there is none.
    pub fn load() -> Result<Self> {
        let path = Self::path();
        if !path.is_file() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(&path).map_err(|e| Error::io(&path, e))?;
        serde_yaml_ng::from_str(&text).map_err(|e| Error::Yaml {
            path: path.clone(),
            source: e,
        })
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let text = serde_yaml_ng::to_string(self).map_err(|e| Error::Yaml {
            path: path.clone(),
            source: e,
        })?;
        fs::write(&path, text).map_err(|e| Error::io(&path, e))
    }

    /// Where this program keeps what it installs (venv, MSST, models, worker)
    /// when the configuration does not say otherwise.
    pub fn data_dir() -> PathBuf {
        if let Some(p) = std::env::var_os(ENV_DATA) {
            return PathBuf::from(p);
        }
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(NAME)
    }

    pub fn work_dir(&self) -> PathBuf {
        self.work_dir
            .as_deref()
            .map(expand)
            .unwrap_or_else(|| Self::data_dir().join("work"))
    }

    pub fn models_dir(&self) -> PathBuf {
        self.models_dir
            .as_deref()
            .map(expand)
            .unwrap_or_else(|| Self::data_dir().join("models"))
    }

    pub fn msst_dir(&self) -> PathBuf {
        self.msst_dir
            .as_deref()
            .map(expand)
            .unwrap_or_else(|| Self::data_dir().join("msst"))
    }

    /// The venv `setup` creates.
    pub fn venv_dir() -> PathBuf {
        Self::data_dir().join("venv")
    }

    /// The interpreter inside a venv directory.
    pub fn venv_python(venv: &Path) -> PathBuf {
        if cfg!(windows) {
            venv.join("Scripts").join("python.exe")
        } else {
            venv.join("bin").join("python")
        }
    }

    /// The Python that runs the inference: the configured one, else the venv's.
    pub fn python(&self) -> PathBuf {
        self.python
            .as_deref()
            .map(expand)
            .unwrap_or_else(|| Self::venv_python(&Self::venv_dir()))
    }
}

/// `~` and `~/…` to the home directory; anything else unchanged.
pub fn expand(path: &Path) -> PathBuf {
    let Some(text) = path.to_str() else {
        return path.to_path_buf();
    };
    if text == "~" {
        return dirs::home_dir().unwrap_or_else(|| path.to_path_buf());
    }
    if let Some(rest) = text.strip_prefix("~/").or_else(|| text.strip_prefix("~\\"))
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest);
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_pairs_are_named_on_the_command_line() {
        assert_eq!(Pairs::parse("none").unwrap(), Pairs::default());
        assert_eq!(Pairs::parse("all").unwrap(), Pairs::ALL);
        assert_eq!(
            Pairs::parse("original, dub").unwrap(),
            Pairs {
                original: true,
                dub: true,
                centre_twice: false
            }
        );
        // The centre twice needs the dub's pairs, and brings them.
        assert_eq!(
            Pairs::parse("centre").unwrap(),
            Pairs {
                original: false,
                dub: true,
                centre_twice: true
            }
        );
        assert!(Pairs::parse("stereo").is_err());
        assert_eq!(Pairs::ALL.describe(), "original,dub+centre");
        assert_eq!(Pairs::default().describe(), "none");
        let lone = Pairs {
            centre_twice: true,
            ..Pairs::default()
        };
        assert_eq!(lone.effective(), Pairs::default());
        assert_eq!(lone.describe(), "none");
    }

    #[test]
    fn a_configuration_without_pairs_runs_every_element_alone() {
        let config: Config = serde_yaml_ng::from_str("keep_work: true\n").unwrap();
        assert_eq!(config.pairs, Pairs::default());
        let config: Config =
            serde_yaml_ng::from_str("pairs:\n  original: true\n  dub: true\n").unwrap();
        assert!(config.pairs.original && config.pairs.dub && !config.pairs.centre_twice);
    }

    use super::*;

    #[test]
    fn unknown_keys_survive_a_round_trip() {
        let text = "encoder: dee\nfuture_key: 42\nmodels:\n  vo: ulin\n";
        let config: Config = serde_yaml_ng::from_str(text).unwrap();
        assert_eq!(config.encoder, Encoder::Dee);
        assert_eq!(config.models.vo, "ulin");
        assert_eq!(config.models.vf, "kf2b");
        assert_eq!(config.hz_opts, DEFAULT_HZ_OPTS);
        assert_eq!(config.iamf_opts, "");
        let back = serde_yaml_ng::to_string(&config).unwrap();
        assert!(back.contains("future_key: 42"), "{back}");
    }

    #[test]
    fn the_engine_is_pymss_until_told_otherwise() {
        let config: Config = serde_yaml_ng::from_str("keep_work: true\n").unwrap();
        assert_eq!(config.inference.engine, Engine::Pymss);
        assert_eq!(config.inference.overlap, 25);
        assert_eq!(config.inference.describe(), "pymss, 25 % overlap");
        let config: Config = serde_yaml_ng::from_str("inference:\n  engine: msst\n").unwrap();
        assert_eq!(config.inference.describe(), "msst");
        let config: Config =
            serde_yaml_ng::from_str("inference: { block_seconds: 120, batch_size: 1 }\n").unwrap();
        assert_eq!(config.inference.engine, Engine::Pymss);
        assert_eq!(
            config.inference.describe(),
            "pymss, 25 % overlap, blocks of 120 s, batches of 1"
        );
        let config: Config =
            serde_yaml_ng::from_str("inference: { engine: pymss, overlap: 0 }\n").unwrap();
        assert_eq!(config.inference.describe(), "pymss, MSST's step");
    }

    #[test]
    fn duplicates_are_cancelled_until_told_otherwise() {
        let config: Config = serde_yaml_ng::from_str("keep_work: true\n").unwrap();
        assert!(config.cancel_duplicates);
        let config: Config = serde_yaml_ng::from_str("cancel_duplicates: false\n").unwrap();
        assert!(!config.cancel_duplicates);
    }

    #[test]
    fn iamf_is_a_third_encoder_with_options_of_its_own() {
        let config: Config =
            serde_yaml_ng::from_str("encoder: iamf\niamf_opts: --codec opus --bitrate 48\n")
                .unwrap();
        assert_eq!(config.encoder, Encoder::Iamf);
        assert_eq!(config.encoder.code(), "iamf");
        assert_eq!(config.iamf_opts, "--codec opus --bitrate 48");
        assert_eq!(config.hz_opts, DEFAULT_HZ_OPTS);
        let back = serde_yaml_ng::to_string(&config).unwrap();
        assert!(back.contains("encoder: iamf"), "{back}");
        assert!(config.encoder.is_iamf());
    }

    #[test]
    fn iamf_with_the_voices_on_the_bed_is_an_encoder_of_its_own() {
        let config: Config =
            serde_yaml_ng::from_str("encoder: iamf-lit\niamf_opts: --codec opus\n").unwrap();
        assert_eq!(config.encoder, Encoder::IamfBed);
        assert_eq!(config.encoder.code(), "iamf-lit");
        assert!(config.encoder.is_iamf());
        assert_eq!(config.iamf_opts, "--codec opus");
        let back = serde_yaml_ng::to_string(&config).unwrap();
        assert!(back.contains("encoder: iamf-lit"), "{back}");
        let again: Config = serde_yaml_ng::from_str(&back).unwrap();
        assert_eq!(again, config);
        assert!(!Encoder::Hz.is_iamf() && !Encoder::Dee.is_iamf());
        // The name it had before still reads.
        let old: Config = serde_yaml_ng::from_str("encoder: iamf-bed\n").unwrap();
        assert_eq!(old.encoder, Encoder::IamfBed);
    }

    #[test]
    fn the_iamf_codec_is_flac_unless_opus_is_asked_for() {
        let config: Config = serde_yaml_ng::from_str("encoder: iamf\n").unwrap();
        assert_eq!(config.iamf_codec, IamfCodec::Flac);
        assert_eq!(config.iamf_bitrate, DEFAULT_IAMF_BITRATE);
        let config: Config =
            serde_yaml_ng::from_str("iamf_codec: opus\niamf_bitrate: 48\n").unwrap();
        assert_eq!(config.iamf_codec, IamfCodec::Opus);
        assert_eq!(config.iamf_bitrate, 48);
        let back = serde_yaml_ng::to_string(&config).unwrap();
        assert!(back.contains("iamf_codec: opus"), "{back}");
    }

    #[test]
    fn the_codec_goes_before_the_free_options_unless_they_name_it() {
        assert_eq!(IamfCodec::Flac.options(64, None, ""), "");
        assert_eq!(IamfCodec::Flac.options(64, Some(16), ""), "--bits 16");
        assert_eq!(
            IamfCodec::Flac.options(64, Some(24), "--overlay-cost 0"),
            "--bits 24 --overlay-cost 0"
        );
        assert_eq!(
            IamfCodec::Opus.options(48, Some(16), ""),
            "--codec opus --bitrate 48"
        );
        assert_eq!(
            IamfCodec::Opus.options(48, None, "--overlay-cost 0"),
            "--codec opus --bitrate 48 --overlay-cost 0"
        );
        // What was typed by hand wins, rather than a flag given twice.
        assert_eq!(
            IamfCodec::Opus.options(48, None, "--codec lpcm"),
            "--codec lpcm"
        );
        assert_eq!(
            IamfCodec::Opus.options(48, None, "--bitrate 96"),
            "--bitrate 96"
        );
        assert_eq!(
            IamfCodec::Opus.options(48, None, "--bitrate=96"),
            "--bitrate=96"
        );
        assert_eq!(
            IamfCodec::Flac.options(48, Some(24), "--bits 16"),
            "--bits 16"
        );
        assert!(!names_codec("--bitsy 3 --overlay-cost 0"));
    }

    #[test]
    fn a_lossless_stream_keeps_the_originals_depth_at_16_or_24() {
        assert_eq!(iamf_bits_for(16), 16);
        assert_eq!(iamf_bits_for(14), 16);
        assert_eq!(iamf_bits_for(17), 24);
        assert_eq!(iamf_bits_for(20), 24);
        assert_eq!(iamf_bits_for(24), 24);
        let config: Config = serde_yaml_ng::from_str("iamf_bits: 16\n").unwrap();
        assert_eq!(config.iamf_bits, Some(16));
        assert_eq!(Config::default().iamf_bits, None);
        let back = serde_yaml_ng::to_string(&Config::default()).unwrap();
        assert!(!back.contains("iamf_bits"), "{back}");
    }

    #[test]
    fn an_empty_file_is_the_defaults() {
        let config: Config = serde_yaml_ng::from_str("{}").unwrap();
        assert_eq!(config, Config::default());
    }
}
