//! The test set: its zones, what the ear said of every output on each, and
//! what was measured.
//!
//! One file, `bench.json`, at the root of the bench directory. A zone's audio
//! is beside it, one directory per zone (see [`crate::layout`]).

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// The set's file, in the bench directory.
pub const FILE: &str = "bench.json";
/// The bench directory, in omnifrog's work directory.
pub const DIR: &str = "_bench";
/// Every file of the bench is mono at this rate, as a job's are.
pub const RATE: u32 = 48_000;
/// A zone is loaded whole to be measured, so it is kept short.
pub const MAX_SECONDS: u32 = 120;

/// The kinds of passage offered when a zone is added. A zone's kind is free
/// text all the same.
pub const KINDS: [&str; 8] = [
    "Dialogue over music",
    "Dialogue over effects",
    "Shout / loud voice",
    "Whisper",
    "Crowd / several voices",
    "Singing",
    "Clean dialogue",
    "Other",
];

/// Which of the two inputs of a zone: the element of the original track, or
/// the channel of the dub.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Vo,
    Vf,
}

impl Side {
    pub const BOTH: [Side; 2] = [Side::Vo, Side::Vf];

    pub fn name(self) -> &'static str {
        match self {
            Side::Vo => "vo",
            Side::Vf => "vf",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Side::Vo => "VO",
            Side::Vf => "VF",
        }
    }

    /// The stem a model is judged on, on this side: what the run keeps of
    /// it. Of the original's element that is everything but the voice; of
    /// the dub's channel, the voice.
    pub fn listened(self) -> Stem {
        match self {
            Side::Vo => Stem::RestVo,
            Side::Vf => Stem::VoiceVf,
        }
    }

    pub fn voice(self) -> Stem {
        match self {
            Side::Vo => Stem::VoiceVo,
            Side::Vf => Stem::VoiceVf,
        }
    }

    pub fn rest(self) -> Stem {
        match self {
            Side::Vo => Stem::RestVo,
            Side::Vf => Stem::RestVf,
        }
    }
}

/// One of the four stems a model makes of a zone: the voice and the rest of
/// each side. (The fifth file of an output, the mix, is made from them.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stem {
    VoiceVo,
    RestVo,
    VoiceVf,
    RestVf,
}

impl Stem {
    pub const ALL: [Stem; 4] = [Stem::VoiceVo, Stem::RestVo, Stem::VoiceVf, Stem::RestVf];

    pub fn name(self) -> &'static str {
        match self {
            Stem::VoiceVo => "voice_vo",
            Stem::RestVo => "rest_vo",
            Stem::VoiceVf => "voice_vf",
            Stem::RestVf => "rest_vf",
        }
    }

    pub fn side(self) -> Side {
        match self {
            Stem::VoiceVo | Stem::RestVo => Side::Vo,
            Stem::VoiceVf | Stem::RestVf => Side::Vf,
        }
    }

    pub fn is_voice(self) -> bool {
        matches!(self, Stem::VoiceVo | Stem::VoiceVf)
    }

    /// The other stem of the same side: what leaks into this one.
    pub fn other(self) -> Stem {
        match self {
            Stem::VoiceVo => Stem::RestVo,
            Stem::RestVo => Stem::VoiceVo,
            Stem::VoiceVf => Stem::RestVf,
            Stem::RestVf => Stem::VoiceVf,
        }
    }
}

/// Why an output was rejected, judged on the stem listened to on that side.
/// It says which remedy to try: a leak wants more removed, a lack more kept,
/// artifacts another family of models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    /// Some of the other signal is left: voice in the rest, or rest in the voice.
    Leak,
    /// The wanted signal is eaten into: thinned or holed.
    Lack,
    /// The sound is damaged: metallic, watery, chopped, pumping.
    Artifacts,
}

impl Reason {
    pub const ALL: [Reason; 3] = [Reason::Leak, Reason::Lack, Reason::Artifacts];

    pub fn label(self) -> &'static str {
        match self {
            Reason::Leak => "leak",
            Reason::Lack => "lack",
            Reason::Artifacts => "artifacts",
        }
    }
}

/// What the ear said of an output on one side of a zone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Neutral,
    /// One of the outputs the others are measured against.
    Reference,
    /// Judged bad. A memory only: it is still measured.
    Rejected(Option<Reason>),
}

/// The ratings of one side of a zone.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Rating {
    /// The outputs rated best. Several make a consensus: the median of
    /// them, sample by sample. None: nothing is measured on this side.
    pub refs: Vec<String>,
    pub rejects: Vec<String>,
    /// A reject without a reason predates the reasons.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub reasons: BTreeMap<String, Reason>,
}

impl Rating {
    pub fn status(&self, key: &str) -> Status {
        if self.refs.iter().any(|k| k == key) {
            Status::Reference
        } else if self.rejects.iter().any(|k| k == key) {
            Status::Rejected(self.reasons.get(key).copied())
        } else {
            Status::Neutral
        }
    }
}

/// One step of the map of the dub around a zone.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Step {
    /// Centre of the window, in frames from the start of the zone (negative:
    /// before it).
    pub centre: i64,
    /// Smoothed offset of the dub, in frames; positive: the dub comes later.
    pub offset: i32,
}

/// What a zone needs of its film to cancel the original's music and effects
/// out of the dub's stems: a zone is a few seconds long, and the map of the
/// dub is measured around it, on the film.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DuplicatesContext {
    /// How far the dubbed track leads the original in the source, in ms.
    pub dub_lead_ms: f64,
    pub steps: Vec<Step>,
    /// Where the music and effects of the other elements of the original
    /// came from, for the record.
    pub others: String,
}

/// A zone: a range of a film, one element of its original and one channel of
/// its dub.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Zone {
    /// The name of its directory: `<film>_o<object>_c<channel>_<start in s>[_n]`.
    pub id: String,
    pub film: String,
    /// Path of the film it was cut from, for the record.
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lang_vo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lang_vf: Option<String>,
    pub object: u32,
    pub channel: String,
    /// Where the zone starts in the film, in frames at [`RATE`].
    pub start: u64,
    /// Where it ends; none: at the end of what was extracted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<u64>,
    pub kind: String,
    pub vo: Rating,
    pub vf: Rating,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duplicates: Option<DuplicatesContext>,
    /// When it was added, RFC 3339.
    pub date: String,
}

impl Zone {
    pub fn rating(&self, side: Side) -> &Rating {
        match side {
            Side::Vo => &self.vo,
            Side::Vf => &self.vf,
        }
    }

    pub fn rating_mut(&mut self, side: Side) -> &mut Rating {
        match side {
            Side::Vo => &mut self.vo,
            Side::Vf => &mut self.vf,
        }
    }

    /// The zone's range in the film's own time: `6:25 – 6:50`.
    pub fn range(&self) -> String {
        let end = match self.end {
            Some(end) => clock(end),
            None => "end".to_string(),
        };
        format!("{} – {end}", clock(self.start))
    }

    pub fn label(&self) -> String {
        format!(
            "{} · object {} / channel {} · {}",
            self.film,
            self.object,
            self.channel,
            self.range()
        )
    }
}

/// Frames at [`RATE`] as `m:ss` or `h:mm:ss`.
pub fn clock(frames: u64) -> String {
    let seconds = frames / RATE as u64;
    let (h, m, s) = (seconds / 3600, seconds / 60 % 60, seconds % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

/// A time as one types it — `83`, `1:23`, `1:02:03`, with decimals if one
/// likes — in frames at [`RATE`].
pub fn parse_clock(text: &str) -> Option<u64> {
    let mut seconds = 0.0f64;
    let parts: Vec<&str> = text.trim().split(':').collect();
    if parts.len() > 3 {
        return None;
    }
    for part in parts {
        let value: f64 = part.trim().replace(',', ".").parse().ok()?;
        if !value.is_finite() || value < 0.0 {
            return None;
        }
        seconds = seconds * 60.0 + value;
    }
    Some((seconds * RATE as f64).round() as u64)
}

/// A stem measured against the reference. Everything is relative to it:
/// close to the reference, not perfect. A value that could not be measured
/// (a silent reference, a stem that is the only reference) is none.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Metric {
    /// This output is one of the references of its side. Alone, it is not
    /// measured against itself; one of several, it is measured against the
    /// consensus like any other.
    pub reference: bool,
    /// dB, higher is closer to the reference; blind to gain.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub si_sdr: Option<f64>,
    /// dB, negative is good: energy of the reference's *other* stem found
    /// in this output, against the useful part.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residue: Option<f64>,
    /// dB, negative is good: the part of the reference the output lacks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loss: Option<f64>,
    /// dB, negative is good: what is neither the target nor the other stem.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<f64>,
    /// Gap to the reference by bands, in dB, averaged over the frames: the
    /// sum of its two parts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gap: Option<f64>,
    /// The part where the output is louder than the reference: a leak.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub excess: Option<f64>,
    /// The part where it is quieter: a lack.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lack: Option<f64>,
    /// Frames the output was found late on the reference by, and shifted
    /// back before measuring.
    pub lag: i32,
}

/// What one output measured on one zone.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Measure {
    /// Time the outputs took to compute, model loading included; 0: unknown.
    pub seconds: f64,
    /// When it was measured, RFC 3339.
    pub date: String,
    pub stems: BTreeMap<Stem, Metric>,
}

impl Measure {
    /// Whether anything was measured. A measure with no stem is what is
    /// left when the references it was against have changed: the time its
    /// outputs took is still true, and kept.
    pub fn measured(&self) -> bool {
        !self.stems.is_empty()
    }
}

/// Which stem of the first model the second one of a cascade works again.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CascadeMode {
    /// The rest, against a leak.
    #[default]
    Rest,
    /// The voice, against a lack.
    Voice,
}

/// Two models one after the other: the first separates, the second works
/// one of its stems again, and what it finds there changes side — nothing
/// is lost, it is moved.
///
/// - `rest`: the rest is the second model's rest of the first's rest; the
///   voice the second finds in it joins the first's voice. Against a leak.
/// - `voice`: the voice is the second's voice of the first's voice; the
///   rest it finds there goes back to the first's rest. Against a lack.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Cascade {
    pub first: String,
    pub second: String,
    pub mode: CascadeMode,
    /// Marked as worth keeping. A note for whoever reads the set: nothing
    /// in the bench acts on it.
    pub kept: bool,
}

impl Cascade {
    /// The key its outputs go by.
    pub fn key(&self) -> String {
        crate::keys::cascade(&self.first, &self.second, self.mode)
    }
}

/// A legacy set's fitted score for one side — the SI-SDR less `a` times the
/// excess and `b` times the lack. It was never adopted and the bench does
/// not compute it; an imported set keeps the numbers for the record.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Fit {
    pub a: f64,
    pub b: f64,
    pub a_fit: f64,
    pub b_fit: f64,
    pub adopted: bool,
    pub agreement: f64,
    pub agreement_fit: f64,
    pub agreement_si_sdr: f64,
    pub agreement_held_out: f64,
    pub pairs: u32,
    pub zones: u32,
    pub date: String,
}

/// Which variants a batch computes from a model's outputs, on the zones
/// that have what it takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Variants {
    /// The duplicates cancelled, the voice moved by the tracks' offset.
    pub cancel: bool,
    /// The same, the voice moved step by step along the map of the dub.
    pub cancel_step: bool,
}

impl Default for Variants {
    fn default() -> Self {
        Self {
            cancel: true,
            cancel_step: false,
        }
    }
}

impl Variants {
    /// The suffixes of the variants taken.
    pub fn suffixes(self) -> Vec<&'static str> {
        [
            (self.cancel, crate::keys::CANCEL),
            (self.cancel_step, crate::keys::CANCEL_STEP),
        ]
        .into_iter()
        .filter_map(|(taken, suffix)| taken.then_some(suffix))
        .collect()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Set {
    pub zones: Vec<Zone>,
    /// `[zone id][key]`: what that output measured on that zone.
    pub measures: BTreeMap<String, BTreeMap<String, Measure>>,
    /// `[key]`: taken in the next batch or not. Absent is taken.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub checked: BTreeMap<String, bool>,
    pub variants: Variants,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub cascades: Vec<Cascade>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub imported_fits: BTreeMap<Side, Fit>,
}

impl Set {
    pub fn path(dir: &Path) -> PathBuf {
        dir.join(FILE)
    }

    /// The set of the bench directory, or an empty one when there is no file
    /// yet. A file that does not read is an error, never an empty set: the
    /// next save would write over it.
    pub fn load(dir: &Path) -> Result<Self> {
        let path = Self::path(dir);
        if !path.is_file() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(&path).map_err(|e| Error::io(&path, e))?;
        serde_json::from_str(&text).map_err(|e| Error::msg(format!("{}: {e}", path.display())))
    }

    /// Write the set, whole or not at all: to a file beside it, then renamed.
    pub fn save(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
        let path = Self::path(dir);
        let text = serde_json::to_string_pretty(self)
            .map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
        let part = path.with_extension("json.part");
        fs::write(&part, text).map_err(|e| Error::io(&part, e))?;
        fs::rename(&part, &path).map_err(|e| Error::io(&path, e))
    }

    pub fn zone(&self, id: &str) -> Option<&Zone> {
        self.zones.iter().find(|z| z.id == id)
    }

    pub fn measure(&self, zone: &str, key: &str) -> Option<&Measure> {
        self.measures.get(zone)?.get(key)
    }

    /// Take a zone out of the set, with its files: its inputs, every output,
    /// its references, its measures. There is no way back. The film it was
    /// cut from, if it is in the bench, is not touched.
    pub fn remove_zone(&mut self, dir: &Path, id: &str) -> Result<()> {
        let Some(place) = self.zones.iter().position(|z| z.id == id) else {
            return Err(Error::msg(format!("no zone `{id}` in the set")));
        };
        // A zone's directory is its id, one name under the bench directory:
        // anything else is not removed.
        let one_name = Path::new(id).components().count() == 1
            && !matches!(id, "" | "." | "..")
            && !id.contains(['/', '\\']);
        if !one_name {
            return Err(Error::msg(format!("`{id}` is not a zone's directory")));
        }
        let path = dir.join(id);
        match fs::remove_dir_all(&path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(Error::io(&path, e)),
            _ => {}
        }
        self.zones.remove(place);
        self.measures.remove(id);
        self.save(dir)
    }

    /// Add a cascade to the set, unless it is there already. Whether it
    /// was added.
    pub fn add_cascade(&mut self, cascade: Cascade) -> bool {
        let key = cascade.key();
        if self.cascades.iter().any(|c| c.key() == key) {
            return false;
        }
        self.cascades.push(cascade);
        true
    }

    /// Take a cascade out of the set, by its key. Its outputs stay where
    /// they are; they are only no longer computed. Whether it was there.
    pub fn remove_cascade(&mut self, key: &str) -> bool {
        let before = self.cascades.len();
        self.cascades.retain(|c| c.key() != key);
        self.cascades.len() != before
    }

    /// Whether the next batch takes this key.
    pub fn is_checked(&self, key: &str) -> bool {
        self.checked.get(key).copied().unwrap_or(true)
    }

    /// Every key the set knows of: rated or measured on some zone.
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = Vec::new();
        let mut add = |key: &String| {
            if !keys.contains(key) {
                keys.push(key.clone());
            }
        };
        for zone in &self.zones {
            for side in Side::BOTH {
                let rating = zone.rating(side);
                rating.refs.iter().chain(&rating.rejects).for_each(&mut add);
            }
        }
        for measures in self.measures.values() {
            measures.keys().for_each(&mut add);
        }
        keys
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Set {
        let mut zone = Zone {
            id: "Film_o12_cC_723".into(),
            film: "Film".into(),
            object: 12,
            channel: "C".into(),
            start: 723 * RATE as u64,
            end: Some(735 * RATE as u64),
            kind: KINDS[0].into(),
            ..Zone::default()
        };
        zone.vo.refs.push("ulpi".into());
        zone.vf.rejects.push("bd2".into());
        zone.vf.reasons.insert("bd2".into(), Reason::Lack);
        let mut set = Set::default();
        let mut measure = Measure::default();
        measure.stems.insert(
            Stem::RestVo,
            Metric {
                reference: true,
                ..Metric::default()
            },
        );
        measure.stems.insert(
            Stem::VoiceVf,
            Metric {
                si_sdr: Some(12.5),
                lag: -3,
                ..Metric::default()
            },
        );
        set.measures
            .entry(zone.id.clone())
            .or_default()
            .insert("ulpi".into(), measure);
        set.zones.push(zone);
        set.checked.insert("bd2".into(), false);
        set
    }

    #[test]
    fn a_set_comes_back_as_it_was_saved() {
        let dir = std::env::temp_dir().join(format!("omnifrog-bench-set-{}", std::process::id()));
        let set = sample();
        set.save(&dir).unwrap();
        assert_eq!(Set::load(&dir).unwrap(), set);
        assert!(!dir.join("bench.json.part").exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn no_file_is_an_empty_set_and_a_broken_one_an_error() {
        let dir =
            std::env::temp_dir().join(format!("omnifrog-bench-broken-{}", std::process::id()));
        assert_eq!(Set::load(&dir).unwrap(), Set::default());
        fs::create_dir_all(&dir).unwrap();
        fs::write(Set::path(&dir), "{ not json").unwrap();
        assert!(Set::load(&dir).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_zone_is_removed_with_its_files_and_its_measures() {
        let dir =
            std::env::temp_dir().join(format!("omnifrog-bench-remove-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut set = sample();
        let id = set.zones[0].id.clone();
        let file = dir.join(&id).join("outputs/ulpi/voice_vo.wav");
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(&file, "").unwrap();
        // A neighbour that must stay.
        let other = dir.join("films/Some.Film/film.json");
        fs::create_dir_all(other.parent().unwrap()).unwrap();
        fs::write(&other, "{}").unwrap();
        set.save(&dir).unwrap();

        assert!(set.remove_zone(&dir, "nobody").is_err());
        // An id that would reach outside a zone's own directory is refused,
        // even if the set holds it.
        let mut odd = set.clone();
        odd.zones[0].id = "films/Some.Film".into();
        assert!(odd.remove_zone(&dir, "films/Some.Film").is_err());
        assert!(other.is_file());

        set.remove_zone(&dir, &id).unwrap();
        assert!(!dir.join(&id).exists());
        assert!(other.is_file());
        let saved = Set::load(&dir).unwrap();
        assert!(saved.zones.is_empty() && saved.measures.is_empty());
        // Ratings of other zones and the checks are not its to take.
        assert!(!saved.is_checked("bd2"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_status_is_read_from_the_lists() {
        let set = sample();
        let zone = &set.zones[0];
        assert_eq!(zone.vo.status("ulpi"), Status::Reference);
        assert_eq!(zone.vf.status("bd2"), Status::Rejected(Some(Reason::Lack)));
        assert_eq!(zone.vf.status("ulpi"), Status::Neutral);
        assert!(set.is_checked("ulpi"));
        assert!(!set.is_checked("bd2"));
        assert_eq!(set.keys(), ["ulpi", "bd2"]);
    }

    #[test]
    fn a_range_is_told_in_the_film_s_time() {
        let set = sample();
        assert_eq!(set.zones[0].range(), "12:03 – 12:15");
        assert_eq!(clock(3723 * RATE as u64), "1:02:03");
        let open = Zone {
            start: 0,
            end: None,
            ..Zone::default()
        };
        assert_eq!(open.range(), "0:00 – end");
    }

    #[test]
    fn a_time_is_read_as_it_is_typed() {
        assert_eq!(parse_clock("83"), Some(83 * RATE as u64));
        assert_eq!(parse_clock(" 1:23 "), Some(83 * RATE as u64));
        assert_eq!(parse_clock("1:02:03"), Some(3723 * RATE as u64));
        assert_eq!(
            parse_clock("12:02,5"),
            Some(722 * RATE as u64 + RATE as u64 / 2)
        );
        assert_eq!(parse_clock(""), None);
        assert_eq!(parse_clock("1:2:3:4"), None);
        assert_eq!(parse_clock("-3"), None);
        assert_eq!(parse_clock("abc"), None);
    }

    #[test]
    fn the_stems_of_a_side_are_each_other_s_other() {
        for stem in Stem::ALL {
            assert_eq!(stem.other().other(), stem);
            assert_eq!(stem.other().side(), stem.side());
            assert_ne!(stem.other().is_voice(), stem.is_voice());
        }
        assert_eq!(Side::Vo.listened(), Stem::RestVo);
        assert_eq!(Side::Vf.listened(), Stem::VoiceVf);
    }
}
