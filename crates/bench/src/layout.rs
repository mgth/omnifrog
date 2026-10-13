//! Where a zone's audio is.
//!
//! ```text
//! _bench/
//!   bench.json
//!   <zone>/
//!     1.extract/vo_<object>.wav  vf_<channel>.wav  [others_vo.wav]
//!     outputs/<key>/voice_vo.wav rest_vo.wav voice_vf.wav rest_vf.wav mix.wav  loudness.json
//!     ref_vo.voice.wav ref_vo.rest.wav ref_vf.voice.wav ref_vf.rest.wav
//! ```
//!
//! A zone is a job in small: its two inputs where a job extracts its
//! elements, then one directory per output with the five files a model makes
//! of them, and the references — the consensus of the outputs rated best.

use crate::set::{Side, Stem, Zone};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const EXTRACT_DIR: &str = "1.extract";
pub const OUTPUTS_DIR: &str = "outputs";
/// The sum of the music and effects of the other elements of the original,
/// written with the zone's duplicates context.
pub const OTHERS_FILE: &str = "others_vo.wav";
/// The fifth file of an output: the rest of the original with the voice of
/// the dub, as a run would mix them.
pub const MIX_FILE: &str = "mix.wav";
pub const LOUDNESS_FILE: &str = "loudness.json";
/// What ends the name of an output's directory while it is being made: it
/// takes its key only once every file is in it.
pub const PART: &str = ".part";

pub fn zone_dir(dir: &Path, zone: &Zone) -> PathBuf {
    dir.join(&zone.id)
}

pub fn extract_dir(dir: &Path, zone: &Zone) -> PathBuf {
    zone_dir(dir, zone).join(EXTRACT_DIR)
}

/// The zone's input on one side, named as a job names its elements.
pub fn extract(dir: &Path, zone: &Zone, side: Side) -> PathBuf {
    let name = match side {
        Side::Vo => format!("vo_{}.wav", zone.object),
        Side::Vf => format!("vf_{}.wav", zone.channel),
    };
    extract_dir(dir, zone).join(name)
}

pub fn others(dir: &Path, zone: &Zone) -> PathBuf {
    extract_dir(dir, zone).join(OTHERS_FILE)
}

pub fn outputs_dir(dir: &Path, zone: &Zone) -> PathBuf {
    zone_dir(dir, zone).join(OUTPUTS_DIR)
}

pub fn output_dir(dir: &Path, zone: &Zone, key: &str) -> PathBuf {
    outputs_dir(dir, zone).join(key)
}

pub fn stem_file(stem: Stem) -> String {
    format!("{}.wav", stem.name())
}

pub fn output(dir: &Path, zone: &Zone, key: &str, stem: Stem) -> PathBuf {
    output_dir(dir, zone, key).join(stem_file(stem))
}

pub fn mix(dir: &Path, zone: &Zone, key: &str) -> PathBuf {
    output_dir(dir, zone, key).join(MIX_FILE)
}

pub fn reference_file(stem: Stem) -> String {
    let part = if stem.is_voice() { "voice" } else { "rest" };
    format!("ref_{}.{part}.wav", stem.side().name())
}

pub fn reference(dir: &Path, zone: &Zone, stem: Stem) -> PathBuf {
    zone_dir(dir, zone).join(reference_file(stem))
}

/// Whether the output has both stems of a side: what it takes to be
/// measured there, or to be a reference.
pub fn side_present(dir: &Path, zone: &Zone, key: &str, side: Side) -> bool {
    output(dir, zone, key, side.voice()).is_file() && output(dir, zone, key, side.rest()).is_file()
}

pub fn outputs_present(dir: &Path, zone: &Zone, key: &str) -> bool {
    Side::BOTH
        .iter()
        .all(|&side| side_present(dir, zone, key, side))
}

pub fn reference_present(dir: &Path, zone: &Zone, side: Side) -> bool {
    reference(dir, zone, side.voice()).is_file() && reference(dir, zone, side.rest()).is_file()
}

/// The keys that have a directory among the zone's outputs, sorted.
pub fn keys_on_disk(dir: &Path, zone: &Zone) -> Vec<String> {
    let Ok(entries) = fs::read_dir(outputs_dir(dir, zone)) else {
        return Vec::new();
    };
    let mut keys: Vec<String> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        // An output still being made, or left half-made by a run that was
        // killed, is no output.
        .filter(|name| !name.ends_with(PART))
        .collect();
    keys.sort();
    keys
}

/// The loudness of an output's files, in LKFS, by file name without its
/// extension: `voice_vo`, `rest_vo`, `voice_vf`, `rest_vf`, `mix`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Loudness(pub BTreeMap<String, f64>);

/// The name the gain of an output's mix goes by in its loudness file.
pub const MIX_GAIN: &str = "mix_gain_db";

impl Loudness {
    /// The gain the dub's voice was given in the output's mix, as a ratio,
    /// when it was kept. An output of an imported legacy set has none.
    ///
    /// Kept because it cannot always be found again: a zone cut out of a
    /// longer range has the mix of that range, made at the gain the levels
    /// had over all of it, while its loudness is measured again on the cut.
    pub fn mix_gain(&self) -> Option<f64> {
        self.0.get(MIX_GAIN).map(|db| 10f64.powf(db / 20.0))
    }

    pub fn set_mix_gain(&mut self, ratio: f64) {
        self.0.insert(MIX_GAIN.to_string(), 20.0 * ratio.log10());
    }

    pub fn path(dir: &Path, zone: &Zone, key: &str) -> PathBuf {
        output_dir(dir, zone, key).join(LOUDNESS_FILE)
    }

    /// What was measured of the output; nothing when the file is absent or
    /// does not read — the loudness is a display, measured again at need.
    pub fn load(dir: &Path, zone: &Zone, key: &str) -> Self {
        fs::read_to_string(Self::path(dir, zone, key))
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, dir: &Path, zone: &Zone, key: &str) -> Result<()> {
        let path = Self::path(dir, zone, key);
        let text = serde_json::to_string_pretty(self)
            .map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
        fs::write(&path, text).map_err(|e| Error::io(&path, e))
    }

    pub fn of(&self, stem: Stem) -> Option<f64> {
        self.0.get(stem.name()).copied()
    }

    pub fn mix(&self) -> Option<f64> {
        self.0.get("mix").copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_files_of_a_zone_are_where_the_plan_says() {
        let zone = Zone {
            id: "Film_o12_cC_723".into(),
            object: 12,
            channel: "C".into(),
            ..Zone::default()
        };
        let dir = Path::new("bench");
        let at = |path: PathBuf| path.to_string_lossy().replace('\\', "/");
        assert_eq!(
            at(extract(dir, &zone, Side::Vo)),
            "bench/Film_o12_cC_723/1.extract/vo_12.wav"
        );
        assert_eq!(
            at(extract(dir, &zone, Side::Vf)),
            "bench/Film_o12_cC_723/1.extract/vf_C.wav"
        );
        assert_eq!(
            at(output(dir, &zone, "ulpi+cancel", Stem::RestVf)),
            "bench/Film_o12_cC_723/outputs/ulpi+cancel/rest_vf.wav"
        );
        assert_eq!(
            at(mix(dir, &zone, "ulpi")),
            "bench/Film_o12_cC_723/outputs/ulpi/mix.wav"
        );
        assert_eq!(
            at(reference(dir, &zone, Stem::VoiceVo)),
            "bench/Film_o12_cC_723/ref_vo.voice.wav"
        );
        assert_eq!(
            at(reference(dir, &zone, Stem::RestVf)),
            "bench/Film_o12_cC_723/ref_vf.rest.wav"
        );
    }
}
