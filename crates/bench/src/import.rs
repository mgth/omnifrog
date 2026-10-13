//! Bringing in a legacy test set.
//!
//! A legacy set is a `_JeuxEssai` directory: `jeu.json` and one directory
//! per zone, laid out as a job of the tool that made it. The import copies a
//! zone's audio into the bench's layout and translates the set — French
//! names to English ones, and the legacy model keys to the catalogue's.
//!
//! A model's key here is the legacy set's short code for it, so the
//! translation is a table of legacy keys against those codes. A model the
//! catalogue does not have keeps its legacy key and its outputs: they are
//! *foreign*, measured like the others and never computed again.
//!
//! Two things are left behind on purpose. The log-spectral distance was
//! replaced by the gap by bands, which agrees with the ear where it did not;
//! the fitted score was never adopted, and only its numbers are kept.

use crate::layout::{self, Loudness};
use crate::set::{
    Cascade, CascadeMode, DuplicatesContext, Fit, KINDS, Measure, Metric, Rating, Reason, Set,
    Side, Stem, Step, Zone,
};
use crate::{Error, Result, keys};
use serde::{Deserialize, Deserializer};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

/// The legacy set file, in its `_JeuxEssai`.
pub const FILE: &str = "jeu.json";

/// The legacy output key of every model of that catalogue, against its short
/// code — the key of the same model in omnifrog's catalogue.
pub const MODELS: [(&str, &str); 42] = [
    ("bandit_v2", "bd2"),
    ("bandit_v2_hq", "bd2q"),
    ("large_inst", "ulin"),
    ("leap_inst", "ulpi"),
    ("leap_xe_inst", "ulxi"),
    ("hyperace_v2_inst", "uhai"),
    ("gabox_fullness_v4", "gfu4"),
    ("melband_roformer_inst_v1e_plus", "kv1e"),
    ("melband_roformer_inst_v2", "kv2"),
    ("melband_roformer_instvoc_duality_v1", "kd1"),
    ("melband_roformer_instvox_duality_v2", "kd2"),
    ("mel_band_roformer_instrumental_becruily", "bci"),
    ("mel_band_roformer_kim_ft2_bleedless_unwa", "kf2b"),
    ("model_bs_roformer_ep_317_sdr_12.9755", "vpx"),
    ("mel_band_roformer_instrumental_instv8_gabox", "giv8"),
    ("mel_band_roformer_instrumental_bleedless_v3_gabox", "gbl3"),
    ("mel_band_roformer_instrumental_fullness_v3_gabox", "gfu3"),
    ("mel_band_roformer_instrumental_fv7z_gabox", "gfv7z"),
    ("mel_band_roformer_instrumental_fv8_gabox", "gfv8"),
    ("mel_band_roformer_instrumental_fvx_gabox", "gfvx"),
    ("gabox_fv9", "gfv9"),
    ("gabox_flowers_v10", "gfl10"),
    ("becruily_deux", "bcd"),
    ("bs_roformer_instrumental_resurrection_unwa", "rsi"),
    ("MDX23C-8KFFT-InstVoc_HQ_2", "mdx23"),
    ("leap_voc", "ulpv"),
    ("leap_xe_voc", "ulxv"),
    ("anvuew_ft1", "aft1"),
    ("hyperace_v2_voc", "uhav"),
    ("big_beta7", "ubb7"),
    ("gabox_voc_fv7", "gvf7"),
    ("bandit_v2_fra", "bd2f"),
    ("mel_band_roformer_vocals_fv4_gabox", "gvf4"),
    ("mel_band_roformer_vocals_fv5_gabox", "gvf5"),
    ("mel_band_roformer_vocals_fv6_gabox", "gvf6"),
    ("mel_band_roformer_vocals_v2_gabox", "gvv2"),
    ("bs_roformer_vocals_gabox", "gbv"),
    ("mel_band_roformer_vocals_becruily", "bcv"),
    ("mel_band_roformer_kim_ft3_unwa", "kf3"),
    ("mel_band_roformer_kim_ft2_unwa", "kf2"),
    ("bs_roformer_vocals_revive_v2_unwa", "rvv2"),
    ("UVR-MDX-NET-Voc_FT", "mdxv"),
];

/// The legacy variant suffixes against the bench's, longest first.
const VARIANTS: [(&str, &str); 2] = [("+annulp", keys::CANCEL_STEP), ("+annul", keys::CANCEL)];

/// The legacy kinds of passage, in the order of [`KINDS`].
const KINDS_FR: [&str; 8] = [
    "Dialogue sur musique",
    "Dialogue sur effets",
    "Cri / voix forte",
    "Chuchotement",
    "Foule / voix multiples",
    "Chant",
    "Dialogue propre",
    "Autre",
];

/// The bench's key for a legacy one. `known` says whether the catalogue
/// has a model by that key; one it does not have keeps the legacy key.
pub fn key(old: &str, known: &impl Fn(&str) -> bool) -> String {
    let (model, suffix) = VARIANTS
        .iter()
        .find_map(|(from, to)| old.strip_suffix(from).map(|model| (model, *to)))
        .unwrap_or((old, ""));
    let model = MODELS
        .iter()
        .find(|(theirs, code)| *theirs == model && known(code))
        .map_or(model, |(_, code)| code);
    format!("{model}{suffix}")
}

/// What an import did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Report {
    /// Zones brought in.
    pub zones: Vec<String>,
    /// Zones the bench already had, left as they were.
    pub skipped: Vec<String>,
    /// Every key that came with them.
    pub keys: BTreeSet<String>,
    /// Those of them the catalogue does not know.
    pub foreign: BTreeSet<String>,
    pub files: u64,
    pub bytes: u64,
    pub warnings: Vec<String>,
}

/// Import the legacy test set at `from` into the bench directory `dir`.
///
/// A zone the bench already has is left alone, its ratings and measures too:
/// an import adds, it never writes over what was rated here since. `note` is
/// told the zone being copied.
pub fn run(
    from: &Path,
    dir: &Path,
    known: impl Fn(&str) -> bool,
    mut note: impl FnMut(&str),
) -> Result<Report> {
    let file = from.join(FILE);
    let text = fs::read_to_string(&file).map_err(|e| Error::io(&file, e))?;
    let old: OldSet =
        serde_json::from_str(&text).map_err(|e| Error::msg(format!("{}: {e}", file.display())))?;
    let translate = |old: &str| key(old, &known);

    let mut set = Set::load(dir)?;
    let mut report = Report::default();
    for old_zone in &old.zones {
        if set.zone(&old_zone.id).is_some() {
            report.skipped.push(old_zone.id.clone());
            continue;
        }
        note(&old_zone.id);
        let zone = zone(old_zone, &translate, &mut report.warnings);
        copy_zone(
            &from.join(&old_zone.id),
            dir,
            &zone,
            &translate,
            &mut report,
        )?;

        let mut measures = BTreeMap::new();
        let times = old_times(&from.join(&old_zone.id));
        for (old_key, old_measure) in old.measures.get(&old_zone.id).into_iter().flatten() {
            let mut measure = measure(old_measure, &zone.id, old_key, &mut report.warnings);
            if measure.seconds <= 0.0 {
                measure.seconds = times.get(old_key).copied().unwrap_or(0.0);
            }
            measures.insert(translate(old_key), measure);
        }
        report.keys.extend(measures.keys().cloned());
        for side in Side::BOTH {
            let rating = zone.rating(side);
            report
                .keys
                .extend(rating.refs.iter().chain(&rating.rejects).cloned());
        }
        if !measures.is_empty() {
            set.measures.insert(zone.id.clone(), measures);
        }
        report.zones.push(zone.id.clone());
        set.zones.push(zone);
    }

    for (old_key, &checked) in &old.checked {
        set.checked.entry(translate(old_key)).or_insert(checked);
    }
    for cascade in &old.cascades {
        let cascade = Cascade {
            first: translate(&cascade.a),
            second: translate(&cascade.b),
            mode: if cascade.mode == "voix" {
                CascadeMode::Voice
            } else {
                CascadeMode::Rest
            },
            kept: cascade.kept,
        };
        if !set.cascades.contains(&cascade) {
            set.cascades.push(cascade);
        }
    }
    for (name, fit) in &old.fits {
        let side = match name.as_str() {
            "VO" => Side::Vo,
            "VF" => Side::Vf,
            other => {
                report
                    .warnings
                    .push(format!("a fitted score for `{other}`, which is no side"));
                continue;
            }
        };
        set.imported_fits.entry(side).or_insert_with(|| Fit {
            a: fit.a,
            b: fit.b,
            a_fit: fit.a_fit,
            b_fit: fit.b_fit,
            adopted: fit.adopted,
            agreement: fit.agreement,
            agreement_fit: fit.agreement_fit,
            agreement_si_sdr: fit.agreement_si_sdr,
            agreement_held_out: fit.agreement_held_out,
            pairs: fit.pairs,
            zones: fit.zones,
            date: fit.date.clone(),
        });
    }
    report.foreign = report
        .keys
        .iter()
        .filter(|key| !known(keys::base(key)))
        .cloned()
        .collect();
    set.save(dir)?;
    Ok(report)
}

fn zone(old: &OldZone, translate: &impl Fn(&str) -> String, warnings: &mut Vec<String>) -> Zone {
    let mut rating = |refs: &[String],
                      rejects: &[String],
                      reasons: &BTreeMap<String, String>,
                      side: Side| {
        let mut out = Rating {
            refs: refs.iter().map(|k| translate(k)).collect(),
            rejects: rejects.iter().map(|k| translate(k)).collect(),
            reasons: BTreeMap::new(),
        };
        for (old_key, reason) in reasons {
            let reason = match reason.as_str() {
                "fuite" => Reason::Leak,
                "manque" => Reason::Lack,
                "artefacts" => Reason::Artifacts,
                other => {
                    warnings.push(format!(
                        "{}: `{old_key}` rejected on {} for `{other}`, which is no reason; kept without one",
                        old.id,
                        side.label()
                    ));
                    continue;
                }
            };
            out.reasons.insert(translate(old_key), reason);
        }
        out
    };
    let vo = rating(&old.refs_vo, &old.rejects_vo, &old.reasons_vo, Side::Vo);
    let vf = rating(&old.refs_vf, &old.rejects_vf, &old.reasons_vf, Side::Vf);
    let kind = KINDS_FR
        .iter()
        .position(|fr| *fr == old.kind)
        .map_or(old.kind.clone(), |i| KINDS[i].to_string());
    Zone {
        id: old.id.clone(),
        film: old.film.clone(),
        source: old.source.clone(),
        lang_vo: old.lang_vo.clone(),
        lang_vf: old.lang_vf.clone(),
        object: old.object,
        channel: old.channel.clone(),
        start: old.start.max(0) as u64,
        end: (old.end >= 0).then_some(old.end as u64),
        kind,
        vo,
        vf,
        duplicates: old.duplicates.as_ref().map(|d| DuplicatesContext {
            dub_lead_ms: d.dub_lead_ms,
            steps: d
                .steps
                .iter()
                .map(|s| Step {
                    centre: s.centre,
                    offset: s.offset,
                })
                .collect(),
            others: d.others.clone(),
        }),
        date: old.date.clone(),
    }
}

fn measure(old: &OldMeasure, zone: &str, old_key: &str, warnings: &mut Vec<String>) -> Measure {
    let mut measure = Measure {
        seconds: old.seconds,
        date: old.date.clone(),
        stems: BTreeMap::new(),
    };
    for (name, m) in &old.stems {
        let stem = match name.as_str() {
            "voixVO" => Stem::VoiceVo,
            "sansVO" => Stem::RestVo,
            "voixVF" => Stem::VoiceVf,
            "sansVF" => Stem::RestVf,
            other => {
                warnings.push(format!(
                    "{zone}: `{old_key}` measured on `{other}`, which is no stem; left out"
                ));
                continue;
            }
        };
        measure.stems.insert(
            stem,
            Metric {
                reference: m.reference,
                si_sdr: m.si_sdr,
                residue: m.residue,
                loss: m.loss,
                artifacts: m.artifacts,
                gap: m.gap,
                excess: m.excess,
                lack: m.lack,
                lag: m.lag,
            },
        );
    }
    measure
}

/// The legacy names for the five files of an output, against the bench's.
fn output_files(zone: &Zone) -> [(String, String); 5] {
    let (o, c) = (zone.object, &zone.channel);
    [
        (
            format!("voix_vo.o{o}.0.wav"),
            layout::stem_file(Stem::VoiceVo),
        ),
        (
            format!("sansvoix_vo.o{o}.0.wav"),
            layout::stem_file(Stem::RestVo),
        ),
        (
            format!("voix_vf.c{c}.0.wav"),
            layout::stem_file(Stem::VoiceVf),
        ),
        (
            format!("sansvoix_vf.c{c}.0.wav"),
            layout::stem_file(Stem::RestVf),
        ),
        (format!("mix.o{o}.c{c}.0.wav"), layout::MIX_FILE.to_string()),
    ]
}

/// Copy a file if there is one to copy; whether there was.
fn copy(source: &Path, destination: &Path, report: &mut Report) -> Result<bool> {
    if !source.is_file() {
        return Ok(false);
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    report.bytes += fs::copy(source, destination).map_err(|e| Error::io(source, e))?;
    report.files += 1;
    Ok(true)
}

fn copy_zone(
    from: &Path,
    dir: &Path,
    zone: &Zone,
    translate: &impl Fn(&str) -> String,
    report: &mut Report,
) -> Result<()> {
    let extraction = from.join("1.Extraction");
    for side in Side::BOTH {
        let destination = layout::extract(dir, zone, side);
        let name = destination.file_name().unwrap_or_default();
        if !copy(&extraction.join(name), &destination, report)? {
            report.warnings.push(format!(
                "{}: no {} in its extraction; the zone cannot be computed again",
                zone.id,
                name.to_string_lossy()
            ));
        }
    }
    copy(
        &extraction.join("autres_vo.wav"),
        &layout::others(dir, zone),
        report,
    )?;
    for stem in Stem::ALL {
        let part = if stem.is_voice() { "voix" } else { "sansvoix" };
        let name = format!("ref_{}.{part}.wav", stem.side().name());
        copy(
            &from.join(name),
            &layout::reference(dir, zone, stem),
            report,
        )?;
    }

    let tests = from.join("Tests");
    let loudness = old_loudness(&tests);
    let files = output_files(zone);
    let mut old_keys: Vec<String> = fs::read_dir(&tests)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    old_keys.sort();
    for old_key in old_keys {
        let key = translate(&old_key);
        let mut copied = 0;
        for (theirs, ours) in &files {
            let destination = layout::output_dir(dir, zone, &key).join(ours);
            if copy(&tests.join(&old_key).join(theirs), &destination, report)? {
                copied += 1;
            }
        }
        if copied == 0 {
            continue;
        }
        report.keys.insert(key.clone());
        let mut lufs = Loudness::default();
        for (theirs, ours) in &files {
            let value = loudness.get(&old_key).and_then(|of| of.get(theirs));
            if let Some(value) = value.copied().flatten() {
                lufs.0
                    .insert(ours.trim_end_matches(".wav").to_string(), value);
            }
        }
        if !lufs.0.is_empty() {
            lufs.save(dir, zone, &key)?;
        }
    }
    Ok(())
}

/// The loudness a legacy set recorded of a zone's outputs: `[key][file name]`.
fn old_loudness(tests: &Path) -> BTreeMap<String, BTreeMap<String, Option<f64>>> {
    old_tests(tests)
        .map(|t| {
            t.outputs
                .into_iter()
                .map(|(key, files)| (key, files.into_iter().map(|(f, v)| (f, v.0)).collect()))
                .collect()
        })
        .unwrap_or_default()
}

/// The time a legacy set noted for the outputs of a zone: `[key]` in
/// seconds.
fn old_times(zone: &Path) -> BTreeMap<String, f64> {
    old_tests(&zone.join("Tests"))
        .map(|t| t.times)
        .unwrap_or_default()
}

fn old_tests(tests: &Path) -> Option<OldTests> {
    let text = fs::read_to_string(tests.join("tests.json")).ok()?;
    serde_json::from_str(&text).ok()
}

/// A number that may be written `"NaN"` or `"Infinity"`, as .NET writes the
/// ones JSON has no literal for; anything not finite is no value.
fn finite<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<f64>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Raw {
        Number(f64),
        #[allow(dead_code)]
        Text(String),
        Null,
    }
    Ok(match Raw::deserialize(deserializer)? {
        Raw::Number(value) if value.is_finite() => Some(value),
        _ => None,
    })
}

#[derive(Deserialize)]
struct Finite(#[serde(deserialize_with = "finite")] Option<f64>);

#[derive(Deserialize, Default)]
#[serde(default)]
struct OldSet {
    #[serde(rename = "Zones")]
    zones: Vec<OldZone>,
    #[serde(rename = "Mesures")]
    measures: BTreeMap<String, BTreeMap<String, OldMeasure>>,
    #[serde(rename = "Coches")]
    checked: BTreeMap<String, bool>,
    #[serde(rename = "Cascades")]
    cascades: Vec<OldCascade>,
    #[serde(rename = "Calages")]
    fits: BTreeMap<String, OldFit>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct OldZone {
    #[serde(rename = "Id")]
    id: String,
    #[serde(rename = "Film")]
    film: String,
    #[serde(rename = "Source")]
    source: String,
    #[serde(rename = "LangueVO")]
    lang_vo: Option<String>,
    #[serde(rename = "LangueVF")]
    lang_vf: Option<String>,
    #[serde(rename = "Objet")]
    object: u32,
    #[serde(rename = "Canal")]
    channel: String,
    #[serde(rename = "Debut")]
    start: i64,
    #[serde(rename = "Fin")]
    end: i64,
    #[serde(rename = "Type")]
    kind: String,
    #[serde(rename = "RefsVO")]
    refs_vo: Vec<String>,
    #[serde(rename = "RefsVF")]
    refs_vf: Vec<String>,
    #[serde(rename = "RejetsVO")]
    rejects_vo: Vec<String>,
    #[serde(rename = "RejetsVF")]
    rejects_vf: Vec<String>,
    #[serde(rename = "MotifsVO")]
    reasons_vo: BTreeMap<String, String>,
    #[serde(rename = "MotifsVF")]
    reasons_vf: BTreeMap<String, String>,
    #[serde(rename = "Doublons")]
    duplicates: Option<OldDuplicates>,
    #[serde(rename = "Date")]
    date: String,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct OldDuplicates {
    #[serde(rename = "CalageVfMs")]
    dub_lead_ms: f64,
    #[serde(rename = "Paliers")]
    steps: Vec<OldStep>,
    #[serde(rename = "Autres")]
    others: String,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct OldStep {
    #[serde(rename = "Centre")]
    centre: i64,
    #[serde(rename = "Decalage")]
    offset: i32,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct OldMeasure {
    #[serde(rename = "TempsS")]
    seconds: f64,
    #[serde(rename = "Date")]
    date: String,
    #[serde(rename = "Stems")]
    stems: BTreeMap<String, OldMetric>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct OldMetric {
    #[serde(rename = "Reference")]
    reference: bool,
    #[serde(rename = "SiSdr", deserialize_with = "finite")]
    si_sdr: Option<f64>,
    #[serde(rename = "Residu", deserialize_with = "finite")]
    residue: Option<f64>,
    #[serde(rename = "Perte", deserialize_with = "finite")]
    loss: Option<f64>,
    #[serde(rename = "Artefacts", deserialize_with = "finite")]
    artifacts: Option<f64>,
    #[serde(rename = "Ecart", deserialize_with = "finite")]
    gap: Option<f64>,
    #[serde(rename = "EcartTrop", deserialize_with = "finite")]
    excess: Option<f64>,
    #[serde(rename = "EcartManque", deserialize_with = "finite")]
    lack: Option<f64>,
    #[serde(rename = "Lag")]
    lag: i32,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct OldCascade {
    #[serde(rename = "A")]
    a: String,
    #[serde(rename = "B")]
    b: String,
    #[serde(rename = "Mode")]
    mode: String,
    #[serde(rename = "Retenue")]
    kept: bool,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct OldFit {
    #[serde(rename = "A")]
    a: f64,
    #[serde(rename = "B")]
    b: f64,
    #[serde(rename = "ACale")]
    a_fit: f64,
    #[serde(rename = "BCale")]
    b_fit: f64,
    #[serde(rename = "Adopte")]
    adopted: bool,
    #[serde(rename = "Accord")]
    agreement: f64,
    #[serde(rename = "AccordCale")]
    agreement_fit: f64,
    #[serde(rename = "AccordSiSdr")]
    agreement_si_sdr: f64,
    #[serde(rename = "AccordValide")]
    agreement_held_out: f64,
    #[serde(rename = "Paires")]
    pairs: u32,
    #[serde(rename = "Zones")]
    zones: u32,
    #[serde(rename = "Date")]
    date: String,
}

/// The legacy `tests.json` of a zone, of which the bench takes the loudness
/// and the times.
#[derive(Deserialize, Default)]
#[serde(default)]
struct OldTests {
    #[serde(rename = "Algos")]
    outputs: BTreeMap<String, BTreeMap<String, Finite>>,
    #[serde(rename = "Temps")]
    times: BTreeMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// The catalogue as far as these tests go: every code but the one
    /// the legacy set ran through another engine.
    fn known(key: &str) -> bool {
        key != "mdxv" && MODELS.iter().any(|(_, code)| *code == key)
    }

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("omnifrog-bench-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(path: &Path, content: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    const JEU: &str = r#"{
      "Zones": [{
        "Id": "Film_o12_cC_723", "Film": "Film", "Source": "/films/Film.mkv",
        "LangueVO": "en", "LangueVF": null, "Objet": 12, "Canal": "C",
        "Debut": 34704000, "Fin": -1, "Type": "Dialogue sur musique",
        "RefsVO": ["leap_inst"], "RefsVF": ["big_beta7", "UVR-MDX-NET-Voc_FT"],
        "RejetsVO": ["bandit_v2"], "RejetsVF": ["leap_inst+annul"],
        "MotifsVO": {"bandit_v2": "fuite"}, "MotifsVF": {"leap_inst+annul": "manque"},
        "Doublons": {"CalageVfMs": 78, "Paliers": [{"Centre": -240000, "Decalage": 3280}], "Autres": "10/10"},
        "Date": "2026-09-06T08:12:55.3375355+02:00"
      }],
      "Mesures": {"Film_o12_cC_723": {
        "leap_inst": {"TempsS": 0, "Date": "2026-09-28T13:53:34+02:00", "Stems": {
          "voixVO": {"Reference": true, "SiSdr": "NaN", "Residu": "NaN", "Perte": "NaN", "Artefacts": "NaN", "Lsd": "NaN", "Ecart": "NaN", "EcartTrop": "NaN", "EcartManque": "NaN", "Lag": 0},
          "voixVF": {"Reference": false, "SiSdr": 32.5, "Residu": -32.25, "Perte": -53, "Artefacts": -48, "Lsd": 6.5, "Ecart": 2.25, "EcartTrop": 2, "EcartManque": 0.25, "Lag": -3}}},
        "leap_inst+annul": {"TempsS": 0, "Date": "2026-09-28T13:53:35+02:00", "Stems": {
          "sansVF": {"Reference": false, "SiSdr": -3.5, "Lag": 0}}}
      }},
      "Coches": {"bandit_v2": false, "UVR-MDX-NET-Voc_FT": true},
      "Cascades": [{"A": "leap_inst", "B": "big_beta7", "Mode": "voix", "Retenue": true}],
      "Calages": {"VO": {"A": 0, "B": 0, "ACale": 2, "BCale": 0.5, "Adopte": false, "Accord": 0.92, "AccordCale": 0.93, "AccordSiSdr": 0.92, "AccordValide": 0.91, "Paires": 2875, "Zones": 8, "Date": "2026-09-25T21:07:47+02:00"}}
    }"#;

    const TESTS: &str = r#"{
      "Debut": "0:00", "Fin": "", "Objet": 12, "Canal": "C",
      "Algos": {
        "leap_inst": {"voix_vo.o12.0.wav": -35.5, "sansvoix_vo.o12.0.wav": -31.75, "voix_vf.cC.0.wav": "NaN", "mix.o12.cC.0.wav": -25.25},
        "leap_inst+annul": {"voix_vf.cC.0.wav": -30.5}
      },
      "Temps": {"leap_inst+annul": 0.125}
    }"#;

    fn legacy_set(from: &Path) {
        write(&from.join(FILE), JEU);
        let zone = from.join("Film_o12_cC_723");
        write(&zone.join("1.Extraction/vo_12.wav"), "vo");
        write(&zone.join("1.Extraction/vf_C.wav"), "vf");
        write(&zone.join("1.Extraction/autres_vo.wav"), "others");
        write(&zone.join("ref_vo.voix.wav"), "ref vo voice");
        write(&zone.join("ref_vo.sansvoix.wav"), "ref vo rest");
        write(&zone.join("Tests/tests.json"), TESTS);
        write(&zone.join("Tests/vo.o12.0.wav"), "the player's copy");
        for key in ["leap_inst", "leap_inst+annul", "UVR-MDX-NET-Voc_FT"] {
            let out = zone.join("Tests").join(key);
            write(&out.join("voix_vo.o12.0.wav"), "1");
            write(&out.join("sansvoix_vo.o12.0.wav"), "22");
            write(&out.join("voix_vf.cC.0.wav"), "333");
            write(&out.join("sansvoix_vf.cC.0.wav"), "4444");
            write(&out.join("mix.o12.cC.0.wav"), "55555");
        }
        fs::create_dir_all(zone.join("Tests/empty_model")).unwrap();
    }

    #[test]
    fn a_key_takes_the_code_the_catalogue_knows() {
        assert_eq!(key("leap_inst", &known), "ulpi");
        assert_eq!(key("leap_inst+annul", &known), "ulpi+cancel");
        assert_eq!(key("leap_inst+annulp", &known), "ulpi+cancel-step");
        assert_eq!(key("model_bs_roformer_ep_317_sdr_12.9755", &known), "vpx");
        // A model the catalogue does not have keeps its legacy key.
        assert_eq!(key("UVR-MDX-NET-Voc_FT", &known), "UVR-MDX-NET-Voc_FT");
        assert_eq!(
            key("UVR-MDX-NET-Voc_FT+annul", &known),
            "UVR-MDX-NET-Voc_FT+cancel"
        );
        assert_eq!(key("somebody_else", &known), "somebody_else");
    }

    #[test]
    fn the_table_names_every_model_once() {
        let theirs: BTreeSet<_> = MODELS.iter().map(|(theirs, _)| theirs).collect();
        let codes: BTreeSet<_> = MODELS.iter().map(|(_, code)| code).collect();
        assert_eq!(theirs.len(), MODELS.len());
        assert_eq!(codes.len(), MODELS.len());
    }

    #[test]
    fn a_zone_comes_in_with_its_ratings_measures_and_files() {
        let from = scratch("import-from");
        let dir = scratch("import-to");
        legacy_set(&from);
        let mut noted = Vec::new();
        let report = run(&from, &dir, known, |zone| noted.push(zone.to_string())).unwrap();

        assert_eq!(noted, ["Film_o12_cC_723"]);
        assert_eq!(report.zones, ["Film_o12_cC_723"]);
        assert!(report.skipped.is_empty());
        assert_eq!(
            report.keys.iter().collect::<Vec<_>>(),
            ["UVR-MDX-NET-Voc_FT", "bd2", "ubb7", "ulpi", "ulpi+cancel"]
        );
        assert_eq!(
            report.foreign.iter().collect::<Vec<_>>(),
            ["UVR-MDX-NET-Voc_FT"]
        );
        // Two extracts, the others, two references, three outputs of five.
        assert_eq!(report.files, 20);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);

        let set = Set::load(&dir).unwrap();
        let zone = &set.zones[0];
        assert_eq!(zone.kind, "Dialogue over music");
        assert_eq!((zone.start, zone.end), (34_704_000, None));
        assert_eq!(zone.lang_vo.as_deref(), Some("en"));
        assert_eq!(zone.lang_vf, None);
        assert_eq!(zone.vo.refs, ["ulpi"]);
        assert_eq!(zone.vf.refs, ["ubb7", "UVR-MDX-NET-Voc_FT"]);
        assert_eq!(zone.vo.reasons["bd2"], Reason::Leak);
        assert_eq!(zone.vf.reasons["ulpi+cancel"], Reason::Lack);
        let context = zone.duplicates.as_ref().unwrap();
        assert_eq!(context.dub_lead_ms, 78.0);
        assert_eq!(
            context.steps,
            [Step {
                centre: -240_000,
                offset: 3280
            }]
        );

        let measure = set.measure(&zone.id, "ulpi").unwrap();
        let alone = &measure.stems[&Stem::VoiceVo];
        assert!(alone.reference);
        assert_eq!(alone.si_sdr, None);
        let voice = &measure.stems[&Stem::VoiceVf];
        assert_eq!(
            (voice.si_sdr, voice.residue, voice.loss, voice.artifacts),
            (Some(32.5), Some(-32.25), Some(-53.0), Some(-48.0))
        );
        assert_eq!(
            (voice.gap, voice.excess, voice.lack, voice.lag),
            (Some(2.25), Some(2.0), Some(0.25), -3)
        );
        // The time of a variant is only in the zone's tests.json.
        let variant = set.measure(&zone.id, "ulpi+cancel").unwrap();
        assert_eq!(variant.seconds, 0.125);
        assert_eq!(variant.stems[&Stem::RestVf].si_sdr, Some(-3.5));

        assert!(!set.is_checked("bd2"));
        assert_eq!(
            set.cascades,
            [Cascade {
                first: "ulpi".into(),
                second: "ubb7".into(),
                mode: CascadeMode::Voice,
                kept: true
            }]
        );
        assert_eq!(set.imported_fits[&Side::Vo].pairs, 2875);

        let read = |path: PathBuf| fs::read_to_string(path).unwrap();
        assert_eq!(read(layout::extract(&dir, zone, Side::Vo)), "vo");
        assert_eq!(read(layout::extract(&dir, zone, Side::Vf)), "vf");
        assert_eq!(read(layout::others(&dir, zone)), "others");
        assert_eq!(
            read(layout::reference(&dir, zone, Stem::RestVo)),
            "ref vo rest"
        );
        assert!(!layout::reference_present(&dir, zone, Side::Vf));
        assert_eq!(
            read(layout::output(&dir, zone, "ulpi", Stem::RestVf)),
            "4444"
        );
        assert_eq!(read(layout::mix(&dir, zone, "ulpi+cancel")), "55555");
        assert!(layout::outputs_present(&dir, zone, "UVR-MDX-NET-Voc_FT"));
        assert_eq!(
            layout::keys_on_disk(&dir, zone),
            ["UVR-MDX-NET-Voc_FT", "ulpi", "ulpi+cancel"]
        );

        let loudness = Loudness::load(&dir, zone, "ulpi");
        assert_eq!(loudness.of(Stem::VoiceVo), Some(-35.5));
        assert_eq!(loudness.of(Stem::RestVo), Some(-31.75));
        assert_eq!(loudness.of(Stem::VoiceVf), None);
        assert_eq!(loudness.mix(), Some(-25.25));
        assert_eq!(
            Loudness::load(&dir, zone, "UVR-MDX-NET-Voc_FT"),
            Loudness::default()
        );

        fs::remove_dir_all(&from).unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_zone_the_bench_has_is_left_as_it_is() {
        let from = scratch("again-from");
        let dir = scratch("again-to");
        legacy_set(&from);
        run(&from, &dir, known, |_| {}).unwrap();

        // Rated here since: the import must not bring the old rating back.
        let mut set = Set::load(&dir).unwrap();
        set.zones[0].vo.refs = vec!["bd2".into()];
        set.checked.insert("bd2".into(), true);
        set.save(&dir).unwrap();

        let report = run(&from, &dir, known, |_| {}).unwrap();
        assert!(report.zones.is_empty());
        assert_eq!(report.skipped, ["Film_o12_cC_723"]);
        assert_eq!(report.files, 0);
        let set = Set::load(&dir).unwrap();
        assert_eq!(set.zones.len(), 1);
        assert_eq!(set.zones[0].vo.refs, ["bd2"]);
        assert!(set.is_checked("bd2"));
        assert_eq!(set.cascades.len(), 1);

        fs::remove_dir_all(&from).unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_missing_extract_is_said() {
        let from = scratch("missing-from");
        let dir = scratch("missing-to");
        legacy_set(&from);
        fs::remove_file(from.join("Film_o12_cC_723/1.Extraction/vf_C.wav")).unwrap();
        let report = run(&from, &dir, known, |_| {}).unwrap();
        assert_eq!(report.warnings.len(), 1);
        assert!(
            report.warnings[0].contains("vf_C.wav"),
            "{:?}",
            report.warnings
        );
        fs::remove_dir_all(&from).unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn no_set_to_import_is_an_error() {
        let from = scratch("none-from");
        let dir = scratch("none-to");
        assert!(run(&from, &dir, known, |_| {}).is_err());
        assert!(!Set::path(&dir).exists());
        fs::remove_dir_all(&from).unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }
}
