//! What the track says about itself: Matroska tags of the track.
//!
//! The models and every parameter of the job travel *in* the `.mka` (and
//! the `.mkv`), rather than only in the file's name and in a manifest left
//! on a working disk. They are tags **of the track** — `TargetTypeValue` 30,
//! neither global tags nor an attachment — because that is the one form that
//! follows the track when it is remuxed from the `.mka` into a film's
//! `.mkv`, with mkvmerge as with ffmpeg. Their keys are upper-case with
//! underscores: ffmpeg upper-cases keys when it rewrites them, and reads a
//! `-xxx` suffix as a language code.
//!
//! The keys:
//!
//! - `OMNIFROG_DATE` — when the track was made, in UTC;
//! - `OMNIFROG_MODELS` — `objets: <name> [<checkpoint>] ; voix: <name> [<checkpoint>]`;
//! - `OMNIFROG_MANIFEST` — the job's manifest as JSON, minus what is only a
//!   resume's state and minus anything that names the machine it ran on;
//! - `OMNIFROG_TOOL` — this program and its version, so a track says which
//!   of the two tools made it.

use crate::config::{Encoder, IamfCodec};
use crate::job::Manifest;
use crate::models::Catalogue;
use crate::{Error, NAME, Result, VERSION};
use serde_json::Value;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Write the tags of the new track to `path`, in the XML `mkvmerge --tags`
/// reads (the one `mkvextract tags` writes). mkvmerge adds the track's UID
/// itself for the track `--tags TID:` names.
pub fn write(manifest: &Manifest, catalogue: &Catalogue, path: &Path) -> Result<()> {
    let tags = [
        ("OMNIFROG_DATE", utc_now()),
        ("OMNIFROG_MODELS", models(manifest, catalogue)),
        ("OMNIFROG_MANIFEST", published(manifest)?),
        ("OMNIFROG_TOOL", format!("{NAME} {VERSION}")),
    ];
    let mut xml = String::from(
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n\
         <!DOCTYPE Tags SYSTEM \"matroskatags.dtd\">\n\
         <Tags>\n  <Tag>\n    <Targets>\n      <TargetTypeValue>30</TargetTypeValue>\n    </Targets>\n",
    );
    for (name, value) in tags {
        xml.push_str(&format!(
            "    <Simple>\n      <Name>{}</Name>\n      <String>{}</String>\n    </Simple>\n",
            escape(name),
            escape(&value)
        ));
    }
    xml.push_str("  </Tag>\n</Tags>\n");
    std::fs::write(path, xml).map_err(|e| Error::io(path, e))
}

/// `objets: <name> [<checkpoint>] ; voix: <name> [<checkpoint>]`.
fn models(manifest: &Manifest, catalogue: &Catalogue) -> String {
    let one = |model: &crate::job::ModelRef| match catalogue.get(&model.key) {
        Ok(entry) => format!("{} [{}]", entry.name, model.checkpoint),
        Err(_) => model.checkpoint.clone(),
    };
    format!(
        "objets: {} ; voix: {}",
        one(&manifest.model_original),
        one(&manifest.model_dub)
    )
}

/// The manifest as the track carries it: what describes the track — its
/// sources, models, parameters and measurements — and nothing that is only
/// a resume's bookkeeping or a path on the machine that made it. The source
/// is its file name alone; the stems' signatures, which hold its full path,
/// are dropped with the rest of the state. Of the encoders' free options
/// only those of the encoder that made the track are kept: the others were
/// handed to nothing.
fn published(manifest: &Manifest) -> Result<String> {
    let mut value = serde_json::to_value(manifest)
        .map_err(|e| Error::msg(format!("the manifest as a tag: {e}")))?;
    let Value::Object(object) = &mut value else {
        return Err(Error::msg("the manifest is not an object"));
    };
    for state in ["done", "durations", "out_dir", "output"] {
        object.remove(state);
    }
    if manifest.encoder != Encoder::Hz {
        object.remove("hz_opts");
    }
    if !manifest.encoder.is_iamf() {
        for key in ["iamf_opts", "iamf_codec", "iamf_bitrate"] {
            object.remove(key);
        }
    } else if manifest.iamf_codec != IamfCodec::Opus {
        object.remove("iamf_bitrate");
    }
    if let Some(name) = manifest.source.file_name() {
        object.insert(
            "source".into(),
            Value::String(name.to_string_lossy().into_owned()),
        );
    }
    for separated in ["separate_original", "separate_dub"] {
        if let Some(Value::Object(inner)) = object.get_mut(separated) {
            inner.remove("signature");
        }
    }
    if let Some(Value::Object(master)) = object.get_mut("master") {
        master.remove("inputs");
    }
    if let Some(Value::Object(encode)) = object.get_mut("encode") {
        encode.remove("stream");
    }
    serde_json::to_string(&value).map_err(|e| Error::msg(format!("the manifest as a tag: {e}")))
}

/// Now, as `2026-09-21T18:04:05Z`.
pub fn utc_now() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    utc(seconds)
}

/// A count of seconds since 1970 as a UTC date and time. The civil date is
/// Howard Hinnant's days-from-epoch algorithm, which is exact over the whole
/// proleptic Gregorian calendar and needs no table.
fn utc(seconds: u64) -> String {
    let days = (seconds / 86_400) as i64;
    let second_of_day = seconds % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        second_of_day / 3_600,
        second_of_day / 60 % 60,
        second_of_day % 60
    )
}

/// The five characters XML reserves.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_count_of_seconds_is_the_calendar_date() {
        assert_eq!(utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(utc(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(utc(1_790_013_845), "2026-09-21T18:04:05Z");
        assert_eq!(utc(4_107_542_399), "2100-02-28T23:59:59Z");
    }

    #[test]
    fn a_track_carries_the_options_of_its_own_encoder_only() {
        let mut manifest = crate::job::tests::manifest();
        manifest.encoder = Encoder::Iamf;
        manifest.iamf_opts = "--codec opus".into();
        let tag: Value = serde_json::from_str(&published(&manifest).unwrap()).unwrap();
        assert_eq!(tag["encoder"], "iamf");
        assert_eq!(tag["iamf_opts"], "--codec opus");
        assert!(tag.get("hz_opts").is_none(), "{tag}");
        assert!(tag.get("done").is_none(), "{tag}");
        assert_eq!(tag["source"], "Some.Film.mkv");
        manifest.encoder = Encoder::Hz;
        let tag: Value = serde_json::from_str(&published(&manifest).unwrap()).unwrap();
        assert_eq!(tag["hz_opts"], "--presentations");
        assert!(tag.get("iamf_opts").is_none(), "{tag}");
        // The mode is the encoder the track records, with the same options.
        manifest.encoder = Encoder::IamfBed;
        let tag: Value = serde_json::from_str(&published(&manifest).unwrap()).unwrap();
        assert_eq!(tag["encoder"], "iamf-lit");
        assert_eq!(tag["iamf_opts"], "--codec opus");
        assert!(tag.get("hz_opts").is_none(), "{tag}");
    }

    #[test]
    fn a_track_carries_its_iamf_codec_and_an_opus_bitrate_only() {
        let mut manifest = crate::job::tests::manifest();
        manifest.encoder = Encoder::Iamf;
        let tag: Value = serde_json::from_str(&published(&manifest).unwrap()).unwrap();
        assert_eq!(tag["iamf_codec"], "flac");
        assert!(tag.get("iamf_bitrate").is_none(), "{tag}");
        manifest.iamf_codec = IamfCodec::Opus;
        manifest.iamf_bitrate = 48;
        let tag: Value = serde_json::from_str(&published(&manifest).unwrap()).unwrap();
        assert_eq!(tag["iamf_codec"], "opus");
        assert_eq!(tag["iamf_bitrate"], 48);
        manifest.encoder = Encoder::Hz;
        let tag: Value = serde_json::from_str(&published(&manifest).unwrap()).unwrap();
        assert!(tag.get("iamf_codec").is_none(), "{tag}");
        assert!(tag.get("iamf_bitrate").is_none(), "{tag}");
    }

    #[test]
    fn reserved_characters_are_escaped() {
        assert_eq!(
            escape(r#"a<b & "c">'d'"#),
            "a&lt;b &amp; &quot;c&quot;&gt;&apos;d&apos;"
        );
    }
}
