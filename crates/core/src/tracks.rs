//! Which track is the original and which one is the dub.
//!
//! The original is the first track carrying a scene — that is what there is
//! to re-voice. The dub is chosen by ranking, not by container order:
//! a MKV that lists a VFQ AC-3 5.1 before a VFF DD+ 7.1 would otherwise
//! hand us the Québec dub, and one that opens with an audio description
//! would hand us that. `--vo N` and `--vf N` override both.

use crate::probe::{AudioTrack, Codec, Media};
use crate::text;
use crate::{Error, Result};

/// The track to re-voice: the first one carrying a scene.
pub fn original(media: &Media) -> Result<&AudioTrack> {
    media.audio.iter().find(|t| t.is_spatial()).ok_or_else(|| {
        Error::msg(format!(
            "{}: no spatial track (TrueHD Atmos, E-AC-3 JOC, DTS:X or Auro-3D) among {} audio tracks",
            media.path.display(),
            media.audio.len()
        ))
    })
}

/// The track whose dialogue is taken: the best French one, or the best of
/// the rest when the file has no French at all.
pub fn dub(media: &Media) -> Result<&AudioTrack> {
    let plain: Vec<&AudioTrack> = media.audio.iter().filter(|t| !t.is_spatial()).collect();
    let french: Vec<&AudioTrack> = plain.iter().copied().filter(|t| is_french(t)).collect();
    let candidates = if french.is_empty() { &plain } else { &french };
    candidates
        .iter()
        .copied()
        .min_by_key(|t| key(t))
        .ok_or_else(|| {
            Error::msg(format!(
                "{}: every audio track carries a scene; none is left to take the dialogue from",
                media.path.display()
            ))
        })
}

/// What the ranking compares, smallest first: not a side track, then the
/// home dub before the Québec one, then more channels, a better codec, a
/// higher bit rate, and — everything equal — the container's own order.
fn key(track: &AudioTrack) -> (u8, u8, i64, i64, i64, u32) {
    (
        track.side as u8,
        is_quebec(track) as u8,
        -(track.channels as i64),
        -(codec_rank(track) as i64),
        -(track.bit_rate as i64),
        track.index,
    )
}

/// Quality of a codec, 5 for lossless down to 0 for one we cannot name.
/// A DTS track is only as good as its profile: DTS-HD MA is lossless, HRA
/// is not, and the core alone is a lossy 1.5 Mb/s stream.
fn codec_rank(track: &AudioTrack) -> u8 {
    let profile = track.profile.as_deref().unwrap_or("");
    match &track.codec {
        Codec::TrueHd | Codec::Lossless(_) => 5,
        Codec::Dts if profile.contains("MA") => 5,
        Codec::Dts if profile.contains("HRA") || profile.contains("HR") => 4,
        Codec::Dts | Codec::Eac3 => 3,
        Codec::Ac3 => 2,
        Codec::Other(name) if name.is_empty() => 0,
        Codec::Other(_) => 1,
    }
}

/// Whether the track is in French, however the container spells it.
pub fn is_french(track: &AudioTrack) -> bool {
    matches!(iso3(track.language.as_deref()), "fre" | "fra")
}

/// A Québec dub, which the home dub is preferred to: `fr-CA`, or a title
/// that says so.
pub fn is_quebec(track: &AudioTrack) -> bool {
    if region(track.language.as_deref()).as_deref() == Some("CA") {
        return true;
    }
    let Some(title) = &track.title else {
        return false;
    };
    let words = text::words(title);
    ["vfq", "quebecois", "quebec", "canadian", "canadien"]
        .iter()
        .any(|w| text::has_word(&words, w))
}

/// The primary subtag of a language code, lower-cased: `fr-CA` is `fr`.
fn primary(language: Option<&str>) -> String {
    language
        .unwrap_or("")
        .split(['-', '_'])
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
}

/// The region of a language code, upper-cased: `fr-CA` is `CA`.
pub fn region(language: Option<&str>) -> Option<String> {
    let code = language?;
    let region = code.split(['-', '_']).nth(1)?.trim();
    (region.len() == 2).then(|| region.to_ascii_uppercase())
}

/// The ISO 639-2 code Matroska wants on a track (`fre`, `eng`), from
/// whatever the source carried (`fr`, `fr-CA`, `fre`). `und` when unknown.
pub fn iso3(language: Option<&str>) -> &'static str {
    let primary = primary(language);
    if let Some((_, iso3)) = TABLE.iter().find(|(two, _)| *two == primary) {
        return iso3;
    }
    // A three-letter code is already one; the table only knows the ones a
    // dub is likely to be in, and an unknown one is passed through.
    match LEAVE_AS_IS.iter().find(|code| **code == primary) {
        Some(code) => code,
        None => "und",
    }
}

/// ISO 639-1 to the ISO 639-2/B code Matroska carries.
const TABLE: [(&str, &str); 27] = [
    ("fr", "fre"),
    ("en", "eng"),
    ("es", "spa"),
    ("de", "ger"),
    ("it", "ita"),
    ("pt", "por"),
    ("nl", "dut"),
    ("ru", "rus"),
    ("ja", "jpn"),
    ("ko", "kor"),
    ("zh", "chi"),
    ("pl", "pol"),
    ("sv", "swe"),
    ("da", "dan"),
    ("no", "nor"),
    ("nb", "nob"),
    ("fi", "fin"),
    ("cs", "cze"),
    ("hu", "hun"),
    ("el", "gre"),
    ("tr", "tur"),
    ("ar", "ara"),
    ("he", "heb"),
    ("hi", "hin"),
    ("th", "tha"),
    ("uk", "ukr"),
    ("ro", "rum"),
];

/// Three-letter codes passed through as they are, both the bibliographic
/// and the terminological form of the ones that have two.
const LEAVE_AS_IS: [&str; 34] = [
    "fre", "fra", "eng", "spa", "ger", "deu", "ita", "por", "dut", "nld", "rus", "jpn", "kor",
    "chi", "zho", "pol", "swe", "dan", "nor", "nob", "fin", "cze", "ces", "hun", "gre", "ell",
    "tur", "ara", "heb", "hin", "tha", "ukr", "rum", "ron",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::Spatial;

    fn track(index: u32, codec: Codec, channels: u32, language: &str, title: &str) -> AudioTrack {
        AudioTrack {
            index,
            stream: index,
            codec,
            profile: None,
            channels,
            sample_rate: 48000,
            bit_rate: 640_000,
            language: Some(language.to_string()),
            title: (!title.is_empty()).then(|| title.to_string()),
            default: false,
            start: 0.0,
            side: false,
            spatial: None,
            probe_error: None,
        }
    }

    fn media(audio: Vec<AudioTrack>) -> Media {
        Media {
            path: "/films/Film.mkv".into(),
            duration: 6000.0,
            title: None,
            audio,
            subtitles: 0,
            video: 1,
        }
    }

    fn spatial(mut track: AudioTrack) -> AudioTrack {
        track.spatial = Some(Spatial {
            label: "TrueHD".into(),
            kind: "atmos".into(),
            objects: None,
            fixed: None,
            presentation: None,
            experimental: false,
        });
        track
    }

    #[test]
    fn the_original_is_the_first_track_carrying_a_scene() {
        let media = media(vec![
            track(1, Codec::Ac3, 6, "eng", ""),
            spatial(track(2, Codec::TrueHd, 8, "eng", "")),
            spatial(track(3, Codec::Eac3, 6, "eng", "")),
        ]);
        assert_eq!(original(&media).unwrap().index, 2);
    }

    #[test]
    fn the_home_dub_beats_the_quebec_one_whatever_the_container_order() {
        let media = media(vec![
            spatial(track(1, Codec::TrueHd, 8, "eng", "English Atmos")),
            track(2, Codec::Ac3, 6, "fre", "VFQ AC-3 5.1"),
            track(3, Codec::Eac3, 8, "fre", "VFF DD+ 7.1"),
        ]);
        assert_eq!(dub(&media).unwrap().index, 3);
    }

    #[test]
    fn an_audio_description_is_never_the_dub() {
        let mut ad = track(2, Codec::Eac3, 8, "fre", "Audiodescription");
        ad.side = true;
        let media = media(vec![
            spatial(track(1, Codec::TrueHd, 8, "eng", "")),
            ad,
            track(3, Codec::Ac3, 6, "fre", "VFF"),
        ]);
        assert_eq!(dub(&media).unwrap().index, 3);
    }

    #[test]
    fn french_is_preferred_to_a_better_track_in_another_language() {
        let media = media(vec![
            spatial(track(1, Codec::TrueHd, 8, "eng", "")),
            track(2, Codec::Eac3, 8, "ger", "German 7.1"),
            track(3, Codec::Ac3, 6, "fre", "French 5.1"),
        ]);
        assert_eq!(dub(&media).unwrap().index, 3);
    }

    #[test]
    fn without_french_the_best_of_the_rest_is_taken() {
        let media = media(vec![
            spatial(track(1, Codec::TrueHd, 8, "eng", "")),
            track(2, Codec::Ac3, 6, "ger", ""),
            track(3, Codec::Eac3, 6, "spa", ""),
        ]);
        assert_eq!(dub(&media).unwrap().index, 3);
    }

    #[test]
    fn a_lossless_dts_outranks_a_lossy_one_at_equal_channels() {
        let mut ma = track(2, Codec::Dts, 6, "fre", "VFF DTS-HD MA");
        ma.profile = Some("DTS-HD MA".into());
        let mut core = track(3, Codec::Dts, 6, "fre", "VFF DTS");
        core.profile = Some("DTS".into());
        core.bit_rate = 1_509_000;
        let media = media(vec![
            spatial(track(1, Codec::TrueHd, 8, "eng", "")),
            core,
            ma,
        ]);
        assert_eq!(dub(&media).unwrap().index, 2);
    }

    #[test]
    fn matroska_codes_come_out_of_whatever_the_source_carried() {
        assert_eq!(iso3(Some("fr")), "fre");
        assert_eq!(iso3(Some("fr-CA")), "fre");
        assert_eq!(iso3(Some("fra")), "fra");
        assert_eq!(iso3(Some("chi")), "chi");
        assert_eq!(iso3(Some("klingon")), "und");
        assert_eq!(iso3(None), "und");
    }

    #[test]
    fn a_quebec_dub_is_known_by_its_region_or_its_title() {
        assert!(is_quebec(&track(1, Codec::Ac3, 6, "fr-CA", "")));
        assert!(is_quebec(&track(1, Codec::Ac3, 6, "fre", "VFQ")));
        assert!(is_quebec(&track(1, Codec::Ac3, 6, "fre", "Québécois")));
        assert!(!is_quebec(&track(1, Codec::Ac3, 6, "fre", "VFF")));
    }
}
