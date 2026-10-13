//! The name an output goes by.
//!
//! An output's key is the catalogue key of the model that made it, with a
//! suffix when it is a variant computed from that model's outputs rather than
//! by a model. A key the catalogue does not know is *foreign*: outputs that
//! came from elsewhere (an imported set's audio-separator models), which can be
//! listened to and measured but not computed again.

/// The dub's stems with the original's music and effects cancelled out of
/// them, one alignment for the whole zone.
pub const CANCEL: &str = "+cancel";
/// The same, aligned step by step along the map of the dub.
pub const CANCEL_STEP: &str = "+cancel-step";

/// The variants, longest suffix first so that one is never read as the other.
pub const VARIANTS: [&str; 2] = [CANCEL_STEP, CANCEL];

/// A key split into the model's key and the variant's suffix, if any.
pub fn split(key: &str) -> (&str, Option<&'static str>) {
    for suffix in VARIANTS {
        if let Some(base) = key.strip_suffix(suffix) {
            return (base, Some(suffix));
        }
    }
    (key, None)
}

/// The key of the model a key's outputs come from.
pub fn base(key: &str) -> &str {
    split(key).0
}

/// Whether the key names a variant. A variant has no outputs of its own on
/// the original's side: they are its model's.
pub fn is_variant(key: &str) -> bool {
    split(key).1.is_some()
}

/// What joins the parts of a cascade's key: a character no catalogue key
/// has, and that every file system takes in a directory's name.
pub const THEN: char = '~';

/// The key of a cascade: its two models, in the order they run, and the
/// stem of the first the second works again — `rest` or `voice`.
pub fn cascade(first: &str, second: &str, mode: crate::set::CascadeMode) -> String {
    let mode = match mode {
        crate::set::CascadeMode::Rest => "rest",
        crate::set::CascadeMode::Voice => "voice",
    };
    format!("{first}{THEN}{second}{THEN}{mode}")
}

/// A cascade's key taken apart: its two models and its mode. None for any
/// other key; a variant of a cascade is read through its base.
pub fn stages(key: &str) -> Option<(&str, &str, crate::set::CascadeMode)> {
    let mut parts = base(key).split(THEN);
    let (first, second, mode) = (parts.next()?, parts.next()?, parts.next()?);
    let mode = match mode {
        "rest" => crate::set::CascadeMode::Rest,
        "voice" => crate::set::CascadeMode::Voice,
        _ => return None,
    };
    (parts.next().is_none() && !first.is_empty() && !second.is_empty())
        .then_some((first, second, mode))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_splits_into_its_model_and_its_variant() {
        assert_eq!(split("ulpi"), ("ulpi", None));
        assert_eq!(split("ulpi+cancel"), ("ulpi", Some(CANCEL)));
        assert_eq!(split("ulpi+cancel-step"), ("ulpi", Some(CANCEL_STEP)));
        assert_eq!(base("UVR-MDX-NET-Voc_FT+cancel"), "UVR-MDX-NET-Voc_FT");
        assert!(is_variant("kf2b+cancel-step"));
        assert!(!is_variant("kf2b"));
    }

    #[test]
    fn a_cascade_s_key_names_its_two_models_and_its_mode() {
        use crate::set::CascadeMode;
        let key = cascade("ulpi", "ubb7", CascadeMode::Rest);
        assert_eq!(key, "ulpi~ubb7~rest");
        assert_eq!(stages(&key), Some(("ulpi", "ubb7", CascadeMode::Rest)));
        assert_eq!(
            stages("a~b~voice+cancel"),
            Some(("a", "b", CascadeMode::Voice))
        );
        assert_eq!(base("a~b~voice+cancel-step"), "a~b~voice");
        for other in ["ulpi", "a~b", "a~b~both", "a~b~rest~more", "~b~rest"] {
            assert_eq!(stages(other), None, "{other}");
        }
    }
}
