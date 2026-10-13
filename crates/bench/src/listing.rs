//! What a zone has to show: one row per output, in the catalogue's order.

use crate::keys;
use crate::layout::{self, Loudness};
use crate::set::{Measure, Set, Side, Status, Stem, Zone};
use std::path::Path;

/// One output of a zone, as a list shows it.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub key: String,
    /// The catalogue does not have its model: its outputs came from elsewhere.
    pub foreign: bool,
    /// Which of the four stems are on disk, in the order of [`Stem::ALL`].
    pub stems: [bool; 4],
    pub mix: bool,
    /// What the ear said, in the order of [`Side::BOTH`].
    pub status: [Status; 2],
    pub loudness: Loudness,
    pub measure: Option<Measure>,
}

impl Row {
    pub fn status(&self, side: Side) -> Status {
        self.status[side as usize]
    }

    /// Files of the output on disk, out of five.
    pub fn files(&self) -> usize {
        self.stems.iter().filter(|&&present| present).count() + usize::from(self.mix)
    }
}

/// The rows of a zone: every key that has files, a rating or a measure on
/// it. `order` is the catalogue's keys; a variant follows its model, and the
/// keys the catalogue does not have come last, by name.
pub fn rows(dir: &Path, set: &Set, zone: &Zone, order: &[String]) -> Vec<Row> {
    let mut all = layout::keys_on_disk(dir, zone);
    for side in Side::BOTH {
        let rating = zone.rating(side);
        all.extend(rating.refs.iter().chain(&rating.rejects).cloned());
    }
    all.extend(
        set.measures
            .get(&zone.id)
            .into_iter()
            .flatten()
            .map(|(k, _)| k.clone()),
    );
    let place = |key: &str| {
        let (model, variant) = keys::split(key);
        // A cascade follows the model it starts with.
        let first = keys::stages(key).map_or(model, |(first, _, _)| first);
        let model_place = order.iter().position(|k| k == first).unwrap_or(usize::MAX);
        let variant_place = match variant {
            None => 0,
            Some(keys::CANCEL) => 1,
            Some(_) => 2,
        };
        (model_place, model.to_string(), variant_place)
    };
    all.sort_by_cached_key(|key| place(key));
    all.dedup();

    all.into_iter()
        .map(|key| Row {
            foreign: !known(order, &key),
            stems: Stem::ALL.map(|stem| layout::output(dir, zone, &key, stem).is_file()),
            mix: layout::mix(dir, zone, &key).is_file(),
            status: Side::BOTH.map(|side| zone.rating(side).status(&key)),
            loudness: Loudness::load(dir, zone, &key),
            // A measure emptied by a change of references is no measure to
            // show.
            measure: set
                .measure(&zone.id, &key)
                .filter(|m| m.measured())
                .cloned(),
            key,
        })
        .collect()
}

/// Whether the bench can compute a key again: its model is in the
/// catalogue, or it is a cascade of two that are.
pub fn known(order: &[String], key: &str) -> bool {
    let has = |model: &str| order.iter().any(|k| k == model);
    match keys::stages(key) {
        Some((first, second, _)) => has(first) && has(second),
        None => has(keys::base(key)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::set::Reason;
    use std::fs;

    #[test]
    fn a_zone_lists_what_it_has_in_the_catalogue_s_order() {
        let dir = std::env::temp_dir().join(format!("omnifrog-bench-rows-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut zone = Zone {
            id: "Film_o12_cC_723".into(),
            object: 12,
            channel: "C".into(),
            ..Zone::default()
        };
        zone.vo.refs.push("ulpi".into());
        zone.vf.rejects.push("zzz_elsewhere".into());
        zone.vf.reasons.insert("zzz_elsewhere".into(), Reason::Leak);
        let mut measured = Measure::default();
        measured.stems.insert(Stem::VoiceVf, Default::default());
        let mut set = Set::default();
        set.measures
            .entry(zone.id.clone())
            .or_default()
            .insert("bd2".into(), measured);
        // Emptied by a change of references: its time is kept in the set,
        // and it is no measure to show.
        set.measures.get_mut(&zone.id).unwrap().insert(
            "ulpi".into(),
            Measure {
                seconds: 3.0,
                ..Measure::default()
            },
        );
        for (key, stems) in [
            ("ulpi+cancel", &[Stem::VoiceVf, Stem::RestVf][..]),
            ("ulpi", &Stem::ALL[..]),
            ("Elsewhere", &[Stem::VoiceVo][..]),
        ] {
            for &stem in stems {
                let path = layout::output(&dir, &zone, key, stem);
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(path, "").unwrap();
            }
        }
        fs::write(layout::mix(&dir, &zone, "ulpi"), "").unwrap();

        let order = ["bd2".to_string(), "ulpi".to_string()];
        let rows = rows(&dir, &set, &zone, &order);
        let keys: Vec<&str> = rows.iter().map(|r| r.key.as_str()).collect();
        assert_eq!(
            keys,
            ["bd2", "ulpi", "ulpi+cancel", "Elsewhere", "zzz_elsewhere"]
        );
        // Measured, with nothing on disk.
        assert_eq!(rows[0].files(), 0);
        assert!(rows[0].measure.is_some());
        assert_eq!(rows[1].files(), 5);
        assert!(rows[1].measure.is_none());
        assert_eq!(rows[1].status(Side::Vo), Status::Reference);
        assert_eq!(rows[1].status(Side::Vf), Status::Neutral);
        assert_eq!(rows[2].stems, [false, false, true, true]);
        assert!(!rows[2].foreign);
        assert!(rows[3].foreign);
        // A cascade of two models of the catalogue is no stranger, and
        // comes after the model it starts with; with one that is not, it is.
        assert!(known(&order, "ulpi~bd2~rest+cancel"));
        assert!(!known(&order, "ulpi~Elsewhere~rest"));
        assert!(!known(&order, "Elsewhere"));
        assert_eq!(
            rows[4].status(Side::Vf),
            Status::Rejected(Some(Reason::Leak))
        );
        fs::remove_dir_all(&dir).unwrap();
    }
}
