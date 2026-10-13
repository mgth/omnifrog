//! Rating an output by ear.
//!
//! On each side of a zone an output is a reference, a reject, or neither.
//! When the list of references of a side changes, the consensus is written
//! again — or removed with the last of them — and what was measured on that
//! side of the zone, now against an old reference, is dropped: it is to be
//! measured again. The other side's measures were against its own
//! references and stand; so does the time an output took to compute, which
//! no reference has anything to do with.

use crate::set::{Set, Side, Status, Stem};
use crate::{Error, Result, consensus, layout};
use std::fs;
use std::path::Path;

/// Give an output its status on one side of a zone, and save. A reference
/// must have both stems of that side on the zone. Whether the references
/// changed — and with them the consensus, what was measured on that side
/// of the zone being dropped. On an error nothing is changed.
pub fn change(
    dir: &Path,
    set: &mut Set,
    zone_id: &str,
    key: &str,
    side: Side,
    status: Status,
) -> Result<bool> {
    let place = set
        .zones
        .iter()
        .position(|z| z.id == zone_id)
        .ok_or_else(|| Error::msg(format!("no zone `{zone_id}` in the set")))?;
    if status == Status::Reference && !layout::side_present(dir, &set.zones[place], key, side) {
        return Err(Error::msg(format!(
            "`{key}` has no {} outputs on this zone yet: compute them first",
            side.label()
        )));
    }
    let zone = &mut set.zones[place];
    let rating = zone.rating_mut(side);
    let was_reference = rating.refs.iter().any(|k| k == key);
    rating.refs.retain(|k| k != key);
    rating.rejects.retain(|k| k != key);
    rating.reasons.remove(key);
    match status {
        Status::Reference => rating.refs.push(key.to_string()),
        Status::Rejected(reason) => {
            rating.rejects.push(key.to_string());
            if let Some(reason) = reason {
                rating.reasons.insert(key.to_string(), reason);
            }
        }
        Status::Neutral => {}
    }
    // Changing a reject's reason alone leaves the references as they are:
    // nothing to measure again.
    let changed = was_reference != (status == Status::Reference);
    if changed {
        if !consensus::write(dir, zone, side)? {
            for stem in [side.voice(), side.rest()] {
                remove(&layout::reference(dir, zone, stem))?;
            }
        }
        // Only what was measured against this side's references is stale.
        for measure in set.measures.get_mut(zone_id).into_iter().flatten() {
            measure.1.stems.retain(|stem, _| stem.side() != side);
        }
    }
    set.save(dir)?;
    Ok(changed)
}

fn remove(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(Error::io(path, e)),
        _ => Ok(()),
    }
}

/// The stems a change of rating on a side makes stale.
pub fn stems(side: Side) -> [Stem; 2] {
    [side.voice(), side.rest()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio;
    use crate::set::{Measure, RATE, Reason, Zone};
    use std::path::PathBuf;

    fn bench(name: &str) -> (PathBuf, Set) {
        let dir =
            std::env::temp_dir().join(format!("omnifrog-bench-rate-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let zone = Zone {
            id: "zone".into(),
            object: 1,
            channel: "C".into(),
            ..Zone::default()
        };
        for (key, level) in [("a", 0.25f32), ("b", 0.75)] {
            for stem in [Stem::VoiceVf, Stem::RestVf] {
                let samples = vec![level; 64];
                audio::write(&layout::output(&dir, &zone, key, stem), &samples, RATE).unwrap();
            }
        }
        // An output measured on both sides, that took twelve seconds and a
        // half to compute.
        let mut measure = Measure {
            seconds: 12.5,
            ..Measure::default()
        };
        for stem in Stem::ALL {
            measure.stems.insert(stem, Default::default());
        }
        let mut set = Set::default();
        set.measures
            .entry("zone".into())
            .or_default()
            .insert("a".into(), measure);
        set.zones.push(zone);
        (dir, set)
    }

    #[test]
    fn a_reference_writes_the_consensus_and_drops_the_measures() {
        let (dir, mut set) = bench("reference");
        assert!(change(&dir, &mut set, "zone", "a", Side::Vf, Status::Reference).unwrap());
        // What was measured against the dub's references is gone; what was
        // measured on the original's side stands, and so does the time.
        let measure = set.measure("zone", "a").unwrap();
        let left: Vec<Stem> = measure.stems.keys().copied().collect();
        assert_eq!(left, [Stem::VoiceVo, Stem::RestVo]);
        assert_eq!(measure.seconds, 12.5);
        let zone = set.zones[0].clone();
        assert_eq!(zone.vf.refs, ["a"]);
        let reference = layout::reference(&dir, &zone, Stem::VoiceVf);
        assert_eq!(audio::read(&reference).unwrap()[0], 0.25);
        // A second one: the consensus of two is their mean.
        assert!(change(&dir, &mut set, "zone", "b", Side::Vf, Status::Reference).unwrap());
        assert_eq!(audio::read(&reference).unwrap()[0], 0.5);
        // Saved as it went.
        assert_eq!(Set::load(&dir).unwrap().zones[0].vf.refs, ["a", "b"]);
        // Measured again, the dub's side is back and the time is the one
        // the separation took: nothing else could say it again.
        let stop = std::sync::atomic::AtomicBool::new(false);
        crate::measure::batch(&dir, &mut set, &["zone".to_string()], &stop, |_| {}).unwrap();
        let measure = set.measure("zone", "a").unwrap();
        assert!(measure.stems.contains_key(&Stem::VoiceVf));
        assert_eq!(measure.seconds, 12.5);

        // The last reference gone, so are the files.
        assert!(change(&dir, &mut set, "zone", "a", Side::Vf, Status::Neutral).unwrap());
        assert_eq!(audio::read(&reference).unwrap()[0], 0.75);
        let rejected = Status::Rejected(Some(Reason::Leak));
        assert!(change(&dir, &mut set, "zone", "b", Side::Vf, rejected).unwrap());
        assert!(!reference.exists());
        assert_eq!(set.zones[0].vf.status("b"), rejected);
        assert_eq!(set.measure("zone", "a").unwrap().seconds, 12.5);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_reject_or_its_reason_leaves_the_measures_alone() {
        let (dir, mut set) = bench("reject");
        let leak = Status::Rejected(Some(Reason::Leak));
        assert!(!change(&dir, &mut set, "zone", "a", Side::Vf, leak).unwrap());
        assert_eq!(set.measure("zone", "a").unwrap().stems.len(), 4);
        let lack = Status::Rejected(Some(Reason::Lack));
        assert!(!change(&dir, &mut set, "zone", "a", Side::Vf, lack).unwrap());
        assert_eq!(set.zones[0].vf.status("a"), lack);
        assert_eq!(set.zones[0].vf.rejects, ["a"]);
        assert!(
            !change(
                &dir,
                &mut set,
                "zone",
                "a",
                Side::Vf,
                Status::Rejected(None)
            )
            .unwrap()
        );
        assert!(set.zones[0].vf.reasons.is_empty());
        assert!(!change(&dir, &mut set, "zone", "a", Side::Vf, Status::Neutral).unwrap());
        assert_eq!(set.zones[0].vf.status("a"), Status::Neutral);
        assert_eq!(set.measures.len(), 1);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_reference_needs_its_outputs() {
        let (dir, mut set) = bench("missing");
        // Nothing on the original's side, and no such key at all.
        assert!(change(&dir, &mut set, "zone", "a", Side::Vo, Status::Reference).is_err());
        assert!(
            change(
                &dir,
                &mut set,
                "zone",
                "nobody",
                Side::Vf,
                Status::Reference
            )
            .is_err()
        );
        assert!(change(&dir, &mut set, "elsewhere", "a", Side::Vf, Status::Neutral).is_err());
        assert!(set.zones[0].vo.refs.is_empty());
        assert_eq!(set.measures.len(), 1);
        // A reject needs nothing.
        assert!(
            !change(
                &dir,
                &mut set,
                "zone",
                "nobody",
                Side::Vo,
                Status::Rejected(None)
            )
            .unwrap()
        );
        fs::remove_dir_all(&dir).unwrap();
    }
}
