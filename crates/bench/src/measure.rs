//! Measuring the outputs of a zone against its references.
//!
//! A side with no reference is not measured. An output that is the only
//! reference of its side is marked as such and not measured there — it
//! would be compared with itself; one of several is measured against their
//! consensus like any other, with the mark. A reject is measured too: the
//! rating keeps it out of the reference, not out of the table.

use crate::set::{Measure, Metric, Set, Side, Zone};
use crate::{Result, audio, consensus, keys, layout, metrics};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// The two stems of a side's reference: the voice, then the rest.
pub type Reference = [Vec<f32>; 2];

/// Where a zone's references are taken from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum From {
    /// The files written when the zone was last rated.
    Files,
    /// The outputs rated best, their consensus computed again.
    Outputs,
}

/// The references of a zone, in the order of [`Side::BOTH`]; none for a
/// side that has no rating or no file.
pub fn references(dir: &Path, zone: &Zone, from: From) -> Result<[Option<Reference>; 2]> {
    let side = |side: Side| -> Result<Option<Reference>> {
        if zone.rating(side).refs.is_empty() {
            return Ok(None);
        }
        match from {
            From::Outputs => consensus::of(dir, zone, side),
            From::Files if !layout::reference_present(dir, zone, side) => Ok(None),
            From::Files => Ok(Some([
                audio::read(&layout::reference(dir, zone, side.voice()))?,
                audio::read(&layout::reference(dir, zone, side.rest()))?,
            ])),
        }
    };
    Ok([side(Side::Vo)?, side(Side::Vf)?])
}

/// Measure one output of a zone. `seconds` is what its outputs took to
/// compute, kept from the measure it replaces.
pub fn output(
    dir: &Path,
    zone: &Zone,
    key: &str,
    references: &[Option<Reference>; 2],
    seconds: f64,
) -> Result<Measure> {
    let mut measure = Measure {
        seconds,
        date: omnifrog_core::tags::utc_now(),
        stems: BTreeMap::new(),
    };
    for side in Side::BOTH {
        // A variant's outputs on the original's side are its model's,
        // measured on the model's own row.
        if side == Side::Vo && keys::is_variant(key) {
            continue;
        }
        let Some([voice, rest]) = &references[side as usize] else {
            continue;
        };
        let rating = zone.rating(side);
        let reference = rating.refs.iter().any(|k| k == key);
        if reference && rating.refs.len() == 1 {
            for stem in [side.voice(), side.rest()] {
                measure.stems.insert(
                    stem,
                    Metric {
                        reference: true,
                        ..Metric::default()
                    },
                );
            }
            continue;
        }
        if !layout::side_present(dir, zone, key, side) {
            continue;
        }
        let out_voice = audio::read(&layout::output(dir, zone, key, side.voice()))?;
        let out_rest = audio::read(&layout::output(dir, zone, key, side.rest()))?;
        // The voice: what leaks into it is the rest. The rest: the voice.
        measure.stems.insert(
            side.voice(),
            metrics::metric(voice, rest, &out_voice, reference),
        );
        measure.stems.insert(
            side.rest(),
            metrics::metric(rest, voice, &out_rest, reference),
        );
    }
    Ok(measure)
}

/// Measure the given keys of a zone, several at once. A key with nothing to
/// measure — no file, no reference — gives no measure.
pub fn zone(
    dir: &Path,
    set: &Set,
    zone: &Zone,
    keys: &[String],
    from: From,
) -> Result<Vec<(String, Measure)>> {
    let references = references(dir, zone, from)?;
    let measures = keys
        .par_iter()
        .map(|key| {
            let seconds = set.measure(&zone.id, key).map_or(0.0, |m| m.seconds);
            output(dir, zone, key, &references, seconds).map(|measure| (key.clone(), measure))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(measures
        .into_iter()
        .filter(|(_, measure)| !measure.stems.is_empty())
        .collect())
}

/// The keys a batch takes on a zone: those with files there that are
/// checked — a variant with its model.
pub fn keys_of(dir: &Path, set: &Set, zone: &Zone) -> Vec<String> {
    layout::keys_on_disk(dir, zone)
        .into_iter()
        .filter(|key| set.is_checked(key) && set.is_checked(keys::base(key)))
        .collect()
}

/// What a batch says as it goes.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// A zone is being measured: its place, out of how many.
    Zone {
        id: String,
        index: usize,
        total: usize,
    },
    /// It is done, with this many outputs measured.
    Measured { id: String, outputs: usize },
}

/// Measure every checked output of the given zones against the reference
/// files, and save the set after each zone. `stop` ends the batch between
/// two zones; what was measured stays. Whether it ran to the end.
pub fn batch(
    dir: &Path,
    set: &mut Set,
    zones: &[String],
    stop: &AtomicBool,
    mut event: impl FnMut(Event),
) -> Result<bool> {
    for (index, id) in zones.iter().enumerate() {
        if stop.load(Ordering::Relaxed) {
            return Ok(false);
        }
        let Some(zone) = set.zone(id).cloned() else {
            continue;
        };
        event(Event::Zone {
            id: id.clone(),
            index,
            total: zones.len(),
        });
        let keys = keys_of(dir, set, &zone);
        let measures = self::zone(dir, set, &zone, &keys, From::Files)?;
        let outputs = measures.len();
        set.measures.entry(id.clone()).or_default().extend(measures);
        set.save(dir)?;
        event(Event::Measured {
            id: id.clone(),
            outputs,
        });
    }
    Ok(true)
}

/// What measuring a zone again finds against what the set and the files
/// hold, nothing written: the test of a port, and of a change to the
/// metrics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Check {
    /// Outputs measured again.
    pub outputs: usize,
    /// How far the measures are from the stored ones.
    pub measures: Difference,
    /// The output furthest from its stored measure.
    pub worst: Option<(String, f64)>,
    /// Outputs the set has no measure of.
    pub unmeasured: usize,
    /// For each side with a reference, in the order of [`Side::BOTH`]: the
    /// samples of the consensus computed again that are not the file's, out
    /// of how many.
    pub references: [Option<(usize, usize)>; 2],
}

/// Measure a zone again and compare, writing nothing.
pub fn check(dir: &Path, set: &Set, zone: &Zone) -> Result<Check> {
    let mut check = Check::default();
    let keys = keys_of(dir, set, zone);
    for (key, measure) in self::zone(dir, set, zone, &keys, From::Files)? {
        check.outputs += 1;
        let Some(stored) = set.measure(&zone.id, &key) else {
            check.unmeasured += 1;
            continue;
        };
        let difference = difference(stored, &measure);
        if check
            .worst
            .as_ref()
            .is_none_or(|(_, worst)| difference.largest > *worst)
        {
            check.worst = Some((key, difference.largest));
        }
        check.measures.add(difference);
    }
    let files = references(dir, zone, From::Files)?;
    let again = references(dir, zone, From::Outputs)?;
    for side in Side::BOTH {
        let (Some(files), Some(again)) = (&files[side as usize], &again[side as usize]) else {
            continue;
        };
        let (mut unlike, mut total) = (0, 0);
        for (file, again) in files.iter().zip(again) {
            total += file.len().max(again.len());
            unlike += file.len().abs_diff(again.len());
            unlike += file
                .iter()
                .zip(again)
                .filter(|(file, again)| audio::quantized(**file) != audio::quantized(**again))
                .count();
        }
        check.references[side as usize] = Some((unlike, total));
    }
    Ok(check)
}

/// How far a measure is from another: the largest difference over every
/// value of every stem, in dB, and whether something is measured in one and
/// not in the other.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Difference {
    pub largest: f64,
    /// Values present on one side only, lags that differ, marks that differ.
    pub unlike: usize,
    pub values: usize,
}

impl Difference {
    pub fn add(&mut self, other: Difference) {
        self.largest = self.largest.max(other.largest);
        self.unlike += other.unlike;
        self.values += other.values;
    }
}

pub fn difference(a: &Measure, b: &Measure) -> Difference {
    let mut difference = Difference::default();
    let stems: std::collections::BTreeSet<_> = a.stems.keys().chain(b.stems.keys()).collect();
    for stem in stems {
        let (Some(a), Some(b)) = (a.stems.get(stem), b.stems.get(stem)) else {
            difference.unlike += 1;
            continue;
        };
        if a.lag != b.lag || a.reference != b.reference {
            difference.unlike += 1;
        }
        for (a, b) in [
            (a.si_sdr, b.si_sdr),
            (a.residue, b.residue),
            (a.loss, b.loss),
            (a.artifacts, b.artifacts),
            (a.gap, b.gap),
            (a.excess, b.excess),
            (a.lack, b.lack),
        ] {
            match (a, b) {
                (Some(a), Some(b)) => {
                    difference.values += 1;
                    difference.largest = difference.largest.max((a - b).abs());
                }
                (None, None) => {}
                _ => difference.unlike += 1,
            }
        }
    }
    difference
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::set::{RATE, Stem};
    use std::fs;
    use std::path::PathBuf;

    fn noise(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                0.25 * ((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
            })
            .collect()
    }

    fn mix(a: &[f32], b: &[f32], share: f32) -> Vec<f32> {
        a.iter().zip(b).map(|(a, b)| a + share * b).collect()
    }

    /// A zone of one second with three outputs on both sides: a clean one,
    /// one that leaks a tenth of the other stem, and a variant.
    fn bench(name: &str) -> (PathBuf, Set) {
        let dir =
            std::env::temp_dir().join(format!("omnifrog-bench-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let zone = Zone {
            id: "zone".into(),
            object: 3,
            channel: "C".into(),
            ..Zone::default()
        };
        let n = RATE as usize;
        for (side, seed) in [(Side::Vo, 10), (Side::Vf, 20)] {
            let (voice, rest) = (noise(seed, n), noise(seed + 1, n));
            let put = |key: &str, stem: Stem, samples: &[f32]| {
                audio::write(&layout::output(&dir, &zone, key, stem), samples, RATE).unwrap();
            };
            put("clean", side.voice(), &voice);
            put("clean", side.rest(), &rest);
            put("leaky", side.voice(), &mix(&voice, &rest, 0.1));
            put("leaky", side.rest(), &mix(&rest, &voice, 0.1));
            put("leaky+cancel", side.voice(), &mix(&voice, &rest, 0.01));
            put("leaky+cancel", side.rest(), &mix(&rest, &voice, 0.01));
        }
        let mut set = Set::default();
        set.zones.push(zone);
        (dir, set)
    }

    #[test]
    fn a_side_without_a_reference_is_not_measured() {
        let (dir, mut set) = bench("no-reference");
        set.zones[0].vf.refs = vec!["clean".into()];
        consensus::write(&dir, &mut set.zones[0], Side::Vf).unwrap();
        let zone = set.zones[0].clone();
        let measures = self::zone(&dir, &set, &zone, &["leaky".into()], From::Files).unwrap();
        let stems: Vec<Stem> = measures[0].1.stems.keys().copied().collect();
        assert_eq!(stems, [Stem::VoiceVf, Stem::RestVf]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_only_reference_is_marked_and_the_others_measured_against_it() {
        let (dir, mut set) = bench("measure");
        for side in Side::BOTH {
            set.zones[0].rating_mut(side).refs = vec!["clean".into()];
            assert!(consensus::write(&dir, &mut set.zones[0], side).unwrap());
        }
        let stop = AtomicBool::new(false);
        let mut events = Vec::new();
        let done = batch(&dir, &mut set, &["zone".into()], &stop, |e| events.push(e)).unwrap();
        assert!(done);
        assert_eq!(
            events,
            [
                Event::Zone {
                    id: "zone".into(),
                    index: 0,
                    total: 1
                },
                Event::Measured {
                    id: "zone".into(),
                    outputs: 3
                }
            ]
        );
        // Saved as it went.
        let set = Set::load(&dir).unwrap();
        let clean = set.measure("zone", "clean").unwrap();
        assert_eq!(clean.stems.len(), 4);
        assert!(
            clean
                .stems
                .values()
                .all(|m| m.reference && m.si_sdr.is_none())
        );
        assert!(!clean.date.is_empty());

        let leaky = set.measure("zone", "leaky").unwrap();
        for stem in Stem::ALL {
            let m = &leaky.stems[&stem];
            assert!(!m.reference);
            assert!((m.si_sdr.unwrap() - 20.0).abs() < 0.2, "{stem:?} {m:?}");
            assert!((m.residue.unwrap() + 20.0).abs() < 0.2, "{stem:?} {m:?}");
            assert!(m.gap.is_some());
        }
        // A variant is measured on the dub's side only.
        let variant = set.measure("zone", "leaky+cancel").unwrap();
        let stems: Vec<Stem> = variant.stems.keys().copied().collect();
        assert_eq!(stems, [Stem::VoiceVf, Stem::RestVf]);
        assert!((variant.stems[&Stem::VoiceVf].si_sdr.unwrap() - 40.0).abs() < 0.5);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn one_of_several_references_is_measured_against_their_consensus() {
        let (dir, mut set) = bench("several");
        set.zones[0].vf.refs = vec!["clean".into(), "leaky".into(), "gone".into()];
        consensus::write(&dir, &mut set.zones[0], Side::Vf).unwrap();
        // A reference that has no files is dropped from the list.
        assert_eq!(set.zones[0].vf.refs, ["clean", "leaky"]);
        let zone = set.zones[0].clone();
        let keys = keys_of(&dir, &set, &zone);
        let from_files = self::zone(&dir, &set, &zone, &keys, From::Files).unwrap();
        let clean = &from_files.iter().find(|(k, _)| k == "clean").unwrap().1;
        let voice = &clean.stems[&Stem::VoiceVf];
        assert!(voice.reference);
        // The consensus of two is their mean: the clean one is half a leak away.
        assert!((voice.si_sdr.unwrap() - 26.0).abs() < 0.3, "{voice:?}");

        // Computed again from the outputs, the references are the files',
        // within the 24 bits a file keeps.
        let from_outputs = self::zone(&dir, &set, &zone, &keys, From::Outputs).unwrap();
        let mut all = Difference::default();
        for ((_, a), (_, b)) in from_files.iter().zip(&from_outputs) {
            all.add(difference(a, b));
        }
        assert_eq!(all.unlike, 0);
        assert!(all.values > 0 && all.largest < 0.01, "{all:?}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_unchecked_key_is_left_out_and_a_stop_ends_the_batch() {
        let (dir, mut set) = bench("unchecked");
        set.zones[0].vf.refs = vec!["clean".into()];
        consensus::write(&dir, &mut set.zones[0], Side::Vf).unwrap();
        set.checked.insert("leaky".into(), false);
        let zone = set.zones[0].clone();
        // The variant goes with its model.
        assert_eq!(keys_of(&dir, &set, &zone), ["clean"]);

        let stop = AtomicBool::new(true);
        let done = batch(&dir, &mut set, &["zone".into()], &stop, |_| {}).unwrap();
        assert!(!done);
        assert!(set.measures.is_empty());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_difference_counts_what_is_unlike() {
        let mut a = Measure::default();
        a.stems.insert(
            Stem::VoiceVf,
            Metric {
                si_sdr: Some(10.0),
                gap: Some(1.0),
                ..Metric::default()
            },
        );
        let mut b = a.clone();
        assert_eq!(difference(&a, &b).unlike, 0);
        assert_eq!(difference(&a, &b).values, 2);
        let m = b.stems.get_mut(&Stem::VoiceVf).unwrap();
        m.si_sdr = Some(10.5);
        m.gap = None;
        m.lag = 2;
        let d = difference(&a, &b);
        assert_eq!((d.largest, d.unlike, d.values), (0.5, 2, 1));
        b.stems.insert(Stem::RestVf, Metric::default());
        assert_eq!(difference(&a, &b).unlike, 3);
    }
}
