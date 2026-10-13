//! The batch: every checked model on every zone that lacks its outputs,
//! then the measures.
//!
//! One model at a time, loaded once and run over the zones that need it —
//! loading a checkpoint costs more than separating a zone. The set's
//! loudness files are written as it goes, so a batch that is stopped, or
//! that fails on one model, keeps what it made. It stops between two zones:
//! a zone is a few seconds of work, not worth interrupting.

use crate::outputs::{self, Separate, Trajectory};
use crate::set::{Set, Zone};
use crate::{Result, keys, layout, measure, variants};
use omnifrog_core::config::Config;
use omnifrog_core::models::{self, Catalogue};
use omnifrog_core::progress;
use omnifrog_core::worker::Worker;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// A model of the catalogue, as far as the batch needs to know it.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub key: String,
    pub name: String,
    /// The rate the model is told its input is at.
    pub rate: u32,
}

/// What a batch says as it goes.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// A model is being loaded, to run over this many zones; its place
    /// among the models that have work, out of how many.
    Model {
        key: String,
        zones: usize,
        index: usize,
        total: usize,
    },
    /// The model in hand is this far into a zone.
    Zone {
        key: String,
        zone: String,
        done: f64,
    },
    Separated {
        key: String,
        zone: String,
        seconds: f64,
    },
    /// A variant was computed from a model's outputs: its key, and the
    /// energy taken out of the dub's voice, in dB below it.
    Variant {
        key: String,
        zone: String,
        removed_db: f64,
    },
    /// A model could not run; the batch goes on with the next.
    Failed {
        key: String,
        why: String,
    },
    /// Something the worker said that is worth the log.
    Note(String),
    Measure(measure::Event),
}

/// The models of the catalogue, as a batch takes them.
pub fn models_of(catalogue: &Catalogue) -> Vec<Model> {
    catalogue
        .models
        .iter()
        .map(|m| Model {
            key: m.key.clone(),
            name: m.name.clone(),
            rate: m.rate,
        })
        .collect()
}

/// Those of the models the set has checked for the next batch.
pub fn checked(set: &Set, models: &[Model]) -> Vec<Model> {
    models
        .iter()
        .filter(|model| set.is_checked(&model.key))
        .cloned()
        .collect()
}

/// What loads a model for the real thing: its files fetched if they are
/// not there, then the core's worker — and with it the lock that keeps two
/// separations off the same device.
pub fn worker<'a>(
    config: &'a Config,
    catalogue: &'a Catalogue,
) -> impl FnMut(&Model, &dyn Fn(Event)) -> Result<Box<dyn Separate>> + 'a {
    move |model, event| {
        let model = catalogue.get(&model.key)?;
        // The core reports from its own threads; what it says is passed on
        // once the model is loaded.
        let said = Mutex::new(Vec::new());
        let progress = |e: progress::Event| {
            if let progress::Event::Message { text } = e {
                said.lock().unwrap_or_else(|e| e.into_inner()).push(text);
            }
        };
        let started = models::fetch(model, config, &progress)
            .and_then(|local| Worker::start(config, model, &local, &progress));
        for text in said.into_inner().unwrap_or_else(|e| e.into_inner()) {
            event(Event::Note(text));
        }
        Ok(Box::new(started?))
    }
}

/// A zone to separate, and where its element of the original is over it
/// when that is known.
#[derive(Debug, Clone, Copy)]
pub struct Target<'a> {
    pub zone: &'a Zone,
    pub trajectory: &'a Trajectory,
}

/// The models with work to do, each with the targets — by their place — it
/// has no outputs on.
pub fn plan<'a>(
    dir: &Path,
    targets: &[Target],
    models: &'a [Model],
) -> Vec<(&'a Model, Vec<usize>)> {
    models
        .iter()
        .filter(|model| !keys::is_variant(&model.key))
        .filter_map(|model| {
            let missing: Vec<usize> = (0..targets.len())
                .filter(|&i| !layout::outputs_present(dir, targets[i].zone, &model.key))
                .collect();
            (!missing.is_empty()).then_some((model, missing))
        })
        .collect()
}

/// What loads a model: given the model and where to say what happens, a
/// separator with that model in it.
pub type Start<'a> = dyn FnMut(&Model, &dyn Fn(Event)) -> Result<Box<dyn Separate>> + 'a;

/// What each separation took, in seconds, by zone and key.
pub type Seconds = BTreeMap<(String, String), f64>;

/// Separate what the given models lack on the targets. `start` loads a
/// model; a model that does not load, or fails on a zone, is said and left,
/// and the batch goes on. Whether it ran to the end, and what it took.
pub fn separate(
    dir: &Path,
    targets: &[Target],
    models: &[Model],
    start: &mut Start,
    stop: &AtomicBool,
    event: &dyn Fn(Event),
) -> Result<(bool, Seconds)> {
    let work = plan(dir, targets, models);
    let total = work.len();
    let mut seconds = Seconds::new();
    for (index, (model, missing)) in work.into_iter().enumerate() {
        if stop.load(Ordering::Relaxed) {
            return Ok((false, seconds));
        }
        event(Event::Model {
            key: model.key.clone(),
            zones: missing.len(),
            index,
            total,
        });
        let mut separator = match start(model, event) {
            Ok(separator) => separator,
            Err(why) => {
                event(Event::Failed {
                    key: model.key.clone(),
                    why: why.to_string(),
                });
                continue;
            }
        };
        for target in missing.into_iter().map(|i| targets[i]) {
            if stop.load(Ordering::Relaxed) {
                return Ok((false, seconds));
            }
            let id = &target.zone.id;
            let started = Instant::now();
            let made = outputs::make(
                dir,
                target.zone,
                &model.key,
                model.rate,
                target.trajectory,
                separator.as_mut(),
                &mut |done| {
                    event(Event::Zone {
                        key: model.key.clone(),
                        zone: id.clone(),
                        done,
                    });
                },
            );
            match made {
                Ok(_) => {
                    let took = started.elapsed().as_secs_f64();
                    seconds.insert((id.clone(), model.key.clone()), took);
                    event(Event::Separated {
                        key: model.key.clone(),
                        zone: id.clone(),
                        seconds: took,
                    });
                }
                Err(why) => {
                    event(Event::Failed {
                        key: model.key.clone(),
                        why: format!("{id}: {why}"),
                    });
                    // A worker that failed may be gone: the next model
                    // starts its own.
                    break;
                }
            }
        }
    }
    Ok((true, seconds))
}

/// Run the cascades taken on the zones where their first model has its
/// outputs and they have none: one load per second model, shared by every
/// cascade that ends with it. `catalogue` is every model, checked or not —
/// a cascade may end with one that is not separated on its own. What each
/// took, and whether it ran to the end.
#[allow(clippy::too_many_arguments)]
pub fn cascades(
    dir: &Path,
    set: &Set,
    zones: &[Zone],
    catalogue: &[Model],
    taken: &dyn Fn(&str) -> bool,
    start: &mut Start,
    stop: &AtomicBool,
    event: &dyn Fn(Event),
) -> Result<(bool, Seconds)> {
    let mut seconds = Seconds::new();
    // The work, by second model: (cascade, zone).
    let mut work: Vec<(&Model, Vec<(&crate::set::Cascade, &Zone)>)> = Vec::new();
    for cascade in set.cascades.iter().filter(|c| taken(&c.key())) {
        let Some(second) = catalogue.iter().find(|m| m.key == cascade.second) else {
            event(Event::Failed {
                key: cascade.key(),
                why: format!("no model `{}` in the catalogue", cascade.second),
            });
            continue;
        };
        let todo: Vec<(&crate::set::Cascade, &Zone)> = zones
            .iter()
            .filter(|zone| {
                layout::outputs_present(dir, zone, &cascade.first)
                    && !layout::outputs_present(dir, zone, &cascade.key())
            })
            .map(|zone| (cascade, zone))
            .collect();
        if todo.is_empty() {
            continue;
        }
        match work.iter_mut().find(|(model, _)| model.key == second.key) {
            Some((_, list)) => list.extend(todo),
            None => work.push((second, todo)),
        }
    }
    let total = work.len();
    for (index, (model, todo)) in work.into_iter().enumerate() {
        if stop.load(Ordering::Relaxed) {
            return Ok((false, seconds));
        }
        event(Event::Model {
            key: model.key.clone(),
            zones: todo.len(),
            index,
            total,
        });
        let mut separator = match start(model, event) {
            Ok(separator) => separator,
            Err(why) => {
                event(Event::Failed {
                    key: model.key.clone(),
                    why: why.to_string(),
                });
                continue;
            }
        };
        for (cascade, zone) in todo {
            if stop.load(Ordering::Relaxed) {
                return Ok((false, seconds));
            }
            let key = cascade.key();
            let started = Instant::now();
            let made = outputs::make_cascade(
                dir,
                zone,
                cascade,
                model.rate,
                separator.as_mut(),
                &mut |done| {
                    event(Event::Zone {
                        key: key.clone(),
                        zone: zone.id.clone(),
                        done,
                    });
                },
            );
            match made {
                Ok(_) => {
                    let took = started.elapsed().as_secs_f64();
                    seconds.insert((zone.id.clone(), key.clone()), took);
                    event(Event::Separated {
                        key,
                        zone: zone.id.clone(),
                        seconds: took,
                    });
                }
                Err(why) => {
                    event(Event::Failed {
                        key,
                        why: format!("{}: {why}", zone.id),
                    });
                    break;
                }
            }
        }
    }
    Ok((true, seconds))
}

/// Compute the variants the set asks for that are not there yet, on the
/// zones that have what they take: for every output `bases` takes, each
/// suffix of [`crate::set::Variants`] whose key is checked. A variant that
/// fails is said and left. What each took, and whether it ran to the end.
pub fn variants(
    dir: &Path,
    set: &Set,
    zones: &[Zone],
    bases: &dyn Fn(&str) -> bool,
    stop: &AtomicBool,
    event: &dyn Fn(Event),
) -> (bool, Seconds) {
    let mut seconds = Seconds::new();
    let suffixes = set.variants.suffixes();
    for zone in zones.iter().filter(|zone| variants::possible(dir, zone)) {
        for base in layout::keys_on_disk(dir, zone) {
            if keys::is_variant(&base)
                || !bases(&base)
                || !layout::outputs_present(dir, zone, &base)
            {
                continue;
            }
            for suffix in &suffixes {
                let key = format!("{base}{suffix}");
                if !set.is_checked(&key) || layout::outputs_present(dir, zone, &key) {
                    continue;
                }
                if stop.load(Ordering::Relaxed) {
                    return (false, seconds);
                }
                let started = Instant::now();
                match variants::make(dir, zone, &base, suffix) {
                    Ok((_, removed_db)) => {
                        seconds.insert(
                            (zone.id.clone(), key.clone()),
                            started.elapsed().as_secs_f64(),
                        );
                        event(Event::Variant {
                            key,
                            zone: zone.id.clone(),
                            removed_db,
                        });
                    }
                    Err(why) => event(Event::Failed {
                        key,
                        why: format!("{}: {why}", zone.id),
                    }),
                }
            }
        }
    }
    (true, seconds)
}

/// Separate what the given models lack on zones of the set, run the
/// cascades and compute the variants of the outputs `bases` takes, then
/// measure every checked output of those zones. `catalogue` is every model,
/// for the second stage of a cascade. Whether it ran to the end.
#[allow(clippy::too_many_arguments)]
pub fn compute(
    dir: &Path,
    set: &mut Set,
    zones: &[String],
    models: &[Model],
    catalogue: &[Model],
    bases: &dyn Fn(&str) -> bool,
    start: &mut Start,
    stop: &AtomicBool,
    event: &dyn Fn(Event),
) -> Result<bool> {
    let chosen: Vec<Zone> = zones
        .iter()
        .filter_map(|id| set.zone(id).cloned())
        .collect();
    let targets: Vec<Target> = chosen
        .iter()
        .map(|zone| Target {
            zone,
            trajectory: &[],
        })
        .collect();
    let (mut done, mut seconds) = separate(dir, &targets, models, start, stop, event)?;
    if done {
        let (finished, took) = cascades(dir, set, &chosen, catalogue, bases, start, stop, event)?;
        done = finished;
        seconds.extend(took);
    }
    if done {
        let (finished, took) = variants(dir, set, &chosen, bases, stop, event);
        done = finished;
        seconds.extend(took);
    }
    // The time an output took goes into the set before the measure reads
    // it back.
    for ((zone, key), took) in seconds {
        set.measures
            .entry(zone)
            .or_default()
            .entry(key)
            .or_default()
            .seconds = took;
    }
    if !done {
        set.save(dir)?;
        return Ok(false);
    }
    measure::batch(dir, set, zones, stop, |e| event(Event::Measure(e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outputs::Split;
    use crate::outputs::tests::bench;
    use crate::set::{Side, Stem};
    use crate::{Error, consensus};
    use std::cell::RefCell;
    use std::fs;

    fn models() -> Vec<Model> {
        ["a", "b", "c"]
            .iter()
            .map(|key| Model {
                key: key.to_string(),
                name: key.to_uppercase(),
                rate: 44_100,
            })
            .collect()
    }

    #[test]
    fn what_is_missing_is_separated_and_everything_measured() {
        let (dir, zone) = bench("compute");
        let mut set = Set::default();
        set.zones.push(zone);
        set.checked.insert("c".into(), false);
        let stop = AtomicBool::new(false);
        let events = RefCell::new(Vec::new());
        let mut loaded = Vec::new();
        let mut start = |model: &Model, _: &dyn Fn(Event)| -> Result<Box<dyn Separate>> {
            loaded.push(model.key.clone());
            if model.key == "b" {
                return Err(Error::msg("no checkpoint"));
            }
            Ok(Box::new(Split {
                voice: 0.5,
                off_by: 0,
            }))
        };
        let zones = ["zone".to_string()];
        let taken = checked(&set, &models());
        let done = compute(
            &dir,
            &mut set,
            &zones,
            &taken,
            &models(),
            &|_| true,
            &mut start,
            &stop,
            &|e| events.borrow_mut().push(e),
        )
        .unwrap();
        assert!(done);
        // The unchecked model is not loaded; the one that fails is said.
        assert_eq!(loaded, ["a", "b"]);
        let events = events.into_inner();
        assert!(events.contains(&Event::Failed {
            key: "b".into(),
            why: "no checkpoint".into()
        }));
        assert!(
            matches!(&events[0], Event::Model { key, zones: 1, index: 0, total: 2 } if key == "a")
        );
        let zone = set.zones[0].clone();
        assert!(layout::outputs_present(&dir, &zone, "a"));
        assert!(!layout::outputs_present(&dir, &zone, "b"));

        // With a reference, a second batch separates nothing and measures;
        // the time of the first separation is kept.
        set.zones[0].vf.refs = vec!["a".into()];
        consensus::write(&dir, &mut set.zones[0], Side::Vf).unwrap();
        fs::remove_dir_all(layout::output_dir(&dir, &zone, "b")).ok();
        let mut none = |model: &Model, _: &dyn Fn(Event)| -> Result<Box<dyn Separate>> {
            assert_eq!(model.key, "b", "only what is missing is loaded");
            Err(Error::msg("still no checkpoint"))
        };
        compute(
            &dir,
            &mut set,
            &zones,
            &taken,
            &models(),
            &|_| true,
            &mut none,
            &stop,
            &|_| {},
        )
        .unwrap();
        let measure = set.measure("zone", "a").unwrap();
        assert!(measure.seconds > 0.0);
        assert!(measure.stems[&Stem::VoiceVf].reference);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_stop_ends_the_batch_before_the_next_model() {
        let (dir, zone) = bench("stop");
        let mut set = Set::default();
        set.zones.push(zone);
        let stop = AtomicBool::new(true);
        let mut start = |_: &Model, _: &dyn Fn(Event)| -> Result<Box<dyn Separate>> {
            panic!("nothing is loaded once stopped")
        };
        let zones = ["zone".to_string()];
        let done = compute(
            &dir,
            &mut set,
            &zones,
            &models(),
            &models(),
            &|_| true,
            &mut start,
            &stop,
            &|_| {},
        )
        .unwrap();
        assert!(!done);
        let target = Target {
            zone: &set.zones[0],
            trajectory: &[],
        };
        assert_eq!(plan(&dir, &[target], &models()).len(), 3);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn cascades_run_after_their_first_model_one_load_per_second_model() {
        use crate::set::{Cascade, CascadeMode};
        let (dir, zone) = bench("cascades");
        let mut set = Set::default();
        set.zones.push(zone);
        // Two cascades end with `b`, one starts with a model that is not
        // separated here, one is unchecked.
        for (first, second, mode) in [
            ("a", "b", CascadeMode::Rest),
            ("a", "b", CascadeMode::Voice),
            ("c", "a", CascadeMode::Rest),
            ("a", "c", CascadeMode::Rest),
        ] {
            assert!(set.add_cascade(Cascade {
                first: first.into(),
                second: second.into(),
                mode,
                kept: false,
            }));
        }
        assert!(!set.add_cascade(Cascade {
            first: "a".into(),
            second: "b".into(),
            mode: CascadeMode::Rest,
            kept: true,
        }));
        set.checked.insert("a~c~rest".into(), false);
        let stop = AtomicBool::new(false);
        let loaded = RefCell::new(Vec::new());
        let mut start = |model: &Model, _: &dyn Fn(Event)| -> Result<Box<dyn Separate>> {
            loaded.borrow_mut().push(model.key.clone());
            Ok(Box::new(Split {
                voice: 0.5,
                off_by: 0,
            }))
        };
        let zones = ["zone".to_string()];
        let only_a: Vec<Model> = models().into_iter().filter(|m| m.key == "a").collect();
        let checked = set.checked.clone();
        let bases = move |key: &str| checked.get(key).copied().unwrap_or(true);
        let done = compute(
            &dir,
            &mut set,
            &zones,
            &only_a,
            &models(),
            &bases,
            &mut start,
            &stop,
            &|_| {},
        )
        .unwrap();
        assert!(done);
        // `a` for itself, then `b` once for the two cascades that end with
        // it; nothing for the cascade whose first model has no outputs, nor
        // for the unchecked one.
        assert_eq!(*loaded.borrow(), ["a", "b"]);
        let zone = set.zones[0].clone();
        assert!(layout::outputs_present(&dir, &zone, "a~b~rest"));
        assert!(layout::outputs_present(&dir, &zone, "a~b~voice"));
        assert!(!layout::output_dir(&dir, &zone, "c~a~rest").exists());
        assert!(!layout::output_dir(&dir, &zone, "a~c~rest").exists());
        // Again: nothing is loaded, everything is there.
        loaded.borrow_mut().clear();
        compute(
            &dir,
            &mut set,
            &zones,
            &only_a,
            &models(),
            &bases,
            &mut start,
            &stop,
            &|_| {},
        )
        .unwrap();
        assert!(loaded.borrow().is_empty());
        assert!(set.remove_cascade("a~b~rest"));
        assert!(!set.remove_cascade("a~b~rest"));
        assert_eq!(set.cascades.len(), 3);
        fs::remove_dir_all(&dir).unwrap();
    }
}
