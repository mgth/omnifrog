//! A film in the bench: its extraction, kept, and the ranges tried on it.
//!
//! ```text
//! _bench/films/<film>/
//!   film.json                 what was extracted, from where
//!   1.extract/                the whole film, as a job extracts it
//!   trials/o<object>_c<channel>_<start>_<end>/
//!     1.extract/  outputs/    a range of one element and one channel
//! ```
//!
//! A film is extracted whole, once: every element of its original and every
//! channel of its dub, exactly what a run reads. A *trial* is a range of
//! it — one element, one channel — laid out as a zone, so everything that
//! works on a zone works on it: the models are run on it, their outputs
//! listened to. A zone is then cut out of a trial, outputs and all: a few
//! seconds of a longer range the models heard whole, which is how a short
//! zone gets outputs worth measuring.

use crate::layout::{self, Loudness};
use crate::outputs;
use crate::set::{DuplicatesContext, MAX_SECONDS, RATE, Set, Side, Stem, Step, Zone};
use crate::{Error, Result, audio};
use omnifrog_core::config::Config;
use omnifrog_core::master::{MasterSet, Trajectories};
use omnifrog_core::progress::Progress;
use omnifrog_core::{duplicates, extract, probe, tracks};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const FILMS_DIR: &str = "films";
pub const TRIALS_DIR: &str = "trials";
pub const FILE: &str = "film.json";

/// An element of the original's scene.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Element {
    pub id: u32,
    /// The speaker of a bed channel; an object's own label.
    pub name: String,
    pub object: bool,
}

/// What `film.json` keeps of an extraction.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Film {
    /// The name of its directory.
    pub name: String,
    /// The file it was extracted from.
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lang_vo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lang_vf: Option<String>,
    /// The two tracks, counted among the audio tracks from 1.
    pub original: u32,
    pub dub: u32,
    pub elements: Vec<Element>,
    pub channels: Vec<String>,
    /// Frames every file of the extraction holds.
    pub frames: u64,
    /// When each of the two tracks starts in the source, in seconds: the
    /// original's, the dub's. They need not start together, and by how
    /// much the dub leads is where its voice goes in a mix. None for a
    /// film extracted before this was kept; extracting it again fills it
    /// in without reading the film.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub starts: Option<[f64; 2]>,
    /// When it was extracted, RFC 3339.
    pub date: String,
}

/// A name a directory can have: letters, digits, `-`, `.` and `_`; anything
/// else becomes `_`.
pub fn clean(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '-' | '.' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

impl Film {
    /// Write `film.json`.
    pub fn save(&self, dir: &Path) -> Result<()> {
        let path = self.dir(dir).join(FILE);
        let text = serde_json::to_string_pretty(self)
            .map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
        fs::write(&path, text).map_err(|e| Error::io(&path, e))
    }

    pub fn dir(&self, dir: &Path) -> PathBuf {
        dir.join(FILMS_DIR).join(&self.name)
    }

    pub fn extract_dir(&self, dir: &Path) -> PathBuf {
        self.dir(dir).join(layout::EXTRACT_DIR)
    }

    /// Whether everything the extraction wrote is still there.
    pub fn complete(&self, dir: &Path) -> bool {
        let ids: Vec<u32> = self.elements.iter().map(|e| e.id).collect();
        extract::complete(&self.extract_dir(dir), &ids, &self.channels)
    }

    /// Bytes the film takes in the bench, trials included.
    pub fn size(&self, dir: &Path) -> u64 {
        fn of(path: &Path) -> u64 {
            let Ok(entries) = fs::read_dir(path) else {
                return 0;
            };
            entries
                .flatten()
                .map(|entry| match entry.metadata() {
                    Ok(meta) if meta.is_dir() => of(&entry.path()),
                    Ok(meta) => meta.len(),
                    Err(_) => 0,
                })
                .sum()
        }
        of(&self.dir(dir))
    }

    /// Where an object of the film is over a range, in frames of the range:
    /// what weights the loudness of its voice. Empty for a bed channel, and
    /// when the scene's metadata does not read.
    pub fn trajectory(
        &self,
        dir: &Path,
        object: u32,
        start: u64,
        end: u64,
    ) -> Vec<(u64, [f64; 3])> {
        let Ok(master) = MasterSet::find(&self.extract_dir(dir)) else {
            return Vec::new();
        };
        let Ok(trajectories) = Trajectories::read(&master.metadata) else {
            return Vec::new();
        };
        let all = trajectories.of(object);
        // Where it is as the range starts, then every move within it.
        let before = all.iter().rev().find(|(at, _)| *at <= start);
        before
            .map(|(_, position)| (0, *position))
            .into_iter()
            .chain(
                all.iter()
                    .filter(|(at, _)| *at > start && *at < end)
                    .map(|(at, position)| (at - start, *position)),
            )
            .collect()
    }
}

/// The films extracted in the bench, by name.
pub fn films(dir: &Path) -> Vec<Film> {
    let Ok(entries) = fs::read_dir(dir.join(FILMS_DIR)) else {
        return Vec::new();
    };
    let mut films: Vec<Film> = entries
        .flatten()
        .filter_map(|entry| {
            let text = fs::read_to_string(entry.path().join(FILE)).ok()?;
            serde_json::from_str(&text).ok()
        })
        .collect();
    films.sort_by(|a, b| a.name.cmp(&b.name));
    films
}

/// Remove a film from the bench: its extraction and its trials. The zones
/// cut from it are their own files, and stay.
pub fn remove(dir: &Path, film: &Film) -> Result<()> {
    let path = film.dir(dir);
    if film.name.is_empty() || !path.join(FILE).is_file() {
        return Err(Error::msg(format!(
            "{}: not a film of the bench",
            path.display()
        )));
    }
    fs::remove_dir_all(&path).map_err(|e| Error::io(&path, e))
}

/// Extract a film whole into the bench, or find it there already. `original`
/// and `dub` name the two tracks among the audio tracks, from 1; without
/// them, the ones a run would take.
pub fn extract(
    config: &Config,
    dir: &Path,
    source: &Path,
    original: Option<u32>,
    dub: Option<u32>,
    progress: Progress,
) -> Result<Film> {
    let media = probe::media(config, source, progress)?;
    let original = match original {
        Some(index) => media.track(index)?,
        None => tracks::original(&media)?,
    };
    let dub = match dub {
        Some(index) => media.track(index)?,
        None => tracks::dub(&media)?,
    };
    let name: String = clean(&media.stem()).chars().take(80).collect();
    let known = films(dir).into_iter().find(|film| film.name == name);
    if let Some(mut film) = known
        && film.original == original.index
        && film.dub == dub.index
        && film.complete(dir)
    {
        if film.starts.is_none() {
            film.starts = Some([original.start, dub.start]);
            film.save(dir)?;
        }
        return Ok(film);
    }
    let mut film = Film {
        name,
        source: source.to_string_lossy().into_owned(),
        lang_vo: original.language.clone(),
        lang_vf: dub.language.clone(),
        original: original.index,
        dub: dub.index,
        starts: Some([original.start, dub.start]),
        ..Film::default()
    };
    // Another extraction of the same film: the ranges tried on the last one
    // are not ranges of this one.
    forget_trials(dir, &film)?;
    let into = film.extract_dir(dir);
    fs::create_dir_all(&into).map_err(|e| Error::io(&into, e))?;
    let extracted = extract::run(
        config,
        &extract::Plan {
            source,
            original,
            dub,
            dir: &into,
            duration: media.duration,
            seconds: None,
            start: None,
        },
        progress,
    )?;
    film.elements = extracted
        .master
        .channels
        .iter()
        .map(|c| Element {
            id: c.id,
            name: c.name.clone(),
            object: c.object,
        })
        .collect();
    film.channels = extracted.dub.iter().map(|(name, _)| name.clone()).collect();
    film.frames = extracted.frames;
    film.date = omnifrog_core::tags::utc_now();
    film.save(dir)?;
    Ok(film)
}

/// What a zone of a film needs to have its duplicates cancelled, measured
/// while the film's extraction is there: the map of the dub around the
/// zone, how far the dub leads the original, and — written beside the
/// zone's inputs — the sum of the other elements of the original over it.
///
/// Those other elements are taken as decoded: the bench has no separation
/// of them, as a run has when it reaches its own cancellation. They carry
/// some voice a run would have removed first — reverberation, secondary
/// lines — so a little more is taken out here than a run would take.
pub fn context(dir: &Path, film: &Film, zone: &Zone) -> Result<DuplicatesContext> {
    let from = film.extract_dir(dir);
    let lfe = |name: &str| name.eq_ignore_ascii_case("LFE");
    let elements: Vec<&Element> = film.elements.iter().filter(|e| !lfe(&e.name)).collect();
    let original: Vec<PathBuf> = elements
        .iter()
        .map(|e| from.join(format!("{}{}.wav", extract::ORIGINAL, e.id)))
        .collect();
    let dub: Vec<PathBuf> = film
        .channels
        .iter()
        .filter(|name| !lfe(name))
        .map(|name| from.join(format!("{}{name}.wav", extract::DUB)))
        .collect();
    let end = zone.end.unwrap_or(film.frames);
    let steps = duplicates::steps_around(&original, &dub, zone.start as i64, end as i64)?;
    let others: Vec<(PathBuf, i64)> = elements
        .iter()
        .zip(&original)
        .filter(|(e, _)| e.id != zone.object)
        .map(|(_, path)| (path.clone(), zone.start as i64))
        .collect();
    duplicates::write_sum(&others, end - zone.start, &layout::others(dir, zone))?;
    Ok(DuplicatesContext {
        dub_lead_ms: film
            .starts
            .map_or(0.0, |[original, dub]| (original - dub) * 1000.0),
        steps: steps
            .into_iter()
            .map(|(centre, offset)| Step { centre, offset })
            .collect(),
        others: format!(
            "{} other element(s) of the original, as decoded{}",
            others.len(),
            match film.starts {
                Some(_) => "",
                None => "; the tracks' offset is not known for this film: taken as 0",
            }
        ),
    })
}

/// Remove the ranges tried on a film, with what the models made of them.
///
/// A trial is found again by its film, element, channel and range, and its
/// inputs are kept when they are the right length. Extracted again with
/// another track for the dub, the film has the same names and the same
/// lengths: without this, the range would still be the old dub's, and its
/// outputs with it, under the new extraction's description.
pub fn forget_trials(dir: &Path, film: &Film) -> Result<()> {
    let trials = film.dir(dir).join(TRIALS_DIR);
    match fs::remove_dir_all(&trials) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(Error::io(&trials, e)),
        _ => Ok(()),
    }
}

/// A range of a film — one element of its original, one channel of its
/// dub — laid out as a zone that is not in the set. Its two inputs are cut
/// from the film's extraction if they are not there yet.
pub fn trial(
    dir: &Path,
    film: &Film,
    object: u32,
    channel: &str,
    start: u64,
    end: u64,
) -> Result<Zone> {
    if !film.elements.iter().any(|e| e.id == object) {
        return Err(Error::msg(format!("{}: no element {object}", film.name)));
    }
    if !film.channels.iter().any(|c| c == channel) {
        return Err(Error::msg(format!(
            "{}: no channel {channel} in its dub",
            film.name
        )));
    }
    let end = end.min(film.frames);
    if end <= start {
        return Err(Error::msg(format!(
            "an empty range (the film is {} long)",
            crate::set::clock(film.frames)
        )));
    }
    let zone = Zone {
        // A path under the bench directory, as a zone's id is: everything
        // that finds a zone's files finds a trial's.
        id: format!(
            "{FILMS_DIR}/{}/{TRIALS_DIR}/o{object}_c{channel}_{start}_{end}",
            film.name
        ),
        film: film.name.clone(),
        source: film.source.clone(),
        lang_vo: film.lang_vo.clone(),
        lang_vf: film.lang_vf.clone(),
        object,
        channel: channel.to_string(),
        start,
        end: Some(end),
        ..Zone::default()
    };
    let from = film.extract_dir(dir);
    for (side, name) in [
        (Side::Vo, format!("{}{object}.wav", extract::ORIGINAL)),
        (Side::Vf, format!("{}{channel}.wav", extract::DUB)),
    ] {
        let destination = layout::extract(dir, &zone, side);
        if omnifrog_core::wav::frames(&destination) != end - start {
            audio::cut(&from.join(name), &destination, start, end - start)?;
        }
    }
    Ok(zone)
}

/// Whether a zone is a trial rather than one of the set.
pub fn is_trial(zone: &Zone) -> bool {
    zone.id.starts_with(FILMS_DIR) && zone.id.contains(TRIALS_DIR)
}

/// Add a zone to the set, cut from a trial: `range` in frames of the trial,
/// or all of it. The trial's inputs and every output it has are cut to the
/// range, the loudness measured again on what was cut; nothing is rated
/// yet. The zone's id is `<film>_o<object>_c<channel>_<start in s>`, with a
/// number after it when that is taken.
pub fn add_zone(
    dir: &Path,
    set: &mut Set,
    trial: &Zone,
    range: Option<(u64, u64)>,
    kind: &str,
) -> Result<String> {
    let length = trial.end.unwrap_or(trial.start).saturating_sub(trial.start);
    let (from, to) = match range {
        Some((a, b)) => (a.min(length), b.min(length)),
        None => (0, length),
    };
    if to <= from {
        return Err(Error::msg("an empty range"));
    }
    if to - from > MAX_SECONDS as u64 * RATE as u64 {
        return Err(Error::msg(format!(
            "a zone is {MAX_SECONDS} s at most: it is measured whole, in memory"
        )));
    }
    let film: String = clean(&trial.film).chars().take(40).collect();
    let start = trial.start + from;
    let base = format!(
        "{film}_o{}_c{}_{}",
        trial.object,
        trial.channel,
        (start as f64 / RATE as f64).round() as u64
    );
    let mut id = base.clone();
    let mut n = 2;
    while dir.join(&id).exists() || set.zone(&id).is_some() {
        id = format!("{base}_{n}");
        n += 1;
    }
    let mut zone = Zone {
        id: id.clone(),
        start,
        end: Some(trial.start + to),
        kind: kind.to_string(),
        date: omnifrog_core::tags::utc_now(),
        ..trial.clone()
    };
    let made = (|| -> Result<()> {
        for side in Side::BOTH {
            let source = layout::extract(dir, trial, side);
            audio::cut(&source, &layout::extract(dir, &zone, side), from, to - from)?;
        }
        for key in layout::keys_on_disk(dir, trial) {
            let mut any = false;
            let files = Stem::ALL
                .iter()
                .map(|&stem| {
                    (
                        layout::output(dir, trial, &key, stem),
                        layout::output(dir, &zone, &key, stem),
                    )
                })
                .chain([(layout::mix(dir, trial, &key), layout::mix(dir, &zone, &key))]);
            for (source, destination) in files {
                if source.is_file() {
                    audio::cut(&source, &destination, from, to - from)?;
                    any = true;
                }
            }
            if any {
                // What was measured of the trial held over the whole of it.
                // The gain of its mix is another matter: the zone's mix is
                // a cut of the trial's, at the gain that one was made with.
                let mut loudness = outputs::loudness_of(dir, &zone, &key)?;
                if let Some(gain) = Loudness::load(dir, trial, &key).mix_gain() {
                    loudness.set_mix_gain(gain);
                }
                loudness.save(dir, &zone, &key)?;
            }
        }
        Ok(())
    })();
    if let Err(e) = made {
        // Half a zone would be found on disk and taken for a whole one.
        let _ = fs::remove_dir_all(layout::zone_dir(dir, &zone));
        return Err(e);
    }
    // What the duplicate variants take, measured now: the film may not be
    // in the bench for long. Without it the zone is a zone all the same,
    // only without those variants.
    let film = films(dir).into_iter().find(|film| film.name == trial.film);
    if let Some(film) = film.filter(|film| film.complete(dir)) {
        zone.duplicates = context(dir, &film, &zone).ok();
    }
    set.zones.push(zone);
    set.save(dir)?;
    Ok(id)
}

/// The trajectory a trial's element of the original has over it.
pub fn trajectory_of(dir: &Path, film: &Film, trial: &Zone) -> Vec<(u64, [f64; 3])> {
    film.trajectory(
        dir,
        trial.object,
        trial.start,
        trial.end.unwrap_or(film.frames),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outputs::tests::noise;
    use crate::outputs::{Separate, Split};

    /// A film of ten seconds as an extraction leaves it, without the scene's
    /// own files: two elements, one channel.
    fn bench(name: &str) -> (PathBuf, Film) {
        let dir =
            std::env::temp_dir().join(format!("omnifrog-bench-film-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let film = Film {
            name: "Some.Film".into(),
            source: "/films/Some Film.mkv".into(),
            lang_vo: Some("en".into()),
            lang_vf: Some("fr".into()),
            original: 1,
            dub: 2,
            elements: vec![
                Element {
                    id: 3,
                    name: "C".into(),
                    object: false,
                },
                Element {
                    id: 12,
                    name: "12".into(),
                    object: true,
                },
            ],
            channels: vec!["C".into()],
            frames: 10 * RATE as u64,
            starts: None,
            date: String::new(),
        };
        let into = film.extract_dir(&dir);
        let n = 10 * RATE as usize;
        audio::write(&into.join("vo_3.wav"), &noise(3, n, 0.25), RATE).unwrap();
        audio::write(&into.join("vo_12.wav"), &noise(12, n, 0.25), RATE).unwrap();
        audio::write(&into.join("vf_C.wav"), &noise(20, n, 0.125), RATE).unwrap();
        fs::write(
            film.dir(&dir).join(FILE),
            serde_json::to_string(&film).unwrap(),
        )
        .unwrap();
        (dir, film)
    }

    #[test]
    fn a_film_is_found_again_and_removed() {
        let (dir, film) = bench("found");
        assert_eq!(films(&dir), std::slice::from_ref(&film));
        assert!(film.size(&dir) > 3 * 10 * RATE as u64 * 3);
        assert!(remove(&dir, &Film::default()).is_err());
        remove(&dir, &film).unwrap();
        assert!(films(&dir).is_empty());
        assert_eq!(clean("Some Film: the (2nd)"), "Some_Film__the__2nd_");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_trial_is_a_range_of_the_film_laid_out_as_a_zone() {
        let (dir, film) = bench("trial");
        let (start, end) = (2 * RATE as u64, 6 * RATE as u64);
        let trial = trial(&dir, &film, 12, "C", start, end).unwrap();
        assert!(is_trial(&trial));
        assert_eq!((trial.start, trial.end), (start, Some(end)));
        let whole = audio::read(&film.extract_dir(&dir).join("vo_12.wav")).unwrap();
        let cut = audio::read(&layout::extract(&dir, &trial, Side::Vo)).unwrap();
        assert_eq!(cut, &whole[start as usize..end as usize]);
        assert_eq!(
            omnifrog_core::wav::frames(&layout::extract(&dir, &trial, Side::Vf)),
            end - start
        );
        // Opened again, the range is the same files, not cut again…
        let input = layout::extract(&dir, &trial, Side::Vf);
        fs::write(input.with_extension("mark"), "").unwrap();
        super::trial(&dir, &film, 12, "C", start, end).unwrap();
        assert!(input.with_extension("mark").is_file());
        // …until the film is extracted again: with another track for its
        // dub, the same names and the same lengths would be another sound.
        let other: Vec<f32> = noise(99, 10 * RATE as usize, 0.125);
        audio::write(&film.extract_dir(&dir).join("vf_C.wav"), &other, RATE).unwrap();
        forget_trials(&dir, &film).unwrap();
        forget_trials(&dir, &film).unwrap();
        let again = super::trial(&dir, &film, 12, "C", start, end).unwrap();
        let cut = audio::read(&layout::extract(&dir, &again, Side::Vf)).unwrap();
        let whole = audio::read(&film.extract_dir(&dir).join("vf_C.wav")).unwrap();
        assert!(cut == whole[start as usize..end as usize]);
        assert!(!input.with_extension("mark").exists());
        // Past the end of the film: cut there. No such element: said.
        let last = super::trial(&dir, &film, 3, "C", 9 * RATE as u64, 99 * RATE as u64).unwrap();
        assert_eq!(last.end, Some(film.frames));
        assert!(super::trial(&dir, &film, 4, "C", start, end).is_err());
        assert!(super::trial(&dir, &film, 12, "Ls", start, end).is_err());
        assert!(super::trial(&dir, &film, 12, "C", end, start).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_zone_is_cut_out_of_a_trial_with_its_outputs() {
        let (dir, film) = bench("zone");
        let trial = trial(&dir, &film, 12, "C", 2 * RATE as u64, 8 * RATE as u64).unwrap();
        let mut split = Split {
            voice: 0.5,
            off_by: 0,
        };
        let split: &mut dyn Separate = &mut split;
        outputs::make(&dir, &trial, "model", RATE, &[], split, &mut |_| {}).unwrap();

        let mut set = Set::default();
        let range = Some((RATE as u64, 3 * RATE as u64));
        let id = add_zone(&dir, &mut set, &trial, range, "Whisper").unwrap();
        assert_eq!(id, "Some.Film_o12_cC_3");
        let zone = set.zone(&id).unwrap().clone();
        assert!(!is_trial(&zone));
        assert_eq!(
            (zone.start, zone.end),
            (3 * RATE as u64, Some(5 * RATE as u64))
        );
        assert_eq!(zone.kind, "Whisper");
        assert!(zone.vo.refs.is_empty() && !zone.date.is_empty());
        assert_eq!(Set::load(&dir).unwrap().zones.len(), 1);

        // The zone's files are the trial's, from one second in, two long.
        let (a, b) = (RATE as usize, 3 * RATE as usize);
        let of_trial = audio::read(&layout::output(&dir, &trial, "model", Stem::RestVf)).unwrap();
        let of_zone = audio::read(&layout::output(&dir, &zone, "model", Stem::RestVf)).unwrap();
        assert_eq!(of_zone, &of_trial[a..b]);
        let input = audio::read(&layout::extract(&dir, &zone, Side::Vo)).unwrap();
        let whole = audio::read(&film.extract_dir(&dir).join("vo_12.wav")).unwrap();
        assert_eq!(input, &whole[3 * RATE as usize..5 * RATE as usize]);
        assert!(layout::outputs_present(&dir, &zone, "model"));
        assert!(layout::mix(&dir, &zone, "model").is_file());
        // The zone's mix is a cut of the range's: its gain is the range's.
        let of_range = Loudness::load(&dir, &trial, "model").mix_gain();
        assert!(of_range.is_some());
        assert_eq!(Loudness::load(&dir, &zone, "model").mix_gain(), of_range);
        assert!(
            Loudness::load(&dir, &zone, "model")
                .of(Stem::VoiceVo)
                .is_some()
        );

        // The same range again takes another name; the whole trial is too
        // long here to be refused, an empty range is.
        let again = add_zone(&dir, &mut set, &trial, range, "Whisper").unwrap();
        assert_eq!(again, "Some.Film_o12_cC_3_2");
        assert!(add_zone(&dir, &mut set, &trial, Some((5, 5)), "").is_err());
        let whole = add_zone(&dir, &mut set, &trial, None, "").unwrap();
        assert_eq!(whole, "Some.Film_o12_cC_2");
        assert_eq!(set.zones.len(), 3);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_zone_longer_than_the_bench_measures_is_refused() {
        let (dir, mut film) = bench("long");
        // A film said to be long enough; the range is checked before any
        // file is read.
        film.frames = 400 * RATE as u64;
        let long = Zone {
            id: format!("{FILMS_DIR}/{}/{TRIALS_DIR}/long", film.name),
            film: film.name.clone(),
            start: 0,
            end: Some(300 * RATE as u64),
            ..Zone::default()
        };
        let mut set = Set::default();
        let error = add_zone(&dir, &mut set, &long, None, "").unwrap_err();
        assert!(error.to_string().contains("120 s"), "{error}");
        assert!(set.zones.is_empty());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_zone_s_context_is_measured_on_the_film_around_it() {
        // Fifty seconds: two elements, and a dub whose one channel is their
        // sum, 96 samples late.
        let dir =
            std::env::temp_dir().join(format!("omnifrog-bench-context-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let n = 50 * RATE as usize;
        let (a, b) = (noise(12, n, 0.2), noise(3, n, 0.2));
        let late: Vec<f32> = (0..n)
            .map(|i| if i >= 96 { a[i - 96] + b[i - 96] } else { 0.0 })
            .collect();
        let film = Film {
            name: "Some.Film".into(),
            elements: vec![
                Element {
                    id: 1,
                    name: "LFE".into(),
                    object: false,
                },
                Element {
                    id: 3,
                    name: "C".into(),
                    object: false,
                },
                Element {
                    id: 12,
                    name: "12".into(),
                    object: true,
                },
            ],
            channels: vec!["C".into(), "LFE".into()],
            frames: n as u64,
            starts: Some([0.084, 0.006]),
            ..Film::default()
        };
        let into = film.extract_dir(&dir);
        audio::write(&into.join("vo_3.wav"), &b, RATE).unwrap();
        audio::write(&into.join("vo_12.wav"), &a, RATE).unwrap();
        audio::write(&into.join("vf_C.wav"), &late, RATE).unwrap();
        // The LFE of either track is never read: its files are not even there.
        let zone = Zone {
            id: "zone".into(),
            object: 12,
            channel: "C".into(),
            start: 20 * RATE as u64,
            end: Some(25 * RATE as u64),
            ..Zone::default()
        };
        let context = context(&dir, &film, &zone).unwrap();
        assert!((context.dub_lead_ms - 78.0).abs() < 1e-9);
        assert!(!context.steps.is_empty());
        assert!(
            context.steps.iter().all(|s| s.offset == 96),
            "{:?}",
            context.steps
        );
        // A step is told from the zone's start: one of them is at it or
        // within a step of it.
        assert!(
            context
                .steps
                .iter()
                .any(|s| s.centre.abs() <= 5 * RATE as i64)
        );
        assert!(context.others.starts_with("1 other element"));
        // The others are the other element, over the zone, as it is.
        let others = audio::read(&layout::others(&dir, &zone)).unwrap();
        assert_eq!(others.len(), 5 * RATE as usize);
        let step = 1.0 / 8_388_608.0f32;
        for i in [0, 1000, 239_999] {
            assert!((others[i] - b[20 * RATE as usize + i]).abs() <= step, "{i}");
        }
        assert!(crate::variants::possible(
            &dir,
            &Zone {
                duplicates: Some(context),
                ..zone
            }
        ));
        fs::remove_dir_all(&dir).unwrap();
    }
}
