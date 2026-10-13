//! A bench of the inference's device memory, for choosing
//! `inference.block_seconds` and `inference.batch_size` on a smaller card.
//!
//! One element is separated under several settings, each by a worker of its
//! own, exactly as a run would start it: what torch reserved on the device
//! at most, how long the loading and the separation took, and how far the
//! stems moved from the first setting's (the difference's energy under the
//! stem's, in dB). The first setting is the reference: pymss at the
//! configuration's overlap, with the default block and the model's batch.
//!
//! torch's figure leaves out the runtime's own context (about half a GB with
//! ROCm or CUDA) and whatever else the desktop holds: a card needs that much
//! more than the peak.

use crate::config::{Config, DEFAULT_BLOCK_SECONDS, Engine, Inference};
use crate::models::{self, Model};
use crate::progress::{Event, Progress};
use crate::worker::{Stems, Worker};
use crate::{Error, Result, wav};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// One way of running the models.
#[derive(Debug, Clone)]
pub struct Setting {
    pub label: String,
    pub inference: Inference,
}

/// The settings a bench tries: the reference first, then each block, each
/// batch (at the default block), a light overlap, and MSST.
pub fn settings(config: &Config, blocks: &[u32], batches: &[u32]) -> Vec<Setting> {
    let base = Inference {
        engine: Engine::Pymss,
        block_seconds: DEFAULT_BLOCK_SECONDS,
        batch_size: 0,
        ..config.inference
    };
    let mut list = vec![Setting {
        label: base.describe(),
        inference: base,
    }];
    let mut push = |inference: Inference| {
        let label = inference.describe();
        if !list.iter().any(|s| s.label == label) {
            list.push(Setting { label, inference });
        }
    };
    for &block_seconds in blocks {
        push(Inference {
            block_seconds,
            ..base
        });
    }
    for &batch_size in batches {
        push(Inference { batch_size, ..base });
    }
    push(Inference { overlap: 5, ..base });
    push(Inference::msst());
    list
}

/// One line of the result.
#[derive(Debug, Clone)]
pub struct Row {
    pub model: String,
    pub setting: String,
    /// GiB torch reserved at most; `None` on a CPU.
    pub peak_gib: Option<f64>,
    pub load_seconds: f64,
    pub seconds: f64,
    /// Difference from the reference's stems, in dB under them; `None` for
    /// the reference itself.
    pub voice_db: Option<f64>,
    pub rest_db: Option<f64>,
}

/// Separate the first `seconds` of `input` (a mono element, as a job's
/// extraction writes them) with each model under each setting, in `dir`,
/// which is emptied first and removed at the end unless `keep`.
#[allow(clippy::too_many_arguments)]
pub fn run(
    config: &Config,
    models: &[&Model],
    input: &Path,
    seconds: f64,
    settings: &[Setting],
    dir: &Path,
    keep: bool,
    progress: Progress,
) -> Result<Vec<Row>> {
    let format = wav::read_header(input)?;
    let frames = ((seconds * format.sample_rate as f64) as u64).min(format.frames);
    if frames == 0 {
        return Err(Error::msg(format!("{}: no samples", input.display())));
    }
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    let mut rows = Vec::new();
    for model in models {
        let local = models::fetch(model, config, progress)?;
        let here = dir.join(&model.key);
        fs::create_dir_all(&here).map_err(|e| Error::io(&here, e))?;
        // The no-resample trick, as a run hands an element over: the same
        // samples under the rate the model was trained at.
        let element = here.join("element.wav");
        wav::copy_mono(input, &element, frames, model.rate, 1.0, wav::Sample::Float)?;
        let mut reference: Option<Stems> = None;
        for (index, setting) in settings.iter().enumerate() {
            progress(Event::message(format!(
                "{} · {} ({}/{})",
                model.key,
                setting.label,
                index + 1,
                settings.len()
            )));
            let mut this = config.clone();
            this.inference = setting.inference;
            let out = here.join(format!("s{index}"));
            let started = Instant::now();
            let mut worker = Worker::start(&this, model, &local, progress)?;
            let load_seconds = started.elapsed().as_secs_f64();
            let started = Instant::now();
            let stems = worker.separate(std::slice::from_ref(&element), &out, |f| {
                progress(Event::ratio(
                    format!("{} · {}", model.key, setting.label),
                    f,
                ))
            })?;
            let seconds = started.elapsed().as_secs_f64();
            // pymss out of device memory: the element went through MSST, and
            // the row measures MSST's time and peak, not this setting's.
            for note in worker.take_notes() {
                progress(Event::message(format!("  {note}")));
            }
            let peak_gib = worker.device_peak();
            drop(worker);
            let stems = stems.into_iter().next().expect("one input, one pair");
            let (voice_db, rest_db) = match &reference {
                Some(first) => (
                    Some(difference_db(&first.voice, &stems.voice)?),
                    Some(difference_db(&first.rest, &stems.rest)?),
                ),
                None => (None, None),
            };
            let row = Row {
                model: model.key.clone(),
                setting: setting.label.clone(),
                peak_gib,
                load_seconds,
                seconds,
                voice_db,
                rest_db,
            };
            progress(Event::message(format!("  {}", line(&row))));
            rows.push(row);
            if reference.is_none() {
                reference = Some(stems);
            } else if !keep {
                // Two float stems of the element per setting: gigabytes
                // over a bench, and nothing reads them again.
                let _ = fs::remove_dir_all(&out);
            }
        }
    }
    if !keep {
        let _ = fs::remove_dir_all(dir);
    }
    Ok(rows)
}

/// One row as the table prints it.
pub fn line(row: &Row) -> String {
    let db = |v: Option<f64>| v.map_or("ref".to_string(), |v| format!("{v:.1}"));
    format!(
        "{:6} {:48} {:>8} {:>7.0} {:>7.0} {:>7} {:>7}",
        row.model,
        row.setting,
        row.peak_gib
            .map_or("-".to_string(), |p| format!("{p:.1} GiB")),
        row.load_seconds,
        row.seconds,
        db(row.voice_db),
        db(row.rest_db)
    )
}

/// The table's head, in the rows' columns.
pub fn header() -> String {
    format!(
        "{:6} {:48} {:>8} {:>7} {:>7} {:>7} {:>7}",
        "model", "setting", "peak", "load s", "sep s", "voice", "M&E"
    )
}

/// The energy of `b - a` under `a`'s, in dB, both folded to mono.
fn difference_db(a: &Path, b: &Path) -> Result<f64> {
    let mut first = wav::Reader::open(a)?;
    let mut second = wav::Reader::open(b)?;
    let mut x = vec![0f32; wav::BLOCK];
    let mut y = vec![0f32; wav::BLOCK];
    let (mut signal, mut difference) = (0f64, 0f64);
    loop {
        let n = first.read_block(&mut x)?;
        let m = second.read_block(&mut y[..n])?;
        if m == 0 {
            break;
        }
        for (a, b) in x[..m].iter().zip(&y[..m]) {
            signal += (*a as f64) * (*a as f64);
            difference += (*b as f64 - *a as f64).powi(2);
        }
    }
    Ok(10.0 * (difference.max(1e-30) / signal.max(1e-30)).log10())
}

/// Where a bench works when it is not told: beside the jobs.
pub fn default_dir(config: &Config) -> PathBuf {
    config.work_dir().join("bench-vram")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reference_comes_first_and_nothing_is_tried_twice() {
        let config = Config::default();
        let list = settings(&config, &[600, 300, 120], &[1, 2]);
        let labels: Vec<&str> = list.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(
            labels,
            [
                "pymss, 25 % overlap",
                "pymss, 25 % overlap, blocks of 600 s",
                "pymss, 25 % overlap, blocks of 120 s",
                "pymss, 25 % overlap, batches of 1",
                "pymss, 25 % overlap, batches of 2",
                "pymss, 5 % overlap",
                "msst",
            ]
        );
    }
}
