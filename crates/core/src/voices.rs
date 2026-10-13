//! What the dub's separated voices look like, and where they will be put.
//!
//! Once every channel of the dub has had its dialogue lifted out, one
//! question decides how the voices are placed: are the channels really
//! independent, or is every one of them the centre again?
//!
//! Plenty of dubs are delivered as a false multichannel — the same mono
//! dialogue copied across the front, sometimes at a different gain. Placing
//! one voice object per channel then puts the same voice at three places at
//! once, which is not a wider dialogue but a smeared one. So each channel
//! is compared to the centre — correlation, and the residual left after the
//! best gain — and if they are all copies of it, the centre alone is placed,
//! spread over three front objects. Otherwise each channel becomes an
//! object at its speaker's position.
//!
//! Everything is measured on short-term RMS and percentiles rather than on
//! peaks: one isolated click would otherwise make a silent channel look
//! busy.

use crate::extract;
use crate::progress::{Event, Progress};
use crate::wav;
use crate::{Error, Result};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// The short-term window, 100 ms at 48 kHz: long enough that one sample
/// cannot carry it.
const WINDOW: usize = 4800;
/// Above this, in dBFS, a window is counted as holding sound.
const PRESENT: f64 = -60.0;
/// The floor a measurement in decibels is reported at.
const FLOOR: f64 = -144.0;

/// One channel of the dub's dialogue, measured.
#[derive(Debug, Clone, PartialEq)]
pub struct Channel {
    pub name: String,
    /// Linear RMS over the whole channel, which the summary totals.
    pub rms: f64,
    /// dBFS: the average, the loudest sample, the loudest 100 ms window,
    /// and the robust maximum.
    pub rms_db: f64,
    pub peak_db: f64,
    pub window_max_db: f64,
    pub percentile_999_db: f64,
    /// Fraction of the film this channel holds sound over.
    pub presence: f64,
    /// Against the reference channel; `None` for the reference itself.
    pub against: Option<Against>,
}

/// How a channel compares to the centre.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Against {
    /// dBFS of the plain difference.
    pub difference_db: f64,
    pub correlation: f64,
    /// The gain that best fits this channel onto the reference.
    pub gain_db: f64,
    /// dBFS of what is left once that gain is taken out: low means the
    /// channel *is* the reference, louder or quieter.
    pub residual_db: f64,
}

impl Against {
    /// The one-word reading of those four numbers.
    pub fn verdict(&self) -> String {
        if self.difference_db < -80.0 {
            "identical".to_string()
        } else if self.residual_db < -70.0 {
            format!("a copy ({:+.1} dB)", self.gain_db)
        } else if self.correlation > 0.98 {
            "very close".to_string()
        } else if self.correlation > 0.90 {
            "close".to_string()
        } else {
            "independent".to_string()
        }
    }
}

/// What the analysis concluded.
#[derive(Debug, Clone, PartialEq)]
pub struct Analysis {
    pub channels: Vec<Channel>,
    /// Every channel is the reference again (or silent): a false
    /// multichannel, to be placed as one voice spread over the front.
    pub identical: bool,
    pub reference: String,
    pub sample_rate: u32,
    pub frames: u64,
}

impl Analysis {
    /// The names of the channels that hold almost nothing: mixing them in
    /// buys nothing.
    pub fn silent(&self) -> Vec<&str> {
        self.channels
            .iter()
            .filter(|c| c.presence < 0.005)
            .map(|c| c.name.as_str())
            .collect()
    }
}

/// A voice object to add to the scene: which channel it is taken from,
/// where it goes, and its share of the whole.
///
/// The share is relative. The absolute level is set once, by the gain that
/// brings the dub's dialogue to the loudness of the original's, so nothing
/// here is a bet on a level.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceObject {
    pub channel: String,
    pub position: [f64; 3],
    pub weight: f64,
}

/// Compare every channel of the dub's dialogue in `dir` to `reference`.
pub fn analyse(dir: &Path, reference: &str, progress: Progress) -> Result<Analysis> {
    let mut names = channels(dir)?;
    if !names.iter().any(|n| n.eq_ignore_ascii_case(reference)) {
        return Err(Error::msg(format!(
            "no `{reference}` among the separated channels of {}",
            dir.display()
        )));
    }
    // The reference first: its rate and its length are what the rest is
    // read against.
    names.sort_by_key(|n| (!n.eq_ignore_ascii_case(reference), n.clone()));
    let reference_path = dir.join(format!("{}{reference}.wav", extract::DUB_VOICE));

    let mut channels = Vec::with_capacity(names.len());
    let (mut sample_rate, mut frames) = (0, 0);
    for (i, name) in names.iter().enumerate() {
        let path = dir.join(format!("{}{name}.wav", extract::DUB_VOICE));
        let is_reference = name.eq_ignore_ascii_case(reference);
        let measured = one(
            &path,
            (!is_reference).then_some(reference_path.as_path()),
            name,
        )?;
        if is_reference {
            sample_rate = measured.1;
            frames = measured.2;
        }
        channels.push(measured.0);
        progress(Event::items("comparing", i as u64 + 1, names.len() as u64));
    }

    let compared: Vec<&Channel> = channels.iter().filter(|c| c.against.is_some()).collect();
    let identical = compared.is_empty()
        || compared
            .iter()
            .all(|c| c.against.expect("filtered").correlation >= 0.98 || c.presence < 0.01);
    Ok(Analysis {
        channels,
        identical,
        reference: reference.to_string(),
        sample_rate,
        frames,
    })
}

/// The separated channels of a dub, LFE aside — it carries no dialogue and
/// is never re-voiced.
fn channels(dir: &Path) -> Result<Vec<String>> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| Error::io(dir, e))? {
        let path = entry.map_err(|e| Error::io(dir, e))?.path();
        let file = path.file_name().unwrap_or_default().to_string_lossy();
        if let Some(rest) = file.strip_prefix(extract::DUB_VOICE)
            && let Some(name) = rest.strip_suffix(".wav")
            && !name.eq_ignore_ascii_case("LFE")
        {
            names.push(name.to_string());
        }
    }
    Ok(names)
}

/// One channel, walked once: its own statistics, and — when a reference is
/// given — the comparison to it, both files advancing block for block so
/// the samples line up without either being held.
fn one(path: &Path, reference: Option<&Path>, name: &str) -> Result<(Channel, u32, u64)> {
    let mut reader = wav::Reader::open(path)?;
    let sample_rate = reader.format().sample_rate;
    let mut other = reference.map(wav::Reader::open).transpose()?;

    let (mut square_sum, mut peak) = (0.0f64, 0.0f32);
    let (mut window_sum, mut window_count) = (0.0f64, 0usize);
    let mut windows: Vec<f64> = Vec::new();
    // Σxr, Σrr, Σxx and Σ(x-r)², which give the correlation, the best gain
    // and both residuals in one pass.
    let (mut xr, mut rr, mut xx, mut difference) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut compared = 0u64;

    let mut block = vec![0f32; wav::BLOCK];
    let mut theirs = vec![0f32; wav::BLOCK];
    let mut frames = 0u64;
    loop {
        let read = reader.read_block(&mut block)?;
        if read == 0 {
            break;
        }
        let shared = match &mut other {
            Some(other) => read.min(other.read_block(&mut theirs)?),
            None => 0,
        };
        for sample in &block[..read] {
            let square = (*sample as f64) * (*sample as f64);
            square_sum += square;
            peak = peak.max(sample.abs());
            window_sum += square;
            window_count += 1;
            if window_count == WINDOW {
                windows.push((window_sum / WINDOW as f64).sqrt());
                window_sum = 0.0;
                window_count = 0;
            }
        }
        for (x, r) in block[..shared].iter().zip(&theirs[..shared]) {
            let (x, r) = (*x as f64, *r as f64);
            xr += x * r;
            rr += r * r;
            xx += x * x;
            difference += (x - r) * (x - r);
        }
        compared += shared as u64;
        frames += read as u64;
    }

    let rms = (square_sum / frames.max(1) as f64).sqrt();
    let mut sorted = windows.clone();
    sorted.sort_by(f64::total_cmp);
    let channel = Channel {
        name: name.to_string(),
        rms,
        rms_db: db(rms),
        peak_db: db(peak as f64),
        window_max_db: sorted.last().copied().map(db).unwrap_or(FLOOR),
        percentile_999_db: percentile(&sorted, 99.9).map(db).unwrap_or(FLOOR),
        presence: if windows.is_empty() {
            0.0
        } else {
            windows.iter().filter(|w| db(**w) > PRESENT).count() as f64 / windows.len() as f64
        },
        against: reference.map(|_| {
            let gain = if rr > 0.0 { xr / rr } else { 0.0 };
            let residual = (xx - 2.0 * gain * xr + gain * gain * rr).max(0.0);
            Against {
                difference_db: db((difference / compared.max(1) as f64).sqrt()),
                correlation: if xx > 0.0 && rr > 0.0 {
                    xr / (xx * rr).sqrt()
                } else {
                    0.0
                },
                gain_db: db(gain),
                residual_db: db((residual / compared.max(1) as f64).sqrt()),
            }
        }),
    };
    Ok((channel, sample_rate, frames))
}

/// Where each voice goes, and its share.
///
/// Every position here is on the floor, and the encoder depends on it:
/// harlettizer's cost guard refuses an overlay whose voice ends up far from
/// where it belongs, and a voice at a floor speaker's place costs exactly
/// nothing. The day the dub is laid onto the original's *dialogue* objects
/// — the idea this whole thing started from — the positions will be able to
/// rise, and that refusal will arrive with them.
///
/// Takes what the analysis decided rather than the analysis itself, since
/// that is what the manifest keeps between the step that measured and the
/// one that places.
pub fn voice_objects(identical: bool, channels: &[String]) -> Vec<VoiceObject> {
    let names: Vec<&str> = channels.iter().map(String::as_str).collect();
    if names.len() <= 1 {
        return names
            .into_iter()
            .map(|channel| VoiceObject {
                channel: channel.to_string(),
                position: [0.0, 1.0, 0.0],
                weight: 1.0,
            })
            .collect();
    }
    if identical {
        // One voice, spread over three front places: centred, but not so
        // pointlike that a character glued to a side speaker sounds wrong.
        let source = if names.contains(&"C") { "C" } else { names[0] };
        return vec![
            VoiceObject {
                channel: source.to_string(),
                position: [0.0, 1.0, 0.0],
                weight: 1.0,
            },
            VoiceObject {
                channel: source.to_string(),
                position: [-1.0, 1.0, 0.0],
                weight: 0.5,
            },
            VoiceObject {
                channel: source.to_string(),
                position: [1.0, 1.0, 0.0],
                weight: 0.5,
            },
        ];
    }
    // A 7.1 dub has a rear pair, which puts its side pair at the sides; a
    // 5.1 one has only the pair the bed calls side, and it is at the back.
    let rears = names.iter().any(|n| *n == "Lrs" || *n == "Rrs");
    names
        .into_iter()
        .map(|channel| VoiceObject {
            channel: channel.to_string(),
            position: position(channel, rears),
            weight: 1.0,
        })
        .collect()
}

/// The speaker position of a bed channel, in [-1, 1]: x left to right,
/// y back to screen, z on the floor.
pub(crate) fn position(name: &str, rears: bool) -> [f64; 3] {
    match name {
        "C" => [0.0, 1.0, 0.0],
        "L" => [-1.0, 1.0, 0.0],
        "R" => [1.0, 1.0, 0.0],
        "Lss" if rears => [-1.0, 0.0, 0.0],
        "Rss" if rears => [1.0, 0.0, 0.0],
        "Lss" | "Lrs" => [-1.0, -1.0, 0.0],
        "Rss" | "Rrs" => [1.0, -1.0, 0.0],
        _ => [0.0, 1.0, 0.0],
    }
}

/// The report the job keeps beside the stems: the numbers the verdict was
/// read off, for the day the verdict looks wrong.
pub fn report(analysis: &Analysis) -> String {
    let mut out = String::new();
    let seconds = if analysis.sample_rate > 0 {
        analysis.frames as f64 / analysis.sample_rate as f64
    } else {
        0.0
    };
    let _ = writeln!(
        out,
        "===== the dub's voices, against {} =====",
        analysis.reference
    );
    let _ = writeln!(
        out,
        "{seconds:.1} s ({} samples at {} Hz)\n",
        analysis.frames, analysis.sample_rate
    );
    let _ = writeln!(
        out,
        "{:<6} {:>8} {:>8} {:>10} {:>8} {:>9}",
        "channel", "RMS", "peak", "window max", "p99.9", "presence"
    );
    for c in &analysis.channels {
        let _ = writeln!(
            out,
            "{:<6} {:>8.1} {:>8.1} {:>10.1} {:>8.1} {:>8.0} %",
            c.name,
            c.rms_db,
            c.peak_db,
            c.window_max_db,
            c.percentile_999_db,
            c.presence * 100.0
        );
    }
    let _ = writeln!(out, "\nagainst {}", analysis.reference);
    let _ = writeln!(
        out,
        "{:<6} {:>10} {:>8} {:>7} {:>8}  verdict",
        "channel", "diff", "corr", "gain", "residual"
    );
    for c in &analysis.channels {
        if let Some(a) = c.against {
            let _ = writeln!(
                out,
                "{:<6} {:>10.1} {:>8.4} {:>7.1} {:>8.1}  {}",
                c.name,
                a.difference_db,
                a.correlation,
                a.gain_db,
                a.residual_db,
                a.verdict()
            );
        }
    }
    let total = analysis
        .channels
        .iter()
        .map(|c| c.rms * c.rms)
        .sum::<f64>()
        .sqrt();
    let alone = analysis
        .channels
        .iter()
        .find(|c| c.name.eq_ignore_ascii_case(&analysis.reference))
        .map(|c| c.rms)
        .unwrap_or(0.0);
    let _ = writeln!(
        out,
        "\nevery channel together against {} alone: {:+.1} dB",
        analysis.reference,
        db(total) - db(alone)
    );
    let silent = analysis.silent();
    if !silent.is_empty() {
        let _ = writeln!(
            out,
            "almost silent (under 0.5 % presence): {}",
            silent.join(", ")
        );
    }
    let _ = writeln!(
        out,
        "\n{}",
        if analysis.identical {
            format!(
                "the channels are copies of {}: one voice, spread over three front objects",
                analysis.reference
            )
        } else {
            "the channels are independent: one voice object each, at its speaker's position"
                .to_string()
        }
    );
    let _ = writeln!(
        out,
        "\n`peak` is the loudest single sample and moves with one click; `p99.9` is the\n\
         robust maximum. A wide gap between them is an isolated peak, not a loud channel."
    );
    out
}

/// Where the report is written.
pub fn report_path(dir: &Path) -> PathBuf {
    dir.join("voices.txt")
}

fn db(linear: f64) -> f64 {
    if linear <= 1e-9 {
        FLOOR
    } else {
        20.0 * linear.log10()
    }
}

/// The p-th percentile of an already-sorted slice.
fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let at = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted.get(at.min(sorted.len() - 1)).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel(name: &str, correlation: f64, presence: f64) -> Channel {
        Channel {
            name: name.to_string(),
            rms: 0.1,
            rms_db: -20.0,
            peak_db: -6.0,
            window_max_db: -10.0,
            percentile_999_db: -11.0,
            presence,
            against: (name != "C").then_some(Against {
                difference_db: -30.0,
                correlation,
                gain_db: 0.0,
                residual_db: -30.0,
            }),
        }
    }

    fn names(a: &Analysis) -> Vec<String> {
        a.channels.iter().map(|c| c.name.clone()).collect()
    }

    fn analysis(channels: Vec<Channel>, identical: bool) -> Analysis {
        Analysis {
            channels,
            identical,
            reference: "C".into(),
            sample_rate: 48000,
            frames: 48000,
        }
    }

    #[test]
    fn a_false_multichannel_puts_one_voice_across_the_front() {
        let a = analysis(
            vec![
                channel("C", 1.0, 0.3),
                channel("L", 0.999, 0.3),
                channel("R", 0.999, 0.3),
            ],
            true,
        );
        let objects = voice_objects(a.identical, &names(&a));
        assert_eq!(objects.len(), 3);
        assert!(objects.iter().all(|o| o.channel == "C"));
        assert_eq!(objects[0].position, [0.0, 1.0, 0.0]);
        assert_eq!(objects[0].weight, 1.0);
        assert_eq!(objects[1].weight, 0.5);
    }

    #[test]
    fn independent_channels_each_become_an_object_at_their_speaker() {
        let a = analysis(
            vec![
                channel("C", 0.2, 0.3),
                channel("L", 0.1, 0.3),
                channel("R", 0.1, 0.3),
                channel("Lss", 0.1, 0.3),
                channel("Rss", 0.1, 0.3),
            ],
            false,
        );
        let objects = voice_objects(a.identical, &names(&a));
        assert_eq!(objects.len(), 5);
        // No rear pair: the bed's side pair is the 5.1 surround, at the back.
        let lss = objects.iter().find(|o| o.channel == "Lss").unwrap();
        assert_eq!(lss.position, [-1.0, -1.0, 0.0]);
        assert!(objects.iter().all(|o| o.weight == 1.0));
        assert!(objects.iter().all(|o| o.position[2] == 0.0));
    }

    #[test]
    fn a_seven_one_dub_puts_its_side_pair_at_the_sides() {
        let a = analysis(
            vec![
                channel("C", 0.2, 0.3),
                channel("L", 0.1, 0.3),
                channel("R", 0.1, 0.3),
                channel("Lss", 0.1, 0.3),
                channel("Rss", 0.1, 0.3),
                channel("Lrs", 0.1, 0.3),
                channel("Rrs", 0.1, 0.3),
            ],
            false,
        );
        let objects = voice_objects(a.identical, &names(&a));
        let at = |name: &str| objects.iter().find(|o| o.channel == name).unwrap().position;
        assert_eq!(at("Lss"), [-1.0, 0.0, 0.0]);
        assert_eq!(at("Lrs"), [-1.0, -1.0, 0.0]);
    }

    #[test]
    fn a_mono_dub_puts_its_one_voice_in_the_middle() {
        let a = analysis(vec![channel("C", 1.0, 0.3)], true);
        let objects = voice_objects(a.identical, &names(&a));
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].position, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn the_verdict_tells_a_copy_from_an_independent_channel() {
        let copy = Against {
            difference_db: -20.0,
            correlation: 0.999,
            gain_db: -6.0,
            residual_db: -85.0,
        };
        assert_eq!(copy.verdict(), "a copy (-6.0 dB)");
        let same = Against {
            difference_db: -120.0,
            correlation: 1.0,
            gain_db: 0.0,
            residual_db: -120.0,
        };
        assert_eq!(same.verdict(), "identical");
        let independent = Against {
            difference_db: -10.0,
            correlation: 0.3,
            gain_db: -1.0,
            residual_db: -12.0,
        };
        assert_eq!(independent.verdict(), "independent");
    }

    #[test]
    fn a_report_names_every_channel_and_the_conclusion() {
        let a = analysis(
            vec![channel("C", 1.0, 0.3), channel("L", 0.999, 0.001)],
            true,
        );
        let text = report(&a);
        assert!(text.contains("channel"));
        assert!(text.contains(" C "));
        assert!(text.contains("almost silent"), "{text}");
        assert!(text.contains("spread over three front objects"), "{text}");
    }
}
