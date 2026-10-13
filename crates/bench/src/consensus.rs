//! The reference of a side: the consensus of the outputs rated best.
//!
//! One output rated best is the reference as it is. Several are aligned on
//! the first and their median is taken, sample by sample: it follows no one
//! model, and the artifact of a single one does not get in.

use crate::Result;
use crate::audio;
use crate::layout;
use crate::metrics;
use crate::set::{RATE, Side, Zone};
use std::path::Path;

/// The median of several outputs, sample by sample (of two, the mean), each
/// shifted onto the first by the lag found for it, over their common length.
pub fn consensus(outputs: &[Vec<f32>]) -> Vec<f32> {
    let Some(first) = outputs.first() else {
        return Vec::new();
    };
    if outputs.len() == 1 {
        return first.clone();
    }
    let n = outputs.iter().map(Vec::len).min().unwrap_or(0);
    let lags: Vec<i64> = outputs
        .iter()
        .enumerate()
        .map(|(k, output)| match k {
            0 => 0,
            _ => metrics::lag(first, output) as i64,
        })
        .collect();
    let mut values = vec![0f32; outputs.len()];
    let middle = values.len() / 2;
    (0..n)
        .map(|i| {
            for ((value, output), lag) in values.iter_mut().zip(outputs).zip(&lags) {
                let j = i as i64 + lag;
                *value = if j >= 0 && (j as usize) < output.len() {
                    output[j as usize]
                } else {
                    0.0
                };
            }
            values.sort_unstable_by(f32::total_cmp);
            if values.len() % 2 == 1 {
                values[middle]
            } else {
                0.5 * (values[middle - 1] + values[middle])
            }
        })
        .collect()
}

/// The two stems of a side's reference, from the outputs rated best that
/// have both stems on disk. None when no reference has its files.
pub fn of(dir: &Path, zone: &Zone, side: Side) -> Result<Option<[Vec<f32>; 2]>> {
    let members = members(dir, zone, side);
    if members.is_empty() {
        return Ok(None);
    }
    let stem = |stem| -> Result<Vec<f32>> {
        let outputs = members
            .iter()
            .map(|key| audio::read(&layout::output(dir, zone, key, stem)))
            .collect::<Result<Vec<_>>>()?;
        Ok(consensus(&outputs))
    };
    Ok(Some([stem(side.voice())?, stem(side.rest())?]))
}

/// The references of a side that have both of its stems on disk.
pub fn members(dir: &Path, zone: &Zone, side: Side) -> Vec<String> {
    zone.rating(side)
        .refs
        .iter()
        .filter(|key| layout::side_present(dir, zone, key, side))
        .cloned()
        .collect()
}

/// Write the reference files of a side. A reference that has lost its files
/// is dropped from the zone's list first; with none left, nothing is
/// written. Whether there is a reference.
pub fn write(dir: &Path, zone: &mut Zone, side: Side) -> Result<bool> {
    let members = members(dir, zone, side);
    zone.rating_mut(side).refs = members;
    let Some([voice, rest]) = of(dir, zone, side)? else {
        return Ok(false);
    };
    audio::write(&layout::reference(dir, zone, side.voice()), &voice, RATE)?;
    audio::write(&layout::reference(dir, zone, side.rest()), &rest, RATE)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_output_is_its_own_consensus() {
        assert_eq!(consensus(&[vec![0.1, 0.2, 0.3]]), [0.1, 0.2, 0.3]);
        assert!(consensus(&[]).is_empty());
    }

    #[test]
    fn two_are_averaged_and_three_give_their_median() {
        let a = vec![0.0, 0.2, 0.4, 0.6];
        let b = vec![0.2, 0.2, 0.0, 0.2];
        assert_eq!(consensus(&[a.clone(), b.clone()]), [0.1, 0.2, 0.2, 0.4]);
        // The third is wild on one sample: it does not get in.
        let c = vec![0.1, 9.0, 0.3, 0.3, 0.7];
        assert_eq!(consensus(&[a, b, c]), [0.1, 0.2, 0.3, 0.3]);
    }

    #[test]
    fn a_late_output_is_aligned_before_the_median() {
        let n = 48_000;
        let mut state = 0x1234_5678_9abc_def1u64;
        let first: Vec<f32> = (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect();
        let late: Vec<f32> = (0..n)
            .map(|i| if i >= 5 { first[i - 5] } else { 0.0 })
            .collect();
        let both = consensus(&[first.clone(), late]);
        assert_eq!(&both[..n - 5], &first[..n - 5]);
    }
}
