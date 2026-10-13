//! How many bits the source master was really mastered at.
//!
//! A lossless coder drops the low bits no sample sets for nothing, and codes
//! every other bit it is handed, signal or not. Two things here hand it bits
//! that are not signal. The separation writes its stems in float, so an
//! element without its dialogue comes back with all 24 bits set; and
//! harlettizer's mix of a voice into an element is a sum over real
//! weights, whose low bits are the noise of the multiplication. The
//! re-voiced master cannot say where to cut: every one of its bits is
//! arithmetic. The original can: it is the master as it was delivered.
//!
//! One TrueHD Atmos master carries six dead bits on its objects, so it was
//! mastered at 18. Its separated elements came back at 24, and the stream
//! coded from them was 6.07 GB where the disc's is 3.82. Rounded to 18 as
//! they are written — the error is some 110 dB down, far under what the
//! separation itself leaves — and with harlettizer rounding its mixes to
//! the same depth, it is 3.56 GB: no bit the disc did not carry.
//!
//! The measure is the OR of every sample read: a bit that is clear in the OR
//! was never set by any of them. The fewest dead bits over every waveform
//! and every window wins — never coarser than the finest channel at its
//! finest moment.

use crate::extract;
use crate::wav;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// What harlettizer takes when told nothing: careful, and what is used
/// when the original cannot answer.
pub const DEFAULT: u32 = 20;
/// harlettizer's floor: below it the format's dead-bit field can no longer
/// say what was rounded away.
pub const MINIMUM: u32 = 17;
/// Spread over the film: one window could land on a passage poorer than
/// the rest.
const WINDOWS: u64 = 8;
/// About four seconds at 48 kHz: 600 kB a waveform a window.
const WINDOW: u64 = 200_000;

/// The depth of the original's waveforms in `dir`, between [`MINIMUM`] and
/// 24; [`DEFAULT`] with the reason when it cannot be told.
pub fn of_original(dir: &Path) -> (u32, Option<String>) {
    match measured(dir) {
        Ok(bits) => (bits.clamp(MINIMUM, 24), None),
        Err(why) => (DEFAULT, Some(why)),
    }
}

/// The depth of the original's waveforms in `dir` as it is, 1 to 24, with
/// no floor: what a lossless IAMF stream is told to keep. The reason when it
/// cannot be told.
pub fn measured(dir: &Path) -> Result<u32, String> {
    let mut paths: Vec<_> = match std::fs::read_dir(dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                let name = p.file_name().unwrap_or_default().to_string_lossy();
                name.starts_with(extract::ORIGINAL) && name.ends_with(".wav")
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    paths.sort();
    if paths.is_empty() {
        return Err(format!("no waveform of the original in {}", dir.display()));
    }
    let mut dead = u32::MAX;
    for path in &paths {
        match dead_bits(path) {
            Ok(Some(bits)) => dead = dead.min(bits),
            Ok(None) => {}
            Err(why) => return Err(why),
        }
        if dead == 0 {
            break;
        }
    }
    if dead == u32::MAX {
        return Err("the original is silent wherever it was sampled".to_string());
    }
    Ok((24 - dead.min(23)).max(1))
}

/// Whether `path` carries nothing below `bits` bits wherever it is read:
/// what a stem written by an earlier version, at the worker's full 24, is
/// told apart by. Digital silence is at every depth.
pub fn within(path: &Path, bits: u32) -> Result<bool, String> {
    Ok(dead_bits(path)?.is_none_or(|dead| dead >= 24 - bits.min(24)))
}

/// The low bits no sample of `path` sets, over [`WINDOWS`] windows; `None`
/// for a file that is digital silence wherever it was read.
fn dead_bits(path: &Path) -> Result<Option<u32>, String> {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let format = wav::read_header(path).map_err(|e| e.to_string())?;
    if format.float || format.bits != 24 || format.channels != 1 {
        return Err(format!(
            "{name} is {}-bit{}, {} channel(s): the measure reads mono 24-bit PCM",
            format.bits,
            if format.float { " float" } else { "" },
            format.channels
        ));
    }
    let frames = format.frames;
    if frames == 0 {
        return Ok(None);
    }
    let mut file = File::open(path).map_err(|e| format!("{name}: {e}"))?;
    let window = frames.min(WINDOW);
    let mut buffer = vec![0u8; window as usize * 3];
    let mut dead = u32::MAX;
    for w in 0..WINDOWS {
        let start = if frames > window {
            (frames - window) * w / (WINDOWS - 1)
        } else {
            0
        };
        file.seek(SeekFrom::Start(format.data_offset + start * 3))
            .map_err(|e| format!("{name}: {e}"))?;
        let mut filled = 0;
        while filled < buffer.len() {
            match file.read(&mut buffer[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) => return Err(format!("{name}: {e}")),
            }
        }
        let (samples, _) = buffer[..filled].as_chunks::<3>();
        let or = samples.iter().fold(0u32, |or, s| {
            or | u32::from(s[0]) | u32::from(s[1]) << 8 | u32::from(s[2]) << 16
        });
        // Digital silence says nothing, and must not claim 24 dead bits.
        if or == 0 {
            continue;
        }
        dead = dead.min(or.trailing_zeros());
        if dead == 0 {
            break;
        }
    }
    Ok((dead != u32::MAX).then_some(dead))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::{BufWriter, Write};

    fn waveform(path: &Path, samples: impl Iterator<Item = i32>, frames: u64) {
        let mut out = BufWriter::new(File::create(path).unwrap());
        wav::write_header(
            &mut out,
            wav::Format {
                channels: 1,
                sample_rate: 48000,
                bits: 24,
                float: false,
                frames,
                data_offset: 0,
            },
        )
        .unwrap();
        for s in samples.take(frames as usize) {
            out.write_all(&s.to_le_bytes()[..3]).unwrap();
        }
        out.flush().unwrap();
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("omnifrog-depth-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_master_at_eighteen_bits_is_measured_at_eighteen() {
        let dir = scratch("18");
        // Six low bits never set, the sign bit set half the time.
        let samples = (0..).map(|i: i32| ((i % 1000) - 500) << 6);
        waveform(&dir.join("vo_10.wav"), samples, 50_000);
        assert_eq!(of_original(&dir), (18, None));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_finest_waveform_decides() {
        let dir = scratch("finest");
        waveform(
            &dir.join("vo_10.wav"),
            (0..).map(|i: i32| (i % 700) << 8),
            50_000,
        );
        waveform(
            &dir.join("vo_11.wav"),
            (0..).map(|i: i32| (i % 700) << 3),
            50_000,
        );
        assert_eq!(of_original(&dir).0, 21);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn silence_or_nothing_falls_back_to_the_default_and_says_why() {
        let dir = scratch("silent");
        let (depth, why) = of_original(&dir);
        assert_eq!(depth, DEFAULT);
        assert!(why.is_some());
        waveform(&dir.join("vo_3.wav"), std::iter::repeat(0), 10_000);
        let (depth, why) = of_original(&dir);
        assert_eq!(depth, DEFAULT);
        assert!(why.unwrap().contains("silent"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stem_is_within_a_depth_when_no_sample_sets_a_bit_below_it() {
        let dir = scratch("within");
        let path = dir.join("sons_10.wav");
        waveform(&path, (0..).map(|i: i32| ((i % 900) - 450) << 6), 50_000);
        assert_eq!(within(&path, 18), Ok(true));
        assert_eq!(within(&path, 20), Ok(true));
        assert_eq!(within(&path, 17), Ok(false));
        waveform(&path, (0..).map(|i: i32| (i % 900) - 450), 50_000);
        assert_eq!(within(&path, 18), Ok(false));
        assert_eq!(within(&path, 24), Ok(true));
        waveform(&path, std::iter::repeat(0), 10_000);
        assert_eq!(within(&path, 18), Ok(true));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_coarse_master_is_held_at_the_floor() {
        let dir = scratch("coarse");
        waveform(
            &dir.join("vo_10.wav"),
            (0..).map(|i: i32| (i % 50) << 12),
            20_000,
        );
        assert_eq!(of_original(&dir).0, MINIMUM);
        let _ = fs::remove_dir_all(&dir);
    }
}
