//! The interleaved audio of a master set: a CAF of 24-bit big-endian PCM.
//!
//! A master set carries every waveform in one file, frame after frame, and
//! Dee reads nothing else. So the re-voiced master's waveforms — each a
//! mono WAV — are woven into one here, a block at a time: every input is
//! read `BLOCK` frames at a go, the frames are interleaved, and the block is
//! written. A film's CAF is twenty-odd gigabytes, and none of it is held.
//!
//! The byte order is the only transformation: a WAV is little-endian and a
//! master set's CAF is big-endian, so each three-byte sample is written
//! back to front. Nothing is decoded, so nothing can round.

use crate::progress::{Event, Progress};
use crate::wav;
use crate::{Error, Result};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const BYTES: usize = 3;

/// Interleave `inputs` — mono 24-bit WAVs, in the master set's order —
/// into a CAF at `destination`, `frames` frames long. An input shorter than
/// that is padded with silence; one longer is cut.
pub fn interleave(
    inputs: &[PathBuf],
    destination: &Path,
    frames: u64,
    sample_rate: f64,
    progress: Progress,
) -> Result<()> {
    if inputs.is_empty() {
        return Err(Error::msg("a master set needs at least one waveform"));
    }
    let mut readers = Vec::with_capacity(inputs.len());
    for input in inputs {
        let format = wav::read_header(input)?;
        if format.channels != 1 || format.bits != 24 || format.float {
            return Err(Error::msg(format!(
                "{}: {} channel(s) of {}-bit{} samples; a master set's waveform is mono 24-bit PCM",
                input.display(),
                format.channels,
                format.bits,
                if format.float { " float" } else { "" }
            )));
        }
        let mut file = File::open(input).map_err(|e| Error::io(input, e))?;
        file.seek(SeekFrom::Start(format.data_offset))
            .map_err(|e| Error::io(input, e))?;
        readers.push((file, format.frames));
    }

    let channels = inputs.len();
    let mut out = BufWriter::with_capacity(
        1 << 20,
        File::create(destination).map_err(|e| Error::io(destination, e))?,
    );
    write_header(&mut out, sample_rate, channels as u32, frames)
        .map_err(|e| Error::io(destination, e))?;

    let mut columns = vec![vec![0u8; wav::BLOCK * BYTES]; channels];
    let mut frame = vec![0u8; wav::BLOCK * BYTES * channels];
    let mut written = 0u64;
    let mut last = u64::MAX;
    while written < frames {
        let count = (frames - written).min(wav::BLOCK as u64) as usize;
        for ((file, held), column) in readers.iter_mut().zip(&mut columns) {
            // What is left of this input, and silence past its end.
            let available = held.saturating_sub(written).min(count as u64) as usize;
            let bytes = &mut column[..count * BYTES];
            file.read_exact(&mut bytes[..available * BYTES])
                .map_err(|e| Error::msg(format!("reading a waveform: {e}")))?;
            bytes[available * BYTES..].fill(0);
        }
        for f in 0..count {
            let base = f * channels * BYTES;
            for (c, column) in columns.iter().enumerate() {
                let from = &column[f * BYTES..f * BYTES + BYTES];
                let to = &mut frame[base + c * BYTES..base + c * BYTES + BYTES];
                // Little-endian in, big-endian out.
                to[0] = from[2];
                to[1] = from[1];
                to[2] = from[0];
            }
        }
        out.write_all(&frame[..count * channels * BYTES])
            .map_err(|e| Error::io(destination, e))?;
        written += count as u64;
        let percent = written * 100 / frames.max(1);
        if percent != last {
            last = percent;
            progress(Event::ratio("interleaving", written as f64 / frames as f64));
        }
    }
    out.flush().map_err(|e| Error::io(destination, e))
}

/// `caff`, a `desc` chunk for 24-bit big-endian integer PCM, and the head
/// of the `data` chunk; the samples follow.
fn write_header(
    out: &mut impl Write,
    sample_rate: f64,
    channels: u32,
    frames: u64,
) -> std::io::Result<()> {
    let data = frames * channels as u64 * BYTES as u64;
    out.write_all(b"caff")?;
    out.write_all(&1u16.to_be_bytes())?;
    out.write_all(&0u16.to_be_bytes())?;
    out.write_all(b"desc")?;
    out.write_all(&32i64.to_be_bytes())?;
    out.write_all(&sample_rate.to_be_bytes())?;
    out.write_all(b"lpcm")?;
    // Format flags: integer, big-endian — both bits clear.
    out.write_all(&0u32.to_be_bytes())?;
    out.write_all(&(channels * BYTES as u32).to_be_bytes())?;
    out.write_all(&1u32.to_be_bytes())?;
    out.write_all(&channels.to_be_bytes())?;
    out.write_all(&(BYTES as u32 * 8).to_be_bytes())?;
    out.write_all(b"data")?;
    // The data chunk opens with a four-byte edit count.
    out.write_all(&(4 + data as i64).to_be_bytes())?;
    out.write_all(&0u32.to_be_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn mono(path: &Path, samples: &[i32]) {
        let mut out = BufWriter::new(File::create(path).unwrap());
        wav::write_header(
            &mut out,
            wav::Format {
                channels: 1,
                sample_rate: 48000,
                bits: 24,
                float: false,
                frames: samples.len() as u64,
                data_offset: 0,
            },
        )
        .unwrap();
        for s in samples {
            out.write_all(&s.to_le_bytes()[..3]).unwrap();
        }
        out.flush().unwrap();
    }

    #[test]
    fn frames_are_interleaved_big_endian_and_a_short_input_is_padded() {
        let dir = std::env::temp_dir().join(format!("omnifrog-caf-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let (a, b, caf) = (dir.join("a.wav"), dir.join("b.wav"), dir.join("x.caf"));
        mono(&a, &[0x010203, 0x040506, -1]);
        mono(&b, &[0x0A0B0C]);
        interleave(&[a, b], &caf, 3, 48000.0, &crate::progress::quiet).unwrap();

        let bytes = fs::read(&caf).unwrap();
        assert_eq!(&bytes[0..4], b"caff");
        let rate = f64::from_be_bytes(bytes[20..28].try_into().unwrap());
        assert_eq!(rate, 48000.0);
        let channels = u32::from_be_bytes(bytes[44..48].try_into().unwrap());
        assert_eq!(channels, 2);
        // 8 file header + 44 desc + 12 data head + 4 edit count.
        let data = &bytes[68..];
        assert_eq!(data.len(), 3 * 2 * 3);
        assert_eq!(&data[0..6], &[0x01, 0x02, 0x03, 0x0A, 0x0B, 0x0C]);
        assert_eq!(&data[6..12], &[0x04, 0x05, 0x06, 0, 0, 0]);
        assert_eq!(&data[12..18], &[0xFF, 0xFF, 0xFF, 0, 0, 0]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_float_waveform_is_refused_rather_than_misread() {
        let dir = std::env::temp_dir().join(format!("omnifrog-caf-f-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f.wav");
        let mut out = BufWriter::new(File::create(&path).unwrap());
        wav::write_header(
            &mut out,
            wav::Format {
                channels: 1,
                sample_rate: 48000,
                bits: 32,
                float: true,
                frames: 1,
                data_offset: 0,
            },
        )
        .unwrap();
        out.write_all(&0.5f32.to_le_bytes()).unwrap();
        drop(out);
        assert!(
            interleave(
                &[path],
                &dir.join("x.caf"),
                1,
                48000.0,
                &crate::progress::quiet
            )
            .is_err()
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
