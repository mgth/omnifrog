//! A zone's audio, whole in memory.
//!
//! A zone is a few seconds to two minutes of mono at 48 kHz: small enough to
//! measure as one array, where a job walks its hours in blocks.

use crate::{Error, Result};
use omnifrog_core::wav::{self, BLOCK, Format, Reader};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Every sample of a WAV, its channels averaged into one.
pub fn read(path: &Path) -> Result<Vec<f32>> {
    let mut reader = Reader::open(path)?;
    let mut samples = vec![0f32; reader.format().frames as usize];
    let mut at = 0;
    while at < samples.len() {
        let to = (at + BLOCK).min(samples.len());
        let read = reader.read_block(&mut samples[at..to])?;
        if read == 0 {
            break;
        }
        at += read;
    }
    samples.truncate(at);
    Ok(samples)
}

/// A sample as the 24-bit value a file keeps of it. One half-way between
/// two steps goes to the even one: a consensus of two or four outputs is
/// full of such samples, and the reference written here is then the very
/// file an imported legacy set came with.
pub fn quantized(sample: f32) -> i32 {
    (sample * 8_388_608.0)
        .round_ties_even()
        .clamp(-8_388_608.0, 8_388_607.0) as i32
}

/// Write samples as a mono 24-bit WAV, whole or not at all.
pub fn write(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    let part = path.with_extension("wav.part");
    let file = File::create(&part).map_err(|e| Error::io(&part, e))?;
    let mut writer = BufWriter::new(file);
    let format = Format {
        channels: 1,
        sample_rate,
        bits: 24,
        float: false,
        frames: samples.len() as u64,
        data_offset: 0,
    };
    wav::write_header(&mut writer, format).map_err(|e| Error::io(&part, e))?;
    for &sample in samples {
        let value = quantized(sample);
        writer
            .write_all(&value.to_le_bytes()[..3])
            .map_err(|e| Error::io(&part, e))?;
    }
    writer.flush().map_err(|e| Error::io(&part, e))?;
    drop(writer);
    fs::rename(&part, path).map_err(|e| Error::io(path, e))
}

/// Copy `frames` frames of a WAV from `start` into a file of its own, the
/// samples as they are — no decoding, whatever their depth — and whole or
/// not at all. The source is never read entire: an element of a film is a
/// gigabyte, and a zone a few seconds of it. A range that runs past the end
/// is cut there; the frames written are returned.
pub fn cut(source: &Path, destination: &Path, start: u64, frames: u64) -> Result<u64> {
    let format = wav::read_header(source)?;
    let start = start.min(format.frames);
    let frames = frames.min(format.frames - start);
    let size = format.bytes_per_frame();
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    let mut input = File::open(source).map_err(|e| Error::io(source, e))?;
    input
        .seek(SeekFrom::Start(format.data_offset + start * size))
        .map_err(|e| Error::io(source, e))?;
    let part = destination.with_extension("wav.part");
    let file = File::create(&part).map_err(|e| Error::io(&part, e))?;
    let mut writer = BufWriter::new(file);
    wav::write_header(&mut writer, Format { frames, ..format }).map_err(|e| Error::io(&part, e))?;
    let copied = std::io::copy(&mut input.take(frames * size), &mut writer)
        .map_err(|e| Error::io(&part, e))?;
    if copied != frames * size {
        return Err(Error::msg(format!(
            "{}: shorter than its header says",
            source.display()
        )));
    }
    writer.flush().map_err(|e| Error::io(&part, e))?;
    drop(writer);
    fs::rename(&part, destination).map_err(|e| Error::io(destination, e))?;
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_come_back_as_they_were_written_to_24_bits() {
        let path =
            std::env::temp_dir().join(format!("omnifrog-bench-audio-{}.wav", std::process::id()));
        let step = 1.0 / 8_388_608.0f32;
        // Exact steps, two ties (to the even step) and both ends.
        let samples = [
            0.0,
            0.5,
            -0.5,
            3.0 * step,
            2.5 * step,
            3.5 * step,
            2.0,
            -2.0,
        ];
        write(&path, &samples, 48_000).unwrap();
        assert!(!path.with_extension("wav.part").exists());
        assert_eq!(wav::read_header(&path).unwrap().sample_rate, 48_000);
        let back = read(&path).unwrap();
        assert_eq!(
            back,
            [
                0.0,
                0.5,
                -0.5,
                3.0 * step,
                2.0 * step,
                4.0 * step,
                1.0 - step,
                -1.0
            ]
        );
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_range_is_cut_out_as_it_is() {
        let dir = std::env::temp_dir().join(format!("omnifrog-bench-cut-{}", std::process::id()));
        let source = dir.join("whole.wav");
        let step = 1.0 / 8_388_608.0f32;
        let samples: Vec<f32> = (0..1000).map(|i| i as f32 * step).collect();
        write(&source, &samples, 48_000).unwrap();
        let part = dir.join("deep/part.wav");
        assert_eq!(cut(&source, &part, 100, 50).unwrap(), 50);
        assert_eq!(read(&part).unwrap(), &samples[100..150]);
        // Past the end: what there is.
        assert_eq!(cut(&source, &part, 990, 50).unwrap(), 10);
        assert_eq!(read(&part).unwrap(), &samples[990..]);
        assert_eq!(cut(&source, &part, 5000, 50).unwrap(), 0);
        assert!(cut(&dir.join("absent.wav"), &part, 0, 1).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }
}
