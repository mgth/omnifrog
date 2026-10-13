//! WAV: reading a header, streaming samples, writing a mono file.
//!
//! Everything this pipeline hands around between steps is a WAV: ffmpeg
//! writes the dub's channels, the decoder writes the original's waveforms,
//! the separation worker writes its stems. A header says whether a file is
//! whole, how long it is and at what rate, which is what a resume needs to
//! tell a finished element from a half-written one.
//!
//! Nothing is ever held whole. A film's channel is a gigabyte and a step
//! may walk seven of them at once, so a reader hands out blocks and a
//! writer takes them; the peak cost of any step here is its block size.

use crate::{Error, Result};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Frames a block holds: 64 k frames is 192 kB of 24-bit mono, small enough
/// to walk a dozen files at once and large enough that the syscalls vanish.
pub const BLOCK: usize = 1 << 16;

/// What a WAV header states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Format {
    pub channels: u16,
    pub sample_rate: u32,
    pub bits: u16,
    /// IEEE float samples rather than signed integers: what the separation
    /// worker writes, and what a stem needs so a peak past ±1 survives.
    pub float: bool,
    /// Frames the `data` chunk holds.
    pub frames: u64,
    /// Where the samples start.
    pub data_offset: u64,
}

impl Format {
    pub fn bytes_per_sample(&self) -> u64 {
        self.bits as u64 / 8
    }

    pub fn bytes_per_frame(&self) -> u64 {
        self.channels as u64 * self.bytes_per_sample()
    }

    pub fn data_bytes(&self) -> u64 {
        self.frames * self.bytes_per_frame()
    }

    pub fn seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.frames as f64 / self.sample_rate as f64
        }
    }
}

/// The header of the WAV at `path`.
pub fn read_header(path: &Path) -> Result<Format> {
    let mut file = File::open(path).map_err(|e| Error::io(path, e))?;
    let length = file.metadata().map_err(|e| Error::io(path, e))?.len();
    header(&mut file, length).map_err(|e| name(path, e))
}

/// Frames the WAV at `path` holds, or 0 when it is not a readable WAV: what
/// a resume asks about a file a previous run left behind.
pub fn frames(path: &Path) -> u64 {
    read_header(path).map(|f| f.frames).unwrap_or(0)
}

/// Whether the file is a whole WAV of exactly `frames` frames. A run
/// interrupted mid-write leaves a shorter one, and a file left by another
/// parameter's run a different one.
pub fn holds(path: &Path, frames: u64) -> bool {
    read_header(path).is_ok_and(|f| f.frames == frames)
}

/// Copy a WAV changing nothing but the sample rate its header states.
///
/// This is the no-resample trick the separation depends on: the models are
/// trained at 44.1 kHz and MSST would resample a 48 kHz input to get there
/// and back, twice over, for every element of a film. Labelling the very
/// same samples 44.1 kHz hands the model exactly what it expects, and the
/// stems come back to be re-labelled 48 kHz. Not one sample is decoded,
/// and the file is copied in blocks.
pub fn copy_rate(source: &Path, destination: &Path, sample_rate: u32) -> Result<Format> {
    let mut reader = File::open(source).map_err(|e| Error::io(source, e))?;
    let length = reader.metadata().map_err(|e| Error::io(source, e))?.len();
    let format = header(&mut reader, length).map_err(|e| name(source, e))?;
    create_parent(destination)?;
    let mut writer =
        BufWriter::new(File::create(destination).map_err(|e| Error::io(destination, e))?);
    write_header(
        &mut writer,
        Format {
            sample_rate,
            ..format
        },
    )
    .map_err(|e| Error::io(destination, e))?;
    let mut left = format.data_bytes();
    let mut buffer = vec![0u8; BLOCK];
    while left > 0 {
        let want = left.min(buffer.len() as u64) as usize;
        let read = reader
            .read(&mut buffer[..want])
            .map_err(|e| Error::io(source, e))?;
        if read == 0 {
            break;
        }
        writer
            .write_all(&buffer[..read])
            .map_err(|e| Error::io(destination, e))?;
        left -= read as u64;
    }
    writer.flush().map_err(|e| Error::io(destination, e))?;
    Ok(Format {
        sample_rate,
        ..format
    })
}

/// Write two mono WAVs as one stereo WAV — `left` on the left, `right` on
/// the right — under a header that states `sample_rate`, cut to the shorter
/// of the two.
///
/// What a pair of elements is handed to a stereo model as. Like
/// [`copy_rate`] it decodes nothing: the two files have to be of the same
/// sample format, and their samples are interleaved byte for byte.
pub fn pair(left: &Path, right: &Path, destination: &Path, sample_rate: u32) -> Result<Format> {
    let open = |path: &Path| -> Result<(File, Format)> {
        let mut file = File::open(path).map_err(|e| Error::io(path, e))?;
        let length = file.metadata().map_err(|e| Error::io(path, e))?.len();
        let format = header(&mut file, length).map_err(|e| name(path, e))?;
        if format.channels != 1 {
            return Err(Error::msg(format!(
                "{}: {} channels, where a pair is made of two mono files",
                path.display(),
                format.channels
            )));
        }
        Ok((file, format))
    };
    let (mut l, lf) = open(left)?;
    let (mut r, rf) = open(right)?;
    if (lf.bits, lf.float) != (rf.bits, rf.float) {
        return Err(Error::msg(format!(
            "{} and {}: two sample formats, which a pair cannot interleave",
            left.display(),
            right.display()
        )));
    }
    let format = Format {
        channels: 2,
        sample_rate,
        frames: lf.frames.min(rf.frames),
        ..lf
    };
    create_parent(destination)?;
    let mut writer =
        BufWriter::new(File::create(destination).map_err(|e| Error::io(destination, e))?);
    write_header(&mut writer, format).map_err(|e| Error::io(destination, e))?;
    let size = lf.bytes_per_sample() as usize;
    let mut a = vec![0u8; BLOCK * size];
    let mut b = vec![0u8; BLOCK * size];
    let mut out = vec![0u8; 2 * BLOCK * size];
    let mut left_frames = format.frames;
    while left_frames > 0 {
        let take = (left_frames as usize).min(BLOCK);
        l.read_exact(&mut a[..take * size])
            .map_err(|e| Error::io(left, e))?;
        r.read_exact(&mut b[..take * size])
            .map_err(|e| Error::io(right, e))?;
        for f in 0..take {
            out[2 * f * size..(2 * f + 1) * size].copy_from_slice(&a[f * size..(f + 1) * size]);
            out[(2 * f + 1) * size..(2 * f + 2) * size]
                .copy_from_slice(&b[f * size..(f + 1) * size]);
        }
        writer
            .write_all(&out[..2 * take * size])
            .map_err(|e| Error::io(destination, e))?;
        left_frames -= take as u64;
    }
    writer.flush().map_err(|e| Error::io(destination, e))?;
    Ok(format)
}

/// How [`copy_mono`] writes a sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sample {
    /// IEEE float, which no peak is clipped by: what a stem kept for a
    /// measurement wants.
    Float,
    /// 24-bit PCM rounded to its `n` highest bits, the others left clear;
    /// `Pcm(24)` keeps them all. A lossless coder drops the bits no sample
    /// sets for nothing, and codes every other one.
    Pcm(u32),
}

/// Read a WAV folded to mono and write it as a mono WAV of exactly
/// `frames` frames — zero-padded when the source is shorter, cut when it is
/// longer — at `sample_rate`, scaled by `gain`, in blocks. Returns how many
/// samples had to be clipped to fit 24 bits; a float file clips none.
///
/// This is how a stem comes back from the worker — folded to mono,
/// re-labelled 48 kHz, squared up with the element it came from — and how a
/// dubbed voice is brought to its level. `form` says how; PCM is what
/// the rest of the pipeline reads.
pub fn copy_mono(
    source: &Path,
    destination: &Path,
    frames: u64,
    sample_rate: u32,
    gain: f32,
    form: Sample,
) -> Result<u64> {
    mix_mono(
        &[Pick::Fold(source)],
        destination,
        frames,
        sample_rate,
        gain,
        form,
    )
}

/// What [`mix_mono`] reads from a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pick<'a> {
    /// Every channel, averaged: a mono file, or a stem the worker wrote with
    /// the same element on both sides.
    Fold(&'a Path),
    /// One channel alone, from 0: one element of a stem made from a pair.
    Channel(&'a Path, usize),
}

impl Pick<'_> {
    fn open(self) -> Result<Reader> {
        match self {
            Pick::Fold(path) => Reader::open(path),
            Pick::Channel(path, channel) => Reader::open_channel(path, channel),
        }
    }
}

/// [`copy_mono`] from the average of several picks: the same element
/// estimated twice, from two different pairs, and written once. One pick is
/// a plain copy.
pub fn mix_mono(
    sources: &[Pick],
    destination: &Path,
    frames: u64,
    sample_rate: u32,
    gain: f32,
    form: Sample,
) -> Result<u64> {
    if sources.is_empty() {
        return Err(Error::msg(format!(
            "{}: nothing to write it from",
            destination.display()
        )));
    }
    let mut readers = sources
        .iter()
        .map(|pick| pick.open())
        .collect::<Result<Vec<_>>>()?;
    let share = 1.0 / readers.len() as f32;
    create_parent(destination)?;
    let file = File::create(destination).map_err(|e| Error::io(destination, e))?;
    let mut writer = BufWriter::new(file);
    let float = form == Sample::Float;
    let bits = if float { 32 } else { 24 };
    // One step of the kept depth, in 24-bit units, and the highest value a
    // sample on that grid can take: rounded up to full scale, a peak would
    // set every bit the rounding cleared.
    let step = match form {
        Sample::Pcm(kept) => (1u32 << (24 - kept.clamp(1, 24))) as f32,
        Sample::Float => 1.0,
    };
    let top = 8_388_608.0 - step;
    write_header(
        &mut writer,
        Format {
            channels: 1,
            sample_rate,
            bits,
            float,
            frames,
            data_offset: 0,
        },
    )
    .map_err(|e| Error::io(destination, e))?;
    let per_sample = bits as usize / 8;
    let mut samples = vec![0f32; BLOCK];
    let mut other = vec![0f32; BLOCK];
    let mut bytes = vec![0u8; BLOCK * per_sample];
    let mut written = 0u64;
    let mut clipped = 0u64;
    while written < frames {
        let (first, rest) = readers.split_first_mut().expect("at least one source");
        let mut read = first.read_block(&mut samples)?;
        // The picks advance in step; the shortest one ends the mix.
        for reader in rest {
            read = read.min(reader.read_block(&mut other[..read])?);
            for (sample, add) in samples[..read].iter_mut().zip(&other) {
                *sample += add;
            }
        }
        if read == 0 {
            break;
        }
        let take = read.min((frames - written) as usize);
        let mut at = 0;
        for sample in &samples[..take] {
            let sample = sample * share * gain;
            if float {
                bytes[at..at + 4].copy_from_slice(&sample.to_le_bytes());
            } else {
                // Exact in f32: a 24-bit value and a power of two apart.
                let scaled = (sample * 8_388_608.0 / step).round() * step;
                if !(-8_388_608.0..=top).contains(&scaled) {
                    clipped += 1;
                }
                let value = scaled.clamp(-8_388_608.0, top) as i32;
                bytes[at..at + 3].copy_from_slice(&value.to_le_bytes()[..3]);
            }
            at += per_sample;
        }
        writer
            .write_all(&bytes[..at])
            .map_err(|e| Error::io(destination, e))?;
        written += take as u64;
    }
    // The source was shorter: the rest is silence, so the file is the
    // length the element it belongs to is.
    bytes.fill(0);
    while written < frames {
        let take = ((frames - written) as usize).min(BLOCK);
        writer
            .write_all(&bytes[..take * per_sample])
            .map_err(|e| Error::io(destination, e))?;
        written += take as u64;
    }
    writer.flush().map_err(|e| Error::io(destination, e))?;
    Ok(clipped)
}

/// A WAV read in blocks, folded to mono.
///
/// What every measurement walks: several of these advance in step over the
/// voices of a film without any of them being held.
pub struct Reader {
    file: File,
    format: Format,
    buffer: Vec<u8>,
    left: u64,
    /// One channel alone instead of the fold; clamped to the last one, so
    /// the right side of a mono file is its only channel.
    channel: Option<usize>,
}

impl Reader {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).map_err(|e| Error::io(path, e))?;
        let length = file.metadata().map_err(|e| Error::io(path, e))?.len();
        let format = header(&mut file, length).map_err(|e| name(path, e))?;
        Ok(Self {
            file,
            format,
            buffer: vec![0u8; BLOCK * format.bytes_per_frame().max(1) as usize],
            left: format.frames,
            channel: None,
        })
    }

    /// A reader of one channel, from 0, rather than of the fold.
    pub fn open_channel(path: &Path, channel: usize) -> Result<Self> {
        let mut reader = Self::open(path)?;
        reader.channel = Some(channel.min(reader.format.channels.max(1) as usize - 1));
        Ok(reader)
    }

    pub fn format(&self) -> Format {
        self.format
    }

    /// Fill `out` with the next frames, every channel averaged into one;
    /// returns how many frames were read, 0 at the end.
    pub fn read_block(&mut self, out: &mut [f32]) -> Result<usize> {
        let frame = self.format.bytes_per_frame() as usize;
        let want = out
            .len()
            .min(self.buffer.len() / frame.max(1))
            .min(self.left as usize);
        if want == 0 {
            return Ok(0);
        }
        let need = want * frame;
        let mut filled = 0;
        while filled < need {
            let read = self
                .file
                .read(&mut self.buffer[filled..need])
                .map_err(|e| Error::msg(format!("reading samples: {e}")))?;
            if read == 0 {
                break;
            }
            filled += read;
        }
        let frames = filled / frame;
        let channels = self.format.channels as usize;
        let per_sample = self.format.bytes_per_sample() as usize;
        for (f, out) in out.iter_mut().enumerate().take(frames) {
            if let Some(c) = self.channel {
                *out = sample(
                    &self.buffer[(f * channels + c) * per_sample..],
                    self.format.bits,
                    self.format.float,
                );
                continue;
            }
            let mut total = 0.0;
            for c in 0..channels {
                total += sample(
                    &self.buffer[(f * channels + c) * per_sample..],
                    self.format.bits,
                    self.format.float,
                );
            }
            *out = if channels > 1 {
                total / channels as f32
            } else {
                total
            };
        }
        self.left -= frames as u64;
        Ok(frames)
    }
}

/// Write a WAV header; the caller writes `format.data_bytes()` after it.
pub fn write_header(out: &mut impl Write, format: Format) -> std::io::Result<()> {
    let block_align = format.bytes_per_frame() as u32;
    let data = format.data_bytes() as u32;
    out.write_all(b"RIFF")?;
    out.write_all(&(36 + data).to_le_bytes())?;
    out.write_all(b"WAVE")?;
    out.write_all(b"fmt ")?;
    out.write_all(&16u32.to_le_bytes())?;
    out.write_all(&if format.float { 3u16 } else { 1u16 }.to_le_bytes())?;
    out.write_all(&format.channels.to_le_bytes())?;
    out.write_all(&format.sample_rate.to_le_bytes())?;
    out.write_all(&(format.sample_rate * block_align).to_le_bytes())?;
    out.write_all(&(block_align as u16).to_le_bytes())?;
    out.write_all(&format.bits.to_le_bytes())?;
    out.write_all(b"data")?;
    out.write_all(&data.to_le_bytes())
}

/// One sample at the start of `bytes`, as a number in [-1, 1] — except a
/// float one, which is whatever was written.
pub(crate) fn sample(bytes: &[u8], bits: u16, float: bool) -> f32 {
    if float {
        return f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    }
    match bits {
        16 => i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32_768.0,
        24 => {
            // The three bytes land in the top of an i32, which both
            // sign-extends them and scales them by 256 — so the divisor is
            // the 32-bit one.
            i32::from_le_bytes([0, bytes[0], bytes[1], bytes[2]]) as f32 / 2_147_483_648.0
        }
        32 => i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32 / 2_147_483_648.0,
        _ => 0.0,
    }
}

/// The header of a stream `length` bytes long, from its start.
fn header(source: &mut (impl Read + Seek), length: u64) -> Result<Format> {
    let mut riff = [0u8; 12];
    source
        .read_exact(&mut riff)
        .map_err(|_| Error::msg("too short to be a WAV file"))?;
    if &riff[0..4] != b"RIFF" || &riff[8..12] != b"WAVE" {
        return Err(Error::msg("not a WAV file"));
    }
    let (mut channels, mut sample_rate, mut bits, mut float) = (0u16, 0u32, 0u16, false);
    let mut offset = 12u64;
    let mut chunk = [0u8; 8];
    while source.read_exact(&mut chunk).is_ok() {
        let size = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]) as u64;
        let body = offset + 8;
        match &chunk[0..4] {
            b"fmt " => {
                let mut fmt = vec![0u8; size.min(40) as usize];
                source
                    .read_exact(&mut fmt)
                    .map_err(|_| Error::msg("short fmt chunk"))?;
                if fmt.len() < 16 {
                    return Err(Error::msg("short fmt chunk"));
                }
                channels = u16::from_le_bytes([fmt[2], fmt[3]]);
                sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
                bits = u16::from_le_bytes([fmt[14], fmt[15]]);
                // An extensible header says PCM or float in the first two
                // bytes of its sub-format, where a plain one says it
                // outright.
                let tag = match u16::from_le_bytes([fmt[0], fmt[1]]) {
                    0xFFFE if fmt.len() >= 26 => u16::from_le_bytes([fmt[24], fmt[25]]),
                    tag => tag,
                };
                float = tag == 3;
                if tag != 1 && tag != 3 {
                    return Err(Error::msg(format!(
                        "sample format {tag}, which this reader does not decode"
                    )));
                }
            }
            b"data" => {
                if channels == 0 || bits == 0 {
                    return Err(Error::msg("data before fmt, which this reader cannot read"));
                }
                // A writer that streamed to a pipe leaves the size at zero
                // or at its maximum; what is on disk is then the truth.
                let available = length.saturating_sub(body);
                let bytes = match size {
                    0 | 0xFFFF_FFFF => available,
                    size => size.min(available),
                };
                let frame = channels as u64 * (bits as u64 / 8);
                return Ok(Format {
                    channels,
                    sample_rate,
                    bits,
                    float,
                    frames: bytes.checked_div(frame).unwrap_or(0),
                    data_offset: body,
                });
            }
            _ => {}
        }
        // Chunks are padded to an even length; seeking to the next one
        // rather than over the current one keeps a chunk that was only
        // partly read (`fmt `) from putting the reader out of step.
        offset = body + size + (size & 1);
        source
            .seek(SeekFrom::Start(offset))
            .map_err(|_| Error::msg("truncated chunk"))?;
    }
    Err(Error::msg("no data chunk"))
}

fn create_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    Ok(())
}

/// Put the file's name in front of a message that has none.
fn name(path: &Path, error: Error) -> Error {
    match error {
        Error::Message(text) => Error::msg(format!("{}: {text}", path.display())),
        error => error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A mono 24-bit header as ffmpeg writes one, optionally with the
    /// `LIST` chunk it likes to add between `fmt ` and `data`.
    fn wav(list_chunk: bool, data_size: u32, samples: usize) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend(b"RIFF");
        v.extend(0u32.to_le_bytes());
        v.extend(b"WAVE");
        v.extend(b"fmt ");
        v.extend(16u32.to_le_bytes());
        v.extend(1u16.to_le_bytes()); // PCM
        v.extend(1u16.to_le_bytes()); // mono
        v.extend(48000u32.to_le_bytes());
        v.extend(144000u32.to_le_bytes());
        v.extend(3u16.to_le_bytes());
        v.extend(24u16.to_le_bytes());
        if list_chunk {
            v.extend(b"LIST");
            v.extend(5u32.to_le_bytes());
            v.extend(b"INFOx");
            v.push(0); // padded to an even length
        }
        v.extend(b"data");
        v.extend(data_size.to_le_bytes());
        v.extend(std::iter::repeat_n(0u8, samples * 3));
        v
    }

    fn read(bytes: Vec<u8>) -> Result<Format> {
        let length = bytes.len() as u64;
        header(&mut Cursor::new(bytes), length)
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("omnifrog-wav-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A 24-bit mono file of these integer samples.
    fn mono24(path: &Path, samples: &[i32]) {
        let mut out = Vec::new();
        write_header(
            &mut out,
            Format {
                channels: 1,
                sample_rate: 48_000,
                bits: 24,
                float: false,
                frames: samples.len() as u64,
                data_offset: 0,
            },
        )
        .unwrap();
        for s in samples {
            out.extend(&s.to_le_bytes()[..3]);
        }
        std::fs::write(path, out).unwrap();
    }

    /// A float stereo file, as the worker writes a stem.
    fn stereo_float(path: &Path, left: &[f32], right: &[f32]) {
        let mut out = Vec::new();
        write_header(
            &mut out,
            Format {
                channels: 2,
                sample_rate: 44_100,
                bits: 32,
                float: true,
                frames: left.len() as u64,
                data_offset: 0,
            },
        )
        .unwrap();
        for (l, r) in left.iter().zip(right) {
            out.extend(l.to_le_bytes());
            out.extend(r.to_le_bytes());
        }
        std::fs::write(path, out).unwrap();
    }

    fn floats(path: &Path) -> Vec<f32> {
        let mut reader = Reader::open(path).unwrap();
        let mut out = vec![0f32; reader.format().frames as usize];
        let read = reader.read_block(&mut out).unwrap();
        out.truncate(read);
        out
    }

    #[test]
    fn a_pair_interleaves_two_mono_files_byte_for_byte() {
        let dir = scratch("pair");
        let (l, r, both) = (dir.join("l.wav"), dir.join("r.wav"), dir.join("lr.wav"));
        mono24(&l, &[1, -2, 3, 4]);
        mono24(&r, &[100, 200, -300]);
        let format = pair(&l, &r, &both, 44_100).unwrap();
        assert_eq!(
            (format.channels, format.sample_rate, format.frames),
            (2, 44_100, 3)
        );
        assert_eq!(
            read_header(&both).unwrap(),
            Format {
                data_offset: 44,
                ..format
            }
        );
        let bytes = std::fs::read(&both).unwrap();
        let samples: Vec<i32> = bytes[44..]
            .chunks(3)
            .map(|b| i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8)
            .collect();
        assert_eq!(samples, [1, 100, -2, 200, 3, -300]);
        // Two formats cannot be interleaved.
        let float = dir.join("f.wav");
        stereo_float(&float, &[0.0], &[0.0]);
        assert!(pair(&l, &float, &both, 44_100).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn one_side_of_a_stem_or_the_average_of_two() {
        let dir = scratch("pick");
        let (a, b, out) = (dir.join("a.wav"), dir.join("b.wav"), dir.join("out.wav"));
        stereo_float(&a, &[0.5, -0.25, 0.125], &[0.1, 0.2, 0.3]);
        stereo_float(&b, &[0.0, 0.0, 0.0], &[0.3, -0.25, 0.125]);
        let write = |picks: &[Pick]| {
            mix_mono(picks, &out, 3, 48_000, 1.0, Sample::Float).unwrap();
            floats(&out)
        };
        assert_eq!(write(&[Pick::Channel(&a, 0)]), [0.5, -0.25, 0.125]);
        assert_eq!(write(&[Pick::Channel(&a, 1)]), [0.1, 0.2, 0.3]);
        let close = |got: Vec<f32>, want: [f32; 3]| {
            assert!(
                got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-6),
                "{got:?} != {want:?}"
            );
        };
        // The fold of the pair is its two sides averaged, as before.
        close(write(&[Pick::Fold(&a)]), [0.3, -0.025, 0.2125]);
        // The dub's centre: the left of one pass and the right of another.
        close(
            write(&[Pick::Channel(&a, 0), Pick::Channel(&b, 1)]),
            [0.4, -0.25, 0.125],
        );
        // A side past the last one is the last one.
        assert_eq!(write(&[Pick::Channel(&a, 5)]), [0.1, 0.2, 0.3]);
        assert!(mix_mono(&[], &out, 3, 48_000, 1.0, Sample::Float).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_chunk_between_fmt_and_data_is_stepped_over() {
        let format = read(wav(true, 300, 100)).unwrap();
        assert_eq!(format.channels, 1);
        assert_eq!(format.sample_rate, 48000);
        assert_eq!(format.bits, 24);
        assert!(!format.float);
        assert_eq!(format.frames, 100);
        assert_eq!(format.bytes_per_frame(), 3);
    }

    #[test]
    fn a_size_left_unwritten_is_taken_from_the_file() {
        assert_eq!(read(wav(false, 0xFFFF_FFFF, 50)).unwrap().frames, 50);
        assert_eq!(read(wav(false, 0, 50)).unwrap().frames, 50);
    }

    #[test]
    fn a_truncated_file_counts_the_frames_it_really_holds() {
        assert_eq!(read(wav(false, 300, 40)).unwrap().frames, 40);
    }

    #[test]
    fn something_that_is_not_a_wav_is_refused() {
        assert!(read(b"not a wav at all, not even close".to_vec()).is_err());
        assert!(read(b"RIF".to_vec()).is_err());
    }

    #[test]
    fn a_header_this_crate_wrote_reads_back() {
        for (bits, float) in [(24u16, false), (32, true), (16, false)] {
            let mut bytes = Vec::new();
            let written = Format {
                channels: 2,
                sample_rate: 44100,
                bits,
                float,
                frames: 7,
                data_offset: 0,
            };
            write_header(&mut bytes, written).unwrap();
            bytes.extend(std::iter::repeat_n(0u8, written.data_bytes() as usize));
            let back = read(bytes).unwrap();
            assert_eq!(back.channels, 2);
            assert_eq!(back.sample_rate, 44100);
            assert_eq!(back.bits, bits);
            assert_eq!(back.float, float);
            assert_eq!(back.frames, 7);
        }
    }

    /// The 24-bit samples of a mono file `copy_mono` wrote, and how many it
    /// clipped, from float `samples`.
    fn copied(name: &str, samples: &[f32], form: Sample) -> (Vec<i32>, u64) {
        let dir = std::env::temp_dir().join(format!("omnifrog-wav-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (source, destination) = (dir.join("float.wav"), dir.join("pcm.wav"));
        let mut bytes = Vec::new();
        let format = Format {
            channels: 1,
            sample_rate: 48000,
            bits: 32,
            float: true,
            frames: samples.len() as u64,
            data_offset: 0,
        };
        write_header(&mut bytes, format).unwrap();
        for s in samples {
            bytes.extend(s.to_le_bytes());
        }
        std::fs::write(&source, bytes).unwrap();
        let frames = samples.len() as u64;
        let clipped = copy_mono(&source, &destination, frames, 48000, 1.0, form).unwrap();
        let offset = read_header(&destination).unwrap().data_offset as usize;
        let read = std::fs::read(&destination).unwrap();
        let (triples, _) = read[offset..].as_chunks::<3>();
        let written = triples
            .iter()
            .map(|b| i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8)
            .collect();
        let _ = std::fs::remove_dir_all(&dir);
        (written, clipped)
    }

    #[test]
    fn a_stem_is_rounded_to_the_depth_it_is_kept_at() {
        let samples = [0.3f32, -0.3, 1e-7, -1.0, 0.999_999, 0.5];
        let (full, clipped) = copied("24", &samples, Sample::Pcm(24));
        assert_eq!(
            full,
            [2_516_583, -2_516_583, 1, -8_388_608, 8_388_600, 4_194_304]
        );
        assert_eq!(clipped, 0);
        // At 18 bits a step is 64: every sample lands on it, the nearest
        // one, and the top of the grid is a step under full scale.
        let (rounded, clipped) = copied("18", &samples, Sample::Pcm(18));
        assert_eq!(
            rounded,
            [2_516_608, -2_516_608, 0, -8_388_608, 8_388_544, 4_194_304]
        );
        assert!(rounded.iter().all(|s| s % 64 == 0));
        assert_eq!(clipped, 1);
    }

    #[test]
    fn samples_decode_to_the_same_number_whatever_the_depth() {
        // Half of full scale, as each depth writes it.
        assert!((sample(&[0x00, 0x40], 16, false) - 0.5).abs() < 1e-6);
        assert!((sample(&[0x00, 0x00, 0x40], 24, false) - 0.5).abs() < 1e-6);
        assert!((sample(&[0x00, 0x00, 0x00, 0x40], 32, false) - 0.5).abs() < 1e-6);
        assert!((sample(&0.5f32.to_le_bytes(), 32, true) - 0.5).abs() < 1e-9);
        // And the sign survives the three-byte case.
        assert!((sample(&[0x00, 0x00, 0xC0], 24, false) + 0.5).abs() < 1e-6);
    }
}
