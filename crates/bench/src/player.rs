//! The player's engine: what the audio callback plays, and what the window
//! tells it.
//!
//! One track at a time, held whole in memory — a zone is short. Changing the
//! track keeps the position and the state, so two outputs are compared at
//! the very same sample, with a few milliseconds of crossfade instead of a
//! click. The loop is exact: the sample after the last of `[A, B)` is the
//! first. A track may carry a second signal added at a gain — the rest of
//! one model with the voice of another, mixed as it plays, no file written.
//!
//! Nothing here opens a device: [`Source::fill`] is what a callback calls,
//! and what the tests call.

use crate::set::RATE;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Frames a change of track is faded over: 5 ms.
pub const CROSSFADE: usize = 240;

/// What is played: a signal, and perhaps another added to it.
#[derive(Debug, Clone, Default)]
pub struct Track {
    pub base: Arc<[f32]>,
    /// A second signal and its gain, added over the length of the first.
    pub overlay: Option<(Arc<[f32]>, f32)>,
}

impl Track {
    pub fn of(base: Arc<[f32]>) -> Self {
        Self {
            base,
            overlay: None,
        }
    }

    pub fn len(&self) -> usize {
        self.base.len()
    }

    pub fn is_empty(&self) -> bool {
        self.base.is_empty()
    }

    fn sample(&self, at: usize) -> f32 {
        let base = self.base.get(at).copied().unwrap_or(0.0);
        match &self.overlay {
            Some((overlay, gain)) => base + gain * overlay.get(at).copied().unwrap_or(0.0),
            None => base,
        }
    }
}

/// The gain that brings a voice to the level of another: of the voice the
/// rest's own model took out, as its mix does. No gain when either loudness
/// is unknown.
pub fn mix_gain(reference_lkfs: Option<f64>, voice_lkfs: Option<f64>) -> f32 {
    match (reference_lkfs, voice_lkfs) {
        (Some(reference), Some(voice)) if reference.is_finite() && voice.is_finite() => {
            10f64.powf((reference - voice) / 20.0) as f32
        }
        _ => 1.0,
    }
}

struct Shared {
    track: Mutex<Arc<Track>>,
    /// Counts the tracks loaded: the callback takes the lock only when it
    /// has changed.
    generation: AtomicU64,
    playing: AtomicBool,
    /// Where the next frame is read, in frames of the track.
    position: AtomicU64,
    /// Counts the seeks, so a callback that was running does not write its
    /// own position over one.
    seeks: AtomicU64,
    loop_start: AtomicU64,
    /// Past the last frame of the loop; `u64::MAX`: the end of the track.
    loop_end: AtomicU64,
}

/// The window's side of the player.
#[derive(Clone)]
pub struct Player {
    shared: Arc<Shared>,
}

impl Default for Player {
    fn default() -> Self {
        Self::new()
    }
}

impl Player {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Shared {
                track: Mutex::new(Arc::new(Track::default())),
                generation: AtomicU64::new(0),
                playing: AtomicBool::new(false),
                position: AtomicU64::new(0),
                seeks: AtomicU64::new(0),
                loop_start: AtomicU64::new(0),
                loop_end: AtomicU64::new(u64::MAX),
            }),
        }
    }

    /// The callback's side. One per stream.
    pub fn source(&self) -> Source {
        Source {
            shared: self.shared.clone(),
            track: Arc::new(Track::default()),
            generation: 0,
            previous: None,
            fading: 0,
            fraction: 0.0,
        }
    }

    /// Play this track from where the last one was, playing or paused as
    /// it was.
    pub fn load(&self, track: Track) {
        *self.shared.track.lock().unwrap_or_else(|e| e.into_inner()) = Arc::new(track);
        self.shared.generation.fetch_add(1, Ordering::Release);
    }

    pub fn len(&self) -> usize {
        self.shared
            .track
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn play(&self) {
        self.shared.playing.store(true, Ordering::Relaxed);
    }

    pub fn pause(&self) {
        self.shared.playing.store(false, Ordering::Relaxed);
    }

    pub fn playing(&self) -> bool {
        self.shared.playing.load(Ordering::Relaxed)
    }

    /// The frame played next.
    pub fn position(&self) -> u64 {
        self.shared.position.load(Ordering::Relaxed)
    }

    pub fn seek(&self, frame: u64) {
        self.shared.position.store(frame, Ordering::Relaxed);
        self.shared.seeks.fetch_add(1, Ordering::Release);
    }

    /// Loop over `[start, end)`; none: over the whole track. A position
    /// outside the new loop goes to its start.
    pub fn set_loop(&self, range: Option<(u64, u64)>) {
        let (start, end) = match range {
            Some((a, b)) if b > a => (a, b),
            _ => (0, u64::MAX),
        };
        self.shared.loop_start.store(start, Ordering::Relaxed);
        self.shared.loop_end.store(end, Ordering::Relaxed);
        let position = self.position();
        if position < start || position >= end {
            self.seek(start);
        }
    }

    /// The loop, when it is not the whole track.
    pub fn looped(&self) -> Option<(u64, u64)> {
        let start = self.shared.loop_start.load(Ordering::Relaxed);
        let end = self.shared.loop_end.load(Ordering::Relaxed);
        (end != u64::MAX).then_some((start, end))
    }
}

/// The callback's side of the player: it owns the track it plays, and looks
/// at the shared state once per buffer.
pub struct Source {
    shared: Arc<Shared>,
    track: Arc<Track>,
    generation: u64,
    /// The track before this one, while the change is faded.
    previous: Option<Arc<Track>>,
    fading: usize,
    /// Where between two frames the device's clock is, when its rate is not
    /// the track's.
    fraction: f64,
}

impl Source {
    /// Fill an interleaved buffer of `channels` channels at `rate`: the
    /// track on every channel, silence when paused or with nothing loaded.
    pub fn fill(&mut self, out: &mut [f32], channels: usize, rate: u32) {
        let generation = self.shared.generation.load(Ordering::Acquire);
        if generation != self.generation {
            // A lock held by the window for the time of one pointer copy;
            // if it is held right now, the next buffer takes the new track.
            if let Ok(track) = self.shared.track.try_lock() {
                let old = std::mem::replace(&mut self.track, track.clone());
                self.generation = generation;
                if !old.is_empty() {
                    self.previous = Some(old);
                    self.fading = CROSSFADE;
                }
            }
        }
        if !self.shared.playing.load(Ordering::Relaxed) || self.track.is_empty() || channels == 0 {
            out.fill(0.0);
            return;
        }
        let seeks = self.shared.seeks.load(Ordering::Acquire);
        let length = self.track.len() as u64;
        let start = self
            .shared
            .loop_start
            .load(Ordering::Relaxed)
            .min(length - 1);
        let end = self
            .shared
            .loop_end
            .load(Ordering::Relaxed)
            .min(length)
            .max(start + 1);
        let mut position = self.shared.position.load(Ordering::Relaxed);
        if position < start || position >= end {
            position = start;
        }
        let step = RATE as f64 / rate.max(1) as f64;
        for frame in out.chunks_exact_mut(channels) {
            let next = if position + 1 >= end {
                start
            } else {
                position + 1
            };
            let mut sample = self.value(position as usize, next as usize);
            if let Some(previous) = &self.previous {
                let old = previous.sample(position as usize);
                let share = self.fading as f32 / CROSSFADE as f32;
                sample = share * old + (1.0 - share) * sample;
                self.fading -= 1;
                if self.fading == 0 {
                    self.previous = None;
                }
            }
            frame.fill(sample);
            self.fraction += step;
            while self.fraction >= 1.0 {
                self.fraction -= 1.0;
                position = if position + 1 >= end {
                    start
                } else {
                    position + 1
                };
            }
        }
        // Unless the window sought meanwhile: its position stands.
        if self.shared.seeks.load(Ordering::Acquire) == seeks {
            self.shared.position.store(position, Ordering::Relaxed);
        }
    }

    /// The track at a frame; between two frames when the device runs at
    /// another rate.
    fn value(&self, at: usize, next: usize) -> f32 {
        let here = self.track.sample(at);
        if self.fraction == 0.0 {
            return here;
        }
        here + self.fraction as f32 * (self.track.sample(next) - here)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize, from: f32) -> Arc<[f32]> {
        (0..n).map(|i| from + i as f32).collect()
    }

    #[test]
    fn nothing_plays_until_asked() {
        let player = Player::new();
        let mut source = player.source();
        let mut out = [9.0f32; 8];
        source.fill(&mut out, 2, RATE);
        assert_eq!(out, [0.0; 8]);
        player.load(Track::of(ramp(100, 0.0)));
        source.fill(&mut out, 2, RATE);
        assert_eq!(out, [0.0; 8]);
        assert_eq!(player.position(), 0);
        player.play();
        source.fill(&mut out, 2, RATE);
        assert_eq!(out, [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        assert_eq!(player.position(), 4);
        player.pause();
        source.fill(&mut out, 2, RATE);
        assert_eq!(out, [0.0; 8]);
        assert_eq!(player.position(), 4);
    }

    #[test]
    fn the_loop_is_exact() {
        let player = Player::new();
        let mut source = player.source();
        player.load(Track::of(ramp(100, 0.0)));
        player.set_loop(Some((10, 13)));
        assert_eq!(player.position(), 10);
        player.play();
        let mut out = [0.0f32; 8];
        source.fill(&mut out, 1, RATE);
        assert_eq!(out, [10.0, 11.0, 12.0, 10.0, 11.0, 12.0, 10.0, 11.0]);
        assert_eq!(player.position(), 12);
        assert_eq!(player.looped(), Some((10, 13)));
        // No loop: the whole track, over and over.
        player.set_loop(None);
        assert_eq!(player.looped(), None);
        player.seek(98);
        source.fill(&mut out[..4], 1, RATE);
        assert_eq!(&out[..4], [98.0, 99.0, 0.0, 1.0]);
    }

    #[test]
    fn another_track_takes_over_at_the_same_sample() {
        let player = Player::new();
        let mut source = player.source();
        player.load(Track::of(ramp(2000, 0.0)));
        player.play();
        let mut out = vec![0.0f32; 500];
        source.fill(&mut out, 1, RATE);
        // The second track is the first plus a thousand.
        player.load(Track::of(ramp(2000, 1000.0)));
        source.fill(&mut out, 1, RATE);
        assert_eq!(player.position(), 1000);
        // Faded in from the first, at the same position…
        assert_eq!(out[0], 500.0);
        assert!(out[1] > 501.0 && out[1] < 1501.0);
        // …and wholly the second once the fade is over.
        assert_eq!(out[CROSSFADE], 1500.0 + CROSSFADE as f32);
        assert_eq!(out[499], 1999.0);
    }

    #[test]
    fn a_seek_made_during_a_buffer_stands() {
        let player = Player::new();
        let mut source = player.source();
        player.load(Track::of(ramp(100, 0.0)));
        player.play();
        player.seek(40);
        let mut out = [0.0f32; 4];
        source.fill(&mut out, 1, RATE);
        assert_eq!(out, [40.0, 41.0, 42.0, 43.0]);
        // A position past the end is brought back into the loop.
        player.seek(5000);
        source.fill(&mut out, 1, RATE);
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn a_second_signal_is_added_at_its_gain() {
        let player = Player::new();
        let mut source = player.source();
        let voice: Arc<[f32]> = vec![10.0f32; 3].into();
        player.load(Track {
            base: ramp(5, 0.0),
            overlay: Some((voice, 0.5)),
        });
        player.play();
        let mut out = [0.0f32; 5];
        source.fill(&mut out, 1, RATE);
        // The length is the first signal's.
        assert_eq!(out, [5.0, 6.0, 7.0, 3.0, 4.0]);
        // Six decibels up: twice the amplitude, near enough.
        assert!((mix_gain(Some(-30.0), Some(-36.0)) - 1.9953).abs() < 1e-4);
        assert_eq!(mix_gain(None, Some(-36.0)), 1.0);
        assert_eq!(mix_gain(Some(f64::NAN), Some(-36.0)), 1.0);
    }

    #[test]
    fn a_device_at_another_rate_is_fed_at_the_track_s_speed() {
        let player = Player::new();
        let mut source = player.source();
        player.load(Track::of(ramp(48_000, 0.0)));
        player.play();
        // A second of a 96 kHz device is a second of the track.
        let mut out = vec![0.0f32; 96_000];
        source.fill(&mut out, 1, 96_000);
        assert_eq!(player.position(), 0);
        assert_eq!(&out[..5], [0.0, 0.5, 1.0, 1.5, 2.0]);
        // And one of a 44.1 kHz device too, within a frame.
        let mut out = vec![0.0f32; 44_100];
        source.fill(&mut out, 1, 44_100);
        assert!(player.position() <= 1 || player.position() == 47_999);
    }
}
