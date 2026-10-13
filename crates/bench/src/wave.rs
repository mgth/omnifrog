//! What a waveform shows: the envelope of a signal, and the tint of a
//! column by its gap to the reference.

/// Columns an envelope is kept at: fine enough for any window, and a
/// thousand times smaller than the samples it stands for.
pub const COLUMNS: usize = 4096;

/// The lowest and highest sample of each of `columns` equal parts of a
/// signal. What is kept of a track that is shown and not played.
pub fn envelope(samples: &[f32], columns: usize) -> Vec<(f32, f32)> {
    if samples.is_empty() || columns == 0 {
        return Vec::new();
    }
    (0..columns)
        .map(|column| {
            let from = column * samples.len() / columns;
            let to = ((column + 1) * samples.len() / columns).max(from + 1);
            samples[from..to.min(samples.len())]
                .iter()
                .fold((f32::MAX, f32::MIN), |(low, high), &s| {
                    (low.min(s), high.max(s))
                })
        })
        .collect()
}

/// The lowest and highest of an envelope over the columns `[from, to)` of
/// a view of `width` columns: an envelope drawn at any width.
pub fn span(envelope: &[(f32, f32)], column: usize, width: usize) -> (f32, f32) {
    if envelope.is_empty() || width == 0 {
        return (0.0, 0.0);
    }
    let from = (column * envelope.len() / width).min(envelope.len() - 1);
    let to = ((column + 1) * envelope.len() / width).clamp(from + 1, envelope.len());
    envelope[from..to]
        .iter()
        .fold((f32::MAX, f32::MIN), |(low, high), &(l, h)| {
            (low.min(l), high.max(h))
        })
}

/// The scales of the tint, in dB: from where a colour starts to where it is
/// full. An excess is by nature more discreet than a lack — on the test set
/// the references stay under about 1 dB of excess and 2 dB of lack on nine
/// frames out of ten, the rejects for a lack reach 7 to 11 dB of it, those
/// for a leak 1.5 to 4 dB of excess — so each has its own.
pub const EXCESS: (f32, f32) = (1.0, 6.0);
pub const LACK: (f32, f32) = (2.0, 12.0);
pub const TOTAL: (f32, f32) = (2.0, 12.0);
/// Steps a tint is quantised to.
pub const LEVELS: u8 = 16;

/// How a column is tinted: how strongly, and how much of it is the colour
/// of an excess rather than of a lack.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Tint {
    /// 0: the waveform's own colour; 1: the tint alone.
    pub strength: f32,
    /// 1: all excess (red); 0: all lack (yellow).
    pub excess: f32,
}

fn level(value: f32, (min, max): (f32, f32)) -> u8 {
    (((value - min) / (max - min)).clamp(0.0, 1.0) * LEVELS as f32).round() as u8
}

/// The tint of a column from its gap in each direction. With two colours,
/// each direction on its own scale, both at once giving an orange at the
/// strength of the stronger; with one, their sum in red.
pub fn tint(excess: f32, lack: f32, two_colours: bool) -> Tint {
    let (e, l) = if two_colours {
        (level(excess, EXCESS), level(lack, LACK))
    } else {
        (level(excess + lack, TOTAL), 0)
    };
    Tint {
        strength: e.max(l) as f32 / LEVELS as f32,
        excess: if e + l == 0 {
            1.0
        } else {
            e as f32 / (e + l) as f32
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_envelope_keeps_the_extremes_of_each_part() {
        let samples = [0.0, 0.5, -0.25, 0.1, 0.9, -0.8, 0.2, 0.3];
        assert_eq!(
            envelope(&samples, 4),
            [(0.0, 0.5), (-0.25, 0.1), (-0.8, 0.9), (0.2, 0.3)]
        );
        // More columns than samples: each sample is its own extremes.
        assert_eq!(envelope(&samples[..2], 4)[3], (0.5, 0.5));
        assert!(envelope(&[], 4).is_empty());
    }

    #[test]
    fn an_envelope_is_drawn_at_any_width() {
        let e = envelope(&[0.0, 0.5, -0.25, 0.1, 0.9, -0.8, 0.2, 0.3], 4);
        assert_eq!(span(&e, 0, 2), (-0.25, 0.5));
        assert_eq!(span(&e, 1, 2), (-0.8, 0.9));
        // Wider than the envelope: columns repeat.
        assert_eq!(span(&e, 0, 8), (0.0, 0.5));
        assert_eq!(span(&e, 7, 8), (0.2, 0.3));
        assert_eq!(span(&[], 0, 8), (0.0, 0.0));
    }

    #[test]
    fn a_tint_starts_and_saturates_on_its_own_scale() {
        assert_eq!(tint(0.5, 1.0, true).strength, 0.0);
        let leak = tint(6.0, 0.0, true);
        assert_eq!((leak.strength, leak.excess), (1.0, 1.0));
        let lack = tint(0.0, 7.0, true);
        assert_eq!((lack.strength, lack.excess), (0.5, 0.0));
        // Both at once: as strong as the stronger, a colour between.
        let both = tint(6.0, 12.0, true);
        assert_eq!((both.strength, both.excess), (1.0, 0.5));
        // One colour: the sum, all of it "excess".
        let sum = tint(3.0, 4.0, false);
        assert_eq!((sum.strength, sum.excess), (0.5, 1.0));
    }
}
