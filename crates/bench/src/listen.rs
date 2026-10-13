//! Listening to a zone: what is loaded in the player, what the waveforms
//! show, and the ratings given while listening.

use crate::app::App;
use crate::output::{self, Choice, Output};
use crate::prefs::Prefs;
use omnifrog_bench::listing::Row;
use omnifrog_bench::metrics::{self, Gaps};
use omnifrog_bench::player::{self, Player, Track};
use omnifrog_bench::set::{Side, Status, Stem, Zone};
use omnifrog_bench::{audio, layout, rate, wave};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

/// One of the five files of an output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Stem(Stem),
    Mix,
}

impl Kind {
    pub const ALL: [Kind; 5] = [
        Kind::Stem(Stem::VoiceVo),
        Kind::Stem(Stem::RestVo),
        Kind::Stem(Stem::VoiceVf),
        Kind::Stem(Stem::RestVf),
        Kind::Mix,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Kind::Stem(Stem::VoiceVo) => "Voice VO",
            Kind::Stem(Stem::RestVo) => "Rest VO",
            Kind::Stem(Stem::VoiceVf) => "Voice VF",
            Kind::Stem(Stem::RestVf) => "Rest VF",
            Kind::Mix => "Mix",
        }
    }

    /// The input the top waveform shows beside this kind of output.
    pub fn side(self) -> Side {
        match self {
            Kind::Stem(stem) => stem.side(),
            Kind::Mix => Side::Vo,
        }
    }

    pub fn path(self, dir: &Path, zone: &Zone, key: &str) -> PathBuf {
        match self {
            Kind::Stem(stem) => layout::output(dir, zone, key, stem),
            Kind::Mix => layout::mix(dir, zone, key),
        }
    }

    pub fn present(self, row: &Row) -> bool {
        match self {
            Kind::Stem(stem) => row.stems[Stem::ALL.iter().position(|s| *s == stem).unwrap_or(0)],
            Kind::Mix => row.mix,
        }
    }

    pub fn loudness(self, row: &Row) -> Option<f64> {
        match self {
            Kind::Stem(stem) => row.loudness.of(stem),
            Kind::Mix => row.loudness.mix(),
        }
    }
}

/// What the player holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Loaded {
    /// An input of the zone, as it was before any model.
    Original(Side),
    Output {
        key: String,
        kind: Kind,
    },
    /// The rest of one output with the voice of another, mixed as it plays.
    Blend,
}

/// What a waveform shows of a file: its envelope, and its gap to the
/// reference when there is one.
pub struct Wave {
    pub envelope: Vec<(f32, f32)>,
    pub frames: usize,
    pub gaps: Option<Gaps>,
}

type Waves = (u64, Option<Wave>, Vec<(String, Wave)>);

pub struct Listen {
    pub player: Player,
    /// The device, once something was played; an error if it did not open.
    pub output: Option<Result<Output, String>>,
    /// The output chosen; the machine's default when there is none.
    pub chosen: Option<Choice>,
    /// The outputs there were when the list was last opened.
    pub choices: Vec<Choice>,
    /// Whether the list was open at the last frame: the outputs are asked
    /// for when it opens, not at every frame it stays so.
    pub listed: bool,
    /// The file the choice is kept in; none, and it is not kept.
    pub prefs: Option<PathBuf>,
    pub loaded: Option<Loaded>,
    /// The two places of the blend: a rest, a voice. Both taken, they play
    /// together.
    pub blend: [Option<(String, Stem)>; 2],
    /// The kind of file the rows' waveforms show: that of the last one
    /// played.
    pub kind: Option<Kind>,
    pub original: Option<Wave>,
    pub waves: HashMap<String, Wave>,
    /// The height every waveform has, set with the wheel on the top one.
    pub height: f32,
    pub two_colours: bool,
    /// Where a drag on the top waveform started, in frames.
    pub drag_from: Option<u64>,
    /// Counts the loads asked for: what an older one sends is dropped.
    asked: u64,
    tx: Sender<Waves>,
    rx: Receiver<Waves>,
}

pub const HEIGHT: (f32, f32, f32) = (50.0, 200.0, 15.0);

impl Default for Listen {
    fn default() -> Self {
        let (tx, rx) = channel();
        Self {
            player: Player::new(),
            output: None,
            chosen: None,
            choices: Vec::new(),
            listed: false,
            prefs: None,
            loaded: None,
            blend: [None, None],
            kind: None,
            original: None,
            waves: HashMap::new(),
            height: 65.0,
            two_colours: true,
            drag_from: None,
            asked: 0,
            tx,
            rx,
        }
    }
}

impl Listen {
    pub fn blended(&self, key: &str, stem: Stem) -> bool {
        self.blend
            .iter()
            .flatten()
            .any(|(k, s)| k == key && *s == stem)
    }
}

fn wave(samples: &[f32], reference: Option<&[f32]>) -> Wave {
    Wave {
        envelope: wave::envelope(samples, wave::COLUMNS),
        frames: samples.len(),
        // Shifted back by its lag first, as a measure is: an output that is
        // only late is no redder here than in the report.
        gaps: reference.map(|reference| {
            let lag = metrics::lag(reference, samples);
            metrics::gaps(reference, &metrics::shifted(samples, lag))
        }),
    }
}

impl App {
    /// Another zone, or none: the player lets go of the last one.
    pub fn listen_reset(&mut self) {
        let listen = &mut self.listen;
        listen.player.pause();
        listen.player.load(Track::default());
        listen.player.set_loop(None);
        listen.player.seek(0);
        listen.loaded = None;
        listen.blend = [None, None];
        listen.kind = None;
        listen.original = None;
        listen.waves.clear();
        listen.asked += 1;
    }

    /// Play or pause. The device is opened the first time, and asked for
    /// again as long as it did not open: it may have been plugged in since.
    pub fn toggle_play(&mut self) {
        if self.listen.player.playing() {
            self.listen.player.pause();
            return;
        }
        if matches!(self.listen.output, Some(Ok(_))) || self.open_output() {
            self.listen.player.play();
        }
    }

    fn open_output(&mut self) -> bool {
        let output = output::open(&self.listen.player, self.listen.chosen.as_ref());
        if let Err(e) = &output {
            self.status = format!("Audio: {e}");
        }
        let opened = output.is_ok();
        self.listen.output = Some(output);
        opened
    }

    /// What this machine remembers, read from the file it is kept in from
    /// now on.
    pub fn remember_in(&mut self, path: PathBuf) {
        self.listen.chosen = Prefs::load(&path).output;
        self.listen.prefs = Some(path);
    }

    /// Ask the system which outputs there are. Nothing is opened.
    pub fn list_outputs(&mut self) {
        let choices = output::choices().unwrap_or_else(|e| {
            self.status = format!("Audio: {e}");
            Vec::new()
        });
        self.outputs_listed(choices);
    }

    /// Take the outputs the system gave. The one chosen is found among
    /// them by its id; the name it has now replaces the one that was kept.
    pub fn outputs_listed(&mut self, choices: Vec<Choice>) {
        let renamed = self.listen.chosen.as_ref().and_then(|chosen| {
            choices
                .iter()
                .find(|c| c.is(chosen) && c.name != chosen.name)
                .cloned()
        });
        self.listen.choices = choices;
        if renamed.is_some() {
            self.listen.chosen = renamed;
            self.keep_output();
        }
    }

    /// Whether the output chosen is one of those last listed; the default
    /// always is.
    pub fn output_listed(&self) -> bool {
        self.listen
            .chosen
            .as_ref()
            .is_none_or(|chosen| self.listen.choices.iter().any(|c| c.is(chosen)))
    }

    fn keep_output(&mut self) {
        if let Some(path) = &self.listen.prefs {
            let prefs = Prefs {
                output: self.listen.chosen.clone(),
            };
            if let Err(e) = prefs.save(path) {
                self.status = format!("Preferences: {e}");
            }
        }
    }

    /// Choose where the sound goes, `None` being the machine's default. A
    /// stream that is open moves there at once, and plays on if it was
    /// playing; a bench that holds no device still holds none.
    pub fn choose_output(&mut self, choice: Option<Choice>) {
        let same = match (&self.listen.chosen, &choice) {
            (Some(chosen), Some(choice)) => chosen.is(choice),
            (None, None) => true,
            _ => false,
        };
        if same {
            return;
        }
        self.listen.chosen = choice;
        self.status.clear();
        self.keep_output();
        let open = matches!(self.listen.output, Some(Ok(_)));
        // The stream that was open goes before the next one comes: two
        // would both move the position.
        self.listen.output = None;
        if open && !self.open_output() {
            self.listen.player.pause();
        }
    }

    fn read(&mut self, path: &Path) -> Option<Arc<[f32]>> {
        match audio::read(path) {
            Ok(samples) => Some(samples.into()),
            Err(e) => {
                self.status = e.to_string();
                None
            }
        }
    }

    /// Load an input of the zone as it was before any model.
    pub fn listen_original(&mut self, side: Side) {
        let Some(zone) = self.zone_shown() else {
            return;
        };
        let path = layout::extract(&self.dir, zone, side);
        let Some(samples) = self.read(&path) else {
            return;
        };
        self.listen.player.load(Track::of(samples));
        self.listen.loaded = Some(Loaded::Original(side));
        self.listen.blend = [None, None];
        self.show(Kind::Stem(side.listened()));
    }

    /// A click on one of the five buttons of a row: load that file, playing
    /// or paused as the player was, at the same position.
    pub fn listen_output(&mut self, key: &str, kind: Kind) {
        let Some(zone) = self.zone_shown() else {
            return;
        };
        let path = kind.path(&self.dir, zone, key);
        let Some(samples) = self.read(&path) else {
            return;
        };
        self.listen.player.load(Track::of(samples));
        self.listen.loaded = Some(Loaded::Output {
            key: key.to_string(),
            kind,
        });
        self.listen.blend = [None, None];
        self.show(kind);
    }

    /// A Ctrl+click on a stem: put it in its place of the blend, or take it
    /// out. A rest and a voice, of two models if one likes: the voice is
    /// brought to the level of the voice the rest's own model took out, as
    /// that model's mix does, and the two play together.
    pub fn listen_blend(&mut self, key: &str, stem: Stem) {
        // The file playing joins the blend it is the other half of.
        if self.listen.blend.iter().all(Option::is_none)
            && let Some(Loaded::Output {
                key: playing,
                kind: Kind::Stem(other),
            }) = self.listen.loaded.clone()
            && other.is_voice() != stem.is_voice()
        {
            self.listen.blend[usize::from(other.is_voice())] = Some((playing, other));
        }
        let place = usize::from(stem.is_voice());
        let chosen = Some((key.to_string(), stem));
        self.listen.blend[place] = if self.listen.blend[place] == chosen {
            None
        } else {
            chosen
        };
        match self.listen.blend.clone() {
            [Some((rest_key, rest)), Some((voice_key, voice))] => {
                let Some(zone) = self.zone_shown().cloned() else {
                    return;
                };
                let lkfs = |key: &str, stem: Stem| {
                    self.rows
                        .iter()
                        .find(|row| row.key == key)
                        .and_then(|row| row.loudness.of(stem))
                };
                let gain = player::mix_gain(lkfs(&rest_key, rest.other()), lkfs(&voice_key, voice));
                let base = self.read(&layout::output(&self.dir, &zone, &rest_key, rest));
                let over = self.read(&layout::output(&self.dir, &zone, &voice_key, voice));
                let (Some(base), Some(over)) = (base, over) else {
                    return;
                };
                self.listen.player.load(Track {
                    base,
                    overlay: Some((over, gain)),
                });
                self.listen.loaded = Some(Loaded::Blend);
                self.status = format!(
                    "Blend: {} of {rest_key} + {} of {voice_key} at {:+.1} dB",
                    Kind::Stem(rest).label(),
                    Kind::Stem(voice).label(),
                    20.0 * gain.log10()
                );
                self.show(Kind::Mix);
            }
            [Some((key, stem)), None] | [None, Some((key, stem))] => {
                let blend = self.listen.blend.clone();
                self.listen_output(&key, Kind::Stem(stem));
                self.listen.blend = blend;
            }
            [None, None] => {}
        }
    }

    /// Show the waveforms of a kind of file: every row's, and the input of
    /// that side on top. Read on a thread of their own; the sound has
    /// already changed.
    fn show(&mut self, kind: Kind) {
        if self.listen.kind == Some(kind) && !self.listen.waves.is_empty() {
            return;
        }
        self.listen.kind = Some(kind);
        self.load_waves();
    }

    /// Read the waveforms again: the kind shown changed, or the reference
    /// they are tinted against.
    pub fn load_waves(&mut self) {
        let (Some(kind), Some(zone)) = (self.listen.kind, self.zone_shown().cloned()) else {
            return;
        };
        self.listen.asked += 1;
        let asked = self.listen.asked;
        let keys: Vec<String> = self
            .rows
            .iter()
            .filter(|row| kind.present(row))
            .map(|row| row.key.clone())
            .collect();
        let (dir, tx, ctx) = (self.dir.clone(), self.listen.tx.clone(), self.ctx.clone());
        std::thread::spawn(move || {
            let original = audio::read(&layout::extract(&dir, &zone, kind.side()))
                .ok()
                .map(|samples| wave(&samples, None));
            let reference = match kind {
                Kind::Stem(stem) => audio::read(&layout::reference(&dir, &zone, stem)).ok(),
                Kind::Mix => None,
            };
            let waves = keys
                .par_iter()
                .filter_map(|key| {
                    let samples = audio::read(&kind.path(&dir, &zone, key)).ok()?;
                    Some((key.clone(), wave(&samples, reference.as_deref())))
                })
                .collect();
            let _ = tx.send((asked, original, waves));
            ctx.request_repaint();
        });
    }

    /// Take in the waveforms read since the last frame.
    pub fn drain_listen(&mut self) {
        while let Ok((asked, original, waves)) = self.listen.rx.try_recv() {
            if asked == self.listen.asked {
                self.listen.original = original;
                self.listen.waves = waves.into_iter().collect();
            }
        }
    }

    /// A click on a ★ or in the menu of a ✗: recorded at once. When the
    /// references change, the consensus is written again and the zone is to
    /// be measured again.
    pub fn rate(&mut self, key: &str, side: Side, status: Status) {
        let Some(id) = self.zone_shown().map(|z| z.id.clone()) else {
            return;
        };
        if self.tab == crate::app::Tab::Film {
            return;
        }
        match rate::change(&self.dir, &mut self.set, &id, key, side, status) {
            Ok(changed) => {
                self.status = if changed {
                    format!(
                        "{} references of this zone changed: measure it again (Report, Metrics only).",
                        side.label()
                    )
                } else {
                    String::new()
                };
                self.refresh_rows();
                if changed && self.listen.kind.is_some_and(|k| k != Kind::Mix) {
                    self.load_waves();
                }
            }
            Err(e) => self.status = e.to_string(),
        }
    }
}
