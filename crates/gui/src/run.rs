//! A task the window follows: its steps, where the current one is, what
//! it said, and how it ended.

use crate::child::{Heard, Running};
use crate::task::{Reply, Request};
use egui::{Color32, RichText, Ui};
use omnifrog_core::job::Step;
use omnifrog_core::progress::{self, Event, Unit};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Lines of log kept; a film's run says a few thousand.
const LOG_LINES: usize = 5000;

/// How much of a run each step is, in percent: the two separations are
/// most of it (measured on one film: extraction 3 min, the original
/// 3 h 37, the dub 41 min, the encoding 8 min).
const WEIGHTS: [f64; 6] = [2.0, 70.0, 16.0, 3.0, 7.0, 2.0];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Setup,
    Fetch,
    Dub,
}

pub enum Ending {
    Done(Option<PathBuf>),
    Failed(String),
    Stopped,
}

/// The fraction the current step last reported.
struct Fraction {
    what: String,
    value: Option<f64>,
    text: String,
    /// When this `what` first reported, and at what value: the rate an
    /// estimate of the time left is drawn from.
    since: Instant,
    from: f64,
}

/// How long a step took, as the worker measured it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct StepTime {
    took: Duration,
    /// Done by an earlier run of the job: the time is the one it recorded.
    earlier: bool,
}

pub struct Run {
    pub kind: Kind,
    pub title: String,
    worker: Option<Running>,
    stopping: bool,
    started: Instant,
    pub ending: Option<(Ending, Duration)>,
    /// For a run: the step under way (index into [`Step::ALL`]) and the
    /// first this run did — the earlier ones were done by a previous run.
    step: Option<usize>,
    first_step: Option<usize>,
    step_started: Instant,
    /// What each step took, by index into [`Step::ALL`], once it is done.
    took: Vec<Option<StepTime>>,
    /// The most of the current step any fraction has shown, so the overall
    /// bar never goes back when a step moves to its next sub-task.
    step_best: f64,
    /// The most of the whole run the overall bar has shown: a step started
    /// early counts towards it, and the step before it is not quite whole
    /// when it ends, so a step change would otherwise take a little back.
    overall_best: f64,
    fraction: Option<Fraction>,
    /// Later steps already under way beside the current one, by index into
    /// [`Step::ALL`]: the original's separation during the extraction. A
    /// fraction named after a step (see [`Step::labelled`]) is that step's,
    /// whichever step it is reported in.
    ahead: Vec<(usize, Fraction)>,
    pub log: VecDeque<String>,
}

impl Run {
    pub fn start(
        kind: Kind,
        title: impl Into<String>,
        request: &Request,
        ctx: &egui::Context,
    ) -> Self {
        let mut run = Self::new(kind, title);
        let ctx = ctx.clone();
        match Running::start(request, move || ctx.request_repaint()) {
            Ok(worker) => run.worker = Some(worker),
            Err(e) => run.finish(
                Ending::Failed(format!("could not start the worker: {e}")),
                Instant::now(),
            ),
        }
        run
    }

    fn new(kind: Kind, title: impl Into<String>) -> Self {
        Self {
            kind,
            title: title.into(),
            worker: None,
            stopping: false,
            started: Instant::now(),
            ending: None,
            step: None,
            first_step: None,
            step_started: Instant::now(),
            took: vec![None; Step::ALL.len()],
            step_best: 0.0,
            overall_best: 0.0,
            fraction: None,
            ahead: Vec::new(),
            log: VecDeque::new(),
        }
    }

    pub fn active(&self) -> bool {
        self.ending.is_none()
    }

    pub fn stop(&mut self) {
        if let Some(worker) = &mut self.worker {
            self.stopping = true;
            worker.stop();
            self.push_log("stopping…".into());
        }
    }

    /// Take in what the worker said; returns true when the task just ended.
    pub fn pump(&mut self) -> bool {
        let Some(worker) = &self.worker else {
            return false;
        };
        let replies = worker.poll();
        let mut ended = false;
        for Heard { at, reply } in replies {
            match reply {
                Reply::Progress { event } => self.heard(event, at),
                Reply::Done { output } => {
                    self.finish(Ending::Done(output), at);
                    ended = true;
                }
                Reply::Failed { error } => {
                    let ending = if self.stopping {
                        Ending::Stopped
                    } else {
                        self.push_log(format!("error: {error}"));
                        Ending::Failed(error)
                    };
                    self.finish(ending, at);
                    ended = true;
                }
            }
            if ended {
                break;
            }
        }
        ended
    }

    /// The task ended `at`, which is when the worker said so: not when the
    /// window read it.
    fn finish(&mut self, ending: Ending, at: Instant) {
        self.ending = Some((ending, at.saturating_duration_since(self.started)));
        self.worker = None;
        self.fraction = None;
        self.ahead.clear();
    }

    fn push_log(&mut self, line: String) {
        if self.log.len() == LOG_LINES {
            self.log.pop_front();
        }
        self.log.push_back(line);
    }

    #[cfg(test)]
    fn event(&mut self, event: Event) {
        self.heard(event, Instant::now());
    }

    /// Take in `event`, which the worker said `at`.
    fn heard(&mut self, event: Event, at: Instant) {
        self.take(event, at);
        if let Some(overall) = self.overall_now() {
            self.overall_best = self.overall_best.max(overall);
        }
    }

    fn take(&mut self, event: Event, at: Instant) {
        let took_line = event.took_line();
        match event {
            Event::Step {
                number,
                label,
                total,
            } => {
                let index = (number as usize).saturating_sub(1).min(Step::ALL.len() - 1);
                self.first_step.get_or_insert(index);
                self.step = Some(index);
                self.step_started = at;
                self.step_best = 0.0;
                self.fraction = None;
                // A step that started early carries on as the step's own
                // bar, where it was: its rate so far was the step before's,
                // so the time left is measured afresh.
                if let Some(at_index) = self.ahead.iter().position(|(of, _)| *of == index) {
                    let (_, mut early) = self.ahead.remove(at_index);
                    early.since = at;
                    early.from = early.value.unwrap_or(0.0);
                    self.step_best = early.from;
                    self.fraction = Some(early);
                }
                self.ahead.retain(|(of, _)| *of > index);
                self.push_log(format!("── {number}/{total} {label}"));
            }
            Event::Message { text } => self.push_log(text),
            Event::Fraction {
                what,
                done,
                total,
                unit,
            } => {
                let value = total.filter(|t| *t > 0).map(|t| done as f64 / t as f64);
                let text = match (total, unit) {
                    (Some(t), Unit::Bytes) if t > 0 => {
                        format!("{what} · {} / {} MB", done >> 20, t >> 20)
                    }
                    (Some(t), Unit::Items) if t > 0 => format!("{what} · {done} / {t}"),
                    (_, Unit::Bytes) => format!("{what} · {} MB", done >> 20),
                    _ => what.clone(),
                };
                let early = Step::labelled(&what)
                    .map(|step| step.number() as usize - 1)
                    .filter(|of| self.step.is_some_and(|now| *of > now));
                if let Some(of) = early {
                    match self.ahead.iter_mut().find(|(on, _)| *on == of) {
                        Some((_, f)) => {
                            f.value = value;
                            f.text = text;
                        }
                        None => self.ahead.push((
                            of,
                            Fraction {
                                what,
                                value,
                                text,
                                since: at,
                                from: value.unwrap_or(0.0),
                            },
                        )),
                    }
                    return;
                }
                if let Some(v) = value {
                    self.step_best = self.step_best.max(v);
                }
                match &mut self.fraction {
                    Some(f) if f.what == what => {
                        f.value = value;
                        f.text = text;
                    }
                    _ => {
                        self.fraction = Some(Fraction {
                            what,
                            value,
                            text,
                            since: at,
                            from: value.unwrap_or(0.0),
                        })
                    }
                }
            }
            Event::Took {
                number,
                seconds,
                earlier,
                ..
            } => {
                if let Some(line) = took_line {
                    self.push_log(line);
                }
                let index = (number as usize).saturating_sub(1);
                if let Some(slot) = self.took.get_mut(index) {
                    *slot = Some(StepTime {
                        took: Duration::from_secs_f64(seconds.max(0.0)),
                        earlier,
                    });
                }
            }
        }
    }

    /// The whole run, from 0 to 1, by the steps' weights, as far as it has
    /// ever got.
    fn overall(&self) -> Option<f64> {
        self.step.map(|_| self.overall_best)
    }

    /// The whole run now: the steps done, the current one, and what the
    /// steps started early have made.
    fn overall_now(&self) -> Option<f64> {
        let step = self.step?;
        let total: f64 = WEIGHTS.iter().sum();
        let before: f64 = WEIGHTS[..step].iter().sum();
        let early: f64 = self
            .ahead
            .iter()
            .map(|(of, f)| WEIGHTS[*of] * f.value.unwrap_or(0.0))
            .sum();
        Some(((before + WEIGHTS[step] * self.step_best + early) / total).clamp(0.0, 1.0))
    }

    /// How far a later step started early is, from 0 to 1.
    fn early(&self, i: usize) -> Option<f64> {
        self.ahead
            .iter()
            .find(|(of, _)| *of == i)
            .map(|(_, f)| f.value.unwrap_or(0.0))
    }

    pub fn ui(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.strong(&self.title);
            let elapsed = match &self.ending {
                Some((_, took)) => *took,
                None => self.started.elapsed(),
            };
            ui.weak(clock(elapsed));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if self.active() {
                    let label = if self.stopping { "Stopping…" } else { "Stop" };
                    if ui
                        .add_enabled(!self.stopping, egui::Button::new(label))
                        .on_hover_text("Stop the worker and everything it started; a run picks up where it was next time")
                        .clicked()
                    {
                        self.stop();
                    }
                }
            });
        });

        if self.kind == Kind::Dub {
            self.steps_ui(ui);
        }

        if let Some(f) = &self.fraction {
            let mut bar =
                egui::ProgressBar::new(f.value.unwrap_or(0.0) as f32).text(match f.value {
                    Some(v) => {
                        let mut t = format!("{} · {:.0} %", f.text, v * 100.0);
                        if let Some(left) = time_left(f, v) {
                            t.push_str(&format!(" · about {} left", clock(left)));
                        }
                        t
                    }
                    None => f.text.clone(),
                });
            if f.value.is_none() {
                bar = bar.animate(true);
            }
            ui.add(bar);
        } else if self.active() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.weak(self.log.back().map(String::as_str).unwrap_or("starting…"));
            });
        }
        // Each step started early, as a bar of its own under the step's:
        // the work goes on side by side, and so do the bars.
        for (of, f) in &self.ahead {
            let now = self.step.map_or(1, |now| now + 1);
            let text = match f.value {
                Some(v) => format!("{}. {} · {:.0} %", of + 1, f.text, v * 100.0),
                None => format!("{}. {}", of + 1, f.text),
            };
            let mut bar = egui::ProgressBar::new(f.value.unwrap_or(0.0) as f32).text(text);
            if f.value.is_none() {
                bar = bar.animate(true);
            }
            ui.add(bar).on_hover_text(format!(
                "step {} started while step {now} runs: it keeps pace with it, and carries on \
                 as its own step from where it got to",
                of + 1
            ));
        }
        if let (Some(overall), true) = (self.overall(), self.active()) {
            ui.add(
                egui::ProgressBar::new(overall as f32)
                    .desired_height(6.0)
                    .fill(ui.visuals().selection.bg_fill.gamma_multiply(0.6)),
            )
            .on_hover_text(format!(
                "the whole run, about {:.0} % (the separations weigh the most)",
                overall * 100.0
            ));
        }

        match &self.ending {
            Some((Ending::Done(output), _)) => {
                ui.horizontal(|ui| {
                    ui.colored_label(Color32::from_rgb(90, 190, 110), "✔ Done");
                    if let Some(output) = output {
                        ui.label(output.display().to_string());
                        if ui.small_button("Show in folder").clicked() {
                            reveal(output);
                        }
                    }
                });
            }
            Some((Ending::Failed(error), _)) => {
                ui.colored_label(Color32::from_rgb(230, 90, 80), format!("✖ {error}"));
            }
            Some((Ending::Stopped, _)) => {
                ui.colored_label(
                    Color32::from_rgb(220, 170, 60),
                    "Stopped. Starting the same run again picks up at the step it was in.",
                );
            }
            None => {}
        }

        egui::CollapsingHeader::new("Log")
            .id_salt("run-log")
            .default_open(false)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .max_height(ui.available_height().max(120.0))
                    .stick_to_bottom(true)
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        for line in &self.log {
                            ui.label(RichText::new(line).monospace().size(11.5));
                        }
                    });
            });
    }

    fn steps_ui(&self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            for (i, step) in Step::ALL.iter().enumerate() {
                let (mark, color) = match (self.step, self.first_step) {
                    _ if matches!(self.ending, Some((Ending::Done(_), _))) => {
                        ("✔", Color32::from_rgb(90, 190, 110))
                    }
                    (_, Some(first)) if i < first => ("✔", ui.visuals().weak_text_color()),
                    (Some(now), _) if i < now => ("✔", Color32::from_rgb(90, 190, 110)),
                    (Some(now), _) if i == now && self.active() => {
                        ("▶", ui.visuals().selection.stroke.color)
                    }
                    (Some(now), _) if i == now => ("✖", Color32::from_rgb(230, 90, 80)),
                    _ if self.active() && self.early(i).is_some() => {
                        ("▶", ui.visuals().selection.stroke.color)
                    }
                    _ => ("○", ui.visuals().weak_text_color()),
                };
                let mut text = format!("{mark} {}. {}", i + 1, step.label());
                if let Some(took) = self.time_of(i) {
                    text.push_str(&format!(" · {}", clock(took)));
                } else if let (Some(early), true) = (self.early(i), self.active()) {
                    text.push_str(&format!(" · {:.0} %", early * 100.0));
                }
                let label = if Some(i) == self.step && self.active() {
                    RichText::new(text).color(color).strong()
                } else {
                    RichText::new(text).color(color)
                };
                let response = ui.label(label);
                if Some(i) == self.step && self.active() {
                    response.on_hover_text(format!(
                        "in this step for {}",
                        clock(self.step_started.elapsed())
                    ));
                } else if let (Some(early), true) = (self.early(i), self.active()) {
                    response.on_hover_text(format!(
                        "started early, beside step {}: {:.0} % made",
                        self.step.map_or(1, |now| now + 1),
                        early * 100.0
                    ));
                } else if self.first_step.is_some_and(|first| i < first) {
                    response.on_hover_text(match self.took[i] {
                        Some(time) => {
                            format!(
                                "done by an earlier run of this job, in {}",
                                clock(time.took)
                            )
                        }
                        None => "done by an earlier run of this job".to_string(),
                    });
                }
                if i + 1 < Step::ALL.len() {
                    ui.weak("›");
                }
            }
        });
    }
}

impl Run {
    /// How long step `i` took: what the worker said once it was done, or,
    /// for the step a run ended in, from its start to the end.
    fn time_of(&self, i: usize) -> Option<Duration> {
        if let Some(time) = self.took.get(i).copied().flatten() {
            return Some(time.took);
        }
        match &self.ending {
            Some((_, took)) if self.step == Some(i) => {
                Some((self.started + *took).saturating_duration_since(self.step_started))
            }
            _ => None,
        }
    }
}

/// The time left at the rate this fraction has gone since it started.
fn time_left(f: &Fraction, now: f64) -> Option<Duration> {
    let elapsed = f.since.elapsed().as_secs_f64();
    let gone = now - f.from;
    if elapsed < 5.0 || gone < 0.01 {
        return None;
    }
    Some(Duration::from_secs_f64(elapsed * (1.0 - now) / gone))
}

/// « 1:02:03 », « 4:05 ».
pub fn clock(d: Duration) -> String {
    progress::clock(d.as_secs())
}

/// Show a file in the platform's file manager.
pub fn reveal(path: &std::path::Path) {
    let result = if cfg!(windows) {
        std::process::Command::new("explorer")
            .arg(format!("/select,{}", path.display()))
            .spawn()
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open")
            .arg("-R")
            .arg(path)
            .spawn()
    } else {
        let dir = if path.is_dir() {
            path
        } else {
            path.parent().unwrap_or(path)
        };
        std::process::Command::new("xdg-open").arg(dir).spawn()
    };
    let _ = result;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(number: u32) -> Event {
        Event::Step {
            number,
            total: 6,
            label: Step::ALL[number as usize - 1].label().into(),
        }
    }

    #[test]
    fn a_resumed_run_counts_the_steps_before_it_as_done() {
        let mut run = Run::new(Kind::Dub, "film");
        run.event(Event::message("job … starting at step 3/6"));
        assert_eq!(run.overall(), None);
        run.event(step(3));
        assert_eq!((run.first_step, run.step), (Some(2), Some(2)));
        let before = (2.0 + 70.0) / 100.0;
        assert!((run.overall().unwrap() - before).abs() < 1e-9);
        run.event(Event::ratio("dialogue out of the dub", 0.5));
        assert!((run.overall().unwrap() - (before + 0.08)).abs() < 1e-9);
    }

    #[test]
    fn the_overall_bar_never_goes_back_within_a_step() {
        let mut run = Run::new(Kind::Dub, "film");
        run.event(step(4));
        run.event(Event::ratio("duplicates: map", 1.0));
        let full = run.overall().unwrap();
        // The step's next sub-task starts from zero; the bar stays.
        run.event(Event::ratio("duplicates: cancelling", 0.1));
        assert_eq!(run.overall().unwrap(), full);
        assert_eq!(
            run.fraction.as_ref().unwrap().what,
            "duplicates: cancelling"
        );
        run.event(step(5));
        assert!(run.fraction.is_none());
    }

    /// **Work side by side, bars side by side.** The original's separation
    /// starts during the extraction and says how far it is under its own
    /// step's name: it gets a bar of its own beside the decoding's, the
    /// steps row shows it under way, and the whole run counts it. When the
    /// extraction ends, its bar becomes the step's, where it was.
    #[test]
    fn a_step_started_early_has_a_bar_of_its_own() {
        let mut run = Run::new(Kind::Dub, "film");
        let original = Step::SeparateOriginal.label();
        run.event(step(1));
        run.event(Event::bytes("decoding", 50, Some(100)));
        run.event(Event::ratio(original, 0.2));
        assert_eq!(run.fraction.as_ref().unwrap().what, "decoding");
        assert_eq!(run.early(1), Some(0.2));
        assert!((run.overall().unwrap() - (2.0 * 0.5 + 70.0 * 0.2) / 100.0).abs() < 1e-9);
        run.event(Event::ratio(original, 0.3));
        run.event(Event::bytes("decoding", 90, Some(100)));
        assert_eq!(run.early(1), Some(0.3));
        assert_eq!(run.fraction.as_ref().unwrap().what, "decoding");
        let before = run.overall().unwrap();

        // The extraction ends a little short of its estimate: the whole
        // run's bar does not go back, and the separation's carries on where
        // it was.
        run.event(step(2));
        assert_eq!(run.early(1), None);
        let own = run.fraction.as_ref().unwrap();
        assert_eq!((own.what.as_str(), own.value), (original, Some(0.3)));
        let carried = run.overall().unwrap();
        assert!(carried >= before, "{carried} < {before}");
        run.event(Event::ratio(original, 0.35));
        assert!(run.overall().unwrap() > carried);
        // A fraction named after the step under way, or one before it, is
        // the step's own.
        run.event(Event::ratio(Step::Extract.label(), 1.0));
        assert_eq!(run.fraction.as_ref().unwrap().what, Step::Extract.label());
        assert!(run.ahead.is_empty());
    }

    fn took(number: u32, seconds: f64, earlier: bool) -> Event {
        Event::Took {
            number,
            total: 6,
            label: Step::ALL[number as usize - 1].label().into(),
            seconds,
            earlier,
        }
    }

    /// A run's times are when its replies came in, not when the window read
    /// them: a window that is not on screen reads a run's end hours late,
    /// and the run still took what it took. Here it ended at 38:24, and
    /// the window reads it now, before that.
    #[test]
    fn a_run_s_times_are_when_its_replies_came_in() {
        let mut run = Run::new(Kind::Dub, "film");
        let start = run.started;
        let at = |seconds: u64| start + Duration::from_secs(seconds);
        run.heard(took(1, 893.4, true), at(0));
        run.heard(step(4), at(2094));
        run.heard(took(4, 38.7, false), at(2133));
        run.heard(step(5), at(2133));
        run.finish(Ending::Failed("refused".into()), at(2304));

        assert_eq!(run.ending.as_ref().unwrap().1, Duration::from_secs(2304));
        assert_eq!(clock(run.time_of(0).unwrap()), "14:53");
        assert_eq!(run.time_of(1), None, "no time was ever said for it");
        assert_eq!(clock(run.time_of(3).unwrap()), "0:38");
        // The step it ended in: from its start to the end.
        assert_eq!(run.time_of(4), Some(Duration::from_secs(171)));
        assert_eq!(run.time_of(5), None);
        assert!(
            run.log.iter().any(|line| line == "4/6 master took 0:38"),
            "{:?}",
            run.log
        );
    }

    #[test]
    fn the_panel_draws_in_every_state() {
        let ctx = egui::Context::default();
        let mut runs = vec![Run::new(Kind::Dub, "film"), Run::new(Kind::Setup, "setup")];
        runs[0].event(took(1, 893.4, true));
        runs[0].event(step(2));
        runs[0].event(Event::bytes("decoding", 3 << 20, None));
        let mut early = Run::new(Kind::Dub, "film");
        early.event(step(1));
        early.event(Event::bytes("decoding", 3 << 20, Some(8 << 20)));
        early.event(Event::ratio(Step::SeparateOriginal.label(), 0.2));
        runs.push(early);
        runs[1].event(Event::bytes("gfu3", 10 << 20, Some(100 << 20)));
        let mut done = Run::new(Kind::Dub, "film");
        done.finish(Ending::Done(Some("/x/film.mka".into())), Instant::now());
        runs.push(done);
        let mut failed = Run::new(Kind::Fetch, "fetch");
        failed.finish(Ending::Failed("no network".into()), Instant::now());
        runs.push(failed);
        for run in &mut runs {
            let mut output = ctx.run_ui(Default::default(), |ui| run.ui(ui));
            // No renderer takes the font atlas here.
            output.textures_delta.clear();
        }
    }
}
