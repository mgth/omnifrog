//! The long work, run by a second copy of this program.
//!
//! Building the Python environment, downloading gigabytes of models and
//! re-voicing a film take minutes to hours and start programs of their own
//! (pip, ffmpeg, harletty, the inference worker, the encoder). The window
//! hands each of them to a copy of itself started with [`WORKER_FLAG`]: one
//! request as a JSON line on its standard input, one [`Reply`] per line on
//! its standard output. Stopping a run is then killing that copy and
//! everything under it, which the core needs no cancellation for — a job
//! picks up where its manifest says — and a crash in the core ends the
//! run, not the window.

use omnifrog_core::config::{Config, Encoder, Inference};
use omnifrog_core::models::{self, Catalogue};
use omnifrog_core::pipeline;
use omnifrog_core::progress::Event;
use omnifrog_core::setup::{self, Gpu};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use std::path::PathBuf;

/// The argument that makes this program a worker rather than a window.
pub const WORKER_FLAG: &str = "--worker";

/// What the window asks a worker for.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "task", rename_all = "snake_case")]
pub enum Request {
    /// The machine made ready: the inference settings written, the Python
    /// environment built for `gpu`, then the models fetched.
    Setup {
        gpu: String,
        force: bool,
        inference: Option<Inference>,
        fetch: Vec<String>,
    },
    /// Download models, by catalogue key.
    Fetch { keys: Vec<String> },
    /// Re-voice a film.
    Dub(Dub),
}

/// A run, as the window words it; the rest is the configuration's.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Dub {
    pub source: PathBuf,
    pub original: Option<u32>,
    pub dub: Option<u32>,
    pub seconds: Option<f64>,
    /// Where in the film a trial starts, in seconds.
    #[serde(default)]
    pub start: Option<f64>,
    pub mkv: bool,
    pub encoder: Option<Encoder>,
    pub gain_db: f64,
    pub cancel_duplicates: Option<bool>,
    pub out_dir: Option<PathBuf>,
    pub restart: bool,
}

/// One line of a worker's standard output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
pub enum Reply {
    Progress {
        event: Event,
    },
    /// The work is done; `output` is the track, for a run.
    Done {
        output: Option<PathBuf>,
    },
    Failed {
        error: String,
    },
}

pub fn gpu_from_label(label: &str) -> Gpu {
    match label {
        "cuda" => Gpu::Cuda,
        "rocm" => Gpu::Rocm,
        "mps" => Gpu::Mps,
        "cpu" => Gpu::Cpu,
        _ => Gpu::Auto,
    }
}

/// The worker's whole life: read the request, do it, say how it ended.
pub fn serve() -> std::process::ExitCode {
    let mut line = String::new();
    let mut stdin = std::io::stdin().lock();
    if stdin.read_line(&mut line).is_err() {
        return std::process::ExitCode::FAILURE;
    }
    drop(stdin);
    watch_parent();
    let reply = match serde_json::from_str::<Request>(&line) {
        Err(e) => Reply::Failed {
            error: format!("the window sent a request this worker does not read: {e}"),
        },
        Ok(request) => match std::panic::catch_unwind(|| perform(&request)) {
            Ok(Ok(output)) => Reply::Done { output },
            Ok(Err(error)) => Reply::Failed { error },
            Err(panic) => Reply::Failed {
                error: format!(
                    "the worker crashed: {}",
                    panic
                        .downcast_ref::<String>()
                        .map(String::as_str)
                        .or_else(|| panic.downcast_ref::<&str>().copied())
                        .unwrap_or("no message")
                ),
            },
        },
    };
    let failed = matches!(reply, Reply::Failed { .. });
    send(&reply);
    if failed {
        std::process::ExitCode::FAILURE
    } else {
        std::process::ExitCode::SUCCESS
    }
}

fn send(reply: &Reply) {
    if let Ok(text) = serde_json::to_string(reply) {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{text}");
        let _ = out.flush();
    }
}

fn perform(request: &Request) -> Result<Option<PathBuf>, String> {
    let progress = |event: Event| send(&Reply::Progress { event });
    let text = |e: omnifrog_core::Error| e.to_string();
    match request {
        Request::Setup {
            gpu,
            force,
            inference,
            fetch,
        } => {
            let mut config = Config::load().map_err(text)?;
            if let Some(inference) = inference {
                config.inference = *inference;
                config.save().map_err(text)?;
                progress(Event::message(format!(
                    "inference: {}",
                    config.inference.describe()
                )));
            }
            setup::run(
                &mut config,
                &setup::Options {
                    gpu: gpu_from_label(gpu),
                    python: None,
                    force: *force,
                },
                &progress,
            )
            .map_err(text)?;
            fetch_models(&config, fetch, &progress)?;
            Ok(None)
        }
        Request::Fetch { keys } => {
            let config = Config::load().map_err(text)?;
            fetch_models(&config, keys, &progress)?;
            Ok(None)
        }
        Request::Dub(dub) => {
            let config = Config::load().map_err(text)?;
            let options = pipeline::Options {
                source: dub.source.clone(),
                original: dub.original,
                dub: dub.dub,
                seconds: dub.seconds,
                start: dub.start,
                mkv: dub.mkv,
                encoder: dub.encoder,
                gain_db: dub.gain_db,
                hz_opts: None,
                iamf_opts: None,
                iamf_codec: None,
                iamf_bitrate: None,
                iamf_bits: None,
                pairs: None,
                cancel_duplicates: dub.cancel_duplicates,
                out_dir: dub.out_dir.clone(),
                restart: dub.restart,
            };
            let outcome = pipeline::run(&config, &options, &progress).map_err(text)?;
            Ok(Some(outcome.output))
        }
    }
}

fn fetch_models(
    config: &Config,
    keys: &[String],
    progress: &(dyn Fn(Event) + Sync),
) -> Result<(), String> {
    if keys.is_empty() {
        return Ok(());
    }
    let catalogue = Catalogue::load().map_err(|e| e.to_string())?;
    for key in keys {
        let model = catalogue.get(key).map_err(|e| e.to_string())?;
        if model.local(config).all_present() {
            progress(Event::message(format!("{key}: already on disk")));
            continue;
        }
        progress(Event::message(format!("fetching {key} — {}", model.name)));
        models::fetch(model, config, progress).map_err(|e| format!("fetching {key}: {e}"))?;
    }
    Ok(())
}

/// When the window goes away without stopping the worker — it crashed, it
/// was killed — its end of the pipe closes; the worker then stops too, with
/// everything it started. (On Windows the job object does it: the window's
/// handle to it was the last one.)
fn watch_parent() {
    std::thread::spawn(|| {
        let mut sink = Vec::new();
        let _ = std::io::Read::read_to_end(&mut std::io::stdin().lock(), &mut sink);
        #[cfg(unix)]
        // SAFETY: plain calls; the group is signalled only when this worker
        // leads it, as the window starts it — never the group of a shell
        // that started a worker by hand.
        unsafe {
            if libc::getpgrp() == libc::getpid() {
                libc::kill(0, libc::SIGTERM);
            }
        }
        std::process::exit(1);
    });
}
