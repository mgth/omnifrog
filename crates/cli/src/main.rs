//! The omnifrog command line: one thin layer over `omnifrog-core` that
//! turns arguments into calls and progress events into lines on stderr.

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use omnifrog_core::config::{Config, Encoder, IamfCodec, Pairs};
use omnifrog_core::doctor::{self, Status};
use omnifrog_core::models::{self, Catalogue};
use omnifrog_core::progress::{Event, Unit};
use omnifrog_core::setup::{self, Gpu};
use omnifrog_core::tools::{self, Tool};
use omnifrog_core::{bench, pipeline, probe, tracks};
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Parser)]
#[command(
    name = "omnifrog",
    version,
    about = "Re-voices a spatial-audio film with the dialogue of its dubbed track"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// What is installed, what is missing, and whether a run could start
    Doctor,
    /// Build the Python environment (torch for this GPU + MSST) the inference runs in
    Setup {
        /// Which torch to install: auto detects ROCm (Linux, /dev/kfd), Metal (Apple silicon), then CUDA (nvidia-smi), else CPU
        #[arg(long, value_enum, default_value_t = GpuArg::Auto)]
        gpu: GpuArg,
        /// The base interpreter (3.11 or 3.12) to build the venv from; found on PATH otherwise
        #[arg(long)]
        python: Option<PathBuf>,
        /// Rebuild the venv and the MSST checkout even if they look complete
        #[arg(long)]
        force: bool,
    },
    /// The separation models: list, choose, download
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
    /// The effective configuration and where it lives
    Config,
    /// List the audio tracks of a file and say which would be taken
    Probe {
        file: PathBuf,
        /// Index of the original's track among the audio tracks (1-based)
        #[arg(long)]
        vo: Option<u32>,
        /// Index of the dubbed track (1-based)
        #[arg(long)]
        vf: Option<u32>,
    },
    /// Re-voice a film
    Dub {
        file: PathBuf,
        /// Index of the original's track among the audio tracks (1-based); default: the first spatial one
        #[arg(long)]
        vo: Option<u32>,
        /// Index of the dubbed track (1-based); default: the best French one
        #[arg(long)]
        vf: Option<u32>,
        /// Write a full .mkv (video + the new track + the source's tracks, subtitles and chapters) instead of an .mka
        #[arg(long)]
        mkv: bool,
        /// Which encoder (default: the configuration's): hz and dee write TrueHD Atmos; iamf writes IAMF objects (harlettizer, into an .mka) with the voices panned onto them as hz does, iamf-lit with the voices on the bed
        #[arg(long, value_enum)]
        encoder: Option<EncoderArg>,
        /// Read only this many seconds of the film, for a trial run
        #[arg(long)]
        seconds: Option<f64>,
        /// Read from this point of the film rather than its beginning, for a trial of a scene: seconds or [h:]mm:ss; the run starts at the original's first keyframe at or after it
        #[arg(long, value_parser = parse_time)]
        start: Option<f64>,
        /// Offset of the dub's voices against the original's dialogue, in dB
        #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
        gain_db: f64,
        /// Options passed on to `harlettizer encode`, replacing the configuration's: --hz-opts "--presentations --overlay-cost 0"
        #[arg(long, allow_hyphen_values = true)]
        hz_opts: Option<String>,
        /// Options passed on to `harlettizer iamf` with --encoder iamf or iamf-lit, replacing the configuration's: --iamf-opts "--codec opus --bitrate 48"; empty = harlettizer's defaults (FLAC, lossless)
        #[arg(long, allow_hyphen_values = true)]
        iamf_opts: Option<String>,
        /// The IAMF codec, replacing the configuration's: flac, lossless, or opus, lossy (a harlettizer built with its opus feature)
        #[arg(long, value_enum)]
        iamf_codec: Option<IamfCodecArg>,
        /// Opus only: kilobits a second for each channel, replacing the configuration's (64 puts a 7.1.4 bed at about 720 kbit/s)
        #[arg(long, value_parser = clap::value_parser!(u32).range(6..=256))]
        iamf_bitrate: Option<u32>,
        /// FLAC only: bits a sample, replacing the configuration's: source (16 for a master at 16 bits or fewer, 24 otherwise), 16 or 24
        #[arg(long, value_parser = parse_bits)]
        iamf_bits: Option<Bits>,
        /// Hand the models two elements at once, replacing the configuration's: a list of original (the original's elements, neighbours from the centre), dub (the dub's channels, left with right), centre (and the dub's centre twice, beside L and R), all, none
        #[arg(long, value_parser = parse_pairs)]
        pairs: Option<Pairs>,
        /// Leave the duplicates alone: the dub's voices as separated, on the dub's clock (default: the configuration's cancel_duplicates, on unless set to false)
        #[arg(long, conflicts_with = "cancel_duplicates")]
        no_cancel_duplicates: bool,
        /// Cancel the duplicates between the original and the dub, whatever the configuration says
        #[arg(long)]
        cancel_duplicates: bool,
        /// Temporaries here instead of the configuration's work directory
        #[arg(long)]
        work: Option<PathBuf>,
        /// Write the track here; default: beside the source
        #[arg(long)]
        out: Option<PathBuf>,
        /// Start over, whatever the job's manifest says is done
        #[arg(long)]
        restart: bool,
        /// One JSON object per line on stderr instead of progress lines
        #[arg(long)]
        json: bool,
    },
    /// Measure the separation's device memory under several settings, to choose inference.block_seconds and inference.batch_size for a smaller card
    BenchVram {
        /// A mono element, as a job's extraction writes them (1.extract/vo_<id>.wav)
        file: PathBuf,
        /// Models to try, by key; default: the configuration's two
        #[arg(long, value_delimiter = ',')]
        models: Vec<String>,
        /// Seconds of the element to separate; longer than a block, so blocks are joined
        #[arg(long, default_value_t = 1500.0)]
        seconds: f64,
        /// pymss blocks to try, in seconds, besides the default
        #[arg(long, value_delimiter = ',', default_value = "600,120,60")]
        blocks: Vec<u32>,
        /// Batch sizes to try, at the default block
        #[arg(long, value_delimiter = ',', default_value = "1,2,4")]
        batches: Vec<u32>,
        /// Work here instead of <work_dir>/bench-vram
        #[arg(long)]
        work: Option<PathBuf>,
        /// Keep every setting's stems
        #[arg(long)]
        keep: bool,
    },
}

#[derive(Subcommand)]
enum ModelsCommand {
    /// The catalogue, with what is on disk
    List,
    /// Choose the two models a run uses
    Use {
        /// Key of the model that removes the dialogue from the original's elements
        #[arg(long)]
        vo: Option<String>,
        /// Key of the model that extracts the dialogue from the dubbed track
        #[arg(long)]
        vf: Option<String>,
    },
    /// Download what is missing (default: the two selected models)
    Fetch { keys: Vec<String> },
    /// Add or override catalogue entries from a YAML of the same shape
    Add { file: PathBuf },
}

#[derive(Clone, Copy, ValueEnum)]
enum GpuArg {
    Auto,
    Cuda,
    Rocm,
    Mps,
    Cpu,
}

impl From<GpuArg> for Gpu {
    fn from(g: GpuArg) -> Self {
        match g {
            GpuArg::Auto => Gpu::Auto,
            GpuArg::Cuda => Gpu::Cuda,
            GpuArg::Rocm => Gpu::Rocm,
            GpuArg::Mps => Gpu::Mps,
            GpuArg::Cpu => Gpu::Cpu,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum EncoderArg {
    Hz,
    Dee,
    Iamf,
    #[value(name = "iamf-lit", alias = "iamf-bed")]
    IamfBed,
}

/// A FLAC depth asked for: `None` is the original's.
#[derive(Clone, Copy)]
struct Bits(Option<u32>);

fn parse_bits(s: &str) -> Result<Bits, String> {
    match s {
        "source" | "auto" => Ok(Bits(None)),
        "16" => Ok(Bits(Some(16))),
        "24" => Ok(Bits(Some(24))),
        _ => Err("source, 16 or 24".into()),
    }
}

/// Seconds, or `[h:]mm:ss` with a fraction of a second if wanted.
fn parse_time(s: &str) -> Result<f64, String> {
    let mut seconds = 0.0;
    for part in s.split(':') {
        let value: f64 = part
            .trim()
            .parse()
            .map_err(|_| format!("`{s}`: seconds, or [h:]mm:ss"))?;
        seconds = seconds * 60.0 + value;
    }
    if seconds < 0.0 || !seconds.is_finite() || s.split(':').count() > 3 {
        return Err(format!("`{s}`: seconds, or [h:]mm:ss"));
    }
    Ok(seconds)
}

#[derive(Clone, Copy, ValueEnum)]
enum IamfCodecArg {
    Flac,
    Opus,
}

impl From<IamfCodecArg> for IamfCodec {
    fn from(c: IamfCodecArg) -> Self {
        match c {
            IamfCodecArg::Flac => IamfCodec::Flac,
            IamfCodecArg::Opus => IamfCodec::Opus,
        }
    }
}

impl From<EncoderArg> for Encoder {
    fn from(e: EncoderArg) -> Self {
        match e {
            EncoderArg::Hz => Encoder::Hz,
            EncoderArg::Dee => Encoder::Dee,
            EncoderArg::Iamf => Encoder::Iamf,
            EncoderArg::IamfBed => Encoder::IamfBed,
        }
    }
}

/// Progress on stderr: messages as lines, fractions as one line rewritten
/// in place on a terminal (and at every percent otherwise) — or, with
/// `--json`, one object per line, which is what a front end reads.
struct Reporter {
    tty: bool,
    json: bool,
    state: Mutex<(String, u64)>,
}

impl Reporter {
    fn new(json: bool) -> Self {
        Self {
            tty: std::io::stderr().is_terminal() && !json,
            json,
            state: Mutex::new((String::new(), u64::MAX)),
        }
    }

    fn report(&self, event: Event) {
        let mut err = std::io::stderr().lock();
        if self.json {
            if let Ok(line) = serde_json::to_string(&event) {
                let _ = writeln!(err, "{line}");
            }
            return;
        }
        let mut state = self.state.lock().unwrap();
        let event = match event {
            Event::Step {
                number,
                total,
                label,
            } => Event::message(format!("{number}/{total} {label}")),
            other => other,
        };
        match event {
            Event::Step { .. } => unreachable!("worded as a message above"),
            Event::Message { text } => {
                if self.tty && !state.0.is_empty() {
                    let _ = writeln!(err);
                    state.0.clear();
                }
                let _ = writeln!(err, "{text}");
            }
            Event::Fraction {
                what,
                done,
                total,
                unit,
            } => {
                // A step that reports every block would otherwise write
                // thousands of identical lines into a log.
                let step = match total {
                    Some(t) if t > 0 => done * 100 / t,
                    _ => done >> 24,
                };
                let same = state.0 == what;
                if same && state.1 == step && !self.tty {
                    return;
                }
                if !same && self.tty && !state.0.is_empty() {
                    let _ = writeln!(err);
                }
                *state = (what.clone(), step);
                let text = match (total, unit) {
                    (Some(t), Unit::Bytes) if t > 0 => format!(
                        "  {what}: {} % ({} / {} MB)",
                        done * 100 / t,
                        done >> 20,
                        t >> 20
                    ),
                    (Some(t), Unit::Items) if t > 0 => {
                        format!("  {what}: {done} / {t} ({} %)", done * 100 / t)
                    }
                    (Some(t), Unit::Ratio) if t > 0 => {
                        format!("  {what}: {} %", done * 100 / t)
                    }
                    (_, Unit::Bytes) => format!("  {what}: {} MB", done >> 20),
                    (_, Unit::Items | Unit::Ratio) => format!("  {what}: {done}"),
                };
                if self.tty {
                    let _ = write!(err, "\r{text}\x1b[K");
                    if total.is_some_and(|t| done >= t) {
                        let _ = writeln!(err);
                        state.0.clear();
                    }
                } else {
                    let _ = writeln!(err, "{text}");
                }
            }
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    // Not for the commands that feed a worker on a pipe: a worker that died
    // (a GPU reset takes it down) would take this program down with it at
    // the next write, before it could say why.
    if !matches!(cli.command, Command::Dub { .. } | Command::BenchVram { .. }) {
        reset_sigpipe();
    }
    let reporter = Reporter::new(matches!(cli.command, Command::Dub { json: true, .. }));
    let progress = |e: Event| reporter.report(e);
    match cli.command {
        Command::Doctor => cmd_doctor(),
        Command::Setup { gpu, python, force } => {
            let mut config = Config::load()?;
            setup::run(
                &mut config,
                &setup::Options {
                    gpu: gpu.into(),
                    python,
                    force,
                },
                &progress,
            )?;
            eprintln!("configuration written to {}", Config::path().display());
            Ok(())
        }
        Command::Models { command } => cmd_models(command, &progress),
        Command::Config => cmd_config(),
        Command::Probe { file, vo, vf } => cmd_probe(file, vo, vf, &progress),
        Command::BenchVram {
            file,
            models,
            seconds,
            blocks,
            batches,
            work,
            keep,
        } => cmd_bench_vram(
            &file, &models, seconds, &blocks, &batches, work, keep, &progress,
        ),
        Command::Dub {
            file,
            vo,
            vf,
            mkv,
            encoder,
            seconds,
            start,
            gain_db,
            hz_opts,
            iamf_opts,
            iamf_codec,
            iamf_bitrate,
            iamf_bits,
            pairs,
            no_cancel_duplicates,
            cancel_duplicates,
            work,
            out,
            restart,
            json: _,
        } => cmd_dub(
            pipeline::Options {
                source: file,
                original: vo,
                dub: vf,
                seconds,
                start,
                mkv,
                encoder: encoder.map(Into::into),
                gain_db,
                hz_opts,
                iamf_opts,
                iamf_codec: iamf_codec.map(Into::into),
                iamf_bitrate,
                iamf_bits: iamf_bits.map(|b| b.0),
                pairs,
                cancel_duplicates: match (cancel_duplicates, no_cancel_duplicates) {
                    (true, _) => Some(true),
                    (_, true) => Some(false),
                    _ => None,
                },
                out_dir: out,
                restart,
            },
            work,
            &progress,
        ),
    }
}

/// What the file holds, and which two tracks a run would take.
fn cmd_probe(
    file: PathBuf,
    vo: Option<u32>,
    vf: Option<u32>,
    progress: &(dyn Fn(Event) + Sync),
) -> Result<()> {
    let config = Config::load()?;
    let media = probe::media(&config, &file, progress)?;
    println!(
        "{}  ({} audio, {} subtitle, {} video tracks, {})",
        media.path.display(),
        media.audio.len(),
        media.subtitles,
        media.video,
        duration(media.duration)
    );
    let options = pipeline::Options {
        source: file,
        original: vo,
        dub: vf,
        ..Default::default()
    };
    let chosen = pipeline::choose(&media, &options);
    let (original, dub) = match &chosen {
        Ok((original, dub)) => (Some(original.index), Some(dub.index)),
        Err(_) => (None, None),
    };
    for track in &media.audio {
        let role = match Some(track.index) {
            i if i == original => "vo",
            i if i == dub => "vf",
            _ => "  ",
        };
        println!("{role} {:>2}  {}", track.index, track.summary());
        if let Some(why) = &track.probe_error {
            println!("        the decoder could not read it: {why}");
        }
    }
    match chosen {
        Ok((original, dub)) => {
            println!(
                "\nwould re-voice track {} ({}) with the dialogue of track {} ({}, {})",
                original.index,
                original
                    .spatial
                    .as_ref()
                    .map(|s| s.label.as_str())
                    .unwrap_or("?"),
                dub.index,
                dub.layout(),
                tracks::iso3(dub.language.as_deref())
            );
            Ok(())
        }
        Err(e) => Err(e.into()),
    }
}

/// Re-voice a film, and say where the run got to.
fn cmd_dub(
    options: pipeline::Options,
    work: Option<PathBuf>,
    progress: &(dyn Fn(Event) + Sync),
) -> Result<()> {
    let mut config = Config::load()?;
    if let Some(work) = work {
        config.work_dir = Some(work);
    }
    let outcome = pipeline::run(&config, &options, progress)?;
    // The one line on standard output: the track, for a script to pick up.
    println!("{}", outcome.output.display());
    Ok(())
}

/// The device-memory bench: a table on standard output.
#[allow(clippy::too_many_arguments)]
fn cmd_bench_vram(
    file: &Path,
    keys: &[String],
    seconds: f64,
    blocks: &[u32],
    batches: &[u32],
    work: Option<PathBuf>,
    keep: bool,
    progress: &(dyn Fn(Event) + Sync),
) -> Result<()> {
    let config = Config::load()?;
    let catalogue = Catalogue::load()?;
    let keys: Vec<String> = if keys.is_empty() {
        vec![config.models.vo.clone(), config.models.vf.clone()]
    } else {
        keys.to_vec()
    };
    let models = keys
        .iter()
        .map(|key| catalogue.get(key))
        .collect::<omnifrog_core::Result<Vec<_>>>()?;
    let settings = bench::settings(&config, blocks, batches);
    let dir = work.unwrap_or_else(|| bench::default_dir(&config));
    let rows = bench::run(
        &config, &models, file, seconds, &settings, &dir, keep, progress,
    )?;
    println!("{}", bench::header());
    for row in &rows {
        println!("{}", bench::line(row));
    }
    println!(
        "peak = what torch reserved; the runtime's context (about 0.5 GB) and the desktop come on top; voice, M&E = dB of difference under the first setting's stems"
    );
    Ok(())
}

/// Seconds as a person reads them.
fn duration(seconds: f64) -> String {
    if seconds <= 0.0 {
        return "duration unknown".to_string();
    }
    let total = seconds.round() as u64;
    format!("{}:{:02}:{:02}", total / 3600, total / 60 % 60, total % 60)
}

fn cmd_doctor() -> Result<()> {
    let config = Config::load()?;
    let checks = doctor::run(&config);
    let width = checks.iter().map(|c| c.name.len()).max().unwrap_or(8);
    let mut missing = 0;
    for check in &checks {
        let mark = match check.status {
            Status::Ok => "ok ",
            Status::Warning => "?  ",
            Status::Missing => "MISSING",
        };
        if check.status == Status::Missing {
            missing += 1;
        }
        println!("{mark:8} {:width$}  {}", check.name, check.detail);
    }
    if missing > 0 {
        bail!("{missing} thing(s) missing");
    }
    Ok(())
}

fn cmd_models(command: ModelsCommand, progress: &(dyn Fn(Event) + Sync)) -> Result<()> {
    let mut config = Config::load()?;
    let catalogue = Catalogue::load()?;
    match command {
        ModelsCommand::List => {
            let dir = config.models_dir();
            println!("models in {}", dir.display());
            println!(
                "{:<6} {:<6} {:<8} {:<18} {:<8} name",
                "key", "code", "role", "type", "on disk"
            );
            for model in catalogue.visible() {
                let local = model.local(&config);
                let present = if local.all_present() {
                    "yes"
                } else if local.checkpoint.is_file() {
                    "partly"
                } else {
                    "no"
                };
                let selected = if config.models.vo == model.key {
                    " ← vo"
                } else if config.models.vf == model.key {
                    " ← vf"
                } else {
                    ""
                };
                println!(
                    "{:<6} {:<6} {:<8} {:<18} {:<8} {}{selected}",
                    model.key,
                    model.code,
                    model.role.label(),
                    model.model_type,
                    present,
                    model.name
                );
            }
            Ok(())
        }
        ModelsCommand::Use { vo, vf } => {
            if vo.is_none() && vf.is_none() {
                bail!("give --vo KEY and/or --vf KEY");
            }
            if let Some(vo) = vo {
                catalogue.get(&vo)?;
                config.models.vo = vo;
            }
            if let Some(vf) = vf {
                catalogue.get(&vf)?;
                config.models.vf = vf;
            }
            config.save()?;
            println!(
                "vo: {}  vf: {}  ({})",
                config.models.vo,
                config.models.vf,
                Config::path().display()
            );
            Ok(())
        }
        ModelsCommand::Fetch { keys } => {
            let keys = if keys.is_empty() {
                vec![config.models.vo.clone(), config.models.vf.clone()]
            } else {
                keys
            };
            for key in keys {
                let model = catalogue.get(&key)?;
                let local = models::fetch(model, &config, progress)
                    .with_context(|| format!("fetching {key}"))?;
                println!("{key}: {}", local.checkpoint.display());
            }
            Ok(())
        }
        ModelsCommand::Add { file } => {
            let keys = Catalogue::add_from(&file)?;
            println!(
                "added or replaced {} in {}",
                keys.join(", "),
                Config::user_catalogue_path().display()
            );
            Ok(())
        }
    }
}

fn cmd_config() -> Result<()> {
    let config = Config::load()?;
    println!(
        "file:        {}{}",
        Config::path().display(),
        if Config::path().is_file() {
            ""
        } else {
            " (absent: defaults)"
        }
    );
    println!(
        "catalogue:   {} (user additions)",
        Config::user_catalogue_path().display()
    );
    println!("work_dir:    {}", config.work_dir().display());
    println!("models_dir:  {}", config.models_dir().display());
    println!("python:      {}", config.python().display());
    println!("msst_dir:    {}", config.msst_dir().display());
    for tool in Tool::here() {
        println!(
            "{:<12} {}",
            format!("{}:", tool.name()),
            tools::find(&config, tool)
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "(not found)".into())
        );
    }
    println!(
        "models:      vo {}  vf {}",
        config.models.vo, config.models.vf
    );
    println!(
        "encoder:     {} ({})",
        config.encoder.code(),
        config.encoder.describe()
    );
    println!("hz_opts:     {}", config.hz_opts);
    println!("iamf_opts:   {}", config.iamf_opts);
    println!(
        "iamf_codec:  {}",
        match config.iamf_codec {
            IamfCodec::Opus => format!("opus, {} kbit/s a channel", config.iamf_bitrate),
            IamfCodec::Flac => match config.iamf_bits {
                Some(bits) => format!("flac, {bits} bits"),
                None => "flac, at the original's depth".to_string(),
            },
        }
    );
    println!("pairs:       {}", config.pairs.describe());
    println!(
        "duplicates:  {}",
        if config.cancel_duplicates {
            "cancelled"
        } else {
            "left alone"
        }
    );
    println!("inference:   {}", config.inference.describe());
    println!("keep_work:   {}", config.keep_work);
    Ok(())
}

/// `--pairs original,dub`: which separations hand the models two elements.
fn parse_pairs(list: &str) -> std::result::Result<Pairs, String> {
    Pairs::parse(list)
}

/// Let a closed pipe end the program quietly, as every other command-line
/// tool does, instead of panicking in `println!`.
fn reset_sigpipe() {
    #[cfg(unix)]
    // SAFETY: restoring the default disposition of SIGPIPE before anything
    // else runs; no handler is installed, nothing else refers to it.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}
