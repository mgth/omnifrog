//! The re-voiced master set, encoded: as TrueHD Atmos by harlettizer or by
//! Dee, or as IAMF by harlettizer.
//!
//! The two TrueHD encoders read the same master set and write the same kind
//! of stream; they differ in what they do with it.
//!
//! **harlettizer**, the open encoder, runs as an *overlay*: the original's
//! elements are kept exactly as they were — their metadata element for
//! element, their samples copied wherever no voice reaches them — and only
//! the voices are panned onto them. The one thing to decide besides the
//! number of voices is how deep to round the few elements a voice was mixed
//! into, and that is measured on the original ([`crate::depth`]). Every
//! other option is harlettizer's own, passed through as the configuration's
//! `hz_opts` says: a new option of the encoder asks nothing of this program.
//! An overlay that falls outside one of its bounds is refused, loudly, and
//! the run fails with the encoder's own words — there is no quiet fallback
//! to a mode nobody chose.
//!
//! **Dee** re-clusters the whole scene into twelve, fourteen or sixteen
//! elements, as a delivered stream carries. It runs under Wine on Linux, it
//! cannot open a path with an accent in it, and under memory pressure it can
//! hit an access violation after which it stops making progress without ever
//! exiting: that line is watched for, and Dee is killed with a message that
//! says why.
//!
//! **IAMF** is the open format a browser or a television decodes, and
//! harlettizer writes it straight into an `.mka` of one track — the remux
//! takes that as it would take a raw stream. It never goes through TrueHD:
//! harlettizer reads the master set and codes IAMF v2.0 objects from it,
//! with no bound of sixteen. Two ways to carry the voices:
//!
//! - **`iamf`**, the overlay: the original's elements kept as they are, and
//!   the voices panned onto them exactly as the TrueHD overlay pans them —
//!   the same objects, the same projection, the same guards, refused the
//!   same way;
//! - **`iamf-lit`**: the voices put on the fixed bed, and the original's
//!   other objects left independent.
//!
//! Every other option is harlettizer's own, passed through as `iamf_opts`
//! says.
//!
//! Dee reads the master set's interleaved CAF, which is built from the mono
//! waveforms just before and removed once the stream is written: it is
//! twenty-odd gigabytes a film, and the waveforms it is made of are still
//! there. A harlettizer that has `--mono-prefix` reads those waveforms where
//! they are, linked under one prefix, and no CAF is written at all; one that
//! has not gets the CAF too.

use crate::config::{Config, Encoder, IamfCodec, iamf_bits_for, names_codec};
use crate::job::Job;
use crate::master::{REVOICED, Revoiced};
use crate::progress::{Event, Progress};
use crate::run::{self, Stream};
use crate::tools::{self, Tool};
use crate::{Error, Result, caf, depth};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// What an encode left.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoded {
    /// `3.master/vf.thd`, or `3.master/vf.mka` for IAMF.
    pub stream: PathBuf,
    /// The command, then everything the encoder said.
    pub log: PathBuf,
    /// The line worth reading without opening the log.
    pub summary: String,
    /// What the mixed elements were rounded to, harlettizer's overlay only.
    pub depth: Option<u32>,
}

/// The most elements a TrueHD stream carries, and so the most an overlay can
/// keep the original's elements as.
pub const MOST_ELEMENTS: usize = 16;

/// Whether harlettizer can overlay voices onto an original of `elements`
/// waveforms, and if not, what to do instead.
///
/// An overlay keeps every element of the original as it is, so the stream
/// has as many as the original — and a TrueHD stream holds sixteen. A DTS:X
/// 7.1.4 with five objects decodes to seventeen, and cannot be overlaid; it
/// is not the kind of thing a quiet fallback to a fold should decide, so
/// the run stops and says what would work.
pub fn overlay_fits(elements: usize) -> Result<()> {
    if elements <= MOST_ELEMENTS {
        return Ok(());
    }
    Err(Error::msg(format!(
        "the original has {elements} elements and a TrueHD stream carries {MOST_ELEMENTS}, so \
         harlettizer cannot keep them all and overlay the voices onto them. Encode with Dee \
         (`--encoder dee`), which re-clusters the whole scene into {MOST_ELEMENTS}, or as IAMF \
         (`--encoder iamf`), which has no such bound; the extraction is kept."
    )))
}

/// The stream's path inside `3.master/`: a raw TrueHD stream, or the `.mka`
/// harlettizer writes IAMF into — `vf-bed.mka` with the voices on the bed,
/// so either IAMF sits beside the other.
pub fn stream(job: &Job, encoder: Encoder) -> PathBuf {
    let name = match encoder {
        Encoder::Hz | Encoder::Dee => format!("{REVOICED}.thd"),
        Encoder::Iamf => format!("{REVOICED}.mka"),
        Encoder::IamfBed => format!("{REVOICED}-bed.mka"),
    };
    job.master_dir().join(name)
}

/// The option of `harlettizer iamf` that says what becomes of the voices:
/// `--overlay` panned onto the master's objects, `--voices-to-bed` put on
/// the bed. `None` for an encoder that is not IAMF.
pub(crate) fn voices_flag(encoder: Encoder) -> Option<&'static str> {
    match encoder {
        Encoder::Iamf => Some("--overlay"),
        Encoder::IamfBed => Some("--voices-to-bed"),
        Encoder::Hz | Encoder::Dee => None,
    }
}

/// Encode with harlettizer, as an overlay of the voices onto the original.
pub fn harlettizer(
    config: &Config,
    job: &Job,
    revoiced: &Revoiced,
    hz_opts: &str,
    progress: Progress,
) -> Result<Encoded> {
    let program = tools::require(config, Tool::Harlettizer)?;
    overlay_fits(revoiced.inputs.len() - revoiced.voices)?;
    let options = split(hz_opts)?;
    let out = stream(job, Encoder::Hz);
    let log = job.master_dir().join(format!("harlettizer_{REVOICED}.log"));

    let depth = fold_depth(job, &options, progress);

    // The waveforms where they are, when this harlettizer reads them that
    // way; otherwise the CAF its header names.
    let audio = if reads_mono(&program) {
        progress(Event::message(
            "harlettizer reads the waveforms where they are (--mono-prefix): no CAF is written",
        ));
        Audio::Mono(link_waveforms(job, revoiced)?)
    } else {
        Audio::Interleaved(interleave(job, revoiced, progress)?)
    };
    let _ = fs::remove_file(&out);
    let mut command = crate::run::command(&program);
    command.arg("encode").arg(&revoiced.header).args([
        "--overlay",
        &revoiced.voices.to_string(),
        "--progress",
    ]);
    if let Audio::Mono(prefix) = &audio {
        command.arg("--mono-prefix").arg(prefix);
    }
    if let Some(bits) = depth {
        command.args(["--fold-depth", &bits.to_string()]);
    }
    command.args(&options).arg("--out").arg(&out);
    progress(Event::message(format!(
        "harlettizer encode, overlay of {} voices",
        revoiced.voices
    )));
    let (said, result) = run_harlettizer(command, &log, progress)?;
    if let Err(e) = result {
        let refusals = said.refusals();
        if !refusals.is_empty() {
            return Err(Error::msg(refused(
                &refusals,
                Encoder::Hz,
                hz_opts,
                &out,
                &log,
            )));
        }
        return Err(said.failure(e, &log));
    }
    wrote(&out)?;
    audio.discard(config);
    Ok(Encoded {
        stream: out,
        log,
        summary: overlay_summary(&said.lines),
        depth,
    })
}

/// The depth the elements an overlay mixes a voice into are rounded to:
/// the original's, measured, unless the options already say it — they are
/// the user's, and a flag given twice is an error rather than a preference.
fn fold_depth(job: &Job, options: &[String], progress: Progress) -> Option<u32> {
    if options.iter().any(|o| o.starts_with("--fold-depth")) {
        return None;
    }
    let (bits, why) = depth::of_original(&job.extract_dir());
    progress(Event::message(match why {
        None => {
            format!("the original is mastered at {bits} bits; mixed elements are rounded to it")
        }
        Some(why) => format!("{why}: mixed elements are rounded to {bits} bits"),
    }));
    Some(bits)
}

/// The depth a FLAC stream is coded at when nothing chose it: from the
/// original's, as measured on its waveforms. `None` when the codec is not
/// FLAC, when a depth was chosen (it is among `options` then), or when the
/// original cannot tell — harlettizer then codes 24.
fn flac_depth(job: &Job, options: &[String], progress: Progress) -> Option<u32> {
    let manifest = &job.manifest;
    if manifest.iamf_codec != IamfCodec::Flac
        || manifest.iamf_bits.is_some()
        || names_codec(&options.join(" "))
    {
        return None;
    }
    match depth::measured(&job.extract_dir()) {
        Ok(original) => {
            let bits = iamf_bits_for(original);
            progress(Event::message(format!(
                "FLAC at {bits} bits: the original is mastered at {original}"
            )));
            Some(bits)
        }
        Err(why) => {
            progress(Event::message(format!(
                "{why}: FLAC at harlettizer's 24 bits"
            )));
            None
        }
    }
}

/// Encode with harlettizer as IAMF objects, into an `.mka` of one track:
/// the voices panned onto the master's objects (`encoder: iamf`) or put on
/// the bed (`encoder: iamf-lit`). No TrueHD anywhere, and no bound of
/// sixteen. The overlay rounds what it mixes to the original's depth, as the
/// TrueHD one does, so that the two carry the same scene.
pub fn iamf(
    config: &Config,
    job: &Job,
    revoiced: &Revoiced,
    encoder: Encoder,
    iamf_opts: &str,
    progress: Progress,
) -> Result<Encoded> {
    let program = tools::require(config, Tool::Harlettizer)?;
    let help = iamf_help(&program)?;
    iamf_mode(&program, &help, encoder)?;
    let options = split(iamf_opts)?;
    let out = stream(job, encoder);
    let log = job
        .master_dir()
        .join(format!("harlettizer_{}_{REVOICED}.log", encoder.code()));
    let depth = match encoder {
        Encoder::Iamf => fold_depth(job, &options, progress),
        _ => None,
    };
    let flac_bits = flac_depth(job, &options, progress);
    let options: Vec<String> = depth
        .map(|bits| vec!["--fold-depth".to_string(), bits.to_string()])
        .unwrap_or_default()
        .into_iter()
        .chain(
            flac_bits
                .map(|bits| vec!["--bits".to_string(), bits.to_string()])
                .unwrap_or_default(),
        )
        .chain(options)
        .collect();
    let audio = if help.contains("--mono-prefix") {
        progress(Event::message(
            "harlettizer reads the waveforms where they are (--mono-prefix): no CAF is written",
        ));
        Audio::Mono(link_waveforms(job, revoiced)?)
    } else {
        Audio::Interleaved(interleave(job, revoiced, progress)?)
    };
    let _ = fs::remove_file(&out);
    let command = iamf_command(
        &program,
        &revoiced.header,
        encoder,
        revoiced.voices,
        &out,
        &audio,
        &options,
    );
    let what = match encoder {
        Encoder::IamfBed => format!("{} voices on the bed", revoiced.voices),
        _ => format!("overlay of {} voices", revoiced.voices),
    };
    progress(Event::message(if iamf_opts.trim().is_empty() {
        format!("harlettizer iamf, {what}")
    } else {
        format!("harlettizer iamf, {what}, {}", iamf_opts.trim())
    }));
    let (said, result) = run_harlettizer(command, &log, progress)?;
    if let Err(e) = result {
        // The overlay's guards are the TrueHD overlay's, and say so the
        // same way.
        let refusals = said.refusals();
        if encoder == Encoder::Iamf && !refusals.is_empty() {
            return Err(Error::msg(refused(
                &refusals, encoder, iamf_opts, &out, &log,
            )));
        }
        return Err(said.failure(e, &log));
    }
    wrote(&out)?;
    audio.discard(config);
    let summary = match encoder {
        Encoder::Iamf => [overlay_summary(&said.lines), iamf_summary(&said.lines)]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
            .join("; "),
        _ => iamf_summary(&said.lines),
    };
    Ok(Encoded {
        stream: out,
        log,
        summary,
        depth,
    })
}

/// `harlettizer iamf <master> --objects --overlay|--voices-to-bed <voices>
/// --out <mka> --progress [--mono-prefix <p>] <options>`: the options last
/// — the depth this program measured, then the user's, as they are for
/// `encode`. `--objects` is left to them when they already give it — a
/// flag given twice is an error, not a preference.
fn iamf_command(
    program: &Path,
    header: &Path,
    encoder: Encoder,
    voices: usize,
    out: &Path,
    audio: &Audio,
    options: &[String],
) -> Command {
    let mut command = crate::run::command(program);
    command.arg("iamf").arg(header);
    if !options.iter().any(|o| o == "--objects") {
        command.arg("--objects");
    }
    if let Some(flag) = voices_flag(encoder) {
        command.args([flag, &voices.to_string()]);
    }
    command.arg("--out").arg(out).arg("--progress");
    if let Audio::Mono(prefix) = audio {
        command.arg("--mono-prefix").arg(prefix);
    }
    command.args(options);
    command
}

/// What `harlettizer iamf --help` says, or why this harlettizer cannot
/// encode IAMF into an `.mka`: it has no `iamf` at all, or one that writes
/// only a standalone `.iamf`. Asked of the binary, as [`reads_mono`] asks.
pub(crate) fn iamf_help(program: &Path) -> Result<String> {
    let out = crate::run::command(program)
        .args(["iamf", "--help"])
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| Error::io(program, e))?;
    let help = String::from_utf8_lossy(&out.stdout);
    if !out.status.success() || !help.contains("--out") {
        return Err(Error::msg(format!(
            "{} has no `iamf` command; encoder: iamf needs a harlettizer that writes IAMF into \
             Matroska (.mka)",
            program.display()
        )));
    }
    if !into_matroska(&help) {
        return Err(Error::msg(format!(
            "{} writes IAMF only as a standalone .iamf stream; encoder: iamf needs a newer \
             harlettizer, one that writes it into Matroska (.mka)",
            program.display()
        )));
    }
    Ok(help.into_owned())
}

/// Whether an `iamf --help` offers Matroska output.
fn into_matroska(help: &str) -> bool {
    help.contains(".mka") || help.to_ascii_lowercase().contains("matroska")
}

/// Whether an `iamf --help` offers what `encoder` asks of it, and if not
/// the reason: `--overlay` for `iamf`, `--voices-to-bed` for `iamf-lit`.
pub(crate) fn iamf_mode(program: &Path, help: &str, encoder: Encoder) -> Result<()> {
    let Some(flag) = voices_flag(encoder) else {
        return Ok(());
    };
    if offers(help, flag) {
        return Ok(());
    }
    Err(Error::msg(format!(
        "{}: its `iamf` has no {flag}; encoder: {} needs a harlettizer that {}",
        program.display(),
        encoder.code(),
        match encoder {
            Encoder::IamfBed => "puts the voices on the bed of an IAMF stream",
            _ => "overlays the voices onto the master's objects in an IAMF stream",
        }
    )))
}

/// Whether a `--help` names `flag` as an option of its own, not as the head
/// of a longer one (`--overlay` is not `--overlay-cost`).
fn offers(help: &str, flag: &str) -> bool {
    help.match_indices(flag).any(|(at, _)| {
        !help[at + flag.len()..]
            .chars()
            .next()
            .is_some_and(|c| c == '-' || c.is_ascii_alphanumeric())
    })
}

/// The harlettizer an IAMF run encodes with, refused with the reason when
/// it cannot: asked before a run starts as well as by the encode, so hours
/// of separation are not spent to learn it at the end.
pub fn require_iamf(config: &Config, encoder: Encoder) -> Result<PathBuf> {
    let program = tools::require(config, Tool::Harlettizer)?;
    let help = iamf_help(&program)?;
    iamf_mode(&program, &help, encoder)?;
    Ok(program)
}

/// Everything harlettizer said, sorted the way a failure is reported.
struct Said {
    /// Every line that was not a progress report, for the log.
    lines: Vec<String>,
    /// Its `error:` lines.
    errors: Vec<String>,
    /// Its last line on stderr that was neither an error nor a note.
    last_complaint: Option<String>,
}

impl Said {
    /// The overlay's refusals among its errors, `error: overlay: …`.
    fn refusals(&self) -> Vec<&String> {
        self.errors
            .iter()
            .filter(|e| e.starts_with("error: overlay:"))
            .collect()
    }

    /// The error a failed run ends with: harlettizer's own words.
    fn failure(&self, e: Error, log: &Path) -> Error {
        match self.errors.first().or(self.last_complaint.as_ref()) {
            Some(why) => Error::msg(format!("harlettizer failed: {why} (see {})", log.display())),
            None => e,
        }
    }
}

/// Run harlettizer with `--progress`, report how far it got, and write the
/// command and everything it said to `log`: what it said, and how it ended.
fn run_harlettizer(command: Command, log: &Path, progress: Progress) -> Result<(Said, Result<()>)> {
    let line = describe(&command);
    let mut lines = Vec::new();
    let mut errors = Vec::new();
    let mut last_complaint = None;
    let result = run::lines(command, |stream, text| {
        if let Some(percent) = percentage(text, "progress") {
            progress(Event::ratio("encoding", percent));
            return;
        }
        if stream == Stream::Stderr {
            if text.starts_with("error:") {
                errors.push(text.to_string());
            } else if !text.starts_with("note:") && !text.trim().is_empty() {
                // A failure that is not a refusal says so on its own line,
                // `harlettizer: <file>: <why>`; the notes before it are not
                // the reason.
                last_complaint = Some(text.to_string());
            }
        }
        if !text.trim().is_empty() {
            lines.push(text.to_string());
        }
    });
    write_log(log, &line, &lines)?;
    let said = Said {
        lines,
        errors,
        last_complaint,
    };
    Ok((said, result))
}

/// A run that ended well wrote its file, or says it did not.
fn wrote(out: &Path) -> Result<()> {
    if out.is_file() {
        return Ok(());
    }
    Err(Error::msg(format!(
        "harlettizer ended well but wrote no {}",
        out.display()
    )))
}

/// Encode with Dee: the whole scene re-clustered into `clusters` elements
/// — twelve, fourteen or sixteen, the nearest at or above what the
/// original carried.
pub fn dee(
    config: &Config,
    job: &Job,
    revoiced: &Revoiced,
    clusters: usize,
    progress: Progress,
) -> Result<Encoded> {
    let program = tools::require(config, Tool::Dee)?;
    let dir = job.master_dir();
    if !dir.to_string_lossy().is_ascii() {
        return Err(Error::msg(format!(
            "{}: Dee cannot open a path with an accent in it; move work_dir somewhere plain",
            dir.display()
        )));
    }
    let clusters = match clusters {
        0..=12 => 12,
        13..=14 => 14,
        _ => 16,
    };
    let out = stream(job, Encoder::Dee);
    let log = dir.join(format!("dee_{REVOICED}.log"));
    let xml = dir.join(format!("dee_{REVOICED}.xml"));
    let temporary = dir.join("_dee_tmp");
    fs::create_dir_all(&temporary).map_err(|e| Error::io(&temporary, e))?;
    let audio = interleave(job, revoiced, progress)?;
    let _ = fs::remove_file(&out);
    fs::write(
        &xml,
        job_xml(
            &file_name(&revoiced.header),
            &dir,
            &file_name(&out),
            &dir,
            &temporary,
            clusters,
        ),
    )
    .map_err(|e| Error::io(&xml, e))?;

    let mut command = if tools::needs_wine(&program) {
        let wine = tools::require(config, Tool::Wine)?;
        let mut c = crate::run::command(wine);
        c.arg(&program);
        c
    } else {
        crate::run::command(&program)
    };
    command
        .arg("--xml")
        .arg(&xml)
        .arg("--progress")
        .arg("--log-file")
        .arg(&log)
        .arg("--stdout");
    progress(Event::message(format!("Dee, {clusters} spatial clusters")));

    let mut violation = false;
    let finished = run::lines_until(command, |_, text| {
        if text.to_ascii_lowercase().contains("access violation") {
            violation = true;
            return false;
        }
        if let Some(percent) = percentage(text, "Overall progress:") {
            progress(Event::ratio("encoding", percent));
        }
        true
    });
    let _ = fs::remove_dir_all(&temporary);
    if violation {
        return Err(Error::msg(format!(
            "Dee hit an access violation and was stopped — under Wine it hangs after one instead \
             of exiting. It is most often memory pressure: close what can be closed and run again. \
             Its log: {}",
            log.display()
        )));
    }
    finished?;
    if !out.is_file() {
        return Err(Error::msg(format!(
            "Dee ended well but wrote no {}; its log: {}",
            out.display(),
            log.display()
        )));
    }
    discard(config, &audio);
    Ok(Encoded {
        stream: out,
        log,
        summary: format!("Dee, {clusters} spatial clusters"),
        depth: None,
    })
}

/// The master set's interleaved CAF, built from its waveforms unless one of
/// the right size is already there.
fn interleave(job: &Job, revoiced: &Revoiced, progress: Progress) -> Result<PathBuf> {
    let audio = job.master_dir().join(format!("{REVOICED}.atmos.audio"));
    // 8 file header + 44 desc + 12 data header + 4 edit count.
    let expected = 68 + revoiced.frames * revoiced.inputs.len() as u64 * 3;
    if fs::metadata(&audio).is_ok_and(|m| m.len() == expected) {
        return Ok(audio);
    }
    progress(Event::message(format!(
        "interleaving {} waveforms into {}",
        revoiced.inputs.len(),
        audio.display()
    )));
    let temporary = audio.with_extension("audio.part");
    caf::interleave(
        &revoiced.inputs,
        &temporary,
        revoiced.frames,
        48000.0,
        progress,
    )?;
    fs::rename(&temporary, &audio).map_err(|e| Error::io(&audio, e))?;
    Ok(audio)
}

/// The CAF goes once the stream is written, unless the job is to be kept
/// whole: it is the largest file of the job and the cheapest to rebuild.
fn discard(config: &Config, audio: &Path) {
    if !config.keep_work {
        let _ = fs::remove_file(audio);
    }
}

/// The option of harlettizer's that declines the bound a refusal names;
/// `None` for a refusal no option declines.
fn declining(refusal: &str) -> Option<&'static str> {
    if refusal.contains("cost the worst source") {
        Some("--overlay-cost")
    } else if refusal.contains("° from where it asked to be") {
        Some("--overlay-drift")
    } else if refusal.contains("left no element holding") {
        Some("--overlay-spread")
    } else if refusal.contains("movement nobody asked for") {
        Some("--overlay-wobble")
    } else if refusal.contains("from the level it asked for") {
        Some("--overlay-level")
    } else {
        None
    }
}

/// What a refused overlay tells the user: the encoder's own lines, where
/// the stream it wrote anyway is, and what would get past each refusal.
/// The TrueHD overlay (`encoder: hz`) and the IAMF one (`encoder: iamf`)
/// have the same guards, under the same names.
///
/// The `--hz-opts` (or `--iamf-opts`) it offers keeps the options already
/// in use — the flag replaces them, and a suggestion that dropped
/// `--presentations` would quietly take the real folds out of the stream.
fn refused(refusals: &[&String], encoder: Encoder, in_use: &str, out: &Path, log: &Path) -> String {
    let mut text = String::from("harlettizer refused the overlay:\n");
    for refusal in refusals {
        text.push_str(&format!("  {refusal}\n"));
    }
    if out.is_file() {
        text.push_str(&format!(
            "The stream was written anyway and left at {} so it can be heard; the whole account \
             is in {}.\n",
            out.display(),
            log.display()
        ));
    } else {
        text.push_str(&format!("The whole account is in {}.\n", log.display()));
    }
    let flags: Vec<&str> = refusals.iter().filter_map(|r| declining(r)).collect();
    if !flags.is_empty() {
        let mut options = in_use.trim().to_string();
        for flag in &flags {
            if !options.contains(flag) {
                if !options.is_empty() {
                    options.push(' ');
                }
                options.push_str(&format!("{flag} 0"));
            }
        }
        let option = if encoder.is_iamf() {
            "--iamf-opts"
        } else {
            "--hz-opts"
        };
        text.push_str(&format!(
            "To accept it as it is, decline the bound that refused it — the separation is kept, \
             only the encode runs again:\n  {option} \"{options}\"\n"
        ));
    }
    if flags.len() < refusals.len() {
        text.push_str(
            "Some of it no option declines (voices with no element to carry them, or samples \
             clipped); a lower --gain-db helps the second.\n",
        );
    }
    text.push_str(if encoder.is_iamf() {
        "Or put the voices on the bed (--encoder iamf-lit), the master's other objects left as \
         they are."
    } else {
        "Or encode with Dee (--encoder dee), which re-clusters the whole scene."
    });
    text
}

/// How the encoder is handed the master's audio.
enum Audio {
    /// The CAF the header names.
    Interleaved(PathBuf),
    /// `<prefix>_<n>.wav`, one per waveform, linked to where they already
    /// are.
    Mono(PathBuf),
}

impl Audio {
    fn discard(&self, config: &Config) {
        match self {
            Audio::Interleaved(path) => discard(config, path),
            Audio::Mono(prefix) => {
                if !config.keep_work {
                    let _ = unlink_waveforms(prefix);
                }
            }
        }
    }
}

/// Whether this harlettizer reads a master's waveforms from their own
/// files. Asked of the binary rather than inferred from a version: it is
/// what the binary says it takes that decides what can be passed to it.
fn reads_mono(program: &Path) -> bool {
    crate::run::command(program)
        .args(["encode", "--help"])
        .stdin(std::process::Stdio::null())
        .output()
        .is_ok_and(|out| String::from_utf8_lossy(&out.stdout).contains("--mono-prefix"))
}

/// Link the master's waveforms, in its order, under one prefix — the names
/// `--mono-prefix` reads. A hard link where the filesystem allows it (it
/// always does inside one job), a copy where it does not. A hard link shares
/// its file, which is safe because no step here rewrites a waveform in
/// place: each writes a new file and renames it over the old, and a link
/// keeps what it was made to.
fn link_waveforms(job: &Job, revoiced: &Revoiced) -> Result<PathBuf> {
    let prefix = job.master_dir().join("wave");
    unlink_waveforms(&prefix)?;
    for (n, input) in revoiced.inputs.iter().enumerate() {
        let link = waveform(&prefix, n);
        if fs::hard_link(input, &link).is_err() {
            fs::copy(input, &link).map_err(|e| Error::io(&link, e))?;
        }
    }
    Ok(prefix)
}

fn unlink_waveforms(prefix: &Path) -> Result<()> {
    let mut n = 0;
    loop {
        let path = waveform(prefix, n);
        if !path.exists() {
            return Ok(());
        }
        fs::remove_file(&path).map_err(|e| Error::io(&path, e))?;
        n += 1;
    }
}

/// `<prefix>_<n>.wav`, the convention `harletty decode --mono-prefix` writes
/// and `harlettizer encode --mono-prefix` reads.
fn waveform(prefix: &Path, n: usize) -> PathBuf {
    let mut name = prefix.as_os_str().to_owned();
    name.push(format!("_{n}.wav"));
    PathBuf::from(name)
}

/// `progress 42%` from harlettizer, `Overall progress: 42.9.` from Dee.
fn percentage(line: &str, label: &str) -> Option<f64> {
    let rest = line.trim().strip_prefix(label)?.trim_start();
    let number: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == ',')
        .collect();
    let number = number.trim_end_matches(['.', ',']).replace(',', ".");
    let value: f64 = number.parse().ok()?;
    Some((value / 100.0).clamp(0.0, 1.0))
}

/// What an overlay says of itself, in one line: how much of the original
/// went through untouched, how far the voices ended up from their places,
/// and any reservation the encoder had. The whole account is in the log.
fn overlay_summary(said: &[String]) -> String {
    let field = |key: &str| field(said, key);
    // The head of a line, up to the first of its separators: the tails are
    // useful and long, and belong in the log.
    let head = |text: String, cuts: &[&str]| {
        let end = cuts
            .iter()
            .filter_map(|cut| text.find(cut))
            .filter(|at| *at > 0)
            .min()
            .unwrap_or(text.len());
        text[..end].trim().to_string()
    };
    let mut parts = Vec::new();
    if let Some(what) = field("overlay") {
        parts.push(what);
    }
    if let Some(untouched) = field("untouched") {
        parts.push(head(untouched, &[" ("]));
    }
    if let Some(drift) = field("drift") {
        parts.push(head(drift, &[","]));
    }
    if let Some(fell_back) = field("fell back") {
        parts.push(format!(
            "put on the bed outright: {}",
            head(fell_back, &[" ("])
        ));
    }
    if let Some(stranded) = field("stranded") {
        parts.push(format!("with nowhere to go: {}", head(stranded, &[" —"])));
    }
    let notes = said
        .iter()
        .filter(|line| line.starts_with("  note "))
        .count();
    if notes > 0 {
        parts.push(format!(
            "{notes} reservation{} of the encoder's, in the log",
            if notes > 1 { "s" } else { "" }
        ));
    }
    parts.join("; ")
}

/// What an IAMF encode says of itself, in one line: what the programme
/// became, how it was coded, how loud it measured, and any warning. The
/// rest is in the log.
fn iamf_summary(said: &[String]) -> String {
    let mut parts = Vec::new();
    parts.extend(field(said, "programme"));
    // `<units> temporal units of <n>, <samples> samples, <coding>[, …]`.
    parts.extend(
        field(said, "encoded").and_then(|encoded| encoded.split(", ").nth(2).map(str::to_string)),
    );
    parts.extend(field(said, "loudness"));
    // The voices on the bed: their element and how far a listener may move
    // it, then their own loudness — the heads of the `dialogue` and
    // `anchored` lines.
    parts.extend(field(said, "dialogue").map(|line| head(&line, ";")));
    parts.extend(field(said, "anchored").map(|line| head(&line, ",")));
    parts.extend(
        said.iter()
            .filter_map(|line| line.strip_prefix("warning: "))
            .map(str::to_string),
    );
    parts.join("; ")
}

/// `line` up to the first `separator`, or all of it.
fn head(line: &str, separator: &str) -> String {
    line.split(separator)
        .next()
        .unwrap_or(line)
        .trim()
        .to_string()
}

/// The value of a line of harlettizer's summary, `  <key>  <value>`.
fn field(said: &[String], key: &str) -> Option<String> {
    said.iter().find_map(|line| {
        line.strip_prefix("  ")
            .and_then(|rest| rest.strip_prefix(key))
            .filter(|rest| rest.starts_with(' '))
            .map(|rest| rest.trim().to_string())
    })
}

fn write_log(log: &Path, command: &str, said: &[String]) -> Result<()> {
    let mut text =
        String::with_capacity(command.len() + said.iter().map(|l| l.len() + 1).sum::<usize>() + 2);
    text.push_str(command);
    text.push_str("\n\n");
    for line in said {
        text.push_str(line);
        text.push('\n');
    }
    fs::write(log, text).map_err(|e| Error::io(log, e))
}

/// The command as a person would type it, for the head of the log: a log
/// that does not say how the stream was made is no use six months later.
fn describe(command: &Command) -> String {
    let quote = |text: &str| {
        if text.is_empty() || text.contains(|c: char| c.is_whitespace() || c == '"' || c == '\'') {
            format!("'{}'", text.replace('\'', "'\\''"))
        } else {
            text.to_string()
        }
    };
    std::iter::once(command.get_program())
        .chain(command.get_args())
        .map(|part| quote(&part.to_string_lossy()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Split an option string as a shell would at its simplest: on whitespace,
/// with single or double quotes around anything that holds some.
pub fn split(text: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut started = false;
    for c in text.chars() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => current.push(c),
            None if c == '"' || c == '\'' => {
                quote = Some(c);
                started = true;
            }
            None if c.is_whitespace() => {
                if started {
                    out.push(std::mem::take(&mut current));
                    started = false;
                }
            }
            None => {
                current.push(c);
                started = true;
            }
        }
    }
    if quote.is_some() {
        return Err(Error::msg(format!(
            "unbalanced quote in the options `{text}`"
        )));
    }
    if started {
        out.push(current);
    }
    Ok(out)
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The job Dee runs: the official `encode_to_dthd` template of Dee 5.2.1,
/// its defaults kept (film_light DRC, 1770-4 loudness with dialogue
/// intelligence), the spatial clusters set to the original's.
fn job_xml(
    input: &str,
    input_dir: &Path,
    output: &str,
    output_dir: &Path,
    temporary: &Path,
    clusters: usize,
) -> String {
    format!(
        r#"<?xml version="1.0"?>
<job_config>
  <input>
    <audio>
      <atmos_mezz version="1">
        <file_name>{input}</file_name>
        <timecode_frame_rate>not_indicated</timecode_frame_rate>
        <offset>auto</offset>
        <ffoa>auto</ffoa>
        <storage>
          <local>
            <path>{input_dir}</path>
          </local>
        </storage>
      </atmos_mezz>
    </audio>
  </input>
  <filter>
    <audio>
      <encode_to_dthd version="1">
        <loudness_measurement>
          <metering_mode>1770-4</metering_mode>
          <dialogue_intelligence>true</dialogue_intelligence>
          <speech_threshold>15</speech_threshold>
        </loudness_measurement>
        <timecode_frame_rate>not_indicated</timecode_frame_rate>
        <start>first_frame_of_action</start>
        <end>end_of_file</end>
        <time_base>file_position</time_base>
        <prepend_silence_duration>0</prepend_silence_duration>
        <append_silence_duration>0</append_silence_duration>
        <atmos_presentation>
          <drc_profile>film_light</drc_profile>
          <spatial_clusters>{clusters}</spatial_clusters>
          <legacy_authoring_compatibility>true</legacy_authoring_compatibility>
        </atmos_presentation>
        <presentation_8ch>
          <drc_profile>film_light</drc_profile>
          <surround_3db_attenuation>true</surround_3db_attenuation>
        </presentation_8ch>
        <presentation_6ch>
          <drc_profile>film_light</drc_profile>
          <surround_3db_attenuation>true</surround_3db_attenuation>
        </presentation_6ch>
        <presentation_2ch>
          <drc_profile>film_light</drc_profile>
          <drc_default_on>true</drc_default_on>
          <format>stereo</format>
        </presentation_2ch>
        <optimize_data_rate>false</optimize_data_rate>
        <embedded_timecodes>
          <starting_timecode>off</starting_timecode>
          <frame_rate>auto</frame_rate>
        </embedded_timecodes>
        <log_format>txt</log_format>
      </encode_to_dthd>
    </audio>
  </filter>
  <output>
    <mlp version="1">
      <file_name>{output}</file_name>
      <storage>
        <local>
          <path>{output_dir}</path>
        </local>
      </storage>
    </mlp>
  </output>
  <misc>
    <temp_dir>
      <clean_temp>true</clean_temp>
      <path>{temporary}</path>
    </temp_dir>
  </misc>
</job_config>
"#,
        input_dir = input_dir.display(),
        output_dir = output_dir.display(),
        temporary = temporary.display(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refusal_says_which_option_declines_it_and_keeps_the_rest() {
        let cost = "error: overlay: 22.1 % of the voiced blocks cost the worst source more than 0.50 of its own gain vector, 1.514 at its worst".to_string();
        let drift = "error: overlay: 7.0 % of the blocks a source was audible in put it more than 30° from where it asked to be, which is past the 5 % this refuses at".to_string();
        assert_eq!(declining(&cost), Some("--overlay-cost"));
        assert_eq!(declining(&drift), Some("--overlay-drift"));
        let said = refused(
            &[&cost, &drift],
            Encoder::Hz,
            "--presentations",
            Path::new("/job/vf.thd"),
            Path::new("/job/log"),
        );
        assert!(
            said.contains(r#"--hz-opts "--presentations --overlay-cost 0 --overlay-drift 0""#),
            "{said}"
        );
        assert!(!said.contains("no option declines"), "{said}");
        let stranded = "error: overlay: 40 source-blocks (6.0 %) were audible with no element to carry them at all".to_string();
        assert_eq!(declining(&stranded), None);
        let said = refused(
            &[&stranded],
            Encoder::Hz,
            "",
            Path::new("/job/vf.thd"),
            Path::new("/job/log"),
        );
        assert!(
            said.contains("no option declines") && said.contains("--encoder dee"),
            "{said}"
        );
        assert!(!said.contains("--hz-opts"), "{said}");
    }

    #[test]
    fn an_iamf_overlay_is_refused_as_the_truehd_one_is() {
        let level = "error: overlay: 3.0 % of the voiced blocks put a source more than 3 dB from the level it asked for".to_string();
        let said = refused(
            &[&level],
            Encoder::Iamf,
            "--codec opus",
            Path::new("/job/nowhere/vf.mka"),
            Path::new("/job/log"),
        );
        assert!(
            said.contains(r#"--iamf-opts "--codec opus --overlay-level 0""#),
            "{said}"
        );
        assert!(!said.contains("--hz-opts"), "{said}");
        assert!(said.contains("--encoder iamf-lit"), "{said}");
        assert!(!said.contains("--encoder dee"), "{said}");
        // No stream to point at when none was left.
        assert!(!said.contains("written anyway"), "{said}");
        assert!(said.contains("The whole account is in /job/log"), "{said}");
    }

    #[test]
    fn an_overlay_needs_the_original_to_fit_a_stream() {
        assert!(overlay_fits(12).is_ok());
        assert!(overlay_fits(16).is_ok());
        let e = overlay_fits(17).unwrap_err().to_string();
        assert!(
            e.contains("17 elements") && e.contains("--encoder dee"),
            "{e}"
        );
    }

    #[test]
    fn waveforms_are_linked_under_the_names_mono_prefix_reads() {
        let dir = std::env::temp_dir().join(format!("omnifrog-link-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let prefix = dir.join("wave");
        assert_eq!(waveform(&prefix, 3), dir.join("wave_3.wav"));
        for n in 0..3 {
            fs::write(waveform(&prefix, n), b"x").unwrap();
        }
        unlink_waveforms(&prefix).unwrap();
        assert!(!waveform(&prefix, 0).exists());
        assert!(!waveform(&prefix, 2).exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn options_split_as_a_shell_would() {
        assert_eq!(
            split("--presentations --overlay-cost 0").unwrap(),
            ["--presentations", "--overlay-cost", "0"]
        );
        assert_eq!(
            split(r#"--config "/a path/c.yaml"  --fast"#).unwrap(),
            ["--config", "/a path/c.yaml", "--fast"]
        );
        assert_eq!(split("  ").unwrap(), Vec::<String>::new());
        assert_eq!(split("''").unwrap(), [""]);
        assert!(split(r#"--config "unfinished"#).is_err());
    }

    #[test]
    fn progress_is_read_off_either_encoder() {
        assert_eq!(percentage("progress 42%", "progress"), Some(0.42));
        assert_eq!(percentage("progress 100%", "progress"), Some(1.0));
        let dee = percentage("Overall progress: 42.9.", "Overall progress:").unwrap();
        assert!((dee - 0.429).abs() < 1e-9);
        assert_eq!(percentage("wrote vf.thd", "progress"), None);
    }

    #[test]
    fn an_overlay_summary_is_the_heads_of_its_lines() {
        let said: Vec<String> = [
            "  overlay      21 elements kept, 5 sources panned onto them",
            "  untouched    96.3 % of the element-blocks copied through, neither mixed nor rounded (934 of 970); 40.0 % of the programme's blocks touched nothing at all",
            "  drift        3.12° mean between where a source asked to be and where its carriers put it, 19.96° at its worst, over 2.00 audible sources a block",
            "  beds         7 of 21 elements are bed channels, all 7 declared by the master",
            "  note         a source reached 19.9° from where it asked to be",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let summary = overlay_summary(&said);
        assert_eq!(
            summary,
            "21 elements kept, 5 sources panned onto them; \
             96.3 % of the element-blocks copied through, neither mixed nor rounded; \
             3.12° mean between where a source asked to be and where its carriers put it; \
             1 reservation of the encoder's, in the log"
        );
    }

    #[test]
    fn the_dee_job_names_what_it_reads_and_writes() {
        let xml = job_xml(
            "vf.atmos",
            Path::new("/job/3.master"),
            "vf.thd",
            Path::new("/job/3.master"),
            Path::new("/job/3.master/_dee_tmp"),
            16,
        );
        assert!(xml.contains("<file_name>vf.atmos</file_name>"));
        assert!(xml.contains("<spatial_clusters>16</spatial_clusters>"));
        assert!(xml.contains("<path>/job/3.master/_dee_tmp</path>"));
        assert!(xml.starts_with("<?xml"));
    }

    #[test]
    fn the_iamf_command_says_the_mode_and_puts_the_users_options_last() {
        let command = |encoder, audio: &Audio, options: &str| {
            describe(&iamf_command(
                Path::new("harlettizer"),
                Path::new("/job/3.master/vf.atmos"),
                encoder,
                5,
                Path::new("/job/3.master/vf.mka"),
                audio,
                &split(options).unwrap(),
            ))
        };
        let mono = Audio::Mono(PathBuf::from("/job/3.master/wave"));
        assert_eq!(
            command(
                Encoder::Iamf,
                &mono,
                "--fold-depth 20 --codec opus --bitrate 48"
            ),
            "harlettizer iamf /job/3.master/vf.atmos --objects --overlay 5 \
             --out /job/3.master/vf.mka --progress --mono-prefix /job/3.master/wave \
             --fold-depth 20 --codec opus --bitrate 48"
        );
        assert_eq!(
            command(Encoder::IamfBed, &mono, ""),
            "harlettizer iamf /job/3.master/vf.atmos --objects --voices-to-bed 5 \
             --out /job/3.master/vf.mka --progress --mono-prefix /job/3.master/wave"
        );
        // No prefix for a harlettizer that reads the CAF, and `--objects`
        // once when the options already give it.
        let caf = Audio::Interleaved(PathBuf::from("/job/3.master/vf.atmos.audio"));
        assert_eq!(
            command(
                Encoder::Iamf,
                &caf,
                "--fold-depth 20 --objects --codec lpcm"
            ),
            "harlettizer iamf /job/3.master/vf.atmos --overlay 5 --out /job/3.master/vf.mka \
             --progress --fold-depth 20 --objects --codec lpcm"
        );
    }

    #[test]
    fn either_iamf_has_a_stream_of_its_own() {
        let config = Config {
            work_dir: Some("/work".into()),
            ..Config::default()
        };
        let job = Job::new(&config, crate::job::tests::manifest());
        let master = job.master_dir();
        assert_eq!(stream(&job, Encoder::Hz), master.join("vf.thd"));
        assert_eq!(stream(&job, Encoder::Dee), master.join("vf.thd"));
        assert_eq!(stream(&job, Encoder::Iamf), master.join("vf.mka"));
        assert_eq!(stream(&job, Encoder::IamfBed), master.join("vf-bed.mka"));
    }

    #[test]
    fn a_harlettizer_without_the_mode_is_refused_for_it() {
        let both = "  --objects\n  --overlay <VOICES>  ...\n  --overlay-cost <X>\n  \
                    --voices-to-bed <VOICES>  ...\n  --out <OUT>  .mka";
        let program = Path::new("harlettizer");
        assert!(iamf_mode(program, both, Encoder::Iamf).is_ok());
        assert!(iamf_mode(program, both, Encoder::IamfBed).is_ok());
        // `--overlay-cost` alone is not `--overlay`.
        let guards_only = "  --overlay-cost <X>\n  --out <OUT>  .mka";
        let e = iamf_mode(program, guards_only, Encoder::Iamf)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("no --overlay") && e.contains("encoder: iamf"),
            "{e}"
        );
        let e = iamf_mode(program, guards_only, Encoder::IamfBed)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("no --voices-to-bed") && e.contains("encoder: iamf-lit"),
            "{e}"
        );
        // Nothing to ask for the TrueHD encoders.
        assert!(iamf_mode(program, "", Encoder::Hz).is_ok());
        assert!(offers("--overlay\n", "--overlay"));
        assert!(offers("--overlay <N>", "--overlay"));
        assert!(!offers("--overlay-drift", "--overlay"));
    }

    #[test]
    fn only_a_harlettizer_that_writes_matroska_encodes_iamf() {
        assert!(!into_matroska(
            "The output is a standalone `.iamf` stream; `docs/iamf.md` has the FFmpeg line"
        ));
        assert!(into_matroska(
            "--out <OUT>  Write the stream here; a name ending in .mka"
        ));
        assert!(into_matroska("written into Matroska, one track"));
    }

    #[test]
    fn an_iamf_summary_is_the_programme_and_its_loudness() {
        let said: Vec<String> = [
            "  programme    12 bed channels, 11 objects, rendered to 7.1.4 on the room's cube",
            "  encoded      3515 temporal units of 4096, 14395520 samples, FLAC 24-bit, 7 substreams",
            "  loudness     -27.4 LKFS on 7.1.4, -25.9 LKFS on stereo",
            "  headroom     0 samples clipped, the bed peaking at 0.71 of full scale before the codec's integers",
            "  wrote        /job/3.master/vf.mka, 1 bytes, 1 kbit/s",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(
            iamf_summary(&said),
            "12 bed channels, 11 objects, rendered to 7.1.4 on the room's cube; FLAC 24-bit; \
             -27.4 LKFS on 7.1.4, -25.9 LKFS on stereo"
        );
        let mut clipped = said.clone();
        clipped.push("warning: the render clipped 3 samples".into());
        assert!(iamf_summary(&clipped).ends_with("; the render clipped 3 samples"));
    }

    #[test]
    fn the_voices_on_the_bed_say_where_they_went_and_how_loud() {
        let said: Vec<String> = [
            "  programme    11 objects, the bed as an LFE element, a 7.1 dialogue element: 13 elements, 20 channels, IAMF v2.0 advanced-2",
            "  encoded      235 temporal units of 4096, 960000 samples, FLAC 24-bit",
            "  dialogue     the last 5 objects rendered into a 7.1 element of their own on the room's cube, the smallest layout that holds where they go; a listener may move it 12 dB either way",
            "  loudness     -18.2 LKFS on 7.1.4, -16.1 LKFS on stereo",
            "  anchored     the dialogue element alone at -36.1 LKFS on 7.1.4, -36.0 LKFS on stereo, stated as the loudness anchored on dialogue",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(
            iamf_summary(&said),
            "11 objects, the bed as an LFE element, a 7.1 dialogue element: 13 elements, 20 channels, \
             IAMF v2.0 advanced-2; FLAC 24-bit; -18.2 LKFS on 7.1.4, -16.1 LKFS on stereo; the last 5 \
             objects rendered into a 7.1 element of their own on the room's cube, the smallest layout \
             that holds where they go; the dialogue element alone at -36.1 LKFS on 7.1.4"
        );
    }

    #[test]
    fn a_logged_command_can_be_pasted_back() {
        let mut c = crate::run::command("harlettizer");
        c.args(["encode", "/a b/vf.atmos", "--overlay", "5"]);
        assert_eq!(
            describe(&c),
            "harlettizer encode '/a b/vf.atmos' --overlay 5"
        );
    }
}
