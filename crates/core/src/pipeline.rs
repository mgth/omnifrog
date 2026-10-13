//! A run, from a file to a track: the steps in order.
//!
//! What the command line asks for and what an egui client would ask for is
//! the same thing — one call, one stream of events — so the order of the
//! steps, the choice of tracks and the decision of where to pick up all
//! live here rather than in a front end.

use crate::config::{Config, Encoder, IamfCodec, Pairs};
use crate::job::{EncodeState, Job, Manifest, MasterState, SeparateState, Step};
use crate::master::{MasterSet, Revoiced};
use crate::models::Catalogue;
use crate::probe::{AudioTrack, Media};
use crate::progress::{Event, Progress};
use crate::{Error, Result, encode, extract, master, probe, remux, separate, tracks, voices, wav};
use std::path::PathBuf;
use std::time::Instant;

/// What a run was asked for. Everything absent is the configuration's.
#[derive(Debug, Clone, Default)]
pub struct Options {
    pub source: PathBuf,
    /// Rank of the original among the audio tracks, from 1.
    pub original: Option<u32>,
    pub dub: Option<u32>,
    /// Read only this many seconds, for a trial.
    pub seconds: Option<f64>,
    /// Read from this point of the film, in seconds, for a trial of a scene.
    pub start: Option<f64>,
    /// A full `.mkv` rather than an `.mka` of the new track alone.
    pub mkv: bool,
    pub encoder: Option<Encoder>,
    /// Offset of the dub's voices against the original's dialogue.
    pub gain_db: f64,
    pub hz_opts: Option<String>,
    /// Options of `harlettizer iamf`, replacing the configuration's.
    pub iamf_opts: Option<String>,
    /// The IAMF codec, replacing the configuration's.
    pub iamf_codec: Option<IamfCodec>,
    /// Opus kilobits a second for each channel, replacing the configuration's.
    pub iamf_bitrate: Option<u32>,
    /// FLAC's depth, replacing the configuration's: `Some(None)` is the
    /// original's.
    pub iamf_bits: Option<Option<u32>>,
    /// Which separations work in pairs, replacing the configuration's.
    pub pairs: Option<Pairs>,
    /// Cancel the duplicates between the original and the dub, replacing
    /// the configuration's.
    pub cancel_duplicates: Option<bool>,
    /// Where the track goes; beside the source when absent.
    pub out_dir: Option<PathBuf>,
    /// Start over, whatever the manifest says is done.
    pub restart: bool,
}

/// Where a run got to.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub dir: PathBuf,
    /// The track.
    pub output: PathBuf,
    /// The first step this run did, `None` when there was nothing to do.
    pub resumed_at: Option<Step>,
}

/// Re-voice a film.
pub fn run(config: &Config, options: &Options, progress: Progress) -> Result<Outcome> {
    if options.mkv && options.seconds.is_some() {
        // A trial is a track to listen to; a film whose new track stops
        // after a minute, around a picture that goes on for two hours, is
        // not one — and cutting the picture is not something this program does.
        return Err(Error::msg(
            "--seconds makes a trial, which is an .mka to listen to; --mkv writes the whole film",
        ));
    }
    if options.mkv && options.start.is_some_and(|start| start > 0.0) {
        return Err(Error::msg(
            "--start makes a trial, which is an .mka to listen to; --mkv writes the whole film",
        ));
    }
    let media = probe::media(config, &options.source, progress)?;
    let (original, dub) = choose(&media, options)?;
    progress(Event::message(format!(
        "original: track {} · {}",
        original.index,
        original.summary().trim_end()
    )));
    progress(Event::message(format!(
        "dub:      track {} · {}",
        dub.index,
        dub.summary().trim_end()
    )));

    let catalogue = Catalogue::load()?;
    let model_original = catalogue.get(&config.models.vo)?;
    let model_dub = catalogue.get(&config.models.vf)?;
    let mut manifest = Manifest::new(&media, original, dub, (model_original, model_dub), config);
    manifest.seconds = options.seconds;
    manifest.start = options.start.filter(|start| *start > 0.0);
    manifest.mkv = options.mkv;
    manifest.gain_db = options.gain_db;
    if let Some(encoder) = options.encoder {
        manifest.encoder = encoder;
    }
    if let Some(hz_opts) = &options.hz_opts {
        manifest.hz_opts = hz_opts.clone();
    }
    if let Some(iamf_opts) = &options.iamf_opts {
        manifest.iamf_opts = iamf_opts.clone();
    }
    if let Some(codec) = options.iamf_codec {
        manifest.iamf_codec = codec;
    }
    if let Some(bitrate) = options.iamf_bitrate {
        manifest.iamf_bitrate = bitrate;
    }
    if let Some(bits) = options.iamf_bits {
        manifest.iamf_bits = bits;
    }
    if let Some(pairs) = options.pairs {
        manifest.pairs = pairs.effective();
    }
    if let Some(cancel) = options.cancel_duplicates {
        manifest.cancel_duplicates = cancel;
    }
    manifest.out_dir = options.out_dir.clone();

    let mut job = Job::new(config, manifest);
    let (previous, complaint) = job.previous();
    if let Some(complaint) = complaint {
        progress(Event::message(complaint));
    }
    // What the previous run measured and left behind, before anything is
    // asked about it: `gone` looks the recorded files up on disk.
    if let Some(previous) = &previous {
        job.manifest.carry_over(previous);
    }
    let from = if options.restart {
        Some(Step::Extract)
    } else {
        let by_parameters = job.manifest.resume_at(previous.as_ref());
        match (by_parameters, gone(&job, progress)) {
            (Some(step), Some(missing)) => Some(step.min(missing)),
            (None, missing) => missing,
            (step, None) => step,
        }
    };
    if let Some(from) = from {
        // Everything from the resumption point on is to be done again.
        job.manifest.done = job.manifest.done.min(from.number() - 1);
    }
    // A harlettizer that cannot write IAMF into an .mka, or not in the mode
    // asked for, is known now, not
    // after the hours of separation before the encode — and before the
    // manifest is touched, so a refused run leaves the job as it was.
    if job.manifest.encoder.is_iamf() && from.is_some_and(|from| from <= Step::Encode) {
        encode::require_iamf(config, job.manifest.encoder)?;
    }
    job.save()?;
    let outcome = Outcome {
        dir: job.dir.clone(),
        output: job.output(),
        resumed_at: from,
    };
    let Some(from) = from else {
        progress(Event::message(format!(
            "job {} — every step is done with these parameters",
            job.dir.display()
        )));
        return Ok(outcome);
    };
    progress(Event::message(format!(
        "job {} — starting at step {}/{} ({})",
        job.dir.display(),
        from.number(),
        Step::ALL.len(),
        from.label()
    )));
    for step in Step::ALL {
        if step < from {
            continue;
        }
        let started = Instant::now();
        match step {
            Step::Extract => {
                announce(step, progress);
                let extracted = extract::run(
                    config,
                    &extract::Plan {
                        source: &media.path,
                        original,
                        dub,
                        dir: &job.extract_dir(),
                        duration: media.duration,
                        seconds: options.seconds,
                        start: job.manifest.start,
                    },
                    progress,
                )?;
                job.manifest.extract = Some(crate::job::ExtractState {
                    source_codec: extracted.master.source_codec.clone(),
                    elements: extracted.master.channels.iter().map(|c| c.id).collect(),
                    objects: extracted.master.objects().count(),
                    dub_channels: extracted.dub.iter().map(|(n, _)| n.clone()).collect(),
                    frames: extracted.frames,
                    begin: extracted.begin,
                });
            }
            Step::SeparateOriginal => {
                announce(step, progress);
                let master = MasterSet::find(&job.extract_dir())?;
                let signature = separate::signature(
                    &job.manifest.model_original,
                    &job.manifest.source,
                    job.manifest.original.index,
                    job.manifest.seconds,
                    job.manifest.start,
                    separate::mode_original(job.manifest.pairs),
                );
                claim(&mut job, separate::Stage::Original, &signature)?;
                let done = separate::original(
                    config,
                    &job,
                    model_original,
                    &master,
                    job.manifest.pairs,
                    progress,
                )?;
                job.manifest.separate_original = Some(SeparateState {
                    signature,
                    elements: done.elements,
                    lkfs: done.lkfs,
                    identical: None,
                });
            }
            Step::SeparateDub => {
                announce(step, progress);
                let channels = job
                    .manifest
                    .extract
                    .as_ref()
                    .map(|state| state.dub_channels.clone())
                    .ok_or_else(|| {
                        Error::msg("the extraction did not record the dub's channels")
                    })?;
                let signature = separate::signature(
                    &job.manifest.model_dub,
                    &job.manifest.source,
                    job.manifest.dub.index,
                    job.manifest.seconds,
                    job.manifest.start,
                    separate::mode_dub(job.manifest.pairs),
                );
                claim(&mut job, separate::Stage::Dub, &signature)?;
                let (done, analysis) = separate::dub(
                    config,
                    &job,
                    model_dub,
                    &channels,
                    job.manifest.pairs,
                    progress,
                )?;
                job.manifest.separate_dub = Some(SeparateState {
                    signature,
                    elements: done.elements,
                    lkfs: done.lkfs,
                    identical: Some(analysis.identical),
                });
                if let Some(gain) = job.manifest.voice_gain_db() {
                    progress(Event::message(format!(
                        "the dub's voices will be levelled by {gain:+.2} dB"
                    )));
                }
            }
            Step::Master => {
                announce(step, progress);
                let dub_state = job
                    .manifest
                    .separate_dub
                    .clone()
                    .ok_or_else(|| Error::msg("the dub's separation recorded nothing"))?;
                let identical = dub_state.identical.unwrap_or(false);
                // The levelling, or — when either side could not be
                // measured — only what the run asked for, said out loud.
                let gain = match job.manifest.voice_gain_db() {
                    Some(gain) => gain,
                    None => {
                        progress(Event::message(format!(
                            "a loudness is missing, so the voices are not levelled against the \
                             original: only --gain-db ({:+.1} dB) applies",
                            job.manifest.gain_db
                        )));
                        job.manifest.gain_db
                    }
                };
                let placed = voices::voice_objects(identical, &dub_state.elements);
                let revoiced = master::revoice(
                    &job,
                    &placed,
                    identical,
                    dub_state.elements.len(),
                    gain,
                    job.manifest.cancel_duplicates,
                    progress,
                )?;
                job.manifest.master = Some(MasterState {
                    voices: revoiced.voices,
                    gain_db: revoiced.gain_db,
                    frames: revoiced.frames,
                    inputs: revoiced
                        .inputs
                        .iter()
                        .map(|p| p.strip_prefix(&job.dir).unwrap_or(p).to_path_buf())
                        .collect(),
                    clipped: revoiced.clipped,
                    duplicates: revoiced.duplicates,
                });
            }
            Step::Encode => {
                announce(step, progress);
                let state = job
                    .manifest
                    .master
                    .clone()
                    .ok_or_else(|| Error::msg("the master set recorded nothing"))?;
                let revoiced = Revoiced {
                    header: job
                        .master_dir()
                        .join(master::REVOICED)
                        .with_extension("atmos"),
                    inputs: state.inputs.iter().map(|p| job.dir.join(p)).collect(),
                    voices: state.voices,
                    frames: state.frames,
                    gain_db: state.gain_db,
                    clipped: state.clipped,
                    duplicates: state.duplicates,
                };
                let encoded = match job.manifest.encoder {
                    Encoder::Hz => {
                        let options = job.manifest.hz_opts.clone();
                        encode::harlettizer(config, &job, &revoiced, &options, progress)?
                    }
                    encoder @ (Encoder::Iamf | Encoder::IamfBed) => {
                        let options = job.manifest.encoder_options();
                        encode::iamf(config, &job, &revoiced, encoder, &options, progress)?
                    }
                    Encoder::Dee => {
                        let clusters = job
                            .manifest
                            .extract
                            .as_ref()
                            .map(|e| e.elements.len())
                            .unwrap_or(16);
                        encode::dee(config, &job, &revoiced, clusters, progress)?
                    }
                };
                progress(Event::message(format!(
                    "wrote {}: {}",
                    encoded.stream.display(),
                    encoded.summary
                )));
                job.manifest.encode = Some(EncodeState {
                    encoder: job.manifest.encoder,
                    stream: encoded
                        .stream
                        .strip_prefix(&job.dir)
                        .unwrap_or(&encoded.stream)
                        .to_path_buf(),
                    summary: encoded.summary,
                    depth: encoded.depth,
                });
            }
            Step::Remux => {
                announce(step, progress);
                let stream = job
                    .manifest
                    .encode
                    .as_ref()
                    .map(|state| job.dir.join(&state.stream))
                    .ok_or_else(|| Error::msg("the encode recorded no stream"))?;
                let tags = job.master_dir().join("tags.xml");
                crate::tags::write(&job.manifest, &catalogue, &tags)?;
                let output = job.output();
                let name = remux::track_name(job.manifest.encoder);
                if job.manifest.mkv {
                    remux::mkv(
                        config,
                        &media.path,
                        &job.manifest.original,
                        &job.manifest.dub,
                        &stream,
                        name,
                        &tags,
                        &output,
                        progress,
                    )?;
                } else {
                    remux::mka(
                        config,
                        &stream,
                        name,
                        &tags,
                        job.manifest.original_begin(),
                        &output,
                        progress,
                    )?;
                }
                progress(Event::message(format!("wrote {}", output.display())));
                job.manifest.output = Some(output);
            }
        }
        job.finished(step, started.elapsed().as_secs_f64())?;
        // Known once the master set is on disk, and hours of separation
        // before the encoder would have said it. Checked after the step is
        // recorded, so a run with another encoder carries on from here.
        if step == Step::Extract
            && job.manifest.encoder == Encoder::Hz
            && let Some(extracted) = &job.manifest.extract
        {
            encode::overlay_fits(extracted.elements.len())?;
        }
    }
    if !config.keep_work {
        // Tens of gigabytes a film, and nothing in them the track needs.
        job.clean()?;
        progress(Event::message(format!(
            "removed the job's temporaries from {} (keep_work: true keeps them)",
            job.dir.display()
        )));
    }
    Ok(outcome)
}

/// Make a separation's directory this run's before its first element.
///
/// Stems made for another model or another track are not this run's; they
/// go before anything trusts their length. The signature is recorded right
/// away rather than when the step ends, so a separation interrupted halfway
/// — a GPU reset, a closed laptop — keeps the elements it finished and the
/// next run only does the rest: recorded at the end, an unfinished step
/// looked like another run's, and hours of stems were thrown away.
fn claim(job: &mut Job, stage: separate::Stage, signature: &str) -> Result<()> {
    let dir = job.separate_dir();
    let state = match stage {
        separate::Stage::Original => &mut job.manifest.separate_original,
        separate::Stage::Dub => &mut job.manifest.separate_dub,
    };
    if state.as_ref().is_some_and(|s| s.signature == signature) {
        return Ok(());
    }
    separate::clear(&dir, stage)?;
    // Nothing measured yet: no element, no loudness (0 is "not measured").
    *state = Some(SeparateState {
        signature: signature.to_string(),
        elements: Vec::new(),
        lkfs: 0.0,
        identical: None,
    });
    job.save()
}

/// The earliest finished step whose files are no longer all there.
///
/// The manifest says how far a run got; the disk says whether what it got
/// is still there. A stem someone deleted, a directory a cleaner emptied,
/// a file a full disk left half-written: the step that made it runs again
/// rather than the next one reading a hole.
fn gone(job: &Job, progress: Progress) -> Option<Step> {
    let complain = |step: Step, what: &str| {
        progress(Event::message(format!(
            "{what} — step {}/{} ({}) runs again",
            step.number(),
            Step::ALL.len(),
            step.label()
        )));
        Some(step)
    };
    match &job.manifest.extract {
        Some(state)
            if !extract::complete(&job.extract_dir(), state.elements(), &state.dub_channels) =>
        {
            return complain(Step::Extract, "the extraction's files are not all there");
        }
        None if job.manifest.done > 0 => {
            return complain(Step::Extract, "the manifest records no extraction");
        }
        _ => {}
    }
    for (step, stage, state) in [
        (
            Step::SeparateOriginal,
            separate::Stage::Original,
            &job.manifest.separate_original,
        ),
        (
            Step::SeparateDub,
            separate::Stage::Dub,
            &job.manifest.separate_dub,
        ),
    ] {
        if let Some(state) = state
            && !separate::complete(job, stage, &state.elements)
        {
            return complain(step, "some of the separated stems are not there");
        }
    }
    if let Some(state) = &job.manifest.separate_original
        && !separate::at_depth(job, &state.elements)
    {
        return complain(
            Step::SeparateOriginal,
            "the original's elements carry bits below the depth it was mastered at",
        );
    }
    if let Some(state) = &job.manifest.master {
        let header = job
            .master_dir()
            .join(master::REVOICED)
            .with_extension("atmos");
        let whole = MasterSet::read(&header).is_ok()
            && state
                .inputs
                .iter()
                .all(|input| wav::holds(&job.dir.join(input), state.frames));
        if !whole {
            return complain(Step::Master, "the re-voiced master set is not all there");
        }
    }
    if let Some(state) = &job.manifest.encode
        && !std::fs::metadata(job.dir.join(&state.stream)).is_ok_and(|m| m.len() > 0)
    {
        return complain(Step::Encode, "the encoded stream is not there");
    }
    if let Some(output) = &job.manifest.output
        && !output.is_file()
    {
        return complain(
            Step::Remux,
            "the track that was written is not there any more",
        );
    }
    None
}

/// « 1/6 extraction », the line a step opens with.
fn announce(step: Step, progress: Progress) {
    progress(Event::Step {
        number: step.number(),
        total: Step::ALL.len() as u32,
        label: step.label().to_string(),
    });
}

/// The two tracks a run works on: the ones asked for, or the ones the
/// ranking picks.
pub fn choose<'a>(media: &'a Media, options: &Options) -> Result<(&'a AudioTrack, &'a AudioTrack)> {
    let original = match options.original {
        Some(index) => {
            let track = media.track(index)?;
            if !track.is_spatial() {
                return Err(Error::msg(format!(
                    "track {index} ({}, {}) carries no scene to re-voice{}",
                    track.codec.name(),
                    track.layout(),
                    match &track.probe_error {
                        Some(why) => format!(" — the decoder said: {why}"),
                        None => String::new(),
                    }
                )));
            }
            track
        }
        None => tracks::original(media)?,
    };
    let dub = match options.dub {
        Some(index) => media.track(index)?,
        None => tracks::dub(media)?,
    };
    if dub.index == original.index {
        return Err(Error::msg(format!(
            "track {} cannot be both the original and the dub",
            dub.index
        )));
    }
    if dub.is_spatial() {
        return Err(Error::msg(format!(
            "track {} carries a scene ({}); only its channels would be taken, and a dub is a channel track",
            dub.index,
            dub.spatial.as_ref().map(|s| s.label.as_str()).unwrap_or("")
        )));
    }
    Ok((original, dub))
}
