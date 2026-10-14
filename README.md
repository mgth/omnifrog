# omnifrog

omnifrog re-voices a spatial-audio film with the dialogue of its dubbed track, from a command
line or a window (`omnifrog-gui`, see [The window](#the-window)). One command, a film in, a track
out:

```
omnifrog dub film.mkv                 # picks the original and the dubbed track, writes film.Omnifrog-<vo>-<vf>-hz.mka
omnifrog dub film.mkv --vo 2 --vf 1 --mkv
```

What it does, in order: `ffmpeg | harletty` decodes the original (TrueHD Atmos, E-AC-3 JOC, DTS:X,
Auro-3D) into one mono WAV per element and splits the dubbed track into channels; a separation model
removes the dialogue from every element of the original and another extracts it from every channel
of the dub (by default Gabox's Mel-Band Fullness V3 for the original, Kim's Mel-Band FT2 Bleedless
for the dub); the music and effects the dubbed voices still carry — the original's, a few tens of
milliseconds late, heard as an echo — are cancelled and the voices put on the original's clock; they
are levelled to the original's dialogue (BS.1770), placed as static objects, and the scene is encoded as TrueHD Atmos by harlettizer (the original's elements kept as
they are, the voices panned onto them) or by Dee — or as IAMF by harlettizer, the voices panned
onto the objects the same way or put on the bed; mkvmerge writes the `.mka` or the full `.mkv`,
with the models and every parameter of the job as Matroska tags of the track.

## Status

`dub` runs the whole pipeline — reading a file's tracks, choosing the two,
decoding the original into a master set while the dub is split into channels, lifting the dialogue
out of both, levelling one against the other, writing the re-voiced master set, encoding it with
harlettizer or Dee, and writing the `.mka` (or the full `.mkv`) with the job in its tags. The window
does the same, see [The window](#the-window). What is left is Windows: built and tested under
Wine, not yet on a real Windows machine.

Checked on two films, one DTS:X 7.1.4+5 (8 bed and 9 objects) and one TrueHD Atmos (11 objects),
elements and dub channels alike. In the loudness, a bed's surround channels are weighted as
BS.1770 says rather than as front (see `crates/core/src/separate.rs`).

With harlettizer the voices are *overlaid*: the original's elements are kept as they are and the
voices panned onto them, which needs the original to fit a TrueHD stream's sixteen elements. A
DTS:X 7.1.4 with five objects is seventeen; for it the run stops right after the extraction and
says to use Dee, which re-clusters the whole scene. A harlettizer that has `--mono-prefix` reads
the master's waveforms where they are; one that has not is handed an interleaved CAF, some twenty
gigabytes a film, deleted once the stream is written.

`--encoder iamf` and `--encoder iamf-lit` write [IAMF](https://aomediacodec.github.io/iamf/) instead
of TrueHD — the open format browsers and televisions decode — with `harlettizer iamf`, straight into
an `.mka` of one `A_IAMF` track that the remux takes as it takes a TrueHD stream. Nothing in either
goes through TrueHD: harlettizer reads the re-voiced master set and codes it as IAMF v2.0 objects,
with no bound of sixteen elements. The two differ in what becomes of the voices, K of them being the
voices the master step appended:

- **`iamf`** — strictly the TrueHD overlay's scene: the same objects, the voices projected onto them
  the same way, under the same guards (`--overlay-cost`, `--overlay-drift`, …, with the same
  defaults), the elements a voice is mixed into rounded to the original's depth as `hz` rounds them
  (`--fold-depth`, unless `iamf_opts` gives it). A refused overlay fails the run as `hz` does, naming the option that would decline the
  guard as an `--iamf-opts` to pass, or `--encoder iamf-lit`. Written as `3.master/vf.mka`, the
  track named `IAMF Français`, the output `<film>.Omnifrog-<model codes>-iamf.mka`.

  ```
  harlettizer iamf <job>/3.master/vf.atmos --objects --overlay K --out <job>/3.master/vf.mka --progress [--mono-prefix <job>/3.master/wave] [--fold-depth BITS] <iamf_opts>
  ```

- **`iamf-lit`** — the voices on the fixed bed, the original's other objects independent. Written
  as `3.master/vf-bed.mka`, the track named `IAMF Français (voix sur le lit)`, the output
  `<film>.Omnifrog-<model codes>-iamf-lit.mka` — beside the other, not over it.

  ```
  harlettizer iamf <job>/3.master/vf.atmos --objects --voices-to-bed K --out <job>/3.master/vf-bed.mka --progress [--mono-prefix <job>/3.master/wave] <iamf_opts>
  ```

Both take `iamf_opts` (or `--iamf-opts "…"` on `dub`) as it is, after their own flags: `--codec
flac|lpcm|opus` (FLAC by default, lossless), `--bitrate <kbit/s a channel>` for Opus, `--bits 16|24`,
and for `iamf` the overlay's guards. Few decoders read IAMF v2.0 objects yet. Each needs a harlettizer
whose `iamf` writes Matroska and has its flag (`--overlay`, `--voices-to-bed`); one that has not is
refused before the run starts, and `doctor` says which modes the configured one has. The track's
`OMNIFROG_MANIFEST` records the mode as its `encoder`. Changing the encoder — from one IAMF mode to
the other too — or its options redoes the encode and the remux only; the options of an encoder not
in use change nothing.

## Commands

```
omnifrog doctor                       what is installed and what is missing
omnifrog setup [--gpu auto|cuda|rocm|mps|cpu] [--python PATH] [--force]
                                         builds the Python environment (torch + MSST) this program runs its inference in
omnifrog models list                  the catalogue, with what is on disk
omnifrog models use --vo KEY --vf KEY the two models a run uses
omnifrog models fetch [KEY…]          download (default: the two selected)
omnifrog models add FILE.yaml         add or override catalogue entries
omnifrog config                       the effective configuration and where it is
omnifrog probe FILE                   the audio tracks, and which two a run would take
omnifrog bench-vram ELEMENT.wav       device memory and time of the separation under several settings
omnifrog dub FILE [--vo N] [--vf N] [--mkv] [--encoder hz|dee|iamf|iamf-lit] [--seconds N]
                     [--gain-db X] [--hz-opts "…"] [--iamf-opts "…"] [--pairs LIST] [--no-cancel-duplicates]
                     [--out DIR] [--work DIR] [--restart] [--json]
                                         re-voice a film
```

`--vo` and `--vf` count the audio tracks from 1, as `probe` prints them. Left out, the original is
the first track carrying a scene and the dub is the best French one: never a side track (audio
description, commentary, hearing-impaired), the home dub before the Québec one, then more channels,
a better codec, a higher bit rate, and the container's order to break a tie.

The track is written beside the film unless `--out DIR` says otherwise, as
`<film>.Omnifrog-<model codes>-<hz|dee|iamf|iamf-lit>.mka`: one dotted segment after the film's name, so
everything before the last dot is still the film. It carries, as tags of the track,
`OMNIFROG_DATE`, `OMNIFROG_MODELS`, `OMNIFROG_MANIFEST` (the job's parameters and measurements,
without a path from the machine that made it) and `OMNIFROG_TOOL`. `--mkv` writes the whole film instead: the source's picture, subtitles, chapters and
attachments, the new track (default), the original, and the dub as it was. A source whose audio
starts after its picture — a disc remux can start it two seconds in — has the new track shifted by
the same amount, so it plays where the original does.

A run is a job: one directory per film under the working directory, `<film>.omnifrog`, holding

- `1.extract/` — the master set `vo.atmos` and its metadata, one `vo_<id>.wav` per waveform of the
  original, one `vf_<channel>.wav` per channel of the dub;
- `2.separate/` — `sons_<id>.wav` (an element of the original without its dialogue, rounded to the
  depth the original was mastered at: the separation fills all 24 bits, and a lossless coder would
  carry every one),
  `voixvo_<id>.wav` (the dialogue taken out of it, kept to be measured), `voix_<channel>.wav` (the
  dialogue of one of the dub's channels), and `voices.txt`, the report the placement was decided on;
- `3.master/` — the re-voiced master set `vf.atmos` and its metadata (the original's, with one
  static object per voice), `voice_<id>.wav` (each voice at its level, its duplicates cancelled), the stream `vf.thd`
  (`vf.mka` for IAMF, `vf-bed.mka` with the voices on the bed), and the encoder's log with the exact command at its head;
- `manifest.json`.

Once the track is written the job's directory is removed — it is tens of gigabytes a film — unless
`keep_work: true`, which keeps it for another encoder or another container without separating
again.

Running the same film again picks up where the last one stopped, and a parameter that changed sends
it back to the earliest step that parameter decides — another model does not re-extract the film,
another track does. Inside a step the resume is per element: an interruption costs the one element
that was in flight, never the ones already done. `--restart` starts over.

The models are stereo and the elements mono: by default each element is separated alone, written on
both sides. `--pairs` (or `pairs:` in the configuration) hands the model two at once, which halves
the passes and, where both carry the same voice, leaves less of it in the music and effects:

- `original` — the original's elements two by two, spatial neighbours round the room, starting at
  the most central one and its nearer neighbour (the dialogue is usually there; nothing is
  detected, the order alone decides); an odd one out is alone;
- `dub` — the dub's channels left with right (L+R, Lss+Rss, Lrs+Rrs), the centre alone;
- `centre` — `dub`, and the centre separated twice, beside L and beside R, the two averaged; L and R
  come out of those same passes, so it costs no pass;
- `all`, `none`.

Changing it sends the run back to the separation it changes. Measured: a whole film went from
4 h 38 to 1 h 38 with every switch on and the overlap below.

A model's `overlap` in the catalogue replaces its published config's `inference.num_overlap`: the
model sees the audio through overlapping windows and computes every sample that many times, so the
time is proportional to it. Both HyperACE v2 are published at 4 and run at 2, which halves them for
an output 26 to 42 dB below the signal of the one at 4. Kim FT2 Bleedless, the default dub model, is published
at 8 and runs at 2. Under pymss the windows overlap by its own share instead, and the catalogue's
`overlap` only counts at `overlap: 0` (MSST's step) or when the worker falls back to MSST.

**Duplicates.** A voice model leaves part of the dub's own music and effects in the voices it
extracts, and the dub's music and effects are the original's, played again a few tens of
milliseconds later: under the new mix a sharp noise sounds twice. One 2 h 43 film had 165 of them,
heard as an echo or a double attack. By default (`cancel_duplicates: true`,
`--no-cancel-duplicates` for one run, `--cancel-duplicates` to force it) the master step:

- puts the dub's voices on the original's clock — the two tracks are decoded from their own first
  sample, and in the container they need not start together (78 ms apart on that film);
- maps the dubbing's own offset along the film (the dub's mix against the original's, 20 s windows
  every 5 s, ±150 ms; it moves by plateaus, 42 to 90 ms on that film), with the leaks of the
  voices and the attacks that have a twin in the music and effects at their own offset;
- takes out of each voice what is coherent with the original's music and effects at that offset
  (STFT, a local complex gain per 3 frames × 5 bins, weighted by the coherence, never more than the
  voice itself); speech, which does not look like the music and effects, stays.

Neither half goes without the other: moved alone, the voices leave the dubbing's 6 to 36 ms, a comb
filter, which sounds metallic. The report — offset, plateaus, attacks, dB taken out per channel —
is in the progress and in the manifest. Switching it redoes the master step, the encode and the
remux, never a separation. See `crates/core/src/duplicates.rs`.

The separation takes the GPU, and only one run may have it: a lock is taken in the working
directory while a model is loaded, and a second run is told to wait rather than allowed to try.

Progress goes to standard error, a line per percent; `--json` makes it one JSON object per line,
which is what a front end reads.

## The window

`omnifrog-gui` (crate `crates/gui`, egui) does what the command line does, for someone who
would rather not type it:

- **Setup.** It names the graphics card torch will use and its memory (`nvidia-smi`, the AMD
  driver's sysfs on Linux, the machine's memory on Apple silicon), and the inference settings that
  suit it: the largest pymss block the memory holds (600 s is about 15 GB), a batch of one below
  12 GB, MSST below 4 GB. *Set up automatically* writes those settings (unless the configuration
  already has its own), builds the Python environment for that device (the same as `setup`) and
  downloads the two models. It lists what `doctor` lists, with a *Choose…* for every tool and
  where to get a missing one, the two models, and the work and model folders. A first launch on a
  machine that is not ready opens on this tab.
- **Film.** Choose or drop a file: its audio tracks, the original and the dub preselected as
  `dub` would take them (a scene only as the original, a channel track only as the dub); `.mka`
  or `.mkv`, a trial of the first N seconds, the format (TrueHD by harlettizer or by Dee, or IAMF
  — the voices panned onto the objects or on the bed — with its options; kept in the configuration,
  as `encoder` and `iamf_opts`), the voices' offset,
  the duplicates, where the track goes, start over. `omnifrog-gui FILM.mkv` opens with it.
- **Progress.** The six steps (those an earlier run did are marked as such), the current step's
  bar with an estimate of the time left, the whole run's bar weighted by what each step costs, the
  elapsed time and the log. *Stop* ends the run with everything it started; the next run of the
  same film picks up at the step it was in.

The long work — the setup, the downloads, a run — is done by a second copy of the program
(`omnifrog-gui --worker`, one JSON request on its standard input, one reply per line on its
standard output: the core's progress events, then how it ended). The window reads the
configuration, the checks, the device and a film's tracks from `omnifrog-core` directly. The
worker leads a process group on Unix and sits in a job object on Windows, so *Stop*, closing the
window or the window crashing ends pip, ffmpeg, harletty, the inference worker and the encoder
with it. On Linux the worker is started through `/proc/self/exe`, so a window outlives a
rebuild of its binary: the worker is the window's own build, never a newer one. It needs Rust
1.95 (egui 0.36); `rust-toolchain.toml` pins 1.97.1.

## The test bench

`omnifrog-bench` (crate `crates/bench`, egui) is a development tool, not part of a release:
where the separation models are compared. A *zone* is a few seconds of a film — one element of its
original, one channel of its dub — with what every model of the catalogue made of them; the
outputs rated best by ear are the reference, and every other output is measured against it. It
extracts a film whole and cuts zones out of it, runs the models, their cascades and the
duplicates-cancelled variants of their outputs through the core's own worker and cancellation,
plays any two outputs at the very same sample, and ranks the models. It also imports a legacy
test set, and measures it again to the last bit.

See [`crates/bench/README.md`](crates/bench/README.md).

## Configuration

`~/.config/omnifrog/config.yaml` on Linux, `%APPDATA%\omnifrog\config.yaml` on Windows
(`OMNIFROG_CONFIG` overrides; `OMNIFROG_DATA` moves the venv, MSST, models and work directories). Every key is optional:

```yaml
work_dir: /mnt/fast/work         # temporaries, 30-45 GB per film; ASCII only if Dee encodes
models_dir: ~/models             # checkpoints, configs, replacement modules
python: ~/venv/bin/python        # set by `setup`
msst_dir: ~/msst                 # set by `setup`
tools: { harletty: /opt/harletty, harlettizer: /opt/harlettizer, dee: C:\Dee\dee.exe, wine: /usr/bin/wine }
models: { vo: gfu3, vf: kf2b }
encoder: hz                      # hz | dee | iamf (voices panned onto the objects) | iamf-lit (voices on the bed)
hz_opts: "--presentations"
iamf_opts: ""                    # `harlettizer iamf`'s options, either mode; empty = FLAC, lossless
pairs: { original: false, dub: false, centre_twice: false }
cancel_duplicates: true          # the dub's voices on the original's clock, the original's M&E taken out of them
inference: { engine: pymss, overlap: 25, block_seconds: 300, batch_size: 0 }   # see below
keep_work: false
```

Models are listed in `models.yaml` (compiled in); a `models.yaml` beside the configuration adds
to it or overrides entries, key by key. Every model is an MSST checkpoint and config.

By default (`inference: { engine: pymss }`) those same files run through [pymss](https://github.com/pymss-project/pymss)
(MIT, installed by `setup`) instead of MSST's own inference: on five minutes of a film
(7900 XTX) HyperACE v2 Voc takes 10.7 s instead of 25.9 s at the default 25 % overlap, Big Beta 7
8.4 s instead of 22.0 s, and against a ground truth both score the same as under MSST within
0.04 dB. A model pymss cannot load (unwa's Large Inst), one with shifted passes, or a venv where
pymss is missing or does not import goes to MSST, and the progress says so. An element that runs
out of device memory under pymss goes through MSST alone, the log says which, and the next
elements are tried under pymss again. Changing the engine does not redo stems a job already has.

pymss installs into the venv without touching torch, provided the venv's torch is recent enough:
torch >= 2.7.1 is required, otherwise pip would replace it with the PyPI build (no ROCm; CPU only on
Windows). `setup` installs torch 2.10 first, so a venv it built is fine; in a venv of your own (the
configuration's `python`), check `python -c "import torch; print(torch.__version__)"` before
`pip install pymss==2.1.6`.

**A smaller card.** pymss keeps a whole block of an element on the device: `inference.block_seconds`
(300 by default) is what its memory grows with, and outside the fades between blocks the output does
not depend on it, to rounding. An element is cut into equal blocks of at most that plus one window
(the fade), the last one no longer than the others. `inference.batch_size` replaces the model's
published batch (windows computed at once; 0 = the catalogue's, else the author's: 1 for Fullness
V3, Kim FT2 and HyperACE v2 Voc — the catalogue's `batch: 1`, its author's is 2 — and 2 for Big
Beta 7). Each separation prints the most device memory torch reserved; the runtime's context
(about 0.5 GB) and the desktop come on top.

Measured on a 7900 XTX (ROCm), 25 minutes of an element of a 2 h 43 film, pymss at 25 %, with
equal blocks (8 October 2026); the difference is the stems' against the first line's, in dB under
them:

| model | setting | peak | time | difference (voice / M&E) |
|---|---|---|---|---|
| Fullness V3, Kim FT2 | blocks of 300 s (default) | 2.7 GiB | 26 s | — |
| | blocks of 600 s | 2.9 GiB | 25 s | −61 / −60 dB |
| | blocks of 120 s | 2.6 GiB | 27 s | −52 / −51 dB |
| | batches of 2 | 3.8 GiB | 25 s | −79 / −78 dB |
| | batches of 4 | 6.0 GiB | 25 s | −79 / −78 dB |
| | 5 % overlap | 2.6 GiB | 21 s | −32 / −31 dB |
| | `engine: msst` | 2.6 GiB | 64 s | −32 / −31 dB |
| HyperACE v2 Voc | batches of 1 (the catalogue's) | 2.3 GiB | 63 s | — |
| | batches of 2 (its author's) | 3.8 GiB | 85 s | −75 / −74 dB |
| | blocks of 120 s | 2.2 GiB | 68 s | −53 / −52 dB |
| | batches of 4 | 6.4 GiB | 121 s | −74 / −73 dB |
| | `engine: msst` | 2.2 GiB | 121 s | −30 / −29 dB |

Since blocks are equal the block hardly matters on this card: from 600 s to 300 s it takes 0.2 GiB
off, to 120 s another 0.1, and below that nothing more comes off (the model and one window's work
stay). The earlier measurement, when the last block took the remainder, put blocks of 600 s at
5.0 GiB: on these 25 minutes that last block was about 1030 s long. The batch is what costs: 1.1 GiB
more at 2, 3.3 more at 4, for no time on this card, and HyperACE is faster at 1 than at its
published 2. A lower overlap moves the output far more for no memory. Hence the defaults: blocks of
300 s, and `batch: 1` for HyperACE v2 Voc in the catalogue. To choose on another card:

```
omnifrog bench-vram <job>/1.extract/vo_12.wav [--models gfu3,kf2b] [--blocks 600,120,60] [--batches 1,2,4]
```

separates 25 minutes of that element under each setting and prints the peak, the time and how far
the stems moved from the default's.

## Requirements

ffmpeg, ffprobe, mkvmerge, [harletty](https://github.com/harletty/harletty-bridge), and
harlettizer (one whose `iamf` writes `.mka` and has `--overlay` / `--voices-to-bed` for `encoder:
iamf` / `iamf-lit`) or Dee; Python 3.11/3.12 for `setup`. A GPU (CUDA or ROCm) for the separation, which is
hours on a CPU.

## Linux

**Arch.** The AUR has `omnifrog` (the command line, the window and its menu entry), `harlettizer`
(the encoder) and `harletty-bridge` (whose `harletty` is the decoder):

```
paru -S omnifrog harlettizer harletty-bridge
```

**Any other distribution.** `omnifrog-<version>-x86_64.AppImage`, from a release, is the command
line and the window in one file, for x86-64 with glibc 2.35 or newer. Run as it is, it opens the
window; `--cli` as its first argument gives the command line:

```
chmod +x omnifrog-0.1.0-x86_64.AppImage
./omnifrog-0.1.0-x86_64.AppImage                 the window
./omnifrog-0.1.0-x86_64.AppImage --cli doctor    what is installed and what is missing
```

Neither ships the tools a run drives, nor Python: see [Requirements](#requirements).

## Windows

`omnifrog.exe` is one file: it needs nothing beside it but the system's own libraries.

**The installer.** `omnifrog-<version>-windows-x86_64-setup.exe` puts `omnifrog.exe` and
`omnifrog-gui.exe` under `%LOCALAPPDATA%\Programs\Omnifrog`, with an empty `tools\` folder beside
them and a Start-menu shortcut to the window. It is per user and asks for no administrator; `/S`
runs it without a wizard and `/D=<dir>` (last on the line, no quotes) chooses the folder. It
installs none of what the steps below ask for, and its uninstaller leaves the configuration, what
`setup` installed and whatever was put in `tools\`. It is not signed: Windows asks before running
it. `.github/workflows/packages.yml` builds and tests it with the Linux AppImage (see
[Linux](#linux)) — on a pull request that touches `packaging/`, on a run started by hand, and on a
`v*` tag, which also attaches both to a draft release.

1. **Python 3.12** from python.org, with the `py` launcher it installs — the Microsoft Store's
   `python` does not do.
2. **An NVIDIA driver recent enough for CUDA 12.8** (the 570 series or later). `setup` finds the card
   through `nvidia-smi`; `--gpu cuda` says so explicitly.
3. **The tools**, as Windows executables: ffmpeg and ffprobe (any recent build), mkvmerge (MKVToolNix),
   `harletty.exe` (the `harletty-cli-…-windows-x86_64.zip` of a harletty-bridge release) and
   `harlettizer.exe` (a harlettizer release), or `dee.exe` for `encoder: dee`, which runs directly —
   there is no Wine on Windows. Each is looked for in the configuration, then in a `tools\` folder
   beside `omnifrog.exe`, then on `PATH`.
4. Then:

   ```
   omnifrog setup --gpu cuda     a venv with torch 2.10 for CUDA 12.8 and MSST; a few gigabytes
   omnifrog doctor               everything green, and the device torch sees
   omnifrog dub film.mkv
   ```

`omnifrog-gui.exe` is the same with a window: after steps 1 to 3, its *Set up automatically*
is step 4, and the tools can sit in the same `tools\` folder beside it.

The configuration is `%APPDATA%\omnifrog\config.yaml`; the venv, MSST, the models and the jobs are
under `%LOCALAPPDATA%\omnifrog\` unless it says otherwise. Keep `work_dir` on a fast disk with
40 GB free per film, and on a path without accents if Dee encodes.

Python on Windows cannot open a path longer than 260 characters unless long paths were enabled in the
registry, and the separation works about sixty characters below a job's directory. So on Windows the
film's name in that directory is cut to 48 characters, with a hash of the rest; the track itself
keeps the film's full name.

## Licence

GPL-3.0-or-later, see [LICENSE](LICENSE).
