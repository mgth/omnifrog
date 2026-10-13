# omnifrog-bench

The model test bench of omnifrog: where the separation models are compared. A development tool —
it is built with the workspace and checked by its CI, and is not part of a release.

Nobody has the isolated stems of a film mix, so there is no ground truth. The bench works on
*zones*: a few seconds of a film, one element of its original track and one channel of its dub,
with what every model made of them. The outputs rated best by ear are the reference of their side
(several of them: their median, sample by sample), and every other output is measured against it.
All of it is relative — close to the reference, not perfect — and that is enough to rank models
and to see a regression.

It is built on `omnifrog-core`: the worker, the loudness, the extraction and the duplicate
cancellation are the ones a `dub` run uses, so what is measured here is what a run does.

```
omnifrog-bench                         the window
omnifrog-bench --dir DIR …             another bench directory than <work_dir>/_bench
```

## From a film to a ranking

1. **A film in the bench.** `extract FILM.mkv [--vo N] [--vf N]`, or *Extract a film…* in the Film
   tab: the film is extracted whole, once — every element of the original, every channel of the
   dub, the tracks a run would take unless others are named. Tens of gigabytes and several minutes;
   kept until it is removed (`films --remove NAME`, or *Remove*). An extraction cannot be stopped
   from the window. Extracting a film again — with another track for its dub, say — forgets the
   ranges tried on the last extraction.
2. **A range of it.** An element, a channel, two times: *Open* cuts that range out of the
   extraction without reading the rest. It can be listened to at once; *Separate all* runs every
   checked model on it. A few tens of seconds around the passage is the right size: the models
   hear all of it.
3. **A zone.** *Add zone* makes a zone of the loop set on the top waveform (or of the whole
   range): its two inputs and every output are cut to it and their loudness measured again. A zone
   of a few seconds made this way has the outputs of models that heard the half-minute around it.
   `add-zone --film NAME --object 12 --channel C --from 12:02 --to 12:15 --models KEYS --context 15`
   does the three steps without the window.
4. **Rating.** In the Rate tab, ★ makes an output a reference of its side and 🗙 rejects it, with a
   reason: a *leak* (some of the other signal is left), a *lack* (the wanted signal is eaten into),
   *artifacts*. Either is kept at once. When the references of a side change, their consensus is
   written again and what was measured on that side of the zone is dropped; the other side's
   measures stand, and so does the time each output took.
5. **Measuring.** *Compute* in the Report tab, or `separate`, runs what is missing — models,
   cascades, variants — then measures; *Metrics only*, or `measure`, measures alone.
6. **Ranking.** The Report tab, or `report [--side vo|vf] [--sort gap] [--zone ID]`; then one
   model zone by zone, in the Model tab or with `report --model KEY`.

A model alone on a zone shorter than its window (a few seconds) does not do what it does on the
same seconds inside a film: on a zone of two seconds, the dub's voice came out some 40 dB under
what the same model gave with the film around it, and mirroring the input up to the window did
not bring it back. That is why zones are cut from a range rather than separated on their own.

## The window

The zones are on the left. Nothing above a list one clicks down changes height: the controls of a
tab are always there, enabled or not, and the status line at the bottom has its place.

**Rate.** One row per output of the zone chosen: its name and, under it, the SI-SDR and the gap of
the stem each side is judged on; ★ and 🗙 for each side; its five files as buttons carrying their
loudness; its waveform. A click on a button loads that file *at the position and in the state the
player is in*, so two outputs are compared at the very same sample; only ▶ or Space starts and
stops. On the top waveform — the input of the side playing — a click goes there, a drag loops over
that range exactly, a double click lets go of the loop, and the wheel sets the height of every
waveform. Ctrl+click a rest and a voice, of two models if you like, and they play together, the
voice at the level of the voice the rest's own model took out. The rows' waveforms are tinted by
their gap to the reference: red where the output is louder (a leak), yellow where it is quieter
(a lack). *Remove zone*, in two clicks, removes the zone and its files for good.

**Report.** The models summed up over all the zones or the one chosen, one side at a time, sorted
by any column; a checkbox per row says whether the next batch takes it. A line says when a side of
a zone has a single reference — the measure there favours that model's family rather than what the
ear prefers. *Compute*, *Metrics only*, *Stop* (after the zone in hand), and which variants are
computed. A click on a model's name shows it under Model.

**Model.** The report the other way round. The left column lists the models in place of the
zones; one is chosen there, and each zone of the set has its line — where the model's mean comes
from, and where it fails. The same columns and the same sort, the zones' own order to begin with;
the first line is the model's mean over all of them, its line in the report. A zone the model was
not run on has its line too, empty. A click on a zone chooses it for the other tabs: Rate then
plays it.

**Film.** The films of the bench and what they take, the range, *Separate all*, *Add zone*; under
them the same listening bench on the range opened, without the ratings.

**Cascades.** The cascades of the set, and a form to add one.

The player is in the process (cpal). The device is opened the first time something is played and
not before, at 48 kHz when it can. A device that failed to open is asked for again at the next ▶.

**Output**, at the right of the top bar, chooses where the sound goes: *System default*, or one
of the machine's outputs, asked of the system each time the list is opened. A stream that is
playing moves to the output chosen at once. The choice is kept from one run to the next, in
`bench-window.json` beside `config.yaml` — per machine, not in the bench. An output that was chosen
and is no longer there is said so at ▶ and nothing plays: the bench does not go and play somewhere
else on its own.

On Linux the outputs are PipeWire's, under the names the desktop gives them, when it runs; ALSA's
otherwise. With ALSA, whose default may name a card that is not there, the devices `pipewire` then
`pulse` are tried when the default does not open.

## What is measured

Of each stem, against the reference's same stem (the *target*) and its other stem:

| | |
|---|---|
| lag | where the output sits against the reference (±100 ms), undone before anything else |
| SI-SDR | signal to error, blind to gain; dB, higher is closer |
| residue | the reference's other stem found in the output — voice in the rest, rest in the voice; dB, lower is better |
| loss | the part of the target that is missing; dB, lower is better |
| artifacts | what is neither stem of the reference; dB, lower is better |
| gap | how far the spectrum is from the reference's over 24 bands, frame by frame; dB, lower is closer |
| excess, lack | the gap's two directions: where the output is louder (a leak), where it is quieter (a lack) |

Residue, loss and artifacts come from projecting the output, by frames of 200 ms, onto the
reference's two stems. A side is judged on the stem a run keeps of it: what is left of the
element on the original's side, the voice of the channel on the dub's. An output that is the only
reference of its side is not measured there; one of several is measured against their consensus.
A reject is measured like any other.

`measure --dry-run` measures again, compares with the stored measures and reference files, and
writes nothing: the test of a change to any of this.

## Variants and cascades

An output's *key* is its model's catalogue key. Two kinds of keys are built on others.

**Variants**, on zones that have a *duplicates context* (the map of the dub around the zone, how
far the dub's track leads the original's, the sum of the other elements of the original — measured
when a zone is cut from a film). From a model's outputs, what a run would make of them:

- `<key>+cancel` — the original's music and effects cancelled out of the dub's voice, by the
  core's own cancellation; in the mix, the voice moved by the offset of the two tracks;
- `<key>+cancel-step` — the same, the voice moved step by step along the map, with a 20 ms fade
  between two steps.

A variant shares the model's files on the original's side and its mix gain — the very gain the
model's mix was made with, kept in its `loudness.json` — so from one row to the next only the
cancellation and the place of the voice change. The other elements are taken as
decoded — the bench has no separation of them, as a run has — so a little more is taken out than a
run would take.

**Cascades**, `<first>~<second>~rest` or `…~voice`: the first model separates, the second works
one of its stems again, and what it finds changes side. On the rest, against a leak: the voice
still in it joins the voice. On the voice, against a lack: the rest taken with it goes back to the
rest. Voice and rest still sum to what the first model made (to a 24-bit step, short of a sample
pushed past full scale). A cascade is made from the outputs its first model already has on a zone,
so only the second model runs — loaded once for every cascade that ends with it.
`cascade --add FIRST SECOND [--mode voice]`, `cascade --remove KEY`, `cascade`.

## On disk

```
<work_dir>/_bench/
  bench.json                          zones, ratings and reasons, measures, checks, cascades
  <zone>/
    1.extract/vo_<object>.wav  vf_<channel>.wav  [others_vo.wav]
    outputs/<key>/voice_vo.wav rest_vo.wav voice_vf.wav rest_vf.wav mix.wav  loudness.json
    ref_vo.voice.wav ref_vo.rest.wav ref_vf.voice.wav ref_vf.rest.wav
  films/<film>/
    film.json                         the source, the two tracks, the elements and channels
    1.extract/                        the whole film, as a run's first step leaves it
    trials/o<object>_c<channel>_<start>_<end>/     a range, laid out as a zone
```

Every audio file is mono, 48 kHz, 24 bits (`others_vo.wav` is float: a sum of elements goes past
±1). A zone is a job in small. An output is made under its key with `.part` after it and takes
the key once every file is in it: a directory named by a key is a whole output, and what a run
that was killed left is not one. `loudness.json` holds the loudness of the five files and
`mix_gain_db`, the gain the dub's voice has in the mix. `list [--zone ID]` shows the zones or the outputs of one,
`remove-zone ID` removes one.

## A legacy test set

`import <_JeuxEssai>` copies the zones of a legacy test set into this layout and translates the set:
its model keys become the catalogue's (a model's key here is that set's short code for it),
`+annul` and `+annulp` become `+cancel` and `+cancel-step`, ratings, reasons, measures and
duplicate contexts come as they are. Outputs of a model the catalogue does not have keep their own
key: listened to and measured like the others, never computed again. A zone the bench already has
is left alone. Two things are left behind on purpose: the log-spectral distance, which the gap by
bands replaced, and the fitted score, which was never adopted (its numbers are kept).

The arithmetic of the measures follows the legacy set's step for step: `measure --dry-run` on an
imported set finds every stored value again, to the last bit, and the consensus files identical.

## Building

`cargo build --release -p omnifrog-bench` gives `target/release/omnifrog-bench`. It needs Rust
1.95 (egui 0.36) and, on Linux, ALSA's and PipeWire's headers for cpal (`libasound2-dev
libpipewire-0.3-dev libclang-dev`; `alsa-lib pipewire clang`). A machine without PipeWire's library
— one the subcommands are run on over SSH — builds it with `--no-default-features`: the window
then lists ALSA's devices. The
library half of the crate has no UI in it — the set, its files, the measures, the batches; the
window and the subcommands call the same functions, and the tests draw the window without a
screen and never open a device, but for one left out of a plain run that opens this machine's
output: `cargo test -p omnifrog-bench -- --ignored`.
