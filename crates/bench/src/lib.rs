//! UI-agnostic core of the omnifrog test bench.
//!
//! A *test set* is a list of zones: a few seconds of a film, one element of
//! its original track and one channel of its dub, with what every model of
//! the catalogue made of them. Nobody has the isolated stems of a film mix,
//! so there is no ground truth: the outputs rated best by ear are the
//! reference, and every other one is measured against them.
//!
//! Everything here runs without a window. The set and its files are in
//! [`set`] and [`layout`], the names an output goes by in [`keys`], what a
//! zone has to show in [`listing`], and [`import`] brings in a legacy
//! test set. [`consensus`] makes the reference of a side, [`metrics`]
//! compares an output with it, [`measure`] does so over zones and keys, and
//! [`report`] sums a model up. [`rate`] records what the ear said, [`player`]
//! is what plays an output and [`wave`] what a waveform shows of it.
//! [`outputs`] makes the five files of a model on a zone, and [`batch`] runs
//! the models over the zones that lack them. [`film`] keeps a film's
//! extraction, and cuts the ranges tried on it and the zones made of them.
//! [`variants`] computes, from a model's outputs, what a run would make of
//! them once the duplicates are cancelled.

pub mod audio;
pub mod batch;
pub mod consensus;
pub mod film;
pub mod import;
pub mod keys;
pub mod layout;
pub mod listing;
pub mod measure;
pub mod metrics;
pub mod outputs;
pub mod player;
pub mod rate;
pub mod report;
pub mod set;
pub mod variants;
pub mod wave;

pub use omnifrog_core::{Error, Result};
