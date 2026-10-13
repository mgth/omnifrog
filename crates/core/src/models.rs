//! The separation-model catalogue.
//!
//! One YAML compiled into the binary lists the models, with where each file
//! comes from; a second YAML beside the configuration adds to it or
//! overrides an entry, key by key. Every entry runs through the same
//! engine — MSST — because every checkpoint taken from the
//! audio-separator catalogue loads into MSST strictly with the very config
//! audio-separator fetched for it (measured on 23 of them, 2026-09-21); the
//! one exception is an ONNX model that ran on the CPU anyway.
//!
//! A model is downloaded the first time it is needed, or by `models fetch`;
//! a checkpoint that needs converting (Bandit's Lightning checkpoints) is
//! converted by the venv's Python right after.

use crate::config::Config;
use crate::progress::{Event, Progress};
use crate::{Error, Result, download, run};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// The catalogue shipped with the program.
pub const EMBEDDED: &str = include_str!("../../../models.yaml");

/// What a model is for. A "voices" model works for the original too — the
/// worker returns both stems, and what it did not call voice is the rest —
/// so the role is a hint for the listing, never a restriction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Trained to keep the instrumental: removes the dialogue from an element.
    Objects,
    /// Trained to keep the vocals: extracts the dialogue from a channel.
    Voices,
    /// Sold as both.
    Both,
}

impl Role {
    pub fn label(self) -> &'static str {
        match self {
            Role::Objects => "objects",
            Role::Voices => "voices",
            Role::Both => "both",
        }
    }
}

/// One file of a model: where it comes from, what it is called locally.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Source {
    pub url: String,
    pub file: String,
    /// What to do with the download before it is the file the worker loads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub convert: Option<Convert>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Convert {
    /// A PyTorch Lightning checkpoint whose keys carry a `model.` prefix that
    /// MSST's reimplementation does not: keep the state dict, drop the prefix,
    /// drop what the model has no key for (Bandit v2 from Zenodo).
    StripLightning,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    /// Stable identifier: the configuration and the command line name it.
    pub key: String,
    pub name: String,
    /// Short code in the output file's name, 2 to 5 characters, unique.
    pub code: String,
    pub role: Role,
    /// MSST's `model_type`: `bs_roformer`, `mel_band_roformer`, `bandit_v2`, `mdx23c`…
    pub model_type: String,
    pub checkpoint: Source,
    pub config: Source,
    /// A `bs_roformer.py` shipped with the checkpoint that replaces MSST's
    /// class for this model only (unwa Large-Inst, HyperACE).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module: Option<Source>,
    /// Sample rate the worker is told the input is at: 44100 = the samples
    /// stay 48 kHz and only the header says 44.1 (no resampling, the
    /// Roformers' native rate); 48000 = a true header (Bandit).
    #[serde(default = "rate_44100")]
    pub rate: u32,
    /// MSST: shifted passes averaged (1 = none).
    #[serde(default = "one")]
    pub bigshifts: u32,
    /// MSST: test-time augmentation (three times the work).
    #[serde(default)]
    pub tta: bool,
    /// MSST: replaces the published config's `inference.num_overlap`; 0 =
    /// the author's. The model sees the audio through overlapping windows,
    /// every sample computed this many times and averaged, so the time is
    /// proportional to it. Both HyperACE v2 are published at 4 and run at 2:
    /// half the time, for an output 26 to 42 dB below the signal of the
    /// one at 4 on five minutes of a real film.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub overlap: u32,
    /// Replaces the published config's `inference.batch_size` (windows
    /// computed at once); 0 = the author's. The configuration's
    /// `inference.batch_size` wins over it. HyperACE v2 Voc is published at
    /// 2 and runs faster at 1 (70 s against 92 on 25 minutes, 7900 XTX).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub batch: u32,
    /// Kept in the catalogue (for the code of a track already made) but not
    /// listed.
    #[serde(default)]
    pub hidden: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub notes: String,
}

fn rate_44100() -> u32 {
    44100
}
fn one() -> u32 {
    1
}
fn is_zero(value: &u32) -> bool {
    *value == 0
}

/// The three local files of a model.
#[derive(Debug, Clone, PartialEq)]
pub struct Local {
    pub checkpoint: PathBuf,
    pub config: PathBuf,
    pub module: Option<PathBuf>,
}

impl Local {
    pub fn all_present(&self) -> bool {
        self.checkpoint.is_file()
            && self.config.is_file()
            && self.module.as_ref().is_none_or(|m| m.is_file())
    }

    pub fn missing(&self) -> Vec<&Path> {
        let mut v = Vec::new();
        if !self.checkpoint.is_file() {
            v.push(self.checkpoint.as_path());
        }
        if !self.config.is_file() {
            v.push(self.config.as_path());
        }
        if let Some(m) = &self.module
            && !m.is_file()
        {
            v.push(m.as_path());
        }
        v
    }
}

impl Model {
    /// Where the model's files are, or would land: the models directory,
    /// except that a config already in MSST's own `configs/` is used from
    /// there rather than fetched again.
    pub fn local(&self, config: &Config) -> Local {
        let dir = config.models_dir();
        let in_models = dir.join(&self.config.file);
        let in_msst = config.msst_dir().join("configs").join(&self.config.file);
        Local {
            checkpoint: dir.join(&self.checkpoint.file),
            config: if !in_models.is_file() && in_msst.is_file() {
                in_msst
            } else {
                in_models
            },
            module: self.module.as_ref().map(|m| dir.join(&m.file)),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct File {
    #[serde(default)]
    models: Vec<Model>,
}

#[derive(Debug, Clone)]
pub struct Catalogue {
    pub models: Vec<Model>,
}

impl Catalogue {
    /// The embedded catalogue alone.
    pub fn embedded() -> Result<Self> {
        let file: File = serde_yaml_ng::from_str(EMBEDDED).map_err(|e| Error::Yaml {
            path: PathBuf::from("models.yaml (embedded)"),
            source: e,
        })?;
        Ok(Self {
            models: file.models,
        })
    }

    /// The embedded catalogue with the user's file merged over it: an entry
    /// with a known key replaces the shipped one whole, an unknown key is
    /// appended.
    pub fn load() -> Result<Self> {
        let mut catalogue = Self::embedded()?;
        let path = Config::user_catalogue_path();
        if path.is_file() {
            let text = fs::read_to_string(&path).map_err(|e| Error::io(&path, e))?;
            let user: File = serde_yaml_ng::from_str(&text).map_err(|e| Error::Yaml {
                path: path.clone(),
                source: e,
            })?;
            for model in user.models {
                catalogue.merge(model);
            }
        }
        catalogue.check()?;
        Ok(catalogue)
    }

    fn merge(&mut self, model: Model) {
        match self.models.iter_mut().find(|m| m.key == model.key) {
            Some(existing) => *existing = model,
            None => self.models.push(model),
        }
    }

    /// Unique keys and codes, keys the command line can type.
    fn check(&self) -> Result<()> {
        for (i, m) in self.models.iter().enumerate() {
            if m.key.is_empty()
                || !m
                    .key
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            {
                return Err(Error::msg(format!(
                    "model key `{}` must be ASCII letters, digits, `_` or `-`",
                    m.key
                )));
            }
            if !(2..=5).contains(&m.code.len())
                || !m.code.chars().all(|c| c.is_ascii_alphanumeric())
            {
                return Err(Error::msg(format!(
                    "model `{}`: code `{}` must be 2 to 5 ASCII letters or digits",
                    m.key, m.code
                )));
            }
            for other in &self.models[..i] {
                if other.key == m.key {
                    return Err(Error::msg(format!("model key `{}` listed twice", m.key)));
                }
                if other.code == m.code {
                    return Err(Error::msg(format!(
                        "models `{}` and `{}` share the code `{}`",
                        other.key, m.key, m.code
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn get(&self, key: &str) -> Result<&Model> {
        self.models
            .iter()
            .find(|m| m.key == key)
            .ok_or_else(|| Error::UnknownModel {
                key: key.to_string(),
            })
    }

    /// The models a listing shows.
    pub fn visible(&self) -> impl Iterator<Item = &Model> {
        self.models.iter().filter(|m| !m.hidden)
    }

    /// Add or replace entries in the user's catalogue file from another
    /// YAML of the same shape; returns the keys written.
    pub fn add_from(path: &Path) -> Result<Vec<String>> {
        let text = fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
        let added: File = serde_yaml_ng::from_str(&text).map_err(|e| Error::Yaml {
            path: path.to_path_buf(),
            source: e,
        })?;
        let user_path = Config::user_catalogue_path();
        let mut user: File = if user_path.is_file() {
            let t = fs::read_to_string(&user_path).map_err(|e| Error::io(&user_path, e))?;
            serde_yaml_ng::from_str(&t).map_err(|e| Error::Yaml {
                path: user_path.clone(),
                source: e,
            })?
        } else {
            File::default()
        };
        let mut keys = Vec::new();
        for model in added.models {
            keys.push(model.key.clone());
            match user.models.iter_mut().find(|m| m.key == model.key) {
                Some(existing) => *existing = model,
                None => user.models.push(model),
            }
        }
        // The merged result has to be a valid catalogue before it is written.
        let mut merged = Self::embedded()?;
        for m in &user.models {
            merged.merge(m.clone());
        }
        merged.check()?;
        if let Some(parent) = user_path.parent() {
            fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let text = serde_yaml_ng::to_string(&user).map_err(|e| Error::Yaml {
            path: user_path.clone(),
            source: e,
        })?;
        fs::write(&user_path, text).map_err(|e| Error::io(&user_path, e))?;
        Ok(keys)
    }
}

/// Download what is missing of `model` into `models_dir`, converting a
/// checkpoint that needs it. `python` and `msst_dir` are only touched for a
/// conversion.
pub fn fetch(model: &Model, config: &Config, progress: Progress) -> Result<Local> {
    let dir = config.models_dir();
    fs::create_dir_all(&dir).map_err(|e| Error::io(&dir, e))?;
    let local = model.local(config);
    // The config first: a conversion needs it, and it is small.
    if !local.config.is_file() {
        progress(Event::message(format!(
            "{}: fetching {}",
            model.key, model.config.file
        )));
        download::fetch(&model.config.url, &local.config, progress)?;
    }
    if let (Some(source), Some(path)) = (&model.module, &local.module)
        && !path.is_file()
    {
        progress(Event::message(format!(
            "{}: fetching {}",
            model.key, source.file
        )));
        download::fetch(&source.url, path, progress)?;
    }
    if !local.checkpoint.is_file() {
        match model.checkpoint.convert {
            None => {
                progress(Event::message(format!(
                    "{}: fetching {}",
                    model.key, model.checkpoint.file
                )));
                download::fetch(&model.checkpoint.url, &local.checkpoint, progress)?;
            }
            Some(Convert::StripLightning) => {
                let raw = dir.join(format!("{}.lightning", model.checkpoint.file));
                if !raw.is_file() {
                    progress(Event::message(format!(
                        "{}: fetching {} (Lightning checkpoint, to convert)",
                        model.key, model.checkpoint.file
                    )));
                    download::fetch(&model.checkpoint.url, &raw, progress)?;
                }
                progress(Event::message(format!(
                    "{}: converting {}",
                    model.key, model.checkpoint.file
                )));
                strip_lightning(model, &raw, &local, config)?;
                let _ = fs::remove_file(&raw);
            }
        }
    }
    Ok(local)
}

/// Lightning checkpoint → the plain state dict MSST loads, done by the
/// venv's Python because it is the only thing that can read the file.
fn strip_lightning(model: &Model, raw: &Path, local: &Local, config: &Config) -> Result<()> {
    let python = config.python();
    if !python.is_file() {
        return Err(Error::msg(format!(
            "converting {} needs the Python environment: run `setup` first",
            model.checkpoint.file
        )));
    }
    let script = r#"
import sys, torch
msst, model_type, config, raw, out = sys.argv[1:6]
sys.path.insert(0, msst)
from utils.settings import get_model_from_config
model, _ = get_model_from_config(model_type, config)
ck = torch.load(raw, map_location="cpu", weights_only=False)
sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
keys = set(model.state_dict().keys())
new = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.") and k[len("model."):] in keys}
missing = keys - set(new)
if missing:
    raise SystemExit(f"{len(missing)} keys of the model are not in the checkpoint, e.g. {sorted(missing)[:3]}")
model.load_state_dict(new, strict=True)
torch.save(new, out)
print(f"OK {len(new)} tensors")
"#;
    let mut command = run::command(&python);
    command
        .arg("-c")
        .arg(script)
        .arg(config.msst_dir())
        .arg(&model.model_type)
        .arg(&local.config)
        .arg(raw)
        .arg(&local.checkpoint)
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("HIP_VISIBLE_DEVICES", "")
        .env("ROCR_VISIBLE_DEVICES", "");
    run::output(command).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_catalogue_is_valid_and_names_the_defaults() {
        let catalogue = Catalogue::embedded().unwrap();
        catalogue.check().unwrap();
        let defaults = crate::config::Selection::default();
        assert!(catalogue.get(&defaults.vo).is_ok());
        assert!(catalogue.get(&defaults.vf).is_ok());
        assert!(catalogue.models.len() >= 30);
    }

    #[test]
    fn a_duplicate_code_is_refused() {
        let mut catalogue = Catalogue::embedded().unwrap();
        let mut copy = catalogue.models[0].clone();
        copy.key = "another".into();
        catalogue.models.push(copy);
        assert!(catalogue.check().is_err());
    }
}
