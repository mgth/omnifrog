//! The Python environment the inference runs in, built by this program.
//!
//! One venv with torch for the machine's GPU and MSST's dependencies, plus
//! a checkout of MSST at a pinned commit. Every step is skipped when its
//! result is already there, so the command can be run again after a cut
//! download or on a machine that already has half of it.

use crate::config::{Config, Engine, Inference};
use crate::progress::{Event, Progress};
use crate::{Error, Result, download, run, tools};
use std::fs;
use std::path::{Path, PathBuf};

/// The torch the working environments were built with.
pub const TORCH_VERSION: &str = "2.10.0";

/// MSST (ZFTurbo/Music-Source-Separation-Training) at the commit the worker
/// was written against.
pub const MSST_COMMIT: &str = "ccc011abf7f89dd7922bb2888d48493b575c0289";

/// What MSST imports for the models the catalogue lists (bs/mel-band
/// Roformer, Bandit v2, MDX23C), pinned to the versions of a venv that works.
pub const REQUIREMENTS: &[&str] = &[
    "numpy==2.4.6",
    "librosa==0.11.0",
    "omegaconf==2.3.1",
    "ml_collections==1.1.0",
    "soundfile==0.14.0",
    "tqdm==4.68.3",
    "pandas==3.0.3",
    "matplotlib==3.11.0",
    "requests==2.34.2",
    "PyYAML==6.0.3",
    "pytorch-lightning==2.6.5",
    "beartype==0.22.9",
    "einops==0.8.2",
    "rotary-embedding-torch==0.9.1",
    "soxr==1.1.0",
    // The other engine, on the same checkpoints (`inference: { engine: pymss }`);
    // brings pymss-core and av, and leaves torch alone as long as it is
    // >= 2.7.1, which TORCH_VERSION is: an older torch would be replaced with
    // the PyPI build (no ROCm; CPU only on Windows).
    "pymss==2.1.6",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gpu {
    Auto,
    Cuda,
    Rocm,
    /// Apple silicon's Metal, which PyPI's own macOS torch drives.
    Mps,
    Cpu,
}

impl Gpu {
    /// Which torch index serves this GPU.
    pub fn torch_index(self) -> &'static str {
        match self {
            Gpu::Cuda => "https://download.pytorch.org/whl/cu128",
            Gpu::Rocm => "https://download.pytorch.org/whl/rocm7.0",
            Gpu::Mps => "https://pypi.org/simple",
            Gpu::Cpu => "https://download.pytorch.org/whl/cpu",
            Gpu::Auto => detect().torch_index(),
        }
    }

    pub fn resolved(self) -> Gpu {
        match self {
            Gpu::Auto => detect(),
            other => other,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Gpu::Auto => "auto",
            Gpu::Cuda => "cuda",
            Gpu::Rocm => "rocm",
            Gpu::Mps => "mps",
            Gpu::Cpu => "cpu",
        }
    }
}

/// What this machine has: ROCm when the kernel exposes an AMD compute
/// device (Linux only), Metal on Apple silicon, CUDA when `nvidia-smi`
/// answers, the CPU otherwise.
pub fn detect() -> Gpu {
    if cfg!(target_os = "linux") && Path::new("/dev/kfd").exists() {
        return Gpu::Rocm;
    }
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Gpu::Mps;
    }
    if let Some(smi) = tools::on_path("nvidia-smi") {
        let mut c = run::command(smi);
        c.arg("-L");
        if run::output(c).is_ok_and(|s| s.contains("GPU")) {
            return Gpu::Cuda;
        }
    }
    Gpu::Cpu
}

/// The device the models will run on, as far as the machine says.
#[derive(Debug, Clone, PartialEq)]
pub struct Device {
    /// Never [`Gpu::Auto`].
    pub gpu: Gpu,
    pub name: String,
    /// What the models can use, in bytes: the card's own memory, or on
    /// Apple silicon the whole machine's, which the GPU shares.
    pub memory: Option<u64>,
}

impl Device {
    /// « AMD Radeon RX 7900 XTX · 24 GB · rocm ».
    pub fn describe(&self) -> String {
        match self.memory {
            Some(bytes) => format!(
                "{} · {:.0} GB · {}",
                self.name,
                bytes as f64 / f64::from(1u32 << 30),
                self.gpu.label()
            ),
            None => format!("{} · {}", self.name, self.gpu.label()),
        }
    }
}

/// The device [`detect`] picks, with its name and its memory.
pub fn device() -> Device {
    let gpu = detect();
    let found = match gpu {
        Gpu::Rocm => amd_linux(),
        Gpu::Cuda => nvidia(),
        Gpu::Mps => Some(Device {
            gpu,
            name: sysctl("machdep.cpu.brand_string").unwrap_or_else(|| "Apple silicon".into()),
            memory: sysctl("hw.memsize").and_then(|m| m.parse().ok()),
        }),
        Gpu::Cpu | Gpu::Auto => None,
    };
    found.unwrap_or_else(|| Device {
        gpu: if gpu == Gpu::Auto { Gpu::Cpu } else { gpu },
        name: match gpu {
            Gpu::Rocm => "AMD GPU".into(),
            Gpu::Cuda => "NVIDIA GPU".into(),
            _ => "processor (no GPU torch can use)".into(),
        },
        memory: None,
    })
}

/// The first card `nvidia-smi` lists, which is the one torch takes.
fn nvidia() -> Option<Device> {
    let smi = tools::on_path("nvidia-smi")?;
    let mut c = run::command(smi);
    c.args([
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]);
    let out = run::output(c).ok()?;
    let (name, mib) = out.lines().next()?.rsplit_once(',')?;
    Some(Device {
        gpu: Gpu::Cuda,
        name: name.trim().to_string(),
        memory: mib.trim().parse::<u64>().ok().map(|m| m << 20),
    })
}

/// The AMD card with the most memory: a discrete card rather than the
/// processor's own graphics, which ROCm lists too.
fn amd_linux() -> Option<Device> {
    let mut best: Option<(u64, PathBuf)> = None;
    for entry in fs::read_dir("/sys/class/drm").ok()?.flatten() {
        let dir = entry.path().join("device");
        let vendor = fs::read_to_string(dir.join("vendor")).unwrap_or_default();
        if vendor.trim() != "0x1002" {
            continue;
        }
        let Some(vram) = fs::read_to_string(dir.join("mem_info_vram_total"))
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
        else {
            continue;
        };
        if best.as_ref().is_none_or(|(b, _)| vram > *b) {
            best = Some((vram, dir));
        }
    }
    let (vram, dir) = best?;
    let name = fs::read_to_string(dir.join("product_name"))
        .ok()
        .map(|n| n.trim().to_string())
        .filter(|n| !n.is_empty())
        .or_else(|| lspci_name(&dir))
        .unwrap_or_else(|| "AMD GPU".into());
    Some(Device {
        gpu: Gpu::Rocm,
        name,
        memory: Some(vram),
    })
}

/// The device's name as `lspci -mm` words it: the fourth quoted field.
fn lspci_name(dir: &Path) -> Option<String> {
    let slot = fs::canonicalize(dir).ok()?;
    let slot = slot.file_name()?.to_str()?.to_string();
    let lspci = tools::on_path("lspci")?;
    let mut c = run::command(lspci);
    c.args(["-mm", "-s", &slot]);
    let out = run::output(c).ok()?;
    out.split('"').nth(5).map(|n| n.to_string())
}

fn sysctl(key: &str) -> Option<String> {
    let mut c = run::command("sysctl");
    c.args(["-n", key]);
    run::output(c).ok().filter(|s| !s.is_empty())
}

/// pymss's device memory per second of block, from the one measurement
/// there is: Fullness V3 at 25 %, blocks of 600 s, 15 GB of a 24 GB card.
const GIB_PER_BLOCK_SECOND: f64 = 15.0 / 600.0;

/// What the runtime's own context and the desktop hold, on top of torch's
/// peak.
const RESERVE_GIB: f64 = 2.0;

/// The blocks a recommendation picks from, largest first.
const BLOCKS: [u32; 5] = [600, 300, 180, 120, 60];

/// Inference settings for a device, and why.
#[derive(Debug, Clone, PartialEq)]
pub struct Recommendation {
    pub inference: Inference,
    pub reason: String,
}

/// The largest pymss block the device's memory holds, a batch of one below
/// 12 GB, MSST when not even a minute fits. An estimate from one card:
/// `bench-vram` measures it on this one.
pub fn recommend(device: &Device) -> Recommendation {
    let gib = |bytes: u64| bytes as f64 / f64::from(1u32 << 30);
    let base = Inference::default();
    let Some(memory) = device.memory else {
        return match device.gpu {
            Gpu::Cpu | Gpu::Auto => Recommendation {
                inference: Inference {
                    block_seconds: 120,
                    ..base
                },
                reason: "no GPU: the processor runs the models, many hours a film; blocks of \
                         120 s keep the memory low"
                    .into(),
            },
            _ => Recommendation {
                inference: base,
                reason: "the card's memory is unknown: the defaults, sized for 24 GB".into(),
            },
        };
    };
    let total = gib(memory);
    // Apple silicon shares its memory with everything else the machine runs.
    let usable = if device.gpu == Gpu::Mps {
        total / 2.0 - RESERVE_GIB
    } else {
        total - RESERVE_GIB
    };
    let batch_size = if total < 12.0 { 1 } else { 0 };
    let fits = BLOCKS
        .into_iter()
        .find(|&b| f64::from(b) * GIB_PER_BLOCK_SECOND <= usable);
    match fits {
        Some(block_seconds) => Recommendation {
            inference: Inference {
                block_seconds,
                batch_size,
                ..base
            },
            reason: format!(
                "{total:.0} GB: pymss in blocks of {block_seconds} s (about {:.0} GB at the \
                 peak){}",
                f64::from(block_seconds) * GIB_PER_BLOCK_SECOND + 0.5,
                if batch_size == 1 {
                    ", one window at a time"
                } else {
                    ""
                }
            ),
        },
        None => Recommendation {
            inference: Inference {
                engine: Engine::Msst,
                batch_size: 1,
                ..base
            },
            reason: format!(
                "{total:.0} GB: too little for pymss's blocks; MSST, which needs the least, \
                 about three times slower"
            ),
        },
    }
}

#[derive(Debug, Clone)]
pub struct Options {
    pub gpu: Gpu,
    /// A base interpreter to build the venv from; found on PATH otherwise.
    pub python: Option<PathBuf>,
    /// Rebuild the venv and the MSST checkout even if they look complete.
    pub force: bool,
}

/// A base interpreter: the one given, else `python3.12`, `python3.11`,
/// `py -3.12` (Windows), `python3`, `python` — the first that is 3.11 or 3.12.
pub fn find_python(explicit: Option<&Path>) -> Result<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(p) = explicit {
        candidates.push(p.to_path_buf());
    } else {
        for name in ["python3.12", "python3.11", "python3", "python"] {
            if let Some(p) = tools::on_path(name) {
                candidates.push(p);
            }
        }
        if cfg!(windows)
            && let Some(py) = tools::on_path("py")
        {
            for version in ["3.12", "3.11"] {
                let mut c = run::command(&py);
                c.args([
                    &format!("-{version}"),
                    "-c",
                    "import sys; print(sys.executable)",
                ]);
                if let Ok(exe) = run::output(c) {
                    candidates.push(PathBuf::from(exe));
                }
            }
        }
    }
    let mut seen = Vec::new();
    for candidate in candidates {
        match python_version(&candidate) {
            Some((3, minor)) if (11..=12).contains(&minor) => return Ok(candidate),
            Some((major, minor)) => {
                seen.push(format!("{} is {major}.{minor}", candidate.display()))
            }
            None => seen.push(format!("{} does not run", candidate.display())),
        }
    }
    Err(Error::msg(format!(
        "no Python 3.11 or 3.12 found (torch {TORCH_VERSION} ships wheels for those); install one from python.org or your distribution, or pass --python. Seen: {}",
        if seen.is_empty() {
            "nothing".to_string()
        } else {
            seen.join("; ")
        }
    )))
}

fn python_version(python: &Path) -> Option<(u32, u32)> {
    let mut c = run::command(python);
    c.args([
        "-c",
        "import sys; print(sys.version_info[0], sys.version_info[1])",
    ]);
    let out = run::output(c).ok()?;
    let mut parts = out.split_whitespace().map(|p| p.parse::<u32>().ok());
    Some((parts.next()??, parts.next()??))
}

/// Whether a venv's Python imports torch.
fn has_torch(python: &Path) -> bool {
    let mut c = run::command(python);
    c.args(["-c", "import torch"]);
    run::output(c).is_ok()
}

/// Build (or complete) the environment and record it in `config`.
pub fn run(config: &mut Config, options: &Options, progress: Progress) -> Result<()> {
    let gpu = options.gpu.resolved();
    progress(Event::message(format!(
        "GPU: {} → torch index {}",
        gpu.label(),
        gpu.torch_index()
    )));

    // 1. The venv.
    let venv = Config::venv_dir();
    let venv_python = Config::venv_python(&venv);
    if options.force && venv.exists() {
        progress(Event::message(format!("removing {}", venv.display())));
        fs::remove_dir_all(&venv).map_err(|e| Error::io(&venv, e))?;
    }
    if !venv_python.is_file() {
        let base = find_python(options.python.as_deref())?;
        progress(Event::message(format!(
            "creating the venv at {} from {}",
            venv.display(),
            base.display()
        )));
        if let Some(parent) = venv.parent() {
            fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let mut c = run::command(&base);
        c.arg("-m").arg("venv").arg(&venv);
        run::output(c)?;
    } else {
        progress(Event::message(format!(
            "venv present at {}",
            venv.display()
        )));
    }

    // 2. torch, then the rest. pip decides what is already satisfied.
    if !has_torch(&venv_python) || options.force {
        pip(&venv_python, &["install", "--upgrade", "pip"], progress)?;
        progress(Event::message(format!(
            "installing torch {TORCH_VERSION} ({}); this is a few gigabytes",
            gpu.label()
        )));
        pip(
            &venv_python,
            &[
                "install",
                &format!("torch=={TORCH_VERSION}"),
                &format!("torchaudio=={TORCH_VERSION}"),
                "--index-url",
                gpu.torch_index(),
            ],
            progress,
        )?;
    } else {
        progress(Event::message("torch already installed"));
    }
    let mut args = vec!["install"];
    args.extend(REQUIREMENTS);
    progress(Event::message("installing the MSST dependencies"));
    pip(&venv_python, &args, progress)?;

    // 3. MSST itself.
    let msst = config.msst_dir();
    if options.force && msst.exists() {
        progress(Event::message(format!("removing {}", msst.display())));
        fs::remove_dir_all(&msst).map_err(|e| Error::io(&msst, e))?;
    }
    if msst.join("inference.py").is_file() {
        progress(Event::message(format!(
            "MSST present at {}",
            msst.display()
        )));
    } else {
        install_msst(&venv_python, &msst, progress)?;
    }

    // 4. Record and verify.
    config.python = Some(venv_python.clone());
    config.msst_dir = Some(msst.clone());
    config.save()?;
    let mut c = run::command(&venv_python);
    c.arg("-c")
        .arg("import sys, torch; sys.path.insert(0, sys.argv[1]); import inference; print(torch.__version__, 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu', getattr(torch.version, 'hip', None) or '')")
        .arg(&msst);
    let report = run::output(c)?;
    progress(Event::message(format!("ready: torch {report}")));
    Ok(())
}

fn pip(python: &Path, args: &[&str], progress: Progress) -> Result<()> {
    let mut c = run::command(python);
    c.arg("-m")
        .arg("pip")
        .args(args)
        .arg("--disable-pip-version-check");
    c.env("PIP_PROGRESS_BAR", "off");
    run::lines(c, |_, line| {
        let line = line.trim();
        if !line.is_empty() {
            progress(Event::message(format!("pip: {line}")));
        }
    })
}

/// The pinned MSST commit, as GitHub's zip archive, unpacked by the venv's
/// own Python so nothing has to link a zip reader.
fn install_msst(python: &Path, msst: &Path, progress: Progress) -> Result<()> {
    let url = format!(
        "https://github.com/ZFTurbo/Music-Source-Separation-Training/archive/{MSST_COMMIT}.zip"
    );
    let zip = Config::data_dir().join("msst.zip");
    progress(Event::message(format!(
        "fetching MSST at {}",
        &MSST_COMMIT[..7]
    )));
    download::fetch(&url, &zip, progress)?;
    let staging = Config::data_dir().join("msst.unpack");
    if staging.exists() {
        fs::remove_dir_all(&staging).map_err(|e| Error::io(&staging, e))?;
    }
    let mut c = run::command(python);
    c.arg("-c")
        .arg("import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])")
        .arg(&zip)
        .arg(&staging);
    run::output(c)?;
    let unpacked = fs::read_dir(&staging)
        .map_err(|e| Error::io(&staging, e))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("inference.py").is_file())
        .ok_or_else(|| Error::msg("the MSST archive has no inference.py at its root"))?;
    if let Some(parent) = msst.parent() {
        fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    fs::rename(&unpacked, msst).map_err(|e| Error::io(msst, e))?;
    let _ = fs::remove_dir_all(&staging);
    let _ = fs::remove_file(&zip);
    progress(Event::message(format!(
        "MSST installed at {}",
        msst.display()
    )));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(gpu: Gpu, gib: u64) -> Device {
        Device {
            gpu,
            name: "test".into(),
            memory: Some(gib << 30),
        }
    }

    #[test]
    fn the_block_follows_the_memory() {
        let block = |d: &Device| recommend(d).inference.block_seconds;
        assert_eq!(block(&card(Gpu::Rocm, 24)), 600);
        assert_eq!(block(&card(Gpu::Cuda, 16)), 300);
        assert_eq!(block(&card(Gpu::Cuda, 12)), 300);
        assert_eq!(block(&card(Gpu::Cuda, 8)), 180);
        assert_eq!(block(&card(Gpu::Cuda, 4)), 60);
        // Apple silicon keeps half its memory for the rest of the machine.
        assert_eq!(block(&card(Gpu::Mps, 32)), 300);
        assert_eq!(recommend(&card(Gpu::Cuda, 24)).inference.batch_size, 0);
        assert_eq!(recommend(&card(Gpu::Cuda, 8)).inference.batch_size, 1);
    }

    #[test]
    fn a_small_card_runs_msst() {
        let r = recommend(&card(Gpu::Cuda, 3));
        assert_eq!(r.inference.engine, Engine::Msst);
        assert_eq!(r.inference.batch_size, 1);
    }

    #[test]
    fn an_unknown_card_keeps_the_defaults() {
        let d = Device {
            gpu: Gpu::Cuda,
            name: "?".into(),
            memory: None,
        };
        assert_eq!(recommend(&d).inference, Inference::default());
    }
}
