//! Separating on another machine's GPU, over SSH.
//!
//! A remote is a machine with omnifrog installed and set up there (`omnifrog
//! setup` on it), that `ssh` reaches with a key and no question
//! ([`crate::config::Remote`]). Two commands of that machine's omnifrog are
//! all this side runs there, both hidden from its help:
//!
//! - `remote-check`, once a step: its version, the device torch would run
//!   on, and how much of its GPU's memory is in use ([`probe`]). A remote of
//!   another version, or without a GPU torch can use, is left out.
//! - `remote-worker '<model>' '<inference>' <idle below GiB>`, once a pass
//!   ([`serve`]): the worker of [`crate::worker`], its protocol untouched,
//!   with two words of its own before `PRET`:
//!
//! ```text
//! BUSY <why>     another program holds its GPU, or another run does: nothing was loaded
//! DIR <path>     where the inputs go and the stems come out, on that machine
//! ```
//!
//! The inputs go there and the stems come back through `ssh … cat`, over
//! the connection the first command opened and kept (on Unix): a pass of a
//! two-hour film is about 1.5 GB out and 5 GB back, seconds on a 10 Gb/s
//! link against minutes of inference.
//!
//! **Whatever else runs on that machine comes first.** Its GPU is taken
//! only while the other programs there hold less than `idle_below_gib` of
//! its memory, and it is given back after every pass — the model is loaded
//! again for the next one — so a program that wants it between two passes
//! finds it free. Nothing there is ever unloaded or stopped. A remote that
//! is busy is asked again a minute later; one that fails twice in a row is
//! left out for the rest of the step. This machine's GPU does whatever the
//! remotes do not, so a step never waits on one.

use crate::config::{Config, Inference};
use crate::models::{self, Model};
use crate::progress::Event;
use crate::worker::{self, Lock};
use crate::{Error, Result};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// How a line of a remote's standard error starts when it is meant for the
/// log rather than for an error report.
pub const SAYS: &str = "omnifrog: ";

/// What the two sides must agree on: the worker's script and these words.
/// Bumped with any change to `remote-worker`'s arguments or words.
const PROTOCOL: u32 = 1;

/// The directory of a remote's work directory a pass's files live in.
const DIR: &str = "remote";

/// `text` as one word of a POSIX shell, whatever it holds.
pub fn quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

/// What both sides compute and compare: a remote whose worker speaks
/// another version of the protocol is left out.
pub fn protocol() -> String {
    // FNV-1a: stable across builds and platforms, unlike the std hasher.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in worker::SOURCE.bytes().chain(PROTOCOL.to_le_bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// The way to one machine: `ssh` with the options every call needs.
#[derive(Debug, Clone)]
pub struct Ssh {
    host: String,
    /// On Unix, the socket the first connection leaves open for the next
    /// ones (`ControlMaster`), so a transfer does not pay a handshake.
    control: Option<PathBuf>,
}

impl Ssh {
    pub fn new(host: &str) -> Self {
        let control = cfg!(unix).then(|| {
            dirs::runtime_dir()
                .unwrap_or_else(std::env::temp_dir)
                .join("omnifrog-ssh-%C")
        });
        Self {
            host: host.to_string(),
            control,
        }
    }

    /// `ssh` running `line` in the remote's shell; the streams are the
    /// caller's to set.
    pub fn command(&self, line: &str) -> Command {
        let mut command = crate::run::command("ssh");
        // No terminal, never a question: a key or nothing. A link that dies
        // is noticed within a minute rather than at the next request.
        command.arg("-T").args([
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=4",
        ]);
        if let Some(control) = &self.control {
            command
                .arg("-o")
                .arg("ControlMaster=auto")
                .arg("-o")
                .arg(format!("ControlPath={}", control.display()))
                .arg("-o")
                .arg("ControlPersist=120");
        }
        command.arg("--").arg(&self.host).arg(line);
        command
    }

    /// Run `line` there and return what it printed.
    pub fn run(&self, line: &str) -> Result<String> {
        let output = self
            .command(line)
            .stdin(Stdio::null())
            .output()
            .map_err(|e| self.failed("could not start", e.to_string()))?;
        if !output.status.success() {
            return Err(self.failed(
                &output.status.to_string(),
                String::from_utf8_lossy(&output.stderr).trim().to_string(),
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }

    /// Copy `from` to the path `to` on that machine.
    pub fn upload(&self, from: &Path, to: &str) -> Result<()> {
        let file = fs::File::open(from).map_err(|e| Error::io(from, e))?;
        let output = self
            .command(&format!("cat > {}", quote(to)))
            .stdin(file)
            .stdout(Stdio::null())
            .output()
            .map_err(|e| self.failed("could not start", e.to_string()))?;
        if !output.status.success() {
            return Err(self.failed(
                &output.status.to_string(),
                format!(
                    "sending {}: {}",
                    from.display(),
                    String::from_utf8_lossy(&output.stderr).trim()
                ),
            ));
        }
        Ok(())
    }

    /// Copy the path `from` on that machine to `to`, aside and then
    /// renamed: a file under its real name is always whole.
    pub fn download(&self, from: &str, to: &Path) -> Result<()> {
        let part = to.with_extension("wav.part");
        let file = fs::File::create(&part).map_err(|e| Error::io(&part, e))?;
        let output = self
            .command(&format!("cat {}", quote(from)))
            .stdin(Stdio::null())
            .stdout(file)
            .output()
            .map_err(|e| self.failed("could not start", e.to_string()))?;
        if !output.status.success() {
            let _ = fs::remove_file(&part);
            return Err(self.failed(
                &output.status.to_string(),
                format!(
                    "fetching {from}: {}",
                    String::from_utf8_lossy(&output.stderr).trim()
                ),
            ));
        }
        fs::rename(&part, to).map_err(|e| Error::io(to, e))
    }

    fn failed(&self, status: &str, detail: String) -> Error {
        Error::Command {
            program: format!("ssh {}", self.host),
            status: status.to_string(),
            detail,
        }
    }
}

/// What a remote said about itself.
#[derive(Debug, Clone, PartialEq)]
pub struct Probe {
    /// Its omnifrog's version.
    pub version: String,
    /// Its [`protocol`].
    pub protocol: String,
    /// The device torch runs on there: a GPU's name, or `cpu`.
    pub device: String,
    /// Memory in use and in all on its GPU, in GiB, when it can be read.
    pub vram: Option<(f64, f64)>,
    /// What is missing there for a separation to run.
    pub missing: Vec<String>,
}

impl Probe {
    /// Why this machine leaves the remote out; `None` = it takes passes.
    pub fn unfit(&self) -> Option<String> {
        if let Some(missing) = self.missing.first() {
            return Some(missing.clone());
        }
        if self.protocol != protocol() {
            return Some(format!(
                "its omnifrog is {} and speaks another protocol than this {}: install the same \
                 version there",
                self.version,
                env!("CARGO_PKG_VERSION")
            ));
        }
        if self.device == "cpu" || self.device.is_empty() {
            return Some("torch sees no GPU there".to_string());
        }
        None
    }

    /// One line for a progress message.
    pub fn describe(&self) -> String {
        match self.vram {
            Some((used, total)) => {
                format!("{} · {used:.1} of {total:.0} GiB in use", self.device)
            }
            None => self.device.clone(),
        }
    }

    fn parse(text: &str) -> Self {
        let mut probe = Probe {
            version: String::new(),
            protocol: String::new(),
            device: String::new(),
            vram: None,
            missing: Vec::new(),
        };
        for line in text.lines() {
            let line = line.trim_end();
            if let Some(rest) = line.strip_prefix("OMNIFROG ") {
                let mut words = rest.split_whitespace();
                probe.version = words.next().unwrap_or_default().to_string();
                probe.protocol = words.next().unwrap_or_default().to_string();
            } else if let Some(device) = line.strip_prefix("DEVICE ") {
                probe.device = device.trim().to_string();
            } else if let Some(rest) = line.strip_prefix("VRAM ") {
                let mut words = rest.split_whitespace().map(str::parse::<f64>);
                if let (Some(Ok(used)), Some(Ok(total))) = (words.next(), words.next()) {
                    probe.vram = Some((used, total));
                }
            } else if let Some(why) = line.strip_prefix("MISSING ") {
                probe.missing.push(why.trim().to_string());
            }
        }
        if probe.version.is_empty() {
            probe.missing.insert(
                0,
                "no omnifrog answered there (`command` in `remotes`)".to_string(),
            );
        }
        probe
    }
}

/// Ask `remote` what it is: `remote-check` over SSH.
pub fn probe(remote: &crate::config::Remote, ssh: &Ssh) -> Result<Probe> {
    let text = ssh.run(&format!("{} remote-check", remote.command))?;
    Ok(Probe::parse(&text))
}

/// The remote's side of [`probe`]: what `remote-check` prints.
pub fn check(config: &Config) -> Vec<String> {
    let mut lines = vec![format!(
        "OMNIFROG {} {}",
        env!("CARGO_PKG_VERSION"),
        protocol()
    )];
    let python = config.python();
    if !python.is_file() {
        lines.push(format!(
            "MISSING no Python for the inference at {}: run `omnifrog setup` there",
            python.display()
        ));
    } else {
        let probe = crate::run::command(&python)
            .args([
                "-c",
                "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() \
                 else 'cpu')",
            ])
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output();
        match probe {
            Ok(output) if output.status.success() => {
                let device = String::from_utf8_lossy(&output.stdout);
                lines.push(format!("DEVICE {}", device.trim()));
            }
            _ => lines.push(format!(
                "MISSING torch does not load in {}: run `omnifrog setup` there",
                python.display()
            )),
        }
    }
    if !config.msst_dir().join("inference.py").is_file() {
        lines.push(format!(
            "MISSING no MSST checkout at {}: run `omnifrog setup` there",
            config.msst_dir().display()
        ));
    }
    if let Some((used, total)) = vram() {
        lines.push(format!("VRAM {used:.2} {total:.2}"));
    }
    lines
}

/// The remote's side of a pass: decline if the GPU is someone else's, else
/// fetch the model, say where the files go and become the worker, on this
/// process's own standard streams. Returns the worker's exit code.
pub fn serve(
    config: &Config,
    model: &Model,
    inference: Inference,
    idle_below_gib: f64,
) -> Result<i32> {
    let say = |line: &str| {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{line}");
        let _ = out.flush();
    };
    if let Some(why) = busy(idle_below_gib) {
        say(&format!("BUSY {why}"));
        return Ok(0);
    }
    // A run on this machine, or another machine's pass, has the GPU.
    let lock = match Lock::take(&config.work_dir()) {
        Ok(lock) => lock,
        Err(why) => {
            say(&format!("BUSY {why}"));
            return Ok(0);
        }
    };
    let local = models::fetch(model, config, &|event| {
        if let Event::Message { text } = event {
            eprintln!("{SAYS}{text}");
        }
    })?;
    // One pass at a time here, under the lock: whatever is in it was left
    // by a pass that did not end.
    let dir = config.work_dir().join(DIR);
    if dir.exists() {
        fs::remove_dir_all(&dir).map_err(|e| Error::io(&dir, e))?;
    }
    for sub in ["in", "out"] {
        let path = dir.join(sub);
        fs::create_dir_all(&path).map_err(|e| Error::io(&path, e))?;
    }
    let mut command = worker::command(config, model, &local, inference)?;
    say(&format!("DIR {}", dir.display()));
    let status = command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| Error::Command {
            program: config.python().display().to_string(),
            status: "could not start".into(),
            detail: e.to_string(),
        })?;
    let _ = fs::remove_dir_all(&dir);
    drop(lock);
    Ok(status.code().unwrap_or(1))
}

/// Why the GPU here is someone else's now; `None` = it is free, or there is
/// no telling.
pub fn busy(idle_below_gib: f64) -> Option<String> {
    if idle_below_gib <= 0.0 {
        return None;
    }
    let (used, total) = vram()?;
    (used >= idle_below_gib).then(|| {
        format!(
            "another program holds {used:.1} of its GPU's {total:.0} GiB (it is taken below \
             {idle_below_gib} GiB)"
        )
    })
}

/// Memory in use and in all on this machine's largest GPU, in GiB: the
/// kernel's own counters for an AMD card, `nvidia-smi` for an NVIDIA one.
pub fn vram() -> Option<(f64, f64)> {
    amd_vram(Path::new("/sys/class/drm")).or_else(nvidia_vram)
}

const GIB: f64 = (1u64 << 30) as f64;

/// The largest card under `drm` (`/sys/class/drm`): an integrated GPU's
/// few hundred MB are not where the separation runs.
fn amd_vram(drm: &Path) -> Option<(f64, f64)> {
    let mut best: Option<(f64, f64)> = None;
    for entry in fs::read_dir(drm).ok()?.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // card0, not card0-DP-1.
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let device = entry.path().join("device");
        let read = |file: &str| -> Option<u64> {
            fs::read_to_string(device.join(file))
                .ok()?
                .trim()
                .parse()
                .ok()
        };
        if let (Some(used), Some(total)) = (read("mem_info_vram_used"), read("mem_info_vram_total"))
        {
            let (used, total) = (used as f64 / GIB, total as f64 / GIB);
            if total >= 2.0 && best.is_none_or(|(_, t)| total > t) {
                best = Some((used, total));
            }
        }
    }
    best
}

fn nvidia_vram() -> Option<(f64, f64)> {
    let output = crate::run::command("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_nvidia(&String::from_utf8_lossy(&output.stdout))
}

/// `nvidia-smi`'s lines, in MiB: the card with the most memory.
fn parse_nvidia(text: &str) -> Option<(f64, f64)> {
    text.lines()
        .filter_map(|line| {
            let mut fields = line.split(',').map(|f| f.trim().parse::<f64>());
            match (fields.next(), fields.next()) {
                (Some(Ok(used)), Some(Ok(total))) => Some((used / 1024.0, total / 1024.0)),
                _ => None,
            }
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quoted_word_survives_any_quote_inside_it() {
        assert_eq!(quote("plain"), "'plain'");
        assert_eq!(quote("it's"), "'it'\\''s'");
        assert_eq!(quote(r#"{"a":"b c"}"#), r#"'{"a":"b c"}'"#);
    }

    #[test]
    fn the_protocol_is_the_same_on_both_sides_of_one_build() {
        assert_eq!(protocol(), protocol());
        assert_eq!(protocol().len(), 16);
    }

    #[test]
    fn a_probe_reads_what_remote_check_prints() {
        let probe = Probe::parse(&format!(
            "OMNIFROG 0.1.0 {}\nDEVICE AMD Radeon RX 7900 XTX\nVRAM 0.42 23.98\n",
            protocol()
        ));
        assert_eq!(probe.version, "0.1.0");
        assert_eq!(probe.device, "AMD Radeon RX 7900 XTX");
        assert_eq!(probe.vram, Some((0.42, 23.98)));
        assert_eq!(probe.unfit(), None);
        assert_eq!(
            probe.describe(),
            "AMD Radeon RX 7900 XTX · 0.4 of 24 GiB in use"
        );
    }

    #[test]
    fn a_remote_without_a_gpu_or_of_another_version_is_left_out() {
        let cpu = Probe::parse(&format!("OMNIFROG 0.1.0 {}\nDEVICE cpu\n", protocol()));
        assert!(cpu.unfit().unwrap().contains("no GPU"));
        let other = Probe::parse("OMNIFROG 0.0.9 0123456789abcdef\nDEVICE Some GPU\n");
        assert!(other.unfit().unwrap().contains("another protocol"));
        let missing = Probe::parse(&format!(
            "OMNIFROG 0.1.0 {}\nMISSING no MSST checkout\n",
            protocol()
        ));
        assert_eq!(missing.unfit().unwrap(), "no MSST checkout");
        let nothing = Probe::parse("bash: omnifrog: command not found\n");
        assert!(nothing.unfit().unwrap().contains("no omnifrog answered"));
    }

    #[test]
    fn the_largest_amd_card_is_the_one_counted() {
        let root = std::env::temp_dir().join(format!("omnifrog-drm-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let card = |name: &str, used: u64, total: u64| {
            let device = root.join(name).join("device");
            fs::create_dir_all(&device).unwrap();
            fs::write(device.join("mem_info_vram_used"), format!("{used}\n")).unwrap();
            fs::write(device.join("mem_info_vram_total"), format!("{total}\n")).unwrap();
        };
        let gib = 1u64 << 30;
        card("card0", 16 << 20, 512 << 20);
        card("card1", 2 * gib, 24 * gib);
        fs::create_dir_all(root.join("card1-DP-1")).unwrap();
        assert_eq!(amd_vram(&root), Some((2.0, 24.0)));
        fs::remove_dir_all(&root).unwrap();
        assert_eq!(amd_vram(&root), None);
    }

    #[test]
    fn nvidia_smi_is_read_in_mib() {
        assert_eq!(parse_nvidia("1024, 8192\n512, 24576\n"), Some((0.5, 24.0)));
        assert_eq!(parse_nvidia("[N/A], [N/A]\n"), None);
    }
}
