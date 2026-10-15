//! What is installed, what is missing, and whether a run could start.

use crate::config::{Config, Encoder};
use crate::models::Catalogue;
use crate::{encode, run, setup, tools, worker};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    Ok,
    Warning,
    Missing,
}

#[derive(Debug, Clone)]
pub struct Check {
    pub name: String,
    pub status: Status,
    pub detail: String,
}

fn check(name: &str, status: Status, detail: impl Into<String>) -> Check {
    Check {
        name: name.to_string(),
        status,
        detail: detail.into(),
    }
}

/// Every check, in the order a person would fix them.
pub fn run(config: &Config) -> Vec<Check> {
    let mut checks = Vec::new();

    let path = Config::path();
    checks.push(check(
        "config",
        if path.is_file() {
            Status::Ok
        } else {
            Status::Warning
        },
        if path.is_file() {
            path.display().to_string()
        } else {
            format!("{} (absent: defaults in use)", path.display())
        },
    ));

    let device = setup::device();
    let advice = setup::recommend(&device);
    checks.push(check(
        "device",
        if device.gpu == setup::Gpu::Cpu {
            Status::Warning
        } else {
            Status::Ok
        },
        format!(
            "{} — for it: {}{}",
            device.describe(),
            advice.inference.describe(),
            if config.inference == advice.inference {
                " (configured)".to_string()
            } else {
                format!(" (configured: {})", config.inference.describe())
            }
        ),
    ));

    let work = config.work_dir();
    checks.push(check(
        "work dir",
        if work.is_dir() || work.parent().is_some_and(|p| p.is_dir()) {
            Status::Ok
        } else {
            Status::Warning
        },
        match (work.is_dir(), work.to_str().is_some_and(|s| s.is_ascii())) {
            (true, true) => work.display().to_string(),
            (true, false) => format!("{} (not ASCII: Dee cannot use it)", work.display()),
            (false, _) => format!("{} (will be created)", work.display()),
        },
    ));

    for tool in tools::Tool::here() {
        let needed = match tool {
            tools::Tool::Harlettizer => config.encoder == Encoder::Hz || config.encoder.is_iamf(),
            tools::Tool::Dee => config.encoder == Encoder::Dee,
            tools::Tool::Wine => {
                config.encoder == Encoder::Dee
                    && tools::find(config, tools::Tool::Dee).is_some_and(|d| tools::needs_wine(&d))
            }
            _ => true,
        };
        let found = tool_check(config, tool, needed);
        let usable = found.status == Status::Ok;
        checks.push(found);
        if tool == tools::Tool::Harlettizer
            && usable
            && let Some(harlettizer) = tools::find(config, tool)
        {
            checks.push(iamf(config, &harlettizer));
        }
    }

    let python = config.python();
    if python.is_file() {
        match torch_report(&python, &config.msst_dir()) {
            Ok(report) => checks.push(check(
                "python",
                Status::Ok,
                format!("{report} — {}", python.display()),
            )),
            Err(detail) => checks.push(check(
                "python",
                Status::Missing,
                format!("{} — {detail}", python.display()),
            )),
        }
    } else {
        checks.push(check(
            "python",
            Status::Missing,
            format!("{} absent: run `setup`", python.display()),
        ));
    }

    let msst = config.msst_dir();
    checks.push(check(
        "msst",
        if msst.join("inference.py").is_file() {
            Status::Ok
        } else {
            Status::Missing
        },
        if msst.join("inference.py").is_file() {
            msst.display().to_string()
        } else {
            format!("{} has no inference.py: run `setup`", msst.display())
        },
    ));

    match worker::ensure() {
        Ok(path) => checks.push(check("worker", Status::Ok, path.display().to_string())),
        Err(e) => checks.push(check("worker", Status::Missing, e.to_string())),
    }

    match Catalogue::load() {
        Ok(catalogue) => {
            for (what, key) in [
                ("model vo", &config.models.vo),
                ("model vf", &config.models.vf),
            ] {
                match catalogue.get(key) {
                    Ok(model) => {
                        let local = model.local(config);
                        let missing = local.missing();
                        checks.push(check(
                            what,
                            if missing.is_empty() {
                                Status::Ok
                            } else {
                                Status::Warning
                            },
                            if missing.is_empty() {
                                format!("{key} — {} ({})", model.name, local.checkpoint.display())
                            } else {
                                format!(
                                    "{key} — {}: {} to fetch (`models fetch`)",
                                    model.name,
                                    missing.len()
                                )
                            },
                        ));
                    }
                    Err(_) => checks.push(check(
                        what,
                        Status::Missing,
                        format!("`{key}` is not in the catalogue"),
                    )),
                }
            }
        }
        Err(e) => checks.push(check("catalogue", Status::Missing, e.to_string())),
    }
    checks
}

/// Where an external program is, and whether it starts: a file found is not
/// yet a program — a library or an archive chosen by mistake, or a build
/// for another system, is there and cannot run.
fn tool_check(config: &Config, tool: tools::Tool, needed: bool) -> Check {
    let missing = if needed {
        Status::Missing
    } else {
        Status::Warning
    };
    let not_needed = if needed {
        ""
    } else {
        " — not needed by the configured encoder"
    };
    let Some(found) = tools::find(config, tool) else {
        return check(
            tool.name(),
            missing,
            format!("not found ({}){not_needed}", tool.purpose()),
        );
    };
    let broken = |why: String| {
        let hint = tool
            .which_file()
            .map(|hint| format!(" — {hint}"))
            .unwrap_or_default();
        check(
            tool.name(),
            missing.clone(),
            format!("{} — {why}{hint}{not_needed}", found.display()),
        )
    };
    if let Some(why) = tools::not_a_program(&found) {
        return broken(why.to_string());
    }
    // Asking costs a start of the program: 15 s for Wine, which boots its
    // prefix — not worth it for a tool nothing uses.
    if !needed {
        return check(tool.name(), Status::Ok, found.display().to_string());
    }
    match tools::version(&found, tool) {
        tools::Version::CannotStart(why) => broken(format!("cannot start: {why}")),
        tools::Version::Started(version) if !version.is_empty() => check(
            tool.name(),
            Status::Ok,
            format!("{version} — {}", found.display()),
        ),
        tools::Version::Started(_) | tools::Version::NotAsked => {
            check(tool.name(), Status::Ok, found.display().to_string())
        }
    }
}

/// Whether the harlettizer found writes IAMF into an `.mka`, and in which
/// of the two modes: missing when the configured encoder needs what it
/// lacks, a note otherwise.
fn iamf(config: &Config, harlettizer: &Path) -> Check {
    let help = match encode::iamf_help(harlettizer) {
        Ok(help) => help,
        Err(e) if config.encoder.is_iamf() => {
            return check("iamf", Status::Missing, e.to_string());
        }
        Err(e) => {
            return check(
                "iamf",
                Status::Warning,
                format!("{e} — not needed by the configured encoder"),
            );
        }
    };
    let (has, lacks): (Vec<Encoder>, Vec<Encoder>) = [Encoder::Iamf, Encoder::IamfBed]
        .into_iter()
        .partition(|encoder| encode::iamf_mode(harlettizer, &help, *encoder).is_ok());
    let flags = |encoders: &[Encoder]| {
        encoders
            .iter()
            .map(|encoder| {
                format!(
                    "{} (encoder: {})",
                    encode::voices_flag(*encoder).unwrap_or_default(),
                    encoder.code()
                )
            })
            .collect::<Vec<_>>()
    };
    let detail = match (flags(&has).as_slice(), flags(&lacks).as_slice()) {
        (_, []) => {
            return check(
                "iamf",
                Status::Ok,
                "harlettizer writes IAMF into .mka, the voices panned onto the objects (encoder: \
                 iamf) or on the bed (encoder: iamf-lit)",
            );
        }
        ([], lacking) => format!(
            "{}: its `iamf` has neither {}",
            harlettizer.display(),
            lacking.join(" nor ")
        ),
        (having, lacking) => format!(
            "{}: its `iamf` has no {}, only {}",
            harlettizer.display(),
            lacking.join(", "),
            having.join(", ")
        ),
    };
    if lacks.contains(&config.encoder) {
        check("iamf", Status::Missing, detail)
    } else {
        check(
            "iamf",
            Status::Warning,
            format!("{detail} — not needed by the configured encoder"),
        )
    }
}

/// torch's version and device, with MSST importable, from the venv itself.
fn torch_report(python: &Path, msst: &Path) -> Result<String, String> {
    let mut c = run::command(python);
    c.arg("-c")
        .arg("import sys, torch; sys.path.insert(0, sys.argv[1]); dev = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'; hip = getattr(torch.version, 'hip', None); print(f\"torch {torch.__version__}, device {dev}\" + (f\" (ROCm {hip})\" if hip else ''))")
        .arg(msst)
        .env("PYTHONWARNINGS", "ignore");
    run::output(c).map_err(|e| match e {
        crate::Error::Command { detail, .. } => detail
            .lines()
            .last()
            .unwrap_or("torch does not import")
            .to_string(),
        other => other.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::Tool;
    use std::path::PathBuf;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("omnifrog-doctor-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn harletty_at(path: PathBuf) -> Config {
        let mut config = Config::default();
        config.tools.harletty = Some(path);
        config
    }

    #[test]
    fn a_file_that_cannot_start_is_missing_not_ok() {
        let dir = scratch("start");
        // Text, and on Unix not executable: neither system can start it.
        let path = dir.join(format!("harletty{}", std::env::consts::EXE_SUFFIX));
        std::fs::write(&path, "not a program\n").unwrap();
        let found = tool_check(&harletty_at(path.clone()), Tool::Harletty, true);
        assert_eq!(found.status, Status::Missing, "{}", found.detail);
        assert!(found.detail.contains("cannot start"), "{}", found.detail);
        #[cfg(windows)]
        assert!(
            found.detail.contains("not a Windows program")
                && found.detail.contains("(os error 193)"),
            "{}",
            found.detail
        );
        assert!(
            found.detail.contains(&path.display().to_string()),
            "{}",
            found.detail
        );
        assert!(found.detail.contains("harletty-cli"), "{}", found.detail);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_library_is_refused_without_starting_it() {
        let dir = scratch("library");
        let path = dir.join("harletty_bridge.dll");
        std::fs::write(&path, "").unwrap();
        let config = harletty_at(path);
        let needed = tool_check(&config, Tool::Harletty, true);
        assert_eq!(needed.status, Status::Missing);
        assert!(needed.detail.contains("a library"), "{}", needed.detail);
        let unused = tool_check(&config, Tool::Harletty, false);
        assert_eq!(unused.status, Status::Warning);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_program_that_starts_is_ok_even_when_it_prints_nothing() {
        // This test binary starts, rejects the version flag and prints
        // nothing on its standard output.
        let config = harletty_at(std::env::current_exe().unwrap());
        let found = tool_check(&config, Tool::Harletty, true);
        assert_eq!(found.status, Status::Ok, "{}", found.detail);
    }

    #[test]
    fn a_configured_path_that_is_not_there_is_missing() {
        let config = harletty_at(scratch("absent").join("harletty"));
        let found = tool_check(&config, Tool::Harletty, true);
        assert_eq!(found.status, Status::Missing);
        assert!(found.detail.starts_with("not found"), "{}", found.detail);
    }
}
