//! HTTP download of one large file: streamed to `<dest>.part`, renamed once
//! complete, so an interruption never leaves a truncated checkpoint that
//! looks whole. No token, no authentication header: the catalogue only names
//! public repositories, which also keeps an expired Hugging Face token from
//! failing a public download the way `huggingface_hub` does.

use crate::progress::{Event, Progress};
use crate::{Error, Result};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

const BLOCK: usize = 1 << 20;

/// Fetch `url` into `dest`, reporting bytes as they land.
pub fn fetch(url: &str, dest: &Path, progress: Progress) -> Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    let part = dest.with_extension(match dest.extension().and_then(|e| e.to_str()) {
        Some(ext) => format!("{ext}.part"),
        None => "part".to_string(),
    });
    let result = stream(url, dest, &part, progress);
    if result.is_err() {
        let _ = fs::remove_file(&part);
    }
    result
}

fn stream(url: &str, dest: &Path, part: &Path, progress: Progress) -> Result<()> {
    let response = ureq::get(url)
        .timeout(std::time::Duration::from_secs(60 * 60 * 6))
        .call()
        .map_err(|e| Error::Download {
            url: url.to_string(),
            detail: match e {
                ureq::Error::Status(code, _) => format!("HTTP {code}"),
                ureq::Error::Transport(t) => t.to_string(),
            },
        })?;
    let total = response
        .header("content-length")
        .and_then(|v| v.parse::<u64>().ok());
    let what = dest
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("download")
        .to_string();
    let mut reader = response.into_reader();
    let mut out = File::create(part).map_err(|e| Error::io(part, e))?;
    let mut buffer = vec![0u8; BLOCK];
    let mut done = 0u64;
    progress(Event::bytes(what.clone(), done, total));
    loop {
        let read = reader.read(&mut buffer).map_err(|e| Error::Download {
            url: url.to_string(),
            detail: e.to_string(),
        })?;
        if read == 0 {
            break;
        }
        out.write_all(&buffer[..read])
            .map_err(|e| Error::io(part, e))?;
        done += read as u64;
        progress(Event::bytes(what.clone(), done, total));
    }
    out.flush().map_err(|e| Error::io(part, e))?;
    drop(out);
    if let Some(total) = total
        && total != done
    {
        return Err(Error::Download {
            url: url.to_string(),
            detail: format!("{done} bytes received, {total} announced"),
        });
    }
    fs::rename(part, dest).map_err(|e| Error::io(dest, e))?;
    Ok(())
}
