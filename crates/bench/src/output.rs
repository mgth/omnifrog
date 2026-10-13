//! The audio device the player plays through.
//!
//! Opened the first time something is played, never before: a bench left
//! open in a corner holds no device. 48 kHz is asked for, the rate of every
//! file here; a device that cannot do it is fed through the player's own
//! interpolation.
//!
//! The device is the one chosen in the window, or the machine's default
//! when none is. On Linux the outputs are PipeWire's when it runs, and
//! ALSA's otherwise. ALSA's default is whatever its configuration names,
//! which may be a card that is not plugged in: where it does not open, the
//! sound server is asked for by name.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use omnifrog_bench::player::Player;
use omnifrog_bench::set::RATE;
use serde::{Deserialize, Serialize};

pub struct Output {
    /// Dropped with the window: the stream stops with it.
    _stream: cpal::Stream,
    pub name: String,
    pub rate: u32,
}

/// An output as the list shows it and as the choice is kept: the name is
/// what is read, the id what it is found again by.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Choice {
    pub id: String,
    pub name: String,
}

impl Choice {
    /// Whether both are the same output. The id says so: the name is what
    /// the system calls it today, and that may change.
    pub fn is(&self, other: &Choice) -> bool {
        self.id == other.id
    }
}

/// The outputs there are to choose from, in the order the system gives
/// them. Nothing is opened.
pub fn choices() -> Result<Vec<Choice>, String> {
    let devices = cpal::default_host()
        .output_devices()
        .map_err(|e| e.to_string())?;
    let mut choices: Vec<Choice> = Vec::new();
    for device in devices {
        let (Ok(id), Ok(description)) = (device.id(), device.description()) else {
            continue;
        };
        let id = id.to_string();
        if is_output(&id, description.direction()) && choices.iter().all(|c| c.id != id) {
            choices.push(Choice {
                id,
                name: description.name().to_string(),
            });
        }
    }
    Ok(choices)
}

/// PipeWire lists, beside the outputs, the programs that are playing and
/// its own default under two more names. An output is what it gives as
/// taking sound and giving it back; the default is the choice of none.
fn is_output(id: &str, direction: cpal::DeviceDirection) -> bool {
    match id.strip_prefix("pipewire:") {
        Some(node) => direction == cpal::DeviceDirection::Duplex && node != "sink_default",
        None => true,
    }
}

/// The sound servers' devices, as ALSA names them, in the order they are
/// tried after the default one. No other host has devices of these names.
const SERVERS: [&str; 2] = ["pipewire", "pulse"];

/// Open the output chosen, or the default one. An output that was chosen
/// and is not there is an error, not a reason to play somewhere else: what
/// is plugged in its place may be turned up for something other than this.
pub fn open(player: &Player, chosen: Option<&Choice>) -> Result<Output, String> {
    let host = cpal::default_host();
    if let Some(choice) = chosen {
        let device = host
            .output_devices()
            .map_err(|e| e.to_string())?
            .find(|device| device.id().is_ok_and(|id| id.to_string() == choice.id))
            .ok_or_else(|| format!("{} is not there", choice.name))?;
        return open_on(&device, player);
    }
    let default = host
        .default_output_device()
        .ok_or("no audio output device")?;
    // The default device's failure is the one reported: it is the one the
    // machine is set up to use.
    let failure = match open_on(&default, player) {
        Ok(output) => return Ok(output),
        Err(e) => e,
    };
    SERVERS
        .iter()
        .filter_map(|server| named(&host, server))
        .find_map(|device| open_on(&device, player).ok())
        .ok_or(failure)
}

fn named(host: &cpal::Host, name: &str) -> Option<cpal::Device> {
    host.output_devices()
        .ok()?
        .find(|device| device.id().is_ok_and(|id| id.id() == name))
}

fn open_on(device: &cpal::Device, player: &Player) -> Result<Output, String> {
    let name = device
        .description()
        .map_or_else(|_| "audio output".to_string(), |d| d.name().to_string());
    let supported = device
        .supported_output_configs()
        .map_err(|e| format!("{name}: {e}"))?
        .filter(|c| c.sample_format() == cpal::SampleFormat::F32)
        .filter(|c| (c.min_sample_rate()..=c.max_sample_rate()).contains(&RATE))
        // Stereo if there is a choice: the track goes to every channel.
        .min_by_key(|c| (c.channels() as i32 - 2).abs())
        .map(|c| c.with_sample_rate(RATE));
    let supported = match supported {
        Some(config) => config,
        None => device
            .default_output_config()
            .map_err(|e| format!("{name}: {e}"))?,
    };
    if supported.sample_format() != cpal::SampleFormat::F32 {
        return Err(format!(
            "{name}: no float output ({:?} only)",
            supported.sample_format()
        ));
    }
    let config = supported.config();
    let (channels, rate) = (config.channels as usize, config.sample_rate);
    let mut source = player.source();
    let stream = device
        .build_output_stream(
            config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| source.fill(data, channels, rate),
            |error| eprintln!("omnifrog-bench: audio: {error}"),
            None,
        )
        .map_err(|e| format!("{name}: {e}"))?;
    stream.play().map_err(|e| format!("{name}: {e}"))?;
    Ok(Output {
        _stream: stream,
        name,
        rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// For a machine that plays sound, which the ones the tests run on
    /// usually are not: `cargo test -p omnifrog-bench -- --ignored`.
    /// Nothing is heard, the player is left paused.
    #[test]
    #[ignore = "opens the machine's audio output"]
    fn the_outputs_of_this_machine_open() {
        let player = Player::new();
        let output = open(&player, None).expect("the default output");
        println!("default: {} at {} Hz", output.name, output.rate);
        drop(output);
        for choice in choices().expect("the outputs") {
            match open(&player, Some(&choice)) {
                Ok(output) => println!("{}: {} at {} Hz", choice.id, output.name, output.rate),
                Err(e) => println!("{}: {e}", choice.id),
            }
        }
    }

    #[test]
    fn a_program_playing_is_not_an_output() {
        use cpal::DeviceDirection::{Duplex, Output};
        assert!(is_output("pipewire:alsa_output.usb", Duplex));
        assert!(!is_output("pipewire:some-player", Output));
        assert!(!is_output("pipewire:sink_default", Duplex));
        assert!(!is_output("pipewire:output_default", Output));
        assert!(is_output("alsa:hdmi:CARD=0,DEV=0", Output));
        assert!(is_output("wasapi:{0.0.0.00000000}", Output));
    }
}
