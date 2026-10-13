//! The omnifrog window.
//!
//! One program, two lives: started plainly it is the window; started with
//! [`task::WORKER_FLAG`] it is the worker the window hands the long work
//! to (see [`task`]). Everything quick — the configuration, the checks, the
//! device, a film's tracks — the window asks `omnifrog-core` for directly.

// No console window behind the application on Windows.
#![cfg_attr(windows, windows_subsystem = "windows")]

mod app;
mod child;
mod run;
mod task;
mod view_film;
mod view_setup;

fn main() -> std::process::ExitCode {
    let first = std::env::args_os().nth(1);
    if first.as_deref() == Some(std::ffi::OsStr::new(task::WORKER_FLAG)) {
        return task::serve();
    }
    // `omnifrog-gui FILM.mkv`: the film opened at once, as a file
    // manager's « Open with » starts it.
    let film = first.map(std::path::PathBuf::from).filter(|p| p.is_file());
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 780.0])
            .with_min_inner_size([760.0, 560.0])
            .with_title("omnifrog")
            .with_drag_and_drop(true),
        ..Default::default()
    };
    match eframe::run_native(
        "omnifrog",
        options,
        Box::new(|cc| Ok(Box::new(app::App::new(&cc.egui_ctx, film)))),
    ) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("omnifrog: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
