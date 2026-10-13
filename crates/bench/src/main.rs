//! The omnifrog test bench.
//!
//! Started plainly it is the window; started with a subcommand it does the
//! same work without one — for a night of computing over SSH, and for the
//! tests. Both call the same functions of `omnifrog-bench`.

mod app;
mod cli;
mod listen;
mod output;
mod prefs;
#[cfg(test)]
mod tests;
mod view_cascades;
mod view_film;
mod view_model;
mod view_rate;
mod view_report;
mod view_zones;

use clap::Parser;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args = cli::Args::parse();
    let result = cli::bench_dir(args.dir.as_deref()).and_then(|dir| match args.command {
        Some(command) => cli::run(command, &dir),
        None => window(dir, args.tab),
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("omnifrog-bench: {error}");
            ExitCode::FAILURE
        }
    }
}

fn window(dir: std::path::PathBuf, tab: cli::TabArg) -> Result<(), String> {
    let tab = match tab {
        cli::TabArg::Rate => app::Tab::Rate,
        cli::TabArg::Report => app::Tab::Report,
        cli::TabArg::Model => app::Tab::Model,
        cli::TabArg::Film => app::Tab::Film,
        cli::TabArg::Cascades => app::Tab::Cascades,
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1500.0, 900.0])
            .with_min_inner_size([900.0, 560.0])
            .with_title(app::TITLE),
        ..Default::default()
    };
    eframe::run_native(
        app::TITLE,
        options,
        Box::new(move |cc| {
            let mut app = app::App::new(&cc.egui_ctx, dir, tab);
            app.remember_in(prefs::Prefs::path());
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| e.to_string())
}
