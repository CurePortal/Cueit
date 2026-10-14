//! Command line interface: analyze files and export cues headlessly.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use clap::{Parser, Subcommand, ValueEnum};

use cueit_core::batch::{self, Job, JobStatus};
use cueit_core::exporters;
use cueit_core::libraries::{self, LibrarySource};
use cueit_core::types::TrackData;

#[derive(Parser)]
#[command(name = "cueit", about = "Automatic cue-point detection for DJ libraries")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum Export {
    Json,
    Csv,
    Rekordbox,
    Traktor,
    Mixxx,
    Serato,
    None,
}

#[derive(Subcommand)]
enum Cmd {
    /// Analyze files and/or folders.
    Analyze {
        /// Audio files or folders.
        paths: Vec<String>,
        /// Also analyze every track in a detected DJ library.
        #[arg(long, value_parser = ["rekordbox", "traktor"])]
        library: Option<String>,
        /// Path to a specific library xml/nml.
        #[arg(long)]
        library_file: Option<String>,
        #[arg(long, value_enum, default_value_t = Export::Json)]
        export: Export,
        /// Output directory.
        #[arg(long, default_value = ".")]
        out: String,
        /// Parallel worker threads (default: min(cpus-1, 8)).
        #[arg(long)]
        workers: Option<usize>,
        /// Do not descend into subfolders.
        #[arg(long)]
        no_recursive: bool,
        /// Analyze only the first N tracks.
        #[arg(long)]
        limit: Option<usize>,
        /// Disable downbeat-based grids.
        #[arg(long)]
        no_downbeats: bool,
        /// Mixxx database path (for --export mixxx).
        #[arg(long, default_value = "~/.mixxx/mixxxdb.sqlite")]
        mixxx_db: String,
        /// Disable the neural beat/key models (use the DSP engine only).
        #[arg(long)]
        no_models: bool,
        /// Print beats/downbeats/sections/cues for debugging.
        #[arg(long)]
        debug: bool,
        /// Cap average CPU usage (percent, 5-100) so the machine stays responsive.
        #[arg(long, default_value_t = 100)]
        cpu_limit: u32,
    },
    /// List detected DJ libraries and track counts.
    Libraries,
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Libraries => cmd_libraries(),
        Cmd::Analyze {
            paths,
            library,
            library_file,
            export,
            out,
            workers,
            no_recursive,
            limit,
            no_downbeats,
            mixxx_db,
            no_models,
            debug,
            cpu_limit,
        } => {
            cueit_core::analysis::set_models_enabled(!no_models);
            cueit_core::batch::set_cpu_limit(cpu_limit);
            cmd_analyze(
                paths,
                library,
                library_file,
                export,
                out,
                workers,
                no_recursive,
                limit,
                !no_downbeats,
                mixxx_db,
                debug,
            )
        }
    }
}

fn cmd_libraries() -> std::process::ExitCode {
    let libs = libraries::detect_libraries();
    if libs.is_empty() {
        println!("no DJ libraries auto-detected");
        return std::process::ExitCode::SUCCESS;
    }
    for lib in libs {
        match libraries::load_tracks(&lib) {
            Ok(tracks) => println!("{:<22} {:>6} tracks   {}", lib.name, tracks.len(), lib.locator),
            Err(e) => println!("{:<22}   error: {e}   {}", lib.name, lib.locator),
        }
    }
    std::process::ExitCode::SUCCESS
}

#[allow(clippy::too_many_arguments)]
fn cmd_analyze(
    paths: Vec<String>,
    library: Option<String>,
    library_file: Option<String>,
    export: Export,
    out: String,
    workers: Option<usize>,
    no_recursive: bool,
    limit: Option<usize>,
    use_downbeats: bool,
    mixxx_db: String,
    debug: bool,
) -> std::process::ExitCode {
    let _ = std::fs::create_dir_all(&out);

    let mut all_paths = paths;
    if let Some(kind) = library.clone().or_else(|| library_file.as_ref().map(|_| "rekordbox".into()))
    {
        let locator = match &library_file {
            Some(f) => f.clone(),
            None => libraries::default_paths()
                .into_iter()
                .find(|(k, _)| k == &kind)
                .map(|(_, v)| v)
                .unwrap_or_default(),
        };
        if locator.is_empty() || !Path::new(&locator).exists() {
            eprintln!("library '{kind}' not found at {locator}");
            return std::process::ExitCode::from(2);
        }
        let source = LibrarySource::new(kind.clone(), kind.clone(), locator);
        match libraries::load_tracks(&source) {
            Ok(tracks) => {
                eprintln!("library {}: {} tracks", source.name, tracks.len());
                all_paths.extend(tracks);
            }
            Err(e) => {
                eprintln!("{e}");
                return std::process::ExitCode::from(2);
            }
        }
    }

    let mut paths = batch::discover(&all_paths, !no_recursive);
    if let Some(n) = limit {
        paths.truncate(n);
    }
    if paths.is_empty() {
        eprintln!("no audio files found");
        return std::process::ExitCode::from(2);
    }

    let workers = workers.unwrap_or_else(batch::default_workers);
    let total = paths.len();
    eprintln!("analyzing {total} file(s) with {workers} threads…");

    let counter = AtomicUsize::new(0);
    let print_lock = Mutex::new(());
    let on_done = |job: &Job| {
        let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
        let _guard = print_lock.lock().unwrap();
        let name = job.name();
        let name = if name.len() > 44 { format!("{}…", &name[..41]) } else { name };
        match &job.track {
            Some(t) => eprintln!(
                "  [{n:>5}/{total}] {:6.1} {} {:<5} {} cues  {name}",
                t.analysis.bpm,
                t.analysis.key,
                t.analysis.scale,
                t.cues.len(),
            ),
            None if job.status == JobStatus::Error => eprintln!("  [{n:>5}/{total}] FAIL  {name}: {}", job.error),
            None => {}
        }
    };

    let result = batch::run_batch(&paths, workers, use_downbeats, Some(&on_done), None);
    let tracks = result.tracks();

    let errors: Vec<&Job> = result.jobs.iter().filter(|j| j.status == JobStatus::Error).collect();
    if !errors.is_empty() {
        eprintln!("\n{} failure(s):", errors.len());
        for j in errors.iter().take(20) {
            eprintln!("  {}: {}", j.name(), j.error);
        }
    }

    if total <= 15 {
        for t in &tracks {
            print_track(t);
            if debug {
                debug_track(t);
            }
        }
    }

    if matches!(export, Export::None) || tracks.is_empty() {
        eprintln!("\ndone: {} analysed, {} failed.", tracks.len(), errors.len());
        return std::process::ExitCode::SUCCESS;
    }

    write_exports(export, &tracks, &out, &mixxx_db);
    std::process::ExitCode::SUCCESS
}

fn write_exports(export: Export, tracks: &[TrackData], out: &str, mixxx_db: &str) {
    match export {
        Export::Rekordbox | Export::Traktor => {
            let name = if matches!(export, Export::Rekordbox) { "rekordbox.xml" } else { "collection.nml" };
            let path = Path::new(out).join(name);
            let res = if matches!(export, Export::Rekordbox) {
                exporters::export_rekordbox(tracks, &path)
            } else {
                exporters::export_traktor(tracks, &path)
            };
            match res {
                Ok(p) => println!("\nwrote {p}"),
                Err(e) => eprintln!("export failed: {e}"),
            }
        }
        Export::Mixxx => {
            for t in tracks {
                match exporters::export_mixxx(t, mixxx_db) {
                    Ok(r) => println!("  -> {r}"),
                    Err(e) => eprintln!("  -> mixxx failed for {}: {e}", t.title()),
                }
            }
        }
        Export::Serato => {
            for t in tracks {
                match exporters::export_serato(t) {
                    Ok(p) => println!("  -> {p}"),
                    Err(e) => eprintln!("  -> serato failed for {}: {e}", t.title()),
                }
            }
        }
        _ => {
            for t in tracks {
                let base = Path::new(&t.path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "track".into());
                let res = match export {
                    Export::Json => {
                        exporters::export_json(t, Path::new(out).join(format!("{base}.cueit.json")))
                    }
                    Export::Csv => {
                        exporters::export_csv(t, Path::new(out).join(format!("{base}.cueit.csv")))
                    }
                    _ => unreachable!(),
                };
                match res {
                    Ok(p) => println!("  -> {p}"),
                    Err(e) => eprintln!("  -> export failed for {base}: {e}"),
                }
            }
        }
    }
}

fn print_track(t: &TrackData) {
    println!("\n{}", t.path);
    println!(
        "  {:.1} BPM  |  {} {} ({})  |  {:.1}s",
        t.analysis.bpm, t.analysis.key, t.analysis.scale, t.analysis.camelot, t.analysis.duration
    );
    for c in &t.cues {
        let mm = (c.time / 60.0).floor();
        let ss = c.time - mm * 60.0;
        println!("    {:<12} {:02}:{:05.2}  {}", c.name, mm as i64, ss, c.color);
    }
}

fn debug_track(t: &TrackData) {
    let a = &t.analysis;
    let n = a.beats.len().min(10);
    eprintln!("  beats[0..{n}]:     {:?}", &a.beats[..n]);
    let d = a.downbeats.len().min(6);
    eprintln!("  downbeats[0..{d}]: {:?}", &a.downbeats[..d]);
    if a.beats.len() > 1 {
        let mut g: Vec<f64> = a.beats.windows(2).map(|w| w[1] - w[0]).collect();
        g.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let med = g[g.len() / 2];
        eprintln!(
            "  beats n={} median_gap={:.4}s -> {:.1} BPM",
            a.beats.len(),
            med,
            if med > 0.0 { 60.0 / med } else { 0.0 }
        );
    }
    eprintln!("  phrase_length {} | downbeat_phase {} | db_conf {:.2}", a.phrase_length, a.downbeat_phase, a.downbeat_confidence);
    for s in &a.sections {
        eprintln!("  section {:<9} {:7.2}-{:7.2}  energy {:.2}", s.kind, s.start, s.end, s.energy);
    }
    for c in &t.cues {
        let nearest = a
            .downbeats
            .iter()
            .cloned()
            .min_by(|x, y| (x - c.time).abs().partial_cmp(&(y - c.time).abs()).unwrap());
        match nearest {
            Some(db) => eprintln!("  cue     {:<9} {:7.2}  nearest downbeat {:7.2}  d={:+.2}", c.name, c.time, db, c.time - db),
            None => eprintln!("  cue     {:<9} {:7.2}", c.name, c.time),
        }
    }
}
