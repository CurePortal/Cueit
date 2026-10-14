"""Command line interface: analyze files and export cues headlessly."""

from __future__ import annotations

import argparse
import os
import sys

from .batch import default_workers, discover, run_batch
from .libraries import LibrarySource, default_paths, detect_libraries, load_tracks
from . import exporters


def _print_track(t: exporters.TrackData):
    print(f"\n{t.path}")
    print(f"  {t.analysis.bpm:.1f} BPM  |  {t.analysis.key} {t.analysis.scale} "
          f"({t.analysis.camelot})  |  {t.analysis.duration:.1f}s")
    for c in t.cues:
        mm, ss = divmod(c.time, 60)
        print(f"    {c.name:<12} {int(mm):02d}:{ss:05.2f}  {c.color}")


def _warn_if_mixxx_running():
    """Mixxx holds the sqlite DB open; warn if it looks like it is running."""
    import subprocess
    try:
        out = subprocess.run(["pgrep", "-x", "mixxx"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        out = ""
    if out:
        print("\nWARNING: Mixxx appears to be running (pid %s). Close it before "
              "writing to the library, or your changes may be overwritten or the "
              "write may fail.\n" % out, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cueit", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyze files and/or folders")
    a.add_argument("paths", nargs="*", help="audio files or folders")
    a.add_argument("--library", choices=["mixxx", "rekordbox", "traktor"],
                   help="also analyze every track in a detected DJ library")
    a.add_argument("--library-file", help="path to a specific library db/xml/nml")
    a.add_argument("--export", default="json",
                   choices=["json", "csv", "rekordbox", "traktor", "mixxx", "serato", "none"])
    a.add_argument("--out", default=".", help="output directory")
    a.add_argument("--workers", type=int, default=None,
                   help=f"parallel worker processes (default: {default_workers()})")
    a.add_argument("--no-recursive", action="store_true", help="do not descend into subfolders")
    a.add_argument("--limit", type=int, default=None, help="analyze only the first N tracks")
    a.add_argument("--timeout", type=float, default=600.0,
                   help="per-track timeout in seconds (default 600; 0 disables)")
    a.add_argument("--no-downbeats", action="store_true")
    a.add_argument("--mixxx-db", default="~/.mixxx/mixxxdb.sqlite")

    lb = sub.add_parser("libraries", help="list detected DJ libraries and track counts")
    lb.add_argument("paths", nargs="*")

    g = sub.add_parser("gui", help="launch the graphical interface")
    g.add_argument("paths", nargs="*")

    args = parser.parse_args(argv)

    if args.cmd == "gui":
        from .app import main as gui_main
        gui_main(args.paths)
        return 0

    if args.cmd == "libraries":
        libs = detect_libraries()
        if not libs:
            print("no DJ libraries auto-detected")
            return 0
        for lib in libs:
            try:
                n = len(load_tracks(lib))
                print(f"{lib.name:22} {n:>6} tracks   {lib.locator}")
            except Exception as exc:  # noqa: BLE001
                print(f"{lib.name:22}   error: {exc}   {lib.locator}")
        return 0

    os.makedirs(args.out, exist_ok=True)
    all_paths = list(args.paths)
    if args.library or args.library_file:
        if args.library_file:
            source = LibrarySource("Custom", args.library or "mixxx", args.library_file)
        else:
            locator = default_paths().get(args.library)
            if not locator or not os.path.exists(locator):
                print(f"library '{args.library}' not found at {locator}", file=sys.stderr)
                return 2
            source = LibrarySource(args.library.capitalize(), args.library, locator)
        lib_tracks = load_tracks(source)
        print(f"library {source.name}: {len(lib_tracks)} tracks", file=sys.stderr)
        all_paths.extend(lib_tracks)

    paths = discover(all_paths, recursive=not args.no_recursive)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("no audio files found", file=sys.stderr)
        return 2

    workers = args.workers or default_workers()
    total = len(paths)
    print(f"analyzing {total} file(s) with {workers} workers…", file=sys.stderr)

    progress = {"n": 0}
    incremental_mixxx = args.export == "mixxx"
    if incremental_mixxx:
        _warn_if_mixxx_running()
        print("writing cues to the Mixxx library as each track finishes…", file=sys.stderr)

    def on_start(job):
        pass

    def on_done(job):
        progress["n"] += 1
        n = progress["n"]
        name = job.name if len(job.name) < 44 else job.name[:41] + "…"
        if job.track is None:
            if job.status != "cancelled":
                print(f"  [{n:>5}/{total}] FAIL  {name}: {job.error}", file=sys.stderr)
            return
        a = job.track.analysis
        note = ""
        if incremental_mixxx:
            # Write each track to the library immediately so a long run is safe
            # and resumable, and so you can watch it take effect live.
            try:
                exporters.export_mixxx(job.track, db_path=args.mixxx_db)
                note = "-> mixxx"
            except Exception as exc:  # noqa: BLE001
                note = f"-> mixxx FAIL: {exc}"
        print(f"  [{n:>5}/{total}] {a.bpm:6.1f} {a.key} {a.scale:<5} "
              f"{len(job.track.cues)} cues  {name}  {note}".rstrip(), file=sys.stderr)

    result = run_batch(
        paths, workers=workers,
        use_downbeats=not args.no_downbeats,
        on_start=on_start, on_done=on_done,
        task_timeout=(args.timeout or None),
    )
    tracks = result.tracks

    errors = [j for j in result.jobs if j.status == "error"]
    if errors:
        print(f"\n{len(errors)} failure(s):", file=sys.stderr)
        for j in errors[:20]:
            print(f"  {j.name}: {j.error}", file=sys.stderr)

    # Detailed per-track dump only for small runs; for big libraries the
    # progress lines above are enough.
    if total <= 15:
        for t in tracks:
            _print_track(t)

    if args.export == "none" or not tracks:
        print(f"\ndone: {len(tracks)} analysed, {len(errors)} failed.", file=sys.stderr)
        return 0

    if incremental_mixxx:
        print(f"\ndone: {len(tracks)} tracks analysed and written to the Mixxx library "
              f"(backup: {os.path.expanduser(args.mixxx_db)}.cueit.bak).", file=sys.stderr)
        return 0

    if args.export in ("rekordbox", "traktor"):
        out = os.path.join(args.out, "rekordbox.xml" if args.export == "rekordbox" else "collection.nml")
        getattr(exporters, f"export_{args.export}")(tracks, out)
        print(f"\nwrote {out}")
    else:
        for t in tracks:
            base = os.path.splitext(os.path.basename(t.path))[0]
            if args.export == "json":
                out = os.path.join(args.out, base + ".cueit.json")
                exporters.export_json(t, out)
            elif args.export == "csv":
                out = os.path.join(args.out, base + ".cueit.csv")
                exporters.export_csv(t, out)
            elif args.export == "serato":
                out = exporters.export_serato(t)
            print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

