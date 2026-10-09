"""Batch analysis with multiprocessing across CPU cores.

Design notes
------------
* A process pool (``spawn``) is used everywhere so Windows/macOS behave like
  Linux and the Qt GUI can host it from a background thread.
* The pool is *terminable*: :func:`run_batch` accepts ``should_stop`` and calls
  :meth:`multiprocessing.pool.Pool.terminate` so a huge library run can be
  cancelled instantly instead of hanging (the Mixed In Key failure mode).
* A per-track ``task_timeout`` guards against a single stuck file: the offending
  job is marked as an error and the pool is restarted for the remaining tracks.
* ``maxtasksperchild`` recycles workers so memory does not creep up over
  thousands of tracks.
* Each worker is pinned to one BLAS/OpenMP thread so N workers never
  oversubscribe the machine.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from .analysis import analyze
from .cue import infer_cues
from .exporters import TrackData

AUDIO_EXTS = (".wav", ".aiff", ".aif", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wv")

# Keep memory bounded on large libraries; users can raise this explicitly.
_MAX_DEFAULT_WORKERS = 8


class _Restart(Exception):
    """Internal: raised to recycle the pool after a task timeout."""


def default_workers() -> int:
    return max(1, min((os.cpu_count() or 2) - 1, _MAX_DEFAULT_WORKERS))


def discover(paths: list[str], recursive: bool = True) -> list[str]:
    """Expand folders into a sorted list of audio file paths."""
    found: list[str] = []
    for p in paths:
        p = os.path.abspath(os.path.expanduser(p))
        if os.path.isdir(p):
            if recursive:
                for root, _dirs, files in os.walk(p):
                    for f in files:
                        if f.lower().endswith(AUDIO_EXTS):
                            found.append(os.path.join(root, f))
            else:
                for f in os.listdir(p):
                    full = os.path.join(p, f)
                    if os.path.isfile(full) and f.lower().endswith(AUDIO_EXTS):
                        found.append(full)
        elif os.path.isfile(p) and p.lower().endswith(AUDIO_EXTS):
            found.append(p)
    return list(dict.fromkeys(found))


@dataclass(eq=False)
class Job:
    path: str
    status: str = "queued"  # queued | analyzing | done | error | cancelled
    error: str = ""
    track: TrackData | None = None

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


@dataclass
class BatchResult:
    jobs: list[Job] = field(default_factory=list)
    cancelled: bool = False

    @property
    def tracks(self) -> list[TrackData]:
        return [j.track for j in self.jobs if j.track is not None]

    @property
    def done(self) -> int:
        return sum(1 for j in self.jobs if j.status in ("done", "error"))


def _init_worker():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    # Lower priority so the GUI stays responsive and audio playback never glitches
    # while a big batch is running. Best-effort: not available on Windows.
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass


def analyze_one(path: str, use_downbeats: bool = True, progress=None) -> TrackData:
    result = analyze(path, progress=progress)
    cues = infer_cues(result, use_downbeats=use_downbeats)
    return TrackData(path=path, analysis=result, cues=cues.cues)


def _worker(args: tuple[str, bool]) -> tuple[str, TrackData | None, str]:
    path, use_downbeats = args
    try:
        return path, analyze_one(path, use_downbeats), ""
    except Exception as exc:  # noqa: BLE001
        return path, None, f"{type(exc).__name__}: {exc}"


def run_batch(
    paths: list[str],
    workers: int | None = None,
    use_downbeats: bool = True,
    on_start=None,
    on_done=None,
    should_stop=None,
    task_timeout: float | None = 600.0,
    on_pool=None,
) -> BatchResult:
    """Analyze *paths* in parallel.

    ``on_start(job)`` / ``on_done(job)`` are called as jobs progress.
    ``should_stop()`` is polled continuously; when true the pool is terminated
    and unfinished jobs are marked ``cancelled``.
    ``task_timeout`` (seconds, per track) marks a stuck file as an error without
    losing the rest of the batch.
    ``on_pool(pool)`` receives the live :class:`multiprocessing.pool.Pool` so it
    can be terminated from another thread (e.g. a GUI Cancel button).
    """
    import multiprocessing

    workers = max(1, workers or default_workers())
    jobs = [Job(path) for path in paths]
    result = BatchResult(jobs)
    if not jobs:
        return result

    ctx = multiprocessing.get_context("spawn")
    remaining = list(jobs)

    def stopped() -> bool:
        return bool(should_stop and should_stop())

    while remaining:
        if stopped():
            break
        pool = ctx.Pool(processes=workers, initializer=_init_worker, maxtasksperchild=25)
        if on_pool:
            on_pool(pool)
        ars: dict = {}
        started: dict[Job, float] = {}
        try:
            for job in remaining:
                job.status = "analyzing"
                if on_start:
                    on_start(job)
                ars[pool.apply_async(_worker, ((job.path, use_downbeats),))] = job
                started[job] = time.time()
            pending = set(ars)
            completed: set[str] = set()
            while pending:
                if stopped():
                    break
                for ar in list(pending):
                    job = ars[ar]
                    if ar.ready():
                        try:
                            _p, track, error = ar.get()
                        except Exception as exc:  # noqa: BLE001
                            track, error = None, f"{type(exc).__name__}: {exc}"
                        _finish(job, track, error, on_done)
                        completed.add(job.path)
                        pending.discard(ar)
                    elif task_timeout and time.time() - started[job] > task_timeout:
                        job.status, job.error = "error", f"timed out after {int(task_timeout)}s"
                        if on_done:
                            on_done(job)
                        completed.add(job.path)
                        pending.discard(ar)
                        raise _Restart()
                if pending:
                    time.sleep(0.05)
            remaining = [j for j in remaining if j.path not in completed]
        except _Restart:
            remaining = [j for j in remaining if j.path not in completed]
        finally:
            if on_pool:
                on_pool(None)
            pool.terminate()
            pool.join()

    if stopped():
        result.cancelled = True
        for j in remaining:
            if j.status in ("queued", "analyzing"):
                j.status = "cancelled"
                if on_done:
                    on_done(j)
    return result


def _finish(job: Job, track: TrackData | None, error: str, on_done):
    if track is not None:
        job.track, job.status, job.error = track, "done", ""
    else:
        job.status, job.error = "error", error
    if on_done:
        on_done(job)
