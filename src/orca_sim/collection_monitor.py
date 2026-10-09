"""Read-only loopback dashboard for a randomized collection and host resources.

Run in an interpreter with psutil; no MuJoCo/LeRobot imports or simulator changes.
Hardware is sampled once every two seconds, independently of browser requests.
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import platform
import plistlib
import re
import subprocess
import threading
import time
from urllib.parse import urlsplit

import psutil

ASSETS = Path(__file__).parent / "assets/monitor"
RECORD = re.compile(
    r"^RECORD sim=([\d.]+)/([\d.]+)s wall=([\d.]+)s frames/camera=(\d+)", re.M
)
PHASE = re.compile(r"^\s*[\d.]+s \| (.+)$", re.M)
FINAL = {"uploaded", "complete_local", "stopped"}


def tail(path, limit=32768):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def worker_progress(folder, attempt):
    log = tail(folder / "record.log")
    matches, phases = list(RECORD.finditer(log)), PHASE.findall(log)
    result = dict(
        attempt=attempt["attempt"],
        seed=attempt["seed"],
        status=attempt["status"],
        pid=attempt.get("pid"),
        sim_seconds=0,
        sim_total=69,
        frames=0,
        phase=phases[-1] if phases else "Initializing",
        percent=0,
    )
    if matches:
        sim, total, wall, frames = matches[-1].groups()
        result.update(
            sim_seconds=float(sim),
            sim_total=float(total),
            recording_wall_seconds=float(wall),
            frames=int(frames),
            percent=min(100, float(sim) / float(total) * 100),
        )
    if attempt["status"] in {"awaiting_package", "packaging", "validated"}:
        result["percent"] = 100
        result["sim_seconds"] = attempt.get("outcome", {}).get(
            "duration_s", result["sim_seconds"]
        )
    return result


def gpu_stats():
    if platform.system() != "Darwin":
        return dict(
            available=False, reason="GPU counters are supported on Apple Silicon only"
        )
    try:
        output = subprocess.run(
            ["/usr/sbin/ioreg", "-a", "-r", "-c", "AGXAccelerator"],
            capture_output=True,
            timeout=3,
            check=True,
        )
        for item in plistlib.loads(output.stdout):
            stats = item.get("PerformanceStatistics", {})
            value = stats.get("Device Utilization %")
            if isinstance(value, (int, float)) and 0 <= value <= 100:
                return dict(
                    available=True,
                    percent=value,
                    renderer_percent=stats.get("Renderer Utilization %"),
                    memory_bytes=stats.get("In use system memory"),
                    model=item.get("model", "Apple GPU"),
                    source="macOS IORegistry",
                )
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        plistlib.InvalidFileException,
    ):
        pass
    return dict(available=False, reason="macOS GPU counters are unavailable")


def process_role(command, root):
    """Scope process details to this collection; never expose unrelated commands."""

    def under(argument):
        return argument == str(root) or argument.startswith(str(root) + "/")

    if "-m" not in command or not any(under(arg) for arg in command):
        return None
    module = command[command.index("-m") + 1] if command[-1] != "-m" else ""
    if module == "orca_sim.demos.collect_randomized":
        # caffeinate repeats the child command; only Python owns collection work.
        if "python" not in Path(command[0]).name.lower():
            return None
        return "Coordinator" if "run" in command else "Assembly / upload"
    if module == "orca_sim.demos.collect_lerobot_smoke":
        return "Recorder" if "record" in command else "Encoder / validator"
    return None


def summarize(root, state, processes, now):
    attempts = state.get("attempts", [])
    accepted = len(state.get("successful_attempts", []))
    target = state.get("target_episodes", 200)
    started = state.get("started_unix", now)
    elapsed = max(0, state.get("finished_unix", now) - started)
    counts = {
        name: sum(a.get("status") == name for a in attempts)
        for name in (
            "recording",
            "awaiting_package",
            "packaging",
            "validated",
            "failed",
        )
    }
    rows = [a for a in attempts if a.get("status") == "recording"]
    if not rows:
        rows = attempts[-state.get("workers", 3) :]
    workers = [worker_progress(root / f"attempt_{a['attempt']:06d}", a) for a in rows]
    coordinator_alive = any(p["role"] == "Coordinator" for p in processes)
    status = state.get("status", "unavailable")
    warnings = []
    if status not in FINAL and not coordinator_alive:
        warnings.append(
            "Collection coordinator is not running. Saved progress may be stale."
        )
    if state.get("error"):
        warnings.append(state["error"])
    rate = accepted * 3600 / elapsed if accepted and elapsed else None
    eta = (
        (target - accepted) * 3600 / rate
        if rate and accepted >= 3 and status == "recording"
        else None
    )
    return dict(
        status=status,
        coordinator_alive=coordinator_alive,
        accepted=accepted,
        target=target,
        percent=100 * accepted / target if target else 0,
        elapsed_seconds=elapsed,
        episodes_per_hour=rate,
        eta_seconds=eta,
        attempts=len(attempts),
        counts=counts,
        workers=workers,
        warnings=warnings,
        repo_id=state.get("repo_id"),
        private=state.get("private"),
        root=str(root),
        randomization=state.get("randomization", ""),
        started_unix=started,
    )


class Sampler:
    def __init__(self, root, interval=2):
        self.root = Path(root).resolve()
        self.interval = interval
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.history = deque(maxlen=150)
        self.processes = {}
        self.snapshot = dict(ready=False)
        self.previous_io = None
        self.previous_time = None
        psutil.cpu_percent(percpu=True)

    def sample(self):
        now = time.time()
        cores = psutil.cpu_percent(percpu=True)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(self.root)
        swap = psutil.swap_memory()
        gpu = gpu_stats()
        rows, current = [], {}
        for process in psutil.process_iter(["pid", "cmdline", "name"]):
            try:
                role = process_role(process.info["cmdline"] or [], self.root)
                if not role:
                    continue
                proc = self.processes.get(process.pid, process)
                cpu = proc.cpu_percent()
                current[process.pid] = proc
                rows.append(
                    dict(
                        pid=process.pid,
                        role=role,
                        cpu_percent=cpu,
                        rss_bytes=proc.memory_info().rss,
                        elapsed_seconds=now - proc.create_time(),
                    )
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        self.processes = current
        io = psutil.disk_io_counters()
        rates = dict(read_mbps=None, write_mbps=None)
        if io and self.previous_io and self.previous_time and now > self.previous_time:
            dt = now - self.previous_time
            rates = dict(
                read_mbps=max(0, io.read_bytes - self.previous_io.read_bytes)
                / dt
                / 1e6,
                write_mbps=max(0, io.write_bytes - self.previous_io.write_bytes)
                / dt
                / 1e6,
            )
        self.previous_io, self.previous_time = io, now
        cpu = sum(cores) / len(cores) if cores else 0
        self.history.append(
            dict(time=now, cpu=cpu, gpu=gpu.get("percent"), memory=memory.percent)
        )
        hardware = dict(
            cpu_percent=cpu,
            cores=cores,
            logical_cores=len(cores),
            physical_cores=psutil.cpu_count(logical=False),
            memory_percent=memory.percent,
            memory_used_bytes=memory.total - memory.available,
            memory_total_bytes=memory.total,
            memory_available_bytes=memory.available,
            swap_used_bytes=swap.used,
            disk_total_bytes=disk.total,
            disk_free_bytes=disk.free,
            disk_used_percent=disk.percent,
            gpu=gpu,
            disk_io=rates,
            load_average=list(psutil.getloadavg()),
            job_cpu_percent=sum(p["cpu_percent"] for p in rows) / max(1, len(cores)),
            job_rss_bytes=sum(p["rss_bytes"] for p in rows),
        )
        try:
            path = self.root / "status.json"
            state = json.loads(path.read_text())
            collection = summarize(self.root, state, rows, now)
            collection["status_age_seconds"] = max(0, now - path.stat().st_mtime)
            if (
                collection["status"] == "recording"
                and collection["status_age_seconds"] > 90
            ):
                collection["warnings"].append(
                    "Progress heartbeat is over 90 seconds old."
                )
        except (OSError, ValueError, KeyError) as error:
            collection = dict(
                status="unavailable", warnings=[f"Cannot read progress: {error}"]
            )
        if disk.free < 12 * 1024**3:
            collection["warnings"].append(
                "Less than 12 GiB free; the collector's disk guard may stop the run."
            )
        events = [
            line
            for line in tail(self.root.with_suffix(".log")).splitlines()
            if line.startswith(
                ("START ", "ACCEPTED ", "FAILED ", "RuntimeError", "Traceback")
            )
        ][-8:]
        with self.lock:
            self.snapshot = dict(
                ready=True,
                sampled_unix=now,
                interval_seconds=self.interval,
                hardware=hardware,
                collection=collection,
                processes=rows,
                history=list(self.history),
                events=events,
            )

    def run(self):
        while not self.stop.is_set():
            try:
                self.sample()
            except Exception as error:
                with self.lock:
                    self.snapshot = dict(
                        ready=False, error=str(error), sampled_unix=time.time()
                    )
            self.stop.wait(self.interval)

    def read(self):
        with self.lock:
            return copy.deepcopy(self.snapshot)


def make_handler(sampler):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            host = self.headers.get("Host", "").split(":", 1)[0]
            if host not in {"127.0.0.1", "localhost"}:
                self.send_error(403)
                return
            path = urlsplit(self.path).path
            if path == "/api/status":
                content = json.dumps(sampler.read(), allow_nan=False).encode()
                kind = "application/json"
            elif path in {"/", "/dashboard.css", "/dashboard.js"}:
                name = "dashboard.html" if path == "/" else path[1:]
                content = (ASSETS / name).read_bytes()
                kind = {
                    "html": "text/html",
                    "css": "text/css",
                    "js": "text/javascript",
                }[name.rsplit(".", 1)[-1]]
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind + "; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'self'",
            )
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    if not (args.root / "status.json").is_file():
        parser.error("--root must contain the collection's status.json")
    sampler = Sampler(args.root)
    thread = threading.Thread(target=sampler.run, daemon=True)
    thread.start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(sampler))
    print(f"DASHBOARD http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        sampler.stop.set()
        server.server_close()


if __name__ == "__main__":
    main()
