import json
from pathlib import Path

import pytest

pytest.importorskip("psutil")
from orca_sim.collection_monitor import process_role, summarize, tail, worker_progress


def test_tail_is_bounded_and_missing_log_is_safe(tmp_path):
    path = tmp_path / "log"
    path.write_bytes(b"x" * 100 + b"last")
    assert tail(path, 4) == "last"
    assert tail(tmp_path / "missing") == ""


def test_reads_latest_simulation_progress_and_control_phase(tmp_path):
    (tmp_path / "record.log").write_text(
        "RECORD sim=5.0/69s wall=40.5s frames/camera=100\n"
        "RECORD sim=10.0/69s wall=80.1s frames/camera=201\n"
        " 10.7s | BOTH PICK CUBES | RIGHT: PALM REST\n"
    )
    value = worker_progress(
        tmp_path, dict(attempt=4, seed=14, status="recording", pid=123)
    )
    assert value["sim_seconds"] == 10 and value["frames"] == 201
    assert value["percent"] == pytest.approx(1000 / 69)
    assert value["phase"].endswith("PALM REST")


def test_process_scope_excludes_unrelated_jobs_and_wrapper():
    root = Path("/work/run")
    record = [
        "python",
        "-m",
        "orca_sim.demos.collect_lerobot_smoke",
        "record",
        "--output-dir",
        "/work/run/attempt_000001/recording",
    ]
    assert process_role(record, root) == "Recorder"
    assert process_role(record, Path("/work/ru")) is None
    assert (
        process_role(["python", "-m", "other.module", "--root", str(root)], root)
        is None
    )
    coordinator = [
        "python",
        "-m",
        "orca_sim.demos.collect_randomized",
        "run",
        "--root",
        str(root),
    ]
    assert process_role(coordinator, root) == "Coordinator"
    assert process_role(["caffeinate", "-i", *coordinator], root) is None


def test_progress_counts_only_validated_and_warns_on_dead_coordinator(tmp_path):
    state = dict(
        status="recording",
        target_episodes=200,
        workers=3,
        started_unix=100,
        successful_attempts=[0, 1, 2],
        attempts=[
            dict(attempt=i, seed=i, status="validated" if i < 3 else "packaging")
            for i in range(4)
        ],
    )
    view = summarize(tmp_path, state, [], 400)
    assert view["accepted"] == 3 and view["percent"] == 1.5
    assert view["counts"]["packaging"] == 1 and view["episodes_per_hour"] == 36
    assert view["warnings"] and not view["coordinator_alive"]
    live = summarize(tmp_path, state, [dict(role="Coordinator")], 400)
    assert live["coordinator_alive"] and not live["warnings"]
    assert live["eta_seconds"] == pytest.approx(19700)
    state.update(status="uploaded", finished_unix=350)
    done = summarize(tmp_path, state, [], 1000)
    assert not done["warnings"] and done["elapsed_seconds"] == 250
    assert done["eta_seconds"] is None


def test_startup_has_no_invented_eta(tmp_path):
    view = summarize(tmp_path, dict(status="recording", started_unix=100), [], 150)
    assert view["eta_seconds"] is None and view["episodes_per_hour"] is None
    json.dumps(view, allow_nan=False)


def test_http_serves_only_read_only_loopback_routes():
    from http.client import HTTPConnection
    from http.server import ThreadingHTTPServer
    from threading import Thread
    from types import SimpleNamespace
    from orca_sim.collection_monitor import make_handler

    sampler = SimpleNamespace(read=lambda: dict(ready=True, marker="test"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(sampler))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        for path in ("/", "/dashboard.css", "/dashboard.js", "/api/status"):
            conn.request("GET", path)
            response = conn.getresponse()
            assert response.status == 200
            assert response.getheader("Cache-Control") == "no-store"
            content = response.read()
            if path == "/api/status":
                assert json.loads(content)["marker"] == "test"
        conn.request("GET", "/api/status", headers={"Host": "untrusted.example"})
        response = conn.getresponse()
        assert response.status == 403
        response.read()
        conn.request("POST", "/api/status")
        response = conn.getresponse()
        assert response.status == 501
        response.read()
        conn.request("GET", "/../../status.json")
        response = conn.getresponse()
        assert response.status == 404
        response.read()
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
