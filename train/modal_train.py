"""Train one save on Modal: the same engine (train/evolve.js), on one or more 64-core machines.

    .venv/bin/modal run --detach train/modal_train.py --slot slot-1 --hours 3.5 --helpers 1

The main machine runs the engine with 79 racing threads; each helper machine adds 80 more over a
token-protected TCP tunnel (train/worker-server.js). The save lives on the Modal volume "neural-racers";
server.js pulls each new generation back so the app shows it live. Each machine costs about $3.15 an hour.
"""
import pathlib
import secrets
import subprocess
import time
import uuid

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORES, WORKERS, HELPER_THREADS = 64, 79, 80  # 80 threads per host: the main machine keeps one for the coordinator

app = modal.App("neural-racers")
volume = modal.Volume.from_name("neural-racers", create_if_missing=True)
helpers_board = modal.Dict.from_name("neural-racers-helpers", create_if_missing=True)
image = (
    modal.Image.from_registry("node:22-slim", add_python="3.12")
    .add_local_dir(ROOT / "js", "/app/js")
    .add_local_dir(ROOT / "train", "/app/train", ignore=["*.py", "__pycache__"])
    .add_local_file(ROOT / "models/originals/tab-field.json", "/app/models/originals/tab-field.json")
)


# A helper lives only while the main machine's heartbeat does: if the engine is interrupted or stopped,
# the helper shuts itself down instead of billing for nothing. No retries: a restarted helper would have a
# new address the engine never learns.
@app.function(image=image, cpu=CORES, memory=24576, timeout=5 * 3600)
def helper(run: str, index: int, token: str, deadline: float):
    server = subprocess.Popen(["node", "/app/train/worker-server.js", "--port", "9000", "--threads", str(HELPER_THREADS), "--token", token])
    with modal.forward(9000, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        helpers_board[f"{run}:{index}"] = f"{host}:{port}"
        born = time.time()
        while server.poll() is None and time.time() < deadline and not helpers_board.get(f"{run}:done", False):
            beat = helpers_board.get(f"{run}:beat", born)
            if time.time() - beat > 180:
                print("no heartbeat from the main machine for 3 minutes; stopping")
                break
            time.sleep(10)
    server.terminate()


@app.function(image=image, cpu=CORES, memory=24576, timeout=5 * 3600, volumes={"/vol": volume},
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def train(slot: str, deadline: float, helpers: int):
    # The main machine starts its own helpers, so they don't depend on the laptop that launched the run.
    # Each attempt (a preempted run is retried) gets a fresh run id and token.
    run, token = uuid.uuid4().hex, secrets.token_hex(16)
    helpers_board[f"{run}:beat"] = time.time()
    calls = [helper.spawn(run, i, token, deadline + 120) for i in range(helpers)]
    try:
        # wait for the helpers to come up (container start plus tunnel), then train with whoever made it
        addresses, waited = [], time.time()
        while len(addresses) < helpers and time.time() - waited < 360:
            addresses = [a for i in range(helpers) if (a := helpers_board.get(f"{run}:{i}"))]
            helpers_board[f"{run}:beat"] = time.time()
            time.sleep(3)
        print(f"helpers connected: {len(addresses)} of {helpers}")
        # a helper that Modal restarts publishes a new address; the engine re-reads this file and reconnects
        remotes_file, seen = pathlib.Path("/tmp/remotes.txt"), set(addresses)
        remotes_file.write_text("\n".join(addresses) + "\n")
        command = ["node", "/app/train/evolve.js", "--dir", f"/vol/slots/{slot}", "--workers", str(WORKERS), "--token", token, "--remote-file", str(remotes_file)]
        if addresses:
            command += ["--remote", ",".join(addresses)]
        engine = subprocess.Popen(command)
        # commit often: the laptop sees each generation, and a preempted run resumes from the last one
        while engine.poll() is None and time.time() < deadline:
            time.sleep(20)
            helpers_board[f"{run}:beat"] = time.time()
            for i in range(helpers):
                address = helpers_board.get(f"{run}:{i}")
                if address and address not in seen:
                    seen.add(address)
                    with remotes_file.open("a") as f:
                        f.write(address + "\n")
                    print(f"helper {i} is back at a new address")
            volume.commit()
        if engine.poll() is None:
            engine.terminate()
        engine.wait()
        volume.commit()
        if time.time() < deadline:
            raise RuntimeError(f"engine exited early with code {engine.returncode}")
    finally:
        helpers_board[f"{run}:done"] = True
        for call in calls:
            call.cancel()


@app.local_entrypoint()
def main(slot: str, hours: float, helpers: int = 0):
    local = ROOT / "models" / "slots" / slot
    # the remote copy mirrors the laptop's save exactly: a from-scratch save uploads only its meta.json
    # (start: scratch + seed), so the engine founds twenty brand-new random brains
    try:
        volume.remove_file(f"/slots/{slot}", recursive=True)
    except (FileNotFoundError, modal.exception.NotFoundError, modal.exception.InvalidError):
        pass  # nothing there yet
    with volume.batch_upload(force=True) as batch:
        for name in ("meta.json", "state.json", "summary.json"):
            if (local / name).exists():
                batch.put_file(local / name, f"/slots/{slot}/{name}")
    train.remote(slot, time.time() + hours * 3600, helpers)
