"""Start, check and stop a Modal training run without tying it to this laptop.

The job is deployed (modal deploy train/modal_train.py) and started with spawn, so it keeps running if
the laptop sleeps, loses Wi-Fi or the server restarts; a `modal run` client dying would take it down.

    .venv/bin/python train/modal_launch.py start --slot slot-3 --hours 3.5 --helpers 1 [--until 92]   → prints the call id
      (--until: stop once that generation is saved, even if hours remain)
    .venv/bin/python train/modal_launch.py start --slot slot-7 --hours 3 --kind ppo [--until 500]     → gradient learning
      (one GPU machine; --until: stop after that many learning iterations; resumes from the save's ppo/checkpoint.pt)
    .venv/bin/python train/modal_launch.py status --call fc-...                          → running | done | failed
    .venv/bin/python train/modal_launch.py stop                                          → stops every machine now
    .venv/bin/python train/modal_launch.py pilot [--hours 0.25]                          → gradient learning's speed
      pilot: 4 runs side by side, waits for them, downloads them to models/ppo-pilot/ (train/ppo/pilot-report.js)
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP, VOLUME = "neural-racers", "neural-racers"
MODAL = str(ROOT / ".venv" / "bin" / "modal")


# what a run needs from the laptop's save: evolution's state and ladder, or gradient learning's checkpoint and metrics
# (the checkpoint carries the save's history; state and summary keep the remote copy whole until the next generation)
UPLOADS = {
    "train": ("meta.json", "state.json", "summary.json", "ladder.json", "rating.json"),
    "ppo": ("meta.json", "start.json", "config.json", "state.json", "summary.json", "ppo/checkpoint.pt", "ppo/metrics.jsonl"),
}


def start(slot: str, hours: float, helpers: int, until: int = 0, kind: str = "train"):
    subprocess.run([MODAL, "deploy", str(ROOT / "train" / "modal_train.py")], check=True, capture_output=True)
    volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
    local = ROOT / "models" / "slots" / slot
    # the remote copy mirrors the laptop's save exactly: a from-scratch save uploads only its meta.json
    # (start: scratch + seed), so the engine founds twenty brand-new random brains
    try:
        volume.remove_file(f"/slots/{slot}", recursive=True)
    except (FileNotFoundError, modal.exception.NotFoundError, modal.exception.InvalidError):
        pass  # nothing there yet
    with volume.batch_upload(force=True) as batch:
        for name in UPLOADS[kind]:
            if (local / name).exists():
                batch.put_file(local / name, f"/slots/{slot}/{name}")
    deadline = time.time() + hours * 3600
    if kind == "ppo":
        call = modal.Function.from_name(APP, "learn").spawn(slot, deadline, until)
    else:
        call = modal.Function.from_name(APP, "train").spawn(slot, deadline, helpers, until)
    print(call.object_id)


# the learning-speed pilot: the fixed learner (critic warm-up, the rate capped, noise held) from the same starting
# weights, 150 iterations of the real batch, a noise-free evaluation every 10, each machine capped at --hours.
# What it compared (run 2026-10-04, models/ppo-pilot): the defaults then (held 8, steer 0.01) against one change each:
# a looser limit on each update, a faster learning rate (start and ceiling), and steering noise held half as long but
# larger. Held 4 won and is the default now, so a rerun's "defaults" is that and its "hold4" the same again
PILOTS = {
    "speed": {
        "pilot-defaults": ["--hold", "8", "--sigma0", "0.01,0.2"],
        "pilot-looser": ["--hold", "8", "--sigma0", "0.01,0.2", "--kl-target", "0.03", "--clip", "0.3"],
        "pilot-lr": ["--hold", "8", "--sigma0", "0.01,0.2", "--lr", "1e-3"],
        "pilot-hold4": ["--hold", "4", "--sigma0", "0.015,0.2"],
    },
    # from scratch at race distance, seeing the whole race: the wall-rider gets parked minutes after the scraping that
    # does it, beyond gamma 0.999's 33 seconds. A: 2.8 minutes ahead; B: 5.5 minutes and more of the real outcome in
    # each advantage; C: A plus twice the steering exploration, to find the way off the wall
    "foresight": {
        "fs-a": ["--gamma", "0.9998", "--lam", "0.95"],
        "fs-b": ["--gamma", "0.9999", "--lam", "0.97"],
        "fs-c": ["--gamma", "0.9998", "--lam", "0.95", "--sigma0", "0.03,0.1"],
    },
}
PPO_PER_HOUR = 1.81


def pilot(hours: float, set_name: str = "speed", iterations: int = 150, every: int = 10):
    PILOT = PILOTS[set_name]
    subprocess.run([MODAL, "deploy", str(ROOT / "train" / "modal_train.py")], check=True, capture_output=True)
    volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
    for name in PILOT:
        try:
            volume.remove_file(f"/slots/{name}", recursive=True)
        except (FileNotFoundError, modal.exception.NotFoundError, modal.exception.InvalidError):
            pass
    learn, t0 = modal.Function.from_name(APP, "learn"), time.time()
    calls = {name: learn.spawn(name, t0 + hours * 3600, iterations, [*flags, "--generation-every", str(every)]) for name, flags in PILOT.items()}
    print(f"pilot {set_name}: {len(calls)} runs started ({', '.join(c.object_id for c in calls.values())})", flush=True)
    local, log = ROOT / "models" / "ppo-pilot" / ("" if set_name == "speed" else set_name), {}
    local.mkdir(parents=True, exist_ok=True)
    for name, call in calls.items():
        try:
            call.get(timeout=hours * 3600 + 900)
            outcome = "done"
        except Exception as error:  # out of retries or past the cap: whatever it saved still comes home
            outcome = f"failed: {type(error).__name__}"
        log[name] = dict(flags=PILOT[name], call=call.object_id, outcome=outcome, minutes=round((time.time() - t0) / 60, 2))
        print(f"  {name} ({' '.join(PILOT[name]) or 'defaults'}): {outcome} after {log[name]['minutes']} min", flush=True)
    for name in calls:
        subprocess.run([MODAL, "volume", "get", "--force", VOLUME, f"/slots/{name}", str(local)], check=False, capture_output=True)
    # what each machine cost: billed by the second from start to finish, at most this (the last finisher's time for all)
    minutes = max(entry["minutes"] for entry in log.values())
    (local / "pilot.json").write_text(json.dumps(dict(started=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)), hours_cap=hours, runs=log,
                                                      cost_at_most=round(len(log) * minutes / 60 * PPO_PER_HOUR, 2)), indent=1))
    print(f"pilot finished in {minutes:.1f} min; at most ${len(log) * minutes / 60 * PPO_PER_HOUR:.2f}; saved in {local}")


def status(call_id: str):
    try:
        modal.FunctionCall.from_id(call_id).get(timeout=0)
        print("done")
    except TimeoutError:
        print("running")
    except (ConnectionError, OSError) as error:  # this laptop can't reach Modal: say so, don't call it over
        print(f"unknown: {type(error).__name__}")
    except Exception as error:  # cancelled, crashed, or out of retries
        print(f"failed: {type(error).__name__}")


def stop():
    # stopping the app terminates every container at once, helpers included; the next start redeploys
    subprocess.run([MODAL, "app", "stop", "--yes", APP], check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "status", "stop", "pilot"])
    parser.add_argument("--slot")
    parser.add_argument("--hours", type=float, default=0.25)
    parser.add_argument("--helpers", type=int, default=0)
    parser.add_argument("--until", type=int, default=0)
    parser.add_argument("--call")
    parser.add_argument("--kind", choices=["train", "ppo"], default="train")
    parser.add_argument("--set", default="speed", choices=list(PILOTS))
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--every", type=int, default=10)
    a = parser.parse_args()
    {"start": lambda: start(a.slot, a.hours, a.helpers, a.until, a.kind), "status": lambda: status(a.call), "stop": stop,
     "pilot": lambda: pilot(a.hours, a.set, a.iterations, a.every)}[a.action]()
    sys.stdout.flush()
