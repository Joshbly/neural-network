"""Start, check and stop a Modal training run without tying it to this laptop.

The job is deployed (modal deploy train/modal_train.py) and started with spawn, so it keeps running if
the laptop sleeps, loses Wi-Fi or the server restarts; a `modal run` client dying would take it down.

    .venv/bin/python train/modal_launch.py start --slot slot-3 --hours 3.5 --helpers 1 [--until 92]   → prints the call id
      (--until: stop once that generation is saved, even if hours remain)
    .venv/bin/python train/modal_launch.py status --call fc-...                          → running | done | failed
    .venv/bin/python train/modal_launch.py stop                                          → stops every machine now
"""
import argparse
import pathlib
import subprocess
import sys
import time

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP, VOLUME = "neural-racers", "neural-racers"
MODAL = str(ROOT / ".venv" / "bin" / "modal")


def start(slot: str, hours: float, helpers: int, until: int = 0):
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
        for name in ("meta.json", "state.json", "summary.json", "ladder.json", "rating.json"):
            if (local / name).exists():
                batch.put_file(local / name, f"/slots/{slot}/{name}")
    call = modal.Function.from_name(APP, "train").spawn(slot, time.time() + hours * 3600, helpers, until)
    print(call.object_id)


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
    parser.add_argument("action", choices=["start", "status", "stop"])
    parser.add_argument("--slot")
    parser.add_argument("--hours", type=float)
    parser.add_argument("--helpers", type=int, default=0)
    parser.add_argument("--until", type=int, default=0)
    parser.add_argument("--call")
    a = parser.parse_args()
    {"start": lambda: start(a.slot, a.hours, a.helpers, a.until), "status": lambda: status(a.call), "stop": stop}[a.action]()
    sys.stdout.flush()
