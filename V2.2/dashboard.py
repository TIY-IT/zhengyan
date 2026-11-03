"""Local web dashboard for the HITRAN LBL + surrogate + retrieval pipeline.

    python dashboard.py            # opens http://127.0.0.1:8765 in the browser
    python dashboard.py --port 9000 --no-browser

Only the Python standard library is needed.  The server listens on 127.0.0.1
only, every POST needs the per-launch token embedded in the page, and the
pipeline scripts are started with an argument list (never through a shell).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
UI = HERE / "dashboard_ui" / "index.html"
TOKEN = secrets.token_urlsafe(24)

STAGES = [  # id, title, script, relative weight for the overall bar
    ("forward", "① 物理正演（LBL + 两流）", "reproduce_lbl_twostream.py", 5.0),
    ("train", "② 训练替代模型", "train_surrogate.py", 4.0),
    ("invert", "③ 参数反演", "invert_surrogate.py", 3.0),
    ("visualize", "④ 生成图像", "visualize_results.py", 1.0),
]
STAGE_IDS = [s[0] for s in STAGES]

DEFAULTS = {
    "python": sys.executable,
    "output_dir": str(HERE / "outputs"),
    "data_source": "auto",
    "hitemp_root": r"H:\dataset\Hitran\HITEMP",
    "line_file": "",
    "windows": "1200-1400,1900-2300,2800-3200",
    "grid_step": 0.1,
    "ml_samples": 600,
    "device": "auto",
    "gpu_dtype": "float32",
    "skip_gpu_error": True,
    "isotope_scale": "",
    "epochs": 2000,
    "inversion_samples": 0,
    "starts": 5,
    "window_weights": "",
    "stages": STAGE_IDS,
}


class Pipeline:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.reset()

    def reset(self) -> None:
        self.running = False
        self.stop_requested = False
        self.log: list[str] = []
        self.stages = {sid: self._blank(sid, title) for sid, title, _, _ in STAGES}
        self.selected: list[str] = []
        self.config: dict = {}
        self.started = 0.0
        self.finished = 0.0
        self.message = "就绪"

    @staticmethod
    def _blank(sid: str, title: str) -> dict:
        return {"id": sid, "title": title, "status": "idle", "fraction": 0.0,
                "detail": "", "eta": "", "started": 0.0, "elapsed": 0.0, "returncode": None}

    # ------------------------------------------------------------------ commands
    def build_command(self, sid: str, cfg: dict) -> list[str]:
        py, out = cfg["python"], cfg["output_dir"]
        script = str(HERE / dict((s[0], s[2]) for s in STAGES)[sid])
        base = [py, "-u", script]
        if sid == "forward":
            cmd = base + ["--output-dir", out, "--device", cfg["device"], "--windows", cfg["windows"],
                          "--grid-step", str(cfg["grid_step"]), "--ml-samples", str(cfg["ml_samples"]),
                          "--gpu-dtype", cfg["gpu_dtype"], "--data-source", cfg["data_source"]]
            if cfg["data_source"] in ("auto", "hitemp"):
                cmd += ["--hitemp-root", cfg["hitemp_root"]]
            if cfg["data_source"] == "line-file":
                cmd += ["--line-file", cfg["line_file"]]
            if cfg["isotope_scale"]:
                cmd += ["--isotope-scale", cfg["isotope_scale"]]
            cmd += ["--skip-gpu-error"] if cfg["skip_gpu_error"] else ["--gpu-error"]
            return cmd
        if sid == "train":
            return base + ["--input-dir", out, "--device", cfg["device"], "--epochs", str(cfg["epochs"])]
        if sid == "invert":
            cmd = base + ["--input-dir", out, "--device", cfg["device"],
                          "--inversion-samples", str(cfg["inversion_samples"]), "--starts", str(cfg["starts"])]
            if cfg["window_weights"]:
                cmd += ["--window-weights", cfg["window_weights"]]
            return cmd
        return base + ["--input-dir", out]

    # ------------------------------------------------------------------ run loop
    def start(self, cfg: dict) -> None:
        with self.lock:
            if self.running:
                raise RuntimeError("已有任务正在运行")
            self.reset()
            self.config = cfg
            self.selected = [s for s in STAGE_IDS if s in cfg["stages"]]
            for sid in self.selected:
                self.stages[sid]["status"] = "pending"
            self.running = True
            self.started = time.time()
            self.message = "运行中"
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self) -> None:
        with self.lock:
            self.stop_requested = True
            proc = self.proc
        if proc and proc.poll() is None:
            self._kill(proc)

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

    def _run(self) -> None:
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "ZY_PROGRESS": "1"}
        failed = False
        for sid in self.selected:
            if self.stop_requested:
                break
            st = self.stages[sid]
            st.update(status="running", started=time.time(), fraction=0.0, detail="启动中…")
            cmd = self.build_command(sid, self.config)
            self._log(f"$ {' '.join(cmd)}")
            try:
                proc = subprocess.Popen(
                    cmd, cwd=HERE, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                    start_new_session=(os.name != "nt"),
                )
            except OSError as exc:
                st.update(status="failed", detail=f"无法启动：{exc}")
                failed = True
                break
            with self.lock:
                self.proc = proc
            tracker = ProgressTracker(sid, len(self.config["windows"].split(",")) or 1)
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\r\n")
                for part in line.split("\r"):
                    if part.strip() and not part.startswith("@@PROGRESS"):
                        self._log(part)
                    tracker.feed(part, st)
                st["elapsed"] = time.time() - st["started"]
            code = proc.wait()
            st.update(returncode=code, elapsed=time.time() - st["started"])
            if self.stop_requested:
                st.update(status="stopped", detail="已手动停止")
                break
            if code != 0:
                st.update(status="failed", detail=f"退出码 {code}，请查看日志")
                failed = True
                break
            st.update(status="done", fraction=1.0, detail="完成", eta="")
        with self.lock:
            self.proc = None
            self.running = False
            self.finished = time.time()
            for sid in self.selected:
                if self.stages[sid]["status"] == "pending":
                    self.stages[sid]["status"] = "skipped"
            self.message = "已停止" if self.stop_requested else ("失败" if failed else "全部完成")

    def _log(self, text: str) -> None:
        self.log.append(text)
        if len(self.log) > 20000:
            del self.log[:5000]

    # ------------------------------------------------------------------ snapshot
    def snapshot(self, since: int) -> dict:
        weights = {s[0]: s[3] for s in STAGES}
        total = sum(weights[s] for s in self.selected) or 1.0
        done = sum(weights[s] * (1.0 if self.stages[s]["status"] == "done" else self.stages[s]["fraction"])
                   for s in self.selected)
        overall = done / total
        elapsed = (time.time() if self.running else self.finished or time.time()) - self.started if self.started else 0
        eta = ""
        if self.running and 0.02 < overall < 0.999:
            eta = f"{elapsed * (1 - overall) / overall:.0f}"
        return {
            "running": self.running, "message": self.message, "overall": overall,
            "elapsed": elapsed, "eta_seconds": eta,
            "stages": [self.stages[s] | {"selected": s in self.selected} for s in STAGE_IDS],
            "log": self.log[since:], "next": len(self.log),
        }


class ProgressTracker:
    """Turn script output into a 0-1 fraction for one stage."""

    def __init__(self, sid: str, window_count: int) -> None:
        self.sid, self.window_count = sid, max(window_count, 1)
        self.window_index = -1
        self.phase = 0.0

    def feed(self, line: str, st: dict) -> None:
        if self.sid == "forward":
            self._forward(line, st)
        elif self.sid == "train":
            m = re.search(r"核模型 (\d+)/(\d+)", line)
            if m:
                frac = int(m.group(1)) / max(int(m.group(2)), 1)
                st.update(fraction=max(st["fraction"], min(frac, 0.99)), detail=line.strip()[:90])
            elif "FINAL_TEST" in line:
                st.update(fraction=0.99, detail="最终测试集评估")
        elif self.sid == "invert":
            m = re.search(r"retrieval (\d+)/(\d+)", line)
            if m:
                frac = int(m.group(1)) / max(int(m.group(2)), 1)
                st.update(fraction=min(frac, 0.99), detail=f"反演样本 {m.group(1)}/{m.group(2)}")
        else:
            if line.startswith("完成"):
                st.update(fraction=0.99, detail=line.strip()[:90])

    def _forward(self, line: str, st: dict) -> None:
        if line.startswith("=== Running spectral window"):
            self.window_index += 1
            st["detail"] = line.strip("= ").strip()
            return
        if not line.startswith("@@PROGRESS"):
            return
        try:
            _, label, frac_text, detail, eta = line.split("\t")
            frac = float(frac_text)
        except ValueError:
            return
        read_share = 0.15
        if label.startswith("Reading"):
            overall = read_share * frac
        else:
            # Order inside one window: reference atmosphere -> save/plot data -> ML sampling.
            phases = {"Reference atmosphere": (0.0, 0.10), "Saving results": (0.10, 0.15),
                      "ML forward model": (0.15, 1.0)}
            lo, hi = phases.get(label, (0.0, 1.0))
            win = max(self.window_index, 0)
            overall = read_share + (1 - read_share) * (win + lo + (hi - lo) * frac) / self.window_count
        st.update(fraction=max(st["fraction"], min(overall, 0.995)), detail=f"{label}  {detail}", eta=eta)


PIPE = Pipeline()


def clean_config(raw: dict) -> dict:
    """Validate everything coming from the browser (types, ranges, choices)."""
    def choice(key, options):
        if raw.get(key) not in options:
            raise ValueError(f"{key} 取值无效")
        return raw[key]

    def number(key, lo, hi, kind=float):
        try:
            value = kind(raw[key])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{key} 必须是数字") from None
        if not lo <= value <= hi:
            raise ValueError(f"{key} 需在 {lo} 到 {hi} 之间")
        return value

    def text(key, pattern, allow_empty=True):
        value = str(raw.get(key, "")).strip()
        if not value and allow_empty:
            return ""
        if not re.fullmatch(pattern, value):
            raise ValueError(f"{key} 格式不正确")
        return value

    def path(key):
        value = str(raw.get(key, "")).strip()
        if "\x00" in value or value.startswith("-"):
            raise ValueError(f"{key} 路径无效")
        return value

    stages = raw.get("stages", [])
    if not isinstance(stages, list) or not stages or any(s not in STAGE_IDS for s in stages):
        raise ValueError("请至少选择一个有效步骤")
    cfg = {
        "python": path("python") or sys.executable,
        "output_dir": path("output_dir") or DEFAULTS["output_dir"],
        "data_source": choice("data_source", ["auto", "hitemp", "lbl", "line-file"]),
        "hitemp_root": path("hitemp_root"), "line_file": path("line_file"),
        "windows": text("windows", r"\d+(\.\d+)?-\d+(\.\d+)?(,\d+(\.\d+)?-\d+(\.\d+)?)*", allow_empty=False),
        "grid_step": number("grid_step", 0.001, 10.0),
        "ml_samples": number("ml_samples", 0, 1_000_000, int),
        "device": choice("device", ["auto", "cpu", "gpu"]),
        "gpu_dtype": choice("gpu_dtype", ["float32", "float64"]),
        "skip_gpu_error": bool(raw.get("skip_gpu_error", True)),
        "isotope_scale": text("isotope_scale", r"[A-Za-z0-9]+:\d+=[0-9.eE+-]+(,[A-Za-z0-9]+:\d+=[0-9.eE+-]+)*"),
        "epochs": number("epochs", 1, 10_000_000, int),
        "inversion_samples": number("inversion_samples", 0, 1_000_000, int),
        "starts": number("starts", 1, 1000, int),
        "window_weights": text("window_weights", r"[0-9.]+-[0-9.]+=[0-9.eE+-]+(,[0-9.]+-[0-9.]+=[0-9.eE+-]+)*"),
        "stages": stages,
    }
    return cfg


def list_results(output_dir: Path) -> dict:
    out = {"images": [], "metrics": None, "files": []}
    if not output_dir.is_dir():
        return out
    for pattern in ("visualizations/**/*.png", "window_*/*.png", "*.png"):
        for p in sorted(output_dir.glob(pattern)):
            out["images"].append(p.relative_to(output_dir).as_posix())
    metrics = output_dir / "inversion_metrics.json"
    if metrics.is_file():
        try:
            out["metrics"] = json.loads(metrics.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    out["files"] = [p.relative_to(output_dir).as_posix() for p in sorted(output_dir.glob("**/*.csv"))][:80]
    out["images"] = out["images"][:200]
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "ZhengyanDashboard/1.0"

    def log_message(self, *_args) -> None:  # keep the terminal quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._send(403, b"forbidden", "text/plain")
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            html = UI.read_text(encoding="utf-8").replace("__TOKEN__", TOKEN)
            return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        if url.path == "/api/defaults":
            return self._json({**DEFAULTS, **PIPE.config})  # a reloaded page shows the active run's settings
        if url.path == "/api/state":
            since = int(parse_qs(url.query).get("since", ["0"])[0] or 0)
            return self._json(PIPE.snapshot(max(since, 0)))
        if url.path == "/api/results":
            out = Path(PIPE.config.get("output_dir", DEFAULTS["output_dir"]))
            return self._json(list_results(out))
        if url.path.startswith("/files/"):
            base = Path(PIPE.config.get("output_dir", DEFAULTS["output_dir"])).resolve()
            target = (base / unquote(url.path[len("/files/"):])).resolve()
            ok_type = {".png": "image/png", ".csv": "text/csv; charset=utf-8", ".txt": "text/plain; charset=utf-8"}
            if base in target.parents and target.is_file() and target.suffix.lower() in ok_type:
                return self._send(200, target.read_bytes(), ok_type[target.suffix.lower()])
        return self._send(404, b'{"error":"not found"}')

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_ok() or self.headers.get("X-Token") != TOKEN:
            return self._send(403, b'{"error":"forbidden"}')
        length = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(min(length, 1_000_000)) or b"{}")
        except ValueError:
            return self._json({"error": "JSON 无效"}, 400)
        path = urlparse(self.path).path
        try:
            if path == "/api/start":
                PIPE.start(clean_config(data))
            elif path == "/api/stop":
                PIPE.stop()
            else:
                return self._send(404, b'{"error":"not found"}')
        except (ValueError, RuntimeError) as exc:
            return self._json({"error": str(exc)}, 400)
        self._json({"ok": True})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"仪表盘已启动：{url}  （Ctrl+C 退出）", flush=True)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        PIPE.stop()
        print("\n已退出。")


if __name__ == "__main__":
    main()
