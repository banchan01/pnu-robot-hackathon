"""소방 구조 로봇 지휘 콘솔 웹 서버.

표준 라이브러리만 사용하므로 추가 설치가 필요 없다.
실행:  python web/server.py            (기본 포트 8000)
       python web/server.py --port 8080 --ctrl controllers/tb3_mission

컨트롤러와의 연결은 파일 두 개뿐이다.
- 읽기: <ctrl>/output/status.json, live_map.png, live_cam.jpg  (bridge.py 가 기록)
- 쓰기: <ctrl>/commands.txt  (commands.py 가 한 줄씩 읽고 비움)
- 쓰기: <ctrl>/manual.json   (수동 조종 속도 v, w 와 시각. 컨트롤러가 매 tick 읽음)

컨트롤러는 파일에서 첫 줄만 읽고 파일을 비우므로, 서버는 명령을 큐에 쌓아 두고
파일이 비어 있을 때마다 한 줄씩 넘긴다. 모드 한 개가 명령 여러 개로 풀려도 순서가 유지된다.
"""
import argparse
import json
import os
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
MODES_FILE = os.path.join(HERE, "modes.json")


class Console:
    """명령 큐와 모드 정의, 지휘 기록을 관리한다."""

    def __init__(self, ctrl_dir):
        self.ctrl_dir = ctrl_dir
        self.live_pointer = os.path.join(ctrl_dir, "live_output_dir.txt")
        self.command_file = os.path.join(ctrl_dir, "commands.txt")
        self.manual_file = os.path.join(ctrl_dir, "manual.json")
        self.drive_mode = "auto"                 # auto | manual (서버가 마지막으로 보낸 값)
        self.last_manual = {"v": 0.0, "w": 0.0}
        self.queue = deque()
        self.lock = threading.Lock()
        self.command_log = deque(maxlen=100)     # 지휘관이 보낸 명령 기록
        self.current_mode = None
        self._modes_mtime = 0.0
        self._modes = []
        threading.Thread(target=self._pump, daemon=True).start()

    pinned_output = None          # --output 으로 고정한 폴더

    @property
    def output_dir(self):
        """--output 으로 고정했으면 그곳, 아니면 기본 output/ 과 포인터 폴더 중 status.json 이 더 최신인 곳."""
        if self.pinned_output:
            return self.pinned_output
        cands = [os.path.join(self.ctrl_dir, "output")]
        try:
            with open(self.live_pointer, "r", encoding="utf-8") as f:
                d = f.read().strip()
            if d and os.path.isdir(d) and d not in cands:
                cands.append(d)
        except OSError:
            pass
        best, best_m = cands[0], -1.0
        for d in cands:
            try:
                m = os.path.getmtime(os.path.join(d, "status.json"))
            except OSError:
                continue
            if m > best_m:
                best, best_m = d, m
        return best

    # ---------- 모드 ----------
    def modes(self):
        try:
            m = os.path.getmtime(MODES_FILE)
            if m != self._modes_mtime:
                with open(MODES_FILE, "r", encoding="utf-8") as f:
                    self._modes = json.load(f).get("modes", [])
                self._modes_mtime = m
        except (OSError, ValueError):
            pass
        return self._modes

    def set_mode(self, mode_id):
        for m in self.modes():
            if m["id"] == mode_id:
                self.current_mode = mode_id
                for c in m["commands"]:
                    self.enqueue(c, source=f"mode:{mode_id}")
                return m
        return None

    # ---------- 수동 조종 ----------
    def set_manual(self, v, w):
        """속도를 manual.json 에 기록한다. 컨트롤러는 0.7초보다 오래된 값은 무시하므로 계속 갱신해야 한다."""
        v = max(-0.22, min(0.22, float(v)))
        w = max(-2.0, min(2.0, float(w)))
        self.last_manual = {"v": v, "w": w}
        tmp = self.manual_file + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"v": v, "w": w, "ts": time.time()}, f)
            os.replace(tmp, self.manual_file)
        except OSError:
            pass

    def set_drive_mode(self, mode):
        if mode not in ("auto", "manual"):
            return False
        self.set_manual(0.0, 0.0)
        self.drive_mode = mode
        self.enqueue("manual" if mode == "manual" else "auto", source="drive_mode")
        return True

    # ---------- 명령 ----------
    def enqueue(self, command, source="manual"):
        command = " ".join(command.strip().split())
        if not command:
            return
        with self.lock:
            self.queue.append(command)
            self.command_log.append({"wall_time": time.time(), "command": command, "source": source})

    def _pump(self):
        while True:
            time.sleep(0.1)
            with self.lock:
                if not self.queue:
                    continue
                try:
                    if os.path.exists(self.command_file) and os.path.getsize(self.command_file) > 0:
                        continue          # 컨트롤러가 아직 이전 명령을 읽지 않았다
                    cmd = self.queue.popleft()
                    with open(self.command_file, "w", encoding="utf-8") as f:
                        f.write(cmd + "\n")
                except OSError:
                    continue

    # ---------- 상태 ----------
    def status(self):
        path = os.path.join(self.output_dir, "status.json")
        data = {}
        age = None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            age = time.time() - os.path.getmtime(path)
        except (OSError, ValueError):
            pass
        with self.lock:
            pending = list(self.queue)
            log = list(self.command_log)[-30:]
        data["_server"] = {
            "connected": age is not None and age < 3.0,
            "status_age_s": round(age, 1) if age is not None else None,
            "current_mode": self.current_mode,
            "drive_mode": self.drive_mode,
            "last_manual": self.last_manual,
            "top_view": os.path.exists(os.path.join(self.output_dir, "live_top.jpg")),
            "output_dir": self.output_dir,
            "pending_commands": pending,
            "command_log": log,
            "modes": self.modes(),
        }
        return data

    def file(self, name):
        path = os.path.join(self.output_dir, name)
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError:
            return None


CONSOLE = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):      # 요청 로그는 조용히
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=HTTPStatus.OK):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        p = urlparse(self.path).path
        if p in ("/", "/index.html"):
            with open(os.path.join(STATIC, "index.html"), "rb") as f:
                return self._send(HTTPStatus.OK, f.read(), "text/html; charset=utf-8")
        if p == "/api/status":
            return self._json(CONSOLE.status())
        if p == "/api/modes":
            return self._json({"modes": CONSOLE.modes()})
        if p in ("/api/map.png", "/api/camera.jpg", "/api/top.jpg"):
            name = {"/api/map.png": "live_map.png", "/api/camera.jpg": "live_cam.jpg", "/api/top.jpg": "live_top.jpg"}[p]
            body = CONSOLE.file(name)
            if body is None:
                return self._send(HTTPStatus.NOT_FOUND, b"", "text/plain")
            return self._send(HTTPStatus.OK, body, "image/png" if name.endswith("png") else "image/jpeg")
        self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def do_POST(self):
        p = urlparse(self.path).path
        n = int(self.headers.get("Content-Length", "0") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._json({"ok": False, "error": "bad json"}, HTTPStatus.BAD_REQUEST)
        if p == "/api/command":
            cmd = str(body.get("command", ""))
            if not cmd.strip():
                return self._json({"ok": False, "error": "empty"}, HTTPStatus.BAD_REQUEST)
            CONSOLE.enqueue(cmd, source=str(body.get("source", "manual")))
            return self._json({"ok": True, "queued": cmd})
        if p == "/api/manual":
            if CONSOLE.drive_mode != "manual":
                return self._json({"ok": False, "error": "manual mode off"}, HTTPStatus.CONFLICT)
            CONSOLE.set_manual(body.get("v", 0.0), body.get("w", 0.0))
            return self._json({"ok": True})
        if p == "/api/drive_mode":
            if not CONSOLE.set_drive_mode(str(body.get("mode", ""))):
                return self._json({"ok": False, "error": "mode must be auto|manual"}, HTTPStatus.BAD_REQUEST)
            return self._json({"ok": True, "drive_mode": CONSOLE.drive_mode})
        if p == "/api/mode":
            m = CONSOLE.set_mode(str(body.get("mode", "")))
            if m is None:
                return self._json({"ok": False, "error": "unknown mode"}, HTTPStatus.BAD_REQUEST)
            return self._json({"ok": True, "mode": m["id"], "commands": m["commands"]})
        self._json({"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)


def main():
    global CONSOLE
    ap = argparse.ArgumentParser(description="소방 구조 로봇 지휘 콘솔")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ctrl", default=os.path.join(HERE, "..", "controllers", "tb3_mission"),
                    help="tb3_mission 컨트롤러 폴더")
    ap.add_argument("--output", default=None,
                    help="상태 파일 폴더를 고정한다 (기본: 컨트롤러 output/ 과 live_output_dir.txt 중 최신)")
    a = ap.parse_args()
    CONSOLE = Console(os.path.abspath(a.ctrl))
    if a.output:
        CONSOLE.pinned_output = os.path.abspath(a.output)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"지휘 콘솔: http://{a.host}:{a.port}/   (컨트롤러 폴더: {CONSOLE.ctrl_dir})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
