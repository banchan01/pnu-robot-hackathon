"""외부 명령 수신 (담당 D). 사람과의 소통 창구.

두 경로로 명령을 받는다.
1. 텍스트 파일 commands.txt (컨트롤러 폴더). 한 줄을 읽으면 파일을 비운다.
   예)  stop | resume | return | target red 2 | status
2. Webots 3D 뷰 키보드:  P 정지, R 재개, H 즉시 복귀, S 상태 출력

명령 상태(command_state): run | stopped | return | manual
   manual 상태에서는 manual.json 의 (v, w) 를 안전 계층을 거쳐 그대로 바퀴에 보낸다.
"""
import json
import os
import time

from config import COMMAND_FILE, CTRL_DIR

MANUAL_FILE = os.path.join(CTRL_DIR, "manual.json")     # 웹 수동 조종 속도 {"v","w","ts"}
MANUAL_STALE_S = 0.7                                     # 이보다 오래된 속도는 무시 (데드맨)


class CommandSource:
    def __init__(self, keyboard, log):
        self.keyboard = keyboard
        self.log = log
        self.state = "run"
        self.pending_target = None
        self.want_status = False
        # 시작 시 남아 있는 옛 명령은 버린다
        try:
            open(COMMAND_FILE, "w").close()
        except OSError:
            pass

    def _read_file(self):
        try:
            if not os.path.exists(COMMAND_FILE) or os.path.getsize(COMMAND_FILE) == 0:
                return None
            with open(COMMAND_FILE, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            open(COMMAND_FILE, "w").close()
            return lines[0] if lines else None
        except OSError:
            return None

    def _read_key(self):
        if self.keyboard is None:
            return None
        k = self.keyboard.getKey()
        if k == -1:
            return None
        ch = chr(k).upper() if 0 <= k < 256 else ""
        return {"P": "stop", "R": "resume", "H": "return", "S": "status"}.get(ch)

    def poll(self):
        cmd = self._read_file() or self._read_key()
        if not cmd:
            return None
        parts = cmd.lower().split()
        head = parts[0]
        if head == "stop":
            self.state = "stopped"
            self.log("[명령] 정지")
        elif head == "resume":
            if self.state in ("stopped", "return"):   # 정지·긴급복귀 뒤에도 재개 가능
                self.state = "run"
            self.log("[명령] 재개")
        elif head == "return":
            self.state = "return"
            self.log("[명령] 즉시 복귀")
        elif head == "manual":
            self.state = "manual"
            self.log("[명령] 수동 조종 모드")
        elif head == "auto":
            self.state = "run"
            self.log("[명령] 자동 임무 모드")
        elif head == "target" and len(parts) >= 2:
            color = parts[1]
            count = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else None
            self.pending_target = (color, count)
            self.log(f"[명령] 목표 변경 → {color} {count if count else ''}")
        elif head == "status":
            self.want_status = True
        else:
            self.log(f"[명령] 알 수 없는 명령: {cmd}")
        return {"cmd": head, "arg": " ".join(parts[1:])}

    def manual_cmd(self):
        """수동 조종 속도 (v m/s, w rad/s). 파일이 없거나 오래되면 정지."""
        try:
            with open(MANUAL_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
            if time.time() - float(d.get("ts", 0.0)) > MANUAL_STALE_S:
                return 0.0, 0.0
            return float(d.get("v", 0.0)), float(d.get("w", 0.0))
        except (OSError, ValueError, TypeError):
            return 0.0, 0.0
