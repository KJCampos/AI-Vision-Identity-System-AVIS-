"""
Actions on recognition.

- auto_lock: locks Windows when the owner (you) hasn't been seen for N seconds.
  It only arms after you've been recognized once this session, so it never locks
  you out on startup. Windows doesn't let apps unlock, so locking is the safe direction.
- on_recognized_command: runs a command when someone is recognized,
  e.g. 'start spotify:'. {name} is replaced with who was seen. Has a cooldown.
"""
import ctypes
import logging
import subprocess
import sys
import time


class Presence:
    def __init__(self, owner="", auto_lock=False, lock_after_s=60, command="", command_cooldown_s=300, events=None):
        self.owner = (owner or "").strip()
        self.auto_lock = bool(auto_lock) and bool(self.owner)
        self.lock_after = float(lock_after_s)
        self.command = (command or "").strip()
        self.cmd_cooldown = float(command_cooldown_s)
        self.events = events
        self.last_seen = None
        self.locked = False
        self._last_cmd = {}
        if bool(auto_lock) and not self.owner:
            logging.warning("[ACTION] auto_lock needs 'owner' set to your enrolled name in config.yaml; disabled.")
        if self.auto_lock:
            logging.info("[ACTION] auto-lock armed: locks after %.0fs without seeing %s", self.lock_after, self.owner)

    def update(self, names_seen, now=None):
        """names_seen: enrolled names recognized as live this frame."""
        now = time.time() if now is None else now
        if self.owner and self.owner in names_seen:
            if self.last_seen is None or self.locked:
                self.locked = False
            self.last_seen = now
        if (self.auto_lock and self.last_seen is not None and not self.locked
                and now - self.last_seen > self.lock_after):
            self.locked = True
            self._lock()
        if self.command:
            for name in names_seen:
                if now - self._last_cmd.get(name, 0) >= self.cmd_cooldown:
                    self._last_cmd[name] = now
                    cmd = self.command.replace("{name}", name)
                    logging.info("[ACTION] running: %s", cmd)
                    try:
                        subprocess.Popen(cmd, shell=True)
                    except Exception as e:
                        logging.warning("[ACTION] command failed: %s", e)

    def _lock(self):
        if self.events:
            self.events.emit("auto_lock", self.owner, cooldown_s=0)
        if sys.platform == "win32":
            ctypes.windll.user32.LockWorkStation()
        else:
            logging.info("[ACTION] (lock skipped: not Windows)")
