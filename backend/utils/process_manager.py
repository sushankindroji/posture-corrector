import subprocess
from typing import Dict, Optional
import uuid

class ProcessManager:
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}

    def start_process(self, command: list) -> str:
        process_id = str(uuid.uuid4())
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.processes[process_id] = process
        return process_id

    def stop_process(self, process_id: str) -> bool:
        process = self.processes.get(process_id)
        if process:
            process.terminate()
            del self.processes[process_id]
            return True
        return False
    