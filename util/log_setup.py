"""Logging setup: tee stdout/stderr to both terminal and a timestamped log file."""
import sys
import os
from datetime import datetime


class Tee:
    """Stream wrapper that writes to both the original stream and a log file."""

    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file

    def write(self, data):
        self.original.write(data)
        self.log_file.write(data)

    def flush(self):
        self.original.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


def setup_logging(prefix):
    """Redirect stdout and stderr to both terminal and logs/{prefix}_TIMESTAMP.log."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y.%m.%d_%H:%M:%S')
    log_path = os.path.join(log_dir, f'{prefix}_{timestamp}.log')
    log_file = open(log_path, 'w')

    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    print(f'Logging to {log_path}')
