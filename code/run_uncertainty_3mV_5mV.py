"""Uncertainty fits of all nine cells at 3 and 5 mV voltage noise."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uncertainty_runner as runner

runner.OUT = runner.ROOT / 'output' / 'uncertainty_3mV_5mV'
runner.SIGMAS = [3, 5]

if __name__ == '__main__':
    runner.main()
