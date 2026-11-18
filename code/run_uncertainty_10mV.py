"""Uncertainty fits of all nine cells at 10 mV voltage noise."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uncertainty_runner as runner

runner.OUT = runner.ROOT / 'output' / 'uncertainty_10mV'
runner.SIGMAS = [10]

if __name__ == '__main__':
    runner.main()
