"""Compatibility entry point for the declared ECG augmentation suite."""
import runpy
from pathlib import Path

if __name__=='__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1]/'run_additional_experiments.py'),run_name='__main__')
