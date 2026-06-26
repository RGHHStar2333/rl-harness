#!/usr/bin/env python3
import os
import runpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runpy.run_path(os.path.join(ROOT, "scripts", "training", "pause_training.py"), run_name="__main__")
