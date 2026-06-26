#!/usr/bin/env python3
import os
import runpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runpy.run_path(os.path.join(ROOT, "scripts", "feedback", "l2_check.py"), run_name="__main__")
