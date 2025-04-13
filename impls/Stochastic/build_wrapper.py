import sys
import subprocess

compute_capability = sys.argv[1]

subprocess.run([
    sys.executable,
    "setup.py",
    "bdist_wheel",
    f"--compute-capability={compute_capability}"
])