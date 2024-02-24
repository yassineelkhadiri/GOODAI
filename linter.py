"""Python script to run all linters."""

import sys
import subprocess

PATH = sys.executable.rsplit("/", maxsplit=1)[0]

print("[Running BLACK]", flush=True)
subprocess.run([f"{PATH}/python", "-m", "black", "--check", "."], check=True)

print("[Running FLAKE 8]", flush=True)
subprocess.run(
    [f"{PATH}/python", "-m", "flake8", "--config", "setup.ini", "."], check=True
)


print("[Running MYPY]", flush=True)
subprocess.run(
    [f"{PATH}/python", "-m", "mypy", "--config-file", "setup.ini", "."], check=True
)
