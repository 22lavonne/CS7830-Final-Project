import sys
import subprocess
from pathlib import Path

# directory containing all the executables
# every file directly inside it will get processed
TEST_FILES_DIR = Path("test-files")
# the headless script that analyzes a single binary
HEADLESS_SCRIPT = "ghidra_headless_script.py"

# get every file in the test-files directory. this is done once up front (and sorted) because the headless script
# creates a "projects" folder inside of test-files, and we only want actual binaries (files, not folders)
binaries = sorted(
    p for p in TEST_FILES_DIR.iterdir()
    if p.is_file() and not p.name.startswith(".")
)
print(f"Found {len(binaries)} binaries in {TEST_FILES_DIR}")

# run the headless script on each binary in its own process, so ghidra and ghidrassist start fresh every time
# (ghidrassist keeps state like the analysis database between binaries if they share a process).
# if one fails, report it and move on to the next
failed = []
for i, binary_path in enumerate(binaries, start=1):
    print(f"\n===== [{i}/{len(binaries)}] {binary_path.name} =====", flush=True)
    result = subprocess.run([sys.executable, HEADLESS_SCRIPT, str(binary_path)])
    if result.returncode != 0:
        print(f"FAILED on {binary_path.name} (return code {result.returncode})")
        failed.append(binary_path.name)

print(f"\nDone. {len(binaries) - len(failed)}/{len(binaries)} binaries processed successfully.")
if failed:
    print("Failed binaries:")
    for name in failed:
        print(f"  {name}")