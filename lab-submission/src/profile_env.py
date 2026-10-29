import platform
import os
import json
from pathlib import Path
import argparse
from salt import get_student_params

print("CPU:", platform.processor())
print("Logical cores:", os.cpu_count())
print("OS:", platform.platform())
print("Python:", platform.python_version())

import subprocess

command = (
    "Get-CimInstance Win32_Processor | "
    "Select-Object Name, NumberOfCores, "
    "NumberOfLogicalProcessors, L2CacheSize, L3CacheSize | "
    "ConvertTo-Json"
)

result = subprocess.run(
    ["powershell", "-NoProfile", "-Command", command],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    raise RuntimeError(result.stderr)

cpu_info = json.loads(result.stdout)

print("\nCPU Information:")
print("Name:", cpu_info["Name"])
print("Physical cores:", cpu_info["NumberOfCores"])
print("Logical cores:", cpu_info["NumberOfLogicalProcessors"])
print("L2 Cache (KB):", cpu_info["L2CacheSize"])
print("L3 Cache (KB):", cpu_info["L3CacheSize"])


from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--student-id", required=True)
args = parser.parse_args()
print("Student ID:", args.student_id)
params = get_student_params(args.student_id)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

host_profile = {
    "cpu_model": cpu_info["Name"],
    "microarchitecture": "Tiger Lake",
    "base_frequency_ghz": 2.7,
    "physical_cores": cpu_info["NumberOfCores"],
    "logical_cores": cpu_info["NumberOfLogicalProcessors"],
    "hyper_threading": True,
    "l1_data_cache_kb_per_core": 48,
    "l2_cache_kb_reported_by_windows": cpu_info["L2CacheSize"],
    "l3_cache_kb": cpu_info["L3CacheSize"],
    "cache_line_bytes": 64,
    "os": platform.platform(),
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation()
}

with open(results_dir / "host_profile.json", "w", encoding="utf-8") as f:
    json.dump(host_profile, f, indent=4)

print("Saved: results/host_profile.json")

with open(results_dir / "salted_params.json", "w", encoding="utf-8") as f:
    json.dump(params, f, indent=4)

print("Saved: results/salted_params.json")