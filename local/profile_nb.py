# profile_nb.py
import papermill as pm
import threading, time, psutil, pynvml, sys, json
from datetime import datetime

peak = {"ram_gb": 0, "gpu_gb": 0}
stop_event = threading.Event()

def monitor():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    while not stop_event.is_set():
        peak["ram_gb"] = max(peak["ram_gb"], psutil.virtual_memory().used / 1e9)
        peak["gpu_gb"] = max(peak["gpu_gb"], pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1e9)
        time.sleep(0.5)

threading.Thread(target=monitor, daemon=True).start()

notebook = sys.argv[1]
start = time.time()
pm.execute_notebook(notebook, f"executed_{notebook}")
elapsed = time.time() - start

stop_event.set()

report = {
    "notebook": notebook,
    "timestamp": datetime.now().isoformat(),
    "duration_minutes": round(elapsed / 60, 2),
    "peak_ram_gb": round(peak["ram_gb"], 2),
    "peak_gpu_gb": round(peak["gpu_gb"], 2),
}

# Write to text file
report_path = f"profile_{notebook.replace('.ipynb', '')}.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"Report saved to {report_path}")
print(json.dumps(report, indent=2))