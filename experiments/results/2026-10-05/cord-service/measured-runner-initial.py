"""Launch two verified local vLLM checkpoints and record a frozen CORD study.

Linux GPU host only. Uses an idle, explicitly selected GPU and terminates only
the process groups it creates. Install the serving environment separately.
"""

import argparse
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

import pynvml

from precisionlab.data import sha256
from precisionlab.gate import gate_service
from precisionlab.serving import evaluate_service
from precisionlab.study import check_splits, summarize_repetitions
from download_model import matches


PROCESSOR_FILES = ("chat_template.json", "chat_template.jinja", "preprocessor_config.json",
                   "video_preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
                   "vocab.json", "merges.txt", "added_tokens.json")


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def checkpoint_evidence(bf16, awq, output):
    source = json.loads(Path(__file__).with_name("model-source.json").read_text())
    original = [i for i in source["siblings"] if i["rfilename"] != ".gitattributes"]
    for info in original:
        if not matches(bf16 / info["rfilename"], info):
            raise ValueError("BF16 checkpoint differs from pinned revision: " + info["rfilename"])
    expected = json.loads(Path(__file__).parent.joinpath("results/2026-10-03/awq-export-file-hashes.json").read_text())
    for info in expected["files"]:
        path = awq / info["name"]
        if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
            raise ValueError("AWQ checkpoint differs from recorded export: " + info["name"])
    # Preserve the original export. A new overlay uses exactly the BF16
    # processor/tokenizer bytes, with the recorded AWQ weights and config.
    overlay = output / "awq-controlled"
    overlay.mkdir()
    for path in awq.iterdir():
        if path.is_file() and path.name not in PROCESSOR_FILES:
            (overlay / path.name).symlink_to(path.resolve())
    for name in PROCESSOR_FILES:
        if (bf16 / name).exists():
            (overlay / name).symlink_to((bf16 / name).resolve())
    evidence = {"base_revision": source["sha"], "original_files_verified": len(original),
                "awq_files_verified": len(expected["files"]),
                "awq_export_manifest_sha256": sha256(awq / "compression-manifest.json"),
                "processor_files": {n: sha256(bf16 / n) for n in PROCESSOR_FILES if (bf16 / n).exists()},
                "weight_bytes": {"bf16": sum(f.stat().st_size for f in bf16.glob("*.safetensors")),
                                 "awq": sum(f.stat().st_size for f in awq.glob("*.safetensors"))},
                "scope": "Verified checkpoint files and owned launch arguments, not a hash of weights resident in GPU memory"}
    save(output / "checkpoint-evidence.json", evidence)
    return overlay, evidence


@lru_cache
def gpu_handle(gpu):
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(gpu)


def gpu_snapshot(gpu):
    # Keep one NVML connection instead of starting nvidia-smi for each sample.
    # The CLI's initial device enumeration can exceed a short process timeout.
    handle = gpu_handle(gpu)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {"uuid": pynvml.nvmlDeviceGetUUID(handle), "name": pynvml.nvmlDeviceGetName(handle),
            "driver": pynvml.nvmlSystemGetDriverVersion(), "total_mib": memory.total // 2**20,
            "used_mib": memory.used // 2**20,
            "compute_pids": [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)],
            "utilization_percent": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            "power_watts": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
            "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)}


class Monitor:
    def __init__(self, gpu, output):
        self.gpu, self.output, self.stage = gpu, output, "startup"
        self.stop = threading.Event()
        self.started = time.monotonic()
        self.records, self.errors = [], []
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        with self.output.open("w") as stream:
            while not self.stop.is_set():
                try:
                    record = {"elapsed_seconds": time.monotonic() - self.started, "stage": self.stage, **gpu_snapshot(self.gpu)}
                    self.records.append(record)
                    stream.write(json.dumps(record) + "\n"); stream.flush()
                except (pynvml.NVMLError, OSError) as exc:
                    self.errors.append(type(exc).__name__)
                self.stop.wait(.5)

    def finish(self):
        self.stop.set(); self.thread.join(timeout=10)
        return {"sampling_interval_seconds": .5, "samples": len(self.records), "errors": self.errors,
                "peak_framebuffer_mib_by_stage": {stage: max(int(r["used_mib"]) for r in self.records if r["stage"] == stage)
                    for stage in sorted({r["stage"] for r in self.records})},
                "scope": "Sampled whole-device framebuffer usage on a GPU verified idle before launch; includes weights, KV cache, allocator and workspaces; not torch allocated memory"}


def wait_ready(process, endpoint, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Owned server exited with {process.returncode}; inspect server.log")
        try:
            with urlopen(endpoint + "/models", timeout=2) as response:
                models = json.load(response)
            if any(m.get("id") == "model" for m in models.get("data", [])):
                return models
        except (URLError, TimeoutError, OSError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError("Owned server did not become ready")


def stop_owned(process):
    # A dead API parent can still have an engine child in its process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def server_command(model, protocol, port):
    engine = protocol["engine"]
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model),
            "--served-model-name", "model", "--host", "127.0.0.1", "--port", str(port),
            "--dtype", engine["dtype"], "--tensor-parallel-size", "1", "--max-model-len", str(engine["max_model_len"]),
            "--max-num-seqs", str(engine["max_num_seqs"]), "--max-num-batched-tokens", str(engine["max_num_batched_tokens"]),
            "--kv-cache-memory-bytes", str(engine["kv_cache_memory_bytes"]), "--kv-cache-dtype", engine["kv_cache_dtype"],
            "--enforce-eager", "--no-enable-prefix-caching", "--mm-processor-cache-gb", "0",
            "--mm-processor-kwargs", json.dumps(engine["mm_processor_kwargs"]),
            "--limit-mm-per-prompt", '{"image":1,"video":0}', "--generation-config", "vllm", "--seed", "42"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-model", type=Path, required=True)
    parser.add_argument("--awq-model", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("cord-service-protocol.json"))
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    for name in ("vllm", "transformers"):
        if importlib.metadata.version(name) != protocol[name + "_version"]:
            parser.error(name + " version differs from frozen protocol")
    split_info = check_splits(args.dev, args.test)
    gpu = gpu_snapshot(args.gpu)
    with open("/tmp/precisionlab-" + gpu["uuid"] + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # NVML can include roughly 0.5 GiB of driver reservation even when
        # nvidia-smi displays zero application memory on this Blackwell host.
        if gpu["compute_pids"] or int(gpu["used_mib"]) > 1024:
            parser.error("Selected GPU is already in use; choose an idle GPU")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", args.port))
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "protocol.json").write_bytes(args.protocol.read_bytes())
        save(args.output / "split-evidence.json", split_info)
        print(json.dumps({"stage": "verifying_checkpoints", "splits": split_info}), flush=True)
        awq, checkpoints = checkpoint_evidence(args.bf16_model.resolve(), args.awq_model.resolve(), args.output)
        packages = {n: importlib.metadata.version(n) for n in ("vllm", "torch", "transformers", "compressed-tensors", "pillow", "nvidia-ml-py")}
        controls = {"engine": protocol["engine"], "packages": packages, "gpu": gpu["uuid"],
                    "processor_files": checkpoints["processor_files"]}
        save(args.output / "shared-controls.json", controls)
        study = {"status": "running", "recorded_at": datetime.now(timezone.utc).isoformat(), "gpu": gpu,
                 "packages": packages, "script_sha256": sha256(__file__), "protocol_sha256": sha256(args.protocol), "variants": {}}
        save(args.output / "study.json", study)
        endpoint = f"http://127.0.0.1:{args.port}/v1"
        env = os.environ.copy()
        # vLLM 0.11's NVML mapping parses CUDA_VISIBLE_DEVICES as integer ids.
        # Keep the UUID as evidence and check CUDA's actual selection below.
        env.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "VLLM_USAGE_STATS_DO_NOT_TRACK": "1", "DO_NOT_TRACK": "1", "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})
        env.pop("VLLM_ATTENTION_BACKEND", None)
        if protocol["engine"]["attention_backend"] != "auto":
            env["VLLM_ATTENTION_BACKEND"] = protocol["engine"]["attention_backend"]
        # This owned loopback server has no API key. Do not inherit an unrelated one.
        os.environ.pop("PRECISIONLAB_API_KEY", None)
        probe = subprocess.check_output([sys.executable, "-c", "import torch,json; p=torch.cuda.get_device_properties(0); print(json.dumps({'uuid':str(p.uuid),'name':p.name,'capability':torch.cuda.get_device_capability(0)}))"],
                                        env=env, text=True, timeout=120)
        selected = json.loads(probe.strip().splitlines()[-1])
        if selected["uuid"].removeprefix("GPU-") != gpu["uuid"].removeprefix("GPU-"):
            raise RuntimeError("CUDA and NVML device selections differ; no server launched")
        save(args.output / "cuda-device-evidence.json", selected)
        for mode, model in (("bf16", args.bf16_model.resolve()), ("awq", awq)):
            idle = gpu_snapshot(args.gpu)
            if idle["compute_pids"] or int(idle["used_mib"]) > 1024:
                raise RuntimeError("GPU did not return to idle before next owned launch")
            folder = args.output / mode
            folder.mkdir()
            command = server_command(model, protocol, args.port)
            save(folder / "launch.json", {"argv": command, "environment_controls": {k: env.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "VLLM_ATTENTION_BACKEND", "HF_HUB_OFFLINE", "OMP_NUM_THREADS")}})
            monitor = Monitor(args.gpu, folder / "gpu-samples.jsonl")
            monitor.thread.start()
            tick = time.monotonic()
            process = None
            variant = {"status": "starting", "dev": {}}
            study["variants"][mode] = variant
            try:
                with (folder / "server.log").open("w") as log:
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    print(json.dumps({"stage": "starting_server", "variant": mode}), flush=True)
                    variant["advertised_models"] = wait_ready(process, endpoint, args.startup_timeout)
                    variant["startup_seconds"] = time.monotonic() - tick
                    req = protocol["request"]
                    def evaluate(dataset, split, name, concurrency):
                        monitor.stage = name
                        return evaluate_service(dataset, folder / name, endpoint=endpoint, model="model",
                            base_revision=protocol["base_revision"], split=split, concurrency=concurrency,
                            max_tokens=req["max_tokens"], seed=req["seed"], warmup=req["warmup"],
                            timeout=req["timeout_seconds"], server_config=controls)
                    for concurrency in protocol["dev_concurrency"]:
                        runs = []
                        for repeat in range(protocol["dev_repetitions"]):
                            name = f"dev-c{concurrency}-r{repeat + 1}"
                            profile = evaluate(args.dev, "dev", name, concurrency)
                            runs.append(folder / name)
                            print(json.dumps({"variant": mode, "run": name, "task_em": profile["task_em"], "rps": profile["successful_requests_per_second"]}), flush=True)
                        variant["dev"][str(concurrency)] = summarize_repetitions(args.dev, runs)
                    profile = evaluate(args.test, "test", "test-c1-r1", protocol["test_concurrency"])
                    variant["test_profile"] = profile
                    variant["test_gate"] = gate_service(args.test, folder / "test-c1-r1", protocol["test_gate"])
                    save(folder / "test-gate.json", variant["test_gate"])
                    variant["status"] = "completed"
            except BaseException as exc:
                variant.update(status="failed", error_type=type(exc).__name__)
                study["status"] = "failed"
                raise
            finally:
                if process is not None:
                    stop_owned(process)
                variant["memory"] = monitor.finish()
                save(folder / "memory-profile.json", variant["memory"])
                save(args.output / "study.json", study)
        study["status"] = "completed"
        save(args.output / "study.json", study)
        print(json.dumps({"status": study["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
