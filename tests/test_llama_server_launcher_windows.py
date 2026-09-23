from __future__ import annotations

import os
import shutil
import socket
import subprocess
from pathlib import Path

import pytest


LAUNCHER_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "llama-rpc"
    / "start-distributed-llama-server-windows.ps1"
)

POWERSHELL_PATH = shutil.which("pwsh") or shutil.which("powershell")

pytestmark = pytest.mark.skipif(
    POWERSHELL_PATH is None,
    reason="PowerShell is unavailable",
)


def run_launcher(
    temporary_path: Path,
    *arguments: str,
    rpc_servers: str | None = None,
) -> subprocess.CompletedProcess[str]:
    llama_cpp_directory = temporary_path / "llama.cpp"
    llama_server_path = (
        llama_cpp_directory / "build-rpc-cuda" / "bin" / "Release" / "llama-server.exe"
    )
    llama_server_path.parent.mkdir(parents=True, exist_ok=True)
    llama_server_path.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    llama_server_path.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "LLAMA_CPP_DIRECTORY": str(llama_cpp_directory),
            "LLAMA_MODEL_PATH": "/tmp/model.gguf",
        }
    )
    for variable_name in (
        "LLAMA_GPU_LAYERS",
        "LLAMA_CPU_MOE_LAYERS",
        "LLAMA_CPU_FFN_LAYERS",
        "LLAMA_TENSOR_SPLIT",
        "LLAMA_FIT_TARGET",
    ):
        environment.pop(variable_name, None)
    if rpc_servers is not None:
        environment["LLAMA_RPC_SERVERS"] = rpc_servers
    else:
        environment.pop("LLAMA_RPC_SERVERS", None)

    assert POWERSHELL_PATH is not None
    return subprocess.run(
        [POWERSHELL_PATH, "-NoProfile", "-File", str(LAUNCHER_PATH), *arguments],
        capture_output=True,
        check=False,
        env=environment,
        stdin=subprocess.DEVNULL,
        text=True,
        timeout=30,
    )


def test_local_mode_skips_rpc_and_mtp_by_default(tmp_path: Path) -> None:
    result = run_launcher(tmp_path, "-Local")

    assert result.returncode == 0
    assert "--n-gpu-layers\nall\n" in result.stdout
    assert "--alias\nwindows-local\n" in result.stdout
    assert "--rpc\n" not in result.stdout
    assert "--load-mode\n" not in result.stdout
    assert "--ctx-checkpoints\n0\n" in result.stdout
    assert "--spec-type\n" not in result.stdout


def test_distributed_mode_uses_configured_rpc_server(tmp_path: Path) -> None:
    result = run_launcher(tmp_path, rpc_servers="192.168.0.20:50052")

    assert result.returncode == 0
    assert "--rpc\n192.168.0.20:50052\n" in result.stdout
    assert "--load-mode\nnone\n" in result.stdout
    assert "--fit-target\n512,2048\n" in result.stdout
    assert "--alias\ndistributed-local\n" in result.stdout
    assert "--n-gpu-layers\n" not in result.stdout


def test_mtp_uses_requested_number_of_blocks(tmp_path: Path) -> None:
    result = run_launcher(tmp_path, "-Local", "-Mtp", "-MtpBlocks", "5")

    assert result.returncode == 0
    assert "--spec-type\ndraft-mtp\n" in result.stdout
    assert "--spec-draft-n-max\n5\n" in result.stdout


def test_local_mode_uses_requested_partial_gpu_offload(tmp_path: Path) -> None:
    result = run_launcher(tmp_path, "-Local", "-GpuLayers", "20")

    assert result.returncode == 0
    assert "--n-gpu-layers\n20\n" in result.stdout
    assert "--n-gpu-layers\nall\n" not in result.stdout


def test_distributed_mode_supports_partial_offload_options(tmp_path: Path) -> None:
    result = run_launcher(
        tmp_path,
        "-GpuLayers",
        "24",
        "-CpuMoeLayers",
        "8",
        "-CpuFfnLayers",
        "4",
        rpc_servers="192.168.0.20:50052",
    )

    assert result.returncode == 0
    assert "--n-gpu-layers\n24\n" in result.stdout
    assert "--n-cpu-moe\n8\n" in result.stdout
    assert "--n-cpu-ffn\n4\n" in result.stdout


def test_tensor_split_disables_automatic_fit(tmp_path: Path) -> None:
    result = run_launcher(
        tmp_path,
        "-TensorSplit",
        "1,1,1.1",
        rpc_servers="192.168.0.20:50052,192.168.0.21:50052",
    )

    assert result.returncode == 0
    assert "--fit\noff\n" in result.stdout
    assert "--tensor-split\n1,1,1.1\n" in result.stdout
    assert "--fit-target\n" not in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ("-Local", "-GpuLayers", "half"),
        ("-Local", "-CpuMoeLayers", "-1"),
        ("-Local", "-CpuFfnLayers", "x"),
        ("-Local", "-MtpBlocks", "0"),
    ],
)
def test_invalid_offload_values_fail_without_starting_server(
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    result = run_launcher(tmp_path, *arguments)

    assert result.returncode != 0
    assert "--ctx-size" not in result.stdout


def test_missing_rpc_servers_fails_without_interactive_input(tmp_path: Path) -> None:
    environment_subnet = "203.0.113"
    llama_cpp_directory = tmp_path / "llama.cpp"
    llama_server_path = (
        llama_cpp_directory / "build-rpc-cuda" / "bin" / "Release" / "llama-server.exe"
    )
    llama_server_path.parent.mkdir(parents=True, exist_ok=True)
    llama_server_path.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    llama_server_path.chmod(0o755)

    environment = os.environ.copy()
    environment.update(
        {
            "LLAMA_CPP_DIRECTORY": str(llama_cpp_directory),
            "LLAMA_MODEL_PATH": "/tmp/model.gguf",
            "LLAMA_RPC_SUBNET": environment_subnet,
            "LLAMA_RPC_PORT": "1",
        }
    )
    environment.pop("LLAMA_RPC_SERVERS", None)

    assert POWERSHELL_PATH is not None
    result = subprocess.run(
        [POWERSHELL_PATH, "-NoProfile", "-File", str(LAUNCHER_PATH)],
        capture_output=True,
        check=False,
        env=environment,
        stdin=subprocess.DEVNULL,
        text=True,
        timeout=120,
    )

    assert result.returncode != 0
    assert "LLAMA_RPC_SERVERS" in result.stderr


def test_subnet_scan_discovers_listening_worker(tmp_path: Path) -> None:
    llama_cpp_directory = tmp_path / "llama.cpp"
    llama_server_path = (
        llama_cpp_directory / "build-rpc-cuda" / "bin" / "Release" / "llama-server.exe"
    )
    llama_server_path.parent.mkdir(parents=True, exist_ok=True)
    llama_server_path.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    llama_server_path.chmod(0o755)

    listening_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listening_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listening_socket.bind(("127.0.0.1", 0))
    listening_socket.listen(16)
    listening_port = listening_socket.getsockname()[1]

    environment = os.environ.copy()
    environment.update(
        {
            "LLAMA_CPP_DIRECTORY": str(llama_cpp_directory),
            "LLAMA_MODEL_PATH": "/tmp/model.gguf",
            "LLAMA_RPC_SUBNET": "127.0.0",
            "LLAMA_RPC_PORT": str(listening_port),
        }
    )
    environment.pop("LLAMA_RPC_SERVERS", None)

    assert POWERSHELL_PATH is not None
    try:
        result = subprocess.run(
            [POWERSHELL_PATH, "-NoProfile", "-File", str(LAUNCHER_PATH)],
            capture_output=True,
            check=False,
            env=environment,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=120,
        )
    finally:
        listening_socket.close()

    assert result.returncode == 0
    assert f"--rpc\n127.0.0.1:{listening_port}\n" in result.stdout
