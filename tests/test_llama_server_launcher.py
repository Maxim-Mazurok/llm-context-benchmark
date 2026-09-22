from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


LAUNCHER_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "llama-rpc"
    / "start-distributed-llama-server-macos.sh"
)


def run_launcher(
    temporary_path: Path,
    *arguments: str,
    rpc_servers: str | None = None,
) -> subprocess.CompletedProcess[str]:
    llama_cpp_directory = temporary_path / "llama.cpp"
    llama_server_path = (
        llama_cpp_directory / "build-rpc-metal" / "bin" / "llama-server"
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
    if rpc_servers is not None:
        environment["LLAMA_RPC_SERVERS"] = rpc_servers
    else:
        environment.pop("LLAMA_RPC_SERVERS", None)

    return subprocess.run(
        [str(LAUNCHER_PATH), *arguments],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=5,
    )


def test_local_mode_skips_rpc_and_mtp_by_default(tmp_path: Path) -> None:
    result = run_launcher(tmp_path, "--local")

    assert result.returncode == 0
    assert "--n-gpu-layers\nall\n" in result.stdout
    assert "--alias\nmac-local\n" in result.stdout
    assert "--rpc\n" not in result.stdout
    assert "--spec-type\n" not in result.stdout
    assert "--spec-draft-n-max\n" not in result.stdout


def test_distributed_mode_uses_configured_rpc_server(tmp_path: Path) -> None:
    result = run_launcher(tmp_path, rpc_servers="192.168.0.20:50052")

    assert result.returncode == 0
    assert "--rpc\n192.168.0.20:50052\n" in result.stdout
    assert "--alias\ndistributed-local\n" in result.stdout
    assert "--n-gpu-layers\n" not in result.stdout


@pytest.mark.parametrize(
    ("arguments", "expected_blocks"),
    [
        (("--local", "--mtp"), "3"),
        (("--local", "--mtp", "--mtp-blocks", "5"), "5"),
    ],
)
def test_mtp_uses_requested_number_of_blocks(
    tmp_path: Path,
    arguments: tuple[str, ...],
    expected_blocks: str,
) -> None:
    result = run_launcher(tmp_path, *arguments)

    assert result.returncode == 0
    assert "--spec-type\ndraft-mtp\n" in result.stdout
    assert f"--spec-draft-n-max\n{expected_blocks}\n" in result.stdout


def test_invalid_mtp_blocks_fails_without_starting_server(tmp_path: Path) -> None:
    result = run_launcher(tmp_path, "--local", "--mtp", "--mtp-blocks", "0")

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "MTP blocks must be a positive integer: 0\n"


def test_installed_llama_server_supports_launcher_mtp_options(
    tmp_path: Path,
) -> None:
    llama_cpp_directory = Path(
        os.environ.get("LLAMA_CPP_DIRECTORY", Path.home() / "llama.cpp")
    )
    llama_server_path = (
        llama_cpp_directory / "build-rpc-metal" / "bin" / "llama-server"
    )
    if not llama_server_path.is_file():
        pytest.skip("Installed llama-server is unavailable")

    result = subprocess.run(
        [str(llama_server_path), "--help"],
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0
    assert "--spec-type" in result.stdout
    assert any(
        line.strip().split(maxsplit=1)[0] == "--spec-draft-n-max"
        for line in result.stdout.splitlines()
        if line.strip()
    )

    parse_result = subprocess.run(
        [
            str(llama_server_path),
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            "3",
            "--model",
            str(tmp_path / "missing-model.gguf"),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )

    assert parse_result.returncode != 0
    assert "invalid argument" not in parse_result.stderr
    assert "failed to load model" in parse_result.stderr