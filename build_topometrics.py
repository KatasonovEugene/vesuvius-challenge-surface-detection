#!/usr/bin/env python3
"""
Build and install topometrics without conda (Kaggle-friendly).

1) Installs Python deps (optionally from local wheels).
2) Builds Betti-Matching-3D with CMake.
3) Installs topometrics-3d package.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Iterable


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)


def find_libpython() -> str:
    libdir = sysconfig.get_config_var("LIBDIR") or ""
    libpl = sysconfig.get_config_var("LIBPL") or ""
    ldver = sysconfig.get_config_var("LDVERSION") or sysconfig.get_config_var("VERSION") or ""
    ldlibrary = sysconfig.get_config_var("LDLIBRARY") or ""

    candidates: list[str] = []
    if ldlibrary:
        if libdir:
            candidates.append(os.path.join(libdir, ldlibrary))
        if libpl:
            candidates.append(os.path.join(libpl, ldlibrary))
    if ldver:
        if libdir:
            candidates.append(os.path.join(libdir, f"libpython{ldver}.so"))
            candidates.append(os.path.join(libdir, f"libpython{ldver}.a"))
        if libpl:
            candidates.append(os.path.join(libpl, f"libpython{ldver}.so"))
            candidates.append(os.path.join(libpl, f"libpython{ldver}.a"))

    prefix = sys.prefix
    candidates.append(os.path.join(prefix, "lib", f"libpython{ldver}.so"))
    candidates.append(os.path.join(prefix, "lib", f"libpython{ldver}.a"))

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ""


def install_deps(requirements: Path, wheels_dir: Path | None) -> None:
    cmd = [sys.executable, "-m", "pip", "install"]
    if wheels_dir and wheels_dir.exists():
        cmd += ["--no-index", "--find-links", str(wheels_dir)]
    cmd += ["-r", str(requirements)]
    run(cmd)


def build_betti(project_root: Path) -> None:
    betti_dir = project_root / "external" / "Betti-Matching-3D"
    if not betti_dir.exists():
        raise FileNotFoundError(f"Betti-Matching-3D not found at {betti_dir}")

    build_dir = betti_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    pybin = sys.executable
    py_inc = sysconfig.get_config_var("INCLUDEPY") or sysconfig.get_paths().get("include", "")
    py_lib = find_libpython()

    if not py_lib:
        raise RuntimeError(
            "Could not find libpython for the current interpreter. "
            "Install the Python shared library or use a Python build that provides it."
        )

    prefix_paths: list[str] = []
    if sys.prefix:
        prefix_paths.append(sys.prefix)
    purelib = sysconfig.get_paths().get("purelib", "")
    if purelib:
        prefix_paths.append(purelib)
    cmake_prefix = ";".join(prefix_paths)

    cmake_args = [
        "cmake",
        "-S",
        str(betti_dir),
        "-B",
        str(build_dir),
        f"-DPython_EXECUTABLE={pybin}",
        f"-DPython_INCLUDE_DIR={py_inc}",
        f"-DPython_LIBRARY={py_lib}",
        "-DPYBIND11_FINDPYTHON=ON",
    ]
    if cmake_prefix:
        cmake_args.append(f"-DCMAKE_PREFIX_PATH={cmake_prefix}")

    run(cmake_args)
    run(["cmake", "--build", str(build_dir), "--parallel"])


def install_package(project_root: Path, editable: bool) -> None:
    cmd = [sys.executable, "-m", "pip", "install"]
    if editable:
        cmd.append("-e")
    cmd.append(str(project_root))
    run(cmd)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build topometrics without conda")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip installing requirements")
    parser.add_argument("--skip-build", action="store_true", help="Skip building Betti-Matching-3D")
    parser.add_argument("--no-editable", action="store_true", help="Install package non-editable")
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path(__file__).resolve().parent
    project_root = repo_root / "ext" / "vesuvius_metric_resources" / "topological_metrics_kaggle"
    requirements = project_root / "requirements.txt"
    wheels_dir = repo_root / "ext" / "wheels"

    if not project_root.exists():
        raise FileNotFoundError(f"Project root not found at {project_root}")

    if not args.skip_install:
        install_deps(requirements, wheels_dir)

    if not args.skip_build:
        build_betti(project_root)

    install_package(project_root, editable=not args.no_editable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
