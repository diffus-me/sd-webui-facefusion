#!/usr/bin/env python3

from pathlib import Path
from subprocess import run

import launch
import pkg_resources

_REQUIREMENT_PATH = Path(__file__).absolute().parent / "requirements_webui.txt"
_FFMPEG_INSTALL_SCRIPT = """
#!/bin/bash

set -euo pipefail

if ! dpkg -s ffmpeg &>/dev/null; then
    sudo apt-get update
    sudo apt-get install -y ffmpeg
fi
"""


def _get_comparable_version(version: str) -> tuple:
    return tuple(version.split("."))


def _get_installed_version(package: str) -> str | None:
    try:
        return pkg_resources.get_distribution(package).version
    except Exception:
        return None


def _install_ffmpeg() -> None:
    run(
        _FFMPEG_INSTALL_SCRIPT,
        executable="bash",
        shell=True,
        check=True,
        capture_output=True,
    )


with _REQUIREMENT_PATH.open() as fp:
    try:
        _install_ffmpeg()
        print("sd-webui-facefusion requirement: ffmepg")
    except Exception as error:
        print(error)
        print("Warning: Failed to install 'ffmpeg', some preprocessors may not work.")

    for requirement in fp:
        try:
            requirement = requirement.strip()
            if "==" in requirement:
                name, version = requirement.split("==", 1)
                installed_version = _get_installed_version(name)

                if installed_version == version:
                    continue

                launch.run_pip(
                    f"install -U {requirement}",
                    f"sd-webui-facefusion requirement: changing {name} version from {installed_version} to {version}",
                )
                continue

            if ">=" in requirement:
                name, version = requirement.split(">=", 1)
                installed_version = _get_installed_version(name)

                if installed_version and (
                    _get_comparable_version(installed_version) >= _get_comparable_version(version)
                ):
                    continue

                launch.run_pip(
                    f"install -U {requirement}",
                    f"sd-webui-facefusion requirement: changing {name} version from {installed_version} to {version}",
                )
                continue

            if not launch.is_installed(requirement):
                launch.run_pip(
                    f"install {requirement}",
                    f"sd-webui-facefusion requirement: {requirement}",
                )
        except Exception as error:
            print(error)
            print(f"Warning: Failed to install '{requirement}', some preprocessors may not work.")
