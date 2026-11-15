from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_launcher_files_invoke_canonical_module():
    assert 'python -m demo_cli.cli @demoArguments' in (ROOT / "demo.ps1").read_text()
    assert 'exec python -m demo_cli.cli "$@"' in (ROOT / "demo").read_text()


@pytest.mark.parametrize("shell", ["powershell", "bash"])
def test_launchers_forward_arguments_cwd_and_exit_code(tmp_path, shell):
    launch_root = tmp_path / "launch root"
    launch_root.mkdir()
    module = launch_root / "demo_cli"
    module.mkdir()
    (module / "__init__.py").write_text("")
    (module / "cli.py").write_text("import json, os, sys\nprint(json.dumps([os.getcwd(), sys.argv[1:]]))\nsys.exit(7)\n")
    arguments = ["chat", "--session", "label with spaces", "--model", "exact:tag", "", 'a"b', 'a\\"b', "C:\\with space\\", "--no-color"]
    if shell == "powershell":
        executable = shutil.which("powershell") or shutil.which("pwsh")
        if not executable or os.name != "nt":
            pytest.skip("Windows PowerShell is unavailable")
        launcher = launch_root / "demo.ps1"
        shutil.copyfile(ROOT / "demo.ps1", launcher)
        literals = " ".join("'" + item.replace("'", "''") + "'" for item in [str(launcher), *arguments])
        driver = tmp_path / "driver.ps1"
        driver.write_text("& " + literals + "\nexit $LASTEXITCODE\n")
        command = [executable, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(driver)]
    else:
        git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        executable = str(git_bash) if git_bash.is_file() else shutil.which("bash")
        if not executable:
            pytest.skip("Bash is unavailable")
        launcher = launch_root / "demo"
        shutil.copyfile(ROOT / "demo", launcher)
        launcher.chmod(0o755)
        driver = tmp_path / "driver.sh"
        driver.write_text("exec " + shlex.join([launcher.as_posix(), *arguments]) + "\n")
        command = [executable, str(driver)]
    environment = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(  # noqa: S603 - local launcher with a fake CLI module, no provider call
        command,
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 7, result.stdout + result.stderr
    cwd, args = json.loads(result.stdout)
    assert Path(cwd).resolve() == launch_root
    assert args == arguments


def test_powershell_missing_python_is_concise(tmp_path):
    executable = shutil.which("powershell") or shutil.which("pwsh")
    if not executable or os.name != "nt":
        pytest.skip("Windows PowerShell is unavailable")
    result = subprocess.run(  # noqa: S603 - local launcher, intentionally empty executable search path
        [executable, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "demo.ps1"), "preflight"],
        env=dict(os.environ, PATH=str(tmp_path)), cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1
    assert "Unable to invoke Python" in result.stdout and not result.stderr
