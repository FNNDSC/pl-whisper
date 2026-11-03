"""The spec, the description for ChRIS, and runs through the default template."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

import pytest
from chris_plugin.parameters import serialize
from hostlib import JobSpec, job_run

from pl_whisper import COMMAND, MEDIA_FILTER, SPEC, WHISPER_MODEL_NAME, __version__
from pl_whisper.__main__ import PLUGIN, main, parser


def _tree_make(root: Path) -> tuple[Path, Path]:
    """Build recordings at two depths, in two cases, beside notes and an earlier transcript.

    Args:
        root: the directory to build the input and output trees in.

    Returns:
        The input and output directories, the output one empty.
    """
    inputDir: Path = root / "incoming"
    outputDir: Path = root / "outgoing"
    for rel, names in {
        ".": ["weekly.wav", "agenda.txt", "jobs.jsonl"],
        "2026-11": ["standup.wav", "review.MP4", "review.srt"],
    }.items():
        (inputDir / rel).mkdir(parents=True, exist_ok=True)
        for name in names:
            (inputDir / rel / name).write_text("not really a recording\n")
    outputDir.mkdir()
    return inputDir, outputDir


def _transcribe_standIn(binDir: Path, exitStatus: int = 0) -> None:
    """Write a whisper-transcribe that writes a .txt named after its recording.

    Args:
        binDir: the directory to write it in, put first on ``PATH`` by the test.
        exitStatus: what it exits with.
    """
    script: Path = binDir / "whisper-transcribe"
    script.write_text(
        '#!/bin/sh\nname=$(basename "$1"); stem="${name%.*}"\n'
        'echo "Good morning." > "$2/$stem.txt" && echo "[0.00 --> 1.00] Good morning."\n'
        f"exit {exitStatus}\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


def test_spec_namesWhisper_andAsksForOneGpu() -> None:
    """The spec names the model, carries the template, v and the filter, and asks for a GPU."""
    assert SPEC.name == "pl-whisper" and WHISPER_MODEL_NAME in SPEC.title
    assert WHISPER_MODEL_NAME in SPEC.version
    assert SPEC.defaults == {"exec": COMMAND, "analyzeFileIndex": "v", "fileFilter": MEDIA_FILTER}
    assert COMMAND.split() == [
        "whisper-transcribe",
        "%{inputWorkingDir}/%{inputWorkingFile}",
        "%{outputWorkingDir}",
    ]
    assert SPEC.minGpu == 1 and SPEC.jobMemory == "8Gi"


def test_defaults_runOneTranscriptionPerRecording(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every recording, in either case, is transcribed; notes, records and subtitles are not."""
    inputDir, outputDir = _tree_make(tmp_path)
    binDir: Path = tmp_path / "bin"
    binDir.mkdir()
    _transcribe_standIn(binDir)
    monkeypatch.setenv("PATH", f"{binDir}{os.pathsep}{os.environ['PATH']}")
    options: argparse.Namespace = parser.parse_args(["--verbosity", "0"])
    assert main(options, inputDir, outputDir) == 0
    assert sorted(p.relative_to(outputDir).as_posix() for p in outputDir.rglob("*.txt")) == [
        "2026-11/review.txt",
        "2026-11/standup.txt",
        "weekly.txt",
    ]
    lines: list[str] = (outputDir / "2026-11" / "jobs.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 2
    assert all(r["returncode"] == 0 and r["argv"][0] == "whisper-transcribe" for r in records)
    span = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert span["name"] == "pl-whisper" and span["status"] is True


def test_failingRecordings_failTheRun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A whisper-transcribe that fails every recording fails the run, and says so in the record."""
    inputDir, outputDir = _tree_make(tmp_path)
    binDir: Path = tmp_path / "bin"
    binDir.mkdir()
    _transcribe_standIn(binDir, exitStatus=1)
    monkeypatch.setenv("PATH", f"{binDir}{os.pathsep}{os.environ['PATH']}")
    assert main(parser.parse_args(["--verbosity", "0"]), inputDir, outputDir) == 1
    assert json.loads((outputDir / "jobs.jsonl").read_text())["returncode"] == 1


def test_description_carriesTheDefaults_forChris() -> None:
    """The parser ChRIS reads has the template, v and the filter as defaults, and no --threads."""
    byName = {spec["name"]: spec for spec in serialize(PLUGIN.chrisParser)}
    assert byName["exec"]["default"] == COMMAND and byName["analyzeFileIndex"]["default"] == "v"
    assert byName["fileFilter"]["default"] == MEDIA_FILTER
    assert "threads" not in byName and "threadsSplit" not in byName
    assert parser.parse_args(["--threads", "2"]).threads == 2


def test_manualAndVersion_areItsOwn(capsys: pytest.CaptureFixture[str]) -> None:
    """``--man`` names whisper-transcribe and the model; ``--version`` prints both versions."""
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--man"])
    assert exc.value.code == 0
    out: str = capsys.readouterr().out
    assert "whisper-transcribe" in out and WHISPER_MODEL_NAME in out and "TAGS" in out
    with pytest.raises(SystemExit):
        parser.parse_args(["--version"])
    expected: str = f"pl-whisper {__version__} (Whisper {WHISPER_MODEL_NAME})"
    assert capsys.readouterr().out.strip() == expected


def test_chris_plugin_info_describesThePlugin_withOneGpu() -> None:
    """chris_plugin_info describes the package with one GPU, at least and at most."""
    binDir: Path = Path(sys.executable).parent
    image: str = f"docker.io/fnndsc/pl-whisper:{__version__}"
    spec = JobSpec(
        argv=[str(binDir / "chris_plugin_info"), "--dock-image", image, "pl-whisper"],
        env={"PATH": f"{binDir}{os.pathsep}{os.environ.get('PATH', '')}"},
    )
    result = job_run(spec)
    assert result.status, [*result.errors, result.stderr]
    description = json.loads(result.stdout)
    assert description["type"] == "ds" and description["selfexec"] == "pl-whisper"
    assert description["version"] == __version__ and WHISPER_MODEL_NAME in description["title"]
    assert description["documentation"] == "https://github.com/FNNDSC/pl-whisper"
    assert (description["min_gpu_limit"], description["max_gpu_limit"]) == (1, 1)
    assert description["min_memory_limit"] == "8Gi"
    names: set[str] = {parameter["name"] for parameter in description["parameters"]}
    assert "threads" not in names and "exec" in names
