"""whisper-transcribe: the files it writes, its fallback from GPU to CPU, and its exit status."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pl_whisper import transcribe
from pl_whisper.transcribe import main

SPEECH_RATE: int = 16_000


@dataclass(frozen=True)
class _Piece:
    """A segment as faster-whisper yields it, the fields whisper-transcribe reads.

    Attributes:
        start: its start, in seconds.
        end: its end, in seconds.
        text: its text, with faster-whisper's leading space.
        avg_logprob: the mean log probability of its tokens.
        no_speech_prob: the probability it holds no speech.
    """

    start: float
    end: float
    text: str
    avg_logprob: float = -0.25
    no_speech_prob: float = 0.01


@dataclass(frozen=True)
class _Info:
    """What faster-whisper says about a transcription.

    Attributes:
        language: the language detected.
        language_probability: how sure detection was.
        duration: the recording's length, in seconds.
    """

    language: str = "en"
    language_probability: float = 0.97
    duration: float = 4.0


def _fakeModel_make(failOn: set[str], loaded: list[str]) -> Callable[..., SimpleNamespace]:
    """Make a stand-in for faster-whisper's WhisperModel.

    Args:
        failOn: the devices on which transcription raises, as a missing CUDA library would.
        loaded: a list each load appends its device to, for the test to read.

    Returns:
        A loader called as WhisperModel is, giving a model whose ``transcribe``
        yields two segments lazily, as faster-whisper's does, or fails.
    """

    def model_load(model: str, device: str, compute_type: str) -> SimpleNamespace:
        """Load nothing; remember the device.

        Args:
            model: the model name or path.
            device: the device.
            compute_type: the precision.

        Returns:
            The stand-in model.
        """
        loaded.append(device)

        def pieces_yield() -> Iterator[_Piece]:
            """Yield the segments, failing on the first when the device is to fail.

            Yields:
                Two segments.

            Raises:
                RuntimeError: on a device in ``failOn``.
            """
            if device in failOn:
                raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
            yield _Piece(0.0, 2.0, " Good morning everyone.")
            yield _Piece(2.5, 4.0, " First item.")

        return SimpleNamespace(transcribe=lambda _audio, **_options: (pieces_yield(), _Info()))

    return model_load


def _audio_make(root: Path) -> Path:
    """Make a recording's path; its content is never read, decode_audio is stubbed.

    Args:
        root: the directory to make it in.

    Returns:
        The path.
    """
    path: Path = root / "weekly.final.wav"
    path.write_bytes(b"RIFF")
    return path


def _transcribe_stub(
    monkeypatch: pytest.MonkeyPatch, cudaCount: int, failOn: set[str] | None = None
) -> list[str]:
    """Stub decoding, the GPU count and the model.

    Args:
        monkeypatch: pytest's monkeypatch.
        cudaCount: how many GPUs CTranslate2 is to see.
        failOn: the devices on which transcription fails.

    Returns:
        The list each model load appends its device to.
    """
    loaded: list[str] = []
    monkeypatch.setattr(transcribe, "decode_audio", lambda _path: np.zeros(SPEECH_RATE, np.float32))
    monkeypatch.setattr(transcribe.ctranslate2, "get_cuda_device_count", lambda: cudaCount)
    monkeypatch.setattr(transcribe, "WhisperModel", _fakeModel_make(failOn or set(), loaded))
    return loaded


def test_writesFiveFiles_namedAfterTheRecording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """On a GPU that works: five files, each named after the recording less its last extension."""
    audioFile: Path = _audio_make(tmp_path)
    loaded: list[str] = _transcribe_stub(monkeypatch, cudaCount=1)
    outputDir: Path = tmp_path / "out"
    assert main([str(audioFile), str(outputDir)]) == 0
    assert sorted(p.name for p in outputDir.iterdir()) == [
        f"weekly.final.{ext}" for ext in ("json", "srt", "tsv", "txt", "vtt")
    ]
    assert (outputDir / "weekly.final.txt").read_text() == "Good morning everyone.\nFirst item.\n"
    assert (
        "00:00:02,500 --> 00:00:04,000\nFirst item." in (outputDir / "weekly.final.srt").read_text()
    )
    document = json.loads((outputDir / "weekly.final.json").read_text())
    assert document["device"] == "cuda" and document["compute_type"] == "auto"
    assert document["text"] == "Good morning everyone. First item." and document["language"] == "en"
    assert loaded == ["cuda"]
    captured = capsys.readouterr()
    assert "[     0.00 -->      2.00] Good morning everyone." in captured.out
    assert "2 segments, en, 4.0 s, on cuda" in captured.err


def test_gpuFailure_fallsBackToCpu_andSaysSo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the GPU fails, the CPU runs it in int8, and standard error says why."""
    audioFile: Path = _audio_make(tmp_path)
    loaded: list[str] = _transcribe_stub(monkeypatch, cudaCount=1, failOn={"cuda"})
    outputDir: Path = tmp_path / "out"
    assert main([str(audioFile), str(outputDir), "--formats", "json"]) == 0
    assert loaded == ["cuda", "cpu"]
    document = json.loads((outputDir / "weekly.final.json").read_text())
    assert (document["device"], document["compute_type"]) == ("cpu", "int8")
    assert "cuda failed (RuntimeError: Library libcublas.so.12" in capsys.readouterr().err


def test_noGpu_runsOnCpu_withoutTryingCuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no GPU in sight, auto goes straight to the CPU."""
    audioFile: Path = _audio_make(tmp_path)
    loaded: list[str] = _transcribe_stub(monkeypatch, cudaCount=0)
    assert main([str(audioFile), str(tmp_path / "out"), "--formats", "txt"]) == 0
    assert loaded == ["cpu"]


def test_cudaAsked_noFallback_exits1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--device cuda`` that fails is a failed run: exit 1, no files."""
    audioFile: Path = _audio_make(tmp_path)
    loaded: list[str] = _transcribe_stub(monkeypatch, cudaCount=1, failOn={"cuda"})
    outputDir: Path = tmp_path / "out"
    assert main([str(audioFile), str(outputDir), "--device", "cuda"]) == 1
    assert loaded == ["cuda"] and not outputDir.exists()
    assert "RuntimeError" in capsys.readouterr().err


def test_undecodableRecording_exits1_beforeAnyModel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recording that cannot be decoded fails once, before a model is loaded."""
    audioFile: Path = _audio_make(tmp_path)
    loaded: list[str] = _transcribe_stub(monkeypatch, cudaCount=1)

    def decode_fail(_path: str) -> None:
        """Fail as PyAV does on a file that is not media.

        Args:
            _path: the recording.

        Raises:
            ValueError: always.
        """
        raise ValueError("Invalid data found when processing input")

    monkeypatch.setattr(transcribe, "decode_audio", decode_fail)
    assert main([str(audioFile), str(tmp_path / "out")]) == 1
    assert loaded == []


def test_unknownFormat_isAUsageError(tmp_path: Path) -> None:
    """An unknown format is refused by the parser, with argparse's status 2."""
    audioFile: Path = _audio_make(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main([str(audioFile), str(tmp_path), "--formats", "txt,docx"])
    assert exc.value.code == 2


@pytest.mark.skipif(
    not os.environ.get("WHISPER_TEST_MODEL"), reason="set WHISPER_TEST_MODEL to a model"
)
def test_realModel_transcribesSpeech(tmp_path: Path) -> None:
    """With a real model and a WAV of speech in WHISPER_TEST_AUDIO, a transcript with words."""
    audio: Path = Path(os.environ["WHISPER_TEST_AUDIO"])
    outputDir: Path = tmp_path / "out"
    assert main([str(audio), str(outputDir), "--model", os.environ["WHISPER_TEST_MODEL"]]) == 0
    assert len((outputDir / f"{audio.stem}.txt").read_text().split()) > 3
