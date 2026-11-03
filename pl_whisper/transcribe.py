"""The ``whisper-transcribe`` command: one recording in, its transcript in five formats out.

pl-whisper's template runs this command once per recording. It decodes the recording,
loads the Whisper model with faster-whisper, transcribes, and writes the transcript
beside the recording's counterpart in the output tree as ``<name>.txt``, ``.srt``,
``.vtt``, ``.tsv`` and ``.json``. Each segment is printed as it is recognised.

It exists because the commands that already wrap faster-whisper and openai-whisper
report a file that failed to transcribe on standard error and then exit 0, and
pl-shexec's record of a run is its exit status. Here a recording that cannot be
decoded, a model that cannot be loaded, or a transcription that fails exits 1.

On ``--device auto``, the default, the model runs on the GPU when CTranslate2 sees
one and falls back to the CPU when that fails, a missing CUDA library or a GPU out of
memory, saying so on standard error. The recording is decoded first, so a recording
that cannot be read fails once, not once per device.

Key components:
    - parser_build, options_parse: the command line
    - devices_plan: the devices to try, in order
    - transcript_infer: one transcription on one device
    - transcript_inferOnDevices: the same, falling back down the plan
    - transcript_write: the transcript's files
    - main: the console script's target

Typical usage:
    whisper-transcribe /incoming/weekly.wav /outgoing
    whisper-transcribe --language en --formats txt,srt /incoming/weekly.wav /outgoing

Dependencies:
    - faster-whisper and CTranslate2; for the GPU, CUDA 12's cuBLAS and cuDNN 9
    - the model: ``--model``, else ``$WHISPER_MODEL``, else ``large-v3``, downloaded
      from Hugging Face when it is a name and not a directory
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import ctranslate2
import numpy as np
import numpy.typing as npt
from faster_whisper import WhisperModel, decode_audio

from pl_whisper.formats import RENDERERS
from pl_whisper.models import Segment, TranscribeOptions, Transcript

PROGRAM: str = "whisper-transcribe"
MODEL_ENV: str = "WHISPER_MODEL"
MODEL_FALLBACK: str = "large-v3"
DEVICE_CHOICES: tuple[str, ...] = ("auto", "cuda", "cpu")
TASK_CHOICES: tuple[str, ...] = ("transcribe", "translate")
FORMAT_ALL: str = ",".join(RENDERERS)
# CTranslate2's "auto" picks float16 on a GPU that has it, and what the GPU can do
# otherwise; int8 is the fast and small choice on a CPU.
COMPUTE_BY_DEVICE: dict[str, str] = {"cuda": "auto", "cpu": "int8"}
BEAM_SIZE: int = 5
EXIT_OK: int = 0
EXIT_FAILED: int = 1


def parser_build() -> argparse.ArgumentParser:
    """Build the command line.

    Returns:
        The parser: the recording and the output directory, then the options.

    Example:
        >>> parser_build().parse_args(["a.wav", "out"]).device
        'auto'
    """
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Transcribe one recording with Whisper; write txt, srt, vtt, tsv and json.",
    )
    parser.add_argument("audio", type=Path, help="the recording, any format ffmpeg reads")
    parser.add_argument("outputDir", type=Path, help="the directory to write the transcript to")
    parser.add_argument(
        "--model",
        default=os.environ.get(MODEL_ENV, MODEL_FALLBACK),
        help=f"model name or CTranslate2 model directory; default ${MODEL_ENV}, {MODEL_FALLBACK}",
    )
    parser.add_argument(
        "--language", default=None, help="the language spoken, as en (default: detect)"
    )
    parser.add_argument(
        "--task", choices=TASK_CHOICES, default="transcribe", help="translate gives English"
    )
    parser.add_argument("--formats", default=FORMAT_ALL, help=f"comma-separated, from {FORMAT_ALL}")
    parser.add_argument(
        "--device", choices=DEVICE_CHOICES, default="auto", help="auto tries cuda, then cpu"
    )
    return parser


def options_parse(argv: Sequence[str] | None = None) -> TranscribeOptions:
    """Parse the command line into options.

    Args:
        argv: the arguments, without the program's name; ``None`` reads ``sys.argv``.

    Returns:
        The options, formats split and checked.

    Raises:
        SystemExit: with status 2, from argparse, on a usage error or an unknown format.

    Example:
        >>> options_parse(["a.wav", "out", "--formats", "txt,srt"]).formats
        ('txt', 'srt')
    """
    parser: argparse.ArgumentParser = parser_build()
    namespace: argparse.Namespace = parser.parse_args(argv)
    formats: tuple[str, ...] = tuple(
        name.strip() for name in namespace.formats.split(",") if name.strip()
    )
    unknown: list[str] = [name for name in formats if name not in RENDERERS]
    if unknown or not formats:
        parser.error(
            f"--formats: unknown {', '.join(unknown) or 'empty list'}; choose from {FORMAT_ALL}"
        )
    return TranscribeOptions(
        audio=namespace.audio,
        outputDir=namespace.outputDir,
        model=namespace.model,
        language=namespace.language,
        task=namespace.task,
        formats=formats,
        device=namespace.device,
    )


def devices_plan(device: str, cudaCount: int) -> list[str]:
    """Decide the devices to try, in order.

    Args:
        device: what was asked for, ``auto``, ``cuda`` or ``cpu``.
        cudaCount: how many CUDA devices CTranslate2 sees.

    Returns:
        ``auto`` gives the GPU then the CPU when there is a GPU, else the CPU
        alone; ``cuda`` and ``cpu`` give themselves, with no fallback.

    Example:
        >>> devices_plan("auto", 1), devices_plan("auto", 0), devices_plan("cuda", 0)
        (['cuda', 'cpu'], ['cpu'], ['cuda'])
    """
    if device != "auto":
        return [device]
    return ["cuda", "cpu"] if cudaCount > 0 else ["cpu"]


def transcript_infer(
    audio: npt.NDArray[np.float32], options: TranscribeOptions, device: str
) -> Transcript:
    """Transcribe decoded audio on one device, printing each segment as it comes.

    Args:
        audio: the recording, decoded to 16 kHz mono samples.
        options: the options.
        device: ``cuda`` or ``cpu``.

    Returns:
        The transcript.

    Example:
        >>> transcript_infer(samples, options, "cpu").device  # doctest: +SKIP
        'cpu'
    """
    computeType: str = COMPUTE_BY_DEVICE[device]
    model = WhisperModel(options.model, device=device, compute_type=computeType)
    pieces, info = model.transcribe(
        audio, language=options.language, task=options.task, beam_size=BEAM_SIZE
    )
    segments: list[Segment] = []
    for piece in pieces:
        segment = Segment(
            index=len(segments),
            startTime=piece.start,
            endTime=piece.end,
            text=piece.text.strip(),
            avgLogprob=piece.avg_logprob,
            noSpeechProb=piece.no_speech_prob,
        )
        print(f"[{segment.startTime:9.2f} --> {segment.endTime:9.2f}] {segment.text}", flush=True)
        segments.append(segment)
    return Transcript(
        audio=options.audio,
        model=options.model,
        device=device,
        computeType=computeType,
        language=info.language,
        languageProbability=info.language_probability,
        duration=info.duration,
        segments=tuple(segments),
    )


def transcript_inferOnDevices(
    audio: npt.NDArray[np.float32], options: TranscribeOptions, devices: list[str]
) -> Transcript:
    """Transcribe on the first device that works, saying so when one fails.

    Args:
        audio: the recording, decoded.
        options: the options.
        devices: the devices to try, in order; at least one.

    Returns:
        The transcript from the first device that completed.

    Raises:
        Exception: whatever the last device raised, when every device failed.

    Example:
        >>> transcript_inferOnDevices(samples, options, ["cuda", "cpu"])  # doctest: +SKIP
    """
    for device in devices[:-1]:
        try:
            return transcript_infer(audio, options, device)
        except Exception as err:  # any failure on this device is a reason to try the next
            print(
                f"{PROGRAM}: {device} failed ({type(err).__name__}: {err}); trying {devices[-1]}",
                file=sys.stderr,
            )
    return transcript_infer(audio, options, devices[-1])


def transcript_write(transcript: Transcript, options: TranscribeOptions) -> list[Path]:
    """Write the transcript's files, named after the recording.

    Args:
        transcript: the transcript.
        options: the options, for the output directory and the formats.

    Returns:
        The files written, in the order of ``options.formats``.

    Example:
        >>> transcript_write(transcript, options)  # doctest: +SKIP
        [PosixPath('out/weekly.txt'), PosixPath('out/weekly.srt')]
    """
    options.outputDir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in options.formats:
        path: Path = options.outputDir / f"{options.audio.stem}.{name}"
        path.write_text(RENDERERS[name](transcript), encoding="utf-8")
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command: decode, transcribe, write, and exit with what happened.

    Args:
        argv: the arguments, without the program's name; ``None`` reads ``sys.argv``.

    Returns:
        0 when the transcript was written, 1 when anything failed.

    Example:
        >>> main(["/incoming/weekly.wav", "/outgoing"])  # doctest: +SKIP
        0
    """
    options: TranscribeOptions = options_parse(argv)
    try:
        audio: npt.NDArray[np.float32] = decode_audio(str(options.audio))
        devices: list[str] = devices_plan(options.device, ctranslate2.get_cuda_device_count())
        transcript: Transcript = transcript_inferOnDevices(audio, options, devices)
        written: list[Path] = transcript_write(transcript, options)
    except Exception as err:  # the command's boundary: any failure is a failed transcription
        print(f"{PROGRAM}: {options.audio}: {type(err).__name__}: {err}", file=sys.stderr)
        return EXIT_FAILED
    names: str = ", ".join(path.name for path in written)
    print(
        f"{PROGRAM}: {options.audio.name}: {len(transcript.segments)} segments,"
        f" {transcript.language}, {transcript.duration:.1f} s,"
        f" on {transcript.device}; wrote {names}",
        file=sys.stderr,
    )
    return EXIT_OK
