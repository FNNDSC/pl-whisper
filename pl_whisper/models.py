"""The records pl-whisper works with: a transcript's, and shexec's with every layer below.

A transcription produces timed segments of recognised speech and a little about the
recording: the language heard, how sure the model was of it, the duration, and where
the model ran. These are plain data, rendered to files by ``pl_whisper.formats``.
shexec's records are re-exported so a test or a tool imports from here and nothing
below.

Key components:
    - Segment: one timed span of recognised speech
    - Transcript: a recording's segments and what is known about the run
    - TranscribeOptions: what ``whisper-transcribe`` was asked to do
    - PluginSpec, Plugin and the run records: shexec's, re-exported

Typical usage:
    from pl_whisper.models import Segment, Transcript

Dependencies:
    - shexec.models
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shexec.models import (
    DirNode,
    FileJob,
    HostFacts,
    JobResult,
    JobSpec,
    Plugin,
    PluginSpec,
    RunResult,
    ShExecOptions,
    Span,
    StageResult,
)

__all__: list[str] = [
    "DirNode",
    "FileJob",
    "HostFacts",
    "JobResult",
    "JobSpec",
    "Plugin",
    "PluginSpec",
    "RunResult",
    "Segment",
    "ShExecOptions",
    "Span",
    "StageResult",
    "TranscribeOptions",
    "Transcript",
]


@dataclass(frozen=True)
class Segment:
    """One timed span of recognised speech.

    Attributes:
        index: the segment's position in the transcript, from 0.
        startTime: offset of its start from the start of the recording, in seconds.
        endTime: offset of its end, in seconds.
        text: the recognised text, stripped of surrounding whitespace.
        avgLogprob: the mean log probability of its tokens; nearer 0 is surer.
        noSpeechProb: the model's probability that the span holds no speech.

    Example:
        >>> Segment(0, 0.0, 2.5, "Good morning.", -0.2, 0.01).text
        'Good morning.'
    """

    index: int
    startTime: float
    endTime: float
    text: str
    avgLogprob: float
    noSpeechProb: float


@dataclass(frozen=True)
class Transcript:
    """A recording's segments, and what is known about the run that produced them.

    Attributes:
        audio: the recording transcribed.
        model: the model, as named or as the path it was loaded from.
        device: where it ran, ``cuda`` or ``cpu``.
        computeType: the precision CTranslate2 was asked for, such as ``auto``.
        language: the language spoken, detected or as given, as a code such as ``en``.
        languageProbability: how sure detection was, 1.0 when the language was given.
        duration: the recording's length, in seconds.
        segments: the segments, in time order.

    Example:
        >>> transcript = Transcript(
        ...     Path("weekly.wav"), "large-v3", "cpu", "int8", "en", 0.99, 2.5,
        ...     (Segment(0, 0.0, 2.5, "Good morning.", -0.2, 0.01),),
        ... )
        >>> transcript.text
        'Good morning.'
    """

    audio: Path
    model: str
    device: str
    computeType: str
    language: str
    languageProbability: float
    duration: float
    segments: tuple[Segment, ...]

    @property
    def text(self) -> str:
        """The whole transcript as one string, segments joined by spaces."""
        return " ".join(segment.text for segment in self.segments)


@dataclass(frozen=True)
class TranscribeOptions:
    """What ``whisper-transcribe`` was asked to do.

    Attributes:
        audio: the recording to transcribe.
        outputDir: the directory the transcript files are written to.
        model: a model name such as ``large-v3``, or a directory holding a
            CTranslate2 Whisper model.
        language: the language spoken, or ``None`` to detect it.
        task: ``transcribe``, or ``translate`` for an English translation.
        formats: the file formats to write, from ``txt``, ``srt``, ``vtt``, ``tsv``, ``json``.
        device: ``auto``, ``cuda`` or ``cpu``.

    Example:
        >>> TranscribeOptions(Path("a.wav"), Path("out"), "large-v3", None, "transcribe",
        ...                   ("txt",), "auto").formats
        ('txt',)
    """

    audio: Path
    outputDir: Path
    model: str
    language: str | None
    task: str
    formats: tuple[str, ...]
    device: str
