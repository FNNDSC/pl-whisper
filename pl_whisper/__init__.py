"""pl-whisper: a ChRIS DS plugin that transcribes every recording of its input tree with Whisper.

Built on pl-shexec: this package contributes a spec, the ``whisper-transcribe``
command the spec runs, and the image with the model.

Key components:
    - SPEC, COMMAND, MEDIA_FILTER, WHISPER_MODEL_NAME: what the plugin says about itself
    - Segment, Transcript, TranscribeOptions: a transcription's records
    - the records of shexec and everything below, re-exported through pl_whisper.models

Typical usage:
    from pl_whisper import SPEC

Dependencies:
    - chris_plugin, shexec, faster-whisper
"""

from __future__ import annotations

__version__: str = "1.0.0"

from pl_whisper.models import (
    FileJob,
    JobResult,
    Plugin,
    PluginSpec,
    RunResult,
    Segment,
    Span,
    TranscribeOptions,
    Transcript,
)
from pl_whisper.pl_whisper import COMMAND, MEDIA_FILTER, SPEC, WHISPER_MODEL_NAME

__all__: list[str] = [
    "COMMAND",
    "MEDIA_FILTER",
    "SPEC",
    "WHISPER_MODEL_NAME",
    "FileJob",
    "JobResult",
    "Plugin",
    "PluginSpec",
    "RunResult",
    "Segment",
    "Span",
    "TranscribeOptions",
    "Transcript",
    "__version__",
]
