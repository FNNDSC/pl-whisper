"""A transcript rendered as text, subtitles, a table or JSON.

The five formats are the ones openai-whisper's command writes, in the same layout,
so tools that read its output read these: plain text one segment to a line, SubRip
and WebVTT subtitles, a tab-separated table in milliseconds, and JSON with the
segments and the run's facts.

Key components:
    - timestamp_format: seconds as ``HH:MM:SS,mmm``
    - txt_render, srt_render, vtt_render, tsv_render, json_render: one format each
    - RENDERERS: the renderers by format name

Typical usage:
    from pl_whisper.formats import RENDERERS
    text = RENDERERS["srt"](transcript)

Dependencies:
    - pl_whisper.models
"""

from __future__ import annotations

import json
from collections.abc import Callable

from pl_whisper.models import Segment, Transcript

MS_PER_SECOND: int = 1000
SECONDS_PER_HOUR: int = 3600
SECONDS_PER_MINUTE: int = 60


def timestamp_format(seconds: float, separator: str = ",") -> str:
    """Format an offset as ``HH:MM:SS,mmm``, the form subtitles use.

    Args:
        seconds: the offset, in seconds; negative values are taken as 0.
        separator: what precedes the milliseconds, ``,`` for SubRip, ``.`` for WebVTT.

    Returns:
        The offset, hours always shown, milliseconds rounded.

    Example:
        >>> timestamp_format(3725.5)
        '01:02:05,500'
        >>> timestamp_format(1.0004, ".")
        '00:00:01.000'
    """
    totalMs: int = round(max(seconds, 0.0) * MS_PER_SECOND)
    hours, rest = divmod(totalMs, SECONDS_PER_HOUR * MS_PER_SECOND)
    minutes, rest = divmod(rest, SECONDS_PER_MINUTE * MS_PER_SECOND)
    wholeSeconds, ms = divmod(rest, MS_PER_SECOND)
    return f"{hours:02d}:{minutes:02d}:{wholeSeconds:02d}{separator}{ms:03d}"


def txt_render(transcript: Transcript) -> str:
    r"""Render the transcript as plain text, one segment to a line.

    Args:
        transcript: the transcript.

    Returns:
        The text, ending in a newline; empty when there are no segments.

    Example:
        >>> txt_render(_transcriptExample_build())
        'Good morning.\nFirst item.\n'
    """
    return "".join(f"{segment.text}\n" for segment in transcript.segments)


def srt_render(transcript: Transcript) -> str:
    """Render the transcript as SubRip subtitles.

    Args:
        transcript: the transcript.

    Returns:
        Numbered cues from 1, each a time range and the segment's text.

    Example:
        >>> print(srt_render(_transcriptExample_build()), end="")
        1
        00:00:00,000 --> 00:00:02,500
        Good morning.
        <BLANKLINE>
        2
        00:00:03,000 --> 00:00:04,250
        First item.
        <BLANKLINE>
    """
    return "".join(
        f"{number}\n{_cueTimes_format(segment, ',')}\n{segment.text}\n\n"
        for number, segment in enumerate(transcript.segments, start=1)
    )


def vtt_render(transcript: Transcript) -> str:
    """Render the transcript as WebVTT subtitles.

    Args:
        transcript: the transcript.

    Returns:
        The ``WEBVTT`` header, then one cue per segment.

    Example:
        >>> print(vtt_render(_transcriptExample_build()).splitlines()[2])
        00:00:00.000 --> 00:00:02.500
    """
    cues: str = "".join(
        f"{_cueTimes_format(segment, '.')}\n{segment.text}\n\n" for segment in transcript.segments
    )
    return f"WEBVTT\n\n{cues}"


def tsv_render(transcript: Transcript) -> str:
    r"""Render the transcript as a tab-separated table, times in whole milliseconds.

    Args:
        transcript: the transcript.

    Returns:
        A ``start``, ``end``, ``text`` header, then one row per segment, tabs in
        the text replaced by spaces.

    Example:
        >>> tsv_render(_transcriptExample_build()).splitlines()[1].split("\t")
        ['0', '2500', 'Good morning.']
    """
    rows: str = "".join(
        f"{round(segment.startTime * MS_PER_SECOND)}\t{round(segment.endTime * MS_PER_SECOND)}\t"
        f"{segment.text.replace(chr(9), ' ')}\n"
        for segment in transcript.segments
    )
    return f"start\tend\ttext\n{rows}"


def json_render(transcript: Transcript) -> str:
    """Render the transcript as JSON: the text, the segments and the run's facts.

    The keys are openai-whisper's (``text``, ``segments``, ``language``, and per
    segment ``id``, ``start``, ``end``, ``text``, ``avg_logprob``,
    ``no_speech_prob``), with what this run adds beside them.

    Args:
        transcript: the transcript.

    Returns:
        One JSON object, indented, ending in a newline.

    Example:
        >>> json.loads(json_render(_transcriptExample_build()))["segments"][1]["start"]
        3.0
    """
    document: dict[str, object] = {
        "text": transcript.text,
        "language": transcript.language,
        "language_probability": transcript.languageProbability,
        "duration": transcript.duration,
        "model": transcript.model,
        "device": transcript.device,
        "compute_type": transcript.computeType,
        "segments": [_segmentRecord_build(segment) for segment in transcript.segments],
    }
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def _cueTimes_format(segment: Segment, separator: str) -> str:
    """Format a segment's time range as a subtitle cue's first line.

    Args:
        segment: the segment.
        separator: what precedes the milliseconds.

    Returns:
        ``start --> end``.
    """
    startText: str = timestamp_format(segment.startTime, separator)
    return f"{startText} --> {timestamp_format(segment.endTime, separator)}"


def _segmentRecord_build(segment: Segment) -> dict[str, object]:
    """Build a segment's JSON record, in openai-whisper's keys.

    Args:
        segment: the segment.

    Returns:
        The record, ready for ``json.dumps``.
    """
    return {
        "id": segment.index,
        "start": segment.startTime,
        "end": segment.endTime,
        "text": segment.text,
        "avg_logprob": segment.avgLogprob,
        "no_speech_prob": segment.noSpeechProb,
    }


def _transcriptExample_build() -> Transcript:
    """Build the two-segment transcript the examples render.

    Returns:
        A transcript of two short segments in English.
    """
    from pathlib import Path

    return Transcript(
        audio=Path("weekly.wav"),
        model="large-v3",
        device="cpu",
        computeType="int8",
        language="en",
        languageProbability=0.99,
        duration=4.25,
        segments=(
            Segment(0, 0.0, 2.5, "Good morning.", -0.2, 0.01),
            Segment(1, 3.0, 4.25, "First item.", -0.3, 0.02),
        ),
    )


RENDERERS: dict[str, Callable[[Transcript], str]] = {
    "txt": txt_render,
    "srt": srt_render,
    "vtt": vtt_render,
    "tsv": tsv_render,
    "json": json_render,
}
