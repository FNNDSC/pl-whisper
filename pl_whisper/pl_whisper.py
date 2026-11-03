"""The pl-whisper plugin: one Whisper transcription per recording of an input tree.

Given an input tree holding recordings, the plugin runs ``whisper-transcribe`` once per
recording, which transcribes it with Whisper large-v3 through faster-whisper, on the
GPU ChRIS gives it, and writes the transcript as text, subtitles, a table and JSON,
named after the recording, into the matching directory of the output tree. Any other
command line runs through ``--exec``, a template whose tags resolve to the current
directory and file.

The engine that walks the tree, runs the command per file, records each job and sets
the exit status is pl-shexec's. This module holds what pl-whisper adds to it: the
plugin's name, version, title and resource limits for ChRIS, the default template, and
the text of its manual page, all in one ``PluginSpec`` that shexec's factory builds the
plugin from. The model and ``whisper-transcribe`` are in the image.

Key components:
    - WHISPER_MODEL_NAME: the Whisper model the image carries
    - TRANSCRIBE: the default ``--exec`` template
    - MEDIA_FILTER: the default ``--fileFilter``, the names a recording can have
    - SPEC: the spec

Typical usage:
    from pl_whisper.pl_whisper import SPEC

Dependencies:
    - shexec.models
"""

from __future__ import annotations

from pl_whisper import __version__
from pl_whisper.models import PluginSpec

WHISPER_MODEL_NAME: str = "large-v3"

TRANSCRIBE: str = "whisper-transcribe %{inputWorkingDir}/%{inputWorkingFile} %{outputWorkingDir}"

COMMAND: str = TRANSCRIBE

# One case-blind regular expression, since recordings come named .MP4 as often as .mp4.
# The same set as pl-ffmpeg's: faster-whisper decodes any of them itself, and after
# pl-ffmpeg in the Transcribe pipeline only the .wav files are there to match.
MEDIA_FILTER: str = r"re:(?i)\.(mp4|m4a|m4v|mov|mkv|webm|avi|mp3|wav|flac|ogg|oga|opus|aac|wma)$"

BANNER_ART: str = r"""
          _     _
__      _| |__ (_)___ _ __   ___ _ __
\ \ /\ / / '_ \| / __| '_ \ / _ \ '__|
 \ V  V /| | | | \__ \ |_) |  __/ |
  \_/\_/ |_| |_|_|___/ .__/ \___|_|
                     |_|
"""

BANNER_TITLE: str = f"Whisper {WHISPER_MODEL_NAME} over an input tree"

BANNER_TEXT: str = f"""
                      Whisper {WHISPER_MODEL_NAME} over an input tree

                             -- version {{version}} --

        A ChRIS DS plugin that transcribes every recording of its input
        tree with Whisper {WHISPER_MODEL_NAME}, through faster-whisper, on one GPU, and
        writes each transcript as text, SubRip and WebVTT subtitles, a
        table and JSON, named after the recording, in the matching
        directory of the output tree. Without a usable GPU it runs on the
        CPU, much more slowly. In the Transcribe pipeline it follows
        pl-ffmpeg, which hands it 16 kHz mono WAV.
"""

DESCRIPTION: str = f"""
    DESCRIPTION

        ``pl-whisper`` transcribes recordings with Whisper {WHISPER_MODEL_NAME}. The recordings are
        found in its input tree, the <inputDir> positional argument, /incoming
        in the examples, at any depth. Each transcript is written to the
        output tree, the <outputDir> positional argument, /outgoing, in the
        directory matching the recording's own, in five files named after it:

            /incoming/weekly.wav         /outgoing/weekly.txt   plain text
                                         /outgoing/weekly.srt   SubRip subtitles
                                         /outgoing/weekly.vtt   WebVTT subtitles
                                         /outgoing/weekly.tsv   start, end, text
                                         /outgoing/weekly.json  segments and facts

        A file is a recording when it passes ``--fileFilter``, by default a name
        ending, in any case, .mp4, .m4a, .m4v, .mov, .mkv, .webm, .avi, .mp3,
        .wav, .flac, .ogg, .oga, .opus, .aac or .wma. ``--analyzeFileIndex`` v, the
        default, runs every recording. The default command is

            {TRANSCRIBE}

        whisper-transcribe decodes the recording, detects the language from
        its first thirty seconds, transcribes it, prints each segment as it is
        recognised, and writes the five files. Its own options go in the
        template: --language en to skip detection, --task translate for an
        English translation, --formats txt,srt for fewer files, --device cpu
        to keep off the GPU. The model, {WHISPER_MODEL_NAME}, is in the image, so
        the plugin needs no network.

        In ChRIS the plugin asks for one GPU, and the model runs on it in
        CTranslate2's best precision for that GPU, half precision on most.
        When the GPU cannot be used, a missing CUDA library or too little
        memory, whisper-transcribe says so on standard error and runs on the
        CPU in 8-bit integers, a few times slower than the recording's own
        length. Beside the transcripts, jobs.jsonl records, one line per
        recording, the command, its output with every segment, its exit
        status, the host and the timing. At ``--verbosity`` 2 the segments
        stream to the console.

        One plugin instance transcribes its recordings one after another, on
        its one GPU; ``--threads`` is not among the parameters ChRIS is given.
        A recording that fails is recorded and the next one starts. From a
        shell, ``--threads`` N runs up to N directories at once, each loading the
        model, capped to the memory there is at 8 GiB a run.

        The plugin's own run is timed: one line of JSON on standard error.
        The exit status is 0 when every recording of at least one directory
        completed, 1 otherwise, 2 on a usage error.
"""

BRIEF_EXAMPLE: str = """
        pl-whisper /incoming /outgoing
"""

EXAMPLES: str = """

    EXAMPLES

    Every recording transcribed, in five formats, which is the default:

        pl-whisper /incoming /outgoing

    English recordings, subtitles only:

        pl-whisper                                                              \\
            --exec "whisper-transcribe --language en --formats srt,vtt
                    %{inputWorkingDir}/%{inputWorkingFile} %{outputWorkingDir}" \\
            /incoming /outgoing

    The WAV files only, as pl-ffmpeg leaves them:

        pl-whisper --fileFilter 're:\\.wav$' /incoming /outgoing

    From the image, on a machine with an NVIDIA GPU:

        docker run --rm --gpus 1 -v $PWD/in:/incoming:ro -v $PWD/out:/outgoing:rw \\
            fnndsc/pl-whisper pl-whisper /incoming /outgoing
"""

SPEC: PluginSpec = PluginSpec(
    name="pl-whisper",
    version=f"{__version__} (Whisper {WHISPER_MODEL_NAME})",
    title=f"Whisper {WHISPER_MODEL_NAME}: each recording transcribed",
    category="Utility",
    minMemory="8Gi",
    minCpu="2000m",
    minGpu=1,
    defaults={"exec": TRANSCRIBE, "analyzeFileIndex": "v", "fileFilter": MEDIA_FILTER},
    chrisHidden=("threads", "threadsSplit"),
    jobMemory="8Gi",
    bannerArt=BANNER_ART,
    bannerTitle=BANNER_TITLE,
    bannerText=BANNER_TEXT,
    briefExample=BRIEF_EXAMPLE,
    description=DESCRIPTION,
    examples=EXAMPLES,
)
