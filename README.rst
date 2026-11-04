pl-whisper
==========

**Whisper large-v3 over every recording of an input tree, on a GPU, as a ChRIS plugin.**

``pl-whisper`` is a `ChRIS <https://chrisproject.org/>`_ DS plugin that transcribes each recording found in its input tree, the first positional argument, ``/incoming`` below, with OpenAI's Whisper large-v3 model run by `faster-whisper <https://github.com/SYSTRAN/faster-whisper>`_. Each transcript is written to the output tree, the second positional argument, ``/outgoing``, in the directory matching the recording's own, as five files named after the recording: plain text, SubRip and WebVTT subtitles, a tab-separated table and JSON. The common case needs no flags:

.. code-block:: bash

    pl-whisper /incoming /outgoing

In ChRIS the plugin asks for one GPU. The model is in the image, so it needs no network. It reads any common video or audio file itself, and in the ``Transcribe`` pipeline it follows `pl-ffmpeg <https://github.com/FNNDSC/pl-ffmpeg>`_, which hands it 16 kHz mono WAV.

pl-whisper has no engine of its own. It is `pl-shexec <https://github.com/FNNDSC/pl-shexec>`_, the plugin that runs a command template once per file of a tree, built into an image that holds the model and given ``whisper-transcribe``, this package's own small command, as its default. Walking the tree, choosing the files, running the command and recording each run are pl-shexec's code.

What it runs
------------

The default command is a template that pl-shexec resolves once per recording:

.. code-block:: text

    whisper-transcribe %{inputWorkingDir}/%{inputWorkingFile} %{outputWorkingDir}

``whisper-transcribe`` decodes the recording, detects the language from its first thirty seconds, transcribes it, prints each segment as it is recognised, and writes the five files:

.. code-block:: text

    /incoming/weekly.wav     ->  /outgoing/weekly.txt    one segment to a line
                                 /outgoing/weekly.srt    SubRip subtitles
                                 /outgoing/weekly.vtt    WebVTT subtitles
                                 /outgoing/weekly.tsv    start and end in ms, text
                                 /outgoing/weekly.json   segments, language, device, model

The formats and their layout are openai-whisper's, so tools that read its output read these. The command's own options go in the template:

.. code-block:: bash

    pl-whisper --exec "whisper-transcribe --language en --formats txt,srt
                       %{inputWorkingDir}/%{inputWorkingFile} %{outputWorkingDir}" \
        /incoming /outgoing

``--language`` skips detection, ``--task translate`` gives an English translation, ``--formats`` writes fewer files, ``--device cpu`` keeps off the GPU, ``--model`` names another model or a model directory. ``pl-whisper --man`` lists the plugin's flags, tags and functions; ``whisper-transcribe --help`` the command's.

Why a command of its own
------------------------

The commands that already wrap faster-whisper (``whisper-ctranslate2``) and Whisper itself (``whisper``) report a file that failed to transcribe on standard error and exit 0. pl-shexec's record of a run is its exit status, so with either of them a failed transcription would be recorded as a success. ``whisper-transcribe`` exits 1 when the recording cannot be decoded, the model cannot be loaded or the transcription fails, and 0 only when the files are written.

GPU, and the CPU when there is none
-----------------------------------

The plugin's description asks ChRIS for exactly one GPU, and on Kubernetes the job is scheduled on a node that has one free. On the GPU the model runs in the best precision CTranslate2 finds for it, half precision on most. When the GPU cannot be used, a CUDA library that will not load or too little GPU memory, ``whisper-transcribe`` says so on standard error and runs the recording on the CPU in 8-bit integers, slower than the recording's own length on most machines but finished. When no GPU is visible at all, it goes straight to the CPU. ``jobs.jsonl`` and the transcript's JSON record which device ran.

The image carries CUDA 12's cuBLAS and cuDNN 9 from NVIDIA's pip wheels, which is all CTranslate2 needs; the node's driver, 525 or later, comes in through the NVIDIA container runtime.

Which files are recordings
--------------------------

Every file, at any depth, that passes ``--fileFilter``: by default one regular expression, blind to case, for ``.mp4``, ``.m4a``, ``.m4v``, ``.mov``, ``.mkv``, ``.webm``, ``.avi``, ``.mp3``, ``.wav``, ``.flac``, ``.ogg``, ``.oga``, ``.opus``, ``.aac`` and ``.wma``, the same set as pl-ffmpeg's. ``--analyzeFileIndex v``, the default, runs every recording.

What each run leaves behind
---------------------------

Beside the transcripts of each directory, ``jobs.jsonl`` holds one line per recording: the command, its output with every segment, its exit status, the host and the timing. The plugin's own run is one line of JSON on standard error. A recording that fails is recorded and the run goes on; the plugin exits 0 when every recording of at least one directory completed, 1 otherwise, 2 on a usage error.

One recording at a time
-----------------------

One instance transcribes its recordings one after another on its one GPU. ``--threads`` is not among the parameters ChRIS is given, since each thread would load its own copy of the model onto the one GPU. From a shell, ``--threads N`` runs up to N directories at once, capped at the memory the process may use divided by 8 GiB.

The Transcribe pipeline
-----------------------

``pipelines/Transcribe.yml`` is the pipeline: pl-ffmpeg, then pl-whisper with one GPU, a linear chain. Run on a node holding recordings, it leaves the WAVs in one node and the transcripts in the next. To add it to a CUBE that has both plugins:

.. code-block:: bash

    curl -u user:password -F fname=@pipelines/Transcribe.yml \
        https://cube.example.org/api/v1/pipelines/sourcefiles/

From a local recording, as one script
-------------------------------------

``examples/transcribe.mise`` takes a recording on your own machine through BCH's CUBE in six lines of `chell <https://github.com/FNNDSC/mise>`_, the ChRIS shell: it uploads the recording into ``~/uploads/transcribe/<LABEL>``, runs the ``Transcribe`` pipeline from that folder, which roots a new feed on it, checks that pl-whisper finished, and downloads the pipeline's leaf, the transcripts. Played as

.. code-block:: bash

    play examples/transcribe.mise --param AUDIO=~/meetings/weekly.mp4 --param LABEL=weekly-2026-11-03

it leaves ``weekly.txt``, ``.srt``, ``.vtt``, ``.tsv`` and ``.json`` under ``./data/home_chris_uploads_transcribe_weekly-2026-11-03/``, with the run's ``jobs.jsonl``. A minute of speech takes about a minute and a half, nearly all of it the pipeline. ``PIPELINE`` defaults to the pipeline's ``/bin`` name on BCH's CUBE, ``Transcribe_id293``; another CUBE names it differently.

Running it
----------

From ChRIS, run the ``Transcribe`` pipeline on a node holding recordings, or attach pl-whisper to such a node directly. From the command line, with the image, on a machine with an NVIDIA GPU and the NVIDIA container runtime:

.. code-block:: bash

    docker run --rm --gpus 1 -v $PWD/in:/incoming:ro -v $PWD/out:/outgoing:rw fnndsc/pl-whisper \
        pl-whisper /incoming /outgoing

Without ``--gpus`` the same command runs on the CPU.

Built on pl-shexec
------------------

This plugin is pl-shexec's ``plugin_build`` called with a spec: the name, the version, the title and resource limits ChRIS sees, one GPU among them (``minGpu``, from shexec 2.7), the default template, and the manual's words. The filters, the tags, the records and the exit status are pl-shexec's and are documented there. ``pl-whisper --version`` prints the plugin's version and the model's name; ``docs/history.adoc`` records which model and faster-whisper each plugin version carried.

Development
-----------

.. code-block:: bash

    uv venv --python /usr/bin/python3.14 .venv
    uv pip install --python .venv/bin/python -e ".[dev]"
    .venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/python style_check.py pl_whisper tests
    .venv/bin/pytest -q

The tests run the plugin in-process with a stand-in ``whisper-transcribe`` on ``PATH``, and ``whisper-transcribe`` with a stand-in model, so they need neither a GPU nor a model; the formats' doctests run with them. With ``WHISPER_TEST_MODEL`` set to a model name or directory and ``WHISPER_TEST_AUDIO`` to a WAV of speech, one more test transcribes it for real. ``ruff.toml`` and ``style_check.py`` are the lint configuration of the `style guide <https://github.com/rudolphpienaar/style-guides>`_. The CI workflow lints and tests on Python 3.11 to 3.14, frees the runner's disk, builds the image on every push to ``main``, pushes it to Docker Hub and ghcr.io, and keeps the plugin's description as a build artifact for registering it on a CUBE. The tag must match ``__version__`` in ``pl_whisper/__init__.py``.

``docs/purpose.adoc`` says why the plugin exists, ``docs/architecture.adoc`` how it is built, ``docs/history.adoc`` where it came from.
