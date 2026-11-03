"""The ``pl-whisper`` command: the plugin, built by shexec's factory from this package's spec.

Key components:
    - PLUGIN: the built plugin
    - parser, main: its flags and the console script's target

Typical usage:
    pl-whisper /incoming /outgoing
    pl-whisper --man

Dependencies:
    - shexec.plugin, pl_whisper.pl_whisper
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from shexec.plugin import plugin_build

from pl_whisper.models import Plugin
from pl_whisper.pl_whisper import SPEC

PLUGIN: Plugin = plugin_build(SPEC)
parser: argparse.ArgumentParser = PLUGIN.parser
main: Callable[..., int] = PLUGIN.main
