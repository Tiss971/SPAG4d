"""Shared tqdm wrapper that stays readable when stdout/stderr is redirected to a file.

tqdm normally redraws in place with \\r, which only works on a live TTY -- piped to a
log file (nohup, batch runs, CI), every update becomes its own line and floods the log.
This keeps tqdm for interactive use but raises mininterval under non-TTY output so it
emits periodic snapshots instead.
"""
import sys

from tqdm import tqdm as _tqdm

_NON_TTY_MININTERVAL = 5.0


def log_tqdm(*args, **kwargs):
    if not sys.stderr.isatty():
        kwargs.setdefault("mininterval", _NON_TTY_MININTERVAL)
    return _tqdm(*args, **kwargs)


# Callers also use tqdm.write(...) as a classmethod (e.g. segment_with_flows) --
# expose it here so `from .progress import log_tqdm as tqdm` doesn't break that.
log_tqdm.write = _tqdm.write
