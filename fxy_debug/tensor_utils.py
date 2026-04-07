"""Tensor debugging utilities for PaddlePaddle."""

from __future__ import annotations

import inspect
import re
from typing import Any, Union


def dump_tensor(*tensors: Any, title: str = "") -> None:
    """Print a formatted table showing metadata of PaddlePaddle tensors.

    Automatically captures variable names from the call-site source code.
    For non-Tensor arguments, prints type and truncated value instead.

    Args:
        *tensors: One or more ``paddle.Tensor`` objects (or any values).
        title: Optional title string displayed in the banner line.

    Returns:
        None. Output is printed to stdout.

    Example::

        import paddle
        from fxy_debug import dump_tensor

        x = paddle.randn([2, 3])
        y = paddle.zeros([4, 5], dtype="float16")
        dump_tensor(x, y, title="check shapes")
    """
    try:
        import paddle
    except ImportError:
        raise ImportError(
            "PaddlePaddle is required for dump_tensor. "
            "Install it with: pip install paddlepaddle"
        )

    # --- resolve caller variable names from source ---
    frame = inspect.currentframe()
    assert frame is not None
    caller = frame.f_back
    assert caller is not None

    names: list[str] = []
    try:
        filename = inspect.getfile(caller)
        lineno = caller.f_lineno
        with open(filename, "r") as f:
            lines = f.readlines()
        # read from the call line onward until parentheses balance
        src = ""
        for line in lines[lineno - 1:]:
            src += line
            if src.count("(") <= src.count(")"):
                break
        src = " ".join(src.split())  # collapse whitespace
        m = re.search(r"dump_tensor\((.+)\)", src)
        raw = re.sub(r",?\s*title\s*=\s*['\"].*?['\"]", "", m.group(1) if m else "")
        names = [a.strip() for a in raw.split(",") if a.strip()]
    except Exception:
        pass

    # --- build table rows ---
    headers = ("name", "shape", "strides", "dtype", "place")
    rows: list[tuple[str, ...]] = [headers]

    for i, t in enumerate(tensors):
        name = names[i] if i < len(names) else f"tensor[{i}]"
        if not isinstance(t, paddle.Tensor):
            str_t = str(t) + " " * max(0, 50 - len(str(t)))
            rows.append((
                name,
                "N/A",
                "N/A",
                str(type(t).__name__),
                f"value:{str_t[:50]}",
            ))
            continue
        strides = (
            str(list(t.get_strides())) if hasattr(t, "get_strides") else "N/A"
        )
        rows.append((
            name,
            str(list(t.shape)),
            strides,
            str(t.dtype),
            str(t.place),
        ))

    # --- pretty print ---
    col_widths = [max(len(r[c]) for r in rows) for c in range(len(headers))]
    print(f"\n── dump_tensor {title} {'─' * 40}")
    header_row, *data_rows = rows
    print("  " + "  ".join(f"{header_row[c]:<{col_widths[c]}}" for c in range(len(headers))))
    print("  " + "  ".join("─" * col_widths[c] for c in range(len(headers))))
    for r in data_rows:
        print("  " + "  ".join(f"{r[c]:<{col_widths[c]}}" for c in range(len(headers))))
    print()
