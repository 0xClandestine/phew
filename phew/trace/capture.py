"""Metal GPU trace capture wrapper.

Usage
-----
    with TraceContext("trace.gputrace") as ctx:
        for _ in range(10):
            mx.eval(fn(*args))

Or manually:
    cap = TraceCapture()
    cap.start("trace.gputrace")
    mx.eval(fn(*args))
    cap.stop()

Requirements:
  - MTL_CAPTURE_ENABLED=1 must be set in the environment before the Python
    process starts (set in the CLI layer, not here)
  - The output path must not already exist (Metal silently fails otherwise)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TraceCapture:
    """Thin wrapper around mx.metal.start_capture / stop_capture."""

    active_path: Path | None = field(default=None, init=False)

    def start(self, path: str | Path) -> None:
        import mlx.core as mx

        path = Path(path)
        if path.suffix != ".gputrace":
            raise ValueError(
                f"Trace path must end in .gputrace, got: {path}\n"
                "Metal requires the .gputrace extension."
            )
        if path.exists():
            raise FileExistsError(
                f"Trace output path already exists: {path}\n"
                "Metal will silently fail if the path exists. Remove it first."
            )
        if not mx.metal.is_available():
            raise RuntimeError("Metal is not available on this system.")

        mx.metal.start_capture(str(path))
        self.active_path = path

    def stop(self) -> Path | None:
        import mlx.core as mx

        if self.active_path is None:
            return None
        mx.metal.stop_capture()
        path = self.active_path
        self.active_path = None
        return path


class TraceContext:
    """Context manager for Metal GPU trace capture.

    Example
    -------
        with TraceContext("trace.gputrace") as ctx:
            for _ in range(10):
                mx.eval(fn(*args))
        print(ctx.path)  # Path to the saved trace
    """

    def __init__(self, path: str | Path, n_iters: int = 10) -> None:
        self.path = Path(path)
        self.n_iters = n_iters
        self._capture = TraceCapture()

    def __enter__(self) -> "TraceContext":
        self._capture.start(self.path)
        return self

    def __exit__(self, *_) -> None:
        self._capture.stop()

    @staticmethod
    def check_env() -> bool:
        """Return True if MTL_CAPTURE_ENABLED=1 is set."""
        return os.environ.get("MTL_CAPTURE_ENABLED") == "1"
