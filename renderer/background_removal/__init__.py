"""Background removal subsystem.

Kept as an isolated component (plan §21) so the underlying implementation
(RemBG onnx model today) can be swapped for another provider without
touching the editor or queue.

Background removal happens HERE at render time — the editor deliberately
shows the full generated character image (plan §6)."""
from .rembg import backend_name, remove_background

__all__ = ["backend_name", "remove_background"]