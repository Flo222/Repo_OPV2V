"""Read-only ARCE diagnostic helpers.

All audit helpers are disabled by default and must never modify tensors used by
normal inference.
"""

from .compression_auditor import CompressionAuditor

__all__ = ["CompressionAuditor"]
