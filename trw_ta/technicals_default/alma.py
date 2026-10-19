"""ALMA re-export — canonical implementation lives in ``trw_ta.ta_core``.

Import ``alma`` from ``trw_ta`` or ``trw_ta.ta_core``; this module exists so the
indicator also appears under ``trw_ta.technicals_default``.
"""
from ..ta_core import alma

__all__ = ["alma"]
