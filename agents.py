"""Compatibility shim — agent lives in main.py now."""

from main import SupplyChainAgent, run_pipeline

__all__ = ["SupplyChainAgent", "run_pipeline"]
