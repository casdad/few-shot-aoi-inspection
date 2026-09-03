"""Anomaly detector branches used by the original AOI few-shot scheme."""

from .dinov2_spatial import DINOv2SpatialPrototype
from .multiscale_patch_memory import MultiScalePatchMemory

__all__ = ["DINOv2SpatialPrototype", "MultiScalePatchMemory"]
