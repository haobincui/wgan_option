"""Workbook merge logic for building training-ready XLSX files."""

from wgan_option.merge.merge_svi_core import build_svi_workbook_frames
from wgan_option.merge.merge_vol_core import build_vol_workbook_frames

__all__ = ["build_svi_workbook_frames", "build_vol_workbook_frames"]
