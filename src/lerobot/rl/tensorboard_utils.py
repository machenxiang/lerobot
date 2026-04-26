#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """A helper class to log metrics using TensorBoard."""

    def __init__(self, log_dir: Path, config: dict | None = None):
        self.log_dir = Path(log_dir) / "tensorboard"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.config = config
        logging.info(f"TensorBoard logs will be saved to {self.log_dir}")
        logging.info(f"Start tensorboard with: tensorboard --logdir={log_dir}")

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        """
        Log a dictionary of metrics to TensorBoard.

        Args:
            d: Dictionary of metrics to log
            step: Global step counter
            mode: "train" or "eval"
            custom_step_key: Key to use as step when doing async logging
        """
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        # Get the step value
        if custom_step_key is not None:
            step_value = d.get(custom_step_key, step)
        else:
            step_value = step

        # Log each metric
        for key, value in d.items():
            if isinstance(value, (int, float)):
                tag = f"{mode}/{key}"
                self.writer.add_scalar(tag, value, step_value)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        """Log a video to TensorBoard."""
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        # TensorBoard doesn't support video directly, so we log frames
        # For now, skip video logging as it requires additional processing
        logging.warning("TensorBoard video logging not fully implemented")

    def log_policy(self, checkpoint_dir: Path):
        """Log policy checkpoint info (no-op for TensorBoard)."""
        # TensorBoard doesn't have a direct equivalent to WandB artifacts
        pass

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()