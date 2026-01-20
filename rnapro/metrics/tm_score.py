# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""
TM-score (Template Modeling score) implementation for RNA structure evaluation.

TM-score ranges from 0 to 1:
- TM-score < 0.17: Random structural similarity
- TM-score > 0.5: Same fold
- TM-score ~1.0: Identical structures

Reference:
Zhang & Skolnick (2004) "Scoring function for automated assessment of protein
structure template quality" Proteins, 57: 702-710.
"""

from typing import Optional

import torch
import torch.nn as nn

from rnapro.metrics.rmsd import align_pred_to_true


class TMScore(nn.Module):
    """
    TM-score metric for RNA structure comparison.

    Computes TM-score after optimal superposition of predicted and true structures.
    Uses C1' atoms (or CA atoms for compatibility) for RNA structures.
    """

    def __init__(self, eps: float = 1e-10):
        """
        Args:
            eps: Small constant for numerical stability.
        """
        super(TMScore, self).__init__()
        self.eps = eps

    @staticmethod
    def compute_d0(L_target: int, is_rna: bool = True) -> float:
        """
        Compute normalization factor d0 for TM-score.

        Args:
            L_target: Length of target structure (number of residues).
            is_rna: If True, use RNA-specific d0; otherwise use protein d0.

        Returns:
            d0: Normalization distance threshold.

        Note:
            Original protein formula: d0 = 1.24 * (L_target - 15)^(1/3) - 1.8
            For RNA, we use the same formula as it works well empirically.
        """
        if L_target <= 15:
            # For very short sequences, use a minimum threshold
            d0 = 0.5
        else:
            d0 = 1.24 * ((L_target - 15) ** (1.0 / 3.0)) - 1.8

        # Ensure d0 is at least 0.5 Angstroms
        d0 = max(d0, 0.5)
        return d0

    def forward(
        self,
        pred_coordinate: torch.Tensor,
        true_coordinate: torch.Tensor,
        coordinate_mask: Optional[torch.Tensor] = None,
        use_ca_only: bool = True,
    ) -> torch.Tensor:
        """
        Compute TM-score between predicted and true coordinates.

        Args:
            pred_coordinate: Predicted coordinates.
                Shape: [N_sample, N_atom, 3] or [N_sample, N_residue, 3]
            true_coordinate: Ground truth coordinates.
                Shape: [N_atom, 3] or [N_residue, 3]
            coordinate_mask: Mask for valid coordinates.
                Shape: [N_atom] or [N_residue]
                If None, all coordinates are considered valid.
            use_ca_only: If True and coordinates are all-atom, extract C1'/CA atoms.
                For RNA structures, this should use C1' atoms (alpha-carbon equivalent).

        Returns:
            tm_score: TM-score for each sample.
                Shape: [N_sample]
        """
        # Handle coordinate mask
        if coordinate_mask is None:
            if true_coordinate.dim() == 2:
                coordinate_mask = torch.ones(
                    true_coordinate.shape[0],
                    device=true_coordinate.device,
                    dtype=torch.bool
                )
            else:
                raise ValueError("coordinate_mask must be provided for batched true_coordinate")

        # Ensure boolean mask
        if coordinate_mask.dtype != torch.bool:
            coordinate_mask = coordinate_mask.bool()

        # Apply mask to coordinates
        pred_masked = pred_coordinate[..., coordinate_mask, :]  # [N_sample, N_valid, 3]
        true_masked = true_coordinate[coordinate_mask, :]  # [N_valid, 3]

        # Get target length (number of residues)
        L_target = coordinate_mask.sum().item()

        if L_target == 0:
            # No valid coordinates, return zero score
            return torch.zeros(pred_coordinate.shape[0], device=pred_coordinate.device)

        # Compute d0 normalization factor
        d0 = self.compute_d0(L_target, is_rna=True)
        d0_sq = d0 ** 2

        # Align predicted structure to true structure
        # This performs optimal rotation and translation
        aligned_pred, _, _ = align_pred_to_true(
            pred_pose=pred_masked,
            true_pose=true_masked.unsqueeze(0).expand(pred_masked.shape[0], -1, -1),
            atom_mask=None,  # All masked coordinates are valid
            weight=None,  # Equal weight for all atoms
            allowing_reflection=False,
        )

        # Compute distances between aligned predicted and true coordinates
        # Distance: [N_sample, N_valid]
        distances_sq = torch.sum(
            (aligned_pred - true_masked.unsqueeze(0)) ** 2,
            dim=-1
        )

        # Compute TM-score using the formula:
        # TM-score = (1/L_target) * Σ [1 / (1 + (d_i / d0)^2)]
        tm_score_terms = 1.0 / (1.0 + distances_sq / d0_sq)
        tm_score = torch.mean(tm_score_terms, dim=-1)  # [N_sample]

        return tm_score


class TMScoreMetrics(nn.Module):
    """
    TM-score metrics wrapper compatible with existing evaluation pipeline.

    Computes TM-score for multiple samples and aggregates results.
    """

    def __init__(self, configs):
        super(TMScoreMetrics, self).__init__()
        self.eps = configs.metrics.get("tm_score", {}).get("eps", 1e-10)
        self.configs = configs
        self.tm_score_base = TMScore(eps=self.eps)

        # Whether to compute TM-score on C1' atoms only or all atoms
        self.use_ca_only = configs.metrics.get("tm_score", {}).get("use_ca_only", True)

    def compute_tm_score(self, pred_dict: dict, label_dict: dict) -> dict:
        """
        Compute TM-score for all samples.

        Args:
            pred_dict: Dictionary containing:
                - coordinate: [N_sample, N_atom, 3]
            label_dict: Dictionary containing:
                - coordinate: [N_atom, 3]
                - coordinate_mask: [N_atom]

        Returns:
            Dictionary with:
                - complex: [N_sample] TM-scores for each sample
        """
        tm_score = self.tm_score_base.forward(
            pred_coordinate=pred_dict["coordinate"],
            true_coordinate=label_dict["coordinate"],
            coordinate_mask=label_dict["coordinate_mask"],
            use_ca_only=self.use_ca_only,
        )

        return {"complex": tm_score}

    def aggregate_tm_score(self, tm_score_dict: dict) -> dict:
        """
        Aggregate TM-scores across samples.

        Args:
            tm_score_dict: Dictionary with "complex" key containing [N_sample] scores.

        Returns:
            Dictionary with aggregated metrics (best, worst, mean, median).
        """
        tm_scores = tm_score_dict["complex"]  # [N_sample]
        N_sample = tm_scores.shape[0]
        median_index = N_sample // 2

        aggregated = {
            "tm_score/complex/best": tm_scores.max(),
            "tm_score/complex/worst": tm_scores.min(),
            "tm_score/complex/mean": tm_scores.mean(),
            "tm_score/complex/median": tm_scores.sort(descending=True)[0][median_index],
            "tm_score/complex/random": tm_scores[0],
        }

        return aggregated