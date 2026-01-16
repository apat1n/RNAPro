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

# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import time
import traceback
import warnings
from typing import Any, Mapping, Optional
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset, DistributedSampler

from biotite.structure.atoms import AtomArray
from rnapro.data.tokenizer import TokenArray
from pathlib import Path

from rnapro.data.json_to_feature import SampleDictToFeatures
from rnapro.data.utils import data_type_transform, make_dummy_feature
from rnapro.utils.distributed import DIST_WRAPPER
from rnapro.utils.torch_utils import collate_fn_identity, dict_to_tensor

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="biotite")


def get_inference_dataloader(configs: Any) -> DataLoader:
    """
    Creates and returns a DataLoader for inference using the InferenceDataset.

    Args:
        configs: A configuration object containing the necessary parameters for the DataLoader.

    Returns:
        A DataLoader object configured for inference.
    """
    inference_dataset = InferenceDataset(
        input_json_path=configs.input_json_path,
        dump_dir=configs.dump_dir,
        use_msa=configs.use_msa,
        use_template=configs.use_template,
        template_data=configs.template_data,
        num_templates=configs.num_templates,
        template_idx=configs.template_idx,
        rna_msa_dir=configs.rna_msa_dir,
    )
    sampler = DistributedSampler(
        dataset=inference_dataset,
        num_replicas=DIST_WRAPPER.world_size,
        rank=DIST_WRAPPER.rank,
        shuffle=False,
    )
    dataloader = DataLoader(
        dataset=inference_dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collate_fn_identity,
        num_workers=configs.num_workers,
    )
    return dataloader


class InferenceDataset(Dataset):
    def __init__(
        self,
        input_json_path: str,
        dump_dir: str,
        use_msa: bool = True,
        configs=None,
        use_template: bool = False,
        template_data: str = None,
        num_templates: int = 4,
        template_idx: int = 0,
        rna_msa_dir: str = None,
    ) -> None:

        self.input_json_path = input_json_path
        self.dump_dir = dump_dir
        self.use_msa = use_msa
        self.rna_msa_dir = Path(rna_msa_dir)
        self.rna_msa_seq_limit = 16384
        self.template_idx = template_idx
        self.num_templates = num_templates
        with open(self.input_json_path, "r") as f:
            self.inputs = json.load(f)

        self.ribonanza_net_tokenizer = {
            "A": 0,
            "C": 1,
            "G": 2,
            "U": 3,
            "PAD": 4,
            "X": 5,
        }
        self.use_template = use_template
        logger.info(template_data)
        self.template_features = torch.load(template_data, weights_only=False)
        # print("- Template loaded : "" template_data)

    def _read_msa_file(self, target_id: str) -> Optional[list[str]]:
        """
        Read MSA sequences from {target_id}.MSA.fasta file.

        Args:
            target_id: Target identifier for filename construction.

        Returns:
            List of RNA sequences (with gaps '-') or None if unavailable.

        Format:
            Standard FASTA with RNA nucleotides (A,C,G,U,N) and gaps (-).
        """

        msa_file_path = self.rna_msa_dir / f"{target_id}.MSA.fasta"
        if not msa_file_path.exists():
            logger.info(f"MSA file not found for target {target_id}: {msa_file_path}")
            return None

        try:
            sequences = []
            with open(msa_file_path, "r") as f:
                logger.info(f"Reading MSA file for target {target_id}: {msa_file_path}")
                current_seq = ""
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if current_seq:
                            sequences.append(current_seq)
                            current_seq = ""
                    else:
                        current_seq += line

                # Add the last sequence
                if current_seq:
                    sequences.append(current_seq)

            return sequences if sequences else None

        except (IOError, OSError) as e:
            logger.warning(f"Error reading MSA file for target {target_id}: {e}")
            return None

    def _create_msa_features(
        self,
        single_sample_dict: Mapping[str, Any],
        atom_array: AtomArray,
        token_array: TokenArray,
        crop_start: int = 0,
        crop_end: Optional[int] = None,
    ) -> dict:
        """
        Generate MSA features from FASTA files with cropping alignment.

        Nucleotide Encoding:
            Local indices: A=0, G=1, C=2, U=3, N=4, gap=5
            Unified indices: A=21, G=22, C=23, U=24, N=25, gap=31

        Args:
            single_sample_dict: Sample dict with cropped target sequence.
            atom_array: AtomArray (unused).
            token_array: TokenArray (unused).
            crop_start: MSA crop start position.
            crop_end: MSA crop end position (None if no cropping).

        Returns:
            Dict with keys: msa, has_deletion, deletion_value, deletion_mean,
            profile, rna_unpair_num_alignments. Empty dict if no MSA available.

        Note:
            MSA sequences cropped to match target sequence length.
            Profile is 32D covering proteins (0-20), RNA (21-25), DNA (26-30), gap (31).
        """
        sample_name = single_sample_dict["name"]

        # Read MSA sequences for this target
        msa_sequences = self._read_msa_file(sample_name)
        if not msa_sequences:
            # Return empty MSA features if no data available
            logger.debug(f"No MSA data available for {sample_name}")
            return {}
        if len(msa_sequences) > self.rna_msa_seq_limit:
            msa_sequences = msa_sequences[: self.rna_msa_seq_limit]

        try:
            from rnapro.data.constants import RNA_NT_TO_ID

            # The sequence in single_sample_dict is already cropped, so use it as-is
            sequence = single_sample_dict["sequences"][0]["rnaSequence"]["sequence"]
            seq_len = len(sequence)

            num_msa_seq = len(msa_sequences)

            # Use official RNA mapping that includes gap character at index 5
            rna_mapping_with_gap = RNA_NT_TO_ID

            # Create MSA feature arrays for cropped sequence length using unified residue indices
            # MSA embedder expects integer indices in unified residue space (converts to one-hot internally)
            from rnapro.data.constants import RNA_STD_RESIDUES, STD_RESIDUES_WITH_GAP

            # Mapping from local RNA_NT_TO_ID indices to unified system indices
            rna_local_to_unified = {
                0: RNA_STD_RESIDUES["A"],  # A: 0 -> 21
                1: RNA_STD_RESIDUES["G"],  # G: 1 -> 22
                2: RNA_STD_RESIDUES["C"],  # C: 2 -> 23
                3: RNA_STD_RESIDUES["U"],  # U: 3 -> 24
                4: RNA_STD_RESIDUES["N"],  # N: 4 -> 25
                5: STD_RESIDUES_WITH_GAP["-"],  # Gap: 5 -> 31
            }

            # Create MSA array with integer indices in unified 32D space
            # MSA embedder expects int64 indices, not one-hot encoding
            msa_array = np.full(
                (num_msa_seq, seq_len), 25, dtype=np.int64
            )  # Default to 'N' (index 25)

            for seq_idx, msa_seq in enumerate(msa_sequences):
                # Apply cropping to MSA sequences to match target sequence cropping
                if crop_end is not None:
                    # Crop MSA sequence to match target sequence cropping
                    msa_seq_cropped = (
                        msa_seq[crop_start:crop_end]
                        if len(msa_seq) > crop_start
                        else msa_seq
                    )
                else:
                    msa_seq_cropped = msa_seq

                for pos_idx, nucleotide in enumerate(msa_seq_cropped):
                    if pos_idx < seq_len:  # Ensure we don't exceed sequence length
                        # Map nucleotide to local index first, then to unified index
                        local_idx = rna_mapping_with_gap.get(nucleotide.upper(), 4)
                        unified_idx = rna_local_to_unified[local_idx]
                        # Store unified index directly
                        msa_array[seq_idx, pos_idx] = unified_idx

            # Create deletion features based on gap characters (index 31 in unified space)
            gap_idx = STD_RESIDUES_WITH_GAP["-"]  # Index 31
            has_deletion = (msa_array == gap_idx).astype(
                np.bool_
            )  # True where gaps are present

            # Apply arctan transformation to deletion counts for consistency with standard pipeline
            # Since we have binary gap presence (0 or 1), this maps: 0 -> 0, 1 -> ~0.187
            deletion_counts = has_deletion.astype(
                np.float32
            )  # Treat gap presence as count=1
            deletion_value = (2 / np.pi) * np.arctan(
                deletion_counts / 3
            )  # Standard arctan transform

            # Calculate deletion_mean (mean deletion probability across MSA sequences)
            deletion_mean = np.mean(deletion_value, axis=0).astype(
                np.float32
            )  # Shape: (seq_len,)

            # Create unified 32-dimensional profile by averaging MSA (including gaps)
            # Profile covers all residue types: proteins (0-20) + RNA (21-25) + DNA (26-30) + gap (31)
            # This matches the standard _make_msa_profile implementation used in InferenceMSAFeaturizer
            all_res_types = np.arange(32)
            res_type_hits = msa_array[..., None] == all_res_types[None, ...]
            res_type_counts = res_type_hits.sum(axis=0)
            profile = (res_type_counts / num_msa_seq).astype(np.float32)

            # Create MSA feature dictionary
            msa_features = {
                "msa": msa_array,
                "has_deletion": has_deletion,
                "deletion_value": deletion_value,
                "deletion_mean": deletion_mean,
                "profile": profile,
                "rna_unpair_num_alignments": np.array([num_msa_seq], dtype=np.int32),
            }

            logger.debug(
                f"Created MSA features for {sample_name} with {num_msa_seq} sequences"
            )
            return msa_features

        except Exception as e:
            logger.warning(f"Error generating MSA features for {sample_name}: {e}")
            return {}

    def process_one(
        self,
        single_sample_dict: Mapping[str, Any],
    ) -> tuple[dict[str, torch.Tensor], AtomArray, dict[str, float]]:
        """
        Processes a single sample from the input JSON to generate features and statistics.

        Args:
            single_sample_dict: A dictionary containing the sample data.

        Returns:
            A tuple containing:
                - A dictionary of features.
                - An AtomArray object.
                - A dictionary of time tracking statistics.
        """
        # general features
        t0 = time.time()
        sample2feat = SampleDictToFeatures(
            single_sample_dict,
        )
        seq = single_sample_dict["sequences"][0]["rnaSequence"]["sequence"]
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()
        features_dict["distogram_rep_atom_mask"] = torch.Tensor(
            atom_array.distogram_rep_atom_mask
        ).long()
        entity_poly_type = sample2feat.entity_poly_type
        t1 = time.time()

        msa_features = self._create_msa_features(
            single_sample_dict, atom_array, token_array
        )

        # Make dummy features for not implemented features
        dummy_feats = ["template"]
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
        features_dict = make_dummy_feature(
            features_dict=features_dict,
            dummy_feats=dummy_feats,
        )

        # Transform to right data type
        feat = data_type_transform(feat_or_label_dict=features_dict)
        feat["seq"] = seq

        t2 = time.time()

        data = {}
        data["input_feature_dict"] = feat

        # Add dimension related items
        N_token = feat["token_index"].shape[0]
        N_atom = feat["atom_to_token_idx"].shape[0]
        N_msa = feat["msa"].shape[0]

        stats = {}
        for mol_type in ["ligand", "protein", "dna", "rna"]:
            mol_type_mask = feat[f"is_{mol_type}"].bool()
            stats[f"{mol_type}/atom"] = int(mol_type_mask.sum(dim=-1).item())
            stats[f"{mol_type}/token"] = len(
                torch.unique(feat["atom_to_token_idx"][mol_type_mask])
            )

        N_asym = len(torch.unique(data["input_feature_dict"]["asym_id"]))
        data.update(
            {
                "N_asym": torch.tensor([N_asym]),
                "N_token": torch.tensor([N_token]),
                "N_atom": torch.tensor([N_atom]),
                "N_msa": torch.tensor([N_msa]),
            }
        )

        def formatted_key(key):
            type_, unit = key.split("/")
            if type_ == "protein":
                type_ = "prot"
            elif type_ == "ligand":
                type_ = "lig"
            else:
                pass
            return f"N_{type_}_{unit}"

        data.update(
            {
                formatted_key(k): torch.tensor([stats[k]])
                for k in [
                    "protein/atom",
                    "ligand/atom",
                    "dna/atom",
                    "rna/atom",
                    "protein/token",
                    "ligand/token",
                    "dna/token",
                    "rna/token",
                ]
            }
        )
        data.update({"entity_poly_type": entity_poly_type})
        t3 = time.time()
        time_tracker = {
            "crop": t1 - t0,
            "featurizer": t2 - t1,
            "added_feature": t3 - t2,
        }

        return data, atom_array, time_tracker

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], AtomArray, str]:
        try:
            single_sample_dict = self.inputs[index]
            sample_name = single_sample_dict["name"]
            logger.info(f"Featurizing {sample_name}...")

            data, atom_array, _ = self.process_one(
                single_sample_dict=single_sample_dict
            )
            error_message = ""
        except Exception as e:
            data, atom_array = {}, None
            error_message = f"{e}:\n{traceback.format_exc()}"
            print("error_message", error_message)
        data["sample_name"] = single_sample_dict["name"]
        data["sample_index"] = index

        sequence = [
            self.ribonanza_net_tokenizer[nt] for nt in data["input_feature_dict"]["seq"]
        ]
        sequence = np.array(sequence)
        sequence = torch.tensor(sequence)
        data["input_feature_dict"]["tokenized_seq"] = sequence

        if self.use_template == "masked_templates":
            print(f"- Using template masked templates #{self.template_idx}")
            if self.use_template and data["sample_name"] in self.template_features:
                template_ca = (
                    torch.from_numpy(
                        self.template_features[data["sample_name"]]["xyz"][
                            :, [self.template_idx]
                        ]
                    )
                    .permute(1, 0, 2)
                    .float()
                )
                data["input_feature_dict"]["template_coords"] = template_ca
                data["input_feature_dict"]["template_coords_mask"] = torch.ones(
                    1, len(sequence), dtype=torch.bool
                )
                data["input_feature_dict"]["n_templates"] = torch.tensor([1])
            else:
                template_ca = torch.ones(1, len(sequence), 3)
                data["input_feature_dict"]["template_coords"] = template_ca
                data["input_feature_dict"]["template_coords_mask"] = torch.ones(
                    1, len(sequence), dtype=torch.bool
                )
                data["input_feature_dict"]["n_templates"] = torch.tensor([1])
        elif self.use_template == "ca_precomputed":
            # template_idx selects top-k templates: 0->top1, 1->top2, 2->top3, 3->top4, 4->top5
            template_combinations = [
                [0],
                [0, 1],
                [0, 1, 2],
                [0, 1, 2, 3],
                [0, 1, 2, 3, 4],
            ]
            print(f"- Using template ca precomputed #{self.template_idx} - combinations: {template_combinations[self.template_idx]}")
            template_ca = (
                torch.from_numpy(
                    self.template_features[data["sample_name"]]["xyz"][
                        :, template_combinations[self.template_idx]
                    ]
                )
                .permute(1, 0, 2)
                .float()
            )

            data["input_feature_dict"]["template_coords"] = template_ca
            data["input_feature_dict"]["template_coords_mask"] = torch.ones(
                len(template_ca), len(sequence)
            )
            data["input_feature_dict"]["n_templates"] = torch.tensor([len(template_ca)])
        return data, atom_array, error_message
