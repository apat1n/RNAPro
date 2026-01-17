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

import os
import json
import numpy as np
import pandas as pd
from biotite.structure.io import pdbx


class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None


def create_input_json(sequence, target_id, verbose=False):
    if verbose:
        print("input_no_msa")
    input_json = [
        {
            "sequences": [
                {
                    "rnaSequence": {
                        "sequence": sequence,
                        "count": 1,
                    }
                }
            ],
            "name": target_id,
        }
    ]
    return input_json


def extract_c1_coordinates(cif_file_path):
    try:
        # Read the CIF file using the correct biotite method
        with open(cif_file_path, "r") as f:
            cif_data = pdbx.CIFFile.read(f)

        # Get structure from CIF data
        atom_array = pdbx.get_structure(cif_data, model=1)

        # Clean atom names and find C1' atoms
        atom_names_clean = np.char.strip(atom_array.atom_name.astype(str))
        mask_c1 = atom_names_clean == "C1'"
        c1_atoms = atom_array[mask_c1]

        if len(c1_atoms) == 0:
            print(f"Warning: No C1' atoms found in {cif_file_path}")
            return None

        # Sort by residue ID and return coordinates
        sort_indices = np.argsort(c1_atoms.res_id)
        c1_atoms_sorted = c1_atoms[sort_indices]
        c1_coords = c1_atoms_sorted.coord

        return c1_coords
    except Exception as e:
        print(f"Error extracting C1' coordinates from {cif_file_path}: {e}")
        return None


def process_sequence(sequence, target_id, temp_dir, verbose=False):
    if verbose:
        print(f"Processing {target_id}: {sequence}")

    # Create input JSON
    input_json = create_input_json(sequence, target_id, verbose=verbose)

    # Save JSON to temporary file
    os.makedirs(temp_dir, exist_ok=True)
    input_json_path = os.path.join(temp_dir, f"{target_id}_input.json")
    with open(input_json_path, "w") as f:
        json.dump(input_json, f, indent=4)


def solution_to_submit_df(solution):
    submit_df = []
    for k, s in solution.items():
        df = coord_to_df(s.sequence, s.coord, s.target_id)
        submit_df.append(df)

    submit_df = pd.concat(submit_df)
    return submit_df


def coord_to_df(sequence, coord, target_id):
    L = len(sequence)
    df = pd.DataFrame()
    df["ID"] = [f"{target_id}_{i + 1}" for i in range(L)]
    df["resname"] = [s for s in sequence]
    df["resid"] = [i + 1 for i in range(L)]

    num_coord = len(coord)
    for j in range(num_coord):
        df[f"x_{j+1}"] = coord[j][:, 0]
        df[f"y_{j+1}"] = coord[j][:, 1]
        df[f"z_{j+1}"] = coord[j][:, 2]
    return df


def update_inference_configs(configs, N_token):
    # Setting the default inference configs for different N_token and N_atom
    # when N_token is larger than 3000, the default config might OOM even on a
    # A100 80G GPUS,
    if N_token > 3840:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = False
    elif N_token > 2560:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = True
    else:
        configs.skip_amp.confidence_head = True
        configs.skip_amp.sample_diffusion = True
    return configs


# data helper
def make_dummy_solution(valid_df):
    solution = dotdict()
    for i, row in valid_df.iterrows():
        target_id = row.target_id
        sequence = row.sequence
        solution[target_id] = dotdict(
            target_id=target_id,
            sequence=sequence,
            coord=[],
        )
    return solution
