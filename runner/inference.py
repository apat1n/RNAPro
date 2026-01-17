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
import shutil
import logging
import traceback
from contextlib import nullcontext
from os.path import join as opjoin
from typing import Any, Mapping

import torch
import pandas as pd
import numpy as np

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from runner.dumper import DataDumper

from rnapro.config import parse_configs, parse_sys_args
from rnapro.data.infer_data_pipeline import get_inference_dataloader
from rnapro.model.RNAPro import RNAPro
from rnapro.utils.distributed import DIST_WRAPPER
from rnapro.utils.seed import seed_everything
from rnapro.utils.torch_utils import to_device

from rnapro.utils.inference import (
    update_inference_configs,
    make_dummy_solution,
    solution_to_submit_df,
    process_sequence,
    extract_c1_coordinates,
)


class InferenceRunner(object):
    def __init__(self, configs: Any) -> None:
        if configs.logger == "logging":
            self.logger = logging.getLogger(__name__)

        self.configs = configs

        self.init_env()
        self.init_basics()
        self.init_model()
        self.load_checkpoint()
        self.init_dumper(
            need_atom_confidence=configs.need_atom_confidence,
            sorted_by_ranking_score=configs.sorted_by_ranking_score,
        )

    def init_env(self) -> None:
        self.print(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            + f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device("cuda:{}".format(DIST_WRAPPER.local_rank))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            self.print(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        if self.configs.use_deepspeed_evo_attention:
            env = os.getenv("CUTLASS_PATH", None)
            self.print(f"env: {env}")
            assert (
                env is not None
            ), "if use ds4sci, set `CUTLASS_PATH` env as https://www.deepspeed.ai/tutorials/ds4sci_evoformerattention/"
            if env is not None:
                self.print(
                    "The kernels will be compiled when DS4Sci_EvoformerAttention is called for the first time."
                )
        use_fastlayernorm = os.getenv("LAYERNORM_TYPE", None)
        if use_fastlayernorm == "fast_layernorm":
            self.print(
                "The kernels will be compiled when fast_layernorm is called for the first time."
            )

        self.print("Finished initializing environment !")

    def init_basics(self) -> None:
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)

    def init_model(self) -> None:
        self.model = RNAPro(self.configs).to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters())
        self.print(f"Built model ! Total number of parameters: {num_params:,}")

    def load_checkpoint(self) -> None:
        checkpoint_path = self.configs.load_checkpoint_path

        if not os.path.exists(checkpoint_path):
            raise Exception(f"Given checkpoint path not exist [{checkpoint_path}]")
        self.print(
            f"Loading weights from {checkpoint_path}  (strict={self.configs.load_strict})"
        )
        checkpoint = torch.load(checkpoint_path, self.device)

        sample_key = [k for k in checkpoint["model"].keys()][0]
        # self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module."):]: v for k, v in checkpoint["model"].items()
            }
        self.model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=True,
        )
        self.model.eval()
        # self.print("Finish loading checkpoint.")

    def init_dumper(
        self, need_atom_confidence: bool = False, sorted_by_ranking_score: bool = True
    ):
        self.dumper = DataDumper(
            base_dir=self.dump_dir,
            need_atom_confidence=need_atom_confidence,
            sorted_by_ranking_score=sorted_by_ranking_score,
        )

    def print_dict(self, d):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: ", v.shape)
            else:
                pass
                # print(f"{k}: {v}")

    # Adapted from runner.train.Trainer.evaluate
    @torch.no_grad()
    def predict(self, data: Mapping[str, Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]
        # print("eval_precision: ", eval_precision)
        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )
        #         print('input_feature_dict: ', self.print_dict(data["input_feature_dict"]))
        #         exit(0)

        data = to_device(data, self.device)
        with enable_amp:
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
            )

        return prediction

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            if self.configs.logger == "logging":
                self.logger.info(msg)
            elif self.configs.logger == "print":
                print(msg)

    def update_model_configs(self, new_configs: Any) -> None:
        self.model.configs = new_configs


def infer_predict(runner: InferenceRunner, configs: Any) -> None:
    """
    Infer the sequence for a given runner and configs.

    Args:
        runner (InferenceRunner): The runner to be used.
        configs (ConfigDict): The configurations for the inference.
    """
    # Load the dataloader
    try:
        dataloader = get_inference_dataloader(configs=configs)
    except Exception as e:
        error_message = f"{e}:\n{traceback.format_exc()}"
        runner.print(error_message)
        with open(opjoin(runner.error_dir, "error.txt"), "a") as f:
            f.write(error_message)
        return

    # num_data = len(dataloader.dataset)
    for seed in configs.seeds:
        seed_everything(seed=seed, deterministic=configs.deterministic)
        for batch in dataloader:
            try:
                data, atom_array, data_error_message = batch[0]
                sample_name = data["sample_name"]

                if len(data_error_message) > 0:
                    runner.print(data_error_message)
                    with open(opjoin(runner.error_dir, f"{sample_name}.txt"), "a") as f:
                        f.write(data_error_message)
                    continue

                # runner.print(
                #     (
                #         f"[Rank {DIST_WRAPPER.rank} ({data['sample_index'] + 1}/{num_data})] {sample_name}: "
                #         f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                #         f"N_atom {data['N_atom'].item()}, N_msa {data['N_msa'].item()}"
                #     )
                # )
                new_configs = update_inference_configs(configs, data["N_token"].item())
                runner.update_model_configs(new_configs)

                # Predict
                prediction = runner.predict(data)

                # Save
                runner.dumper.dump(
                    dataset_name="",
                    pdb_id=sample_name,
                    seed=seed,
                    pred_dict=prediction,
                    atom_array=atom_array,
                    entity_poly_type=data["entity_poly_type"],
                )

                runner.print(f"Results saved to {configs.dump_dir}/{sample_name}/seed_{seed}/predictions/")
                torch.cuda.empty_cache()
            except Exception as e:
                error_message = f"[Rank {DIST_WRAPPER.rank}]{data['sample_name']} {e}:\n{traceback.format_exc()}"
                runner.print(error_message)
                # Save error info
                with open(opjoin(runner.error_dir, f"{sample_name}.txt"), "a") as f:
                    f.write(error_message)
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()


def run_ptx(target_id, sequence, configs, solution, template_idx, runner):
    """
    Run the inference for a given target_id, sequence, configs, solution, and template_idx.

    Args:
        target_id (str): The target_id of the sequence.
        sequence (str): The sequence to be inferred.
        configs (ConfigDict): The configurations for the inference.
        solution (DotDict): The solution to be updated.
        template_idx (int): The template index to be used.
        runner (InferenceRunner): The runner to be used.
    """
    # Create directories
    temp_dir = f"./{configs.dump_dir}/input"  # Same as in kaggle_inference.py
    output_dir = f"./{configs.dump_dir}/output"  # Same as in kaggle_inference.py
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    process_sequence(sequence=sequence, target_id=target_id, temp_dir=temp_dir)
    configs.input_json_path = os.path.join(temp_dir, f"{target_id}_input.json")
    configs.template_idx = int(template_idx)

    # Run the inference
    infer_predict(runner, configs)

    # Copy the CIF file to the new path
    cif_file_path = (
        f"{configs.dump_dir}/{target_id}/seed_42/predictions/{target_id}_sample_0.cif"
    )
    cif_new_path = f"{configs.dump_dir}/{target_id}/seed_42/predictions/{target_id}_sample_{template_idx}_new.cif"
    shutil.copy(cif_file_path, cif_new_path)

    # Extract the C1 coordinates
    coord = extract_c1_coordinates(cif_file_path)
    if coord is None:
        coord = np.zeros((len(sequence), 3), dtype=np.float32)
    elif coord.shape[0] < (len(sequence)):
        pad_len = len(sequence) - coord.shape[0]
        pad = np.zeros((pad_len, 3), dtype=np.float32)
        coord = np.concatenate([coord, pad], axis=0)

    # Update the solution
    solution[target_id].coord.append(coord)


def run() -> None:
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s %(funcName)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.WARNING,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    # Silence dataloader logging
    logging.getLogger("rnapro.data").setLevel(logging.WARNING)

    configs_base["use_deepspeed_evo_attention"] = (
        os.environ.get("USE_DEEPSPEED_EVO_ATTENTION", False) == "true"
    )
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(
        configs=configs,
        arg_str=parse_sys_args(),
        fill_required_with_null=True,
    )

    valid_df = pd.read_csv(configs.sequences_csv)
    print(f"\n -> Loaded {len(valid_df)} sequence(s)")

    # Build model and load checkpoint once before looping over sequences

    print('\n -> Building model and loading checkpoint\n')
    runner = InferenceRunner(configs)
    print('\n -> Done, starting inference...')

    solution = make_dummy_solution(valid_df)
    for idx, row in valid_df.iterrows():
        print(f"\n -> Sequence {idx + 1}/{len(valid_df)} : {row.target_id} - {row.sequence}")

        if len(row.sequence) > configs.max_len:
            print(f'Sequence is too long ({len(row.sequence)} > {configs.max_len}), skipping')
            for template_idx in range(5):
                coord = np.zeros((len(row.sequence), 3), dtype=np.float32)
                solution[row.target_id].coord.append(coord)
            continue

        try:
            target_id = row.target_id
            sequence = row.sequence
            for template_idx in range(configs.n_templates_inf):
                print()
                run_ptx(
                    target_id=target_id,
                    sequence=sequence,
                    configs=configs,
                    solution=solution,
                    template_idx=template_idx,
                    runner=runner,
                )
        except Exception as e:
            print(f"Error processing {row.target_id}: {e}")
            continue

    print('\n\n -> Inference done ! Saving to submission.csv')
    submit_df = solution_to_submit_df(solution)
    submit_df = submit_df.fillna(0.0)
    submit_df.to_csv("./submission.csv", index=False)


if __name__ == "__main__":
    run()
