# !/usr/bin/env python
# coding: utf-8


import os
import re
import pandas as pd
from pathlib import Path
import shutil
import torch
import torch.nn as nn
from typing import Dict


def parse_tmscore_output(output: str) -> float:
    matches = re.findall(r'TM-score=\s+([\d.]+)', output)
    if len(matches) < 2:
        raise ValueError('No TM score found in USalign output')
    return float(matches[1])


# ---------------------
# PDB writers
# ---------------------

def sanitize(xyz):
    MIN_COORD = -999.999
    MAX_COORD = 9999.999
    return min(max(xyz, MIN_COORD), MAX_COORD)


def write_target_line(atom_name, atom_serial, residue_name, chain_id, residue_num,
                      x_coord, y_coord, z_coord, occupancy=1.0, b_factor=0.0, atom_type='P') -> str:
    return (
        f'ATOM  {atom_serial:>5d}  {atom_name:4s}{residue_name:>3s} {chain_id:1s}{residue_num:>4d}    '
        f'{sanitize(x_coord):>8.3f}{sanitize(y_coord):>8.3f}{sanitize(z_coord):>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}           '
        f'{atom_type}\n'
    )


def write2pdb(df: pd.DataFrame, xyz_id: int, target_path: str) -> int:
    """
    Write single-chain PDB (chain 'A') using row['resid'] as residue_num.
    Raises exceptions on invalid data.
    """
    resolved_cnt = 0
    with open(target_path, 'w') as fh:
        for _, row in df.iterrows():
            x = row[f'x_{xyz_id}']
            y = row[f'y_{xyz_id}']
            z = row[f'z_{xyz_id}']
            if x > -1e6 and y > -1e6 and z > -1e6:
                resolved_cnt += 1
                resid_num = int(row['resid'])
                fh.write(
                    write_target_line(
                        "C1'",
                        resid_num,
                        row['resname'],
                        'A',
                        resid_num,
                        x,
                        y,
                        z,
                        atom_type='C'
                    )
                )
    return resolved_cnt


def write2pdb_singlechain_native(df_native: pd.DataFrame, xyz_id: int, target_path: str) -> int:
    """
    Write native single-chain using row['resid'] as residue numbers.
    Assumes all required columns exist and are valid.
    """
    df_sorted = df_native.copy()
    df_sorted['__resid_int'] = df_sorted['resid'].astype(int)
    df_sorted = df_sorted.sort_values('__resid_int').reset_index(drop=True)

    resolved_cnt = 0
    with open(target_path, 'w') as fh:
        for _, row in df_sorted.iterrows():
            x = row[f'x_{xyz_id}']
            y = row[f'y_{xyz_id}']
            z = row[f'z_{xyz_id}']
            if x > -1e6 and y > -1e6 and z > -1e6:
                resolved_cnt += 1
                resid_num = int(row['resid'])
                fh.write(
                    write_target_line(
                        "C1'",
                        resid_num,
                        row['resname'],
                        'A',
                        resid_num,
                        x,
                        y,
                        z,
                        atom_type='C'
                    )
                )
    return resolved_cnt


def write2pdb_multichain_from_solution(df_solution: pd.DataFrame, xyz_id: int, target_path: str) -> int:
    """
    Write multi-chain PDB for native solution using columns 'chain' and 'copy' to assign chain letters.
    Expects 'resid' convertible to int and chain/copy present. No fallbacks.
    """
    df_sorted = df_solution.copy()
    df_sorted['__resid_int'] = df_sorted['resid'].astype(int)
    df_sorted = df_sorted.sort_values('__resid_int')

    chain_map = {}
    next_ord = ord('A')
    written = 0
    with open(target_path, 'w') as fh:
        for _, row in df_sorted.iterrows():
            x = row[f'x_{xyz_id}']
            y = row[f'y_{xyz_id}']
            z = row[f'z_{xyz_id}']
            if not (x > -1e6 and y > -1e6 and z > -1e6):
                continue
            chain_val = row['chain']
            copy_key = int(row['copy'])
            g = (str(chain_val), copy_key)
            if g not in chain_map:
                if next_ord <= ord('Z'):
                    ch = chr(next_ord)
                else:
                    ov = next_ord - ord('Z') - 1
                    if ov < 26:
                        ch = chr(ord('a') + ov)
                    else:
                        ch = chr(ord('0') + (ov - 26) % 10)
                chain_map[g] = ch
                next_ord += 1
            chain_id = chain_map[g]
            written += 1
            resid_num = int(row['resid'])
            fh.write(write_target_line("C1'", resid_num, row['resname'], chain_id, resid_num, x, y, z, atom_type='C'))
    return written


def write2pdb_multichain_from_groups(
        df_pred: pd.DataFrame,
        xyz_id: int,
        target_path: str,
        groups_list
) -> tuple[int, list]:
    """
    Write predicted multichain PDB based on a positional groups_list (tuple per residue: (chain, copy)).
    Requires groups_list length == number of residues in df_pred (after sorting).
    Returns (written_count, chain_letters_per_res).
    """
    df_sorted = df_pred.copy()
    df_sorted['__resid_int'] = df_sorted['resid'].astype(int)
    df_sorted = df_sorted.sort_values('__resid_int').reset_index(drop=True)

    if groups_list is None or len(groups_list) != len(df_sorted):
        raise ValueError("groups_list must be provided and match number of residues in predicted df")

    chain_map = {}
    next_ord = ord('A')
    chain_letters = []
    written = 0
    with open(target_path, 'w') as fh:
        for idx, row in df_sorted.iterrows():
            g = groups_list[idx]
            if isinstance(g, tuple):
                gkey = (str(g[0]), int(g[1]))
            else:
                gkey = (str(g), None)
            if gkey not in chain_map:
                if next_ord <= ord('Z'):
                    ch = chr(next_ord)
                else:
                    ov = next_ord - ord('Z') - 1
                    if ov < 26:
                        ch = chr(ord('a') + ov)
                    else:
                        ch = chr(ord('0') + (ov - 26) % 10)
                chain_map[gkey] = ch
                next_ord += 1
            chain_id = chain_map[gkey]
            chain_letters.append(chain_id)
            x = row[f'x_{xyz_id}']
            y = row[f'y_{xyz_id}']
            z = row[f'z_{xyz_id}']
            if x > -1e6 and y > -1e6 and z > -1e6:
                written += 1
                resid_num = int(row['resid'])
                fh.write(
                    write_target_line(
                        "C1'",
                        resid_num,
                        row['resname'],
                        chain_id,
                        resid_num,
                        x,
                        y,
                        z,
                        atom_type='C'
                    )
                )
    return written, chain_letters


def write2pdb_singlechain_permuted_pred(df_pred: pd.DataFrame, xyz_id: int, permuted_indices: list,
                                        target_path: str) -> int:
    """
    Create single-chain PDB by concatenating predicted residues in permuted_indices order.
    Output residue numbers are sequential starting at 1 and increase for every permuted position.
    Raises exception if indices out of range.
    """
    df_sorted = df_pred.copy()
    df_sorted['__resid_int'] = df_sorted['resid'].astype(int)
    df_sorted = df_sorted.sort_values('__resid_int').reset_index(drop=True)

    written = 0
    next_res = 1
    with open(target_path, 'w') as fh:
        for idx in permuted_indices:
            if idx < 0 or idx >= len(df_sorted):
                # strict behavior: raise error for invalid index
                raise IndexError(f"permuted index {idx} out of range for predicted residues")
            row = df_sorted.iloc[idx]
            x = row[f'x_{xyz_id}']
            y = row[f'y_{xyz_id}']
            z = row[f'z_{xyz_id}']
            out_resnum = next_res
            if x > -1e6 and y > -1e6 and z > -1e6:
                written += 1
                fh.write(write_target_line("C1'", out_resnum, row['resname'], 'A', out_resnum, x, y, z, atom_type='C'))
            next_res += 1
    return written


# ---------------------
# USalign wrappers
# ---------------------
def run_usalign_raw(predicted_pdb: str, native_pdb: str, usalign_bin='USalign', align_sequence=False,
                    tmscore=None) -> str:
    cmd = f'{usalign_bin} {predicted_pdb} {native_pdb} -atom " C1\'"'
    if tmscore is not None:
        cmd += f' -TMscore {tmscore}'
        if int(tmscore) == 0:
            cmd += ' -mm 1 -ter 0'
    elif not align_sequence:
        cmd += ' -TMscore 1'
    return os.popen(cmd).read()


def parse_usalign_chain_orders(output: str):
    """
    Parse USalign output for both Structure_1 and Structure_2 chain lists.
    Returns (chain_list_structure1, chain_list_structure2).
    Raises if parsing fails to find either line.
    """
    chain1 = None
    chain2 = None
    for line in output.splitlines():
        line = line.strip()
        if line.startswith('Name of Structure_1:'):
            parts = line.split(':')
            clist = []
            for part in parts[2:]:
                token = part.strip()
                if token == '':
                    continue
                token0 = token.split()[0]
                last = token0.split(',')[-1]
                ch = re.sub(r'[^A-Za-z0-9]', '', last)
                if ch:
                    clist.append(ch)
            chain1 = clist
        elif line.startswith('Name of Structure_2:'):
            parts = line.split(':')
            clist = []
            for part in parts[2:]:
                token = part.strip()
                if token == '':
                    continue
                token0 = token.split()[0]
                last = token0.split(',')[-1]
                ch = re.sub(r'[^A-Za-z0-9]', '', last)
                if ch:
                    clist.append(ch)
            chain2 = clist
    if chain1 is None or chain2 is None:
        raise ValueError("Failed to parse chain orders from USalign output")
    return chain1, chain2


# ---------------------
# Main scoring function (no try/except, no fallbacks)
# ---------------------
def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    usalign_bin_hint: str = None
) -> float:
    """
    Enhanced scoring with chain-permutation handling for multicopy targets.
    This version contains no try/except blocks and will raise on any error.
    """
    # determine usalign binary
    if usalign_bin_hint:
        usalign_bin = usalign_bin_hint
    else:
        # Use the USalign binary from the repository
        repo_usalign = Path(__file__).parent.resolve() / 'USalign'
        if os.path.exists(repo_usalign):
            usalign_bin = str(repo_usalign)
            if not os.access(usalign_bin, os.X_OK):
                os.chmod(usalign_bin, 0o755)
        # Fallback to Kaggle paths if repo binary not found
        elif os.path.exists('/kaggle/input/usalign/USalign') and not os.path.exists('/kaggle/working/USalign'):
            shutil.copy2('/kaggle/input/usalign/USalign', '/kaggle/working/USalign')
            os.chmod('/kaggle/working/USalign', 0o755)
            usalign_bin = '/kaggle/working/USalign'
        elif os.path.exists('/kaggle/working/USalign'):
            usalign_bin = '/kaggle/working/USalign'
        else:
            usalign_bin = 'USalign'  # Fall back to system PATH

    sol = solution.copy()
    sub = submission.copy()
    sol['target_id'] = sol['ID'].apply(lambda x: '_'.join(str(x).split('_')[:-1]))
    sub['target_id'] = sub['ID'].apply(lambda x: '_'.join(str(x).split('_')[:-1]))

    results = []

    for target_id, group_native in sol.groupby('target_id'):
        group_predicted = sub[sub['target_id'] == target_id]
        has_chain_copy = ('chain' in group_native.columns) and ('copy' in group_native.columns)
        is_multicopy = has_chain_copy and (group_native['copy'].astype(float).max() > 1)

        # precompute native models that have coords
        native_with_coords = []
        for native_cnt in range(1, 41):
            native_pdb = f'native_{target_id}_{native_cnt}.pdb'
            resolved_native = write2pdb(group_native, native_cnt, native_pdb)
            if resolved_native > 0:
                native_with_coords.append(native_cnt)
            else:
                if os.path.exists(native_pdb):
                    os.remove(native_pdb)

        if not native_with_coords:
            raise ValueError(f"No native models with coordinates for target {target_id}")

        best_per_pred = []
        for pred_cnt in range(1, 6):
            if not is_multicopy:
                predicted_pdb = f'predicted_{target_id}_{pred_cnt}.pdb'
                resolved_pred = write2pdb(group_predicted, pred_cnt, predicted_pdb)
                if resolved_pred <= 2:
                    # print(f"Predicted model {pred_cnt} for target {target_id} has insufficient coordinates")
                    best_per_pred.append(0.0)
                    continue

                scores = []
                for native_cnt in native_with_coords:
                    native_pdb = f'native_{target_id}_{native_cnt}.pdb'
                    out = run_usalign_raw(predicted_pdb, native_pdb, usalign_bin=usalign_bin, align_sequence=False,
                                          tmscore=1)
                    s = parse_tmscore_output(out)
                    scores.append(s)
                best_per_pred.append(max(scores))

            else:
                # multicopy
                # strict: require chain and copy columns convertible
                gn_sorted = group_native.copy()
                gn_sorted['__resid_int'] = gn_sorted['resid'].astype(int)
                gn_sorted = gn_sorted.sort_values('__resid_int').reset_index(drop=True)
                groups_list = []
                for _, r in gn_sorted.iterrows():
                    chain_val = r['chain']
                    copy_i = int(r['copy'])
                    groups_list.append((chain_val, copy_i))

                # predicted multichain - groups_list must match predicted residue count or error
                dfp_sorted = group_predicted.copy()
                dfp_sorted['__resid_int'] = dfp_sorted['resid'].astype(int)
                dfp_sorted = dfp_sorted.sort_values('__resid_int').reset_index(drop=True)
                if len(groups_list) != len(dfp_sorted):
                    raise ValueError(
                        f"groups_list length ({len(groups_list)}) does not match predicted "
                        f"residue count ({len(dfp_sorted)}) for target {target_id}"
                    )

                predicted_multi_pdb = f'pred_multi_{target_id}_{pred_cnt}.pdb'
                resolved_pred_multi, pred_chain_letters = write2pdb_multichain_from_groups(
                    group_predicted, pred_cnt, predicted_multi_pdb, groups_list
                )
                if resolved_pred_multi == 0:
                    # print(f"Predicted multi model {pred_cnt} for target {target_id} has no coordinates")
                    best_per_pred.append(0.0)
                    continue

                scores = []
                for native_cnt in native_with_coords:
                    native_multi_pdb = f'native_multi_{target_id}_{native_cnt}.pdb'
                    resolved_native_multi = write2pdb_multichain_from_solution(
                        group_native, native_cnt, native_multi_pdb
                    )
                    if resolved_native_multi == 0:
                        continue

                    raw_out = run_usalign_raw(predicted_multi_pdb, native_multi_pdb, usalign_bin=usalign_bin,
                                              align_sequence=True, tmscore=0)
                    chain1, chain2 = parse_usalign_chain_orders(raw_out)  # will raise if parsing fails

                    # build native->pred mapping chain2[i] -> chain1[i]
                    native_to_pred = {n_ch: p_ch for n_ch, p_ch in zip(chain2, chain1)}

                    # canonical native order = chain2 unique in order seen
                    # native_chain_order = []
                    # for ch in chain2:
                    #    if ch not in native_chain_order:
                    #        native_chain_order.append(ch)
                    native_chain_order = list(native_to_pred.keys())
                    native_chain_order.sort()  # this is critical...

                    # predicted chain order by following native chain A,B,...
                    pred_chain_order = [native_to_pred[n_ch] for n_ch in native_chain_order if
                                        native_to_pred.get(n_ch) is not None]

                    # construct pred_positions_by_chain
                    pred_positions_by_chain = {}
                    for idx, ch in enumerate(pred_chain_letters):
                        if ch is None:
                            continue
                        pred_positions_by_chain.setdefault(ch, []).append(idx)

                    # require that each chain in pred_chain_order exists in pred_positions_by_chain
                    pred_chain_order = [p for p in pred_chain_order if p in pred_positions_by_chain]

                    # form permuted indices by concatenation
                    permuted_indices = []
                    for ch in pred_chain_order:
                        permuted_indices.extend(pred_positions_by_chain[ch])
                    # append any remaining
                    for idx in range(len(pred_chain_letters)):
                        if idx not in permuted_indices:
                            permuted_indices.append(idx)

                    # write permuted single-chain predicted and native single-chain
                    pred_single_perm = f'pred_permuted_{target_id}_{pred_cnt}_{native_cnt}.pdb'
                    written_pred_single = write2pdb_singlechain_permuted_pred(group_predicted, pred_cnt,
                                                                              permuted_indices, pred_single_perm)
                    native_single = f'native_single_{target_id}_{native_cnt}.pdb'
                    written_native = write2pdb_singlechain_native(group_native, native_cnt, native_single)

                    if written_pred_single <= 2 or written_native <= 2:
                        raise ValueError(
                            "Insufficient residues after permutation for "
                            f"target {target_id}, pred {pred_cnt}, native {native_cnt}"
                        )

                    out = run_usalign_raw(
                        pred_single_perm,
                        native_single,
                        usalign_bin=usalign_bin,
                        align_sequence=False,
                        tmscore=1,
                    )
                    score_final = parse_tmscore_output(out)
                    scores.append(score_final)

                best_per_pred.append(max(scores))

        results.append(max(best_per_pred))

    if not results:
        pass
        # raise ValueError("No targets scored")
    return float(sum(results) / len(results)) if len(results) > 0 else 0.0


# ---------------------
# TM-Score Metrics for training/evaluation pipeline
# ---------------------
class TMScoreMetrics(nn.Module):
    """TM-score metrics for evaluation pipeline"""

    def __init__(self, configs):
        super(TMScoreMetrics, self).__init__()
        self.configs = configs
        # Get the USalign binary path from the repository
        metrics_dir = Path(__file__).parent.resolve()
        self.usalign_bin = str(metrics_dir / 'USalign')

        # Verify the binary exists
        if not os.path.exists(self.usalign_bin):
            raise FileNotFoundError(f"USalign binary not found at {self.usalign_bin}")

        # Make sure it's executable
        if not os.access(self.usalign_bin, os.X_OK):
            os.chmod(self.usalign_bin, 0o755)

    def compute_tm_score(self, pred_dict: Dict, label_dict: Dict) -> Dict[str, torch.Tensor]:
        """Compute TM-score between predicted and ground truth structures.

        Args:
            pred_dict: Dictionary containing:
                - coordinate: [N_sample, N_atom, 3] predicted coordinates
                - additional metadata
            label_dict: Dictionary containing:
                - coordinate: [N_sample, N_atom, 3] ground truth coordinates
                - additional metadata

        Returns:
            Dictionary with 'tm_scores': [N_sample] tensor of TM-scores
        """
        # Convert tensors to numpy and prepare DataFrames for score function
        pred_coords = pred_dict["coordinate"].detach().cpu().numpy()  # [N_sample, N_atom, 3]
        label_coords = label_dict["coordinate"].detach().cpu().numpy()  # [N_sample, N_atom, 3]

        N_sample = pred_coords.shape[0]
        N_atom = pred_coords.shape[1]

        # Create solution DataFrame (ground truth)
        # Each row represents one atom/residue with coordinates for all samples
        solution_rows = []
        for atom_idx in range(N_atom):
            row = {
                'ID': f'target_{atom_idx}',
                'resid': atom_idx,
                'resname': 'A',  # Default residue name
            }
            for sample_idx in range(N_sample):
                row[f'x_{sample_idx + 1}'] = label_coords[sample_idx, atom_idx, 0]
                row[f'y_{sample_idx + 1}'] = label_coords[sample_idx, atom_idx, 1]
                row[f'z_{sample_idx + 1}'] = label_coords[sample_idx, atom_idx, 2]
            solution_rows.append(row)

        # Create submission DataFrame (predictions)
        # Each row represents one atom/residue with coordinates for all samples
        submission_rows = []
        for atom_idx in range(N_atom):
            row = {
                'ID': f'target_{atom_idx}',
                'resid': atom_idx,
                'resname': 'A',
            }
            for sample_idx in range(N_sample):
                row[f'x_{sample_idx + 1}'] = pred_coords[sample_idx, atom_idx, 0]
                row[f'y_{sample_idx + 1}'] = pred_coords[sample_idx, atom_idx, 1]
                row[f'z_{sample_idx + 1}'] = pred_coords[sample_idx, atom_idx, 2]
            submission_rows.append(row)

        solution_df = pd.DataFrame(solution_rows)
        submission_df = pd.DataFrame(submission_rows)

        # Compute TM-score using the score function
        try:
            tm_score_value = score(solution_df, submission_df, usalign_bin_hint=self.usalign_bin)
            # Return as tensor for each sample (all samples get the same aggregated score for now)
            tm_scores = torch.tensor([tm_score_value] * N_sample, dtype=torch.float32)
        except Exception as e:
            print(f"Warning: TM-score computation failed: {e}")
            tm_scores = torch.zeros(N_sample, dtype=torch.float32)

        return {"tm_scores": tm_scores}

    def aggregate_tm_score(self, tm_score_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Aggregate TM-scores across samples.

        Args:
            tm_score_dict: Dictionary with 'tm_scores': [N_sample] tensor

        Returns:
            Dictionary with aggregated metrics
        """
        tm_scores = tm_score_dict["tm_scores"]

        aggregated = {
            "tm_score/mean": tm_scores.mean().item(),
            "tm_score/max": tm_scores.max().item(),
            "tm_score/min": tm_scores.min().item(),
            "tm_score/median": tm_scores.median().item(),
        }

        return aggregated
