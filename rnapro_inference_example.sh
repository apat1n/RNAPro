
export LAYERNORM_TYPE=torch # fast_layernorm, torch

# Inference parameters (RNAPro)
SEED=42
N_SAMPLE=1
N_STEP=200
N_CYCLE=10

# Paths
DUMP_DIR="./output"

# Set a valid checkpoint file path below
CHECKPOINT_PATH="./rnapro_base.pt"

# Template/MSA settings
TEMPLATE_DATA="./examples/test_templates.pt"
# Note: template_idx supports 5 choices and maps to top-k:
# 0->top1, 1->top2, 2->top3, 3->top4, 4->top5
TEMPLATE_IDX=0

# MSA directory
RNA_MSA_DIR="./msa"

# Sequences to process
SEQUENCES_CSV="./examples/test_sequences.csv"

# RibonanzaNet2 path (keep as-is per request)
RIBONANZA_PATH="./release_data/ribonanzanet2_checkpoint"

# Model selection: keep to an existing key to align defaults (N_step=200, N_cycle=10)
MODEL_NAME="rnapro_base"
mkdir -p "${DUMP_DIR}"

python3 runner/inference.py \
    --model_name "${MODEL_NAME}" \
    --seeds ${SEED} \
    --dump_dir "${DUMP_DIR}" \
    --load_checkpoint_path "${CHECKPOINT_PATH}" \
    --use_msa true \
    --use_template "ca_precomputed" \
    --model.use_template "ca_precomputed" \
    --model.use_RibonanzaNet2 true \
    --model.template_embedder.n_blocks 2 \
    --model.ribonanza_net_path "${RIBONANZA_PATH}" \
    --template_data "${TEMPLATE_DATA}" \
    --template_idx ${TEMPLATE_IDX} \
    --rna_msa_dir "${RNA_MSA_DIR}" \
    --model.N_cycle ${N_CYCLE} \
    --sample_diffusion.N_sample ${N_SAMPLE} \
    --sample_diffusion.N_step ${N_STEP} \
    --load_strict true \
    --num_workers 0 \
    --triangle_attention "cuequivariance" \
    --triangle_multiplicative "cuequivariance" \
    --sequences_csv "${SEQUENCES_CSV}" \
    --max_len 5000 \
    --logger "logging" \
    --n_templates_inf 5

# Notes:
# --triangle_attention supports 'triattention', 'cuequivariance', 'deepspeed', 'torch'
# --triangle_multiplicative supports 'cuequivariance', 'torch'
# --max_len 1000: Sequences longer than max_len will be skipped to avoid oom
# --logger handles logging of the inference runner, supports "logging", "print"
# --n_templates_inf sets the number of inferences to do with different template combinations