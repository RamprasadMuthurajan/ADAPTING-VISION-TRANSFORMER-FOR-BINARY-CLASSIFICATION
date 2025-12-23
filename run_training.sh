#!/bin/bash
# ========================================
# Vision Transformer Training Script
# Binary Classification: Ants vs Bees
# File: ViT-pytorch/run_training.sh
# ========================================

# ========================================
# CONFIGURATION - EDIT THESE VALUES
# ========================================

# Experiment settings
EXPERIMENT_NAME="ants_bees_vit_experiment"
DATA_ROOT="my_dataset"                    # Path to your dataset
PRETRAINED="checkpoint/ViT-B_16.npz"      # Path to pretrained weights
MODEL_TYPE="ViT-B_16"                     # Model variant

# Training hyperparameters
IMG_SIZE=224
TRAIN_BATCH_SIZE=16                       # Reduce to 8 if GPU OOM
EVAL_BATCH_SIZE=32
LEARNING_RATE=0.01
NUM_STEPS=2000
WARMUP_STEPS=200
EVAL_EVERY=50

# Output settings
OUTPUT_DIR="output"

# ========================================
# DO NOT EDIT BELOW THIS LINE
# ========================================

echo "=========================================="
echo "🚀 Starting Vision Transformer Training"
echo "=========================================="
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Dataset:    ${DATA_ROOT}"
echo "Model:      ${MODEL_TYPE}"
echo "=========================================="
echo ""

# Run training
python train_simple.py \
    --name ${EXPERIMENT_NAME} \
    --data_root ${DATA_ROOT} \
    --pretrained_dir ${PRETRAINED} \
    --model_type ${MODEL_TYPE} \
    --output_dir ${OUTPUT_DIR} \
    --img_size ${IMG_SIZE} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --num_steps ${NUM_STEPS} \
    --warmup_steps ${WARMUP_STEPS} \
    --eval_every ${EVAL_EVERY} \
    --decay_type cosine \
    --max_grad_norm 1.0 \
    --weight_decay 0 \
    --seed 42

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Training completed successfully!"
    echo "=========================================="
    echo "📁 Saved models: ${OUTPUT_DIR}/${EXPERIMENT_NAME}/"
    echo "📊 View logs:"
    echo "   tensorboard --logdir logs"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ Training failed! Check error messages above."
    echo "=========================================="
    exit 1
fi