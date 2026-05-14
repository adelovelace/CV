#!/bin/bash

# ==========================================
# Experiment Hyperparameters
# ==========================================
EPOCHS=150
BATCH_SIZE=32
LR=0.0005

# Arrays to loop through
DATASETS=("fer2013" "ckplus")
PRETRAINED_MODELS=("mobilenet" "inception")
FROZEN_LAYERS=("30" "60" "80")

echo "Starting Automated Training Pipeline..."
echo "Target Epochs: $EPOCHS | Batch Size: $BATCH_SIZE | Learning Rate: $LR"
sleep 2

# ==========================================
# PHASE 1: Fine-Tuning Pre-trained Models
# ==========================================
for dataset in "${DATASETS[@]}"; do
    for model in "${PRETRAINED_MODELS[@]}"; do
        for frozen in "${FROZEN_LAYERS[@]}"; do
            echo ""
            echo "======================================================================"
            echo "[PHASE 1] Fine-Tuning: Model=$model | Dataset=$dataset | Frozen Layers=$frozen"
            echo "======================================================================"
            
            # Execute the training script
            python train.py \
                --model $model \
                --dataset $dataset \
                --frozen $frozen \
                --epoch $EPOCHS \
                --batchsize $BATCH_SIZE \
                --lr $LR
                
            echo "[!] Finished: $model on $dataset with $frozen frozen layers."
        done
    done
done

# ==========================================
# PHASE 2: Training Custom Model from Scratch
# ==========================================
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "[PHASE 2] Training from Zero: Model=custom | Dataset=$dataset"
    echo "======================================================================"
    
    # Custom model ignores the frozen argument, and train.py automatically sets trainsize to 48
    python train.py \
        --model custom \
        --dataset $dataset \
        --epoch $EPOCHS \
        --batchsize $BATCH_SIZE \
        --lr $LR
        
    echo "[!] Finished: Custom model on $dataset."
done

echo ""
echo "======================================================================"
echo "[!] ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "======================================================================"