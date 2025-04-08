# conda create -n llava python=3.10 -y
# conda activate llava
# pip install --upgrade pip  # Enable PEP 660 support.
# pip install -e ".[train]"

# pip install --upgrade torch torchvision torchaudio bitsandbytes deepspeed
# pip install --upgrade flash-attn --no-build-isolation

# mkdir -p checkpoints

LLM_VERSION="meta-llama/Meta-Llama-3-8B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

PROMPT_VERSION=plain
PRETRAIN_DATA_VERSION="blip558k"
############### Pretrain ################

BASE_RUN_NAME="llavanext-${LLM_VERSION_CLEAN}-${VISION_MODEL_VERSION_CLEAN}-pretrain_${PRETRAIN_DATA_VERSION}_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

PROMPT_VERSION="llava_llama_3"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-blip558k_pretrain_plain_la_1_6mix_ft"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

 # with necessary torchrun information for distributed training\
torchrun --nproc_per_node=8 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version $PROMPT_VERSION \
    --data_path="/home/ubuntu/dataset/LLaVA-NeXT-Data/llava_next_raw_format/llava_v1_5_mix665k.json" \
    --image_folder /home/ubuntu/dataset/LLaVA-NeXT-Data/llava_next_raw_format/ \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "./checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True

    # --image_aspect_ratio anyres \
    # --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \

    # --pretrain_mm_mlp_adapter="./checkpoints/projectors/${BASE_RUN_NAME}/mm_projector.bin" \

    # --data_path="/home/ubuntu/dataset/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json" \
    # --image_folder /home/ubuntu/dataset/LLaVA-Pretrain/ \

    # --data_path="/home/ubuntu/dataset/LLaVA-NeXT-Data/llava_next_raw_format/llava_v1_5_mix665k.json" \
    # --image_folder /home/ubuntu/dataset/LLaVA-NeXT-Data/llava_next_raw_format/ \