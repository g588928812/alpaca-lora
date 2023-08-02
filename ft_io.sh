python finetune-input-output.py \
    --base_model '/home/g/models/open_llama_7b' \
    --data_path 'data/oa_guanaco_io.jsonl' \
    --wandb_project 'alpaca-lora'

# python finetune.py \
#     --base_model '/home/g/models/open_llama_7b' \
#     --data_path 'data/oa_guanaco_io.jsonl' \
#     --prompt_template_name 'guanaco' \
#     --batch_size 16 \
#     --micro_batch_size 1 \
#     --val_set_size 0.1 

