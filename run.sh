#!/bin/bash

#--dataset_name="ccdv/govreport-summarization"
# --dataset_name="cnn_dailymail" --dataset_config "3.0.0" 
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7,8

# export CUDA_LAUNCH_BLOCKING=1 WANDB_CONSOLE=off TORCH_DISTRIBUTED_DEBUG=INFO
# accelerate launch --config_file='./accelerate.yaml' run_summarization_no_trainer_test.py  --seed=42 --preprocessing_num_workers=1 --weight_decay='0.001' --output_dir="arxiv-summarization_exp/t5mem/base" --per_device_train_batch_size=1 --per_device_eval_batch_size=1    --num_train_epochs=10  --model_name_or_path='t5mem-base' --num_beams=5 --with_tracking --report_to='wandb' --checkpointing_steps='epoch'  --dataset_name="ccdv/arxiv-summarization"   --gradient_accumulation_steps=1 --tokenizer_name='t5-base'


# govreport
# export CUDA_VISIBLE_DEVICES=1
# export CUDA_LAUNCH_BLOCKING=1 WANDB_CONSOLE=off TORCH_DISTRIBUTED_DEBUG=INFO
# accelerate launch --config_file='./accelerate.yaml' run_summarization_no_trainer_test.py  --seed=42 --preprocessing_num_workers=1 --weight_decay='0.001' --output_dir="govreport-summarization_exp/t5mem/memorytest" --per_device_train_batch_size=1 --per_device_eval_batch_size=1    --num_train_epochs=10  --model_name_or_path='t5mem-base' --num_beams=5 --with_tracking --report_to='wandb' --checkpointing_steps='epoch'  --dataset_name="ccdv/govreport-summarization"  --gradient_accumulation_steps=1 --tokenizer_name='t5-base'



# export CUDA_LAUNCH_BLOCKING=1 WANDB_CONSOLE=off TORCH_DISTRIBUTED_DEBUG=INFO
# accelerate launch --config_file='./accelerate.yaml' run_summarization_no_trainer_test.py  --seed=42 --preprocessing_num_workers=1 --weight_decay='0.001' --output_dir="govreport-summarization_exp/t5/base" --per_device_train_batch_size=1 --per_device_eval_batch_size=1    --num_train_epochs=10  --model_name_or_path='t5-base' --num_beams=5 --with_tracking --report_to='wandb' --checkpointing_steps='epoch'  --dataset_name="ccdv/govreport-summarization"  --gradient_accumulation_steps=1 --tokenizer_name='t5-base' --source_prefix='summarize: '


#CNN_DailyMail
# export CUDA_LAUNCH_BLOCKING=1 WANDB_CONSOLE=off TORCH_DISTRIBUTED_DEBUG=INFO
# accelerate launch --config_file='./accelerate.yaml' run_summarization_no_trainer_test.py  --seed=42 --preprocessing_num_workers=1 --weight_decay='0.001' --output_dir="cnn_dailymail_exp/t5/base" --per_device_train_batch_size=1 --per_device_eval_batch_size=1    --num_train_epochs=10  --model_name_or_path='t5-base' --num_beams=5 --with_tracking --report_to='wandb' --checkpointing_steps='epoch' --dataset_name="cnn_dailymail" --dataset_config "3.0.0"   --gradient_accumulation_steps=1 --tokenizer_name='t5-base' --source_prefix='summarize: '


# export CUDA_LAUNCH_BLOCKING=1 WANDB_CONSOLE=off TORCH_DISTRIBUTED_DEBUG=INFO
# accelerate launch --config_file='./accelerate.yaml' run_summarization_no_trainer_test.py  --seed=42 --preprocessing_num_workers=1 --weight_decay='0.001' --output_dir="cnn_dailymail_exp/t5mem/base_384_8mem" --per_device_train_batch_size=1 --per_device_eval_batch_size=1    --num_train_epochs=10  --model_name_or_path='t5mem-base' --num_beams=5 --with_tracking --report_to='wandb' --checkpointing_steps='epoch' --dataset_name="cnn_dailymail" --dataset_config "3.0.0"   --gradient_accumulation_steps=1 --tokenizer_name='t5-base'
  
  


export CUDA_LAUNCH_BLOCKING=1 WANDB_CONSOLE=off TORCH_DISTRIBUTED_DEBUG=INFO  GRADIENT_ACCUMULATION_STEPS=1 

accelerate launch --cpu run_summarization.py --seed=42 --weight_decay='0.001' --output_dir="cnn_dailymail_exp/t5mem/base_384_8mem_7ep_3072_128" --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --num_train_epochs=7  --model_name_or_path='configs/t5mem-base.json' --num_beams=5 --with_tracking --report_to='wandb' --checkpointing_steps='epoch' --dataset_name="cnn_dailymail" --dataset_config="3.0.0" --gradient_accumulation_steps=1 --tokenizer_name='t5-base'


# export CUDA_LAUNCH_BLOCKING=1 WANDB_CONSOLE=off TORCH_DISTRIBUTED_DEBUG=INFO  
# accelerate launch --config_file='./accelerate.yaml' run_summarization_no_trainer_sled.py  --seed=42 --preprocessing_num_workers=1 --weight_decay='0.001' --output_dir="cnn_dailymail_exp/SLED_256-10ep_1024_input_128TrainEvalOutput" --per_device_train_batch_size=1 --per_device_eval_batch_size=1    --num_train_epochs=10  --model_name_or_path='tau/t5-v1_1-base-sled' --num_beams=5 --with_tracking --report_to='wandb' --checkpointing_steps='epoch' --dataset_name="cnn_dailymail" --dataset_config "3.0.0"   --use_slow_tokenizer
  
  
  

  
# # # ###  SAmSum

# export CUDA_LAUNCH_BLOCKING=1 WANDB_CONSOLE=off TORCH_DISTRIBUTED_DEBUG=INFO
# accelerate launch --config_file='./accelerate.yaml' run_summarization_no_trainer_test.py  --seed=42 --preprocessing_num_workers=1 --weight_decay='0.001' --output_dir="samsum_exp/paper/t5mem/base_384_5ep_2acc_4g_1w" --per_device_train_batch_size=1 --per_device_eval_batch_size=1    --num_train_epochs=5  --model_name_or_path='t5mem-base' --num_beams=5 --with_tracking --report_to='wandb' --checkpointing_steps='epoch'  --dataset_name="samsum"  --gradient_accumulation_steps=2 --tokenizer_name='t5-base'



# export CUDA_LAUNCH_BLOCKING=1 WANDB_CONSOLE=off TORCH_DISTRIBUTED_DEBUG=INFO
# accelerate launch --config_file='./accelerate.yaml' run_summarization_no_trainer_test.py  --seed=42 --preprocessing_num_workers=1 --weight_decay='0.001' --output_dir="samsum_exp/paper/t5/base-5ep" --per_device_train_batch_size=1 --per_device_eval_batch_size=1    --num_train_epochs=5  --model_name_or_path='t5-base' --num_beams=5 --with_tracking --report_to='wandb' --checkpointing_steps='epoch'  --dataset_name="samsum"  --gradient_accumulation_steps=1 --tokenizer_name='t5-base'



# export CUDA_LAUNCH_BLOCKING=1 WANDB_CONSOLE=off TORCH_DISTRIBUTED_DEBUG=INFO
# accelerate launch --config_file='./accelerate.yaml' run_summarization_no_trainer_sled.py  --seed=42 --preprocessing_num_workers=1 --weight_decay='0.001' --output_dir="samsum_exp/paper/base/SLED_256-5ep_2" --per_device_train_batch_size=1 --per_device_eval_batch_size=1    --num_train_epochs=5  --model_name_or_path='tau/t5-v1_1-base-sled' --num_beams=5 --with_tracking --report_to='wandb' --checkpointing_steps='epoch'  --dataset_name="samsum"  --gradient_accumulation_steps=1 --tokenizer_name='t5-base'


# export CUDA_LAUNCH_BLOCKING=1 WANDB_CONSOLE=off TORCH_DISTRIBUTED_DEBUG=INFO
# accelerate launch --config_file='./accelerate.yaml' run_summarization_no_trainer_test.py  --seed=42 --preprocessing_num_workers=1 --weight_decay='0.001' --output_dir="samsum_exp/paper/base/longt5-5ep" --per_device_train_batch_size=1 --per_device_eval_batch_size=1    --num_train_epochs=5  --model_name_or_path='google/long-t5-tglobal-base' --num_beams=5 --with_tracking --report_to='wandb' --checkpointing_steps='epoch'  --dataset_name="samsum"  --gradient_accumulation_steps=1 

# export CUDA_LAUNCH_BLOCKING=1 WANDB_CONSOLE=off TORCH_DISTRIBUTED_DEBUG=INFO
# accelerate launch --config_file='./accelerate.yaml' run_summarization_no_trainer_test.py  --seed=42 --preprocessing_num_workers=1 --weight_decay='0.001' --output_dir="samsum_exp/paper/t5mem/base_mem_384_7ep_2" --per_device_train_batch_size=1 --per_device_eval_batch_size=1    --num_train_epochs=7  --model_name_or_path='t5mem-base' --num_beams=5 --with_tracking --report_to='wandb' --checkpointing_steps='epoch'  --dataset_name="samsum"  --gradient_accumulation_steps=1 --tokenizer_name='t5-base'