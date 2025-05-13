import argparse
import os
import math
import sys
import random
import numpy as np
import json
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaTokenizer,
    DataCollatorForSeq2Seq, 
    set_seed,
    TrainingArguments,
    Trainer,
    TrainerCallback,
#    Seq2SeqTrainingArguments,
#    Seq2SeqTrainer,
)
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations.deepspeed import HfDeepSpeedConfig
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
# Monkey patch the DeepSpeedCPUAdam class to avoid the ds_opt_adam error
original_init = DeepSpeedCPUAdam.__init__

def patched_init(self, *args, **kwargs):
    try:
        original_init(self, *args, **kwargs)
    except Exception as e:
        print(f"Caught exception in DeepSpeedCPUAdam.__init__: {e}")
        # Add dummy attribute to avoid AttributeError in __del__
        self.ds_opt_adam = None
        self.opt_id = None

DeepSpeedCPUAdam.__init__ = patched_init

original_del = DeepSpeedCPUAdam.__del__

def patched_del(self):
    try:
        if hasattr(self, 'ds_opt_adam'):
            self.ds_opt_adam.destroy_adam(self.opt_id)
    except Exception as e:
        print(f"Error in DeepSpeedCPUAdam.__del__: {e}")

# Apply the patch
DeepSpeedCPUAdam.__del__ = patched_del

from deepspeed.accelerator import get_accelerator
import psutil
import datetime
import gc

import torch.distributed as dist

if dist.is_available() and not dist.is_initialized():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")

gc.collect()
torch.cuda.empty_cache()

class MemTrainer(Trainer):
    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)

        # Memory logging
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        used = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[{current_time}] Step {self.state.global_step} - Used: {used:.2f}MB | Reserved: {reserved:.2f}MB | Max: {max_mem:.2f}MB")

        get_accelerator().empty_cache()
        return loss

class EmptyCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        get_accelerator().empty_cache()
        return control

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)

        
def seed_everything(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--train_data", type=str, default="")
    
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)

    
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    parser.add_argument('--gradient_checkpointing', action='store_true', default=True)
    

    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--warmup_ratio", type=int, default=0.05)

    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed_file", type=str, default=None)
    
    
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    #deepspeed.init_distributed()

    assert not (args.bf16 and args.fp16)
        
    args.global_rank = torch.distributed.get_rank()
    #args.global_rank = dist.get_rank() if dist.is_initialized() else 0

    
       
    print_rank_0(f'*****{torch.cuda.device_count()}*****')
    print_rank_0(f'*****{torch.distributed.get_world_size()}*****')

    seed_everything(args.seed)


    print_rank_0("model_name_or_path : " + args.model_name_or_path, args.global_rank)
    
    model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32), trust_remote_code=True)
    model.config.pretraining_tp = 1
    #model = model.cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    #model_engine, optimizer, _, _ = deepspeed.initialize(model=model, config=args.deepspeed_file)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=args.deepspeed_file,
    )
    with open("model.txt", "w") as f:
        f.write(str(model))
    f.close()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.model_max_length = 256
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Prepare the data

    try:
        train_dataset = DatasetDict.load_from_disk(args.train_data)['train']
    except:
        train_dataset = Dataset.load_from_disk(args.train_data)

    #making eval dataset 10% of train dataset to ensure no OOM errors.
    split_dataset = train_dataset.train_test_split(test_size=0.1, seed=args.seed)
    eval_dataset = split_dataset['test']
        
    print_rank_0("***** Data load success! *****", args.global_rank)
        
    # Show the training loss with every epoch

    training_args = TrainingArguments(#Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        save_strategy = "no",
        # do_eval=True,
        max_grad_norm=1.0,
        learning_rate=args.learning_rate,
        gradient_checkpointing=args.gradient_checkpointing,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        dataloader_num_workers=1,
        logging_steps=1,
        report_to='tensorboard',
        deepspeed=args.deepspeed_file,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
    )
    
    trainer = MemTrainer(#Seq2SeqTrainer(
        model_engine.module,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=4),
        tokenizer=tokenizer,
        callbacks=[EmptyCacheCallback()],
    )
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(os.path.join(args.output_dir, "model_final"))

    trainer.evaluate()


    

    
if __name__ == "__main__":
    main()
    