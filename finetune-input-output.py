import os
import sys
from typing import List

import fire
import wandb
import torch
import transformers
from transformers import TrainerCallback
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./testruns",
    # training hyperparams
    test_run: bool = False,
    batch_size: int = 32,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 0.0002,
    cutoff_len: int = 2000,
    # val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "lora-alpaca",
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "simple-input-output",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            # f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            # f"wandb_run_name: {wandb_run_name}\n"
            # f"wandb_watch: {wandb_watch}\n"
            # f"wandb_log_model: {wandb_log_model}\n"
            # f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    # Only overwrite environ if wandb param passed
    if not test_run:
        os.environ["WANDB_PROJECT"] = wandb_project
    # else:
    #     wandb.init(mode="disabled")

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = (0)        # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            # if add_eos_token:
            #     user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    data = load_dataset("json", data_files=data_path)

    if test_run:
        data["train"]=data["train"].shuffle().select(range(100))

    val_set_size=round(data["train"].num_rows/10)
    train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)

    if test_run:
        train_val["train"]=train_val["train"].add_item({'input': "### Human: Your name is Open Assistant, right?",
        'output': "### Assistant: Yes, that is my name"})
    train_data = (train_val["train"].shuffle().map(generate_and_tokenize_prompt))
    val_data = (train_val["test"].shuffle().map(generate_and_tokenize_prompt))

    class GenerateTextCallback(TrainerCallback):
        def __init__(self,model, tokenizer, device): 
            self.model = model
            self.tokenizer = tokenizer
            self.device = device

        def generate_text(self,prompt):
            model.eval()
            # Generate text
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token_id = 0
            input_ids =self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.6,
                top_p=0.75,
                top_k=10,
                num_beams=torch.cuda.device_count(),
                num_return_sequences=1
            )
            output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            return output
        def on_evaluate(self, args, state, control, **kwargs):
            prompt = "### Human: Hey, who are you?\n### Assistant:"
            print("------------")
            print("Prompt:",prompt)
            generated_text = self.generate_text(prompt)
            print(f"---\n{generated_text}")
            print("------------")
            prompt = "### Human: Who is open assistant?\n### Assistant:"
            print("------------")
            print("Prompt:",prompt)
            generated_text = self.generate_text(prompt)
            print(f"---\n{generated_text}")
            print("------------")

    generate_text_callback = GenerateTextCallback(model=model,tokenizer=tokenizer, device="cuda")

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            lr_scheduler_type="constant",
            fp16=True,
            logging_steps=round((train_data.num_rows/batch_size)/10)+1,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps" if not test_run else "no",
            eval_steps=round((train_data.num_rows/batch_size)/2)+1,
            save_steps=round((train_data.num_rows/batch_size)/2),
            output_dir=output_dir,
            save_total_limit=40,
            group_by_length=False,
            report_to="wandb" if not test_run else None
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[generate_text_callback],
    )
    model.config.use_cache = False
    model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
