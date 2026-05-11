import os
import warnings
from dataclasses import dataclass
from pathlib import Path
import wandb
import torch
from datasets import load_dataset,load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer,PreTrainedTokenizerBase
import json,random


from trl import (
    ModelConfig,
    ScriptArguments
)

from ppo_utils.ppo_config_medo1 import PPOConfig
from ppo_utils.ppo_trainer_medo1 import PPOTrainer


os.environ["WANDB_MODE"] = "offline"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'



from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    AutoTokenizer,
    HfArgumentParser
)
from transformers.utils import is_flash_attn_2_available, is_peft_available

if is_peft_available():
    from peft import PeftConfig, PeftModel


def get_model_load_kwargs(training_args):
    model_load_kwargs = {}
    using_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true"
    zero_stage = os.environ.get("ACCELERATE_DEEPSPEED_ZERO_STAGE")

    if not (using_deepspeed and zero_stage == "3"):
        model_load_kwargs["low_cpu_mem_usage"] = True

    if training_args.bf16:
        model_load_kwargs["torch_dtype"] = torch.bfloat16
    elif training_args.fp16:
        model_load_kwargs["torch_dtype"] = torch.float16

    if is_flash_attn_2_available():
        model_load_kwargs["attn_implementation"] = "flash_attention_2"
        return model_load_kwargs

    warnings.warn(
        "flash_attn is not installed; falling back to the default attention implementation."
    )
    return model_load_kwargs


def get_reward_model_load_kwargs(training_args):
    reward_model_load_kwargs = get_model_load_kwargs(training_args)
    compute_dtype = torch.float32
    if training_args.bf16:
        compute_dtype = torch.bfloat16
    elif training_args.fp16:
        compute_dtype = torch.float16

    reward_model_load_kwargs.pop("torch_dtype", None)
    reward_model_load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    reward_model_load_kwargs["device_map"] = "auto"
    return reward_model_load_kwargs


def resolve_policy_model_path(model_name_or_path: str) -> tuple[bool, str]:
    adapter_config_path = Path(model_name_or_path) / "adapter_config.json"
    if adapter_config_path.exists():
        return True, str(adapter_config_path.parent)
    return False, model_name_or_path


def load_policy_and_ref_policy(model_name_or_path, model_load_kwargs):
    is_adapter_model, resolved_model_path = resolve_policy_model_path(model_name_or_path)
    if not is_adapter_model:
        ref_policy = AutoModelForCausalLM.from_pretrained(resolved_model_path, **model_load_kwargs)
        policy = AutoModelForCausalLM.from_pretrained(resolved_model_path, **model_load_kwargs)
        return policy, ref_policy, resolved_model_path

    if not is_peft_available():
        raise ImportError("PEFT is required when model_name_or_path points to a LoRA adapter.")

    peft_config = PeftConfig.from_pretrained(resolved_model_path)
    base_model_path = peft_config.base_model_name_or_path
    base_policy = AutoModelForCausalLM.from_pretrained(base_model_path, **model_load_kwargs)
    policy = PeftModel.from_pretrained(base_policy, resolved_model_path)
    return policy, None, resolved_model_path


def ensure_distinct_pad_token(tokenizer):
    if tokenizer.pad_token_id != tokenizer.eos_token_id:
        return False

    if "<|pad|>" in tokenizer.get_vocab():
        tokenizer.pad_token = "<|pad|>"
        return False

    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return True


def apply_training_memory_settings(model, training_args):
    if model is None:
        return

    if hasattr(model, "config"):
        model.config.use_cache = False

    if not training_args.gradient_checkpointing:
        return

    gradient_checkpointing_enable = getattr(model, "gradient_checkpointing_enable", None)
    if callable(gradient_checkpointing_enable):
        gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs,
        )

class ppo_dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length = 1000,debug = 0):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
    
        newdata = []
        for da in self.data:
            if len(da['Open-ended Verifiable Question']) > 0 and len(da['Ground-True Answer']) > 0:
                newdata.append({'question':da['Open-ended Verifiable Question'],'answer':da['Ground-True Answer']})
        print(len(self.data),' -> ',len(newdata))
        self.data = newdata

        self.debug = debug     

    def __getitem__(self, index):
        return self.data[index]

    def get_prompt(self,da):
        message = [{"role": "user", "content": da['question']}]
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        input_token = self.tokenizer(
            prompt,
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )

        da['input_ids'] = input_token["input_ids"]
        return da

    def collate_fn(self, batch):
        data = [ self.get_prompt(da) for da in batch]
        input_ids = [item["input_ids"] for item in data]
        question = [item["question"] for item in data]
        answer = [item["answer"] for item in data]

        max_len = max(len(x) for x in input_ids)
        max_len = min(max_len,self.max_length)
        input_ids = [ [self.tokenizer.pad_token_id]*(max_len-len(item)) + item[:max_len] for item in input_ids]

        if self.debug > 0:
            print('[input_ids]',self.tokenizer.decode(input_ids[-1]))
            print('[question]',question[-1])
            print('[answer]',answer[-1])
            self.debug -= 1
        return {
                "input_ids": torch.LongTensor(input_ids),
                "question": question,
                "answer": answer
            }

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    model_load_kwargs = get_model_load_kwargs(training_args)
    reward_model_load_kwargs = get_reward_model_load_kwargs(training_args)
    value_model_load_kwargs = {k: v for k, v in model_load_kwargs.items() if k != "low_cpu_mem_usage"}

    output_dir = training_args.output_dir
    run_name = training_args.run_name
    if run_name not in output_dir:
        output_dir = os.path.join(output_dir,run_name)
        training_args.output_dir = output_dir
    
    policy, ref_policy, tokenizer_source = load_policy_and_ref_policy(
        model_config.model_name_or_path, model_load_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    added_pad_token = ensure_distinct_pad_token(tokenizer)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, num_labels=2, **reward_model_load_kwargs
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.value_model_path,
        trust_remote_code=model_config.trust_remote_code,
        num_labels=1,
        **value_model_load_kwargs,
    )

    apply_training_memory_settings(policy, training_args)
    apply_training_memory_settings(value_model, training_args)

    if added_pad_token:
        if ref_policy is not None:
            ref_policy.resize_token_embeddings(len(tokenizer))
        policy.resize_token_embeddings(len(tokenizer))

    reward_tokenizer = AutoTokenizer.from_pretrained(training_args.reward_model_path)
    assert tokenizer.pad_token_id != tokenizer.eos_token_id

    training_args.stop_token_id = tokenizer.eos_token_id

    eval_ratio = 0.1
    eval_max_num = 200
    with open(script_args.dataset_name) as f:
        data = json.load(f)
    random.shuffle(data)
    eval_num = min(int(len(data) * eval_ratio),eval_max_num)
    train_dataset = ppo_dataset(data[eval_num:],tokenizer, debug = 1)
    eval_dataset = ppo_dataset(data[:eval_num],tokenizer)

    trainer = PPOTrainer(
        config=training_args,
        processing_class=tokenizer,
        reward_processing_class = reward_tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator = train_dataset.collate_fn
    )
    trainer.train()