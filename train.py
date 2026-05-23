import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32
)

print("모델과 토크나이저를 불러오는 중입니다...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files="data.jsonl", split="train")

training_args = SFTConfig(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    max_steps=15,
    save_steps=15,
    optim="adamw_torch",
    fp16=False,
    use_cpu=True,
    dataset_text_field="text",
    max_length=256,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

print("파인튜닝을 시작합니다!")
trainer.train()

output_dir = "./my-custom-lora"
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"학습 완료! 어댑터가 '{output_dir}' 폴더에 저장되었습니다.")
