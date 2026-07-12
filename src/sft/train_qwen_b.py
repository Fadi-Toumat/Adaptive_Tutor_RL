import os
import torch
import re
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    EarlyStoppingCallback,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

# تثبيت العشوائية لضمان استقرار النتائج
set_seed(42)

# =====================================================================
# 1. الإعدادات الثابتة ومعاملات التشغيل المدمجة (Hyperparameters)
# =====================================================================
BASE_MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
DATA_PATH = "/content/drive/MyDrive/Adaptive_Tutor_RL/data/processed/train_sft.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/Adaptive_Tutor_RL/models/qwen_sft_final_b"

EPOCHS = 5                            
LEARNING_RATE = 2e-4                  
BATCH_SIZE = 2                        
GRADIENT_ACCUMULATION_STEPS = 4       
MAX_LENGTH = 1280                     

def main():
    print(f"🚀 Initiating Ultimate Production Pipeline (Fixed for Latest TRL)...")
    print(f"📦 Targeted Architecture: {BASE_MODEL_ID} | Total Epochs: {EPOCHS}\n" + "="*60)
    
    # 2. تحميل الـ Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. تكوين تكميم النموذج لحماية الذاكرة (4-bit QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    print("📥 Loading Quantized Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    base_model = prepare_model_for_kbit_training(base_model)

    # 4. إعداد الـ LoRA الشامل لجميع الطبقات الخطية
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # 5. دالة إعادة الهيكلة للحوار المتعدد وفرض اللغة الإنجليزية
    def format_conversational_data(example):
        prompt_text = example['prompt']
        completion_text = example['completion']
        
        try:
            user_part = prompt_text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0].strip()
        except:
            user_part = prompt_text
            
        messages = [
            {"role": "system", "content": "You are an expert, patient, and adaptive English language tutor. You must conduct the entire dialogue strictly in English."},
            {"role": "user", "content": user_part}
        ]
        
        turns = completion_text.replace("\\n", "\n").split("\n\n")
        for turn in turns:
            turn = turn.strip()
            if not turn: continue
            if turn.startswith("Teacher:"):
                messages.append({"role": "assistant", "content": turn.replace("Teacher:", "").strip()})
            elif turn.startswith("Student:"):
                messages.append({"role": "user", "content": turn.replace("Student:", "").strip()})
                
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    # 6. تحميل البيانات وتقسيمها
    print("\n📊 Preprocessing Dataset...")
    raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    
    train_dataset = split_dataset["train"].map(format_conversational_data, remove_columns=raw_dataset.column_names)
    eval_dataset = split_dataset["test"].map(format_conversational_data, remove_columns=raw_dataset.column_names)
    
    print(f"✅ Data Formatted. Train samples: {len(train_dataset)} | Validation samples: {len(eval_dataset)}")

    # 7. التكوين الحديث لـ SFTConfig (تنظيف التحذيرات الجانبية)
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,                   
        load_best_model_at_end=True,          
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="none",
        dataset_text_field="text",            
        max_length=MAX_LENGTH                 
    )

    # 8. بناء المدرب وتمرير صمام التوقف الذكي
    # التغيير الجوهري هنا: استبدال tokenizer بـ processing_class للتوافق مع التحديث الأخير
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        processing_class=tokenizer, # <-- تم الإصلاح الجذري هنا
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] 
    )

    # 9. آلية الاستكمال التلقائي الذكي من نقاط الحفظ
    resume_checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        checkpoints = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR) if "checkpoint-" in d]
        if checkpoints:
            resume_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"🔄 Checkpoint detected! Automatically resuming training from: {resume_checkpoint}")

    # 10. إطلاق محرك التدريب
    print("\n🔥 Launching Engine...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 11. تأمين وحفظ الأوزان النهائية للنموذج الأفضل
    print(f"\n💾 Archiving the absolute best weights to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("🎯 Supervised Fine-Tuning Process Finished Successfully!")

if __name__ == "__main__":
    main()
