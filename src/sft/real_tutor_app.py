import torch
import json
import re
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def load_random_word_from_data(file_path):
    """دالة لفتح ملفك وسحب كلمة تلقائياً دون تدخل منك"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines: return "apple", "A1"
            
            random_line = random.choice(lines)
            data = json.loads(random_line)
            
            # استخراج الكلمة والمستوى من نص الـ prompt الخاص بك
            word_match = re.search(r"- Target Word:\s*(.+)", data.get("prompt", ""))
            level_match = re.search(r"- CEFR Proficiency Level:\s*(.+)", data.get("prompt", ""))
            
            word = word_match.group(1).strip() if word_match else "apple"
            level = level_match.group(1).strip() if level_match else "A1"
            return word, level
    except Exception as e:
        print(f"Error reading file: {e}")
        return "apple", "A1"

def launch_real_tutor():
    base_model_id = "Qwen/Qwen1.5-1.8B-Chat"
    adapter_dir = "/content/drive/MyDrive/Adaptive_Tutor_RL/models/qwen_sft_final"
    data_path = "/content/drive/MyDrive/Adaptive_Tutor_RL/data/processed/train_sft.jsonl"
    
    print("📥 Loading AI Tutor System...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto"),
        adapter_dir
    )
    model.eval()
    
    # النظام يسحب الكلمة تلقائياً
    target_word, level = load_random_word_from_data(data_path)
    
    print("\n" + "="*50)
    print(f"🎯 Target Word Selected by System: '{target_word}' (Level: {level})")
    print("="*50 + "\n")
    
    # توجيه صارم للنموذج ليتصرف كمعلم حواري
    system_prompt = (
        f"You are an interactive English tutor. Your task is to teach the word '{target_word}'. "
        "RULES:\n"
        "1. NEVER write the student's response. You are ONLY the teacher.\n"
        "2. Keep your responses short.\n"
        "3. Evaluate the student's answer. If they know the word, praise them. If they don't, explain it with an example."
    )
    
    chat_history = [{"role": "system", "content": system_prompt}]
    
    # الضربة الأولى: النموذج يسألك تلقائياً
    initial_question = f"Hi! Do you know what the word '{target_word}' means?"
    chat_history.append({"role": "assistant", "content": initial_question})
    print(f"🤖 Tutor: {initial_question}")
    
    while True:
        user_input = input("👤 You: ")
        if user_input.lower() == 'exit': break
        
        chat_history.append({"role": "user", "content": user_input})
        
        text = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.4, # حرارة منخفضة ليكون دقيقاً
                repetition_penalty=1.1,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # حماية برمجية: إذا حاول النموذج كتابة "Student:" نقوم بقطع النص
        if "Student:" in generated_text:
            generated_text = generated_text.split("Student:")[0].strip()
            
        print(f"🤖 Tutor: {generated_text}")
        chat_history.append({"role": "assistant", "content": generated_text})

if __name__ == "__main__":
    launch_real_tutor()
