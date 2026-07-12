import torch
import json
import re
import random
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def load_full_word_context(file_path):
    """استخراج الكلمة والمادة العلمية بدقة من ملف التدريب لمنع الهلوسة"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines: return None, None

            random_line = random.choice(lines)
            data = json.loads(random_line)

            prompt_text = data.get("prompt", "")
            match = re.search(r"(### WORD DATA\n.*?)(?=\n### PEDAGOGICAL TASK)", prompt_text, re.DOTALL)

            if match:
                word_data_block = match.group(1).strip()
                word_match = re.search(r"- Target Word:\s*(.+)", word_data_block)
                target_word = word_match.group(1).strip() if word_match else "unknown"
                return target_word, word_data_block
            else:
                return "whenever", "- Target Word: whenever\n- Part of Speech: conjunction\n- CEFR Level: B1\n- Contextual Reference Examples:\n  1. You can ask for help whenever you need it."
    except Exception as e:
        print(f"Error reading file: {e}")
        return "whenever", "- Target Word: whenever"

def launch_real_tutor_v2():
    base_model_id = "Qwen/Qwen1.5-1.8B-Chat"
    adapter_dir = "/content/drive/MyDrive/Adaptive_Tutor_RL/models/qwen_sft_final_b"
    data_path = "/content/drive/MyDrive/Adaptive_Tutor_RL/data/processed/train_sft.jsonl"
    save_log_dir = "/content/drive/MyDrive/Adaptive_Tutor_RL/outputs"

    print("📥 Loading AI Tutor System - V2 [Strict Anchoring + Greedy Mode]...")
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

    target_word, word_data_block = load_full_word_context(data_path)

    print("\n" + "="*50)
    print(f"🎯 Target Word for Test 2: '{target_word}'")
    print("="*50 + "\n")

    # توجيهات الإصدار الثاني: التركيز المطلق على الكلمة المستهدفة والإنجليزية الصارمة
    system_prompt = (
        "You are an interactive, highly accurate English tutor.\n"
        f"Current Topic: You are teaching the specific word: '{target_word}'.\n"
        f"You must base your explanations EXACTLY on this data:\n{word_data_block}\n\n"
        "RULES:\n"
        "1. NEVER write the student's response. You are ONLY the teacher.\n"
        "2. Keep responses short and educational.\n"
        "3. NEVER lose focus on the target word. Every response must guide the student back to understanding this word.\n"
        "4. STRICT ENGLISH ONLY: Conduct 100% of the conversation in English. Reject any other language instantly.\n"
        "5. DO NOT invent outside examples. Use the ones provided above."
    )

    chat_history = [{"role": "system", "content": system_prompt}]

    initial_question = f"Hi! Do you know what the word '{target_word}' means?"
    chat_history.append({"role": "assistant", "content": initial_question})
    print(f"🤖 Tutor (V2): {initial_question}")

    while True:
        user_input = input("👤 You: ").strip()

        if user_input.lower() == 'exit':
            print("\n💾 Archiving Test 2 chat logs...")
            break

        if not user_input:
            continue

        # فحص محلي فوري لمنع أي لغة غير الإنجليزية
        if bool(re.search(r'[\u0600-\u06FF]', user_input)):
            print("🤖 Tutor (V2): Please respond only in English.")
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": "Please respond only in English."})
            continue

        chat_history.append({"role": "user", "content": user_input})

        text = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,          # الطور الحتمي لمنع الهلوسة والشرود الذهني للنموذج
                repetition_penalty=1.2,    # لمنع الإجابات الدائرية المكررة
                eos_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        if "Student:" in generated_text:
            generated_text = generated_text.split("Student:")[0].strip()

        print(f"🤖 Tutor (V2): {generated_text}")
        chat_history.append({"role": "assistant", "content": generated_text})

    # حفظ السجل المخصص للإصدار الثاني
    os.makedirs(save_log_dir, exist_ok=True)
    log_file_path = os.path.join(save_log_dir, "tutor_b_2_session_log.json")
    with open(log_file_path, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=4)

    print(f"✅ Test 2 Log saved to: {log_file_path}\n👋 Goodbye!")

if __name__ == "__main__":
    launch_real_tutor_v2()
