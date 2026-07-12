import json
import random
import os
from transformers import AutoTokenizer

def clean_and_format_examples(examples_list):
    """تنظيف قائمة الأمثلة وتحويلها إلى نص منقط ومفهوم للنماذج الصغيرة"""
    if not examples_list:
        return "No examples provided."
    cleaned_lines = []
    for i, ex in enumerate(examples_list, 1):
        clean_ex = str(ex).strip().replace('\u2009', ' ')
        cleaned_lines.append(f"  {i}. {clean_ex}")
    return "\n".join(cleaned_lines)

def transform_shuffle_and_split_qwen(input_path, train_out_path, val_out_path, test_out_path, train_ratio=0.80, val_ratio=0.10):
    if not os.path.exists(input_path):
        print(f"❌ خطأ: الملف غير موجود في المسار: {input_path}")
        return

    print("📥 Loading tokenizer to format prompt/completion pairs natively...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", trust_remote_code=True)

    formatted_dataset = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                word = data.get("word", "").strip()
                w_type = data.get("type", "").strip()
                level = data.get("level", "").strip()
                examples = data.get("examples", [])
                dialogue = data.get("dialogue", "").strip()
                
                formatted_examples = clean_and_format_examples(examples)
                
                user_content = (
                    f"### WORD DATA\n"
                    f"- Target Word: {word}\n"
                    f"- Part of Speech: {w_type}\n"
                    f"- CEFR Proficiency Level: {level}\n"
                    f"- Contextual Reference Examples:\n{formatted_examples}\n\n"
                    f"### PEDAGOGICAL TASK\n"
                    f"Act as an adaptive tutor. Initiate the structured dialogue from the training data to teach the student the word '{word}' naturally, adhering to the provided CEFR level constraints."
                )
                
                messages = [
                    {"role": "system", "content": "You are an expert, patient, and adaptive English language tutor."},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": dialogue}
                ]
                
                # تطبيق قالب Qwen للحصول على النص الكامل للمحادثة
                full_chat = tokenizer.apply_chat_template(messages, tokenize=False)
                
                # فصل النص بدقة رياضية عند بداية استجابة المساعد (Assistant)
                target_boundary = "<|im_start|>assistant\n"
                if target_boundary in full_chat:
                    parts = full_chat.split(target_boundary, 1)
                    prompt = parts[0] + target_boundary
                    completion = parts[1]
                else:
                    # ميكانيكية احتياطية في حال تغير القالب
                    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False) + target_boundary
                    completion = dialogue + tokenizer.eos_token
                
                # بناء الهيكل الحديث المعتمد للـ SFT الموجه
                formatted_entry = {
                    "prompt": prompt,
                    "completion": completion
                }
                formatted_dataset.append(formatted_entry)
            except Exception as e:
                print(f"⚠️ خطأ في معالجة السطر: {e}")
                continue

    print(f"📦 إجمالي العينات المعالجة والمحسنة لـ Qwen: {len(formatted_dataset)}")
    
    random.seed(42) 
    random.shuffle(formatted_dataset)
    
    total = len(formatted_dataset)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_data = formatted_dataset[:train_end]
    val_data = formatted_dataset[train_end:val_end]
    test_data = formatted_dataset[val_end:]
    
    for path, data in [(train_out_path, train_data), (val_out_path, val_data), (test_out_path, test_data)]:
        with open(path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"✅ تم الحفظ بنجاح وتوليد المجموعات الثلاث بنسق (Prompt/Completion) المعتمد حديثاً.")

if __name__ == "__main__":
    transform_shuffle_and_split_qwen(
        input_path='/content/drive/MyDrive/Adaptive_Tutor_RL/data/processed/final_merged_dataset.jsonl',
        train_out_path='/content/drive/MyDrive/Adaptive_Tutor_RL/data/processed/train_sft.jsonl',
        val_out_path='/content/drive/MyDrive/Adaptive_Tutor_RL/data/processed/val_sft.jsonl',
        test_out_path='/content/drive/MyDrive/Adaptive_Tutor_RL/data/processed/test_sft.jsonl'
    )
