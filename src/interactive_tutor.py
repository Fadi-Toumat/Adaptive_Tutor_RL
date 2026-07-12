import sys
import os
import re
import torch

# تأمين مسارات النظام لضمان استيراد الحزم دون مشاكل
sys.path.append("/content/drive/MyDrive/Adaptive_Tutor_RL")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from src.rl.env import AdaptiveTutorEnv

def load_all_assets():
    """تحميل الأصول العصبية المحدثة بالـ PPO لمرة واحدة في الذاكرة"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("⏳ جاري شحن الأوزان الذكية الجديدة المحدثة بالـ [PPO Final]...")
    
    # تحويل المسار رسمياً للمجلد النهائي المطور لـ Qwen
    qwen_path = "/content/drive/MyDrive/Adaptive_Tutor_RL/models/qwen_ppo_final"
    roberta_path = "/content/drive/MyDrive/Adaptive_Tutor_RL/models/discriminator_roberta_aligned"
    
    # تحميل المولد التفاعلي (Qwen المدرب بالـ PPO)
    qwen_tok = AutoTokenizer.from_pretrained(qwen_path)
    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", device_map="auto" if device == "cuda" else None)
    qwen_model = PeftModel.from_pretrained(base_model, qwen_path).eval()
    if device == "cpu":
        qwen_model = qwen_model.to(device)
    
    # تحميل المميز اللغوي الصارم (RoBERTa)
    roberta_tok = AutoTokenizer.from_pretrained(roberta_path)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_path).to(device).eval()
    
    return qwen_model, qwen_tok, roberta_model, roberta_tok, device

def validate_english_only(user_input):
    """آلية إجبار المستخدم على استخدام اللغة الإنجليزية"""
    cleaned_input = user_input.strip()
    if not cleaned_input:
        return False
    if re.search(r'[\u0600-\u06FF]', cleaned_input):
        return False
    return True

def run_interactive_tutor_session():
    qwen, qwen_tok, roberta, roberta_tok, device = load_all_assets()
    
    # بناء وتهيئة بيئة الـ RL المركزية المصححة لإدارة الجلسة والعدادات برمجياً بصورة موحدة
    env = AdaptiveTutorEnv(
        qwen_model=qwen,
        qwen_tokenizer=qwen_tok,
        roberta_model=roberta,
        roberta_tokenizer=roberta_tok,
        device=device
    )
    
    env.reset()
    
    print("\n" + "="*60)
    print(f"🎓 نظام المدرس التفاعلي الذكي [نسخة الـ PPO المحدثة والمحكومة بالكامل]")
    print("⚠️ ملاحظة: تم إصلاح ثغرات خلط الكلمات والترقيات العشوائية بنجاح.")
    print("="*60)
    
    target_word = env.current_word
    print(f"\n🤖 المدرس: Do you know the meaning of the word '{target_word}'?")

    while True:
        current_level = env.levels[env.current_level_idx]
        user_in = input(f"\n[Word: {target_word} | Level: {current_level}] Student: ").strip()
        
        if user_in.lower() == 'exit':
            print("👋 Session ended. Good luck with your studies!")
            break
            
        if not user_in:
            continue

        # ─── صمام الأمان الأول: إجبار استخدام اللغة الإنجليزية ───
        if not validate_english_only(user_in):
            print("❌ [System Message]: Please respond using English only! Arabic or foreign characters are not allowed.")
            print(f"🤖 المدرس: Let's try again in English. Do you know what '{target_word}' means?")
            continue 

        # ─── صمام الأمان الثاني: التصنيف التكيفي لرد الطالب واتخاذ الفعل ───
        response_clean = user_in.lower().strip()
        unknown_keywords = ["no", "don't know", "dont know", "not sure", "more detail", "explain", "help", "clue", "details", "why"]
        
        if any(kw in response_clean for kw in unknown_keywords):
            action = 1  
        else:
            action = 0  

        # إرسال الفعل والرد مباشرة إلى صمام البيئة المصحح
        _, reward, _, _, info = env.step(action=action, student_response=user_in)
        
        # ─── عرض تفاصيل ومخرجات الفعل المتخذ حياً ───
        if action == 0:
            print(f"🤖 [Action 0 - Proceed]: Perfect! Since you know it, let's move forward.")
            
            if info["current_level"] != current_level:
                print(f"🎉 Congratulations! Mastered {env.up_thresholds.get(current_level, 3)} words consecutively.")
                print(f"🎉 System: You have been promoted to level: {info['current_level']}")
            
            target_word = env.current_word
            print(f"\n🤖 المدرس: Well, how about the word '{target_word}'?")
            
        elif action == 1:
            print("🤖 [Action 1 - Deep Explain]: Generating explanation and examples...")
            print(f"\n🤖 المدرس:\n{info['generated_text']}")
            print(f"\n⚙️ [RL Metrics]: Reward Score: {reward:.4f} | Target Reward: {info.get('target_explanation_reward', 0.0)} | RoBERTa Penalty: {info.get('linguistic_penalty', 0.0):.4f}")
            
            # تحديث الكلمة المستهدفة يظل ثابتاً في حالة الشرح لضمان استقرار النقاش اللغوي
            target_word = env.current_word
            print(f"\n🤖 المدرس: Is this explanation clear enough? Or do you need more details about '{target_word}'?")

if __name__ == "__main__":
    run_interactive_tutor_session()
