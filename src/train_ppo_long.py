
import sys
import os

# الحل: إضافة المسار الرئيسي للمشروع إلى مسار النظام
# هذا يجعل مجلد 'src' معروفاً للمترجم أينما كنت
sys.path.append("/content/drive/MyDrive/Adaptive_Tutor_RL")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from src.rl.env import AdaptiveTutorEnv
from stable_baselines3 import PPO

def train_ppo_tutor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 بدء عملية التدريب (Fine-tuning) باستخدام السياسة الجديدة على: {device}")

    # 1. تحميل الأصول الفعلية
    qwen_path = "/content/drive/MyDrive/Adaptive_Tutor_RL/models/qwen_sft_final_b"
    roberta_path = "/content/drive/MyDrive/Adaptive_Tutor_RL/models/discriminator_roberta_aligned"
    
    roberta_tok = AutoTokenizer.from_pretrained(roberta_path)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_path).to(device).eval()
    
    # 2. تهيئة البيئة المحدثة
    env = AdaptiveTutorEnv(
        roberta_model=roberta_model,
        roberta_tokenizer=roberta_tok,
        device=device
    )

    # 3. إعداد خوارزمية PPO
    # ملاحظة: في أنظمة RL للنصوص، نستخدم عادةً PPO مع واجهة Gym
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=1e-5,
        batch_size=32,
        n_steps=128
    )

    # 4. بدء دورة التدريب
    print("⏳ جاري تدريب النموذج على دالة المكافأة المحدثة (Qwen Explanation + RoBERTa Grammar Penalty)...")
    try:
        model.learn(total_timesteps=1000)
        
        # حفظ النموذج بعد التدريب
        save_path = "/content/drive/MyDrive/Adaptive_Tutor_RL/models/ppo_tutor_checkpoints/best_checkpoint"
        model.save(save_path)
        print(f"✅ تم حفظ النموذج المدرب بنجاح في: {save_path}")
        
    except Exception as e:
        print(f"❌ خطأ أثناء التدريب: {e}")

if __name__ == "__main__":
    train_ppo_tutor()
