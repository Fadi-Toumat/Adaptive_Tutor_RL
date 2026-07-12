import sys
import os
import numpy as np
from stable_baselines3 import PPO

sys.path.append("/content/drive/MyDrive/Adaptive_Tutor_RL")
from src.rl.env import AdaptiveTutorEnv

def run_policy_sanity_check():
    print("="*60)
    print("📊 جاري فحص ذكاء النموذج المستفاد (RL Policy Evaluation)...")
    print("="*60)
    
    env = AdaptiveTutorEnv()
    model_path = "/content/drive/MyDrive/Adaptive_Tutor_RL/models/ppo_tutor_checkpoints/best_model"
    
    if not os.path.exists(model_path + ".zip"):
        print("❌ لم يتم العثور على أفضل نموذج في المسار الفعلي!")
        return
        
    model = PPO.load(model_path, env=env)
    
    # محاكاة حالة اختبارية: الطالب في المستوى A1 (القيمة 0)
    obs = np.array([0.0], dtype=np.float32)
    
    print("\n💡 [الاختبار المعياري 1]: الطالب لغوياً يظهر 'استيعاب كامل' (knows)")
    # نحقن حالة الاستيعاب في البيئة
    env.last_student_input = "Yes, I completely understand this word and everything is fine."
    env.step(action=0) # خطوة وهمية لتحديث كاش الحالة اللغوية داخل النواة
    
    action, _ = model.predict(obs, deterministic=True)
    print(f"🤖 قرار الموديل المدرب: {action}")
    print(f"📌 التفسير: {'✅ ممتاز! اختار التجاوز لأن الطالب مستوعب' if action == 0 else '⚠️ خيار غير دقيق (اختار الشرح رغم المعرفة)'}")
    
    print("\n💡 [الاختبار المعياري 2]: الطالب لغوياً يظهر 'تعثر شديد' (doesnt_know)")
    # نحقن حالة التعثر في البيئة
    env.last_student_input = "I dont know, this is hard and not sure, explain please."
    env.step(action=1) # خطوة وهمية لتحديث كاش الحالة اللغوية داخل النواة
    
    action, _ = model.predict(obs, deterministic=True)
    print(f"🤖 قرار الموديل المدرب: {action}")
    print(f"📌 التفسير: {'✅ ممتاز! اختار الشرح والتعمق لأن الطالب متعثر' if action == 1 else '⚠️ خيار غير دقيق (اختار التجاوز رغم التعثر)'}")

if __name__ == "__main__":
    run_policy_sanity_check()
