import sys
import numpy as np
from stable_baselines3 import PPO

# إضافة مسار المشروع
sys.path.append("/content/drive/MyDrive/Adaptive_Tutor_RL")
from src.rl.env import AdaptiveTutorEnv

def test_model_logic():
    env = AdaptiveTutorEnv()
    # تحميل النموذج الذي قمنا بتعديله
    model_path = "/content/drive/MyDrive/Adaptive_Tutor_RL/checkpoints/best_model"
    model = PPO.load(model_path, env=env)
    
    # حالات اختبارية (Scenario Testing)
    # Scenario: (input_text, target_word)
    scenarios = [
        ("I don't understand", "baby"),  # حالة يجب فيها اختيار Action 1 (Deep Explain)
        ("I know this", "forty"),        # حالة يجب فيها اختيار Action 0 (Proceed)
        ("Not sure", "riot"),           # حالة يجب فيها اختيار Action 1
        ("Yes, I understand", "trigger") # حالة يجب فيها اختيار Action 0
    ]
    
    print("🧪 بدء اختبار المنطق التربوي...")
    obs, _ = env.reset()
    
    for student_input, word in scenarios:
        action, _ = model.predict(obs, deterministic=True)
        # تنفيذ الخطوة
        obs, reward, _, _, info = env.step(action, student_input, word, "Teacher explains " + word)
        
        print(f"\nالكلمة: {word} | رد الطالب: {student_input}")
        print(f"القرار المتخذ: {'PROCEED (0)' if action == 0 else 'DEEP_EXPLAIN (1)'}")
        print(f"نتيجة المكافأة: {reward:.2f} | الحالة: {info['knowledge_status']}")
        
        # منطق التحقق (Verification)
        if info['knowledge_status'] == "doesnt_know" and action == 0:
            print("❌ تحذير: النموذج اتخذ قراراً خاطئاً (انتقل بينما الطالب لا يعرف).")
        elif info['knowledge_status'] == "knows" and action == 1:
            print("⚠️ ملاحظة: النموذج اتخذ قراراً متحفظاً (تعمق في الشرح رغم معرفة الطالب).")
        else:
            print("✅ القرار منطقي.")

if __name__ == "__main__":
    test_model_logic()
