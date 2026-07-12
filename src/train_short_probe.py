
import sys
import os
# إضافة المجلد الرئيسي للمشروع وليس المجلد الفرعي
sys.path.append("/content/drive/MyDrive/Adaptive_Tutor_RL")

# الآن استورد مباشرة من داخل src
from src.rl.env import AdaptiveTutorEnv
from src.state_manager.session_tracker import SessionTracker


from stable_baselines3 import PPO



def probe_training():
    env = AdaptiveTutorEnv()
    tracker = SessionTracker()
    
    # تحميل النموذج أو إنشاء نموذج جديد
    model_path = "/content/drive/MyDrive/Adaptive_Tutor_RL/checkpoints/latest_model"
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, env=env, device="cpu")
    else:
        model = PPO("MlpPolicy", env, verbose=0, device="cpu")

    print("🔍 بدء التدريب الاستكشافي (1,000 خطوة)...")
    
    obs, _ = env.reset()
    student_responses = ["I don't know", "Yes, I know it", "Not sure", "Explain please"]
    
    for step in range(1000):
        # 1. تجهيز سياق الجلسة للبيئة
        current_level = env.levels[int(obs[0])]
        word = tracker.get_next_word(current_level)
        
        # تحديث بيانات البيئة (التي ستقرأها دالة step)
        env.last_student_input = student_responses[step % len(student_responses)]
        env.last_target_word = word
        env.last_teacher_text = "Teacher explains " + word
        
        # 2. اتخاذ القرار
        action, _ = model.predict(obs, deterministic=False)
        
        # 3. خطوة التعلم
        model.learn(total_timesteps=1)
        
        # 4. تحديث المراقبة (البيئة تعيد المراقبة الجديدة بعد التحديث)
        obs, reward, _, _, info = env.step(action)
        
        # 5. التقرير الدوري
        if step % 100 == 0:
            print(f"--- Step {step} | الكلمة: {word} ---")
            print(f"رد الطالب: {env.last_student_input} | القرار: {'PROCEED' if action == 0 else 'DEEP_EXPLAIN'}")
            print(f"المكافأة: {reward:.2f} | الحالة: {info['knowledge_status']}")

    model.save(model_path)
    print("✅ اكتمل التدريب الاستكشافي بنجاح.")

if __name__ == "__main__":
    probe_training()
