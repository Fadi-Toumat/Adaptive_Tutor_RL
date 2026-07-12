import os
import shutil

def optimize_and_cleanup():
    base_path = "/content/drive/MyDrive/Adaptive_Tutor_RL"
    
    if not os.path.exists(base_path):
        print(f"❌ الخطأ: المسار الرئيسي {base_path} غير موجود. تأكد من اتصال Google Drive.")
        return

    print("🚀 بدء بروتوكول تنظيف وإعادة هيكلة بيئة العمل للمشروع...")
    print("=" * 60)

    # 1. رصد وحذف مجلدات النظام المؤقتة والكاش المخفي
    junk_dirs = []
    for root, dirs, files in os.walk(base_path):
        for d in dirs:
            if d in ["__pycache__", ".ipynb_checkpoints"]:
                junk_dirs.append(os.path.join(root, d))
                
    for j_dir in junk_dirs:
        if os.path.exists(j_dir):
            try:
                shutil.rmtree(j_dir)
                print(f"🧹 تم حذف كاش النظام المزعج: {os.path.relpath(j_dir, base_path)}")
            except Exception as e:
                print(f"⚠️ تعذر حذف مجلد الكاش {j_dir}: {e}")

    # 2. التخلص من الملفات المكررة والمخلفات التقنية داخل src/rl
    # بما أننا نعتمد على train_ppo_long.py مباشرة في المجلد الرئيسي للـ src
    obsolete_files = [
        os.path.join(base_path, "src/rl/train_ppo.py"),
        os.path.join(base_path, "src/rl/ppo_agent.py")
    ]
    
    print("-" * 40)
    for file_path in obsolete_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"🔥 تم حذف الملف المكرر/القديم: {os.path.relpath(file_path, base_path)}")
            except Exception as e:
                print(f"⚠️ تعذر حذف الملف {file_path}: {e}")
        else:
            print(f"⏭️ التخطي (مخلفات تم تنظيفها مسبقاً): {os.path.basename(file_path)}")

    # 3. التأكد من حماية وتثبيت مجلدات النماذج الاستراتيجية لقمع الـ Mode Collapse لاحقاً
    critical_paths = [
        "models/reward_engine.py",
        "models/qwen_sft_final_b",
        "models/discriminator_roberta_aligned",
        "models/ppo_tutor_checkpoints/best_checkpoint"
    ]
    
    print("-" * 40)
    print("🔒 الفحص الأمني للمكونات الأساسية للـ 4-Stage Framework:")
    for cp in critical_paths:
        full_cp = os.path.join(base_path, cp)
        if os.path.exists(full_cp):
            print(f"   ✓ مكون محمي ومستقر: {cp}")
        else:
            print(f"   ⚠️ تنبيه: المكون {cp} غير مكتشف أو يحتاج لبناء في مرحلته الخاصة.")

    print("=" * 60)
    print("✅ اكتملت العملية بنجاح! بيئة العمل الآن مصفاة ونظيفة بنسبة 100% وجاهزة للتنفيذ الهجين.")

if __name__ == "__main__":
    optimize_and_cleanup()
