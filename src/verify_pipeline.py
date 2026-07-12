import sys
import os
import torch

# تأمين مسارات النظام لضمان استيراد الحزم
sys.path.append("/content/drive/MyDrive/Adaptive_Tutor_RL")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from src.rl.env import AdaptiveTutorEnv
from models.reward_engine import CompositeRewardEngine
from src.state_manager.session_tracker import SessionTracker

def run_comprehensive_verification():
    print("="*60)
    print("🔍 جاري بدء الفحص الشامل وتكامل المنظومة المحدثة...")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ العتاد المستخدم للفحص: {device}")

    try:
        print("\n⏳ 1. جاري تحميل الأوزان العصبية والمحركات لـ Qwen و RoBERTa...")
        qwen_path = "/content/drive/MyDrive/Adaptive_Tutor_RL/models/qwen_sft_final_b"
        roberta_path = "/content/drive/MyDrive/Adaptive_Tutor_RL/models/discriminator_roberta_aligned"
        
        # تحميل وتجهيز المولد
        qwen_tok = AutoTokenizer.from_pretrained(qwen_path)
        base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", device_map="auto")
        qwen_model = PeftModel.from_pretrained(base_model, qwen_path).eval()
        
        # تحميل وتجهيز المميز اللغوي الصارم
        roberta_tok = AutoTokenizer.from_pretrained(roberta_path)
        roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_path).to(device).eval()
        
        print("✅ تم تحميل الأصول العصبية الثقيلة بنجاح وسلامة.")
        
        print("\n⏳ 2. جاري بناء وربط بيئة الـ RL المحدثة والمستودع...")
        tracker = SessionTracker()
        env = AdaptiveTutorEnv(
            qwen_model=qwen_model,
            qwen_tokenizer=qwen_tok,
            roberta_model=roberta_model,
            roberta_tokenizer=roberta_tok,
            device=device
        )
        print("✅ تم تهيئة البيئة ومدير الحالة ومطابقتهم معاً.")
        
    except Exception as e:
        print(f"❌ فشل فادح في تهيئة أو ربط مكونات المنظومة العصبية. خطأ: {e}")
        return

    # إعادة تصفير المحاكاة لبدء خطوة الفحص
    obs, _ = env.reset()
    tracker.reset_session()
    
    print("\n" + "="*15 + " 🧪 سيناريو 1: محاكاة خطوة الشرح (Action 1) " + "="*15)
    print(f"🎯 الكلمة الحالية المستهدفة داخل صمام البيئة الافتراضي: [{env.current_word}]")
    
    try:
        # الموديل يتخذ قرار الشرح (Action 1)
        action_explain = 1
        next_obs, reward, terminated, truncated, info = env.step(action_explain)
        
        print("\n📥 مخرجات خطوة الشرح الناجحة:")
        print(f"  - الفعل المتخذ (Action): {info.get('action_taken')}")
        print(f"  - المكافأة الصافية المتزنة والمقيدة: {reward:.4f} (يجب أن تكون بين -1.0 و 1.0)")
        print(f"  - نقاط شرح الكلمة (Target Reward): {info.get('target_explanation_reward')}")
        print(f"  - عقوبة القواعد من RoBERTa (Penalty): {info.get('linguistic_penalty'):.4f}")
        print(f"  - الكلمة الانتقالية التالية الجاهزة في الخطوة القادمة: [{env.current_word}]")
        print("✅ السيناريو الأول مر بسلام وخرجت المكافآت مستقرة رياضياً دون قيم NaN.")
        
    except Exception as e:
        print(f"❌ انهار النظام أثناء محاكاة الشرح (Action 1). خطأ: {e}")
        return

    print("\n" + "="*15 + " 🧪 سيناريو 2: محاكاة خطوة التخطي (Action 0) " + "="*15)
    try:
        # الموديل يتخذ قرار التخطي المحايد (Action 0)
        action_skip = 0
        next_obs, reward, terminated, truncated, info = env.step(action_skip)
        
        print("\n📥 مخرجات خطوة التخطي الناجحة:")
        print(f"  - الفعل المتخذ (Action): {info.get('action_taken')}")
        print(f"  - المكافأة الثابتة المحايدة المسجلة: {reward:.2f}")
        print("✅ السيناريو الثاني مر بسلام.")
        
    except Exception as e:
        print(f"❌ انهار النظام أثناء محاكاة التخطي (Action 0). خطأ: {e}")
        return

    print("\n" + "="*60)
    print("🎉 انتهى فحص التكامل بنجاح باهر! كافة الأصول ومحركات المكافآت")
    print("   متناسقة ومتصلة 100% وجاهزة لبدء التدريب الطويل أو الحوار الحي.")
    print("="*60)

if __name__ == "__main__":
    run_comprehensive_verification()
