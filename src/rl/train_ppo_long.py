import sys
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions import Categorical
import numpy as np
import shutil

sys.path.append("/content/drive/MyDrive/Adaptive_Tutor_RL")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from src.rl.env import AdaptiveTutorEnv

def train_ppo_long_term(total_epochs=15, steps_per_epoch=100, initial_lr=5e-6):
    print("="*60)
    print("🚀 إطلاق قطار التدريب الطويل [مسارات معزولة وموحدة]")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️ العتاد المكتشف والمستخدم حالياً للتدريب: [{device.upper()}]")

    # 📌 التحديث الجوهري: توجيه كل المخرجات داخل المجلد الموحد المعزول
    base_models_dir = "/content/drive/MyDrive/Adaptive_Tutor_RL/models"
    ppo_parent_dir = os.path.join(base_models_dir, "ppo_tutor_checkpoints")
    
    last_checkpoint_dir = os.path.join(ppo_parent_dir, "checkpoint_last")
    best_checkpoint_dir = os.path.join(ppo_parent_dir, "checkpoint_best")
    final_output_dir = os.path.join(ppo_parent_dir, "qwen_ppo_final")
    checkpoint_states_file = os.path.join(last_checkpoint_dir, "optimizer_scheduler_states.pt")

    # مسارات الشحن الأساسية الثابتة
    qwen_load_path = os.path.join(base_models_dir, "qwen_sft_final_b")
    roberta_path = os.path.join(base_models_dir, "discriminator_roberta_aligned")

    start_epoch = 1
    best_reward = -float('inf')
    is_resumed = False
    saved_states = None

    if os.path.exists(last_checkpoint_dir) and os.path.exists(checkpoint_states_file):
        try:
            saved_states = torch.load(checkpoint_states_file, map_location=device)
            start_epoch = saved_states.get("next_epoch", 1)
            best_reward = saved_states.get("best_reward", -float('inf'))
            if start_epoch <= total_epochs:
                qwen_load_path = last_checkpoint_dir
                is_resumed = True
                print(f"♻️ [Auto-Resume]: استئناف معزول من الحقبة [{start_epoch}]")
        except Exception as e:
            print(f"⚠️ فشل قراءة أصول الاستئناف: {e}")

    print("\n⏳ جاري شحن النماذج والمميزات والمحولات لبيئة العمل...")
    qwen_tok = AutoTokenizer.from_pretrained(qwen_load_path)
    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", device_map="auto" if device == "cuda" else None)

    qwen_model = PeftModel.from_pretrained(base_model, qwen_load_path, is_trainable=True, low_cpu_mem_usage=False)
    if device == "cpu":
        qwen_model = qwen_model.to(device)

    roberta_tok = AutoTokenizer.from_pretrained(roberta_path)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_path).to(device).eval()

    env = AdaptiveTutorEnv(qwen_model=qwen_model, qwen_tokenizer=qwen_tok, roberta_model=roberta_model, roberta_tokenizer=roberta_tok, device=device)
    
    optimizer = optim.AdamW(qwen_model.parameters(), lr=initial_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    if is_resumed and saved_states is not None:
        try:
            optimizer.load_state_dict(saved_states["optimizer_state_dict"])
            scheduler.load_state_dict(saved_states["scheduler_state_dict"])
            print("⚙️ [State Aligned]: تم استعادة حالة المحسن والمجدول.")
        except Exception as e:
            print(f"⚠️ تضارب في الـ State Dict: {e}")

    print("... انطلاق الحلقات التدريبية المحدثة ✅")

    for epoch in range(start_epoch, total_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n🎬 --- بداية الحقبة التدريبية [{epoch}/{total_epochs}] | معدل التعلم الحالي (LR): {current_lr:.2e} ---")
        
        env.reset()
        epoch_rewards = []
        epoch_losses = []

        policy_weights = torch.randn(2, requires_grad=True, device=device)
        policy_optimizer = optim.Adam([policy_weights], lr=1e-3)

        for step_num in range(1, steps_per_epoch + 1):
            probs = torch.softmax(policy_weights, dim=-1)
            m = Categorical(probs)
            action_tensor = m.sample()
            action = action_tensor.item()

            simulated_responses = ["no", "yes", "detail", "explain more", "why", "i know it", "help me"]
            student_in = np.random.choice(simulated_responses)

            _, reward, _, _, info = env.step(action=action, student_response=student_in)
            epoch_rewards.append(reward)

            log_prob = m.log_prob(action_tensor)
            loss = -log_prob * reward

            policy_optimizer.zero_grad()
            loss.backward()
            policy_optimizer.step()
            epoch_losses.append(loss.item())

            if step_num % 20 == 0:
                print(f" 🟩 خطوة [{step_num}/{steps_per_epoch}] | الكلمة: [{info['word_discussed']}] | القرار: {info['action_taken']} | المكافأة: {reward:.4f}")

        avg_reward = np.mean(epoch_rewards)
        avg_loss = np.mean(epoch_losses)
        print(f"📊 ملخص الحقبة {epoch}: متوسط المكافآت = {avg_reward:.4f} | متوسط الـ Loss = {avg_loss:.4f}")

        scheduler.step(avg_reward)

        print(f"🔄 جاري تحديث وحفظ نقطة التحقق المعزولة للحقبة {epoch}...")
        temp_last_dir = last_checkpoint_dir + "_temp"
        if os.path.exists(temp_last_dir):
            shutil.rmtree(temp_last_dir)
            
        qwen_model.save_pretrained(temp_last_dir)
        qwen_tok.save_pretrained(temp_last_dir)
        
        states_to_freeze = {
            "next_epoch": epoch + 1,
            "best_reward": float(best_reward) if best_reward != -float('inf') else -999.0,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
        }
        torch.save(states_to_freeze, os.path.join(temp_last_dir, "optimizer_scheduler_states.pt"))

        if os.path.exists(last_checkpoint_dir):
            shutil.rmtree(last_checkpoint_dir)
        os.rename(temp_last_dir, last_checkpoint_dir)

        if avg_reward > best_reward:
            print(f"🏆 أعلى مكافأة جديدة: ({avg_reward:.4f})")
            best_reward = avg_reward
            temp_best_dir = best_checkpoint_dir + "_temp"
            if os.path.exists(temp_best_dir):
                shutil.rmtree(temp_best_dir)
            qwen_model.save_pretrained(temp_best_dir)
            qwen_tok.save_pretrained(temp_best_dir)
            if os.path.exists(best_checkpoint_dir):
                shutil.rmtree(best_checkpoint_dir)
            os.rename(temp_best_dir, best_checkpoint_dir)

    print(f"\n💾 جاري حفظ الموديل النهائي المعزول في: {final_output_dir}")
    if os.path.exists(final_output_dir):
        shutil.rmtree(final_output_dir)
    qwen_model.save_pretrained(final_output_dir)
    qwen_tok.save_pretrained(final_output_dir)
    
    if os.path.exists(last_checkpoint_dir):
        shutil.rmtree(last_checkpoint_dir)

    print("\n" + "="*60)
    print("🎉 انتهى التدريب المعزول والنظيف بنجاح!")
    print("="*60)

if __name__ == "__main__":
    train_ppo_long_term(total_epochs=15, steps_per_epoch=100, initial_lr=5e-6)
