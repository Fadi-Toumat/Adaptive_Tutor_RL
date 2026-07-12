import sys
import os
import re
import torch
import gradio as gr

# تأمين مسارات النظام لضمان استيراد الحزم دون مشاكل
sys.path.append("/content/drive/MyDrive/Adaptive_Tutor_RL")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from src.rl.env import AdaptiveTutorEnv

# ─── 1. التحقق من مسارات النماذج المحلية ───
qwen_path = "/content/drive/MyDrive/Adaptive_Tutor_RL/models/ppo_tutor_checkpoints_fixed_lr/qwen_ppo_fixed_lr_final"
roberta_path = "/content/drive/MyDrive/Adaptive_Tutor_RL/models/discriminator_roberta_aligned"

if not os.path.exists(qwen_path):
    raise FileNotFoundError(f"❌ خطأ: المسار غير موجود: '{qwen_path}'")

print("⏳ [Gradio Core] Paths verified successfully. Loading neural components...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# تحميل أوزان المولد المطور بالـ PPO محلياً بأمان
qwen_tok = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", device_map="auto" if device == "cuda" else None)
qwen_model = PeftModel.from_pretrained(base_model, qwen_path).eval()
if device == "cpu":
    qwen_model = qwen_model.to(device)

# تحميل أوزان المميز القواعدي محلياً بأمان
roberta_tok = AutoTokenizer.from_pretrained(roberta_path, local_files_only=True)
roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_path, local_files_only=True).to(device).eval()

# بناء بيئة الـ RL الموحدة لإدارة الجلسة تكيفياً
env = AdaptiveTutorEnv(
    qwen_model=qwen_model,
    qwen_tokenizer=qwen_tok,
    roberta_model=roberta_model,
    roberta_tokenizer=roberta_tok,
    device=device
)
env.reset()

# تهيئة العدادات الفرعية بشكل آمن ومستقر
if not hasattr(env, 'down_streak'):
    env.down_streak = 0
if not hasattr(env, 'last_action_was_explain'):
    env.last_action_was_explain = False

# ─── 2. هندسة إدارة الحالات التكيفية للأزرار والعدادات المستقلة ───
def process_rl_action(action_type, chat_history):
    word_just_discussed = env.current_word
    current_level_str = env.levels[env.current_level_idx]
    
    target_up_streak = env.up_streak
    target_down_streak = env.down_streak
    target_level_idx = env.current_level_idx
    
    threshold_up = env.up_thresholds.get(current_level_str, 3)
    threshold_down = 3 

    # تفكيك منطق التأكيد والتعلم بناءً على رغبتك
    if action_type == "known" or action_type == "understood_next":
        action = 0
        
        if action_type == "known":
            display_msg = f"🟢 Confirmed: I already know the word '{word_just_discussed}'"
            # يزداد العداد فقط إذا كان يعرفها مسبقاً دون وسيط الشرح العصبي
            target_up_streak += 1
            is_remediated = False
        else:
            display_msg = f"🎯 Understood: I have acquired the word '{word_just_discussed}' via neural session."
            # تجميد العداد وحفظ المكاسب السابقة دون تصفير أو زيادة لأنها كلمة مشروحة حديثاً
            is_remediated = True
        
        # فحص عتبة الترقية للأعلى (تتفعل فقط عبر الدفع بكلمات معروفة مسبقاً)
        if target_up_streak >= threshold_up:
            target_level_idx = min(target_level_idx + 1, len(env.levels) - 1)
            target_up_streak = 0
            new_level_str = env.levels[target_level_idx]
            tutor_text = f"🤖 **[Action: Level Up Approved]** Dynamic progression active! You have fully mastered **'{word_just_discussed}'**.\n\n🎉 **Promotion Event:** Advanced to CEFR **Level: {new_level_str}**!"
        else:
            if is_remediated:
                tutor_text = f"🤖 **[Action: Remediation Saved]** Progress registered without incrementing baseline streak (Preserved at {target_up_streak})."
            else:
                tutor_text = f"🤖 **[Action: Progress Registered]** Moving forward inside level {current_level_str}."

        # استدعاء البيئة لمزامنة المكافأة وسحب الكلمة الجديدة
        _, reward, _, _, info = env.step(action=action, student_response="yes")
        
        # فرض القيم المستقرة بعد انتهاء دالة خطوة البيئة
        env.up_streak = target_up_streak
        env.down_streak = target_down_streak
        env.current_level_idx = target_level_idx
        env.last_action_was_explain = False  
        
        next_word = env.current_word
        tutor_text += f"\n\n👉 Next Target Challenge: **Do you know the meaning of '{next_word}'?**"
        
        metrics_html = f"""
        <div style='padding: 12px; border-radius: 6px; background-color: rgba(76,175,80,0.06); border-left: 5px solid #4CAF50; font-family: monospace;'>
            <b style='color:#4CAF50; font-size:13px;'>✅ Action Synchronized & Counters Preserved:</b><br>
            • Reward Earned: <span style='color:green; font-weight:bold;'>+{reward:.4f}</span><br>
            • Promotion Streak: {env.up_streak}/{threshold_up}<br>
            • Demotion Progress: {env.down_streak}/{threshold_down}
        </div>
        """
        
        return (
            chat_history + [{"role": "user", "content": display_msg}, {"role": "assistant", "content": tutor_text}],
            next_word,
            env.levels[env.current_level_idx],
            f"{env.up_streak}/{env.up_thresholds.get(env.levels[env.current_level_idx], 3)}",
            f"{env.down_streak}/{threshold_down}",
            metrics_html,
            gr.update(value="✅ Yes, I already know this word", variant="success", interactive=True),
            gr.update(value="❓ No, I don't know it. Explain, please", variant="warning", interactive=True)
        )

    elif action_type == "unknown" or action_type == "deepen_explanation":
        action = 1
        display_msg = f"❓ Requesting neural analysis/deepening for: '{word_just_discussed}'"
        
        if not env.last_action_was_explain:
            target_down_streak += 1
            env.last_action_was_explain = True  
        
        if target_down_streak >= threshold_down:
            target_level_idx = max(target_level_idx - 1, 0)
            target_down_streak = 0
            new_level_str = env.levels[target_level_idx]
            tutor_text = f"🤖 **[Action: Level Adjusted Down]** Regulating difficulty to protect training curve.\n\n⚠️ **System Notice:** Adapted down to CEFR **Level: {new_level_str}** for optimal reinforcement."
        else:
            tutor_text = f"🤖 **[Action: Neural Explanation Model]** "

        _, reward, _, _, info = env.step(action=action, student_response="no")
        
        env.up_streak = target_up_streak
        env.down_streak = target_down_streak
        env.current_level_idx = target_level_idx
        env.current_word = word_just_discussed  
        
        next_word = env.current_word
        explanation = info.get('generated_text', 'No text generated.')
        
        if "🤖" in tutor_text and len(tutor_text) > 40:
            tutor_text += f"\n\nHere is the breakdown for **'{next_word}'**:\n{explanation}"
        else:
            tutor_text = f"🤖 **[Action: Neural Explanation Model]**\n\n{explanation}"
            
        tutor_text += f"\n\n" + f"─" * 25
        tutor_text += f"\n\n💡 *Now that you've reviewed the context, click '🎯 Understood! Next Word' to advance, or '🔄 Deepen Explanation' for more details.*"
        
        metrics_html = f"""
        <div style='padding: 12px; border-radius: 6px; background-color: rgba(33,150,243,0.05); border-left: 5px solid #2196F3; font-family: monospace; font-size:12px;'>
            <b style='color:#2196F3; font-size:13px;'>🎯 Active Reward Matrix Breakdown:</b><br>
            <table style='width:100%; margin-top:5px; border-collapse: collapse;'>
                <tr style='border-bottom: 1px solid #eee;'><td>Total Balanced Reward:</td><td style='text-align:right; font-weight:bold;'>{reward:.4f}</td></tr>
                <tr style='border-bottom: 1px solid #eee;'><td>Target Insertion Fit:</td><td style='text-align:right; color:green;'>+{info.get('target_word_reward', 0.0):.2f}</td></tr>
                <tr style='border-bottom: 1px solid #eee;'><td>Semantic Cosine Similarity:</td><td style='text-align:right; color:green;'>+{info.get('semantic_similarity_reward', 0.0):.2f}</td></tr>
                <tr style='border-bottom: 1px solid #eee;'><td>CEFR Readability Shift:</td><td style='text-align:right; color:green;'>+{info.get('cefr_readability_reward', 0.0):.2f}</td></tr>
                <tr><td>RoBERTa Linguistic Penalty:</td><td style='text-align:right; color:red;'>{info.get('linguistic_penalty', 0.0):.4f}</td></tr>
            </table>
        </div>
        """
        
        return (
            chat_history + [{"role": "user", "content": display_msg}, {"role": "assistant", "content": tutor_text}],
            next_word,
            env.levels[env.current_level_idx],
            f"{env.up_streak}/{env.up_thresholds.get(env.levels[env.current_level_idx], 3)}",
            f"{env.down_streak}/{threshold_down}",
            metrics_html,
            gr.update(value="🎯 Understood! Test me on next word", variant="success", interactive=True),
            gr.update(value="🔄 Deepen explanation with another example", variant="primary", interactive=True)
        )

def handle_left_button(btn_val, chat_history):
    if "already know" in btn_val:
        return process_rl_action("known", chat_history)
    else:
        return process_rl_action("understood_next", chat_history)

def handle_right_button(btn_val, chat_history):
    if "Explain, please" in btn_val:
        return process_rl_action("unknown", chat_history)
    else:
        return process_rl_action("deepen_explanation", chat_history)

# دالة إنهاء التطبيق وإغلاق الواجهة بشكل آمن وععرض ملخص نهائي
def terminate_session(chat_history):
    summary_text = f"""
    🛑 **[System Notice: Session Terminated Safely]**
    
    📊 **Final Performance Report:**
    • Last Active Level: {env.levels[env.current_level_idx]}
    • Final Promotion Streak: {env.up_streak}
    • Final Demotion Counter: {env.down_streak}
    
    The neural session state has been preserved. You can safely close your browser tab now or hit the reset button to start fresh.
    """
    return (
        chat_history + [{"role": "assistant", "content": summary_text}],
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False)
    )

def reset_entire_system():
    env.reset()
    env.down_streak = 0
    env.last_action_was_explain = False
    initial_word = env.current_word
    initial_level = env.levels[env.current_level_idx]
    
    welcome_text = [{"role": "assistant", "content": f"🎓 Welcome to your Advanced Action-Driven Dynamic RL Tutor.\nLet's start fresh: Do you know the meaning of the word **'{initial_word}'**?"}]
    return (
        welcome_text, 
        initial_word, 
        initial_level, 
        f"0/{env.up_thresholds.get(initial_level, 3)}", 
        "0/3", 
        "<div style='color:gray; padding:10px;'>Session restarted. Independent counters synchronized.</div>",
        gr.update(value="✅ Yes, I already know this word", variant="success", interactive=True),
        gr.update(value="❓ No, I don't know it. Explain, please", variant="warning", interactive=True)
    )

# ─── 3. بناء واجهة المستخدم المعدلة بأزرار الإغلاق والمنطق المحمي ───
with gr.Blocks(title="Adaptive Tutor RL Framework") as demo:
    
    gr.Markdown("""
    # 🎓 Pure RL Action-Driven Adaptive English Tutor
    ### Finite State Machine Interface & Anti-Reward Hacking Protection Panel
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            current_word_box = gr.Textbox(value=env.current_word, label="🎯 Active Target Word", interactive=False)
        with gr.Column(scale=1):
            current_level_box = gr.Textbox(value="A1", label="📊 Current CEFR Level", interactive=False)
        with gr.Column(scale=1):
            streak_box = gr.Textbox(value="0/3", label="🔥 Promotion Streak (Preserved)", interactive=False)
        with gr.Column(scale=1):
            demotion_box = gr.Textbox(value="0/3", label="⚠️ Demotion Progress Counter", interactive=False)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": f"🎓 Let's begin: Do you know the meaning of the word '{env.current_word}'?"}], 
                height=430, 
                show_label=False
            )
            
            with gr.Row():
                left_btn = gr.Button("✅ Yes, I already know this word", variant="success", scale=1)
                right_btn = gr.Button("❓ No, I don't know it. Explain, please", variant="warning", scale=1)
            
            with gr.Row():
                reset_btn = gr.Button("🔄 Reset Engine & Purge All Progress", variant="secondary", scale=1)
                terminate_btn = gr.Button("🛑 Terminate & Save Session", variant="stop", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ RL Engine Real-Time Optimization")
            metrics_display = gr.HTML(label="Live PPO State Data", value="<div style='color:gray; padding:10px;'>Click an action button to initialize environmental tracking...</div>")

    # ربط الأحداث بالمنطق المطور
    left_btn.click(
        handle_left_button,
        inputs=[left_btn, chatbot],
        outputs=[chatbot, current_word_box, current_level_box, streak_box, demotion_box, metrics_display, left_btn, right_btn]
    )
    
    right_btn.click(
        handle_right_button,
        inputs=[right_btn, chatbot],
        outputs=[chatbot, current_word_box, current_level_box, streak_box, demotion_box, metrics_display, left_btn, right_btn]
    )

    reset_btn.click(
        reset_entire_system, 
        outputs=[chatbot, current_word_box, current_level_box, streak_box, demotion_box, metrics_display, left_btn, right_btn]
    )

    terminate_btn.click(
        terminate_session,
        inputs=[chatbot],
        outputs=[chatbot, left_btn, right_btn, terminate_btn]
    )

    # ─── معلومات المبرمج في أسفل الواجهة ───
    gr.Markdown("""
    ---
    <p style="text-align: center; color: var(--body-text-color-subdued, #666); font-size: 0.9em; margin-top: 20px;">
        🛠️ <b>Interface Programmer:</b> Fadi Toumat | ✉️ <a href="mailto:Fadi.n.toumat@gmail.com">Fadi.n.toumat@gmail.com</a>
    </p>
    """)

if __name__ == "__main__":
    demo.launch(inline=True, share=True, theme=gr.themes.Soft())
