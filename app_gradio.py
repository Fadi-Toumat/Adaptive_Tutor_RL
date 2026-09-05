import sys
import os
import re
import torch
import gradio as gr
import threading
import json
import signal

# تأمين مسارات النظام لضمان استيراد الحزم دون مشاكل
sys.path.append("/content/drive/MyDrive/Adaptive_Tutor_RL")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from src.rl.env import AdaptiveTutorEnv

# ─── 1. التحديد التلقائي والديناميكي لمسار تقرير الجلسة ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
REPORT_PATH = os.path.join(BASE_DIR, "last_session_report.json")

print(f"📂 [System Path Discovery] Dynamic session logs will be routed to: {REPORT_PATH}")

active_session_log = []

# ─── 2. التحقق من مسارات النماذج المحلية ───
qwen_path = "/content/drive/MyDrive/Adaptive_Tutor_RL/models/ppo_tutor_checkpoints_fixed_lr/qwen_ppo_fixed_lr_final"
roberta_path = "/content/drive/MyDrive/Adaptive_Tutor_RL/models/discriminator_roberta_aligned"

if not os.path.exists(qwen_path):
    raise FileNotFoundError(f"❌ خطأ: المسار غير موجود: '{qwen_path}'")

print("⏳ [Gradio Core] Paths verified successfully. Loading neural components...")
device = "cuda" if torch.cuda.is_available() else "cpu"

qwen_tok = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", device_map="auto" if device == "cuda" else None)
qwen_model = PeftModel.from_pretrained(base_model, qwen_path).eval()
if device == "cpu":
    qwen_model = qwen_model.to(device)

roberta_tok = AutoTokenizer.from_pretrained(roberta_path, local_files_only=True)
roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_path, local_files_only=True).to(device).eval()

env = AdaptiveTutorEnv(
    qwen_model=qwen_model,
    qwen_tokenizer=qwen_tok,
    roberta_model=roberta_model,
    roberta_tokenizer=roberta_tok,
    device=device
)
env.reset()

env.levels = ["A1", "A2", "B1", "B2", "C1"]

if not hasattr(env, 'down_streak'):
    env.down_streak = 0
if not hasattr(env, 'last_action_was_explain'):
    env.last_action_was_explain = False

# ─── 3. هندسة منطق العدادات المعدل بدقة ───
def process_rl_action(action_type, chat_history):
    global active_session_log
    word_just_discussed = env.current_word
    current_level_str = env.levels[env.current_level_idx]

    target_up_streak = env.up_streak
    target_down_streak = env.down_streak
    target_level_idx = env.current_level_idx

    # تحديد عتبة الترقية بحسب المستوى
    if current_level_str in ["A1", "A2"]:
        threshold_up = 3
    elif current_level_str == "B1":
        threshold_up = 4
    else:
        threshold_up = 5

    threshold_down = 3  # عتبة التخفيض الثابتة

    # 🟢 الحالة الأولى: المستخدم يعرف الكلمة من أول مرة دون شرح (إتقان حقيقي)
    if action_type == "known":
        action = 0
        display_msg = f"🟢 Confirmed: I already know the word '{word_just_discussed}'"
        
        target_up_streak += 1
        target_down_streak = 0  # تصفير عداد التخفيض فقط عند المعرفة المباشرة
        
        active_session_log.append({
            "word": word_just_discussed,
            "status": "Already Known (Mastered)",
            "level": current_level_str
        })

        # فحص الترقية
        if target_up_streak >= threshold_up:
            if target_level_idx < len(env.levels) - 1:
                target_level_idx += 1
                target_up_streak = 0
                target_down_streak = 0
                new_level_str = env.levels[target_level_idx]
                tutor_text = f"🤖 **[Action: Level Up Approved]** Dynamic progression active! You have fully mastered **'{word_just_discussed}'**.\n\n🎉 **Promotion Event:** Advanced to CEFR **Level: {new_level_str}**!"
            else:
                target_up_streak = 0
                tutor_text = f"🤖 **[Action: Max Level Reached]** You have mastered **'{word_just_discussed}'**.\n\n🌟 You are currently at the highest level (**C1**)."
        else:
            tutor_text = f"🤖 **[Action: Progress Registered]** Moving forward inside level {current_level_str} ({target_up_streak}/{threshold_up})."

        _, reward, _, _, info = env.step(action=action, student_response="yes")

        env.up_streak = target_up_streak
        env.down_streak = target_down_streak
        env.current_level_idx = target_level_idx
        env.last_action_was_explain = False

        next_word = env.current_word
        tutor_text += f"\n\n👉 Next Target Challenge: **Do you know the meaning of '{next_word}'?**"

    # 🟡 الحالة الثانية: المستخدم يقر بأنه فهم الكلمة "بعد الشرح" (انتقال للكلمة التالية دون ترقية ودون تصفير التخفيض)
    elif action_type == "understood_next":
        action = 0
        display_msg = f"🎯 Understood: I have acquired the word '{word_just_discussed}' via explanation."
        
        # الكلمة مشروحة: لا ترفع عداد الترقية، ولا تُصفّر عداد التخفيض التراكمي
        active_session_log.append({
            "word": word_just_discussed,
            "status": "Acquired via Explanation",
            "level": current_level_str
        })

        tutor_text = f"🤖 **[Action: Remediation Saved]** Explanation acknowledged for **'{word_just_discussed}'**. Progress saved without streak promotion."

        _, reward, _, _, info = env.step(action=action, student_response="yes")

        env.up_streak = target_up_streak
        env.down_streak = target_down_streak
        env.current_level_idx = target_level_idx
        env.last_action_was_explain = False

        # جلب كلمة جديدة من البيئة
        next_word = env.current_word
        tutor_text += f"\n\n👉 Next Target Challenge: **Do you know the meaning of '{next_word}'?**"

    # 🔴 الحالة الثالثة: لا يعرف الكلمة أو يطلب تعميق الشرح
    elif action_type in ["unknown", "deepen_explanation"]:
        action = 1
        display_msg = f"❓ Requesting neural analysis/deepening for: '{word_just_discussed}'"

        # يُحسب خطأ جديد فقط إذا لم نكن في نفس الكلمة المشروحة سابقاً
        if not env.last_action_was_explain:
            target_down_streak += 1
            target_up_streak = 0  # تصفير الترقية عند الخطأ
            env.last_action_was_explain = True

        already_logged = any(item["word"] == word_just_discussed and "Explanation" in item["status"] for item in active_session_log)
        if not already_logged:
            active_session_log.append({
                "word": word_just_discussed,
                "status": "Under Explanation (Needs Practice)",
                "level": current_level_str
            })

        # فحص شرط التخفيض (عند الوصول لـ 3 أخطاء تراكمية)
        if target_down_streak >= threshold_down:
            if target_level_idx > 0:
                target_level_idx -= 1
                target_down_streak = 0
                target_up_streak = 0
                new_level_str = env.levels[target_level_idx]
                
                # تحديث مستوى البيئة وإعادة ضبط سحب الكلمات من المستوى الأدنى
                env.current_level_idx = target_level_idx
                env.reset()
                
                tutor_text = f"🤖 **[Action: Level Adjust Down]** Regulating difficulty due to 3 consecutive misses.\n\n⚠️ **System Notice:** Adapted down to CEFR **Level: {new_level_str}** for optimal reinforcement."
            else:
                target_down_streak = 0
                tutor_text = f"🤖 **[Action: Minimum Level Reached]** You are at base Level A1. Continuing reinforcement."
        else:
            tutor_text = f"🤖 **[Action: Neural Explanation Model]** "

        _, reward, _, _, info = env.step(action=action, student_response="no")

        env.up_streak = target_up_streak
        env.down_streak = target_down_streak
        env.current_level_idx = target_level_idx

        # في حالة طلب الشرح أو تعميقه نُبقي الكلمة نفسها
        if target_down_streak > 0 or not env.last_action_was_explain:
            env.current_word = word_just_discussed

        next_word = env.current_word
        explanation = info.get('generated_text', 'No text generated.')

        if "🤖" in tutor_text and len(tutor_text) > 40:
            tutor_text += f"\n\nHere is the breakdown for **'{next_word}'**:\n{explanation}"
        else:
            tutor_text = f"🤖 **[Action: Neural Explanation Model]**\n\n{explanation}"

        tutor_text += f"\n\n" + f"─" * 25
        tutor_text += f"\n\n💡 *Now that you've reviewed the context, click '🎯 Understood! Next Word' to advance, or '🔄 Deepen Explanation' for more details.*"

    # تحديث القيم للعرض في الواجهة
    new_curr_level = env.levels[env.current_level_idx]
    new_thresh_up = 3 if new_curr_level in ["A1", "A2"] else (4 if new_curr_level == "B1" else 5)

    metrics_html = f"""
    <div style='padding: 12px; border-radius: 6px; background-color: rgba(33,150,243,0.05); border-left: 5px solid #2196F3; font-family: monospace; font-size:12px;'>
        <b style='color:#2196F3; font-size:13px;'>📊 Active State Tracker:</b><br>
        • Active Level: <b>{new_curr_level}</b><br>
        • Reward Earned: <span style='color:green; font-weight:bold;'>+{reward:.4f}</span><br>
        • Promotion Streak: {env.up_streak}/{new_thresh_up}<br>
        • Demotion Progress: <span style='color:red; font-weight:bold;'>{env.down_streak}/{threshold_down}</span>
    </div>
    """

    if action_type in ["known", "understood_next"]:
        btn_left_update = gr.update(value="✅ Yes, I already know this word", variant="success", interactive=True)
        btn_right_update = gr.update(value="❓ No, I don't know it. Explain, please", variant="warning", interactive=True)
    else:
        btn_left_update = gr.update(value="🎯 Understood! Test me on next word", variant="success", interactive=True)
        btn_right_update = gr.update(value="🔄 Deepen explanation with another example", variant="primary", interactive=True)

    return (
        chat_history + [{"role": "user", "content": display_msg}, {"role": "assistant", "content": tutor_text}],
        env.current_word,
        env.levels[env.current_level_idx],
        f"{env.up_streak}/{new_thresh_up}",
        f"{env.down_streak}/{threshold_down}",
        metrics_html,
        btn_left_update,
        btn_right_update
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

def terminate_session(chat_history):
    global active_session_log

    mastered_count = sum(1 for item in active_session_log if "Mastered" in item["status"])
    explained_count = sum(1 for item in active_session_log if "Explanation" in item["status"] or "Acquired" in item["status"])

    report_payload = {
        "last_active_level": env.levels[env.current_level_idx],
        "final_up_streak": env.up_streak,
        "final_down_streak": env.down_streak,
        "total_words_encountered": len(active_session_log),
        "total_words_mastered": mastered_count,
        "total_words_explained": explained_count,
        "words_log": active_session_log
    }

    try:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, ensure_ascii=False, indent=4)
        save_status = "📂 **[Automated Storage: Saved to Active Directory]**"
    except Exception as e:
        save_status = f"⚠️ **[Storage Error: Could not save report]** ({str(e)})"

    summary_text = f"""
    🛑 **[System Notice: Session Terminated Safely]**

    📊 **Final Performance Report:**
    • Last Active Level: {env.levels[env.current_level_idx]}
    • Final Promotion Streak: {env.up_streak}
    • Final Demotion Counter: {env.down_streak}
    • Total Words Mastered during Session: {mastered_count}
    • Total Words Explained (To review later): {explained_count}

    {save_status}

    ⚠️ **Note:** To completely stop the system, click **"⚠️ Exit & Kill Server Process"** below.
    """
    return (
        chat_history + [{"role": "assistant", "content": summary_text}],
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(interactive=False)
    )

def load_previous_report(chat_history):
    if not os.path.exists(REPORT_PATH):
        return chat_history + [{"role": "assistant", "content": "⚠️ **[System Notice]** No previous learning report found in the current workspace directory."}]

    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        words_log_list = data.get("words_log", [])
        if words_log_list:
            formatted_words = "\n".join([f"• <b>{item['word']}</b> ({item['level']}) - <i>{item['status']}</i>" for item in words_log_list])
        else:
            formatted_words = "*No specific words logged in the previous session history.*"

        history_report = f"""
        📋 **[Loaded Historical Session Report]**

        📈 **Previous Progress Metrics:**
        • **CEFR Final Level reached:** {data.get('last_active_level', 'N/A')}
        • **Streak preserved:** {data.get('final_up_streak', 0)}
        • **Demotion counter state:** {data.get('final_down_streak', 0)}
        • **Total words handled:** {data.get('total_words_encountered', 0)}
        • **Mastered words count:** {data.get('total_words_mastered', 0)}
        • **Words needed review (Explained):** {data.get('total_words_explained', 0)}

        🛣️ **Student Learning Path Logs:**
        {formatted_words}
        """
        return chat_history + [{"role": "assistant", "content": history_report}]
    except Exception as e:
        return chat_history + [{"role": "assistant", "content": f"❌ **Error reading session report:** {str(e)}"}]

def hard_shutdown_system(chat_history):
    shutdown_msg = "🔌 **[System Shutting Down]** Killing process instantly via SIGKILL. Session closed!"
    threading.Timer(0.1, lambda: os.kill(os.getpid(), signal.SIGKILL)).start()
    return chat_history + [{"role": "assistant", "content": shutdown_msg}]

def reset_entire_system():
    global active_session_log
    active_session_log = []
    env.reset()
    env.up_streak = 0
    env.down_streak = 0
    env.last_action_was_explain = False
    initial_word = env.current_word
    initial_level = env.levels[env.current_level_idx]

    welcome_text = [{"role": "assistant", "content": f"🎓 Welcome to your Advanced Action-Driven Dynamic RL Tutor.\nLet's start fresh: Do you know the meaning of the word **'{initial_word}'**?"}]
    return (
        welcome_text,
        initial_word,
        initial_level,
        "0/3",
        "0/3",
        "<div style='color:gray; padding:10px;'>Session restarted. Independent counters synchronized.</div>",
        gr.update(value="✅ Yes, I already know this word", variant="success", interactive=True),
        gr.update(value="❓ No, I don't know it. Explain, please", variant="warning", interactive=True),
        gr.update(visible=True, interactive=True),
        gr.update(visible=False),
        gr.update(interactive=True)
    )

# ─── 4. بناء واجهة المستخدم ───
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
                history_btn = gr.Button("📋 Load Previous Session Report", variant="primary", scale=1)
                terminate_btn = gr.Button("🛑 Terminate & Save Session", variant="stop", scale=1)
                shutdown_btn = gr.Button("⚠️ Exit & Kill Server Process", variant="stop", scale=1, visible=False)

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ RL Engine Real-Time Optimization")
            metrics_display = gr.HTML(label="Live PPO State Data", value="<div style='color:gray; padding:10px;'>Click an action button to initialize environmental tracking...</div>")

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
        outputs=[chatbot, current_word_box, current_level_box, streak_box, demotion_box, metrics_display, left_btn, right_btn, terminate_btn, shutdown_btn, history_btn]
    )

    history_btn.click(
        load_previous_report,
        inputs=[chatbot],
        outputs=[chatbot]
    )

    terminate_btn.click(
        terminate_session,
        inputs=[chatbot],
        outputs=[chatbot, left_btn, right_btn, terminate_btn, shutdown_btn, history_btn]
    )

    shutdown_btn.click(
        hard_shutdown_system,
        inputs=[chatbot],
        outputs=[chatbot]
    )

    gr.Markdown("""
    ---
    <p style="text-align: center; color: var(--body-text-color-subdued, #666); font-size: 0.9em; margin-top: 20px;">
        🧠 <b>AI System Architect & Lead RL Engineer:</b> Fadi Toumat | ✉️ <a href="mailto:Fadi.n.toumat@gmail.com">Fadi.n.toumat@gmail.com</a>
    </p>
    """)

if __name__ == "__main__":
    demo.launch(inline=True, share=True, theme=gr.themes.Soft())
