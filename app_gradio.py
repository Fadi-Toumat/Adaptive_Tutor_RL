import sys
import os
import re
import torch
import gradio as gr
import threading
import json
import signal  # استيراد مكتبة الإشارات للقتل الفوري للعملية

# تأمين مسارات النظام لضمان استيراد الحزم دون مشاكل
sys.path.append("/content/drive/MyDrive/Adaptive_Tutor_RL")

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from src.rl.env import AdaptiveTutorEnv

# ─── 1. التحديد التلقائي والديناميكي لمسار تقرير الجلسة (متوافق مع GitHub) ───
# يقوم بايثون هنا بمعرفة مجلد المشروع النشط تلقائياً أينما تم تشغيل الملف
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
REPORT_PATH = os.path.join(BASE_DIR, "last_session_report.json")

print(f"📂 [System Path Discovery] Dynamic session logs will be routed to: {REPORT_PATH}")

# سجل لتتبع الكلمات والقرارات في الجلسة النشطة الحالية
active_session_log = []

# ─── 2. التحقق من مسارات النماذج المحلية ───
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

# ─── 3. هندسة إدارة الحالات التكيفية للأزرار والعدادات المستقلة ───
def process_rl_action(action_type, chat_history):
    global active_session_log
    word_just_discussed = env.current_word
    current_level_str = env.levels[env.current_level_idx]
    
    target_up_streak = env.up_streak
    target_down_streak = env.down_streak
    target_level_idx = env.current_level_idx
    
    threshold_up = env.up_thresholds.get(current_level_str, 3)
    threshold_down = 3 

    if action_type == "known" or action_type == "understood_next":
        action = 0
        
        if action_type == "known":
            display_msg = f"🟢 Confirmed: I already know the word '{word_just_discussed}'"
            target_up_streak += 1
            is_remediated = False
            active_session_log.append({
                "word": word_just_discussed,
                "status": "Already Known (Mastered)",
                "level": current_level_str
            })
        else:
            display_msg = f"🎯 Understood: I have acquired the word '{word_just_discussed}' via neural session."
            is_remediated = True
            active_session_log.append({
                "word": word_just_discussed,
                "status": "Acquired via Explanation (Mastered)",
                "level": current_level_str
            })
        
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

        _, reward, _, _, info = env.step(action=action, student_response="yes")
        
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
        
        already_logged_explanation = any(item["word"] == word_just_discussed and "Explanation" in item["status"] for item in active_session_log)
        if not already_logged_explanation:
            active_session_log.append({
                "word": word_just_discussed,
                "status": "Under Explanation (Needs Practice)",
                "level": current_level_str
            })
            
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

# دالة إنهاء التطبيق وحفظ الحالة الفورية إلى ملف التقرير التلقائي
def terminate_session(chat_history):
    global active_session_log
    
    mastered_count = sum(1 for item in active_session_log if "Mastered" in item["status"])
    explained_count = sum(1 for item in active_session_log if "Explanation" in item["status"])
    
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
    
    ⚠️ **Note:** To completely stop the system, release CUDA memory, and kill the backend server process, click the newly appeared button **"⚠️ Exit & Kill Server Process"** below.
    """
    return (
        chat_history + [{"role": "assistant", "content": summary_text}],
        gr.update(interactive=False),  # left_btn
        gr.update(interactive=False),  # right_btn
        gr.update(visible=False),      # terminate_btn
        gr.update(visible=True),       # shutdown_btn
        gr.update(interactive=False)   # history_btn
    )

# دالة لقراءة التقرير من الملف التلقائي وعرضه بشكل منسق في الشات
def load_previous_report(chat_history):
    if not os.path.exists(REPORT_PATH):
        return chat_history + [{"role": "assistant", "content": "⚠️ **[System Notice]** No previous learning report found in the current workspace directory. Finish a session first to generate one."}]
    
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

# إغلاق فوري وعنيف للنظام دون أي تأخير عن طريق النواة مباشرة
def hard_shutdown_system(chat_history):
    shutdown_msg = "🔌 **[System Shutting Down]** Killing process instantly via SIGKILL. Session closed!"
    
    # مهلة 0.1 ثانية فقط لكي تصل رسالة الوداع للمتصفح، ثم نطلق إشارة القتل الفوري
    threading.Timer(0.1, lambda: os.kill(os.getpid(), signal.SIGKILL)).start()
    
    return chat_history + [{"role": "assistant", "content": shutdown_msg}]

def reset_entire_system():
    global active_session_log
    active_session_log = [] 
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
