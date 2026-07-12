import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import re
from models.reward_engine import CompositeRewardEngine
from src.state_manager.session_tracker import SessionTracker

class AdaptiveTutorEnv(gym.Env):
    """
    بيئة التعلم التعزيزي المطورة والمحكومة برمجياً وتربوياً (Context-Aware RL Environment).
    إصدار متكامل تماماً مع محرك المكافآت الموزون لحماية التدريب من الـ Reward Hacking.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, qwen_model=None, qwen_tokenizer=None, roberta_model=None, roberta_tokenizer=None, device="cpu"):
        super(AdaptiveTutorEnv, self).__init__()

        self.qwen_model = qwen_model
        self.qwen_tokenizer = qwen_tokenizer
        self.roberta_model = roberta_model
        self.roberta_tokenizer = roberta_tokenizer
        self.device = device

        # استدعاء محرك المكافآت المركب
        self.reward_engine = CompositeRewardEngine()
        self.tracker = SessionTracker()

        self.levels = ['A1', 'A2', 'B1', 'B2', 'C1']
        self.current_level_idx = 0

        self.up_streak = 0
        self.down_streak = 0
        self.up_thresholds = {'A1': 3, 'A2': 3, 'B1': 4, 'B2': 4}
        self.down_threshold = 2

        # سحب الكلمة البدئية الأولى من المستودع الحقيقي
        self.current_word = self.tracker.get_next_word(self.levels[self.current_level_idx])

        self.action_space = spaces.Discrete(2)
        # مصفوفة الحالة (State Vector) المبدئية لحين دمج الـ Embeddings الكاملة لـ PPO
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(768,), dtype=np.float32)

    def _get_fallback_obs(self):
        return np.zeros((768,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_level_idx = 0
        self.up_streak = 0
        self.down_streak = 0
        self.tracker.reset_session()
        self.current_word = self.tracker.get_next_word(self.levels[self.current_level_idx])
        return self._get_fallback_obs(), {}

    def step(self, action, student_response="yes", external_generated_text=None):
        current_level = self.levels[self.current_level_idx]
        info = {}

        # تثبيت اسم الكلمة الحالية التي يدور حولها الموقف التعليمي الحالي
        info["word_discussed"] = self.current_word

        # ─── 1. التصنيف اللغوي المرن والمطور (Root Matching) ───
        response_clean = student_response.lower().strip()
        unknown_roots = ["no", "dont", "not", "detail", "explain", "help", "clue", "why"]

        if any(root in response_clean for root in unknown_roots):
            knowledge_status = "unknown"
        else:
            knowledge_status = "known"

        info["knowledge_status"] = knowledge_status

        # ─── 2. تحديث العدادات والمستويات الحتمية مشروطاً ───
        if knowledge_status == "known":
            self.up_streak += 1
            self.down_streak = 0
            if self.up_streak >= self.up_thresholds.get(current_level, 3):
                if self.current_level_idx < len(self.levels) - 1:
                    self.current_level_idx += 1
                    self.up_streak = 0
                    current_level = self.levels[self.current_level_idx]
        else:
            self.down_streak += 1
            self.up_streak = 0
            if self.down_streak >= self.down_threshold:
                if self.current_level_idx > 0:
                    self.current_level_idx -= 1
                    self.down_streak = 0
                    current_level = self.levels[self.current_level_idx]

        info["up_streak"] = self.up_streak
        info["down_streak"] = self.down_streak
        info["current_level"] = current_level

        # ─── 3. مصفوفة الثواب والعقاب التكيفية والمحكومة بالسياق ───
        reward = 0.0

        if knowledge_status == "unknown":
            # الطالب لا يعرف الكلمة ويحتاج إلى شرح
            if action == 1:
                # صياغة البرومبت المحسن الصارم الموجه للمولد
                prompt = (
                    f"<|im_start|>system\n"
                    f"You are a strict CEFR English Tutor. Explain the target word directly and briefly. Provide ONLY 1 or 2 examples max. Do NOT use introductory sentences like 'Sure' or 'Okay'. No chitchat.\n"
                    f"<|im_end|>\n"
                    f"<|im_start|>user\n"
                    f"Explain the word '{self.current_word}' simply with examples for level {current_level}.\n"
                    f"<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )

                if self.qwen_model and self.qwen_tokenizer:
                    inputs = self.qwen_tokenizer(prompt, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.qwen_model.generate(
                            **inputs,
                            max_new_tokens=120,
                            temperature=0.3,
                            do_sample=True,
                            pad_token_id=self.qwen_tokenizer.eos_token_id
                        )
                    generated_text = self.qwen_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                else:
                    generated_text = external_generated_text or f"Explanation of '{self.current_word}'."

                if self.roberta_model and self.roberta_tokenizer:
                    # تعديل حيوي: تمرير الـ student_response والالتزام بتوقيع الدالة الجديد
                    raw_reward, breakdown = self.reward_engine.compute_pure_rewards(
                        original_simple_text=student_response,
                        generated_text=generated_text,
                        target_word=self.current_word,
                        roberta_model=self.roberta_model,
                        roberta_tokenizer=self.roberta_tokenizer,
                        device=self.device
                    )
                    # الاعتماد على الجائزة المركبة الإجمالية والنظيفة من المحرك مباشرة
                    reward = raw_reward
                else:
                    reward, breakdown = 0.5, {}

                info.update(breakdown)
                info["generated_text"] = generated_text
                info["action_taken"] = "Deep Explain"

            else:
                # رفع قيمة عقوبة الكسل التربوي لإجبار الوكيل على اتخاذ القرار الصحيح
                reward = -0.7500
                info["action_taken"] = "Proceed/Skip (Forced Lazy Error)"
                info["generated_text"] = "Skipped despite student needing explanation."

        else:
            # الطالب يعرف الكلمة بالفعل (known) ولا يحتاج هدر وقته
            if action == 0:
                reward = 0.2000
                info["action_taken"] = "Proceed/Skip"
                info["generated_text"] = "Perfect! Let's move to the next word."
            else:
                reward = -0.3000
                info["action_taken"] = "Deep Explain (Annoying Over-Explanation Error)"
                info["generated_text"] = "Generated explanation unnecessarily for a known word."

        # ─── 4. ضمان انتقال قطار الكلمات دائماً للأمام كسرًا للتكرار ───
        self.current_word = self.tracker.get_next_word(current_level)

        terminated = False
        truncated = False

        return self._get_fallback_obs(), reward, terminated, truncated, info
