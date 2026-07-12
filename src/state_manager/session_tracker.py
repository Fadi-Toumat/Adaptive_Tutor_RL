import pandas as pd
import os
import random

class SessionTracker:
    """
    مدير الجلسات وحالة الحوار (State Manager) المطور:
    مسؤول حصرياً عن إدارة مستودع الكلمات وسحبها بشكل حتمي بناءً على المستوى
    الذي يمرره له المحرك الخارجي (ملف التفاعل الحي أو سكربت التدريب)،
    مما يضمن عزل بيئة الـ RL تماماً عن منطق إدارة مستويات الكلمات.
    """
    def __init__(self, csv_path="/content/drive/MyDrive/Adaptive_Tutor_RL/data/processed/repo_words.csv"):
        self.csv_path = csv_path
        # القائمة مطابقة تماماً لحدود مستودع بياناتك الفعلي (الحد الأقصى C1)
        self.levels = ['A1', 'A2', 'B1', 'B2', 'C1']
        self.df = self._load_vocabulary_csv()
        self.used_words = {level: set() for level in self.levels}

    def _load_vocabulary_csv(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"❌ ملف الكلمات غير موجود في المسار المحدد: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        
        # حماية إضافية: تنظيف النصوص وتوحيد حالة الأحرف لضمان دقة الفلترة
        df['level'] = df['level'].astype(str).str.upper().str.strip()
        df['word'] = df['word'].astype(str).str.strip()
        return df

    def get_next_word(self, target_level):
        """
        سحب كلمة بناءً على المستوى الحالي المطلق الذي يحدده مدير الحوار الخارجي.
        آمنة تماماً ضد التكرار الفوري وضد انهيار البيانات.
        """
        target_level = target_level.upper().strip()
        
        # حماية في حال تمرير مستوى خارج النطاق بالخطأ
        if target_level not in self.used_words:
            target_level = "A1"

        # تصفية البيانات حسب المستوى المطلوب
        level_df = self.df[self.df['level'] == target_level]
        
        # استبعاد الكلمات التي تم استخدامها مسبقاً في هذه الجلسة
        available_words = level_df[~level_df['word'].isin(self.used_words[target_level])]

        # ─── صمام أمان (Anti-Failure) ───
        # إذا استهلك المستخدم جميع كلمات هذا المستوى، نقوم بتصفير سجل الذاكرة له وإعادة التدوير
        if available_words.empty:
            self.used_words[target_level].clear()
            available_words = level_df

        if not available_words.empty:
            selected_word = random.choice(available_words['word'].values)
            self.used_words[target_level].add(selected_word)
        else:
            selected_word = "hello" # كلمة طوارئ في حال كان المستوى فارغاً تماماً في الـ CSV

        return selected_word

    def reset_session(self):
        """إعادة تهيئة الجلسة وتصفير سجلات الكلمات المستخدمة لبدء حوار جديد"""
        self.used_words = {level: set() for level in self.levels}
