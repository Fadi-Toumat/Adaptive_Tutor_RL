import json
import os
import re

def extract_with_teacher_boundary(jsonl_path):
    print("🔍 جاري الفحص الميكانيكي النقي (فحص نص الطالب قبل كلمة النداء Teacher/Sir)...")
    
    if not os.path.exists(jsonl_path):
        print("❌ خطأ: ملف البيانات غير موجود!")
        return

    doesnt_know_count = 0
    knows_count = 0
    total_processed = 0

    # كلمات النفي الصريحة المتفق عليها بدون أي حساب للطول
    neg_keywords = ["not sure", "no", "don't", "dont", "not really", "confusing", "confused"]

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                dialogue_text = data.get("dialogue", "")
                
                lines = dialogue_text.split('\n')
                student_response = None
                for l in lines:
                    clean_line = l.strip()
                    if clean_line.lower().startswith("student:"):
                        student_response = clean_line[len("student:"):].strip()
                        break
                
                if student_response:
                    total_processed += 1
                    response_lower = student_response.lower()
                    
                    # تطبيق فكرتك الذكية: قص النص عند أول ظهور لكلمة النداء لحماية الأمثلة اللاحقة
                    boundary_match = re.search(r"\b(teacher|sir|madam)\b", response_lower)
                    if boundary_match:
                        # أخذ النص الواقع قبل كلمة النداء فقط
                        target_text = response_lower[:boundary_match.start()].strip()
                    else:
                        # إذا لم توجد كلمة نداء، نفحص النص كاملاً
                        target_text = response_lower

                    # الفحص الميكانيكي التبادلي عن كلمات النفي داخل النطاق المستهدف فقط
                    # استخدام حدود الكلمات \b لمنع التداخل مع كلمات مثل kNOw أو NOun
                    has_negation = any(re.search(rf"\b{k}\b", target_text) for k in neg_keywords)
                    
                    if has_negation:
                        doesnt_know_count += 1
                    else:
                        knows_count += 1
            except Exception as e:
                continue

    print("\n==================================================")
    print("📊 [نتائج فحص حدود النداء والكلمات المفتاحية]")
    print("==================================================")
    print(f"🔹 إجمالي السجلات المقروءة: {total_processed}")
    print(f"🔹 عداد صنف (لا يعرف الطالب) ❌:  {doesnt_know_count}")
    print(f"🔹 عداد صنف (يعرف الطالب)    ✅:  {knows_count}")
    print("==================================================\n")

if __name__ == "__main__":
    dataset_path = "/content/drive/MyDrive/Adaptive_Tutor_RL/data/processed/final_merged_dataset.jsonl"
    extract_with_teacher_boundary(dataset_path)
