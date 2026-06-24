import re
import time
import pandas as pd
import requests

# ─────────────────────────────
# إعداد Ollama
# ─────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"


# ─────────────────────────────
# LLM Call
# ─────────────────────────────
def call_llm(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=180
            )

            if response.status_code == 200:
                return response.json()["response"].strip()

            print("❌ Response Error:", response.text)

        except Exception as e:
            print("❌ Exception:", e)

        time.sleep(3)

    return None


# ─────────────────────────────
# Step 1: استخراج متطلبات الوظائف
# ─────────────────────────────
def extract_arabjobs_requirements(arabjobs_csv):
    df = pd.read_csv(arabjobs_csv)

    requirements_by_category = {}

    for category in df['job_category'].dropna().unique():
        print(f"🔍 Processing category: {category}")

        cat_df       = df[df['job_category'] == category]
        descriptions = cat_df['description'].dropna().head(30).tolist()
        titles       = cat_df['job_title'].dropna().unique().tolist()
        combined     = "\n---\n".join(descriptions)

        prompt = f"""
بناءً على إعلانات الوظائف العربية دي:

المجال: {category}

المسميات:
{", ".join(titles[:15])}

الإعلانات:
{combined}

استخرج:
- أهم 10 مهارات تقنية
- أهم 5 مهارات شخصية
- المؤهلات المطلوبة
- متوسط سنوات الخبرة

اكتب بشكل مختصر ومنظم بالعربي.
"""

        result = call_llm(prompt)

        if result:
            requirements_by_category[category] = result
            print("   ✅ Done")

        time.sleep(2)

    return requirements_by_category


# ─────────────────────────────
# Step 2: ترجمة + اقتراحات
# ─────────────────────────────
def process_cv(cv_text, category_requirements):
    prompt = f"""
أنت خبير توظيف في السوق العربي.

متطلبات السوق:
{category_requirements}

السيرة الذاتية:
{cv_text}

المطلوب:
1- ترجمة السيرة الذاتية للعربية بشكل احترافي مع تغيير الأسماء والشركات والجامعات لعربية
2- كتابة 5 اقتراحات تحسين محددة بناءً على متطلبات السوق العربي

اكتب الناتج بالشكل ده بالظبط:

[السيرة الذاتية بالعربي]
(اكتب السيرة الذاتية كاملة هنا)

[اقتراحات التحسين]
1- (المشكلة: ... | الحل: ... | السبب: ...)
2- (المشكلة: ... | الحل: ... | السبب: ...)
3- (المشكلة: ... | الحل: ... | السبب: ...)
4- (المشكلة: ... | الحل: ... | السبب: ...)
5- (المشكلة: ... | الحل: ... | السبب: ...)
"""
    return call_llm(prompt)


# ─────────────────────────────
# فصل الـ output لكولمين
# ─────────────────────────────
def parse_output(raw_output):
    arabic_cv   = ""
    suggestions = ""

    if not raw_output:
        return arabic_cv, suggestions

    cv_match = re.search(
        r'\[السيرة الذاتية بالعربي\](.*?)(\[اقتراحات التحسين\]|$)',
        raw_output,
        re.DOTALL
    )
    if cv_match:
        arabic_cv = cv_match.group(1).strip()

    suggestions_match = re.search(
        r'\[اقتراحات التحسين\](.*?)$',
        raw_output,
        re.DOTALL
    )
    if suggestions_match:
        suggestions = suggestions_match.group(1).strip()

    return arabic_cv, suggestions


# ─────────────────────────────
# Main Pipeline
# ─────────────────────────────
def run_pipeline(kaggle_csv, arabjobs_csv, output_csv, start_from=0, limit=50):

    print("📂 تحميل البيانات...")
    kaggle_df = pd.read_csv(kaggle_csv)

    print("\n📊 استخراج متطلبات السوق...")
    requirements = extract_arabjobs_requirements(arabjobs_csv)
    print(f"\n✅ تم استخراج {len(requirements)} مجال\n")

    category_mapping = {
        'Information Technology' : 'تكنولوجيا المعلومات',
        'HR'                     : 'موارد بشرية',
        'Finance'                : 'مالية ومحاسبة',
        'Healthcare'             : 'رعاية صحية',
        'Engineering'            : 'هندسة',
        'Marketing'              : 'تسويق',
        'Sales'                  : 'مبيعات',
        'Education'              : 'تعليم',
        'Accountant'             : 'مالية ومحاسبة',
        'Business-Development'   : 'مبيعات',
        'Advocate'               : 'قانون ومحاماة',
        'Arts'                   : 'إعلام وتصميم',
        'Automobile'             : 'سيارات وميكانيك',
        'Aviation'               : 'هندسة',
        'Banking'                : 'مالية ومحاسبة',
        'Chef'                   : 'سياحة ومطاعم',
        'Construction'           : 'هندسة',
        'Consultant'             : 'إدارة وسكرتارية',
        'Designer'               : 'إعلام وتصميم',
        'Digital-Media'          : 'إعلام وتصميم',
        'Public-Relations'       : 'تسويق',
        'Teacher'                : 'تعليم',
    }

    results = []
    subset  = kaggle_df.iloc[start_from : start_from + limit]

    for i, (_, row) in enumerate(subset.iterrows()):
        actual_idx   = start_from + i + 1
        print(f"🔄 CV {actual_idx} | {row['Category']}")

        mapped_cat       = category_mapping.get(row['Category'], None)
        cat_requirements = requirements.get(mapped_cat, list(requirements.values())[0])

        # ── retry لو arabic_cv أو suggestions فاضيين ──
        arabic_cv    = ""
        suggestions  = ""
        max_attempts = 3

        for attempt in range(max_attempts):
            raw_output            = process_cv(row['Resume_str'], cat_requirements)
            arabic_cv, suggestions = parse_output(raw_output)

            if arabic_cv and suggestions:
                break

            print(f"   ⚠️ ناقص بيانات (محاولة {attempt+1}/{max_attempts}) — بيحاول تاني...")
            time.sleep(3)

        if not arabic_cv or not suggestions:
            print(f"   ❌ فشل بعد {max_attempts} محاولات — تخطى")
            continue

        results.append({
            "id"          : row['ID'],
            "category"    : row['Category'],
            "original_cv" : row['Resume_str'],
            "arabic_cv"   : arabic_cv,
            "suggestions" : suggestions
        })

        if (i + 1) % 25 == 0:
            pd.DataFrame(results).to_csv(
                f"backup_{actual_idx}.csv",
                index=False,
                encoding='utf-8-sig'
            )
            print(f"💾 Backup saved at CV {actual_idx}")

        time.sleep(2)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n🎉 Done! Saved {len(results)} CVs to {output_csv}")


# ─────────────────────────────
# Run
# ─────────────────────────────
if __name__ == "__main__":
    run_pipeline(
        kaggle_csv   = "Resume.csv",
        arabjobs_csv = "ArabJobs.csv",
        output_csv   = "arabic_cvs_output.csv",
        start_from   = 0,    # ← غيّرها لو عايز تكمل من CV معين
        limit        = 2484
    )