import re
import os
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
المسميات: {", ".join(titles[:15])}
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
        raw_output, re.DOTALL
    )
    if cv_match:
        arabic_cv = cv_match.group(1).strip()

    suggestions_match = re.search(
        r'\[اقتراحات التحسين\](.*?)$',
        raw_output, re.DOTALL
    )
    if suggestions_match:
        suggestions = suggestions_match.group(1).strip()

    return arabic_cv, suggestions


# ─────────────────────────────
# Process single CV with retry
# ─────────────────────────────
def process_with_retry(cv_text, cat_requirements, max_attempts=3):
    for attempt in range(max_attempts):
        raw_output             = process_cv(cv_text, cat_requirements)
        arabic_cv, suggestions = parse_output(raw_output)

        if arabic_cv and suggestions:
            return arabic_cv, suggestions

        print(f"   ⚠️ ناقص بيانات (محاولة {attempt+1}/{max_attempts}) — بيحاول تاني...")
        time.sleep(3)

    print(f"   ❌ فشل بعد {max_attempts} محاولات — تخطى")
    return None, None


# ─────────────────────────────
# Main Pipeline
# ─────────────────────────────
def run_pipeline(kaggle_csv, arabjobs_csv, output_csv):

    print("📂 تحميل البيانات...")
    kaggle_df  = pd.read_csv(kaggle_csv)

    # ── تحميل الـ output القديم لو موجود ──
    if os.path.exists(output_csv):
        output_df = pd.read_csv(output_csv, encoding='utf-8-sig')
        print(f"📋 الـ output القديم فيه {len(output_df)} CV")
    else:
        output_df = pd.DataFrame(columns=['id', 'category', 'original_cv', 'arabic_cv', 'suggestions'])
        print("📋 مفيش output قديم — هيبدأ من الأول")

    # ── بناء lookup سريع بالـ ID من الكاجل ──
    kaggle_lookup = kaggle_df.set_index('ID')

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

    results    = output_df.to_dict('records')
    done_ids   = set(output_df['id'].astype(str).tolist())

    # ── Phase 1: فكس الناقص في الـ output ────────────
    incomplete = output_df[
        output_df['arabic_cv'].isna()   | (output_df['arabic_cv']   == '') |
        output_df['suggestions'].isna() | (output_df['suggestions'] == '')
    ]
    print(f"🔧 Phase 1: فيه {len(incomplete)} CV ناقصين في الـ output\n")

    for idx, row in incomplete.iterrows():
        cv_id = int(row['id'])
        print(f"   🔄 فكس ID: {cv_id} | {row['category']}")

        if cv_id not in kaggle_lookup.index:
            print(f"   ⚠️ مش موجود في الأوريجينال — تخطى")
            continue

        cv_text          = kaggle_lookup.loc[cv_id, 'Resume_str']
        mapped_cat       = category_mapping.get(row['category'], None)
        cat_requirements = requirements.get(mapped_cat, list(requirements.values())[0])

        arabic_cv, suggestions = process_with_retry(cv_text, cat_requirements)

        if arabic_cv and suggestions:
            # حدّث الـ results list مباشرة
            for r in results:
                if str(r['id']) == str(cv_id):
                    r['arabic_cv']   = arabic_cv
                    r['suggestions'] = suggestions
                    break
            print(f"   ✅ تم الفكس")

        time.sleep(2)

    print(f"\n✅ Phase 1 خلص!\n")

    # ── Phase 2: بروسيس اللي مش في الـ output خالص ──
    pending_df    = kaggle_df[~kaggle_df['ID'].astype(str).isin(done_ids)]
    total_pending = len(pending_df)
    print(f"🚀 Phase 2: فاضل {total_pending} CV جديد\n")

    for i, (_, row) in enumerate(pending_df.iterrows()):
        print(f"🔄 CV {i+1}/{total_pending} | ID: {row['ID']} | {row['Category']}")

        mapped_cat       = category_mapping.get(row['Category'], None)
        cat_requirements = requirements.get(mapped_cat, list(requirements.values())[0])

        arabic_cv, suggestions = process_with_retry(row['Resume_str'], cat_requirements)

        if not arabic_cv or not suggestions:
            continue

        results.append({
            "id"          : row['ID'],
            "category"    : row['Category'],
            "original_cv" : row['Resume_str'],
            "arabic_cv"   : arabic_cv,
            "suggestions" : suggestions
        })

        if (i + 1) % 25 == 0:
            pd.DataFrame(results).to_csv(output_csv, index=False, encoding='utf-8-sig')
            pd.DataFrame(results).to_csv(f"backup_{i+1}.csv", index=False, encoding='utf-8-sig')
            print(f"💾 Backup saved at CV {i+1}")

        time.sleep(2)

    pd.DataFrame(results).to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n🎉 Done! Total {len(results)} CVs in {output_csv}")


# ─────────────────────────────
# Run
# ─────────────────────────────
if __name__ == "__main__":
    run_pipeline(
        kaggle_csv   = "Resume.csv",
        arabjobs_csv = "ArabJobs.csv",
        output_csv   = "arabic_cvs_output.csv",
    )