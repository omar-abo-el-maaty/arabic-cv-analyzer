import os
import re
from groq import Groq
import pandas as pd
import time

# Reads the key from an environment variable — never hardcode API keys in
# source code. Set it before running:
#   export GROQ_API_KEY="your-key-here"        (Linux/macOS)
#   $env:GROQ_API_KEY = "your-key-here"          (Windows PowerShell)
client = Groq(api_key=os.environ["GROQ_API_KEY"])


# ── استخراج وقت الانتظار الفعلي من الـ error ─────────
def parse_retry_after(error_message):
    match = re.search(r'Please try again in (\d+)m([\d.]+)s', str(error_message))
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return (minutes * 60) + seconds + 5  # +5 ثانية احتياطي
    return 60  # default لو مش لاقي


# ── LLM Call مع Smart Retry ───────────────────────────
def call_llm(prompt, retries=5):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = parse_retry_after(str(e))
            print(f"⚠️  هستنى {wait:.0f} ثانية ({wait / 60:.1f} دقيقة)... (محاولة {attempt + 1}/{retries})")
            time.sleep(wait)
    print("❌ فشل بعد كل المحاولات، هتخطى الـ CV ده")
    return None


# ── Step 1: استخراج متطلبات ArabJobs ─────────────────
def extract_arabjobs_requirements(arabjobs_csv):
    df = pd.read_csv(arabjobs_csv)

    requirements_by_category = {}

    for category in df['job_category'].dropna().unique():
        cat_df = df[df['job_category'] == category]
        descriptions = cat_df['description'].dropna().head(50).tolist()
        titles = cat_df['job_title'].dropna().unique().tolist()
        combined = "\n---\n".join(descriptions)

        prompt = f"""
 بناءً على إعلانات الوظائف العربية دي في مجال "{category}":

 المسميات الوظيفية الموجودة: {", ".join(titles[:20])}

 الإعلانات:
 {combined}

 استخرج:
 1. أهم 10 مهارات تقنية مطلوبة
 2. أهم 5 مهارات شخصية مطلوبة
 3. الشهادات والمؤهلات المطلوبة
 4. متوسط الخبرة المطلوبة (سنوات)

 الناتج: نص منظم ومختصر بالعربي فقط.
 """
        result = call_llm(prompt)
        if result:
            requirements_by_category[category] = result
            print(f"   ✅ {category}")
        time.sleep(2)

    return requirements_by_category


# ── Step 2: ترجمة الـ CV ──────────────────────────────
def translate_cv(cv_text):
    prompt = f"""
 حوّل السيرة الذاتية الإنجليزية دي لسيرة ذاتية عربية واقعية.

 القواعد:
 - ترجمة مهنية طبيعية باللغة العربية الفصحى
 - استبدل الأسماء الأجنبية بأسماء عربية
 - استبدل الشركات الأمريكية بشركات عربية معروفة (STC, Aramco, Vodafone مصر, بنك مصر...)
 - استبدل الجامعات الأجنبية بجامعات عربية (جامعة القاهرة, KAUST, AUB, جامعة الملك فهد...)
 - المهارات التقنية تفضل إنجليزي (Python, SQL, Excel...)
 - الناتج: نص سيرة ذاتية عربية فقط بدون أي تعليق إضافي

 السيرة الذاتية:
 {cv_text}
 """
    return call_llm(prompt)


# ── Step 3: توليد اقتراحات التحسين ───────────────────
def generate_suggestions(arabic_cv, category_requirements):
    prompt = f"""
 أنت خبير في تطوير السير الذاتية للسوق العربي.

 متطلبات سوق العمل العربي في هذا المجال:
 {category_requirements}

 السيرة الذاتية:
 {arabic_cv}

 المطلوب: اكتب 5 اقتراحات تحسين محددة وعملية لهذه السيرة الذاتية.

 كل اقتراح يوضح:
 - المشكلة الموجودة في الـ CV
 - الحل المقترح بالتحديد
 - ليه مهم لسوق العمل العربي

 الناتج: قائمة مرقمة بالعربي فقط.
 """
    return call_llm(prompt)


# ── Main Pipeline ─────────────────────────────────────
def run_pipeline(kaggle_csv, arabjobs_csv, output_csv, limit=100):
    print("📂 جاري تحميل الداتا...")
    kaggle_df = pd.read_csv(kaggle_csv)

    print("\n📋 جاري استخراج متطلبات سوق العمل من ArabJobs...")
    requirements = extract_arabjobs_requirements(arabjobs_csv)
    print(f"✅ تم استخراج متطلبات {len(requirements)} مجال\n")

    category_mapping = {
        'Information Technology': 'تكنولوجيا المعلومات',
        'HR': 'موارد بشرية',
        'Finance': 'مالية ومحاسبة',
        'Healthcare': 'رعاية صحية',
        'Engineering': 'هندسة',
        'Marketing': 'تسويق',
        'Sales': 'مبيعات',
        'Education': 'تعليم',
    }

    results = []

    for i, row in kaggle_df.head(limit).iterrows():
        print(f"🔄 Processing CV {i + 1}/{limit} | Category: {row['Category']}")

        mapped_cat = category_mapping.get(row['Category'], None)
        cat_requirements = requirements.get(mapped_cat, list(requirements.values())[0])

        arabic_cv = translate_cv(row['Resume_str'])
        time.sleep(2)

        if not arabic_cv:
            print(f"   ⚠️ تخطى CV {i + 1}")
            continue

        suggestions = generate_suggestions(arabic_cv, cat_requirements)
        time.sleep(2)

        results.append({
            'id': row['ID'],
            'category': row['Category'],
            'original_cv': row['Resume_str'],
            'arabic_cv': arabic_cv,
            'suggestions': suggestions
        })

        if (i + 1) % 10 == 0:
            pd.DataFrame(results).to_csv(
                f"backup_{i + 1}.csv",
                index=False,
                encoding='utf-8-sig'
            )
            print(f"   💾 Backup saved at CV {i + 1}")

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ خلص! اتحفظ {len(results)} سيرة ذاتية في {output_csv}")
    return output_df


# ── Run ───────────────────────────────────────────────
if __name__ == "__main__":
    df = run_pipeline(
        kaggle_csv="Resume.csv",
        arabjobs_csv="ArabJobs.csv",
        output_csv="arabic_cvs_with_suggestions.csv",
        limit=2484
    )