"""
balance_dataset_v3.py
======================
شغّله بـ: python balance_dataset_v3.py

✅ الفرق عن v2:
  - TARGET بقى ديناميكي = أكبر عدد موجود في أي class (مش رقم ثابت 400)
  - يعني هيزوّد كل الفئات الناقصة لحد ما توصل لمستوى الفئة المهيمنة (ممتاز)
  - مفيش تقليل لأي داتا أبداً — بس إضافة
  - Resume تلقائي زي v2 (لو arabic_cvs_balanced.csv موجود، يكمل منه)
"""

import re
import time
import random
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ══════════════════════════════════════════════════════
#  ✏️  عدّل هنا بس
# ══════════════════════════════════════════════════════
INPUT_CSV    = "arabic_cvs_with_scores.csv"
OUTPUT_CSV   = "arabic_cvs_balanced.csv"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_URL   = "http://localhost:11434/api/generate"
ROW_RETRIES  = 2
SAVE_EVERY   = 25
EXTRA_BUFFER = 0     # ✅ لو عايز تزود شوية فوق أكبر class (مثلاً 100 يبقى الهدف = max+100)
# ══════════════════════════════════════════════════════

BACKUP_DIR = Path("backups")
BACKUP_DIR.mkdir(exist_ok=True)

CATEGORIES_EN = [
    'INFORMATION-TECHNOLOGY', 'BUSINESS-DEVELOPMENT', 'ADVOCATE', 'CHEF',
    'FINANCE', 'ENGINEERING', 'ACCOUNTANT', 'FITNESS', 'AVIATION', 'SALES',
    'HEALTHCARE', 'CONSULTANT', 'BANKING', 'CONSTRUCTION', 'PUBLIC-RELATIONS',
    'HR', 'DESIGNER', 'ARTS', 'TEACHER', 'APPAREL', 'DIGITAL-MEDIA',
    'AGRICULTURE', 'AUTOMOBILE', 'BPO'
]

CATEGORY_AR = {
    'INFORMATION-TECHNOLOGY': 'تكنولوجيا المعلومات والبرمجة',
    'BUSINESS-DEVELOPMENT'  : 'تطوير الأعمال والإدارة',
    'ADVOCATE'              : 'المحاماة والقانون',
    'CHEF'                  : 'الطهي والمطاعم',
    'FINANCE'               : 'المالية والاستثمار',
    'ENGINEERING'           : 'الهندسة',
    'ACCOUNTANT'            : 'المحاسبة والمراجعة',
    'FITNESS'               : 'اللياقة البدنية والرياضة',
    'AVIATION'              : 'الطيران والملاحة الجوية',
    'SALES'                 : 'المبيعات وخدمة العملاء',
    'HEALTHCARE'            : 'الرعاية الصحية والطب',
    'CONSULTANT'            : 'الاستشارات الإدارية',
    'BANKING'               : 'البنوك والخدمات المصرفية',
    'CONSTRUCTION'          : 'البناء والمقاولات',
    'PUBLIC-RELATIONS'      : 'العلاقات العامة والإعلام',
    'HR'                    : 'الموارد البشرية',
    'DESIGNER'              : 'التصميم الجرافيكي والإبداعي',
    'ARTS'                  : 'الفنون والثقافة',
    'TEACHER'               : 'التعليم والتدريس',
    'APPAREL'               : 'الأزياء والملابس',
    'DIGITAL-MEDIA'         : 'الإعلام الرقمي والتسويق الإلكتروني',
    'AGRICULTURE'           : 'الزراعة والإنتاج الغذائي',
    'AUTOMOBILE'            : 'صناعة السيارات والميكانيكا',
    'BPO'                   : 'خدمات الأعمال الخارجية',
}

CLASS_SCORE_RANGES = {
    'ضعيف' : (42, 55),
    'متوسط': (56, 70),
    'جيد'  : (71, 79),
    'ممتاز': (80, 88),
}

LEVEL_PROMPTS = {
    'ضعيف': {
        'ar_desc' : 'ضعيف جداً ومبتدئ بدون خبرة',
        'qualities': [
            'بدون أي خبرة عملية سابقة',
            'لا يوجد قسم مهارات محدد',
            'CV قصير (لا يقل عن 100 كلمة لكن بسيط جداً)',
            'معلومات تواصل ناقصة جداً',
            'تعليم غير مكتمل أو غير ذي صلة',
            'أسلوب كتابة ضعيف وغير منظم',
            'لا يوجد ملخص مهني على الإطلاق',
        ]
    },
    'متوسط': {
        'ar_desc' : 'متوسط المستوى بخبرة محدودة',
        'qualities': [
            'خبرة عمل سنة أو سنتين فقط',
            'مهارات أساسية في المجال فقط',
            'بريد إلكتروني موجود بدون هاتف أو LinkedIn',
            'تعليم جيد لكن بدون شهادات احترافية',
            'CV متوسط الطول (120-180 كلمة)',
            'بدون إنجازات محددة بأرقام',
        ]
    },
    'جيد': {
        'ar_desc' : 'جيد ومتمكن بخبرة واضحة',
        'qualities': [
            'خبرة عمل 3 إلى 5 سنوات في مجاله',
            'مهارات تقنية متعددة في المجال',
            'بريد وهاتف وربما LinkedIn',
            'شهادة أكاديمية مع شهادة احترافية واحدة',
            'CV منظم (250-350 كلمة)',
            'بعض الإنجازات المحددة بأرقام',
            'ملخص مهني جيد في البداية',
        ]
    },
    'ممتاز': {
        'ar_desc' : 'ممتاز ومتميز بخبرة واسعة',
        'qualities': [
            'خبرة عمل 7 سنوات أو أكثر',
            'قيادة فرق وإدارة مشاريع كبيرة',
            'إنجازات استثنائية بأرقام ونسب مئوية',
            'شهادات احترافية متعددة ومعتمدة دولياً',
            'CV شامل ومفصل (450-600 كلمة)',
            'LinkedIn وGitHub أو Portfolio',
            'ملخص مهني ممتاز يبرز الإنجازات',
        ]
    }
}

ATS_WEIGHTS = {
    'has_experience': 15, 'has_education': 12, 'has_skills': 12,
    'has_summary'   : 8,  'has_email'    : 6,  'has_phone' : 4,
    'good_length'   : 6,  'has_contact'  : 7,
}

MIN_WORDS = 60


def calculate_ats_score(arabic_cv):
    cv = str(arabic_cv).lower()
    checks = {
        'has_experience': any(k in cv for k in ['الخبرات','خبرة العمل','الخبرة المهنية','التجربة']),
        'has_education' : any(k in cv for k in ['التعليم','المؤهلات','الدراسة','الشهادة']),
        'has_skills'    : any(k in cv for k in ['المهارات','الكفاءات','قدرات','المهارة']),
        'has_summary'   : any(k in cv for k in ['ملخص','نبذة','عن نفسي','profile','الملخص']),
        'has_email'     : bool(re.search(r'[\w\.-]+@[\w\.-]+', cv)),
        'has_phone'     : bool(re.search(r'[\+\d][\d\s\-]{8,}', cv)),
        'good_length'   : 80 <= len(cv.split()) <= 800,
        'has_contact'   : any(k in cv for k in ['linkedin','github','portfolio','موقع']),
    }
    struct_score  = sum(ATS_WEIGHTS[k] for k, v in checks.items() if v)
    keyword_score = min(30, len([w for w in cv.split() if len(w) > 4]) // 10)
    return min(int(struct_score + keyword_score), 100)


def call_ollama(prompt, max_retries=3, timeout=120):
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model"  : OLLAMA_MODEL,
                    "prompt" : prompt,
                    "stream" : False,
                    "options": {"temperature": 0.85, "top_p": 0.9, "num_predict": 1400}
                },
                timeout=timeout
            )
            if resp.status_code == 200:
                text = resp.json().get("response", "").strip()
                if text:
                    return text
        except requests.exceptions.Timeout:
            print(f"         ⏱️  Timeout ({attempt+1}/{max_retries})")
        except Exception as e:
            print(f"         ❌ {e} ({attempt+1}/{max_retries})")
        time.sleep(2 ** attempt)
    return None


def _try_generate_once(category_en, level, score):
    cat_ar     = CATEGORY_AR.get(category_en, category_en)
    level_info = LEVEL_PROMPTS[level]
    chosen     = random.sample(level_info['qualities'], min(4, len(level_info['qualities'])))
    q_text     = '\n'.join(f'  - {q}' for q in chosen)

    first = random.choice(['محمد','أحمد','علي','عمر','يوسف','خالد','سارة','مريم','فاطمة','نورا','رنا','هند','لينا','ريم'])
    last  = random.choice(['العمري','الحربي','الشمري','السيد','حسن','إبراهيم','محمود','عبدالله','الزهراني','القحطاني'])
    name  = f"{first} {last}"
    email = f"{re.sub(r'[^a-z]','', first.lower())}{random.randint(10,99)}@gmail.com"
    phone = f"05{random.randint(10000000, 99999999)}"

    cv_text = call_ollama(f"""اكتب سيرة ذاتية عربية كاملة لشخص يعمل في مجال "{cat_ar}".

بيانات الشخص:
- الاسم: {name}
- البريد الإلكتروني: {email}
- رقم الهاتف: {phone}
- مستوى الـ CV: {level_info['ar_desc']}
- الدرجة المستهدفة: {score} من 100

الخصائص المطلوبة:
{q_text}

تعليمات: اكتب باللغة العربية فقط، ابدأ مباشرةً بمحتوى الـ CV بدون أي مقدمة، ولا يقل المحتوى عن {MIN_WORDS} كلمة.

اكتب الـ CV الآن:""")

    if not cv_text or len(cv_text.split()) < MIN_WORDS:
        return None

    suggestions = call_ollama(
        f"""اكتب 3 اقتراحات تحسين مختصرة باللغة العربية (نقاط مرقمة فقط) لهذا الـ CV في مجال "{cat_ar}":
{cv_text[:400]}
الاقتراحات:""",
        timeout=60
    )
    if not suggestions:
        suggestions = f"حسّن الـ CV لرفع مستواه في مجال {cat_ar}."

    return {
        'arabic_cv'  : cv_text,
        'suggestions': suggestions,
        'ats_score'  : calculate_ats_score(cv_text),
    }


def generate_row(category_en, level, score, retries=ROW_RETRIES):
    for attempt in range(retries + 1):
        result = _try_generate_once(category_en, level, score)
        if result:
            return result
        if attempt < retries:
            print(f'(retry {attempt+1}) ', end='', flush=True)
    return None


def get_dist(df):
    return {
        lv: int(((df['suitability_score'] >= lo) & (df['suitability_score'] <= hi)).sum())
        for lv, (lo, hi) in CLASS_SCORE_RANGES.items()
    }


def save_df(df, path):
    df.to_csv(path, index=False, encoding='utf-8-sig')


# ── Main ───────────────────────────────────────────────
if __name__ == '__main__':

    print(f'\n{"="*55}')
    print(f'  Arabic CV Dataset Balancer — v3 (Top-Up Only)')
    print(f'{"="*55}')

    output_path = Path(OUTPUT_CSV)
    input_path  = Path(INPUT_CSV)

    if output_path.exists():
        print(f'\n♻️  {OUTPUT_CSV} موجود بالفعل — هنكمل منه (Resume)')
        df = pd.read_csv(output_path, encoding='utf-8-sig')
    else:
        print(f'\n🆕 {OUTPUT_CSV} غير موجود — هنبدأ من {INPUT_CSV}')
        df = pd.read_csv(input_path, encoding='utf-8-sig')

    print(f'✅ Loaded {len(df):,} rows | Columns: {list(df.columns)}')

    # ── Backup ──────────────────────────────────────────
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    bkp = BACKUP_DIR / f'backup_{ts}.csv'
    df.to_csv(bkp, index=False, encoding='utf-8-sig')
    print(f'💾 Backup → {bkp}')

    sc = df['suitability_score']
    print(f'\n📈 Scores: min={sc.min()}  max={sc.max()}  mean={sc.mean():.1f}  std={sc.std():.1f}')

    # ── ✅ Current distribution + Dynamic Target ─────────
    cur    = get_dist(df)
    TARGET = max(cur.values()) + EXTRA_BUFFER   # ✅ الهدف = أكبر فئة موجودة (مش رقم ثابت)

    print(f'\n📊 Current Distribution:')
    for lv, (lo, hi) in CLASS_SCORE_RANGES.items():
        n   = cur[lv]
        pct = n / len(df) * 100
        bar = '█' * int(pct / 2)
        print(f'   {lv:>6} ({lo}-{hi}): {n:>5}  ({pct:5.1f}%)  {bar}')

    print(f'\n🎯 Dynamic Target (= أكبر فئة + buffer): {TARGET:,}')

    needed       = {lv: max(0, TARGET - cur[lv]) for lv in CLASS_SCORE_RANGES}
    total_needed = sum(needed.values())

    print(f'\n📝 To generate (كل ده إضافة، مفيش حذف):')
    for lv, n in needed.items():
        print(f'   {lv:>6}: {"✅ هو الأعلى" if n==0 else f"+{n:,}"}')
    print(f'   Total : +{total_needed:,}')

    if total_needed == 0:
        print('\n✅ Dataset already balanced! Nothing to generate.')
        exit()

    # تقدير وقت تقريبي (بناءً على ~10-15 ثانية للـ CV الواحد)
    est_hours = total_needed * 12 / 3600
    print(f'\n⏱️  تقدير الوقت التقريبي: {est_hours:.1f} ساعة (حسب سرعة جهازك)')
    print(f'   💡 السكريبت بيحفظ تلقائي كل {SAVE_EVERY} CV، فممكن توقفه وترجعه يكمل في أي وقت\n')

    print(f'🚀 Starting generation of {total_needed:,} CVs...\n')

    generated     = 0
    failed        = 0
    start_time    = time.time()
    max_id        = int(df['id'].max()) + 1
    since_save    = 0

    for level, count in needed.items():
        if count == 0:
            continue

        lo, hi = CLASS_SCORE_RANGES[level]
        print(f'  {"="*50}')
        print(f'  📝 Class "{level}"  →  +{count:,} CVs  (score {lo}-{hi})')
        print(f'  {"="*50}')

        for i in range(count):
            category = random.choice(CATEGORIES_EN)
            score    = random.randint(lo, hi)
            done     = generated + failed
            eta_min  = ((time.time()-start_time)/done*(total_needed-done)/60) if done > 0 else 0

            print(f'  [{done+1:>5}/{total_needed}] {category:<25} score={score}  ', end='', flush=True)

            result = generate_row(category, level, score)

            if result:
                new_row = {
                    'id'               : max_id,
                    'category'         : category,
                    'original_cv'      : '',
                    'arabic_cv'        : result['arabic_cv'],
                    'suggestions'      : result['suggestions'],
                    'ats_score'        : result['ats_score'],
                    'suitability_score': score,
                }
                df = pd.concat([df, pd.DataFrame([new_row])[list(df.columns)]], ignore_index=True)
                max_id     += 1
                generated  += 1
                since_save += 1
                words = len(result['arabic_cv'].split())
                print(f'✅ ({words} words | ATS={result["ats_score"]} | ETA≈{eta_min:.0f}min)')
            else:
                failed += 1
                print('❌ Failed (skipped after retries)')

            if since_save >= SAVE_EVERY:
                save_df(df, output_path)
                print(f'         💾 Saved progress → {output_path}  ({len(df):,} rows total)')
                since_save = 0

            time.sleep(0.3)

        print()

    save_df(df, output_path)

    new_dist = get_dist(df)
    print(f'\n📊 Final Distribution:')
    for lv, (lo, hi) in CLASS_SCORE_RANGES.items():
        n_old = cur[lv]
        n_new = new_dist[lv]
        pct   = n_new / len(df) * 100
        bar   = '█' * int(pct / 2)
        print(f'   {lv:>6}: {n_old:>5} → {n_new:>5} ({pct:5.1f}%) {bar}')

    elapsed = time.time() - start_time
    print(f'\n{"="*55}')
    print(f'💾 Saved  → {output_path}')
    print(f'   Total rows now : {len(df):,}')
    print(f'   ✅ Generated   : {generated:,}')
    print(f'   ❌ Failed      : {failed:,}')
    print(f'   ⏱️  Time        : {elapsed/60:.1f} min')
    print(f'{"="*55}')

    remaining = sum(max(0, TARGET - new_dist[lv]) for lv in CLASS_SCORE_RANGES)
    if remaining > 0:
        print(f'\n⚠️  لسه ناقص {remaining:,} CVs (غالباً بسبب فشل بعض المحاولات).')
        print(f'   شغّل السكريبت تاني وهيكمل تلقائياً من حيث ما توقف.')
    else:
        print(f'\n✅ كل الفئات وصلت لمستوى الفئة المهيمنة!')
