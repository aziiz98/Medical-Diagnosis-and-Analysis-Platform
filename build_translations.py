import json
import os
import time
import pandas as pd
from deep_translator import MyMemoryTranslator

CSV_FILE = 'Final_Cleaned_Disease_Data.csv'
OUTPUT_FILE = 'diseases_ar.json'

# Accurate manual translations for common diseases
manual_translations = {
    'flu': 'الإنفلونزا',
    'common cold': 'نزلة برد',
    'seasonal allergies (hay fever)': 'الحساسية الموسمية',
    'asthma': 'الربو',
    'pneumonia': 'الالتهاب الرئوي',
    'bronchitis': 'التهاب الشعب الهوائية',
    'migraine': 'الصداع النصفي',
    'tension headache': 'صداع التوتر',
    'gastroenteritis': 'التهاب المعدة والأمعاء',
    'gastroesophageal reflux disease (gerd)': 'ارتجاع المريء',
    'anxiety': 'اضطراب القلق',
    'depression': 'الاكتئاب',
    'panic disorder': 'اضطراب الهلع',
    'eczema': 'الأكزيما',
    'acne': 'حب الشباب',
    'psoriasis': 'الصدفية',
    'urinary tract infection': 'التهاب المسالك البولية',
    'hypertension': 'ارتفاع ضغط الدم',
    'diabetes': 'السكري',
    'anemia': 'فقر الدم',
    'sinusitis': 'التهاب الجيوب الأنفية',
    'tonsillitis': 'التهاب اللوزتين',
    'strep throat': 'التهاب الحلق العقدي',
    'arthritis': 'التهاب المفاصل',
    'allergy': 'حساسية',
    'conjunctivitis': 'التهاب الملتحمة',
    'ear infection': 'التهاب الأذن',
    'food poisoning': 'تسمم غذائي',
    'irritable bowel syndrome (ibs)': 'القولون العصبي'
}

# Load existing translations if the script was run before
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        translations = json.load(f)
else:
    translations = {}

# Apply manual translations first
for key, ar_name in manual_translations.items():
    translations[key] = f"{ar_name} ({key})"

# Read unique disease names from the dataset
df = pd.read_csv(CSV_FILE, usecols=['diseases'])
unique_diseases = sorted(df['diseases'].dropna().astype(str).str.strip().unique())

translator = MyMemoryTranslator(source='en-GB', target='ar-SA')
total = len(unique_diseases)
translated_now = 0

print(f"Total unique diseases: {total}")

for i, disease in enumerate(unique_diseases, start=1):
    key = disease.lower()
    if key in translations:
        continue

    try:
        ar_name = translator.translate(disease)
        if ar_name and ar_name.lower() != key:
            translations[key] = f"{ar_name} ({disease})"
        else:
            translations[key] = disease
        translated_now += 1
    except Exception as e:
        print(f"Skipped '{disease}': {e}")
        translations[key] = disease

    # Save progress every 10 diseases
    if translated_now % 10 == 0:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(translations, f, ensure_ascii=False, indent=2)
        print(f"Progress: {i}/{total}")

    time.sleep(0.2)

# Final save
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(translations, f, ensure_ascii=False, indent=2)

print(f"Done. Saved {len(translations)} diseases to {OUTPUT_FILE}.")