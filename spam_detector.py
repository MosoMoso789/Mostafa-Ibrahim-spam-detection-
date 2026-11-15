import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ==========================================
# 1. إعداد البيانات (Dataset Preparation)
# ==========================================
# ملاحظة: يمكنك استبدال هذا الجزء بقراءة ملف CSV حقيقي مثل:
# df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
# df.columns = ['label', 'message']

data = {
    'label': [
        'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam',
        'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam'
    ],
    'message': [
        "Hey, are we still meeting for coffee tomorrow at 10?",
        "CONGRATULATIONS! You won a FREE iPhone 15! Click here to claim now!",
        "Please find attached the lecture notes for Machine Learning.",
        "URGENT: Your bank account has been suspended. Verify your password immediately.",
        "Can you send me the Python code for the graduation project?",
        "WINNER!! As a valued customer you have been selected to receive a $1000 prize!",
        "Don't forget to submit the assignment before midnight.",
        "Free entry in 2 a wkly comp to win FA Cup final tkts! Text WIN to 87121.",
        "Mom asked if you are coming home for dinner tonight.",
        "Get cheap loans with 0% interest! Apply today, no credit check required.",
        "Let's review the slides together before the exam.",
        "You have 1 new voicemail. Call 09061701461 to listen now. Charges apply.",
        "Thanks for helping me understand the RBF kernel earlier!",
        "HOT SINGLES in your area want to meet you! Click the link now.",
        "I'll be 10 minutes late to the lab session.",
        "Claim your free cash reward now by replying YES to this message!"
    ]
}

df = pd.DataFrame(data)

# تحويل التصنيفات إلى أرقام (Ham = 0, Spam = 1)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# ==========================================
# 2. تنظيف وتجهيز النصوص (Text Preprocessing)
# ==========================================
def clean_text(text):
    text = text.lower()  # تحويل الحروف إلى صغيرة
    text = re.sub(r'[^a-z0-9\s]', '', text)  # إزالة علامات الترقيم والرموز الخاصة
    text = re.sub(r'\s+', ' ', text).strip()  # إزالة المسافات الزائدة
    return text

df['clean_message'] = df['message'].apply(clean_text)

# ==========================================
# 3. تقسيم البيانات وتحويل النصوص إلى أرقام (TF-IDF)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'], df['label_num'], test_size=0.25, random_state=42, stratify=df['label_num']
)

# استخدام TF-IDF مع إزالة الكلمات الشائعة (Stop words)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ==========================================
# 4. تدريب وتقييم النماذج (Training & Evaluation)
# ==========================================
models = {
    "Naive Bayes (MultinomialNB)": MultinomialNB(),
    "Logistic Regression": LogisticRegression()
}

for name, model in models.items():
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"{'='*40}")
    
    # تدريب النموذج
    model.fit(X_train_tfidf, y_train)
    
    # التنبؤ على بيانات الاختبار
    y_pred = model.predict(X_test_tfidf)
    
    # حساب مقاييس التقييم المطلوبة
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Accuracy  : {acc:.2f}")
    print(f"Precision : {prec:.2f}")
    print(f"Recall    : {rec:.2f}")
    print(f"F1-Score  : {f1:.2f}")

# ==========================================
# 5. اختبار النموذج على رسائل جديدة (Prediction)
# ==========================================
best_model = models["Naive Bayes (MultinomialNB)"]

def predict_email(email_text):
    cleaned = clean_text(email_text)
    vec = vectorizer.transform([cleaned])
    prediction = best_model.predict(vec)[0]
    return "SPAM (مزعجة)" if prediction == 1 else "HAM (طبيعية)"

print("\n--- اختبار رسائل جديدة ---")
sample_1 = "Congratulations! You won a free gift card, click here to claim."
sample_2 = "Hi, can we reschedule our meeting to 3 PM?"

print(f"Message 1: '{sample_1}' -> {predict_email(sample_1)}")
print(f"Message 2: '{sample_2}' -> {predict_email(sample_2)}")
