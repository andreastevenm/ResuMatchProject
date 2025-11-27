# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
import spacy
import re
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------
# Config
# -----------------------
MODEL_PATH = "model.joblib"
VECT_PATH = "vect.joblib"

# Example small default training dataset (field -> example texts)
# You can replace / expand this with your CSV and call train_model_from_csv
DEFAULT_TRAIN = [
    ("Information Technology", "python developer backend react javascript sql system design"),
    ("Information Technology", "software engineer java spring microservices backend"),
    ("Accounting", "accounting financial statements tax audit bookkeeping excel"),
    ("Accounting", "audit financial reporting budget ledger accounting"),
    ("English Literature", "literature english writing editing novel poetry translation"),
    ("Visual Communication Design", "design illustrator photoshop figma ui ux branding"),
    ("Information Systems", "information systems erp database business process analysis"),
]

# -----------------------
# App init
# -----------------------
app = Flask(__name__)
CORS(app)

nlp = spacy.load("en_core_web_sm")

# -----------------------
# Training / Model utils
# -----------------------
def train_and_save_default_model():
    texts = [t for _, t in DEFAULT_TRAIN]
    labels = [lbl for lbl, _ in DEFAULT_TRAIN]

    vect = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
    X = vect.fit_transform(texts)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, labels)

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vect, VECT_PATH)
    return clf, vect

def load_model_or_train():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
        clf = joblib.load(MODEL_PATH)
        vect = joblib.load(VECT_PATH)
        return clf, vect
    else:
        return train_and_save_default_model()

# Load (or train) model once at startup
clf, vect = load_model_or_train()

# Optional helper for re-training from CSV (not auto-called)
def train_model_from_csv(csv_path):
    # CSV should have columns: field,text
    import csv
    texts = []
    labels = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["field"])
            texts.append(row["text"])
    v = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
    X = v.fit_transform(texts)
    model = LogisticRegression(max_iter=2000)
    model.fit(X, labels)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(v, VECT_PATH)
    return model, v

# -----------------------
# Helpers: extractors & classifiers
# -----------------------
SKILL_LIST = ["python","java","javascript","react","node","sql","html","css",
              "communication","leadership","teamwork","flutter","excel","audit","tax","design","figma","ux","ui"]

def extract_text_from_pdf(file_stream):
    reader = PdfReader(file_stream)
    txt = ""
    for page in reader.pages:
        txt += page.extract_text() or ""
    return txt

def extract_email(text):
    m = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return m[0] if m else None

def extract_phone(text):
    m = re.findall(r"\+?\d[\d\s\-\(\)]{7,}\d", text)
    return m[0] if m else None

def detect_skills(text):
    found = []
    lower = text.lower()
    for s in SKILL_LIST:
        if s in lower:
            found.append(s)
    return found

def predict_job_field(text):
    X = vect.transform([text])
    pred = clf.predict(X)[0]
    probs = clf.predict_proba(X)[0]
    classes = clf.classes_
    # get probability for predicted class
    idx = list(classes).index(pred)
    confidence = round(float(probs[idx]) * 100, 1)
    # also return top 3 candidates
    top = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]
    top_formatted = [{"field": f, "prob": round(float(p)*100,1)} for f,p in top]
    return pred, confidence, top_formatted

# -----------------------
# Routes
# -----------------------
@app.route("/")
def home():
    return "ResuMatch backend running"

@app.route("/analyze", methods=["POST"])
def analyze_resume():
    if "file" not in request.files:
        return jsonify({"error":"No file uploaded"}), 400

    file = request.files["file"]

    try:
        text = extract_text_from_pdf(file)
        if not text.strip():
            return jsonify({"message":"PDF uploaded but contains no readable text","text_preview":""}), 200

        # Basic NLP + extractors
        doc = nlp(text[:20000])  # limit length for speed
        email = extract_email(text)
        phone = extract_phone(text)
        skills = detect_skills(text)
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

        # ML prediction
        predicted_field, confidence, top_candidates = predict_job_field(text)

        # Matching score: combine skills coverage + model confidence (simple formula)
        skill_score = min(100, len(skills) * 15)
        matching_score = round((skill_score * 0.4) + (confidence * 0.6))

        preview = text[:700]

        # Save to DB if database module available (non-blocking)
        try:
            from database import get_connection
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO resume_analysis 
                (filename, email, phone, skills, education, experience, matching_score, predicted_field, field_confidence, text_preview)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                file.filename,
                email,
                phone,
                ", ".join(skills),
                ", ".join(orgs),
                ", ".join(dates),
                matching_score,
                predicted_field,
                confidence,
                preview
            ))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as db_e:
            # don't fail the request just because DB failed; log to console
            print("DB save error:", db_e)

        return jsonify({
            "message":"Resume analyzed",
            "language_detected":"en",
            "email": email,
            "phone": phone,
            "skills_found": skills,
            "education_found": orgs,
            "experience_found": dates,
            "matching_score": matching_score,
            "predicted_field": predicted_field,
            "field_confidence": confidence,
            "top_field_candidates": top_candidates,
            "text_preview": preview
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)
