# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
import spacy
import re
import os
import joblib
from database import get_db_connection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Paths for model/vect (kept for your ML job-field predictor)
MODEL_PATH = "model.joblib"
VECT_PATH = "vect.joblib"

DEFAULT_TRAIN = [
    ("Information Technology", "python developer backend react javascript sql system design"),
    ("Information Technology", "software engineer java spring microservices backend"),
    ("Accounting", "accounting financial statements tax audit bookkeeping excel"),
    ("Accounting", "audit financial reporting budget ledger accounting"),
    ("English Literature", "literature english writing editing novel poetry translation"),
    ("Visual Communication Design", "design illustrator photoshop figma ui ux branding"),
    ("Information Systems", "information systems erp database business process analysis"),
]

app = Flask(__name__)
CORS(app)

# Load spaCy model (make sure en_core_web_sm is installed)
nlp = spacy.load("en_core_web_sm")


# ---------- model training / loading ----------
def train_and_save_default_model():
    texts = [t for _, t in DEFAULT_TRAIN]
    labels = [lbl for lbl, _ in DEFAULT_TRAIN]

    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
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


clf, vect = load_model_or_train()
# ---------- end ML ----------


# ---------- improved skill aliases ----------
SKILL_ALIASES = {
    # languages/frameworks
    "python": "Python", "java": "Java", "javascript": "JavaScript",
    "typescript": "TypeScript", "c#": "C#", "c++": "C++", "go": "Go",
    "golang": "Go", "dart": "Dart", "flutter": "Flutter",
    "react native": "React Native", "reactjs": "React", "react": "React",
    "vue": "Vue.js", "angular": "Angular", "node.js": "Node.js",
    "nodejs": "Node.js", "node": "Node.js", "express": "Express",
    "next.js": "Next.js", "nuxt": "Nuxt.js",
    "spring boot": "Spring Boot", "spring": "Spring", "django": "Django",
    "flask": "Flask", "fastapi": "FastAPI",

    # web/frontend
    "html": "HTML", "css": "CSS", "sass": "Sass", "less": "Less",
    "bootstrap": "Bootstrap", "tailwind": "Tailwind CSS",

    # db / storage / search
    "sql": "SQL", "nosql": "NoSQL", "postgresql": "PostgreSQL", "mysql": "MySQL",
    "mongodb": "MongoDB", "redis": "Redis", "elasticsearch": "Elasticsearch",

    # cloud / infra / devops
    "aws": "AWS", "amazon web services": "AWS", "azure": "Azure",
    "gcp": "GCP", "google cloud": "GCP", "docker": "Docker",
    "kubernetes": "Kubernetes", "terraform": "Terraform", "ci/cd": "CI/CD",
    "jenkins": "Jenkins", "github actions": "GitHub Actions", "ansible": "Ansible",
    "linux": "Linux", "bash": "Bash", "shell": "Shell", "sre": "SRE",

    # messaging / architecture
    "microservices": "Microservices", "rest api": "REST API", "graphql": "GraphQL",
    "kafka": "Kafka", "rabbitmq": "RabbitMQ",

    # data / ml
    "machine learning": "Machine Learning", "ml": "Machine Learning",
    "deep learning": "Deep Learning", "data science": "Data Science",
    "pandas": "pandas", "numpy": "numpy", "scikit-learn": "scikit-learn",
    "tensorflow": "TensorFlow", "pytorch": "PyTorch", "spark": "Apache Spark",

    # misc / tooling
    "git": "Git", "github": "GitHub", "gitlab": "GitLab", "figma": "Figma",
    "ux": "UX", "ui": "UI", "design": "Design", "photoshop": "Photoshop",
    "illustrator": "Illustrator", "excel": "Excel",

    # operations / backend
    "api": "API", "restful": "REST API", "hibernate": "Hibernate",
    "celery": "Celery",

    # security / hardware / other IT
    "cyber": "Cybersecurity", "cybersecurity": "Cybersecurity", "security": "Cybersecurity",
    "hardware": "Computer Hardware", "computer hardware": "Computer Hardware",
    "it support": "IT Support", "support": "IT Support",

    # certifications & keywords often used in CVs
    "certificate": "Certification", "certification": "Certification", "certified": "Certification",
    "training": "Training",

    # keep acronyms as uppercase
    "sql server": "SQL Server",
}
# ---------- end skill aliases ----------


# ---------- helpers ----------
def extract_text_from_pdf(file_stream):
    reader = PdfReader(file_stream)
    txt = ""
    for page in reader.pages:
        txt += page.extract_text() or ""
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def extract_email(text):
    m = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return m[0] if m else None


def extract_phone(text):
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    raw_candidates = re.findall(r"(?:\+?\d[\d\-\s\(\)/]{6,}\d)", text)
    candidates = []
    for cand in raw_candidates:
        # skip year-like tokens
        if re.search(r"\b(19|20)\d{2}\b", cand):
            continue
        digits = re.sub(r"\D", "", cand)
        if 8 <= len(digits) <= 15:
            candidates.append((cand.strip(), digits))

    if not candidates:
        return None

    def score_candidate(item):
        cand, digits = item
        s = 0
        if cand.startswith("+"):
            s += 5
        if digits.startswith("62"):
            s += 3
        if re.match(r"^0?8", digits):
            s += 2
        if 9 <= len(digits) <= 13:
            s += 2
        sep_count = len(re.findall(r"[\s\-\(\)/]", cand))
        s += max(0, 2 - sep_count)
        s += min(2, len(digits) - 7)
        return s

    best = max(candidates, key=score_candidate)[0]
    if best.startswith("+"):
        normalized = "+" + re.sub(r"[^\d]", "", best)
        normalized = re.sub(r"^\+{2,}", "+", normalized)
    else:
        normalized = re.sub(r"[^\d]", "", best)

    final_digits = re.sub(r"\D", "", normalized)
    if not (8 <= len(final_digits) <= 15):
        return None
    return normalized


def detect_skills(text: str, doc=None):
    lower = text.lower()
    normalized = re.sub(r"[^a-z0-9\+\#\.\s\-]", " ", lower)
    found = set()
    keys_sorted = sorted(SKILL_ALIASES.keys(), key=lambda s: -len(s))
    for key in keys_sorted:
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, normalized):
            found.add(SKILL_ALIASES[key])

    if doc is None:
        try:
            doc = nlp(text[:20000])
        except Exception:
            doc = None

    if doc:
        lemmas = {tok.lemma_.lower() for tok in doc if not tok.is_stop and tok.is_alpha}
        for key in keys_sorted:
            if SKILL_ALIASES[key] in found:
                continue
            if " " not in key and key in lemmas:
                found.add(SKILL_ALIASES[key])

    def sort_key(s):
        if s.upper() == s and len(s) <= 5:
            return (0, s)
        return (1, s.lower())

    return sorted(found, key=sort_key)


def predict_job_field(text):
    X = vect.transform([text])
    pred = clf.predict(X)[0]
    probs = clf.predict_proba(X)[0]
    classes = clf.classes_
    idx = list(classes).index(pred)
    confidence = round(float(probs[idx]) * 100, 1)
    top = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]
    top_formatted = [{"field": f, "prob": round(float(p) * 100, 1)} for f, p in top]
    return pred, confidence, top_formatted


def calculate_match_score(found_skills, expected_skills_list, model_confidence):
    # found_skills: list of display names (e.g., "Python")
    if expected_skills_list:
        expected_norm = [s.strip().lower() for s in expected_skills_list if s.strip()]
        if len(expected_norm) == 0:
            return 0
        found_norm = [s.lower() for s in found_skills]
        intersect = set(found_norm) & set(expected_norm)
        score = int((len(intersect) / len(expected_norm)) * 100)
        return score
    else:
        skill_score = min(100, len(found_skills) * 15)
        return round((skill_score * 0.4) + (model_confidence * 0.6))


# ---------- DB insert helper ----------
def insert_resume_to_db(record: dict):
    """
    record keys:
    filename, email, phone_number, skills_found (list), education_found (list),
    matching_score, recommendation, text_preview
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO resume_analysis
            (filename, email, phone_number, skills_found, education_found, matching_score, recommendation, text_preview)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                record.get("filename"),
                record.get("email"),
                record.get("phone_number"),
                record.get("skills_found", []),     # psycopg2 will convert python list -> SQL array
                record.get("education_found", []),
                record.get("matching_score"),
                record.get("recommendation"),
                record.get("text_preview"),
            ),
        )
        conn.commit()
        cur.close()
    except Exception as e:
        # don't raise — log for debug
        print("DB insert error:", e)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


# ---------- routes ----------
@app.route("/")
def home():
    return "ResuMatch backend running"


@app.route("/analyze", methods=["POST"])
def analyze_resume():
    # accept multiple files under key 'files' or single 'file'
    files = []
    if "files" in request.files:
        files = request.files.getlist("files")
    elif "file" in request.files:
        files = [request.files["file"]]
    else:
        return jsonify({"error": "No file uploaded"}), 400

    # expected skills (always convert to lowercase here)
    expected_skills_raw = request.form.get("expected_skills", "")
    expected_skills_list = [s.strip().lower() for s in expected_skills_raw.split(",") if s.strip()]

    results = []

    for file in files:
        try:
            text = extract_text_from_pdf(file)
            if not text.strip():
                results.append({
                    "filename": file.filename,
                    "error": "PDF uploaded but contains no readable text",
                    "text_preview": ""
                })
                continue

            doc = nlp(text[:20000])
            email = extract_email(text)
            phone = extract_phone(text)
            skills = detect_skills(text, doc)
            orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]

            predicted_field, confidence, top_candidates = predict_job_field(text)
            matching_score = calculate_match_score(skills, expected_skills_list, confidence)

            recommendation = "Recommended" if matching_score >= 55 else "Not Suitable"

            preview = text[:700]

            # prepare record for DB (match your table columns)
            record = {
                "filename": file.filename,
                "email": email,
                "phone_number": phone,
                "skills_found": skills,
                "education_found": orgs,
                "matching_score": matching_score,
                "recommendation": recommendation,
                "text_preview": preview
            }

            # insert to DB (best-effort)
            insert_resume_to_db(record)

            results.append({
                "filename": file.filename,
                "message": "Resume analyzed",
                "language_detected": "en",
                "email": email,
                "phone": phone,
                "skills_found": skills,
                "education_found": orgs,
                "matching_score": matching_score,
                "predicted_field": predicted_field,
                "field_confidence": confidence,
                "top_field_candidates": top_candidates,
                "recommendation": recommendation,
                "text_preview": preview
            })

        except Exception as e:
            results.append({
                "filename": getattr(file, "filename", "unknown"),
                "error": str(e)
            })

    return jsonify({"resumes": results}), 200


if __name__ == "__main__":
    app.run(port=5000, debug=True)
