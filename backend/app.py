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

nlp = spacy.load("en_core_web_sm")


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


def train_model_from_csv(csv_path):
    import csv
    texts = []
    labels = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["field"])
            texts.append(row["text"])
    v = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
    X = v.fit_transform(texts)
    model = LogisticRegression(max_iter=2000)
    model.fit(X, labels)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(v, VECT_PATH)
    return model, v


# Skill aliases dictionary
SKILL_ALIASES = {
    "python": "Python", "java": "Java", "javascript": "JavaScript",
    "typescript": "TypeScript", "c#": "C#", "c++": "C++", "go": "Go",
    "golang": "Go", "dart": "Dart", "flutter": "Flutter",
    "react native": "React Native", "reactjs": "React", "react": "React",
    "vue": "Vue.js", "angular": "Angular", "node.js": "Node.js",
    "nodejs": "Node.js", "node": "Node.js", "express": "Express",
    "spring boot": "Spring Boot", "spring": "Spring", "django": "Django",
    "flask": "Flask", "fastapi": "FastAPI", "html": "HTML", "css": "CSS",
    "sass": "Sass", "less": "Less", "bootstrap": "Bootstrap",
    "tailwind": "Tailwind CSS", "sql": "SQL", "nosql": "NoSQL",
    "postgresql": "PostgreSQL", "mysql": "MySQL", "mongodb": "MongoDB",
    "redis": "Redis", "elasticsearch": "Elasticsearch", "aws": "AWS",
    "amazon web services": "AWS", "azure": "Azure", "gcp": "GCP",
    "docker": "Docker", "kubernetes": "Kubernetes", "terraform": "Terraform",
    "ci/cd": "CI/CD", "jenkins": "Jenkins", "github actions": "GitHub Actions",
    "gitlab ci": "GitLab CI", "ansible": "Ansible", "linux": "Linux",
    "bash": "Bash", "shell": "Shell", "microservices": "Microservices",
    "rest api": "REST API", "graphql": "GraphQL", "kafka": "Kafka",
    "rabbitmq": "RabbitMQ", "machine learning": "Machine Learning",
    "ml": "Machine Learning", "data science": "Data Science",
    "pandas": "pandas", "numpy": "numpy", "scikit-learn": "scikit-learn",
    "tensorflow": "TensorFlow", "pytorch": "PyTorch", "spark": "Apache Spark",
    "hadoop": "Hadoop", "tableau": "Tableau", "power bi": "Power BI",
    "git": "Git", "github": "GitHub", "gitlab": "GitLab", "figma": "Figma",
    "ux": "UX", "ui": "UI", "design": "Design", "photoshop": "Photoshop",
    "illustrator": "Illustrator", "excel": "Excel", "hibernate": "Hibernate",
    "celery": "Celery", "api": "API", "nlp": "NLP",
    "natural language processing": "NLP", "devops": "DevOps",
    "site reliability engineering": "SRE", "sre": "SRE",
}


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
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    raw_candidates = re.findall(r"(?:\+?\d[\d\-\s\(\)/]{6,}\d)", text)
    candidates = []

    for cand in raw_candidates:
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
        if cand.startswith("+"): s += 5
        if digits.startswith("62"): s += 3
        if re.match(r"^0?8", digits): s += 2
        if 9 <= len(digits) <= 13: s += 2
        sep_count = len(re.findall(r"[\s\-\(\)/]", cand))
        s += max(0, 2 - sep_count)
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
        except:
            doc = None

    if doc:
        lemmas = {tok.lemma_.lower() for tok in doc if not tok.is_stop and tok.is_alpha}
        for key in keys_sorted:
            if SKILL_ALIASES[key] in found:
                continue
            if " " not in key and key in lemmas:
                found.add(SKILL_ALIASES[key])

    def sort_key(s):
        return (0, s) if s.isupper() and len(s) <= 5 else (1, s.lower())

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


@app.route("/")
def home():
    return "ResuMatch backend running"


@app.route("/analyze", methods=["POST"])
def analyze_resume():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        text = extract_text_from_pdf(file)
        if not text.strip():
            return jsonify({"message": "PDF uploaded but contains no readable text", "text_preview": ""}), 200

        doc = nlp(text[:20000])
        email = extract_email(text)
        phone = extract_phone(text)
        skills = detect_skills(text, doc)
        orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]

        predicted_field, confidence, top_candidates = predict_job_field(text)

        skill_score = min(100, len(skills) * 15)
        matching_score = round((skill_score * 0.4) + (confidence * 0.6))

        preview = text[:700]

        # Save to database (experience removed)
        try:
            from database import get_connection
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO resume_analysis 
                (filename, email, phone, skills, education, matching_score, predicted_field, field_confidence, text_preview)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                file.filename,
                email,
                phone,
                ", ".join(skills),
                ", ".join(orgs),
                matching_score,
                predicted_field,
                confidence,
                preview
            ))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as db_e:
            print("DB save error:", db_e)

        return jsonify({
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
            "text_preview": preview
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
