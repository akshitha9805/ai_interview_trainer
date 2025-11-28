# ---------------------------------------------
#  IntervYou – app.py (Analysis logic improved)
#  *Backend analysis only — UI unchanged*
# ---------------------------------------------
import matplotlib
matplotlib.use("Agg")  # Critical for server PDF charts

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
import os, json, re, random
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Dict, Any


# ---------------------------------------------------------
# FLASK APP SETUP
# ---------------------------------------------------------
app = Flask(__name__)
app.secret_key = "replace-with-secure-key"

app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

VIDEO_FOLDER = os.path.join(app.config["UPLOAD_FOLDER"], "videos")
os.makedirs(VIDEO_FOLDER, exist_ok=True)

LOGO_PATH = os.path.join("static", "images", "logo", "interyou.png")

# ------------------------------
# Correct spaCy model loader for Render
# ------------------------------
import spacy
import en_core_web_sm

nlp = en_core_web_sm.load()
# --------------------------
# ---------------------------------------------------------
# USER DB HELPERS
# ---------------------------------------------------------
USER_DB = "users.json"

def load_users():
    return json.load(open(USER_DB)) if os.path.exists(USER_DB) else {}

def save_users(data):
    json.dump(data, open(USER_DB, "w"), indent=2)

# ---------------------------------------------------------
# RESUME TEXT EXTRACTION
# ---------------------------------------------------------
def extract_resume_text(pdf_path):
    text_chunks = []
    try:
        reader = PdfReader(open(pdf_path, "rb"))
        for page in reader.pages:
            t = page.extract_text() or ""
            if t:
                text_chunks.append(t)
    except:
        return ""
    return "\n".join(text_chunks)


# ---------------------------------------------------------
# NAME EXTRACTION
# ---------------------------------------------------------
def extract_name_from_text(text):
    if not text:
        return "Not specified"

    # first lines heuristic
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines[:5]:
        if re.search(r"[@0-9]", l): continue
        if len(l.split()) <= 4 and 2 <= len(l) <= 50:
            return l

    # spaCy fallback
    doc = nlp(text[:2000])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text

    return "Not specified"


# ---------------------------------------------------------
# STRUCTURED RESUME DATA
# ---------------------------------------------------------
def extract_resume_fields(text):
    text = text or ""

    name = extract_name_from_text(text)

    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    email = email_match.group(0) if email_match else "Not specified"

    tech_keys = ["python","java","sql","excel","power bi","machine learning",
                 "ai","react","django","flask","html","css","javascript",
                 "docker","kubernetes","mongodb"]

    found = [k.capitalize() for k in tech_keys if k in text.lower()]
    if not found:
        found = ["Not specified"]

    edu_match = re.search(r"(Bachelor|Master|B\.Tech|M\.Tech).*?(Computer|Science|Engineering|Technology)",
                          text, re.IGNORECASE)
    education = edu_match.group(0) if edu_match else "Not specified"

    return {
        "name": name,
        "email": email,
        "skills": found,
        "education": education,
        "experience": "Extracted experience",
        "projects": ["Project 1", "Project 2"]
    }


# ---------------------------------------------------------
# QUESTION GENERATOR
# ---------------------------------------------------------
def generate_personalized_questions(text, interview_type="Mixed", n=8):
    base = ["Please introduce yourself.", "Why do you want this job?", "What are your strengths?"]
    return [{"question_text": q} for q in base][:n]


# ---------------------------------------------------------
# TRANSCRIPT ANALYSIS HELPERS
# ---------------------------------------------------------
FILLER_WORDS = {"um","uh","erm","ah","like","you know","i mean","so","actually","basically","right"}
CONFIDENCE_KEYWORDS = {"confident","sure","certain","absolutely","definitely","clearly","strong"}

POSITIVE_WORDS = {"good","well","success","achieved","improved","effective","efficient","strong"}
NEGATIVE_WORDS = {"problem","issue","difficult","struggle","fail","failed","weak","concern"}

STAR_KEYWORDS = {"situation","task","action","result","challenge","led","implemented","designed","improved"}


def safe_text(t):
    return (t or "").strip()


def tokenize_sentences(text):
    # lightweight sentence splitter
    if not text: return []
    # split on .,?,! and newlines
    parts = re.split(r"(?<=[\.\?!])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def count_fillers(text):
    t = text.lower()
    c = 0
    for f in FILLER_WORDS:
        c += t.count(f)
    return c


def contains_keywords(text, keywords):
    t = text.lower()
    for k in keywords:
        if k in t:
            return True
    return False


def proportion_keywords(text, keywords):
    words = re.findall(r"\w+", (text or "").lower())
    if not words: return 0.0
    hit = sum(1 for w in words if w in keywords)
    return hit / len(words)


# Normalize helpers: map raw metric to 0..1

def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def normalize_by_range(x, minv, maxv):
    if maxv == minv: return 0.0
    return clamp01((x - minv) / (maxv - minv))


# Compute scores based on transcript entries
def analyze_transcript(transcript, resume_struct):
    # transcript expected as list of dicts or strings
    entries = []
    if isinstance(transcript, list):
        for t in transcript:
            if isinstance(t, dict):
                # prefer 'answer' or 'text'
                txt = t.get('answer') or t.get('response') or t.get('text') or t.get('transcript') or ''
                entries.append(safe_text(txt))
            else:
                entries.append(safe_text(t))
    elif isinstance(transcript, str):
        entries = [transcript]

    # aggregate text
    full = "\n".join(entries)
    sentences = tokenize_sentences(full)

    total_words = len(re.findall(r"\w+", full))
    total_answers = max(1, len(entries))
    avg_answer_len = total_words / total_answers if total_answers else 0

    # filler detection
    total_fillers = sum(count_fillers(e) for e in entries)
    filler_rate = total_fillers / (total_words+1)

    # confidence: presence of confidence words, short hedging, modal verbs
    conf_kw_prop = proportion_keywords(full, CONFIDENCE_KEYWORDS)
    hedges = sum(full.lower().count(h) for h in ["maybe","perhaps","probably","might","could","sort of","kind of"]) 
    hedges_rate = hedges / (total_words+1)

    # sentiment proxy using positive/negative word overlap
    pos_prop = proportion_keywords(full, POSITIVE_WORDS)
    neg_prop = proportion_keywords(full, NEGATIVE_WORDS)
    sentiment_score = clamp01(pos_prop - neg_prop + 0.5)  # center at 0.5

    # technical accuracy: overlap between resume skills and transcript
    skills = [s.lower() for s in (resume_struct.get('skills') or [])]
    skill_hits = 0
    skill_words = set()
    for s in skills:
        for token in re.findall(r"\w+", s.lower()):
            if token:
                skill_words.add(token)
    if skill_words:
        skill_hits = sum(1 for w in re.findall(r"\w+", full.lower()) if w in skill_words)
    technical_prop = normalize_by_range(skill_hits, 0, max(5, len(skill_words)))

    # behavioral quality: use STAR keyword proportion and answer length consistency
    star_prop = proportion_keywords(full, STAR_KEYWORDS)
    # consistency: variance of answer lengths (lower variance -> more consistent)
    lengths = [len(re.findall(r"\w+", e)) for e in entries]
    if len(lengths) <= 1:
        length_var = 0.0
    else:
        mean = sum(lengths)/len(lengths)
        var = sum((x-mean)**2 for x in lengths)/len(lengths)
        length_var = var
    # map length_var to a 0..1 where low variance -> higher score
    length_score = 1.0 - normalize_by_range(length_var, 0, max(1.0, mean**2))

    # emotion stability: frequent emotional words or punctuation spikes
    exclaims = full.count('!')
    questions = full.count('?')
    punct_score = normalize_by_range(abs(exclaims - questions), 0, max(3, total_answers))
    emotion_stability = clamp01(1.0 - punct_score - (filler_rate*2))

    # clarity: penalize fillers and hedges, reward average answer length in reasonable band
    # reasonable band: avg_answer_len between 8 and 120 words
    len_score = normalize_by_range(avg_answer_len, 8, 120)
    clarity = clamp01(0.6*len_score + 0.3*(1 - filler_rate) + 0.1*sentiment_score)

    # confidence final: combine conf_kw_prop, hedges, sentiment and speaking length
    confidence = clamp01(0.5*conf_kw_prop + 0.2*len_score + 0.2*sentiment_score + 0.1*(1-hedges_rate))

    # answer structure: presence of enumerations and STAR usage
    enum_hits = len(re.findall(r"\b(first|second|third|one|two|three|finally|next)\b", full.lower()))
    enum_prop = normalize_by_range(enum_hits, 0, 5)
    answer_structure = clamp01(0.6*enum_prop + 0.4*star_prop)

    # behavioral quality combine star_prop and length consistency
    behavioral = clamp01(0.6*star_prop + 0.4*length_score)

    # overall score: weighted sum
    weights = {
        'confidence': 0.18,
        'clarity': 0.18,
        'technical': 0.2,
        'behavioral': 0.15,
        'emotion': 0.12,
        'structure': 0.17
    }
    scores = {
        'confidence': confidence,
        'clarity': clarity,
        'technical': technical_prop,
        'behavioral': behavioral,
        'emotion': emotion_stability,
        'structure': answer_structure
    }

    overall = 0.0
    for k,w in weights.items():
        overall += scores.get(k, 0.0) * w
    overall = clamp01(overall)

    # generate human-friendly strengths/weaknesses
    strengths = []
    weaknesses = []

    if scores['technical'] > 0.5:
        strengths.append('Demonstrated relevant technical keywords from resume.')
    else:
        weaknesses.append('Mention more technical skills and real examples tied to your resume.')

    if scores['confidence'] > 0.6:
        strengths.append('Confident delivery and use of assertive language.')
    else:
        weaknesses.append('Work on assertiveness — reduce hedging language like "maybe" or "probably".')

    if scores['clarity'] > 0.6:
        strengths.append('Clear and coherent answers.')
    else:
        weaknesses.append('Reduce filler words and speak in concise sentences to improve clarity.')

    if scores['behavioral'] > 0.55:
        strengths.append('Good use of structured storytelling (STAR).')
    else:
        weaknesses.append('Structure behavioral answers using Situation–Task–Action–Result.')

    if scores['emotion'] < 0.4:
        weaknesses.append('Work on emotional stability — try to avoid sudden tone/punctuation shifts.')
    else:
        strengths.append('Emotion remained steady across answers.')

    if scores['structure'] > 0.6:
        strengths.append('Answers included clear structure / enumerations.')
    else:
        weaknesses.append('Use signposting (first, then, finally) to structure answers.')

    recommended_courses = ["Communication Skills", "Interview Preparation: STAR Method"]
    if scores['technical'] < 0.6:
        recommended_courses.insert(0, "Technical Interview: System Design & Coding")

    summary = (
        f"Computed overall = {round(overall,2)}. "
        f"Confidence={round(confidence,2)}, Clarity={round(clarity,2)}, Technical={round(technical_prop,2)}, "
        f"Behavioral={round(behavioral,2)}, Emotion={round(emotion_stability,2)}, Structure={round(answer_structure,2)}."
    )

    return {
        'scores': scores,
        'overall': overall,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'recommended_courses': recommended_courses,
        'summary': summary,
        # debugging details (also returned so UI can display more granular info if desired)
        'meta': {
            'total_words': total_words,
            'total_answers': total_answers,
            'avg_answer_len': avg_answer_len,
            'filler_count': total_fillers,
            'skill_hits': skill_hits,
            'star_prop': round(star_prop,3),
            'pos_prop': round(pos_prop,3),
            'neg_prop': round(neg_prop,3)
        }
    }


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        users = load_users()
        u = request.form.get("username")
        p = request.form.get("password")
        if u in users and users[u]["password"] == p:
            session["user"] = u
            return redirect(url_for("upload_resume"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")


@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        users = load_users()
        u = request.form.get("username")
        p = request.form.get("password")
        if u in users:
            return render_template("signup.html", error="User exists")
        users[u] = {"password": p}
        save_users(users)
        return redirect(url_for("login"))
    return render_template("signup.html")


# ---------------------------------------------------------
# UPLOAD RESUME
# ---------------------------------------------------------
@app.route("/upload", methods=["GET","POST"])
def upload_resume():
    if request.method == "POST":
        file = request.files.get("resume")
        if not file or not file.filename.endswith(".pdf"):
            return render_template("upload.html", error="Upload PDF only")

        path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
        file.save(path)

        text = extract_resume_text(path)
        session["resume_text"] = text
        session["resume_structured"] = extract_resume_fields(text)
        session["resume_summary"] = " ".join(text.split()[:200])

        return redirect(url_for("resume_summary"))

    return render_template("upload.html")


# ---------------------------------------------------------
# RESUME SUMMARY
# ---------------------------------------------------------
@app.route("/resume_summary", methods=["GET","POST"])
def resume_summary():
    if "resume_text" not in session:
        return redirect(url_for("upload_resume"))

    data = session["resume_structured"]

    if request.method == "POST":
        prefs = {
            "role": request.form.get("role"),
            "interview_type": request.form.get("interview_type"),
            "question_count": int(request.form.get("question_count", 8))
        }

        session["preferences"] = prefs
        session["questions"] = generate_personalized_questions(
            session["resume_text"], prefs["interview_type"], prefs["question_count"]
        )
        session["clips"] = []
        session["transcript"] = []

        return redirect(url_for("interview"))

    return render_template("resume_summary.html", data=data, summary=session["resume_summary"])


# ---------------------------------------------------------
# INTERVIEW PAGE
# ---------------------------------------------------------
@app.route("/interview")
def interview():
    return render_template("interview.html", questions=session.get("questions", []))


# ---------------------------------------------------------
# SAVE VIDEO CLIP
# ---------------------------------------------------------
@app.route("/upload_clip", methods=["POST"])
def upload_clip():
    file = request.files.get("clip")
    idx = request.form.get("qindex", "0")

    user = session.get("user", "candidate")
    dest = os.path.join(VIDEO_FOLDER, user)
    os.makedirs(dest, exist_ok=True)

    filename = f"{user}_q{idx}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.webm"
    file.save(os.path.join(dest, filename))

    clips = session.get("clips", [])
    clips.append({"filename": filename, "qindex": idx})
    session["clips"] = clips

    return jsonify({"status": "ok"})


# ---------------------------------------------------------
# SAVE TRANSCRIPT
# ---------------------------------------------------------
@app.route("/save_transcript", methods=["POST"])
def save_transcript():
    data = request.get_json() or {}
    session["transcript"] = data.get("transcript", [])
    return jsonify({"status": "ok"})


# ---------------------------------------------------------
# FEEDBACK ANALYSIS (IMPROVED)
# ---------------------------------------------------------
@app.route("/feedback_analysis", methods=["POST"])
def feedback_analysis():
    candidate = session.get("user", "Candidate")
    timestamp = datetime.utcnow().isoformat()

    transcript = session.get("transcript", [])
    clips = session.get("clips", [])

    # perform deterministic analysis based on transcript + resume
    resume_struct = session.get('resume_structured', {})
    analysis = analyze_transcript(transcript, resume_struct)

    final = {
        "candidate_name": candidate,
        "timestamp": timestamp,
        "transcript": transcript,
        "clips": clips,
    }

    # map analysis scores into final structure expected by UI
    final['confidence'] = round(analysis['scores']['confidence'], 2)
    final['clarity'] = round(analysis['scores']['clarity'], 2)
    final['technical_accuracy'] = round(analysis['scores']['technical'], 2)
    final['behavioral_quality'] = round(analysis['scores']['behavioral'], 2)
    final['emotion_stability'] = round(analysis['scores']['emotion'], 2)
    final['answer_structure'] = round(analysis['scores']['structure'], 2)

    final['overall_score'] = round(analysis['overall'], 2)

    final['strengths'] = analysis['strengths']
    final['weaknesses'] = analysis['weaknesses']
    final['recommended_courses'] = analysis['recommended_courses']
    final['summary'] = analysis['summary']

    # attach meta for debugging / UI inspection
    final['meta'] = analysis.get('meta', {})

    session['final_feedback'] = final
    return jsonify(final)


# ---------------------------------------------------------
# FEEDBACK PAGE
# ---------------------------------------------------------
@app.route("/feedback")
def feedback():
    return render_template("feedback.html", data=session.get("final_feedback", {}))


# ---------------------------------------------------------
# RADAR CHART (100% FIXED)
# ---------------------------------------------------------
def create_radar_chart(m):
    labels = ["Confidence","Clarity","Technical","Behavioral","Emotion","Structure","Overall"]
    vals = [
        float(m.get("confidence",0)),
        float(m.get("clarity",0)),
        float(m.get("technical_accuracy",0)),
        float(m.get("behavioral_quality",0)),
        float(m.get("emotion_stability",0)),
        float(m.get("answer_structure",0)),
        float(m.get("overall_score",0))
    ]

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    vals_closed = vals + vals[:1]
    angles_closed = angles.tolist() + angles[:1].tolist()

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles_closed, vals_closed, linewidth=2)
    ax.fill(angles_closed, vals_closed, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0,1)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)

    buf.seek(0)
    return buf


# ---------------------------------------------------------
# PDF REPORT (FINAL FIXED VERSION)
# ---------------------------------------------------------
@app.route("/download_report")
def download_report():
    data = session.get("final_feedback", {})
    buf = BytesIO()

    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40)

    styles = getSampleStyleSheet()
    story = []

    # --- LOGO ---
    try:
        logo = ImageReader(LOGO_PATH)
        story.append(RLImage(logo, width=150, height=60))
    except:
        story.append(Paragraph("IntervYou", styles["Title"]))

    story.append(Spacer(1, 12))

    # --- HEADER ---
    story.append(Paragraph("<b>Interview Feedback Report</b>", styles["Heading1"]))
    story.append(Paragraph(f"<b>Candidate:</b> {data.get('candidate_name')}", styles["Normal"]))
    story.append(Paragraph(f"<b>Date:</b> {data.get('timestamp')}", styles["Normal"]))
    story.append(Spacer(1, 20))

    # --- SCORE BADGE ---
    story.append(Paragraph(
        f"<para alignment='center'><font size=16><b>⭐ Overall Score: {data.get('overall_score')}</b></font></para>",
        styles["Normal"]
    ))
    story.append(Spacer(1, 20))

    # --- RADAR CHART ---
    story.append(Paragraph("<b>Performance Radar Chart</b>", styles["Heading2"]))
    story.append(Spacer(1, 10))
    try:
        radar = create_radar_chart(data)
        radar.seek(0)
        story.append(RLImage(radar, width=300, height=300))
    except Exception as e:
        story.append(Paragraph(f"Radar ERROR: {e}", styles["Normal"]))

    story.append(Spacer(1, 20))

    # --- TABLE ---
    table_data = [
        ["Metric", "Score"],
        ["Confidence", data.get("confidence")],
        ["Clarity", data.get("clarity")],
        ["Technical Accuracy", data.get("technical_accuracy")],
        ["Behavioral Quality", data.get("behavioral_quality")],
        ["Emotion Stability", data.get("emotion_stability")],
        ["Answer Structure", data.get("answer_structure")]
    ]

    t = Table(table_data, colWidths=[200, 100])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eef2ff")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ]))

    story.append(t)
    story.append(Spacer(1, 20))

    # --- Strengths ---
    story.append(Paragraph("<b>Strengths</b>", styles["Heading2"]))
    for s in data.get("strengths", []):
        story.append(Paragraph("• " + s, styles["Normal"]))
    story.append(Spacer(1, 12))

    # --- Weaknesses ---
    story.append(Paragraph("<b>Areas to Improve</b>", styles["Heading2"]))
    for w in data.get("weaknesses", []):
        story.append(Paragraph("• " + w, styles["Normal"]))
    story.append(Spacer(1, 12))

    # --- Courses ---
    story.append(Paragraph("<b>Recommended Courses</b>", styles["Heading2"]))
    for c in data.get("recommended_courses", []):
        story.append(Paragraph("• " + c, styles["Normal"]))
    story.append(Spacer(1, 12))

    # --- SUMMARY ---
    story.append(Paragraph("<b>Interview Summary</b>", styles["Heading2"]))
    story.append(Paragraph(data.get("summary", "Good performance"), styles["Normal"]))

    doc.build(story)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="IntervYou_Report.pdf")


# ---------------------------------------------------------
# LOGOUT
# ---------------------------------------------------------
@app.route("/logout")
def logout():
    session.clear()
    return render_template("logout_page.html")


# ---------------------------------------------------------
# RUN APP
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
