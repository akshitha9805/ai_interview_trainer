# ---------------------------------------------
#  IntervYou – Final Stable app.py (Radar Chart FIXED + Logo FIXED)
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
# FEEDBACK ANALYSIS
# ---------------------------------------------------------
@app.route("/feedback_analysis", methods=["POST"])
def feedback_analysis():
    candidate = session.get("user", "Candidate")
    timestamp = datetime.utcnow().isoformat()

    transcript = session.get("transcript", [])
    clips = session.get("clips", [])

    def rnd(): return round(random.uniform(0.6, 0.9), 2)

    final = {
        "candidate_name": candidate,
        "timestamp": timestamp,

        # 🔥 Add them here so feedback page can show them
        "transcript": transcript,
        "clips": clips,
    }

    # scores
    final["confidence"] = rnd()
    final["clarity"] = rnd()
    final["technical_accuracy"] = rnd()
    final["behavioral_quality"] = rnd()
    final["emotion_stability"] = rnd()
    final["answer_structure"] = rnd()

    vals = [
        final["confidence"], final["clarity"], final["technical_accuracy"],
        final["behavioral_quality"], final["emotion_stability"], final["answer_structure"],
    ]
    final["overall_score"] = round(sum(vals)/6, 2)

    final["strengths"] = [
        "Good technical knowledge.",
        "Good behavioral storytelling.",
        "Answers were structured.",
    ]
    final["weaknesses"] = [
        "Work on voice projection.",
        "Improve clarity.",
        "Control pace while answering.",
    ]

    # course recommendation
    resume_struct = session.get("resume_structured", {})
    skills_list = resume_struct.get("skills", [])
    final["recommended_courses"] = ["Python for Everybody", "Communication Skills"]

    # summary
    final["summary"] = (
        f"The candidate scored {final['overall_score']}. "
        f"Strengths include {', '.join(final['strengths'])}. "
        f"Areas to improve: {', '.join(final['weaknesses'])}."
    )

    session["final_feedback"] = final
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
