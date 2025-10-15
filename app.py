from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os, json, re
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-random-key"
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

USER_DB = "users.json"

def load_users():
    if os.path.exists(USER_DB):
        with open(USER_DB, "r") as f:
            return json.load(f)
    return {}

def save_users(data):
    with open(USER_DB, "w") as f:
        json.dump(data, f, indent=2)

# Basic keyword-based question generator
def generate_questions_from_text(text, n_questions=8):
    text = (text or "").lower()
    # extract words, keep alphanum, remove small words
    words = re.findall(r"[a-z0-9]{3,}", text)
    keywords = []
    for w in words:
        if w not in keywords:
            keywords.append(w)
        if len(keywords) >= 20:
            break

    hr = [
        "Tell me about yourself and your background.",
        "Why do you want to work at this company?",
        "Describe a challenge you faced and how you handled it.",
        "Where do you see yourself in 3 years?"
    ]

    tech_templates = [
        lambda k: f"Explain a project where you used {k}. What was your role and the outcome?",
        lambda k: f"How would you solve a performance problem involving {k} in a production system?",
        lambda k: f"What trade-offs did you consider when using {k}?"
    ]

    qlist = []
    qlist.extend(hr)  # include HR by default

    # add technical prompts from keywords
    tech_count = 0
    for k in keywords:
        if tech_count >= 4:
            break
        if k in ("experience","skills","project","working","company"):
            continue
        qlist.append( tech_templates[tech_count % len(tech_templates)](k) )
        tech_count += 1

    # pad with common tech questions
    common_tech = [
        "Describe your debugging process for a tough bug.",
        "How do you design scalable systems?",
        "Explain a data structure you used to optimize performance.",
        "Walk me through the architecture of a system you built."
    ]
    idx = 0
    while len(qlist) < n_questions:
        qlist.append(common_tech[idx % len(common_tech)])
        idx += 1

    return qlist[:n_questions]


# ROUTES
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        users = load_users()
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        if username in users and users[username]["password"] == password:
            session["user"] = username
            return redirect(url_for("upload_resume"))
        return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")

@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        users = load_users()
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        if not username or not password:
            return render_template("signup.html", error="Please provide both username and password.")
        if username in users:
            return render_template("signup.html", error="User already exists.")
        users[username] = {"password": password}
        save_users(users)
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/upload", methods=["GET","POST"])
def upload_resume():
    if "user" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        f = request.files.get("resume")
        if not f:
            return render_template("upload.html", error="No file uploaded.")
        if not (f.filename.lower().endswith(".pdf")):
            return render_template("upload.html", error="Please upload a PDF file.")
        filename = secure_filename(f.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(path)

        # extract text using PyPDF2
        try:
            reader = PdfReader(path)
            pages = []
            for p in reader.pages:
                t = p.extract_text() or ""
                pages.append(t)
            resume_text = "\n".join(pages)
        except Exception as e:
            resume_text = ""
            print("PDF extract error:", e)

        # keep an extracted summary in session
        snippet = " ".join(resume_text.split()[:200])  # first ~200 words
        session["resume_text"] = resume_text
        session["resume_summary"] = snippet

        # create a default set of questions (8)
        session["questions"] = generate_questions_from_text(resume_text, n_questions=8)
        return redirect(url_for("setup_interview"))
    return render_template("upload.html")

@app.route("/setup", methods=["GET","POST"])
def setup_interview():
    if "user" not in session:
        return redirect(url_for("login"))
    resume_summary = session.get("resume_summary","")
    if request.method == "POST":
        # user can edit snippet and choose number of q
        snippet = request.form.get("snippet","").strip()
        n_q = int(request.form.get("n_questions", "8"))
        source_text = snippet or session.get("resume_text","")
        # regenerate questions based on snippet and desired count
        session["questions"] = generate_questions_from_text(source_text, n_questions=n_q)
        return redirect(url_for("interview"))
    return render_template("setup.html", resume_summary=resume_summary, default_questions=len(session.get("questions",[])))

@app.route("/interview")
def interview():
    if "user" not in session:
        return redirect(url_for("login"))
    # pass questions, but client will only display after camera permission
    questions = session.get("questions", [])
    return render_template("interview.html", questions=questions)

@app.route("/feedback", methods=["POST","GET"])
def feedback():
    if request.method == "POST":
        # accept client-side computed metrics or final recordings metadata
        session["feedback_data"] = request.get_json()
        return jsonify({"status":"ok"})
    data = session.get("feedback_data", {})
    return render_template("feedback.html", data=data)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

if __name__ == "__main__":
    # debug True for development
    app.run(host="0.0.0.0", port=5000, debug=True)
