from flask import Flask, render_template, request, redirect

app = Flask(__name__)

# ---------- ROUTES ----------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # (Later: validate credentials here)
        return redirect('/upload')
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        # (Later: Save these details securely to a database)
        print(f"✅ New user created: {username}, {email}")
        return redirect('/login')
    return render_template('signup.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Handle resume upload later
        return redirect('/interview')
    return render_template('upload.html')

@app.route('/interview')
def interview():
    return render_template('interview.html')


# ---------- MAIN ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

    print("🚀 Starting Flask server...")
    app.run(debug=True)

