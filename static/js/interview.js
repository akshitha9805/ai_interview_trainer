<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Upload Resume — IntervYou</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>

<body class="upload-page">

  <!-- Logo Top Center -->
  <div style="text-align:center; margin-top:20px; margin-bottom:10px;">
      <img src="{{ url_for('static', filename='images/logo/interyou.jpeg') }}" 
           alt="IntervYou Logo"
           style="height:70px; object-fit:contain;">
  </div>

  <div class="page-card" style="display:flex; gap:40px; align-items:center; flex-wrap:wrap;">

    <!-- Left Section: Resume Upload -->
    <div style="flex:1; min-width:300px;">
      <h1 style="margin-bottom:6px;">Upload Your Resume</h1>
      <p>Please upload a PDF resume. Your data is processed securely on your device/server.</p>

      {% if error %}
        <div class="error">{{ error }}</div>
      {% endif %}

      <form method="post" action="{{ url_for('upload_resume') }}" enctype="multipart/form-data">
        <div class="upload-box">
          <input type="file" id="resume" name="resume" accept=".pdf" required>
        </div>

        <div style="margin-top:18px;">
          <button class="btn" type="submit">Start Interview</button>
          <a class="btn outline" href="{{ url_for('logout') }}">Logout</a>
        </div>
      </form>
    </div>

    <!-- Right Section: Illustration -->
    <div style="flex:1; text-align:center; min-width:300px;">
      <img src="{{ url_for('static', filename='images/illustrations/resume_check.png') }}"
           alt="Resume Illustration"
           style="width:100%; max-width:360px; border-radius:16px; box-shadow:var(--shadow);">
    </div>

  </div>

</body>
</html>
