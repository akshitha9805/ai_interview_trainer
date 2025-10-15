def generate_questions_from_text(text: str, total_questions: int = 6):
    """
    Simple heuristic-based question generator (stub).
    Returns list of questions (strings). Mix of HR + technical.
    Replace with a proper NLP model later.
    """
    text = (text or "").lower()
    # basic HR questions
    hr = [
        "Tell me about yourself and your background.",
        "Why do you want to work at this company?",
        "Describe a challenge you faced and how you handled it.",
        "Where do you see yourself in 3 years?"
    ]

    # extract candidate keywords (simple)
    import re
    tokens = re.split(r'[^a-z0-9]+', text)
    tokens = [t for t in tokens if len(t) > 2]
    keywords = []
    for t in tokens:
        if t not in keywords:
            keywords.append(t)
        if len(keywords) >= 8:
            break

    tech_templates = [
        lambda k: f"Explain a project where you used {k}. What was your role and outcome?",
        lambda k: f"How would you solve a performance problem involving {k}?",
        lambda k: f"What trade-offs did you consider when designing a system using {k}?"
    ]

    questions = []
    # ensure a few HR
    questions.extend(hr[:min(3, total_questions)])
    # technical from keywords
    tech_count = max(0, total_questions - len(questions))
    i = 0
    for k in keywords:
        if i >= tech_count:
            break
        questions.append(tech_templates[i % len(tech_templates)](k))
        i += 1

    # pad with general technical questions
    common = [
        "Describe your debugging process for a tough bug.",
        "How do you design scalable systems?",
        "Explain a data structure you used to optimize performance."
    ]
    idx = 0
    while len(questions) < total_questions:
        questions.append(common[idx % len(common)])
        idx += 1

    return questions
