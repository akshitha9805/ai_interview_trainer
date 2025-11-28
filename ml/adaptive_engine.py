"""
Adaptive Behavioral Interview Engine (Virtual HR)

🎯 Prompt: Adaptive Behavioral Interview (Virtual HR)
You are a Virtual HR Interviewer conducting the Behavioral Phase of an AI interview.
Your goal is to interact naturally, asking relevant behavioral questions based on the user’s resume and preferences collected earlier.

Behavior Rules:
- Start by greeting and asking for a short self-introduction.
- Analyze each response using ML insights — emotion, confidence, and relevance.
- Adapt your next question dynamically:
  High confidence → Ask deeper or company/role-specific leadership questions.
  Moderate confidence → Ask teamwork, adaptability, or communication-based questions.
  Low confidence → Ask supportive, motivational, or self-reflective questions.
- Never repeat questions.
- Personalize questions using the resume context (role, company, skills, projects).
- Maintain a natural HR tone and summarize session performance at the end.
"""

import random

class AdaptiveQuestionEngine:
    def __init__(self, resume_data=None, preferences=None):
        self.resume = resume_data or {}
        self.prefs = preferences or {}
        self.asked = []
        self._greeted = False

    # --- Helper to personalize questions ---
    def _make_personalized(self):
        role = (self.prefs.get("role") or "").lower()
        company = (self.prefs.get("company") or "").lower()

        questions = []
        if "data" in role:
            questions.append("Tell me about a time when you analyzed complex data to solve a problem.")
        if "engineer" in role:
            questions.append("Describe a technical challenge you faced and how you overcame it.")
        if "google" in company:
            questions.append("Google values innovation — can you share a time you brought a new idea to your team?")
        if "infosys" in company:
            questions.append("Infosys emphasizes teamwork — describe a situation where you collaborated effectively with others.")

        # Add skill-based questions
        for skill in self.resume.get("skills", [])[:3]:
            questions.append(f"Can you describe how you used {skill} in a real-world project?")

        # Default fallback questions
        questions += [
            "Describe a time you received feedback and how you used it to improve.",
            "Tell me about a project you are most proud of and why.",
            "Share a situation where you faced a setback and how you handled it."
        ]

        # Deduplicate
        return list(dict.fromkeys(questions))

    # --- Adaptive next question logic ---
    def get_next_question(self, last_answer=None, confidence=0.7):
        if not self._greeted:
            self._greeted = True
            return {
                "hr_response": "Hello Amara Suchitra, welcome to your interview simulation!",
                "next_question": "Could you please introduce yourself briefly before we begin?",
                "confidence_score": 0.7,
                "emotion": "neutral"
            }

        # Determine confidence-driven tone
        if confidence >= 0.8:
            emotion = "positive"
            hr_response = "Great answer! You explained that with confidence."
            q_type = "deep"
        elif confidence >= 0.5:
            emotion = "neutral"
            hr_response = "Thanks for sharing. That was clear and informative."
            q_type = "moderate"
        else:
            emotion = "supportive"
            hr_response = "No worries — take your time. Let's try another one."
            q_type = "simple"

        # Pick next question
        all_qs = self._make_personalized()
        available = [q for q in all_qs if q not in self.asked]
        if not available:
            return {
                "hr_response": "That was a great session! The interview is now complete.",
                "next_question": "",
                "confidence_score": confidence,
                "emotion": emotion
            }

        if q_type == "deep":
            next_q = random.choice(available)
        elif q_type == "moderate":
            next_q = random.choice(available)
        else:
            next_q = random.choice(available[:2])

        self.asked.append(next_q)

        return {
            "hr_response": hr_response,
            "next_question": next_q,
            "confidence_score": confidence,
            "emotion": emotion
        }
