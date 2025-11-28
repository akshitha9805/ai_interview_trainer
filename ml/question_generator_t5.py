# ml/question_generator_t5.py

from transformers import pipeline
import random
import json

class AdaptiveQuestionGenerator:
    """
    Generates realistic interview questions based on resume summary, skills,
    projects, company, and role using a FLAN-T5 or GPT-style model.
    """

    def __init__(self, model_name="google/flan-t5-large"):
        try:
            self.generator = pipeline("text2text-generation", model=model_name)
        except Exception as e:
            print("⚠️ Warning: Could not load model, using fallback generator:", e)
            self.generator = None

    def build_prompt(self, resume_summary, skills, projects, company, role, interview_type, difficulty):
        """
        Build the adaptive interviewer prompt.
        """
        prompt = f"""
You are a professional HR interviewer for a digital interview platform called IntervYou.
Generate realistic, non-repetitive, adaptive interview questions based on the candidate’s profile.

🧾 Resume Summary: {resume_summary}
💼 Skills: {', '.join(skills[:10]) if skills else 'Not specified'}
📊 Projects: {', '.join(projects[:5]) if projects else 'Not specified'}
🏢 Target Company: {company}
🎯 Target Role: {role}
💬 Interview Type: {interview_type}
⚙ Difficulty Level: {difficulty}

Rules:
- Do not repeat questions.
- Maintain professional tone and natural language.
- Adapt questions according to difficulty.
- 70% questions should be technical, 30% behavioral.
- If role or company is given, tailor the context.

Return exactly 5 questions in JSON array format:
[
  {{"question_id": 1, "question_text": "string", "difficulty": "string", "type": "Technical/Behavioral"}}
]
"""
        return prompt

    def generate_questions(self, resume_data, preferences, total=8):
        """
        Generate questions using the language model or a simple fallback if unavailable.
        """
        summary = resume_data.get("summary", "")
        skills = resume_data.get("skills", [])
        projects = resume_data.get("projects", [])
        company = preferences.get("company", "a company")
        role = preferences.get("role", "a role")
        interview_type = preferences.get("interview_type", "Mixed")
        difficulty = preferences.get("difficulty", "Medium")

        prompt = self.build_prompt(summary, skills, projects, company, role, interview_type, difficulty)

        # ---- Primary: use FLAN-T5 if available ----
        if self.generator:
            try:
                output = self.generator(prompt, max_length=400, temperature=0.7, num_return_sequences=1)
                text = output[0]["generated_text"]

                # Try to parse model output as JSON
                try:
                    questions = json.loads(text)
                except Exception:
                    lines = [l.strip("-• ") for l in text.split("\n") if len(l.strip()) > 10]
                    questions = [
                        {
                            "question_id": i + 1,
                            "question_text": q,
                            "difficulty": difficulty,
                            "type": "Technical",
                        }
                        for i, q in enumerate(lines[:total])
                    ]
                return questions
            except Exception as e:
                print("⚠️ Model generation failed, fallback mode:", e)

        # ---- Fallback: simple heuristic question builder ----
        tech_templates = [
            lambda s: f"Explain a project where you applied {s}.",
            lambda s: f"What challenges did you face while working with {s}?",
            lambda s: f"How would you improve performance using {s}?",
        ]
        behavioral = [
            "Describe a time you overcame a major challenge at work.",
            "How do you handle pressure and deadlines?",
            "Tell me about a situation where you worked in a team.",
        ]

        all_questions = []

        # 70% technical questions
        for i in range(int(total * 0.7)):
            skill = random.choice(skills or ["Python", "communication", "leadership"])
            q = random.choice(tech_templates)(skill)
            all_questions.append(
                {
                    "question_id": i + 1,
                    "question_text": q,
                    "difficulty": difficulty,
                    "type": "Technical",
                }
            )

        # 30% behavioral questions
        for i in range(int(total * 0.3)):
            q = random.choice(behavioral)
            all_questions.append(
                {
                    "question_id": len(all_questions) + 1,
                    "question_text": q,
                    "difficulty": difficulty,
                    "type": "Behavioral",
                }
            )

        random.shuffle(all_questions)
        return all_questions[:total]
