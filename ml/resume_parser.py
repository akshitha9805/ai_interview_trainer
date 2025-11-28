# ml/resume_parser.py
import re
from typing import List, Dict

SECTION_HEADERS = [
    "technical skills", "technical skill", "skills", "tools & platforms",
    "core concepts", "personal skills", "professional summary",
    "academic qualifications", "academic qualification", "projects",
    "project highlight", "project highlights", "certifications",
    "campus engagement", "personal details", "experience"
]


def _split_sections(text: str) -> Dict[str, str]:
    """
    Naive but effective: find headers in text (case-insensitive),
    split resume into sections by header lines.
    """
    lines = text.splitlines()
    idxs = []
    for i, ln in enumerate(lines):
        low = ln.strip().lower()
        for h in SECTION_HEADERS:
            if low.startswith(h):
                idxs.append((i, h))
                break

    sections = {}
    if not idxs:
        sections["main"] = text
        return sections

    # collect slices
    for k in range(len(idxs)):
        start = idxs[k][0] + 1
        header = idxs[k][1]
        end = idxs[k + 1][0] if k + 1 < len(idxs) else len(lines)
        content = "\n".join(lines[start:end]).strip()
        sections[header] = content
    return sections


def _extract_bullets(section_text: str) -> List[str]:
    if not section_text:
        return []
    bullets = re.findall(r"(?:•|-|\*|\u2022)?\s*([A-Z][A-Za-z0-9 /.+#\-]{2,})", section_text)
    if bullets:
        cleaned = []
        for b in bullets:
            s = re.sub(r"\s{2,}", " ", b).strip()
            if len(s) > 1 and s.lower() not in ("skills",):
                cleaned.append(s)
        return list(dict.fromkeys(cleaned))  # dedupe preserving order
    # fallback: comma separated
    parts = [p.strip() for p in re.split(r"[,\n/;]", section_text) if len(p.strip()) > 1]
    return parts[:50]


def extract_resume_data(text: str) -> Dict:
    t = re.sub(r"\r\n", "\n", text or "")
    t = re.sub(r"\n{2,}", "\n\n", t)

    # email and phone
    email = None
    phone = None
    m = re.search(r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})", t)
    if m:
        email = m.group(1)
    m2 = re.search(r"(\+?\d{1,3}[-.\s]?\d{2,4}[-.\s]?\d{5,8})", t)
    if m2:
        phone = m2.group(1)

    # split into sections
    sections = _split_sections(t)

    # skills
    skills = []
    for key in ("technical skills", "skills", "tools & platforms", "core concepts"):
        if key in sections:
            skills = _extract_bullets(sections[key])
            break

    # education
    education = []
    for key in ("academic qualifications", "academic qualification"):
        if key in sections:
            education = _extract_bullets(sections[key])
            break

    # projects
    projects = []
    for key in ("project highlight", "project highlights", "projects"):
        if key in sections:
            projects = _extract_bullets(sections[key])
            break

    # experience
    experience = []
    if "professional summary" in sections:
        experience.append(sections["professional summary"].strip())
    if "experience" in sections:
        experience.extend([p.strip() for p in sections["experience"].split("\n") if p.strip()])

    # ---------- NAME DETECTION ----------
    name = None
    if "personal details" in sections:
        nm = re.search(r"Name[:\s]*([A-Z][A-Za-z\s]{2,40})", sections["personal details"])
        if nm:
            name = nm.group(1).strip()

    if not name:
        # check top few lines (skip lines with ':', numbers, or '@')
        lines = [ln.strip() for ln in t.strip().splitlines() if ln.strip()]
        for ln in lines[:5]:
            if ":" in ln or "@" in ln or re.search(r"\d", ln):
                continue
            if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$", ln):
                name = ln.strip()
                break

    if name:
        # remove stray words like "Location", "Address", etc.
        name = re.sub(r"\b(Location|Address|Contact|Email)\b.*", "", name, flags=re.I).strip()

    # summary
    summary = sections.get("professional summary") or "\n\n".join(t.split("\n\n")[:2])
    summary = summary.strip()

    return {
        "name": name or "",
        "email": email or "",
        "phone": phone or "",
        "skills": skills,
        "education": education,
        "projects": projects,
        "experience": experience,
        "summary": summary,
        "raw_sections": sections
    }
