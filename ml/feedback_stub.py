def compute_feedback(interview_data: dict):
    """
    interview_data expected shape:
    {
      "recordings": [
         {"q": "...", "transcript": "...", "duration": 12.3},
         ...
      ],
      "meta": { ... }
    }
    Returns a feedback dict with simple heuristics.
    """
    recordings = interview_data.get('recordings', [])
    filler_list = ['um','uh','like','you know','so','actually','basically']
    total_words = 0
    total_time = 0.0001
    total_fillers = 0
    relevance_scores = []

    for r in recordings:
        t = (r.get('transcript') or "").lower()
        words = [w for w in t.split() if w.strip()]
        total_words += len(words)
        total_time += float(r.get('duration', 0.0001))
        # filler count
        for f in filler_list:
            total_fillers += t.count(f)
        # simple relevance: presence of question keywords
        q = (r.get('q') or "").lower()
        qwords = [w for w in q.split() if len(w)>3]
        hits = 0
        for w in qwords:
            if w in t:
                hits += 1
        rel = (hits / max(1, len(qwords))) if qwords else 0
        relevance_scores.append(rel)

    n = max(1, len(recordings))
    avg_rel = sum(relevance_scores)/n if relevance_scores else 0
    wpm = int(total_words / (total_time/60))
    posture = interview_data.get('meta', {}).get('posture', 70)  # if camera-based posture given
    # confidence heuristic
    conf = max(30, min(95, int(60 + (wpm-110)/2 - total_fillers*2 + (avg_rel*100-50)/2)))

    return {
        "content_relevance_pct": int(avg_rel*100),
        "filler_count": total_fillers,
        "wpm": wpm,
        "posture_pct": posture,
        "overall_confidence_pct": conf
    }
