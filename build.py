"""
build.py — fetches recent arXiv AI papers, scores them with Gemini Flash (free tier),
and writes docs/index.html. Designed to run once daily via GitHub Actions cron.

Gemini usage is kept minimal:
  - One batched prompt covers ALL papers in a single API call
  - Uses gemini-1.5-flash (cheapest / free tier eligible)
  - ~60 papers/day → ~1 call/day → essentially free
"""

import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from string import Template

import requests

# Config

ARXIV_URL = "https://export.arxiv.org/api/query"
ARXIV_PARAMS = {
    "search_query": "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV OR cat:stat.ML",
    "sortBy": "submittedDate",
    "sortOrder": "descending",
    "max_results": 60,
}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"  # free tier; don't change to Pro
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

ESSENTIAL_JSON = Path("essential.json")
DOCS_DIR = Path("docs")
OUTPUT_HTML = DOCS_DIR / "index.html"

# arXiv fetch

NS = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_arxiv_papers():
    resp = requests.get(ARXIV_URL, params=ARXIV_PARAMS, timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    papers = []
    for entry in root.findall("atom:entry", NS):
        title = entry.find("atom:title", NS).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", NS).text.strip().replace("\n", " ")
        link_el = entry.find("atom:id", NS)
        link = link_el.text.strip() if link_el is not None else ""
        # convert abs → pdf link
        pdf_link = link.replace("/abs/", "/pdf/") + ".pdf"
        authors = [
            a.find("atom:name", NS).text
            for a in entry.findall("atom:author", NS)
        ]
        published = entry.find("atom:published", NS).text[:10]
        papers.append(
            {
                "title": title,
                "summary": summary[:600],  # trim for prompt size
                "link": link,
                "pdf_link": pdf_link,
                "authors": authors[:4],
                "published": published,
            }
        )
    return papers


# Gemini scoring

PROMPT_TEMPLATE = """\
You are an AI research curator. Below are {n} recent arXiv papers.
Score each paper on two axes (integers 1–10):
  • interest: how broadly interesting/impactful is this to the AI community?
  • niche: how niche/narrow is this? (10 = extremely niche, 1 = wide appeal)

Return ONLY a JSON array with objects: {{"idx": <int>, "interest": <int>, "niche": <int>, "tldr": "<one sentence>"}}
No markdown, no explanation, just the raw JSON array.

Papers:
{papers_block}
"""


def score_papers_with_gemini(papers):
    """Single Gemini call for all papers, minimises quota usage."""
    if not GEMINI_API_KEY:
        print("No GEMINI_API_KEY found; assigning default scores.")
        return [
            {"idx": i, "interest": 5, "niche": 5, "tldr": "Score unavailable (no API key)."}
            for i in range(len(papers))
        ]

    lines = []
    for i, p in enumerate(papers):
        lines.append(f"[{i}] {p['title']}\n    {p['summary'][:300]}")
    papers_block = "\n\n".join(lines)

    prompt = PROMPT_TEMPLATE.format(n=len(papers), papers_block=papers_block)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
    }

    resp = requests.post(GEMINI_URL, json=payload, timeout=60)
    resp.raise_for_status()
    raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    # strip markdown fences if model wraps anyway
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    scores = json.loads(raw)
    return scores


def select_top_papers(papers, scores, n=8):
    """Pick top-n by (interest - niche/2), filtering out very niche ones."""
    scored = []
    score_map = {s["idx"]: s for s in scores}
    for i, paper in enumerate(papers):
        s = score_map.get(i, {"interest": 5, "niche": 5, "tldr": ""})
        if s["niche"] >= 8:
            continue  # skip hyper-niche
        composite = s["interest"] - s["niche"] * 0.5
        scored.append((composite, paper, s))
    scored.sort(key=lambda x: -x[0])
    return scored[:n]


# HTML generation

def load_essential():
    if ESSENTIAL_JSON.exists():
        return json.loads(ESSENTIAL_JSON.read_text())
    return []


def paper_card_html(paper, score_info, badge="NEW"):
    authors_str = ", ".join(paper["authors"])
    if len(paper["authors"]) == 4:
        authors_str += " et al."
    tldr = score_info.get("tldr", "")
    interest = score_info.get("interest", "–")
    return f"""
    <article class="card">
      <div class="card-meta">
        <span class="badge">{badge}</span>
        <span class="date">{paper['published']}</span>
        <span class="score-pill">interest {interest}/10</span>
      </div>
      <h3><a href="{paper['link']}" target="_blank" rel="noopener">{paper['title']}</a></h3>
      <p class="tldr">{tldr}</p>
      <p class="authors">{authors_str}</p>
      <div class="card-links">
        <a href="{paper['link']}" target="_blank">Abstract ↗</a>
        <a href="{paper['pdf_link']}" target="_blank">PDF ↗</a>
      </div>
    </article>"""


def essential_card_html(ep):
    return f"""
    <article class="card essential-card">
      <div class="card-meta">
        <span class="badge badge-classic">CLASSIC</span>
        <span class="date">{ep.get('year', '')}</span>
      </div>
      <h3><a href="{ep.get('link', '#')}" target="_blank" rel="noopener">{ep['title']}</a></h3>
      <p class="tldr">{ep.get('why', '')}</p>
      <p class="authors">{ep.get('authors', '')}</p>
      <div class="card-links">
        <a href="{ep.get('link', '#')}" target="_blank">Read ↗</a>
      </div>
    </article>"""


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>arXiv Digest</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0d0d0d;
    --surface: #161616;
    --surface2: #1e1e1e;
    --border: #2a2a2a;
    --accent: #c8f04e;
    --accent2: #4ef0c8;
    --text: #e8e8e0;
    --muted: #888;
    --badge-new: #c8f04e;
    --badge-classic: #4ef0c8;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    line-height: 1.6;
    min-height: 100vh;
  }

  /* ── Header ── */
  header {
    border-bottom: 1px solid var(--border);
    padding: 2rem 0 1.5rem;
    position: sticky;
    top: 0;
    background: var(--bg);
    z-index: 10;
  }
  .header-inner {
    max-width: 900px;
    margin: 0 auto;
    padding: 0 1.5rem;
    display: flex;
    align-items: baseline;
    gap: 1rem;
    flex-wrap: wrap;
  }
  .wordmark {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    letter-spacing: -0.5px;
    color: var(--text);
  }
  .wordmark em {
    font-style: italic;
    color: var(--accent);
  }
  .tagline {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.05em;
    margin-left: auto;
  }

  /* ── Layout ── */
  main { max-width: 900px; margin: 0 auto; padding: 3rem 1.5rem 6rem; }

  section { margin-bottom: 4rem; }

  .section-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
  }
  .section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    font-weight: 400;
  }
  .section-count {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    background: var(--surface2);
    padding: 0.2rem 0.5rem;
    border-radius: 3px;
  }
  .updated {
    margin-left: auto;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
  }

  /* ── Cards ── */
  .card-grid { display: flex; flex-direction: column; gap: 1px; }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.4rem 1.6rem;
    transition: border-color 0.15s, background 0.15s;
    position: relative;
  }
  .card:hover {
    border-color: #3a3a3a;
    background: var(--surface2);
  }
  .card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 4px 0 0 4px;
    background: var(--accent);
    opacity: 0;
    transition: opacity 0.15s;
  }
  .card:hover::before { opacity: 1; }
  .essential-card::before { background: var(--accent2); }

  .card-meta {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.6rem;
    flex-wrap: wrap;
  }
  .badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    padding: 0.15rem 0.5rem;
    border-radius: 2px;
    background: var(--badge-new);
    color: #0d0d0d;
  }
  .badge-classic {
    background: var(--badge-classic);
  }
  .date {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
  }
  .score-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    margin-left: auto;
  }

  .card h3 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1.35;
    margin-bottom: 0.5rem;
  }
  .card h3 a {
    color: var(--text);
    text-decoration: none;
    transition: color 0.1s;
  }
  .card h3 a:hover { color: var(--accent); }

  .tldr {
    font-size: 0.875rem;
    color: #b0b0a8;
    margin-bottom: 0.5rem;
    line-height: 1.5;
  }
  .authors {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    margin-bottom: 0.75rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .card-links { display: flex; gap: 1rem; }
  .card-links a {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent);
    text-decoration: none;
    letter-spacing: 0.02em;
    transition: opacity 0.1s;
  }
  .card-links a:hover { opacity: 0.7; }
  .essential-card .card-links a { color: var(--accent2); }

  /* ── Footer ── */
  footer {
    border-top: 1px solid var(--border);
    padding: 2rem 1.5rem;
    max-width: 900px;
    margin: 0 auto;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
  }

  @media (max-width: 600px) {
    .wordmark { font-size: 1.4rem; }
    .card { padding: 1.1rem 1.1rem; }
    .tagline { display: none; }
  }
</style>
</head>
<body>

<header>
  <div class="header-inner">
    <div class="wordmark"><em>arxiv</em> digest</div>
    <div class="tagline">AI research · curated daily</div>
  </div>
</header>

<main>

  <section id="recent">
    <div class="section-header">
      <h2 class="section-title">Today's picks</h2>
      <span class="section-count">$recent_count papers</span>
      <span class="updated">updated $updated</span>
    </div>
    <div class="card-grid">
$recent_cards
    </div>
  </section>

  <section id="essential">
    <div class="section-header">
      <h2 class="section-title">Essential reading</h2>
      <span class="section-count">$essential_count papers</span>
    </div>
    <div class="card-grid">
$essential_cards
    </div>
  </section>

</main>

<footer>
  <span>built with arXiv API + Gemini Flash</span>
  <span>·</span>
  <span>papers scored for breadth &amp; impact</span>
  <span>·</span>
  <span>auto-updated 06:00 UTC daily</span>
</footer>

</body>
</html>"""


def write_html(recent_cards_html, essential_cards_html, recent_count, essential_count):
    DOCS_DIR.mkdir(exist_ok=True)
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = Template(HTML_TEMPLATE).substitute(
        recent_cards=recent_cards_html,
        essential_cards=essential_cards_html,
        recent_count=recent_count,
        essential_count=essential_count,
        updated=updated,
    )
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"✅ Written: {OUTPUT_HTML}")


# Main

def main():
    print("📡 Fetching arXiv papers…")
    papers = fetch_arxiv_papers()
    print(f"   Got {len(papers)} papers")

    print("🤖 Scoring with Gemini Flash (1 API call)…")
    scores = score_papers_with_gemini(papers)

    top = select_top_papers(papers, scores, n=8)
    print(f"   Selected {len(top)} papers")

    recent_cards = "".join(
        paper_card_html(paper, score_info) for _, paper, score_info in top
    )

    print("📚 Loading essential papers…")
    essentials = load_essential()
    essential_cards = "".join(essential_card_html(ep) for ep in essentials)

    write_html(recent_cards, essential_cards, len(top), len(essentials))


if __name__ == "__main__":
    main()
