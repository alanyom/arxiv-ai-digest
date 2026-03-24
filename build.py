#!/usr/bin/env python3
"""
build.py — fetches HuggingFace Daily Papers (already curated by the community),
sorts by upvotes, and writes docs/index.html.

No API key needed. No Gemini. Completely free.
Run once daily via GitHub Actions cron.
"""

import json
from datetime import datetime, timezone
from pathlib import Path


import requests

# Config

HF_API_URL = "https://huggingface.co/api/daily_papers"
TOP_N = 10          # how many papers to show in "today's picks"
ESSENTIAL_JSON = Path("essential.json")
DOCS_DIR = Path("docs")
OUTPUT_HTML = DOCS_DIR / "index.html"

# Fetch

def fetch_hf_papers():
    resp = requests.get(HF_API_URL, params={"limit": 100}, timeout=30)
    resp.raise_for_status()
    papers = resp.json()
    # sort by upvotes descending
    papers.sort(key=lambda p: p.get("paper", {}).get("upvotes", 0), reverse=True)
    return papers


def extract(paper_obj):
    """Normalise the HF API response into a flat dict."""
    p = paper_obj.get("paper", {})
    authors_raw = p.get("authors", [])
    authors = [a.get("name", "") for a in authors_raw[:4]]
    if len(authors_raw) > 4:
        authors.append("et al.")
    arxiv_id = p.get("id", "")
    return {
        "title":     p.get("title", "Untitled"),
        "abstract":  p.get("abstract", "")[:400],
        "upvotes":   p.get("upvotes", 0),
        "authors":   ", ".join(authors),
        "published": (p.get("publishedAt") or "")[:10],
        "link":      f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "#",
        "pdf_link":  f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else "#",
        "hf_link":   f"https://huggingface.co/papers/{arxiv_id}" if arxiv_id else "#",
    }

# HTML helpers

def paper_card_html(p):
    return f"""
    <article class="card">
      <div class="card-meta">
        <span class="badge">NEW</span>
        <span class="date">{p['published']}</span>
        <span class="score-pill">▲ {p['upvotes']}</span>
      </div>
      <h3><a href="{p['hf_link']}" target="_blank" rel="noopener">{p['title']}</a></h3>
      <p class="tldr">{p['abstract']}</p>
      <p class="authors">{p['authors']}</p>
      <div class="card-links">
        <a href="{p['hf_link']}" target="_blank">HF ↗</a>
        <a href="{p['link']}" target="_blank">Abstract ↗</a>
        <a href="{p['pdf_link']}" target="_blank">PDF ↗</a>
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

# ── HTML template ─────────────────────────────────────────────────────────────

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
  .wordmark { font-family: 'DM Serif Display', serif; font-size: 1.8rem; letter-spacing: -0.5px; }
  .wordmark em { font-style: italic; color: var(--accent); }
  .tagline { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--muted); letter-spacing: 0.05em; margin-left: auto; }
  main { max-width: 900px; margin: 0 auto; padding: 3rem 1.5rem 6rem; }
  section { margin-bottom: 4rem; }
  .section-header { display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border); }
  .section-title { font-family: 'DM Serif Display', serif; font-size: 1.3rem; font-weight: 400; }
  .section-count { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: var(--muted); background: var(--surface2); padding: 0.2rem 0.5rem; border-radius: 3px; }
  .updated { margin-left: auto; font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--muted); }
  .card-grid { display: flex; flex-direction: column; gap: 1px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 1.4rem 1.6rem; transition: border-color 0.15s, background 0.15s; position: relative; }
  .card:hover { border-color: #3a3a3a; background: var(--surface2); }
  .card::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; border-radius: 4px 0 0 4px; background: var(--accent); opacity: 0; transition: opacity 0.15s; }
  .card:hover::before { opacity: 1; }
  .essential-card::before { background: var(--accent2); }
  .card-meta { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.6rem; flex-wrap: wrap; }
  .badge { font-family: 'DM Mono', monospace; font-size: 0.62rem; font-weight: 500; letter-spacing: 0.1em; padding: 0.15rem 0.5rem; border-radius: 2px; background: var(--accent); color: #0d0d0d; }
  .badge-classic { background: var(--accent2); }
  .date { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: var(--muted); }
  .score-pill { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--accent); margin-left: auto; }
  .card h3 { font-family: 'DM Serif Display', serif; font-size: 1.05rem; font-weight: 400; line-height: 1.35; margin-bottom: 0.5rem; }
  .card h3 a { color: var(--text); text-decoration: none; transition: color 0.1s; }
  .card h3 a:hover { color: var(--accent); }
  .tldr { font-size: 0.875rem; color: #b0b0a8; margin-bottom: 0.5rem; line-height: 1.5; }
  .authors { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--muted); margin-bottom: 0.75rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .card-links { display: flex; gap: 1rem; }
  .card-links a { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--accent); text-decoration: none; transition: opacity 0.1s; }
  .card-links a:hover { opacity: 0.7; }
  .essential-card .card-links a { color: var(--accent2); }
  footer { border-top: 1px solid var(--border); padding: 2rem 1.5rem; max-width: 900px; margin: 0 auto; font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--muted); display: flex; gap: 1rem; flex-wrap: wrap; }
  @media (max-width: 600px) { .wordmark { font-size: 1.4rem; } .card { padding: 1.1rem; } .tagline { display: none; } }
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
      <span class="section-count">$$recent_count papers</span>
      <span class="updated">updated $$updated</span>
    </div>
    <div class="card-grid">
$$recent_cards
    </div>
  </section>
  <section id="essential">
    <div class="section-header">
      <h2 class="section-title">Essential reading</h2>
      <span class="section-count">$$essential_count papers</span>
    </div>
    <div class="card-grid">
$$essential_cards
    </div>
  </section>
</main>
<footer>
  <span>papers sourced from HuggingFace Daily Papers</span>
  <span>·</span>
  <span>ranked by community upvotes</span>
  <span>·</span>
  <span>auto-updated 06:00 UTC daily</span>
</footer>
</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Fetching HuggingFace Daily Papers…")
    raw_papers = fetch_hf_papers()
    papers = [extract(p) for p in raw_papers[:TOP_N]]
    print(f"   Got {len(papers)} papers (top {TOP_N} by upvotes)")

    recent_cards = "".join(paper_card_html(p) for p in papers)

    print("Loading essential papers…")
    essentials = json.loads(ESSENTIAL_JSON.read_text()) if ESSENTIAL_JSON.exists() else []
    essential_cards = "".join(essential_card_html(ep) for ep in essentials)

    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    DOCS_DIR.mkdir(exist_ok=True)
    html = (HTML_TEMPLATE
        .replace("$$recent_cards", recent_cards)
        .replace("$$essential_cards", essential_cards)
        .replace("$$recent_count", str(len(papers)))
        .replace("$$essential_count", str(len(essentials)))
        .replace("$$updated", updated)
    )
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Written: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
