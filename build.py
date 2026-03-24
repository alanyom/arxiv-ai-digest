import httpx
import xml.etree.ElementTree as ET
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime, timezone
import re

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

ARXIV_API = "https://export.arxiv.org/api/query"
NS = "{http://www.w3.org/2005/Atom}"


# Relevance filter heuristic rules (swap for LLM scorer later)

BROAD_KEYWORDS = [
    "language model", "llm", "transformer", "reasoning", "agent", "alignment",
    "fine-tun", "instruction", "in-context", "few-shot", "zero-shot", "prompt",
    "reinforcement learning from human feedback", "rlhf", "reward model",
    "diffusion model", "multimodal", "vision-language", "foundation model",
    "scaling", "emergent", "chain-of-thought", "retrieval-augmented", "rag",
    "benchmark", "evaluation", "safety", "interpretability", "mechanistic",
    "neural network", "deep learning", "representation learning", "pretraining",
    "self-supervised", "attention mechanism", "generative", "inference",
]

# Phrases that suggest very niche
NICHE_PENALTIES = [
    "specific enzyme", "crystal structure", "seismic", "satellite imagery",
    "ancient manuscript", "cytology", "histopathology", "sonar", "lidar point cloud",
    "hyperspectral", "quantum circuit", "genome", "proteomics", "drug discovery",
    "electroencephalog", "fmri", "radiology", "ct scan", "optical coherence",
]

def relevance_score(title: str, summary: str) -> float:
    """
    Returns a float 0.0–1.0. Papers >= 0.35 are shown in the main feed.
    Replace this function body with an LLM call when ready to upgrade.
    """
    text = (title + " " + summary).lower()
    score = 0.0

    for kw in BROAD_KEYWORDS:
        if kw in text:
            score += 0.12

    for kw in NICHE_PENALTIES:
        if kw in text:
            score -= 0.3

    return min(max(score, 0.0), 1.0)


# arXiv fetcher


def fetch_papers(query: str, max_results: int = 60) -> list[dict]:
    params = {
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    }
    resp = httpx.get(ARXIV_API, params=params, timeout=15)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    papers = []

    for entry in root.findall(f"{NS}entry"):
        title = (entry.findtext(f"{NS}title") or "").strip().replace("\n", " ")
        summary = (entry.findtext(f"{NS}summary") or "").strip().replace("\n", " ")
        summary = re.sub(r"\s+", " ", summary)
        arxiv_id = (entry.findtext(f"{NS}id") or "").strip()
        published_raw = entry.findtext(f"{NS}published") or ""
        authors = [a.findtext(f"{NS}name") for a in entry.findall(f"{NS}author")]

        # Categories
        categories = [
            t.get("term", "")
            for t in entry.findall(f"{NS}category")
        ]

        # Parse date
        try:
            pub_dt = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
            days_ago = (datetime.now(timezone.utc) - pub_dt).days
            pub_label = f"{days_ago}d ago" if days_ago < 30 else pub_dt.strftime("%b %Y")
        except Exception:
            pub_label = published_raw[:10]

        # Affiliation heuristic from author names (arXiv doesn't provide it directly)
        affil = authors[0] if authors else "Unknown"

        score = relevance_score(title, summary)
        short_summary = summary[:220] + "…" if len(summary) > 220 else summary

        # Badges
        badge = None
        if days_ago <= 3:
            badge = ("new", "New")
        elif score >= 0.6:
            badge = ("hot", "Popular topic")

        papers.append({
            "title": title,
            "url": arxiv_id,
            "summary": short_summary,
            "authors": authors[:3],
            "categories": categories[:2],
            "pub_label": pub_label,
            "score": score,
            "badge": badge,
        })

    return papers


# Essential / foundational papers — hardcoded, curated list


ESSENTIAL_PAPERS = [
    {
        "year": "2017",
        "title": "Attention Is All You Need",
        "authors": "Vaswani et al. · Google Brain",
        "url": "https://arxiv.org/abs/1706.03762",
        "why": "Introduced the Transformer. Every modern LLM architecture descends from this paper.",
        "tags": ["Architecture", "Attention"],
    },
    {
        "year": "2018",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": "Devlin et al. · Google",
        "url": "https://arxiv.org/abs/1810.04805",
        "why": "Showed that bidirectional pretraining then fine-tuning could dominate NLP benchmarks across the board.",
        "tags": ["Pretraining", "NLP"],
    },
    {
        "year": "2020",
        "title": "Language Models are Few-Shot Learners (GPT-3)",
        "authors": "Brown et al. · OpenAI",
        "url": "https://arxiv.org/abs/2005.14165",
        "why": "Established that scale unlocks emergent in-context learning without any fine-tuning.",
        "tags": ["Scaling", "In-context learning"],
    },
    {
        "year": "2022",
        "title": "Training Language Models to Follow Instructions (InstructGPT)",
        "authors": "Ouyang et al. · OpenAI",
        "url": "https://arxiv.org/abs/2203.02155",
        "why": "The foundational RLHF paper. Shaped how every modern AI assistant is trained.",
        "tags": ["RLHF", "Alignment"],
    },
    {
        "year": "2022",
        "title": "Chain-of-Thought Prompting Elicits Reasoning in LLMs",
        "authors": "Wei et al. · Google",
        "url": "https://arxiv.org/abs/2201.11903",
        "why": "Simple idea with enormous impact: ask the model to think step by step and reasoning quality jumps dramatically.",
        "tags": ["Reasoning", "Prompting"],
    },
    {
        "year": "2022",
        "title": "Training Compute-Optimal LLMs (Chinchilla)",
        "authors": "Hoffmann et al. · DeepMind",
        "url": "https://arxiv.org/abs/2203.15556",
        "why": "Proved most LLMs were undertrained on data relative to compute. Reshaped how labs plan training runs.",
        "tags": ["Scaling Laws", "Training"],
    },
    {
        "year": "2021",
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "authors": "Hu et al. · Microsoft",
        "url": "https://arxiv.org/abs/2106.09685",
        "why": "Made fine-tuning massive models practical on consumer hardware. Now the default fine-tuning technique.",
        "tags": ["Fine-tuning", "Efficiency"],
    },
    {
        "year": "2017",
        "title": "Proximal Policy Optimization Algorithms (PPO)",
        "authors": "Schulman et al. · OpenAI",
        "url": "https://arxiv.org/abs/1707.06347",
        "why": "The RL algorithm underlying most RLHF training pipelines. Essential for understanding how LLMs are aligned.",
        "tags": ["Reinforcement Learning"],
    },
    {
        "year": "2020",
        "title": "Retrieval-Augmented Generation (RAG)",
        "authors": "Lewis et al. · Facebook AI",
        "url": "https://arxiv.org/abs/2005.11401",
        "why": "Introduced the paradigm of grounding LLM outputs with retrieved documents. Foundational for production AI apps.",
        "tags": ["RAG", "Retrieval"],
    },
    {
        "year": "2023",
        "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
        "authors": "Schick et al. · Meta AI",
        "url": "https://arxiv.org/abs/2302.04761",
        "why": "Showed LLMs can learn to call external APIs autonomously. Seed paper for the AI agent paradigm.",
        "tags": ["Agents", "Tool use"],
    },
]


# Routes


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    query = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL"
    all_papers = fetch_papers(query, max_results=60)
    featured = [p for p in all_papers if p["score"] >= 0.35]
    featured.sort(key=lambda p: p["score"], reverse=True)
    featured = featured[:15]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "papers": featured,
        "essential": ESSENTIAL_PAPERS,
        "total_fetched": len(all_papers),
        "total_shown": len(featured),
        "refreshed": datetime.now().strftime("%b %d, %Y · %H:%M"),
    })
