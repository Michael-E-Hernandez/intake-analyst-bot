from pathlib import Path
import re
from functools import lru_cache


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYBOOK_DIR = PROJECT_ROOT / "playbooks"


def _tokenize(text):
    return set(
        re.findall(
            r"[a-zA-Z][a-zA-Z0-9_-]{2,}",
            str(text).lower(),
        )
    )


def _stage_section(text, stage):
    """
    Prefer the stage-specific section when a playbook contains
    '## Intake Methodology' and/or '## Analysis Methodology'.
    Falls back to the whole playbook.
    """
    stage = str(stage).strip().lower()

    heading = (
        "intake methodology"
        if stage == "intake"
        else "analysis methodology"
    )

    pattern = re.compile(
        rf"(?ims)^##\s+{re.escape(heading)}\s*$"
        rf"(.*?)(?=^##\s+|\Z)"
    )

    match = pattern.search(text)

    if match:
        title_match = re.search(
            r"(?m)^#\s+(.+?)\s*$",
            text,
        )
        title = (
            title_match.group(1).strip()
            if title_match
            else "Analyst Playbook"
        )
        return f"# {title}\n\n## {heading.title()}\n{match.group(1).strip()}"

    return text.strip()


@lru_cache(maxsize=1)
def load_playbooks():
    if not PLAYBOOK_DIR.exists():
        raise FileNotFoundError(
            "Analyst playbook directory was not found:\n"
            f"{PLAYBOOK_DIR}\n\n"
            "Create a playbooks folder under the project root."
        )

    files = sorted(PLAYBOOK_DIR.glob("*.md"))

    if not files:
        raise RuntimeError(
            "No analyst playbooks were found in:\n"
            f"{PLAYBOOK_DIR}"
        )

    playbooks = []

    for path in files:
        text = path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        playbooks.append(
            {
                "name": path.stem,
                "path": path,
                "text": text,
                "tokens": _tokenize(text),
            }
        )

    if not playbooks:
        raise RuntimeError(
            "Analyst playbook files exist, but all are empty."
        )

    return playbooks


def retrieve_playbooks(query, stage="analysis", top_k=3):
    """
    Lightweight local RAG retrieval.

    This intentionally retrieves analyst methodology, not business facts.
    It uses deterministic lexical similarity so the prototype does not
    require a vector database. The returned text is then injected into
    the LLM prompt as retrieved methodology context.
    """
    query_tokens = _tokenize(query)

    if not query_tokens:
        query_tokens = {"analysis"}

    scored = []

    for playbook in load_playbooks():
        section = _stage_section(
            playbook["text"],
            stage,
        )

        section_tokens = _tokenize(section)

        overlap = query_tokens & section_tokens

        # Weighted overlap: direct matches matter most.
        score = len(overlap)

        # Small boosts for common analytical intent families.
        intent_terms = {
            "compare": {"compare", "difference", "different", "versus", "across", "vary"},
            "diagnose": {"why", "driver", "drivers", "cause", "issue", "problem", "dissatisfaction", "unhappy"},
            "trend": {"trend", "time", "change", "changed", "month", "year", "period"},
            "segment": {"segment", "group", "groups", "cohort", "demographic", "population", "age"},
            "understand": {"understand", "describe", "profile", "characterize", "who"},
            "evaluate": {"evaluate", "performance", "rate", "goal", "target", "kpi"},
        }

        query_lower = str(query).lower()
        name_lower = playbook["name"].lower()

        for family, terms in intent_terms.items():
            if any(term in query_lower for term in terms):
                if family in name_lower:
                    score += 4
                if any(term in section_tokens for term in terms):
                    score += 1

        scored.append(
            (
                score,
                playbook["name"],
                section,
            )
        )

    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    selected = [
        item
        for item in scored
        if item[0] > 0
    ][:max(1, int(top_k))]

    # Safe fallback: use the most general/highest-ranked documents.
    if not selected:
        selected = scored[:max(1, int(top_k))]

    blocks = []

    for score, name, section in selected:
        blocks.append(
            f"RETRIEVED PLAYBOOK: {name}\n"
            f"RETRIEVAL SCORE: {score}\n\n"
            f"{section}"
        )

    return "\n\n" + ("\n\n" + ("=" * 60) + "\n\n").join(blocks)
