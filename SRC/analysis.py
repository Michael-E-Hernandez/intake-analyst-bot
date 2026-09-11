from pathlib import Path

import ast

import json

import os

import re

import subprocess

import sys

import tempfile

import textwrap

from dotenv import load_dotenv

from google import genai

from rag import retrieve_playbooks



# =========================================================

# CONFIG

# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:

    raise RuntimeError("GEMINI_API_KEY was not found in the project .env file.")

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.6-flash"



# =========================================================

# BASIC HELPERS

# =========================================================

def load_text_file(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(f"Required file does not exist:\n{path}")

    text = path.read_text(encoding="utf-8")

    if not text.strip():

        raise ValueError(f"Required text file is empty:\n{path}")

    return text



def load_json_file(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(f"Required file does not exist:\n{path}")

    with path.open("r", encoding="utf-8") as f:

        return json.load(f)



def _extract_json_object(text):

    if not text:

        raise RuntimeError("Gemini returned an empty response.")

    text = text.strip()

    text = re.sub(r"^\`\`\`json\s*", "", text, flags=re.I)

    text = re.sub(r"^\`\`\`\s*", "", text)

    text = re.sub(r"\s*\`\`\`$", "", text)

    first = text.find("{")

    if first < 0:

        raise ValueError("No JSON object was found in Gemini output.")

    depth = 0

    in_string = False

    escape = False

    for i in range(first, len(text)):

        ch = text[i]

        if in_string:

            if escape:

                escape = False

            elif ch == "\\":

                escape = True

            elif ch == '"':

                in_string = False

            continue

        if ch == '"':

            in_string = True

        elif ch == "{":

            depth += 1

        elif ch == "}":

            depth -= 1

            if depth == 0:

                candidate = text[first:i + 1]

                candidate = re.sub(r",\s*([}**\\]**])", r"\1", candidate)

                return json.loads(candidate, strict=False)

    raise ValueError("Gemini JSON object was incomplete.")



def call_gemini_json(prompt, temperature=0.0, max_output_tokens=8192, retries=4):

    last_error = None

    working_prompt = prompt

    for attempt in range(1, retries + 1):

        try:

            interaction = client.interactions.create(

                model=MODEL_NAME,

                input=working_prompt,

                generation_config={

                    "temperature": temperature,

                    "max_output_tokens": max_output_tokens,

                    "top_p": 0.95,

                    "thinking_level": "low",

                },

            )

            return _extract_json_object(interaction.output_text)

        except Exception as exc:

            last_error = exc

            working_prompt = (

                prompt

                + "\n\nYour previous response could not be parsed. Return ONE valid JSON object only. "

                  "Do not use markdown fences or commentary outside JSON."

            )

    raise RuntimeError(f"Gemini repeatedly failed to return valid JSON: {last_error}")



# =========================================================

# AGENT TOOL POLICY

# =========================================================

# The analysis agent is intentionally not restricted to a fixed menu such as

# grouped_mean / grouped_rate / regression / numeric_binning. It can write the

# analysis it needs using broad analytical tools. We still sandbox execution.

DISALLOWED_AST = (

    ast.Import,

    ast.ImportFrom,

    ast.Global,

    ast.Nonlocal,

    ast.With,

    ast.AsyncWith,

    ast.Lambda,

    ast.ClassDef,

)

DISALLOWED_NAMES = {

    "open", "exec", "eval", "compile", "__import__", "input",

    "os", "sys", "subprocess", "socket", "requests", "urllib",

    "shutil", "pathlib", "pickle", "joblib",

}



def validate_generated_code(code):

    try:

        tree = ast.parse(code)

    except SyntaxError as exc:

        raise ValueError(f"Generated analysis code has a syntax error: {exc}")

    for node in ast.walk(tree):

        if isinstance(node, DISALLOWED_AST):

            raise ValueError(f"Generated code used disallowed syntax: {type(node).__name__}")

        if isinstance(node, ast.Name) and node.id in DISALLOWED_NAMES:

            raise ValueError(f"Generated code referenced disallowed name: {node.id}")

        if isinstance(node, ast.Attribute) and str(node.attr).startswith("__"):

            raise ValueError("Generated code attempted dunder attribute access.")

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):

            if node.func.id in DISALLOWED_NAMES:

                raise ValueError(f"Generated code attempted disallowed call: {node.func.id}")

    return True



# =========================================================

# PLANNER

# =========================================================

def build_agent_plan(metadata, inventory, business_environment, business_question, requested_output):

    try:

        playbooks = retrieve_playbooks(

            query=business_question,

            stage="analysis",

            top_k=3,

        )

    except Exception as exc:

        playbooks = f"RAG retrieval unavailable: {exc}"

    prompt = f"""

You are the ANALYSIS AGENT for an AI analyst system.

The stakeholder has already completed analyst intake and HUMAN-CONFIRMED the

business question below. Your job is now to decide HOW to answer it.

CORE ARCHITECTURE RULE:

\- RAG gives you analyst methodology and lenses.

\- YOU decide the analysis dynamically.

\- Do NOT force the question into a predefined menu of analysis types.

\- You may perform descriptive analysis, segmentation, derived variables,

  statistical tests, regression, clustering, text analysis, comparisons,

  transformations, joins, or other defensible analysis supported by the data.

\- Do not invent unsupported fields or business meanings.

\- Do not make causal claims from observational descriptive evidence.

\- Use actual computation for numeric claims.

\- Inspect distributions before choosing arbitrary numeric cut points when useful.

\- Preserve analytical grain and be cautious with one-to-many joins.

ANALYST INVESTIGATION BEHAVIOR:

\- Treat this as the FIRST investigation pass, not necessarily the final answer.

\- Start with the broad patterns needed to answer the confirmed question.

\- Look for contrasts, concentrations, outliers, subgroup differences, and relationships among the confirmed factors.

\- When multiple confirmed factors may overlap, calculate evidence that helps distinguish a surface-level pattern from a more specific underlying pattern.

\- Do not stop at "which group is highest" when the connected data can defensibly show how the factors relate to one another.

\- Produce enough compact evidence for a second reasoning step to decide whether deeper analysis is warranted.

- When multiple confirmed factors are available, compute enough evidence to identify
  the strongest initial signal and retain group counts/denominators needed for safe
  interaction or subgroup follow-up analysis.

\- Do not add factors outside the human-confirmed scope merely because they are available.

CONFIRMED BUSINESS QUESTION:

{business_question}

REQUESTED OUTPUT:

{requested_output}

TECHNICAL METADATA:

{json.dumps(metadata, ensure_ascii=False)[:28000]}

SEMANTIC DATA INVENTORY:

{inventory[:18000]}

BUSINESS ENVIRONMENT / CAPABILITY MAP:

{business_environment[:18000]}

RETRIEVED ANALYST METHODOLOGY:

{playbooks[:14000]}

AVAILABLE EXECUTION TOOLS INSIDE THE SANDBOX:

\- pd : pandas

\- np : numpy

\- scipy_stats : scipy.stats when installed

\- sm : statsmodels.api when installed

\- sklearn modules exposed when installed

\- load_table(filename) : loads a connected CSV by exact filename

\- list_tables() : lists connected CSV filenames

\- semantic_classify(records, instruction) : asks the LLM to classify/interpret

  batches of text records according to YOUR instruction and returns JSON rows.

\- to_records(dataframe, limit=200) : converts a dataframe to JSON-safe records

You must generate Python code that performs the actual analysis. The code should

freely decide the correct analytical steps rather than selecting from hard-coded

operation names.

REQUIRED CODE CONTRACT:

1\. Do NOT import anything. Libraries/tools are preloaded.

2\. Do NOT access filesystem/network directly. Use load_table / semantic_classify.

3\. Do NOT use lambda functions. Use normal named helper functions instead.

4\. Assign a JSON-serializable dictionary to RESULT.

4\. RESULT must contain:

   - records_analyzed: integer

   - evidence: list of {{"title": str, "data": list[dict] or dict}}

   - analysis_notes: list[str]

   - visualizations: list of objects, each with title, chart_type (bar/line/scatter), data (list[dict])

5\. Keep evidence compact: normally <= 200 rows per table.

6\. If the question includes written text, semantic_classify can be used with a

   dynamically chosen instruction. It is not limited to predefined themes.

7\. Do not fabricate results if an analysis is unsupported; record the limitation.

Return JSON ONLY:

{{

  "analysis_reasoning_summary": "short explanation of the analytical approach",

  "code": "complete Python code that assigns RESULT",

  "expected_checks": ["validation/check to perform", "..."]

}}

"""

    last_validation_error = None

    working_prompt = prompt

    for generation_attempt in range(1, 4):

        response = call_gemini_json(

            working_prompt,

            temperature=0.05,

            max_output_tokens=12000,

            retries=4,

        )

        code = str(response.get("code") or "").strip()

        if not code:

            last_validation_error = "Analysis agent did not generate executable code."

        else:

            try:

                validate_generated_code(code)

                return response

            except Exception as exc:

                last_validation_error = str(exc)

        working_prompt = (

            prompt

            + f"""

The previous generated code was rejected by the sandbox validator.

VALIDATION ERROR:

{last_validation_error}

Rewrite the analysis code so it obeys every sandbox rule.

Important:

\- Do NOT use lambda functions.

\- Do NOT use imports.

\- Do NOT use filesystem or network access.

\- Use normal named helper functions instead of lambda.

\- Still answer the full confirmed business question.

Return ONE valid JSON object only.

"""

        )

    raise RuntimeError(

        f"Analysis agent repeatedly generated code rejected by the sandbox: "

        f"{last_validation_error}"

    )



# =========================================================

# SANDBOX RUNNER

# =========================================================

def _runner_source(data_dir, generated_code, result_path):

    # This file is generated by trusted application code. The model-generated

    # body has already passed AST validation and receives only controlled tools.

    return f'''

import json

import math

import os

from pathlib import Path

import pandas as pd

import numpy as np

try:

    from scipy import stats as scipy_stats

except Exception:

    scipy_stats = None

try:

    import statsmodels.api as sm

except Exception:

    sm = None

try:

    from sklearn import metrics as sklearn_metrics

    from sklearn import linear_model as sklearn_linear_model

    from sklearn import cluster as sklearn_cluster

    from sklearn import preprocessing as sklearn_preprocessing

    from sklearn import model_selection as sklearn_model_selection

except Exception:

    sklearn_metrics = None

    sklearn_linear_model = None

    sklearn_cluster = None

    sklearn_preprocessing = None

    sklearn_model_selection = None

from dotenv import load_dotenv

from google import genai

DATA_DIR = Path({str(Path(data_dir).resolve())!r})

RESULT_PATH = Path({str(Path(result_path).resolve())!r})

PROJECT_ROOT = Path({str(PROJECT_ROOT.resolve())!r})

load_dotenv(PROJECT_ROOT / ".env")

_api_key = os.getenv("GEMINI_API_KEY")

_llm_client = genai.Client(api_key=_api_key) if _api_key else None

_MODEL = {MODEL_NAME!r}



def list_tables():

    return [p.name for p in sorted(DATA_DIR.glob("*.csv"))]



def load_table(filename):

    name = Path(str(filename)).name

    path = (DATA_DIR / name).resolve()

    if path.parent != DATA_DIR.resolve() or not path.exists() or path.suffix.lower() != ".csv":

        raise ValueError(f"Connected CSV not found: {{filename}}")

    return pd.read_csv(path, low_memory=False)



def _json_safe(value):

    if isinstance(value, dict):

        return {{str(k): _json_safe(v) for k, v in value.items()}}

    if isinstance(value, (list, tuple)):

        return [_json_safe(v) for v in value]

    if isinstance(value, pd.DataFrame):

        return [_json_safe(r) for r in value.to_dict(orient="records")]

    if isinstance(value, pd.Series):

        return [_json_safe(v) for v in value.tolist()]

    if isinstance(value, (np.integer,)):

        return int(value)

    if isinstance(value, (np.floating,)):

        if np.isnan(value) or np.isinf(value):

            return None

        return float(value)

    if isinstance(value, (np.bool_,)):

        return bool(value)

    if pd.isna(value) if not isinstance(value, (str, bytes, dict, list, tuple)) else False:

        return None

    return value



def to_records(df, limit=200):

    if not isinstance(df, pd.DataFrame):

        raise TypeError("to_records expects a pandas DataFrame")

    return _json_safe(df.head(int(limit)).to_dict(orient="records"))



def _extract_json(text):

    first = text.find("[")

    last = text.rfind("]")

    if first >= 0 and last > first:

        return json.loads(text[first:last+1])

    first = text.find("{{")

    last = text.rfind("}}")

    if first >= 0 and last > first:

        obj = json.loads(text[first:last+1])

        return obj if isinstance(obj, list) else obj.get("rows", [])

    raise ValueError("No JSON array found in semantic classification output")



def semantic_classify(records, instruction, batch_size=80):

    if _llm_client is None:

        raise RuntimeError("GEMINI_API_KEY is unavailable for semantic_classify")

    if isinstance(records, pd.Series):

        records = records.tolist()

    records = list(records)

    output = []

    for start in range(0, len(records), int(batch_size)):

        batch = records[start:start+int(batch_size)]

        payload = []

        for i, item in enumerate(batch):

            if isinstance(item, dict):

                row = dict(item)

            else:

                row = {{"text": None if pd.isna(item) else str(item)}}

            row["_row_index"] = start + i

            payload.append(row)

        prompt = f"""You are performing a semantic analysis subtask inside a data-analysis agent.

Instruction: {{instruction}}

Return ONLY a JSON array. Return one output object per input object. Preserve _row_index.

Input records:\n{{json.dumps(payload, ensure_ascii=False, default=str)}}"""

        interaction = _llm_client.interactions.create(

            model=_MODEL,

            input=prompt,

            generation_config={{"temperature": 0.0, "max_output_tokens": 8192, "thinking_level": "low"}},

        )

        output.extend(_extract_json(interaction.output_text))

    return output



# ---------------- MODEL-GENERATED ANALYSIS ----------------

{generated_code}

# ----------------------------------------------------------

if "RESULT" not in globals() or not isinstance(RESULT, dict):

    raise RuntimeError("Generated analysis did not assign a RESULT dictionary")

RESULT_PATH.write_text(

    json.dumps(_json_safe(RESULT), ensure_ascii=False, indent=2, default=str),

    encoding="utf-8",

)

'''



def execute_agent_code(data_dir, code, timeout_seconds=180):

    validate_generated_code(code)

    with tempfile.TemporaryDirectory(prefix="ai_analyst_") as tmp:

        tmp = Path(tmp)

        runner = tmp / "runner.py"

        result_path = tmp / "result.json"

        runner.write_text(_runner_source(data_dir, code, result_path), encoding="utf-8")

        completed = subprocess.run(

            [sys.executable, str(runner)],

            capture_output=True,

            text=True,

            timeout=timeout_seconds,

            cwd=str(tmp),

            env=os.environ.copy(),

        )

        if completed.returncode != 0:

            raise RuntimeError(

                "Agent analysis execution failed.\n"

                f"STDOUT:\n{completed.stdout[-5000:]}\n\n"

                f"STDERR:\n{completed.stderr[-5000:]}"

            )

        if not result_path.exists():

            raise RuntimeError("Agent execution finished without producing result.json")

        return json.loads(result_path.read_text(encoding="utf-8"))



# =========================================================

# REPAIR LOOP

# =========================================================

def repair_code(plan, error_text, metadata, inventory, business_environment, business_question, requested_output):

    prompt = f"""

You are repairing Python analysis code for a sandboxed AI analyst.

CONFIRMED BUSINESS QUESTION:

{business_question}

REQUESTED OUTPUT:

{requested_output}

PRIOR REASONING:

{plan.get('analysis_reasoning_summary', '')}

PRIOR CODE:

{plan.get('code', '')}

EXECUTION ERROR:

{error_text[-8000:]}

DATA METADATA:

{json.dumps(metadata, ensure_ascii=False)[:18000]}

SEMANTIC INVENTORY:

{inventory[:9000]}

BUSINESS ENVIRONMENT:

{business_environment[:9000]}

Repair the code without narrowing the stakeholder's confirmed scope.

Remember:

\- no imports

\- no filesystem/network calls

\- no lambda functions

\- use normal named helper functions instead of lambda

\- assign RESULT

Return JSON only:

{{"analysis_reasoning_summary":"...","code":"...","expected_checks":["..."]}}

"""

    last_validation_error = None

    working_prompt = prompt

    for repair_attempt in range(1, 4):

        repaired = call_gemini_json(

            working_prompt,

            temperature=0.0,

            max_output_tokens=12000,

            retries=3,

        )

        code = str(repaired.get("code") or "").strip()

        if not code:

            last_validation_error = "Repair response did not include executable code."

        else:

            try:

                validate_generated_code(code)

                return repaired

            except Exception as exc:

                last_validation_error = str(exc)

        working_prompt = (

            prompt

            + f"""

The repaired code was still rejected by the sandbox validator.

VALIDATION ERROR:

{last_validation_error}

Rewrite it again.

Do NOT use lambda functions.

Use normal named helper functions instead.

Keep the full confirmed business scope.

Return ONE valid JSON object only.

"""

        )

    raise RuntimeError(

        f"Repair agent repeatedly generated code rejected by the sandbox: "

        f"{last_validation_error}"

    )



# =========================================================

# ITERATIVE INVESTIGATION / DEEPENING

# =========================================================

def build_followup_plan(

    metadata,

    inventory,

    business_environment,

    business_question,

    requested_output,

    investigation_history,

    round_number,

):

    """

    Review evidence already computed and decide whether another analytical pass

    would materially improve the business answer. If so, generate only the

    additional computation needed for that deeper question.

    """

    try:

        playbooks = retrieve_playbooks(

            query=business_question,

            stage="analysis",

            top_k=3,

        )

    except Exception as exc:

        playbooks = f"RAG retrieval unavailable: {exc}"

    history_json = json.dumps(

        investigation_history,

        ensure_ascii=False,

        default=str,

    )[:52000]

    prompt = f"""

You are the INVESTIGATION REVIEWER inside an AI analyst system.

A human has already confirmed the business question. One or more analytical

passes have already been executed against the connected data. Your job is to

review the ACTUAL COMPUTED EVIDENCE and decide whether a strong analyst would

stop or investigate one level deeper before giving the stakeholder the answer.

CONFIRMED BUSINESS QUESTION:

{business_question}

REQUESTED OUTPUT:

{requested_output}

CURRENT INVESTIGATION ROUND:

{round_number}

COMPUTED INVESTIGATION HISTORY:

{history_json}

TECHNICAL METADATA:

{json.dumps(metadata, ensure_ascii=False)[:20000]}

SEMANTIC DATA INVENTORY:

{inventory[:12000]}

BUSINESS ENVIRONMENT / CAPABILITY MAP:

{business_environment[:12000]}

RETRIEVED ANALYST METHODOLOGY:

{playbooks[:12000]}

\=========================================================

HOW A STRONG ANALYST REVIEWS EVIDENCE

\=========================================================

Do NOT ask: "Can I run another analysis?"

Ask: "Would another analysis materially change or sharpen the business story?"

Ask instead from the evidence: "What did I just learn, and what is the most
decision-relevant unresolved question created by that finding?"

Investigate recursively like a strong business analyst:
FIND a meaningful pattern -> QUESTION what may strengthen, qualify, explain, or
concentrate it -> TEST the best supported follow-up -> IDENTIFY the most useful
segment or interaction -> decide whether another pass materially improves the answer.

Each follow-up must be motivated by evidence already found. Prefer follow-ups that:
- test whether a strong factor changes when combined with another in-scope factor
- examine interactions/cross-segments that may reveal concentrated risk or opportunity
- move from a broad pattern to the specific population where it is concentrated
- test whether a pattern persists after accounting for overlapping in-scope factors
- reconcile descriptive and adjusted views when they differ
- turn a general association into a more actionable evidence-supported segment
- determine the most useful next analytical question created by current evidence

When evaluating combinations, include group sizes or denominators whenever possible
so small groups are not presented as equally reliable.

Continue only when at least one of these is true:

\- a broad pattern may be hiding an important subgroup pattern

\- two or more confirmed factors overlap and need to be reconciled

\- the current evidence identifies WHAT is happening but not enough about WHERE

  the pattern is concentrated within the confirmed scope

\- a descriptive pattern should be checked with an appropriate adjusted or

  statistical analysis because the confirmed question asks about drivers,

  associations, explanatory factors, or relationships

\- results from different views appear to conflict and need reconciliation

\- an obvious comparison, interaction, cross-tab, model, robustness check, or

  text drill-down is needed to support the business interpretation

STOP when:

\- the business question is already answered with enough evidence

\- another pass would merely repeat the same ranking or percentages

\- deeper analysis would require variables outside the human-confirmed scope

\- the connected data cannot support the next step reliably

\- the extra work would add technical detail without changing the stakeholder's

  understanding

IMPORTANT:

\- The goal is not maximum analysis. The goal is SUFFICIENT ANALYSIS.

\- Prefer the most specific defensible conclusion supported by the data.

\- Distinguish surface patterns from deeper patterns.

\- Never invent data or meanings.

\- Never make causal claims from observational evidence.

\- Keep the stakeholder's confirmed scope intact.

If action="continue", generate Python code for ONLY the next useful analytical

pass. The code may reload connected tables and perform any defensible analysis

using the preloaded tools.

AVAILABLE EXECUTION TOOLS:

\- pd, np, scipy_stats, sm, sklearn modules when installed

\- load_table(filename)

\- list_tables()

\- semantic_classify(records, instruction)

\- to_records(dataframe, limit=200)

CODE RULES:

1\. Do NOT import anything.

2\. Do NOT use lambda functions. Use named helper functions.

3\. Do NOT access filesystem/network directly.

4\. Assign a JSON-serializable dictionary to RESULT.

5\. RESULT must contain:

   - records_analyzed: integer

   - evidence: list of {{"title": str, "data": list[dict] or dict}}

   - analysis_notes: list[str]

   - visualizations: list of objects with title, chart_type, data

6\. Keep evidence compact, normally <= 200 rows per evidence table.

Return ONE JSON object only:

{{

  "action": "stop" or "continue",

  "review_reasoning": "why the evidence is sufficient or what remains unresolved",

  "next_question": "the analytical sub-question for the next pass, or empty string",

  "analysis_reasoning_summary": "short description of the next analytical approach, or empty string",

  "code": "complete Python code assigning RESULT when continuing, otherwise empty string"

}}

"""

    last_validation_error = None

    working_prompt = prompt

    for generation_attempt in range(1, 4):

        response = call_gemini_json(

            working_prompt,

            temperature=0.05,

            max_output_tokens=12000,

            retries=4,

        )

        action = str(response.get("action") or "").strip().lower()

        if action == "stop":

            response["code"] = ""

            return response

        if action != "continue":

            last_validation_error = (

                "Follow-up reviewer must return action='stop' or action='continue'."

            )

        else:

            code = str(response.get("code") or "").strip()

            if not code:

                last_validation_error = (

                    "Follow-up reviewer chose continue but did not generate code."

                )

            else:

                try:

                    validate_generated_code(code)

                    return response

                except Exception as exc:

                    last_validation_error = str(exc)

        working_prompt = (

            prompt

            + f"""

Your previous follow-up response was rejected.

VALIDATION ERROR:

{last_validation_error}

Return a corrected JSON object. If continuing, the code must obey every sandbox

rule, especially: NO imports and NO lambda functions.

"""

        )

    raise RuntimeError(

        "Investigation reviewer repeatedly failed validation: "

        f"{last_validation_error}"

    )



def execute_plan_with_repair(

    data_dir,

    plan,

    metadata,

    inventory,

    business_environment,

    business_question,

    requested_output,

    label="analysis",

):

    """Execute one generated analysis pass with the existing repair loop."""

    current_plan = plan

    last_error = None

    for attempt in range(1, 4):

        try:

            print(f"Executing {label} (attempt {attempt}/3)...")

            result = execute_agent_code(

                data_dir=data_dir,

                code=current_plan["code"],

                timeout_seconds=240,

            )

            return result, current_plan

        except Exception as exc:

            last_error = exc

            if attempt == 3:

                raise

            print(f"{label.capitalize()} execution failed; asking the agent to repair its approach...")

            current_plan = repair_code(

                plan=current_plan,

                error_text=str(exc),

                metadata=metadata,

                inventory=inventory,

                business_environment=business_environment,

                business_question=business_question,

                requested_output=requested_output,

            )

    raise RuntimeError(f"{label.capitalize()} did not produce a result: {last_error}")



def combine_investigation_history(investigation_history):

    """Combine evidence from multiple passes without hiding which pass produced it."""

    combined_evidence = []

    combined_visualizations = []

    technical_notes = []

    records_analyzed = 0

    seen_viz = set()

    for item in investigation_history:

        result = item.get("result") or {}

        try:

            records_analyzed = max(

                records_analyzed,

                int(result.get("records_analyzed", 0) or 0),

            )

        except Exception:

            pass

        for evidence in result.get("evidence", []) or []:

            if not isinstance(evidence, dict):

                continue

            enriched = dict(evidence)

            enriched.setdefault("investigation_round", item.get("round"))

            enriched.setdefault("investigation_question", item.get("question", ""))

            combined_evidence.append(enriched)

        for viz in result.get("visualizations", []) or []:

            if not isinstance(viz, dict):

                continue

            key = (

                str(viz.get("title", "")),

                str(viz.get("chart_type", "")),

            )

            if key in seen_viz:

                continue

            seen_viz.add(key)

            combined_visualizations.append(viz)

        for note in result.get("analysis_notes", []) or []:

            note = str(note).strip()

            if note and note not in technical_notes:

                technical_notes.append(note)

    return {

        "records_analyzed": records_analyzed,

        "evidence": combined_evidence,

        "visualizations": combined_visualizations,

        "analysis_notes": technical_notes,

        "investigation_rounds": len(investigation_history),

    }



# =========================================================

# GROUNDED INTERPRETATION

# =========================================================

def generate_grounded_interpretation(

    business_question,

    requested_output,

    combined_result,

    investigation_history,

):

    history_json = json.dumps(

        investigation_history,

        ensure_ascii=False,

        default=str,

    )[:62000]

    prompt = f"""

You are the final business-insight layer of an AI analyst system.

The system has already completed a HUMAN-CONFIRMED intake and then performed an

ITERATIVE DATA INVESTIGATION. Multiple analytical passes may have been run.

Every numeric claim below came from actual computation against the connected data.

Your job is to turn that evidence into the kind of analytical story a strong

human analyst would give a business stakeholder.

CONFIRMED BUSINESS QUESTION:

{business_question}

REQUESTED OUTPUT:

{requested_output}

FULL INVESTIGATION HISTORY:

{history_json}

COMBINED COMPUTED EVIDENCE:

{json.dumps(combined_result, ensure_ascii=False, default=str)[:52000]}

\=========================================================

CORE STANDARD

\=========================================================

Do NOT merely list the largest percentages or model outputs.

SYNTHESIZE the evidence.

A strong answer should, when supported by the computed evidence:

\- answer the confirmed question directly

\- rank or prioritize the most important patterns

- identify the PRIMARY signal rather than treating every result equally
- show the investigation chain: primary signal -> what strengthens or qualifies it
  -> important interaction/segment -> what that means for the business

\- explain how multiple findings connect to one another

\- distinguish a broad surface pattern from a more specific underlying pattern

\- explain when one factor appears to partly reflect the mix of another factor

\- reconcile descriptive results with adjusted/statistical results when both exist

\- identify where the problem is most concentrated

- when supported, highlight the most decision-relevant COMBINATION of factors,
  not only the strongest individual factor
- distinguish primary findings from reinforcing/supporting findings
- use group sizes to flag fragile small-sample patterns when relevant
- end with the most valuable specific NEXT INVESTIGATION created by the evidence
  when another useful question remains

\- explain what remains uncertain or unsupported

\- point to sensible areas for further investigation without pretending the data

  proves causation

Example of the reasoning quality wanted:

Instead of:

"Sales has the highest turnover. Level 1 is high. Sales Representatives are high."

Prefer, when evidence supports it:

"Sales looks like the highest-turnover department at first, but the deeper view

shows that the problem is concentrated in particular roles and lower job levels.

That means department alone is too broad a target; the more useful retention

question is which high-turnover roles and levels inside those departments need

closer investigation."

That is the level of CROSS-FINDING SYNTHESIS required.

\=========================================================

PLAIN-ENGLISH COMMUNICATION

\=========================================================

Analyze deeply underneath. Explain simply on top.

Everything the stakeholder sees should sound natural in a meeting.

Do not sound like a statistics textbook, academic paper, model report, or AI log.

Avoid stakeholder-facing jargon such as:

\- odds ratio

\- p-value

\- statistically significant predictor

\- regression coefficient

\- multivariate model

\- confidence interval

\- dummy variable

\- standard error

\- multicollinearity

\- dependent / independent variable

If those methods were used, translate their meaning.

Use useful numbers, but do not dump every number. Round where appropriate.

Good:

"About 40% of Sales Representatives left, compared with about 3% of Research Directors."

Good:

"Sales has higher turnover overall, but the role-level breakdown shows that the

problem is not evenly spread across the department."

\=========================================================

CAUSAL DISCIPLINE

\=========================================================

Never turn association into causation.

Say:

\- "is associated with"

\- "was higher among"

\- "appears concentrated in"

\- "the pattern points toward"

\- "is worth investigating"

Do not say a factor caused the outcome unless causal evidence actually exists.

Recommendations must be framed as areas to investigate, not prescriptions that

pretend the analysis proved why the pattern exists.

\=========================================================

REQUIRED OUTPUT QUALITY

\=========================================================

The answer should be richer than a short summary, while remaining easy to read.

Return JSON only:

{{

  "summary": "3-6 sentence executive synthesis that answers the question and connects the major findings.",

  "key_insights": [

    {{

      "title": "Short business-friendly title",

      "insight": "3-6 sentences that explain the evidence, connect it to other findings when relevant, and tell the stakeholder why it matters."

    }}

  ],

  "caveats": [

    "Plain-English limitation or caution that matters to interpretation."

  ]

}}

Use 3-6 key insights when the evidence supports that many.

Before returning, silently ask:

1\. Did I answer the confirmed question, not a different one?

2\. Did I connect the findings instead of listing them?

3\. Did I distinguish broad patterns from more specific patterns?

4\. Did I avoid causal overclaiming?

5\. Would a nontechnical manager understand this immediately?

6\. Did every concrete claim come from computed evidence?

7. Did I identify the primary signal and explain what strengthens, qualifies,
   or concentrates it?
8. If evidence supports an important interaction or high-risk segment, did I
   surface it instead of stopping at individual-factor rankings?
9. If there is a clear next investigation, did I state that specific analytical
   question rather than a generic recommendation?

Return JSON only.

"""

    return call_gemini_json(

        prompt,

        temperature=0.05,

        max_output_tokens=8000,

        retries=4,

    )



# =========================================================

# PUBLIC ENTRY POINT USED BY interface.py

# =========================================================

def run_analysis(

    data_dir,

    metadata_path,

    inventory_path,

    business_environment_path,

    business_question,

    requested_output,

):

    metadata = load_json_file(metadata_path)

    inventory = load_text_file(inventory_path)

    business_environment = load_text_file(business_environment_path)

    # -----------------------------------------------------

    # ROUND 1 — INITIAL INVESTIGATION

    # -----------------------------------------------------

    print("\nCreating RAG-guided initial investigation plan...")

    initial_plan = build_agent_plan(

        metadata=metadata,

        inventory=inventory,

        business_environment=business_environment,

        business_question=business_question,

        requested_output=requested_output,

    )

    first_result, executed_plan = execute_plan_with_repair(

        data_dir=data_dir,

        plan=initial_plan,

        metadata=metadata,

        inventory=inventory,

        business_environment=business_environment,

        business_question=business_question,

        requested_output=requested_output,

        label="initial investigation",

    )

    investigation_history = [

        {

            "round": 1,

            "question": business_question,

            "reasoning": executed_plan.get("analysis_reasoning_summary", ""),

            "review_reasoning": "Initial analytical pass",

            "result": first_result,

        }

    ]

    # -----------------------------------------------------

    # ROUNDS 2-3 — EVIDENCE-DRIVEN DEEPENING

    # -----------------------------------------------------

    max_rounds = 3

    for round_number in range(2, max_rounds + 1):

        print(f"\nReviewing evidence before investigation round {round_number}...")

        followup = build_followup_plan(

            metadata=metadata,

            inventory=inventory,

            business_environment=business_environment,

            business_question=business_question,

            requested_output=requested_output,

            investigation_history=investigation_history,

            round_number=round_number,

        )

        action = str(followup.get("action") or "stop").strip().lower()

        review_reasoning = str(followup.get("review_reasoning") or "").strip()

        if action != "continue":

            print("Investigation reviewer determined the evidence is sufficient.")

            if review_reasoning:

                print(review_reasoning)

            break

        next_question = str(followup.get("next_question") or "").strip()

        print(

            "Deepening investigation: "

            + (next_question if next_question else "additional evidence check")

        )

        followup_result, executed_followup = execute_plan_with_repair(

            data_dir=data_dir,

            plan=followup,

            metadata=metadata,

            inventory=inventory,

            business_environment=business_environment,

            business_question=business_question,

            requested_output=requested_output,

            label=f"investigation round {round_number}",

        )

        investigation_history.append(

            {

                "round": round_number,

                "question": next_question,

                "reasoning": executed_followup.get("analysis_reasoning_summary", ""),

                "review_reasoning": review_reasoning,

                "result": followup_result,

            }

        )

    # -----------------------------------------------------

    # COMBINE ACTUAL EVIDENCE ACROSS ALL ROUNDS

    # -----------------------------------------------------

    combined_result = combine_investigation_history(

        investigation_history

    )

    # -----------------------------------------------------

    # FINAL BUSINESS SYNTHESIS

    # -----------------------------------------------------

    print("\nSynthesizing the complete investigation into a business answer...")

    interpretation = generate_grounded_interpretation(

        business_question=business_question,

        requested_output=requested_output,

        combined_result=combined_result,

        investigation_history=investigation_history,

    )

    visualizations = combined_result.get("visualizations", [])

    if not isinstance(visualizations, list):

        visualizations = []

    # Stakeholder caveats come ONLY from the communication layer so raw

    # technical notes do not leak into the business-facing UI.

    caveats = list(interpretation.get("caveats", []) or [])

    records_analyzed = combined_result.get("records_analyzed", 0)

    try:

        records_analyzed = int(records_analyzed or 0)

    except Exception:

        records_analyzed = 0

    method_parts = []

    for item in investigation_history:

        reasoning = str(item.get("reasoning") or "").strip()

        if reasoning:

            method_parts.append(

                f"Round {item.get('round')}: {reasoning}"

            )

    return {

        "business_question": business_question,

        "requested_output": requested_output,

        "records_analyzed": records_analyzed,

        "summary": interpretation.get("summary", ""),

        "key_insights": interpretation.get("key_insights", []) or [],

        "visualizations": visualizations,

        "caveats": caveats,

        "analysis_evidence": combined_result.get("evidence", []) or [],

        "analysis_method": "\n".join(method_parts),

        "investigation_rounds": len(investigation_history),

        # Analyst/debug fields. The current Streamlit stakeholder view does not

        # need to render these directly.

        "_analysis_notes": combined_result.get("analysis_notes", []) or [],

        "_investigation_history": investigation_history,

        "_agentic_analysis": True,

    }