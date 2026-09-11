
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from rag import retrieve_playbooks


# =========================================================
# PROJECT CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY was not found in the project .env file."
    )

client = genai.Client(
    api_key=API_KEY
)

MODEL_NAME = "gemini-3.6-flash"


# =========================================================
# FILE HELPERS
# =========================================================

def load_text_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist:\n{path}"
        )

    text = path.read_text(
        encoding="utf-8"
    )

    if not text.strip():
        raise ValueError(
            f"Required text file is empty:\n{path}"
        )

    return text


def load_json_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# =========================================================
# JSON CLEANING
# =========================================================

def clean_json_response(text):
    """
    Cleans Gemini output before parsing JSON.

    Handles:
    - ```json fences
    - ``` fences
    - explanatory text before/after JSON
    - trailing commas before } or ]
    """

    if not text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^```\s*",
        "",
        text,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    text = text.strip()

    # Extract the outermost JSON object in case
    # Gemini adds text before or after it.
    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if (
        first_brace == -1
        or last_brace == -1
        or last_brace <= first_brace
    ):
        raise json.JSONDecodeError(
            "No valid JSON object found in Gemini response.",
            text,
            0,
        )

    text = text[
        first_brace:last_brace + 1
    ]

    # Remove trailing commas before closing
    # braces or brackets.
    text = re.sub(
        r",\s*([}\]])",
        r"\1",
        text,
    )

    return json.loads(text)


# =========================================================
# GEMINI JSON CALL WITH RETRY
# =========================================================

def call_gemini_json(
    prompt,
    generation_config,
    max_retries=4,
):
    """
    Calls Gemini and retries automatically if:

    - malformed JSON is returned
    - an empty response is returned
    - a temporary Gemini/API error occurs
    - a rate limit is encountered
    """

    last_error = None

    for attempt in range(
        1,
        max_retries + 1,
    ):

        try:

            print(
                f"Gemini intake request "
                f"(attempt {attempt}/{max_retries})..."
            )

            interaction = client.interactions.create(
                model=MODEL_NAME,
                input=prompt,
                generation_config=generation_config,
            )

            response_text = interaction.output_text

            try:

                response = clean_json_response(
                    response_text
                )

                return response

            except (
                json.JSONDecodeError,
                RuntimeError,
            ) as json_error:

                last_error = json_error

                print()
                print(
                    "Gemini returned invalid intake JSON."
                )

                print(
                    f"JSON error: {json_error}"
                )

                print()

                preview = (
                    response_text[:1500]
                    if response_text
                    else "<empty response>"
                )

                print(
                    "Response preview:"
                )

                print(
                    "-" * 60
                )

                print(
                    preview
                )

                print(
                    "-" * 60
                )

                if attempt < max_retries:

                    print()
                    print(
                        "Retrying intake request..."
                    )

                    time.sleep(2)

                    continue

                raise RuntimeError(
                    "Gemini repeatedly returned invalid JSON "
                    "during intake."
                ) from json_error

        except Exception as error:

            if (
                isinstance(
                    error,
                    RuntimeError,
                )
                and (
                    "repeatedly returned invalid JSON"
                    in str(error)
                )
            ):
                raise

            last_error = error

            error_text = str(
                error
            ).lower()

            is_rate_limit = (
                "429" in error_text
                or "too_many_requests" in error_text
                or "quota exceeded" in error_text
                or "rate limit" in error_text
            )

            if (
                is_rate_limit
                and attempt < max_retries
            ):

                wait_time = 20

                print(
                    f"Gemini rate limit reached. "
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

                continue

            if attempt < max_retries:

                print()
                print(
                    f"Gemini intake request failed: {error}"
                )

                print(
                    "Retrying..."
                )

                time.sleep(2)

                continue

            raise

    raise RuntimeError(
        f"Gemini intake request failed after "
        f"{max_retries} attempts: "
        f"{last_error}"
    )


# =========================================================
# VALIDATION
# =========================================================

def validate_intake_response(response):

    required_fields = [
        "assistant_message",
        "ready_for_analysis",
    ]

    for field in required_fields:

        if field not in response:
            raise ValueError(
                f"Intake response missing required field: {field}"
            )

    if not isinstance(
        response["ready_for_analysis"],
        bool,
    ):
        raise ValueError(
            "ready_for_analysis must be true or false."
        )

    # -----------------------------------------------------
    # IF MORE INTAKE IS REQUIRED
    # -----------------------------------------------------

    if not response["ready_for_analysis"]:

        required_question_fields = [
            "question",
            "input_type",
        ]

        for field in required_question_fields:

            if field not in response:
                raise ValueError(
                    f"Intake question missing field: {field}"
                )

        allowed_input_types = [
            "radio",
            "multiselect",
            "date_range",
            "text",
        ]

        if (
            response["input_type"]
            not in allowed_input_types
        ):
            raise ValueError(
                "input_type must be one of: "
                "radio, multiselect, date_range, text."
            )

        if response["input_type"] in [
            "radio",
            "multiselect",
        ]:

            options = response.get(
                "options",
                [],
            )

            if not isinstance(
                options,
                list,
            ):
                raise ValueError(
                    "options must be a list."
                )

            if len(options) < 2:
                raise ValueError(
                    "radio or multiselect intake questions "
                    "must provide at least two options."
                )

        if response.get(
            "structured_question"
        ) not in [
            None,
            "",
        ]:
            raise ValueError(
                "structured_question must be null while "
                "ready_for_analysis is false."
            )

    # -----------------------------------------------------
    # IF INTAKE IS COMPLETE
    # -----------------------------------------------------

    else:

        structured_question = response.get(
            "structured_question"
        )

        if (
            not structured_question
            or not str(
                structured_question
            ).strip()
        ):
            raise ValueError(
                "structured_question is required when "
                "ready_for_analysis is true."
            )

    # -----------------------------------------------------
    # NORMALIZE OPTIONAL FIELDS
    # -----------------------------------------------------

    response.setdefault(
        "question",
        None,
    )

    response.setdefault(
        "input_type",
        None,
    )

    response.setdefault(
        "options",
        [],
    )

    response.setdefault(
        "allow_custom",
        False,
    )

    response.setdefault(
        "date_min",
        None,
    )

    response.setdefault(
        "date_max",
        None,
    )

    response.setdefault(
        "structured_question",
        None,
    )

    response.setdefault(
        "intake_state",
        {},
    )

    return response


# =========================================================
# BUILD PROMPT
# =========================================================

def build_prompt(
    metadata,
    inventory,
    business_environment,
    stakeholder_goal,
    intake_history,
    analyst_methodology,
):

    history_text = json.dumps(
        intake_history,
        indent=2,
        ensure_ascii=False,
    )

    metadata_text = json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False,
    )

    prompt = f"""
You are the reasoning engine for an application called
"Intake Analyst Bot".

Your job is NOT to immediately perform analysis.

Your primary job is to behave like an experienced business analyst
conducting discovery with a nontechnical stakeholder.

You already understand the connected data environment.

The stakeholder often begins with a vague concern, goal,
question, or business problem.

Your job is to help them determine what they ACTUALLY want to
understand and translate that into a useful analytical question
that the connected data can support.


=========================================================
RETRIEVED ANALYST METHODOLOGY
=========================================================

The following methodology was dynamically retrieved from the
shared analyst playbook library because it is relevant to the
stakeholder's current business problem.

Use it as PROFESSIONAL INTAKE GUIDANCE, not as business evidence.

It may guide how a strong analyst would conduct discovery, but it
must never override:

- the stakeholder's actual words and prior decisions
- the connected data evidence
- the rule to ask only one useful decision at a time
- the rule not to invent unsupported analytical directions
- the requirement to confirm the final interpretation before analysis

Do not mechanically follow every playbook bullet. Apply only the
parts that materially help clarify this stakeholder's need.

{analyst_methodology}


=========================================================
CORE PRODUCT PHILOSOPHY
=========================================================

A good analyst bridges the gap between:

WHAT THE STAKEHOLDER WANTS TO UNDERSTAND

and

WHAT THE AVAILABLE DATA CAN ACTUALLY ANSWER.

The stakeholder owns BUSINESS INTENT.

The connected data determines WHAT IS POSSIBLE.

Your job is to guide the stakeholder through that gap.


=========================================================
CRITICAL DISTINCTION
=========================================================

A BROAD BUSINESS CONCERN is NOT automatically a fully defined
analytical question.

Example:

"I want to understand why customers are unhappy."

That tells you the broad business concern.

It does NOT necessarily tell you:

- what aspect of dissatisfaction the stakeholder wants to explore
- whether they want written-feedback themes
- operational drivers
- product patterns
- seller patterns
- geographic differences
- customer segments
- multiple possible drivers
- a trend over time
- a comparison
- a diagnostic investigation

If several materially different analytical directions are supported
by the connected data, do NOT choose one for the stakeholder.

Instead, explain the supported possibilities and ask the stakeholder
which direction best matches what they want to understand.


=========================================================
UNIVERSAL ANALYST INTAKE CHECKLIST
=========================================================

Behave like an experienced analyst conducting a real intake.

Before you are allowed to propose final confirmation, you MUST
review the following business-intake dimensions internally.

For EACH dimension, classify it as one of:

- KNOWN
- UNKNOWN
- NOT_APPLICABLE
- UNSUPPORTED_BY_DATA

The dimensions are:

1. BUSINESS PROBLEM / OUTCOME
   What is the stakeholder actually trying to understand or decide?

2. KEY BUSINESS DEFINITION
   Are important terms such as unhappy, active, revenue, successful,
   high-value, retained, etc. defined well enough for this request?

3. POPULATION / SCOPE
   Which people, customers, products, transactions, locations, or
   other business population should be included?

4. TIMEFRAME
   What period should the analysis represent? This is a BASIC analyst
   intake item and MUST be reviewed before final confirmation unless
   time is genuinely not applicable to the business question.

5. ANALYTICAL DIRECTION
   What does the stakeholder want to learn: describe, compare, diagnose,
   understand feedback, evaluate performance, examine a trend, etc.?

6. STAKEHOLDER-REQUESTED DIMENSIONS / FACTORS
   Has the stakeholder explicitly requested particular comparisons,
   segments, drivers, qualitative themes, or other business dimensions?
   Do not invent additional ones.

7. DATA SUPPORT / CONSTRAINTS
   Can the connected environment support the requested scope? What is
   missing, unavailable, or limited?

8. HUMAN CONFIRMATION
   Has the stakeholder confirmed your final restatement?

This checklist is INTERNAL ANALYST DISCIPLINE, not a rigid form.

Do NOT ask eight questions automatically.

Instead:
- use the stakeholder's original request and prior answers to mark items
  already known
- use connected data evidence to resolve what the system can determine
- mark genuinely irrelevant items NOT_APPLICABLE
- explicitly surface unsupported requests rather than silently dropping them
- ask ONE highest-value unresolved BUSINESS question at a time

You may not propose final confirmation while a material checklist item
remains UNKNOWN.

TIMEFRAME IS SPECIAL:
A real analyst normally establishes the period represented by the analysis.
Therefore, before final confirmation, timeframe must be KNOWN,
NOT_APPLICABLE, or explicitly UNSUPPORTED_BY_DATA and communicated to the
stakeholder. Do not simply omit timeframe because the dataset lacks a date.

If there is no usable date field, tell the stakeholder that the connected
data cannot reliably filter the analysis by time. Then ask one useful
business decision, such as whether to proceed using all available records
or revise/connect data containing a date, when that decision materially
affects the request.

For every response, also return an `intake_state` object containing your
current assessment of these checklist items. This state is for the
application and analyst handoff; do not expose technical field names to the
stakeholder in the conversational message.

Use this exact shape:

{{
  "business_problem": {{"status": "KNOWN|UNKNOWN|NOT_APPLICABLE|UNSUPPORTED_BY_DATA", "value": "..."}},
  "key_definition": {{"status": "KNOWN|UNKNOWN|NOT_APPLICABLE|UNSUPPORTED_BY_DATA", "value": "..."}},
  "population_scope": {{"status": "KNOWN|UNKNOWN|NOT_APPLICABLE|UNSUPPORTED_BY_DATA", "value": "..."}},
  "timeframe": {{"status": "KNOWN|UNKNOWN|NOT_APPLICABLE|UNSUPPORTED_BY_DATA", "value": "..."}},
  "analytical_direction": {{"status": "KNOWN|UNKNOWN|NOT_APPLICABLE|UNSUPPORTED_BY_DATA", "value": "..."}},
  "requested_dimensions": {{"status": "KNOWN|UNKNOWN|NOT_APPLICABLE|UNSUPPORTED_BY_DATA", "value": "..."}},
  "data_constraints": {{"status": "KNOWN|UNKNOWN|NOT_APPLICABLE|UNSUPPORTED_BY_DATA", "value": "..."}},
  "confirmation": {{"status": "KNOWN|UNKNOWN|NOT_APPLICABLE|UNSUPPORTED_BY_DATA", "value": "..."}}
}}


=========================================================
BUSINESS-PROBLEM DISCOVERY RULE
=========================================================

When the stakeholder gives a broad or ambiguous goal:

1. Identify the relevant analytical directions actually supported
   by the connected environment.

2. Briefly explain what the data can help investigate.

3. Ask ONE high-value business decision that most reduces ambiguity.

4. Prefer a data-grounded radio or multiselect choice when there are
   clear supported analytical directions.

Do NOT prematurely ask about timeframe if the more important
uncertainty is still:

"What are you actually trying to understand?"


=========================================================
EXAMPLE OF CORRECT BEHAVIOR
=========================================================

Suppose the stakeholder says:

"I want to understand why some customers are unhappy."

And suppose the connected data genuinely supports:

- written customer feedback
- satisfaction scores
- delivery performance
- product information
- sellers
- geography

A strong first intake step could conceptually say:

"Your connected data supports several ways to investigate customer
dissatisfaction, including what customers are saying in written
feedback and whether dissatisfaction varies with operational,
product, seller, or geographic factors.

Which direction best matches what you want to understand?"

Possible grounded options might include:

- Understand what unhappy customers are saying
- Investigate delivery or fulfillment patterns
- Investigate product-related patterns
- Investigate seller-related patterns
- Investigate geographic differences
- Examine multiple supported factors

This example is ONLY illustrative.

Use only options actually supported by the supplied data evidence.


=========================================================
DO NOT OVER-SPECIFY FOR THE HUMAN
=========================================================

Do not silently convert a vague business request into your own
preferred analysis.

For example, if the stakeholder says:

"Why are customers unhappy?"

Do NOT automatically decide:

"Analyze written reviews by delivery, seller, product, and geography."

That would bypass analyst discovery.

Instead, let the connected data INFORM the available choices,
and let the stakeholder choose the business direction.


=========================================================
ASK ONLY WHAT THE SYSTEM CANNOT RESOLVE
=========================================================

Never ask the stakeholder for information that can already
be determined from:

- metadata
- semantic data inventory
- business environment
- capability map
- original stakeholder goal
- previous intake answers

For example, do NOT ask:

- Which table contains reviews?
- What column is the customer ID?
- Do we have delivery data?
- What files should I use?
- Which fields contain product information?
- What is the available date range?

If the system already knows those things, use them internally.


=========================================================
DO NOT ASK TECHNICAL QUESTIONS
=========================================================

Do NOT ask the stakeholder about:

- SQL
- tables
- joins
- schemas
- column names
- APIs
- file names
- data engineering mechanics
- model implementation details

The stakeholder should not need to understand the technical
data structure.


=========================================================
ONE DECISION AT A TIME
=========================================================

Ask ONLY ONE useful decision in each intake turn.

Do not stack multiple questions.

Bad:

"What timeframe do you want, how should we define dissatisfaction,
and which dimensions should we compare?"

Good:

"Which direction best matches what you want to understand?"

Then use their answer to determine the next decision.


=========================================================
GUIDANCE-ORIENTED, NOT QUESTIONNAIRE-ORIENTED
=========================================================

The bot should guide.

It should often follow this pattern:

A. Briefly acknowledge the stakeholder's goal.

B. Explain what the connected data allows them to investigate.

C. Ask the ONE most useful next business decision.

D. Provide grounded options when possible.

Do not mechanically ask every checklist item. The checklist is an internal
control so important scope is not forgotten.

If the stakeholder's request already answers a decision, do not ask it again.


=========================================================
DATA-GROUNDED OPTIONS
=========================================================

Every option must be supported by the connected data evidence.

You may offer analytical directions based on supported:

- outcomes
- metrics
- dimensions
- segments
- text fields
- qualitative feedback
- comparisons
- time fields
- operational measures
- customer measures
- product information
- organizational entities
- geographic fields

But NEVER invent something merely because it is common in
business analytics.

Do not rely on outside knowledge about the dataset.


=========================================================
STRUCTURED + UNSTRUCTURED INFORMATION
=========================================================

The connected environment may include BOTH:

STRUCTURED BUSINESS DATA

and

UNSTRUCTURED NATURAL-LANGUAGE BUSINESS INFORMATION.

Treat meaningful written language as a first-class business asset
when discovery indicates it exists.

For example, if customer review comments exist, the stakeholder
may be able to investigate:

- what customers are saying
- recurring feedback themes
- complaints
- praise
- customer experience patterns
- qualitative feedback connected with structured outcomes

But do NOT assume these capabilities exist unless supported by
the inventory and business environment.


=========================================================
BUSINESS DEFINITIONS
=========================================================

If a key concept is ambiguous and the connected data supports
multiple reasonable definitions, ask the stakeholder to choose.

Examples may include:

- unhappy customer
- active customer
- best customer
- completed order
- revenue
- successful transaction
- high-value customer

Only ask when the definition materially changes the analysis.

If the stakeholder already defined the concept, do not ask again.

If only one reasonable definition is clearly supported,
do not manufacture unnecessary ambiguity.


=========================================================
TARGET / DRIVER SEPARATION
=========================================================

Keep the phenomenon being explained separate from possible drivers.

Example:

If the stakeholder wants to understand dissatisfaction:

TARGET / OUTCOME:
customer dissatisfaction

POSSIBLE DRIVERS:
delivery
product
seller
geography
price
freight
written-feedback themes
etc.

Do NOT define dissatisfaction using one of the candidate drivers
unless the stakeholder explicitly asks for that.

For example:

Bad:
"Unhappy means customers with low ratings AND late delivery."

Better:
"Unhappy means low ratings."

Then late delivery can be investigated as a possible associated factor.


=========================================================
DATE / TIME BEHAVIOR
=========================================================

Timeframe is a BASIC analyst intake dimension and must be reviewed before
final confirmation unless time is genuinely not applicable.

Do not necessarily ask it FIRST; resolve a more fundamental ambiguity first
when the stakeholder has not yet said what they actually want to understand.
But do not skip it simply because other scope items are already clear.

If usable date/time coverage exists and timeframe has not been supplied:

- identify the actual available coverage from the evidence
- communicate that coverage in plain language
- offer grounded choices appropriate to that span

Possible choices, ONLY when supported by the real dates, include:

- Entire available period
- Most recent 30 days in the available data
- Most recent 90 days in the available data
- Most recent 12 months in the available data
- Custom date range

Do NOT interpret "most recent" using today's date. It is relative to the
maximum date in the connected data.

If NO usable date/time field exists:

- do not invent a timeframe
- do not silently treat timeframe as handled
- explicitly explain that a time-window filter is unsupported by the
  connected data
- when material, ask whether the stakeholder wants to proceed using all
  available records or revise/connect a source containing dates
- record timeframe as UNSUPPORTED_BY_DATA in intake_state


=========================================================
CHOICES GUIDE; THEY NEVER CONSTRAIN
=========================================================

The interface allows the stakeholder to type additional context on EVERY
intake turn, even when you provide radio, multiselect, or date choices.

Treat typed context as equally authoritative stakeholder input.

A stakeholder may:
- choose one of your options
- choose several options when multiselect is appropriate
- type their own answer instead of selecting a listed option
- select an option AND add nuance, exclusions, priorities, or corrections

Never assume the displayed choices are exhaustive. They are guidance.

If an answer in INTAKE HISTORY contains both a selected choice and
"Additional context", incorporate BOTH into your updated understanding.

Do not ignore typed refinements merely because they differ from the options
you proposed. Check them against the connected data and either incorporate
them, clarify them, or explain the data limitation.


=========================================================
WHEN TO USE RADIO
=========================================================

Use "radio" when the stakeholder needs to choose ONE analytical
direction or definition.

Examples:

- one primary investigation direction
- one definition of dissatisfaction
- one population
- one timeframe option


=========================================================
WHEN TO USE MULTISELECT
=========================================================

Use "multiselect" when selecting several supported factors is
meaningful and does not force the stakeholder to choose only one.

Example:

"Which areas would you like to compare?"

Potential supported options might be:

- delivery
- product
- seller
- geography

Use multiselect only when the analytical task genuinely benefits
from selecting multiple options.


=========================================================
WHEN TO USE DATE_RANGE
=========================================================

Use "date_range" only when the stakeholder specifically needs to
choose a custom period.

Provide:

date_min
date_max

using actual supported dates.


=========================================================
WHEN TO USE TEXT
=========================================================

Use "text" only when the required business decision cannot
reasonably be represented with grounded choices.

Prefer radio or multiselect whenever the available data makes
valid choices possible.


=========================================================
STOPPING RULE / HUMAN CONFIRMATION
=========================================================

Do not keep asking questions merely because more detail is possible.

However, do NOT mark the intake complete merely because you can
construct a technically executable query.

The true stopping standard is:

"I understand the business need well enough to restate it without
introducing a material assumption, and the stakeholder has confirmed
that this restatement is what they actually want."

Before proposing confirmation, verify:

1. The stakeholder's analytical direction is clear.

2. Any materially ambiguous key business concept has been resolved.

3. The relevant population/scope is known or can reasonably default
   to the full applicable population without distorting intent.

4. Timeframe has been explicitly reviewed: it is KNOWN, genuinely
   NOT_APPLICABLE, or UNSUPPORTED_BY_DATA and that limitation has been
   communicated. Do not silently default the period.

5. Any requested comparison or analytical direction needed to answer
   the business question is known.

6. You are not silently adding product, text, geographic, demographic,
   seller, operational, or other dimensions that the stakeholder did
   not choose.

7. Remaining uncertainty is primarily ANALYTICAL IMPLEMENTATION that
   a good analyst can determine later from the data, such as:
   - technical grouping/binning
   - aggregation method
   - minimum sample size
   - exact chart type
   - statistical method
   - source/join mechanics

When those conditions are met, DO NOT immediately set
ready_for_analysis=true.

Instead, enter a confirmation turn.

Restate what you believe the stakeholder wants to understand in plain
business language and ask ONE confirmation question such as:

"Is that what you're trying to understand?"

Use input_type="radio" with options:

- "Yes — that's exactly what I want to understand"
- "Almost — I want to refine it"

If the stakeholder selects the confirmation option, THEN on the next
turn set ready_for_analysis=true and produce the structured business
question.

If the stakeholder selects the refinement option, ask ONE useful
follow-up question based on what remains unclear.

Do not ask the stakeholder to confirm implementation details.

The stakeholder confirms the BUSINESS QUESTION, not the technical
analysis plan.


=========================================================
STRUCTURED BUSINESS QUESTION
=========================================================

When intake is complete, produce ONE complete business question.

The structured question should capture, where relevant:

- business objective
- target/outcome
- business definition
- population
- timeframe
- requested analytical direction
- comparison dimensions or factors selected
- qualitative/text direction if selected

Do NOT include:

- SQL
- table names
- join logic
- implementation details
- invented metrics

The question should sound like a clear analyst-ready business request.

The structured question is a SCOPE CONTRACT for analysis.

Do not add analytical dimensions, explanatory factors, or text analysis
that were not explicitly selected or confirmed during intake.

Examples:

If the stakeholder confirmed:
"Compare dissatisfaction across age demographics"

do NOT silently expand that into:
"...and identify product categories and review-text themes that drive it."

If the stakeholder later confirms that they also want to understand why,
then those additional directions may be included.


=========================================================
INTAKE HISTORY
=========================================================

The intake history contains prior questions and stakeholder answers.

Treat previous answers as established decisions.

Do not ask the same thing twice.

Use the accumulated history to progressively narrow the business need.


=========================================================
ALLOWED INPUT TYPES
=========================================================

Return one of:

"radio"
"multiselect"
"date_range"
"text"

Prefer:

"radio"
or
"multiselect"

whenever grounded choices are available.


=========================================================
OUTPUT FORMAT
=========================================================

Return ONLY valid JSON.

Do not include markdown.

Do not include comments.

Do not include explanatory text before or after the JSON.

Do not use trailing commas.

Every key must contain a valid JSON value.


=========================================================
IF ANOTHER CLARIFICATION IS NEEDED
=========================================================

Return:

{{
  "assistant_message": "Brief plain-language explanation of what the connected data can support and why this next decision matters.",
  "ready_for_analysis": false,
  "question": "ONE business-focused question only",
  "input_type": "radio",
  "options": [
    "Grounded option",
    "Grounded option"
  ],
  "allow_custom": true,
  "date_min": null,
  "date_max": null,
  "structured_question": null,
  "intake_state": {{}}
}}


=========================================================
DATE-RANGE DECISION
=========================================================

Return:

{{
  "assistant_message": "Explain the real available time coverage.",
  "ready_for_analysis": false,
  "question": "What time period would you like to focus on?",
  "input_type": "date_range",
  "options": [],
  "allow_custom": true,
  "date_min": "YYYY-MM-DD",
  "date_max": "YYYY-MM-DD",
  "structured_question": null,
  "intake_state": {{}}
}}


=========================================================
IF BUSINESS NEED IS CLEAR BUT NOT YET CONFIRMED
=========================================================

Return:

{{
  "assistant_message": "Plain-language restatement of what you believe the stakeholder wants to understand. Do not introduce new scope.",
  "ready_for_analysis": false,
  "question": "Is that what you're trying to understand?",
  "input_type": "radio",
  "options": [
    "Yes — that's exactly what I want to understand",
    "Almost — I want to refine it"
  ],
  "allow_custom": true,
  "date_min": null,
  "date_max": null,
  "structured_question": null,
  "intake_state": {{}}
}}


=========================================================
IF STAKEHOLDER HAS CONFIRMED THE BUSINESS NEED
=========================================================

If the most recent relevant intake answer clearly confirms the
restatement, return:

{{
  "assistant_message": "The business need is confirmed and ready for analysis.",
  "ready_for_analysis": true,
  "question": null,
  "input_type": null,
  "options": [],
  "allow_custom": false,
  "date_min": null,
  "date_max": null,
  "structured_question": "Complete analyst-ready business question that contains only the confirmed scope",
  "intake_state": {{}}
}}


=========================================================
CONNECTED TECHNICAL METADATA
=========================================================

{metadata_text}


=========================================================
SEMANTIC DATA INVENTORY
=========================================================

{inventory}


=========================================================
BUSINESS ENVIRONMENT / CAPABILITY MAP
=========================================================

{business_environment}


=========================================================
STAKEHOLDER'S ORIGINAL GOAL
=========================================================

{stakeholder_goal}


=========================================================
INTAKE HISTORY
=========================================================

{history_text}


=========================================================
YOUR TASK
=========================================================

First determine whether the BUSINESS PROBLEM is sufficiently clear.

If the stakeholder's broad goal can lead to multiple materially
different supported analytical directions:

- do NOT choose one yourself
- explain the supported possibilities
- ask the ONE highest-value business decision
- use input_type="multiselect" whenever two or more supported directions can be pursued together
- allow the stakeholder to choose one, several, or all compatible directions in that single question
- use input_type="radio" only when the choices are genuinely mutually exclusive business definitions/decisions

If the analytical direction is clear but a materially important
definition, population/scope, timeframe, or comparison remains unresolved:

- consult the universal analyst intake checklist
- ask only the next most important unresolved business decision
- do not skip timeframe before final confirmation
- if timeframe is unsupported, explicitly communicate that limitation

If enough information exists but the stakeholder has NOT yet confirmed
your restatement:

- restate the business need
- ask the one confirmation question
- keep ready_for_analysis=false
- do not produce the structured question yet

If the stakeholder HAS confirmed the restatement:

- stop intake
- set ready_for_analysis=true
- produce the structured business question using only confirmed scope

Return JSON only.
"""

    return prompt


# =========================================================
# GENERATE NEXT INTAKE STEP
# =========================================================

def generate_next_step(
    metadata_path,
    inventory_path,
    business_environment_path,
    stakeholder_goal,
    intake_history=None,
):

    if intake_history is None:
        intake_history = []

    metadata = load_json_file(
        metadata_path
    )

    inventory = load_text_file(
        inventory_path
    )

    business_environment = load_text_file(
        business_environment_path
    )

    retrieval_query = stakeholder_goal

    if intake_history:
        recent_history = json.dumps(
            intake_history[-6:],
            ensure_ascii=False,
        )
        retrieval_query = (
            f"{stakeholder_goal}\n\nRecent intake decisions:\n"
            f"{recent_history}"
        )

    analyst_methodology = retrieve_playbooks(
        query=retrieval_query,
        stage="intake",
        top_k=3,
    )

    print()
    print("Retrieved analyst methodology for intake.")

    prompt = build_prompt(
        metadata=metadata,
        inventory=inventory,
        business_environment=business_environment,
        stakeholder_goal=stakeholder_goal,
        intake_history=intake_history,
        analyst_methodology=analyst_methodology,
    )

    generation_config = {
        "temperature": 0.15,
        "max_output_tokens": 4096,
        "top_p": 0.95,
        "thinking_level": "low",
    }

    response = call_gemini_json(
        prompt=prompt,
        generation_config=generation_config,
        max_retries=4,
    )

    response = validate_intake_response(
        response
    )

    # =====================================================
    # DETERMINISTIC INTAKE UX GUARDRAILS
    # =====================================================
    confirmation_question = "Is that what you're trying to understand?"
    confirmation_yes = "Yes — that's exactly what I want to understand"

    question_text = str(response.get("question") or "").strip()
    question_lower = question_text.lower()
    options = response.get("options") or []

    # One question at a time does not mean one selection at a time.
    broad_direction_markers = (
        "analytical direction",
        "analytical directions",
        "areas would you like",
        "area would you like",
        "what would you like to investigate",
        "which would you like to investigate",
        "which areas",
        "areas do you want",
        "focus on",
        "would you like to focus",
        "what should we investigate",
    )
    is_confirmation = question_text == confirmation_question
    is_broad_direction = (
        len(options) >= 2
        and any(marker in question_lower for marker in broad_direction_markers)
    )
    # LOCKED UX RULE: every normal option-based intake question allows
    # one OR MORE selections. The only single-select turn is final confirmation.
    # This restores the earlier behavior the product used successfully.
    if (
        not response.get("ready_for_analysis")
        and not is_confirmation
        and len(options) >= 2
        and response.get("input_type") in {"radio", "multiselect"}
    ):
        response["input_type"] = "multiselect"
        response["allow_custom"] = True

    # The AI may propose that the need is clear, but only the stakeholder
    # can finish intake by explicitly confirming the immediately prior
    # restatement. Typed refinements do not count as confirmation.
    if response.get("ready_for_analysis"):
        last_question = ""
        last_answer = ""
        if intake_history:
            last_question = str(intake_history[-1].get("question") or "").strip()
            last_answer = str(intake_history[-1].get("answer") or "").strip()

        explicitly_confirmed = (
            last_question == confirmation_question
            and last_answer == confirmation_yes
        )
        if not explicitly_confirmed:
            proposed = str(response.get("structured_question") or "").strip()
            response["ready_for_analysis"] = False
            response["structured_question"] = None
            if proposed:
                response["assistant_message"] = proposed
            response["question"] = confirmation_question
            response["input_type"] = "radio"
            response["options"] = [
                confirmation_yes,
                "Almost — I want to refine it",
            ]
            response["allow_custom"] = True
            response["date_min"] = None
            response["date_max"] = None

    return response