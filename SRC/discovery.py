from pathlib import Path
import argparse
import os
import sys
import time

from dotenv import load_dotenv
from google import genai


# =========================================================
# PROJECT SETUP
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


# =========================================================
# COMMAND-LINE ARGUMENTS
# =========================================================

def get_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Build a Business Environment Profile "
            "and Analytical Capability Map."
        )
    )

    parser.add_argument(
        "--inventory",
        required=True,
        help="Path to data_inventory.md"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path where business_environment.md will be saved"
    )

    return parser.parse_args()


# =========================================================
# GEMINI CLIENT
# =========================================================

def create_gemini_client():

    # Local development:
    # Load .env when it exists.
    #
    # Streamlit Cloud:
    # GEMINI_API_KEY is supplied through
    # the process environment.

    if ENV_FILE.exists():

        load_dotenv(
            ENV_FILE
        )

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found in the environment."
        )

    return genai.Client(
        api_key=api_key
    )
# =========================================================
# LOAD DATA INVENTORY
# =========================================================

def load_data_inventory(inventory_file: Path):

    if not inventory_file.exists():
        raise FileNotFoundError(
            f"Data inventory was not found:\n"
            f"{inventory_file}\n\n"
            "Run data_inventory.py first."
        )

    inventory_text = inventory_file.read_text(
        encoding="utf-8"
    )

    if not inventory_text.strip():
        raise ValueError(
            "The Data Inventory file is empty."
        )

    return inventory_text


# =========================================================
# CALL GEMINI WITH RETRY
# =========================================================

def call_gemini_with_retry(
    client,
    prompt: str,
    generation_config: dict,
    max_retries: int = 4
):

    for attempt in range(
        1,
        max_retries + 1
    ):

        try:

            print(
                f"Sending request to Gemini "
                f"(attempt {attempt}/{max_retries})..."
            )

            interaction = client.interactions.create(
                model="gemini-3.6-flash",
                input=prompt,
                generation_config=generation_config,
            )

            output_text = interaction.output_text

            if not output_text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return output_text

        except Exception as error:

            error_text = str(error).lower()

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

                wait_time = 35

                print(
                    "\nGemini rate limit reached."
                )

                print(
                    f"Waiting {wait_time} seconds "
                    "before trying again...\n"
                )

                time.sleep(wait_time)

                continue

            raise


# =========================================================
# BUILD BUSINESS ENVIRONMENT
# =========================================================

def build_business_environment(
    client,
    data_inventory: str
):

    prompt = f"""
You are an experienced AI business analyst reviewing an unfamiliar
business data environment.

A previous AI process has already inspected the connected data and created
a DATA INVENTORY.

Your job is to convert that technical understanding into BUSINESS
UNDERSTANDING.

Create a concise:

1. BUSINESS ENVIRONMENT PROFILE
2. ANALYTICAL CAPABILITY MAP

This output will later be used by an AI analyst interface to help
nontechnical stakeholders turn vague business needs into feasible
analytical questions.

The connected environment may contain BOTH structured business data and
unstructured natural-language information.

Structured information may include transactions, customers, products,
operations, financial measures, dates, locations, statuses, ratings,
or other measurable business attributes.

Unstructured information may include reviews, comments, feedback,
complaints, survey responses, support messages, notes, descriptions,
or other natural-language business information.

When meaningful unstructured information exists, treat it as a business
data asset and determine what kinds of business understanding it may
potentially support.

Do NOT analyze the text itself at this stage.


=========================================================
IMPORTANT RULES
=========================================================

1. Use only the supplied Data Inventory as evidence.

2. Do not assume prior knowledge of the specific dataset.

3. Do not use outside information about the dataset.

4. Infer the likely industry and business model from the evidence.

5. Treat interpretations as hypotheses.

6. Use confidence levels:
   - High
   - Medium
   - Low

7. Distinguish between:
   - directly supported
   - reasonably inferred
   - partially supported
   - unsupported

8. Do not invent fields, datasets, measures, relationships,
   text content, themes, or business outcomes.

9. Do not perform actual business analysis.

10. Do not calculate results.

11. Do not generate SQL.

12. Do not generate Python analysis code.

13. Focus on business meaning instead of repeating the technical inventory.

14. Keep the response concise and useful as context for another AI model.

15. Do not assume that useful business information must be structured.

16. If meaningful natural-language fields exist, identify the business
    capabilities they may support without analyzing their actual content.

17. Do not infer actual customer opinions, complaint themes, sentiment,
    praise, problems, or outcomes from sample text.

18. Clearly distinguish between:
    - what the data environment CAN support
    - what an analysis has actually FOUND

    This stage identifies capabilities only.

19. When structured and unstructured information can be related through
    supported entities or relationships, identify the combined analytical
    opportunity.

20. Do not treat free text as isolated from the rest of the business
    environment when supported relationships allow it to be contextualized.


=========================================================
STRUCTURED + UNSTRUCTURED BUSINESS UNDERSTANDING
=========================================================

When the Data Inventory identifies meaningful natural-language business
information, determine what business questions that information could
potentially help investigate.

Potential capability types may include:

- understanding recurring customer concerns
- identifying common complaint or praise themes
- sentiment or satisfaction analysis
- customer experience analysis
- product or service experience analysis
- operational issue discovery
- feedback summarization
- qualitative-to-quantitative structuring
- identifying themes across large volumes of text
- comparing qualitative feedback across measurable business dimensions

These are reasoning examples only.

Include them only when supported by the supplied Data Inventory.

Also determine whether unstructured information can potentially be
connected with structured business context such as:

- customers
- products
- services
- transactions
- orders
- locations
- sellers
- accounts
- operational events
- financial measures
- dates or time periods
- ratings
- outcomes
- other supported dimensions

When such relationships are supported, explain the combined capability.

For example, the general analytical pattern may be:

    qualitative feedback
            +
    structured business context
            ↓
    understanding WHAT people are saying
    and WHERE, WHEN, FOR WHOM, or under WHAT BUSINESS CONDITIONS
    those patterns occur

This is a general reasoning pattern, not a conclusion about the supplied
data.

Do not perform the analysis now.


=========================================================
OUTPUT FORMAT
=========================================================

# BUSINESS ENVIRONMENT PROFILE


## 1. INDUSTRY AND BUSINESS MODEL

Identify:

- Most likely industry/domain
- Likely business model
- Main business participants
- How transactions/value appear to flow
- Whether meaningful qualitative/unstructured business information exists
- Confidence
- Evidence


## 2. CORE BUSINESS PROCESSES

Identify the major processes represented by the connected data.

For each include:

- Process
- Business purpose
- Supporting data/entities
- Whether structured data, unstructured information, or both contribute
- Confidence


## 3. BUSINESS FUNCTIONS / STAKEHOLDERS

Identify likely business functions that could use this data.

For each include:

- Function/stakeholder
- Relevant available information
- Likely analytical interests
- Whether qualitative information could contribute to those interests


# ANALYTICAL CAPABILITY MAP


## 4. SUPPORTED ANALYTICAL CAPABILITIES


### 4.1 Structured Data Capabilities

Identify business analyses supported primarily by structured fields.

For each area include:

- Business area
- What can be measured
- Relevant entities/data
- Example stakeholder questions
- Confidence

Do not answer the example questions.


### 4.2 Unstructured Information Capabilities

If meaningful natural-language business information exists, identify what
kinds of analysis it could potentially support.

For each capability include:

- Business area
- Unstructured information available
- What could potentially be understood
- Relevant entities/data
- Example stakeholder questions
- Confidence

Examples may involve understanding feedback, sentiment, themes, complaints,
praise, experience, support issues, or other qualitative information.

Only include capabilities supported by the Data Inventory.

Do not identify actual themes or conclusions.


### 4.3 Combined Structured + Unstructured Capabilities

Identify analyses that become possible by connecting qualitative
natural-language information with structured business information.

For each capability include:

- Business objective
- Unstructured source
- Structured context
- Supported relationship between them
- What the combination could help the stakeholder understand
- Example stakeholder questions
- Confidence

Focus especially on opportunities where qualitative information could
explain, contextualize, segment, or enrich measurable business outcomes.

Do not perform the analysis.


## 5. LIMITATIONS AND DATA GAPS

Identify:

- Partially supported questions
- Unsupported important questions
- Missing structured data
- Missing or sparse unstructured information
- Text coverage limitations
- Relationship/grain limitations
- What additional data would unlock
- What additional analytical capabilities that data would support

Do not assume missing data that is not relevant to the apparent
business environment.


## 6. CURRENT BUSINESS SCOPE

Separate capabilities into:

### Strongly Supported

Include capabilities where the necessary data and relationships appear
well supported.

### Supported With Limitations

Include capabilities where analysis appears possible but important
coverage, grain, semantic, text, or relationship limitations exist.

### Not Currently Supported

Include important business questions that cannot reasonably be answered
from the connected environment.


## 7. ANALYST INTAKE FOUNDATION

Summarize what the future conversational analyst should know.

Include:

- Major areas the stakeholder can explore
- High-confidence analytical capabilities
- Available qualitative/unstructured information
- Important structured + unstructured analytical opportunities
- Important limitations
- Important uncertainty
- Useful themes for dynamic clarification questions

The future analyst should use knowledge of BOTH the stakeholder's intent
and the available business information to guide discovery.

For example, when a stakeholder asks a broad question, the analyst may
explain which measurable or qualitative directions the connected data
supports and help the stakeholder narrow the request.

Do NOT create a fixed questionnaire.

The future interface should dynamically decide what to ask based on:

- stakeholder intent
- connected data
- business environment
- analytical capabilities
- available structured information
- available unstructured information
- previous stakeholder answers

It should ask only for information that cannot already be inferred from
the available evidence.

The stakeholder should not need to understand schemas, field names,
joins, text-processing methods, or technical implementation details.


=========================================================
FINAL BUSINESS DISCOVERY PRINCIPLE
=========================================================

The purpose of Business Discovery is to determine:

"What can this business potentially understand from the information
it already has?"

That information may exist as:

- structured facts and measurements
- human language and qualitative information
- or relationships between both

The goal is to make large, fragmented business information easier for
people to understand and explore through a guided analytical system.

Business Discovery identifies what CAN be investigated.

It does NOT determine what the analysis has FOUND.

Do not summarize customer opinions.

Do not infer actual complaint themes.

Do not calculate sentiment.

Do not generate business conclusions.

Those tasks belong to the later analysis stage.


=========================================================
DATA INVENTORY
=========================================================

{data_inventory}
"""

    generation_config = {
        "temperature": 0.2,
        "max_output_tokens": 4096,
        "top_p": 0.95,
        "thinking_level": "low",
    }

    return call_gemini_with_retry(
        client=client,
        prompt=prompt,
        generation_config=generation_config,
        max_retries=4
    )


# =========================================================
# SAVE OUTPUT
# =========================================================

def save_business_environment(
    business_environment: str,
    output_file: Path
):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file.write_text(
        business_environment,
        encoding="utf-8"
    )

    if not output_file.exists():
        raise RuntimeError(
            f"Business environment file was not created:\n"
            f"{output_file}"
        )

    if output_file.stat().st_size == 0:
        raise RuntimeError(
            f"Business environment file is empty:\n"
            f"{output_file}"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    args = get_arguments()

    inventory_file = Path(
        args.inventory
    ).expanduser()

    output_file = Path(
        args.output
    ).expanduser()

    print("\n" + "=" * 80)
    print("BUSINESS DISCOVERY")
    print("=" * 80)

    print(
        f"\nInventory input:\n{inventory_file}"
    )

    print(
        f"\nBusiness environment output:\n{output_file}"
    )

    print(
        "\nLoading Data Inventory..."
    )

    data_inventory = load_data_inventory(
        inventory_file
    )

    print(
        "Data Inventory loaded successfully."
    )

    client = create_gemini_client()

    print(
        "\nGemini client ready."
    )

    print(
        "\nBuilding Business Environment Profile "
        "+ Analytical Capability Map...\n"
    )

    business_environment = build_business_environment(
        client,
        data_inventory
    )

    save_business_environment(
        business_environment,
        output_file
    )

    print("\n" + "=" * 80)
    print("BUSINESS DISCOVERY COMPLETE")
    print("=" * 80)

    print(
        f"\nSaved to:\n{output_file}"
    )

    print(
        f"\nFile exists: {output_file.exists()}"
    )

    print(
        f"File size: {output_file.stat().st_size:,} bytes\n"
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print(
            "\nBUSINESS DISCOVERY ERROR",
            file=sys.stderr
        )

        print(
            str(error),
            file=sys.stderr
        )

        sys.exit(1)