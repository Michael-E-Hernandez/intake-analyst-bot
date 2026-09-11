from pathlib import Path
import argparse
import json
import os
import sys

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
        description="Build an AI-generated semantic data inventory."
    )

    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to csv_metadata.json"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path where data_inventory.md will be saved"
    )

    return parser.parse_args()


# =========================================================
# LOAD METADATA
# =========================================================

def load_metadata(metadata_file: Path):

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file does not exist:\n{metadata_file}"
        )

    with metadata_file.open(
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    if not metadata:
        raise ValueError(
            "Metadata file is empty."
        )

    return metadata


# =========================================================
# GEMINI CLIENT
# =========================================================
def create_gemini_client():

    # Local development:
    # Load .env when it exists.
    #
    # Streamlit Cloud:
    # GEMINI_API_KEY is already supplied through
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
# BUILD PROMPT
# =========================================================

def build_prompt(metadata):

    metadata_text = json.dumps(
        metadata,
        indent=2,
        ensure_ascii=False
    )

    return f"""
You are an AI data analyst examining an unfamiliar business data environment.

Your job is to interpret the available dataset metadata and build a semantic
DATA INVENTORY.

Do not perform business analysis yet.

Do not calculate KPIs.

Do not write SQL or Python.

Do not assume business meanings without evidence.

Your role is to understand:

1. what data exists,
2. what the data appears to represent,
3. how datasets may relate,
4. what kinds of business information may be contained in structured and
   unstructured fields,
5. and what analytical possibilities the data environment may support later.

The goal is NOT to answer a stakeholder's business question yet.

The goal is to create an evidence-grounded understanding of the available
data environment so another AI analyst can later guide a stakeholder through
business discovery and analysis.


IMPORTANT DESIGN PRINCIPLES

1. Do not hard-code business meanings.

2. Infer meaning only from evidence such as:
   - file names
   - column names
   - data types
   - sample values
   - row counts
   - uniqueness
   - null patterns
   - repeated identifiers
   - similarities across datasets
   - relationships between neighboring fields
   - apparent structure of textual fields

3. Treat semantic interpretations as hypotheses.

4. Clearly identify uncertainty.

5. Do not invent relationships that the evidence does not support.

6. Distinguish technical evidence from business interpretation.

7. The system must remain industry-agnostic.

8. Do not assume that all useful business information is numeric or
   categorical.

9. Treat meaningful free-text or long-form text fields as potential
   UNSTRUCTURED BUSINESS DATA.

10. The goal is to help another AI analyst understand what data exists before
    a stakeholder asks a business question.


=========================================================
STRUCTURED AND UNSTRUCTURED DATA
=========================================================

Explicitly distinguish between structured business fields and potentially
unstructured business text.

Structured fields may include things such as:

- identifiers
- categories
- amounts
- quantities
- dates
- statuses
- geographic fields
- demographic attributes
- operational measures
- ratings
- binary indicators

Unstructured or semi-structured fields may include things such as:

- customer comments
- reviews
- survey responses
- support-ticket descriptions
- complaint text
- feedback
- notes
- descriptions
- call notes
- open-ended responses
- written explanations
- other natural-language text

Do NOT identify a field as meaningful unstructured business text merely
because its data type is string/object.

Use evidence such as:

- field name
- sample values
- text length or richness visible in samples
- variation in values
- apparent natural-language structure
- relationship to neighboring fields
- apparent business context

For each likely unstructured business text field, identify:

- dataset
- field
- likely semantic purpose
- whose perspective or source it may represent, if inferable
- whether it appears to contain natural-language information
- what kind of business knowledge it may contain
- potentially useful analytical applications
- structured fields or datasets that could potentially contextualize it
- data quality concerns
- semantic uncertainty
- confidence: High / Medium / Low

Potential analytical applications should remain high-level.

Examples of possible capabilities include:

- sentiment understanding
- recurring theme discovery
- complaint categorization
- praise / positive experience discovery
- customer experience analysis
- topic or issue identification
- feedback summarization
- qualitative-to-quantitative structuring
- connecting qualitative feedback with structured business outcomes
- comparing themes across products, customers, locations, time periods,
  transactions, operational conditions, or other available dimensions

Only list capabilities supported by evidence in the metadata.

Do NOT perform the text analysis now.

Do NOT generate themes from the sample values.

Do NOT decide what customers or users are actually saying.

At this stage, identify only that the text appears capable of supporting
those forms of analysis later.


=========================================================
DATASET-LEVEL UNDERSTANDING
=========================================================

For every dataset, identify:

- likely dataset purpose
- likely row grain
- likely business entity or event represented
- identifiers / candidate keys
- measures
- dimensions / descriptive attributes
- time or date fields
- structured analytical fields
- unstructured / free-text business fields, if present
- potentially useful analytical fields
- potential structured + unstructured analytical connections
- data quality concerns
- semantic uncertainty


Then identify likely relationships between datasets.

For each proposed relationship provide:

- dataset A
- field A
- dataset B
- field B
- likely relationship
- evidence
- confidence: High / Medium / Low


Also identify potentially important semantic distinctions.

For example, if multiple fields appear to represent different versions of
the same business concept, explain the distinction rather than choosing one
without evidence.


=========================================================
OUTPUT STRUCTURE
=========================================================

Create the inventory using the following structure:

# DATA INVENTORY


## 1. Environment Overview

Briefly describe the apparent data environment based only on the evidence.

Mention whether the environment appears to contain:

- structured business data
- unstructured business text
- both

If both exist, briefly explain that the environment may support analysis that
connects qualitative information with structured business context.

Do not perform that analysis yet.


## 2. Dataset Inventory

For each dataset:

### Dataset: <filename>

**Likely purpose:**  
...

**Likely grain:**  
...

**Primary business entity/event:**  
...

**Identifiers / candidate keys:**  
...

**Measures:**  
...

**Dimensions / descriptive fields:**  
...

**Time fields:**  
...

**Structured analytical fields:**  
...

**Unstructured / free-text fields:**  
...

**Potential analytical uses:**  
...

**Potential structured + unstructured connections:**  
...

**Data quality observations:**  
...

**Semantic uncertainty:**  
...

**Confidence:** High / Medium / Low


## 3. Unstructured Business Information

If meaningful unstructured or semi-structured natural-language fields appear
to exist, document them here.

For each field:

### Text Field: <dataset>.<field>

**Likely purpose:**  
...

**Likely source / perspective:**  
...

**Evidence that this is natural-language business information:**  
...

**Potential information contained:**  
...

**Potential analytical applications:**  
...

**Potential structured-data context:**  
...

**Data quality considerations:**  
...

**Semantic uncertainty:**  
...

**Confidence:** High / Medium / Low


If no meaningful unstructured text is supported by the metadata, explicitly
state that no clear unstructured business information was identified.


## 4. Cross-Dataset Relationships

Describe likely relationships and their evidence.


## 5. Shared Business Entities

Describe recurring entities that appear across files such as customers,
orders, products, transactions, locations, sellers, accounts, interactions,
cases, etc.

Only include entities supported by the metadata.


## 6. Structured + Unstructured Analytical Opportunities

Identify situations where unstructured business information could potentially
be analyzed alongside structured fields already present in the environment.

Examples of the TYPE of reasoning expected:

- feedback text alongside satisfaction measures
- comments alongside product attributes
- complaint descriptions alongside operational outcomes
- support messages alongside customer characteristics
- written feedback alongside location or time
- qualitative experience information alongside transaction behavior

These are examples of reasoning patterns only.

Do NOT assume any of them exist unless supported by the supplied metadata.

For every opportunity explain:

- unstructured source
- relevant structured context
- what type of business understanding this combination may support
- evidence
- confidence

Do not calculate results.


## 7. Important Semantic Distinctions

Identify fields or concepts that could easily be misunderstood and may need
clarification later.


## 8. Data Quality and Modeling Risks

Identify issues such as:

- duplicate-looking records
- multiple rows per business entity
- missing values
- unclear identifiers
- many-to-many relationships
- inconsistent grains
- ambiguous date fields
- potential aggregation risks
- sparse text fields
- missing comments or feedback
- duplicate textual records
- extremely short text
- language variation
- text that may not actually represent natural-language feedback


## 9. Unresolved Questions

List semantic questions that cannot confidently be answered from metadata
alone.

These are NOT stakeholder intake questions yet.

They are questions about understanding the data itself.

Examples might involve uncertainty about:

- identifier meaning
- entity grain
- meaning of a text field
- whether multiple records represent revisions or duplicates
- whether text belongs to a customer, employee, system, or another source
- how a textual record relates to another business entity

Do not invent questions unnecessarily.


## 10. Data Understanding Summary

Summarize what another AI analyst should know before using this environment.

Include:

- primary business entities
- major structured information
- meaningful unstructured information
- important relationships
- potential for combining qualitative and quantitative information
- major limitations
- important uncertainties

Do not invent facts.

Use explicit confidence language where appropriate.


=========================================================
FINAL RULE
=========================================================

The purpose of this inventory is to make large and potentially fragmented
business data easier for another AI analyst to understand.

The inventory should identify BOTH:

"What structured facts does this environment contain?"

and, when supported:

"What human language or qualitative business information does this
environment contain?"

Do not analyze the business yet.

Do not summarize customer opinions.

Do not infer actual complaint themes.

Do not produce business conclusions.

Only describe what the available data appears capable of supporting.

Here is the discovered CSV metadata:

{metadata_text}
"""


# =========================================================
# CALL GEMINI
# =========================================================

def generate_inventory(
    client,
    prompt
):

    generation_config = {
        "temperature": 0.2,
        "max_output_tokens": 4096,
        "top_p": 0.95,
        "thinking_level": "low",
    }

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


# =========================================================
# SAVE INVENTORY
# =========================================================

def save_inventory(
    inventory_text: str,
    output_file: Path
):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file.write_text(
        inventory_text,
        encoding="utf-8"
    )

    if not output_file.exists():
        raise RuntimeError(
            f"Inventory file was not created:\n{output_file}"
        )

    if output_file.stat().st_size == 0:
        raise RuntimeError(
            f"Inventory file is empty:\n{output_file}"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    args = get_arguments()

    metadata_file = Path(
        args.metadata
    ).expanduser()

    output_file = Path(
        args.output
    ).expanduser()

    print("\n" + "=" * 80)
    print("AI DATA INVENTORY")
    print("=" * 80)

    print(
        f"\nMetadata input:\n{metadata_file}"
    )

    print(
        f"\nInventory output:\n{output_file}"
    )

    metadata = load_metadata(
        metadata_file
    )

    print(
        f"\nLoaded metadata for {len(metadata)} datasets."
    )

    client = create_gemini_client()

    print(
        "\nGemini client ready."
    )

    prompt = build_prompt(
        metadata
    )

    print(
        "\nGenerating semantic data inventory..."
    )

    inventory_text = generate_inventory(
        client,
        prompt
    )

    save_inventory(
        inventory_text,
        output_file
    )

    print("\n" + "=" * 80)
    print("DATA INVENTORY COMPLETE")
    print("=" * 80)

    print(
        f"\nInventory saved to:\n{output_file}"
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
            "\nDATA INVENTORY ERROR",
            file=sys.stderr
        )

        print(
            str(error),
            file=sys.stderr
        )

        sys.exit(1)