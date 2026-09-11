from pathlib import Path
from datetime import date
import subprocess
import sys

import pandas as pd
import streamlit as st

from intake import generate_next_step
from analysis import run_analysis


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "SRC"

CSV_DISCOVERY_SCRIPT = SRC_DIR / "csv_discovery.py"
DATA_INVENTORY_SCRIPT = SRC_DIR / "data_inventory.py"
BUSINESS_DISCOVERY_SCRIPT = SRC_DIR / "discovery.py"

WORKSPACE_DIR = PROJECT_ROOT / "data" / "processed" / "current"

METADATA_FILE = WORKSPACE_DIR / "csv_metadata.json"
INVENTORY_FILE = WORKSPACE_DIR / "data_inventory.md"
BUSINESS_ENVIRONMENT_FILE = WORKSPACE_DIR / "business_environment.md"


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Intake Analyst Bot",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "connected_folder": "",
    "csv_discovery_complete": False,
    "data_inventory_complete": False,
    "business_discovery_complete": False,
    "environment_ready": False,

    # Intake
    "stakeholder_goal": "",
    "intake_history": [],
    "current_intake_step": None,
    "analysis_ready": False,
    "structured_question": None,

    # Requested output
    "output_preference": None,
    "output_ready": False,

    # Final analysis
    "analysis_complete": False,
    "analysis_result": None,

    # Post-intake / post-analysis routing
    "post_intake_path": None,
    "post_analysis_path": None,
    "continuation_instruction": "",
    "analysis_history": [],
}

for key, value in defaults.items():

    if key not in st.session_state:

        if isinstance(value, list):
            st.session_state[key] = []

        else:
            st.session_state[key] = value


# =========================================================
# RESET INTAKE
# =========================================================

def reset_intake():

    st.session_state.stakeholder_goal = ""
    st.session_state.intake_history = []
    st.session_state.current_intake_step = None
    st.session_state.analysis_ready = False
    st.session_state.structured_question = None
    st.session_state.output_preference = None
    st.session_state.output_ready = False
    st.session_state.analysis_complete = False
    st.session_state.analysis_result = None
    st.session_state.post_intake_path = None
    st.session_state.post_analysis_path = None
    st.session_state.continuation_instruction = ""
    st.session_state.analysis_history = []


# =========================================================
# RESET ENVIRONMENT
# =========================================================

def reset_environment():

    st.session_state.connected_folder = ""
    st.session_state.csv_discovery_complete = False
    st.session_state.data_inventory_complete = False
    st.session_state.business_discovery_complete = False
    st.session_state.environment_ready = False

    reset_intake()


# =========================================================
# STEP 1 — CSV DISCOVERY
# =========================================================

def run_csv_discovery(data_folder: Path):

    WORKSPACE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if METADATA_FILE.exists():
        METADATA_FILE.unlink()

    command = [
        sys.executable,
        str(CSV_DISCOVERY_SCRIPT),
        "--data-dir",
        str(data_folder),
        "--output",
        str(METADATA_FILE),
    ]

    process = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    if process.returncode != 0:
        raise RuntimeError(
            "csv_discovery.py failed.\n\n"
            f"{process.stderr}"
        )

    if not METADATA_FILE.exists():
        raise RuntimeError(
            "csv_metadata.json was not created."
        )


# =========================================================
# STEP 2 — DATA INVENTORY
# =========================================================

def run_data_inventory():

    if INVENTORY_FILE.exists():
        INVENTORY_FILE.unlink()

    command = [
        sys.executable,
        str(DATA_INVENTORY_SCRIPT),
        "--metadata",
        str(METADATA_FILE),
        "--output",
        str(INVENTORY_FILE),
    ]

    process = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    if process.returncode != 0:
        raise RuntimeError(
            "data_inventory.py failed.\n\n"
            f"{process.stderr}"
        )

    if not INVENTORY_FILE.exists():
        raise RuntimeError(
            "data_inventory.md was not created."
        )


# =========================================================
# STEP 3 — BUSINESS DISCOVERY
# =========================================================

def run_business_discovery():

    if BUSINESS_ENVIRONMENT_FILE.exists():
        BUSINESS_ENVIRONMENT_FILE.unlink()

    command = [
        sys.executable,
        str(BUSINESS_DISCOVERY_SCRIPT),
        "--inventory",
        str(INVENTORY_FILE),
        "--output",
        str(BUSINESS_ENVIRONMENT_FILE),
    ]

    process = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    if process.returncode != 0:
        raise RuntimeError(
            "discovery.py failed.\n\n"
            f"{process.stderr}"
        )

    if not BUSINESS_ENVIRONMENT_FILE.exists():
        raise RuntimeError(
            "business_environment.md was not created."
        )


# =========================================================
# GET NEXT INTAKE STEP
# =========================================================

def get_next_intake_step():

    return generate_next_step(
        metadata_path=METADATA_FILE,
        inventory_path=INVENTORY_FILE,
        business_environment_path=BUSINESS_ENVIRONMENT_FILE,
        stakeholder_goal=st.session_state.stakeholder_goal,
        intake_history=st.session_state.intake_history,
    )


# =========================================================
# RUN FINAL ANALYSIS
# =========================================================

def execute_final_analysis(question_override=None, requested_output_override=None):

    business_question = (
        question_override
        if question_override is not None
        else st.session_state.structured_question
    )

    requested_output = (
        requested_output_override
        if requested_output_override is not None
        else st.session_state.output_preference
    )

    return run_analysis(
        data_dir=st.session_state.connected_folder,
        metadata_path=METADATA_FILE,
        inventory_path=INVENTORY_FILE,
        business_environment_path=BUSINESS_ENVIRONMENT_FILE,
        business_question=business_question,
        requested_output=requested_output,
    )


# =========================================================
# RESULT NORMALIZATION
# =========================================================

def normalize_insight(insight, index=1):
    """Normalize a key insight for safe rendering."""
    if isinstance(insight, dict):
        title = str(insight.get("title", f"Finding {index}")).strip()
        insight_text = str(insight.get("insight", "")).strip()
    elif isinstance(insight, str):
        title = f"Finding {index}"
        insight_text = insight.strip()
    else:
        title = f"Finding {index}"
        insight_text = str(insight).strip()
    if not title:
        title = f"Finding {index}"
    return title, insight_text


# =========================================================
# ANALYST HANDOFF
# =========================================================

def build_analyst_handoff_text():

    connected_files = []

    if st.session_state.connected_folder:

        connected_path = Path(
            st.session_state.connected_folder
        )

        if connected_path.exists():

            connected_files = sorted(
                connected_path.glob("*.csv")
            )

    lines = []

    lines.append("ANALYST HANDOFF")
    lines.append("=" * 60)
    lines.append("")

    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    lines.append("STATUS")
    lines.append("-" * 60)

    if st.session_state.analysis_complete:

        lines.append(
            "Self-service analysis completed — "
            "ready for analyst review or further analysis."
        )

    else:

        lines.append(
            "Intake complete — ready for analyst review."
        )

    lines.append("")

    # -----------------------------------------------------
    # ORIGINAL REQUEST
    # -----------------------------------------------------

    lines.append("ORIGINAL STAKEHOLDER REQUEST")
    lines.append("-" * 60)

    lines.append(
        st.session_state.stakeholder_goal
        or "Not provided"
    )

    lines.append("")

    # -----------------------------------------------------
    # STRUCTURED QUESTION
    # -----------------------------------------------------

    lines.append("STRUCTURED BUSINESS QUESTION")
    lines.append("-" * 60)

    lines.append(
        st.session_state.structured_question
        or "Not yet defined"
    )

    lines.append("")

    # -----------------------------------------------------
    # INTAKE DECISIONS
    # -----------------------------------------------------

    lines.append("INTAKE & DISCOVERY DECISIONS")
    lines.append("-" * 60)

    if st.session_state.intake_history:

        for index, turn in enumerate(
            st.session_state.intake_history,
            start=1,
        ):

            question = turn.get(
                "question",
                "Question",
            )

            answer = turn.get(
                "answer",
                "",
            )

            if isinstance(
                answer,
                list,
            ):

                answer_text = ", ".join(
                    str(item)
                    for item in answer
                )

            else:

                answer_text = str(
                    answer
                )

            lines.append(
                f"{index}. {question}"
            )

            lines.append(
                f"   Decision: {answer_text}"
            )

            lines.append("")

    else:

        lines.append(
            "No additional clarification was required."
        )

        lines.append("")

    # -----------------------------------------------------
    # CONNECTED ENVIRONMENT
    # -----------------------------------------------------

    lines.append("CONNECTED DATA ENVIRONMENT")
    lines.append("-" * 60)

    lines.append(
        f"Connected CSV sources: "
        f"{len(connected_files)}"
    )

    for csv_file in connected_files:

        lines.append(
            f"- {csv_file.name}"
        )

    lines.append("")

    lines.append(
        "Environment understanding:"
    )

    lines.append(
        "- Data structure profiled"
    )

    lines.append(
        "- Relationships and semantics interpreted"
    )

    lines.append(
        "- Business context understood"
    )

    lines.append(
        "- Analytical capabilities identified"
    )

    lines.append("")

    # -----------------------------------------------------
    # SELF-SERVICE ANALYSIS
    # -----------------------------------------------------

    lines.append("SELF-SERVICE ANALYSIS")
    lines.append("-" * 60)

    if not st.session_state.analysis_complete:

        lines.append(
            "Status: Not performed"
        )

        if st.session_state.output_preference:

            lines.append(
                f"Requested output: "
                f"{st.session_state.output_preference}"
            )

        lines.append("")

        lines.append(
            "The stakeholder stopped after intake. "
            "The structured request is ready for a "
            "human analyst to continue."
        )

    else:

        result = (
            st.session_state.analysis_result
            or {}
        )

        lines.append(
            "Status: Completed"
        )

        lines.append(
            f"Requested output: "
            f"{st.session_state.output_preference}"
        )

        records_analyzed = result.get(
            "records_analyzed"
        )

        if records_analyzed is not None:

            lines.append(
                f"Records analyzed: "
                f"{records_analyzed:,}"
            )

        lines.append("")

        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

        summary = result.get(
            "summary"
        )

        if summary:

            lines.append(
                "ANALYSIS SUMMARY"
            )

            lines.append(
                "-" * 60
            )

            lines.append(
                summary
            )

            lines.append("")

        # -------------------------------------------------
        # KEY FINDINGS
        # -------------------------------------------------

        key_insights = result.get(
            "key_insights",
            [],
        )

        if key_insights:

            lines.append(
                "KEY FINDINGS"
            )

            lines.append(
                "-" * 60
            )

            for index, insight in enumerate(
                key_insights,
                start=1,
            ):

                title, insight_text = normalize_insight(
                    insight,
                    index=index,
                )

                if not insight_text:
                    continue

                lines.append(
                    f"- {title}: {insight_text}"
                )

            lines.append("")

        # -------------------------------------------------
        # VISUALIZATIONS
        # -------------------------------------------------

        visualizations = result.get(
            "visualizations",
            [],
        )

        if visualizations:

            lines.append(
                "VISUALIZATIONS PRODUCED"
            )

            lines.append(
                "-" * 60
            )

            for viz in visualizations:

                title = viz.get(
                    "title",
                    "Visualization",
                )

                chart_type = viz.get(
                    "chart_type",
                    "chart",
                )

                lines.append(
                    f"- {title} ({chart_type})"
                )

            lines.append("")

        # -------------------------------------------------
        # CAVEATS
        # -------------------------------------------------

        caveats = result.get(
            "caveats",
            [],
        )

        if caveats:

            lines.append(
                "ANALYSIS CONSIDERATIONS"
            )

            lines.append(
                "-" * 60
            )

            for caveat in caveats:

                lines.append(
                    f"- {caveat}"
                )

            lines.append("")

    # -----------------------------------------------------
    # FINAL NOTE
    # -----------------------------------------------------

    lines.append("HANDOFF NOTE")
    lines.append("-" * 60)

    if st.session_state.analysis_complete:

        lines.append(
            "The intake context and completed "
            "self-service analysis are preserved so "
            "a human analyst can review, validate, "
            "extend, or continue the work."
        )

    else:

        lines.append(
            "The intake context is preserved so a "
            "human analyst can begin with a defined "
            "business need rather than restarting "
            "stakeholder discovery."
        )

    return "\n".join(
        lines
    )


# =========================================================
# RENDER ANALYST HANDOFF
# =========================================================

def render_analyst_handoff(key_suffix="default"):

    handoff_text = (
        build_analyst_handoff_text()
    )

    with st.expander(
        "📋 View Analyst Handoff Summary",
        expanded=False,
    ):

        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        if st.session_state.analysis_complete:

            st.success(
                "Self-service analysis completed — "
                "ready for analyst review or further analysis."
            )

        else:

            st.success(
                "Intake complete — ready for analyst review."
            )

        # -------------------------------------------------
        # ORIGINAL REQUEST
        # -------------------------------------------------

        st.subheader(
            "Original Stakeholder Request"
        )

        st.write(
            st.session_state.stakeholder_goal
        )

        # -------------------------------------------------
        # STRUCTURED QUESTION
        # -------------------------------------------------

        st.subheader(
            "Structured Business Question"
        )

        st.write(
            st.session_state.structured_question
        )

        # -------------------------------------------------
        # INTAKE HISTORY
        # -------------------------------------------------

        st.subheader(
            "Intake & Discovery Decisions"
        )

        if st.session_state.intake_history:

            for turn in st.session_state.intake_history:

                st.markdown(
                    f"**{turn['question']}**"
                )

                answer = turn[
                    "answer"
                ]

                if isinstance(
                    answer,
                    list,
                ):

                    answer_text = ", ".join(
                        str(item)
                        for item in answer
                    )

                else:

                    answer_text = str(
                        answer
                    )

                st.write(
                    f"→ {answer_text}"
                )

        else:

            st.write(
                "No additional clarification was required."
            )

        # -------------------------------------------------
        # DATA ENVIRONMENT
        # -------------------------------------------------

        st.subheader(
            "Connected Data Environment"
        )

        connected_folder = st.session_state.connected_folder

        if connected_folder:
            connected_path = Path(
                connected_folder
            )

            csv_files = (
                sorted(connected_path.glob("*.csv"))
                if connected_path.exists()
                else []
            )
        else:
            csv_files = []

        st.write(
            f"{len(csv_files)} connected CSV sources"
        )

        with st.expander(
            "View connected sources"
        ):

            for csv_file in csv_files:

                st.write(
                    f"• {csv_file.name}"
                )

        st.write(
            "✓ Data structure profiled"
        )

        st.write(
            "✓ Relationships and semantics interpreted"
        )

        st.write(
            "✓ Business context understood"
        )

        st.write(
            "✓ Analytical capabilities identified"
        )

        # -------------------------------------------------
        # ANALYSIS
        # -------------------------------------------------

        st.subheader(
            "Self-Service Analysis"
        )

        if not st.session_state.analysis_complete:

            st.info(
                "Self-service analysis has not been performed. "
                "The structured request is ready for a human "
                "analyst to continue."
            )

        else:

            result = (
                st.session_state.analysis_result
                or {}
            )

            st.write(
                f"**Requested output:** "
                f"{st.session_state.output_preference}"
            )

            records_analyzed = result.get(
                "records_analyzed"
            )

            if records_analyzed is not None:

                st.write(
                    f"**Records analyzed:** "
                    f"{records_analyzed:,}"
                )

            summary = result.get(
                "summary"
            )

            if summary:

                st.markdown(
                    "**Analysis Summary**"
                )

                st.write(
                    summary
                )

            key_insights = result.get(
                "key_insights",
                [],
            )

            if key_insights:

                st.markdown(
                    "**Key Findings**"
                )

                for index, insight in enumerate(
                    key_insights,
                    start=1,
                ):

                    title, insight_text = normalize_insight(
                        insight,
                        index=index,
                    )

                    if not insight_text:
                        continue

                    st.write(
                        f"• {title}: {insight_text}"
                    )

            visualizations = result.get(
                "visualizations",
                [],
            )

            if visualizations:

                st.markdown(
                    "**Visualizations Produced**"
                )

                for viz in visualizations:

                    title = viz.get(
                        "title",
                        "Visualization",
                    )

                    chart_type = viz.get(
                        "chart_type",
                        "chart",
                    )

                    st.write(
                        f"• {title} ({chart_type})"
                    )

            caveats = result.get(
                "caveats",
                [],
            )

            if caveats:

                unique_caveats = list(
                    dict.fromkeys(
                        str(caveat).strip()
                        for caveat in caveats
                        if str(caveat).strip()
                    )
                )

                st.markdown(
                    "**Analysis Considerations**"
                )

                for caveat in unique_caveats:

                    st.write(
                        f"• {caveat}"
                    )

        # -------------------------------------------------
        # HANDOFF PURPOSE
        # -------------------------------------------------

        st.subheader(
            "Handoff Purpose"
        )

        if st.session_state.analysis_complete:

            st.write(
                "The stakeholder intake and completed analysis "
                "are preserved so a human analyst can review, "
                "validate, extend, or continue the work without "
                "restarting the discovery process."
            )

        else:

            st.write(
                "The stakeholder intake is preserved so a human "
                "analyst can begin with an organized and clearly "
                "defined business request instead of restarting "
                "the discovery process."
            )

        st.divider()

        st.download_button(
            label="Download Analyst Handoff",
            data=handoff_text,
            file_name="analyst_handoff.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"download_analyst_handoff_{key_suffix}",
        )


# =========================================================
# CHART RENDERING
# =========================================================

def render_visualization(viz):

    data = viz.get(
        "data",
        [],
    )

    if not data:
        return

    df = pd.DataFrame(data)

    if df.empty:
        return

    st.subheader(
        viz.get(
            "title",
            "Visualization",
        )
    )

    chart_type = viz.get(
        "chart_type",
        "bar",
    )

    # -----------------------------------------------------
    # Try to infer usable columns from result data
    # -----------------------------------------------------

    columns = list(
        df.columns
    )

    numeric_columns = [
        column
        for column in columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    non_numeric_columns = [
        column
        for column in columns
        if column not in numeric_columns
    ]

    if not numeric_columns:

        st.dataframe(
            df,
            use_container_width=True,
        )

        return

    y_column = numeric_columns[-1]

    if non_numeric_columns:

        x_column = non_numeric_columns[0]

    else:

        x_column = columns[0]

    chart_df = df.copy()

    if x_column in chart_df.columns:

        chart_df = chart_df.set_index(
            x_column
        )

    # -----------------------------------------------------
    # Render
    # -----------------------------------------------------

    if chart_type == "line":

        st.line_chart(
            chart_df[
                [y_column]
            ],
            use_container_width=True,
        )

    elif chart_type == "scatter":

        if len(numeric_columns) >= 2:

            st.scatter_chart(
                df,
                x=numeric_columns[0],
                y=numeric_columns[1],
                use_container_width=True,
            )

        else:

            st.bar_chart(
                chart_df[
                    [y_column]
                ],
                use_container_width=True,
            )

    else:

        st.bar_chart(
            chart_df[
                [y_column]
            ],
            use_container_width=True,
        )

    with st.expander(
        "View chart data"
    ):

        st.dataframe(
            df,
            use_container_width=True,
        )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title(
        "📊 Intake Analyst Bot"
    )

    # -----------------------------------------------------
    # NOT CONNECTED
    # -----------------------------------------------------

    if not st.session_state.environment_ready:

        st.subheader(
            "Connect Data"
        )

        folder_input = st.text_input(
            "Data folder",
            value=st.session_state.connected_folder,
            placeholder=r"G:\My Drive\AI AGENT\data\raw",
        )

        if folder_input:

            selected_folder = Path(
                folder_input
                .strip()
                .strip('"')
                .strip("'")
            )

            if not selected_folder.exists():

                st.error(
                    "That folder does not exist."
                )

            elif not selected_folder.is_dir():

                st.error(
                    "The selected path is not a folder."
                )

            else:

                csv_files = sorted(
                    selected_folder.glob("*.csv")
                )

                if not csv_files:

                    st.warning(
                        "No CSV files were found."
                    )

                else:

                    st.success(
                        f"Found {len(csv_files)} CSV files."
                    )

                    with st.expander(
                        "View detected files"
                    ):

                        for csv_file in csv_files:

                            st.write(
                                f"• {csv_file.name}"
                            )

                    if st.button(
                        "Connect Data",
                        type="primary",
                        use_container_width=True,
                    ):

                        try:

                            st.session_state.connected_folder = str(
                                selected_folder.resolve()
                            )

                            with st.status(
                                "Understanding data environment...",
                                expanded=True,
                            ) as status:

                                st.write(
                                    "Profiling data structure..."
                                )

                                run_csv_discovery(
                                    selected_folder
                                )

                                st.session_state.csv_discovery_complete = True

                                st.write(
                                    "✓ Structure profiled"
                                )

                                st.write(
                                    "Interpreting relationships and meaning..."
                                )

                                run_data_inventory()

                                st.session_state.data_inventory_complete = True

                                st.write(
                                    "✓ Semantic inventory created"
                                )

                                st.write(
                                    "Understanding business environment..."
                                )

                                run_business_discovery()

                                st.session_state.business_discovery_complete = True

                                st.write(
                                    "✓ Business context understood"
                                )

                                st.write(
                                    "✓ Analytical capabilities identified"
                                )

                                st.session_state.environment_ready = True

                                status.update(
                                    label="Data environment ready",
                                    state="complete",
                                    expanded=False,
                                )

                            st.rerun()

                        except Exception as error:

                            st.error(
                                "Data connection failed."
                            )

                            st.code(
                                str(error)
                            )

    # -----------------------------------------------------
    # CONNECTED
    # -----------------------------------------------------

    else:

        st.subheader(
            "Connected Environment"
        )

        connected_path = Path(
            st.session_state.connected_folder
        )

        csv_files = sorted(
            connected_path.glob("*.csv")
        )

        st.success(
            f"✓ {len(csv_files)} CSV files connected"
        )

        st.divider()

        st.subheader(
            "Data Understanding"
        )

        st.success(
            "✓ Structure profiled"
        )

        st.success(
            "✓ Relationships and semantics interpreted"
        )

        st.success(
            "✓ Business context understood"
        )

        st.divider()

        st.subheader(
            "Analytical Readiness"
        )

        st.success(
            "✓ Capabilities identified"
        )

        st.success(
            "✓ READY FOR INTAKE"
        )

        st.divider()

        if INVENTORY_FILE.exists():

            with st.expander(
                "View Data Inventory"
            ):

                st.text(
                    INVENTORY_FILE.read_text(
                        encoding="utf-8"
                    )
                )

        if BUSINESS_ENVIRONMENT_FILE.exists():

            with st.expander(
                "View Capability Map"
            ):

                st.text(
                    BUSINESS_ENVIRONMENT_FILE.read_text(
                        encoding="utf-8"
                    )
                )

        st.divider()

        if st.button(
            "New Business Question",
            use_container_width=True,
        ):

            reset_intake()

            st.rerun()

        if st.button(
            "Connect Different Data",
            use_container_width=True,
        ):

            reset_environment()

            st.rerun()


# =========================================================
# MAIN WORKSPACE
# =========================================================

st.title(
    "Intake Analyst Bot"
)


# =========================================================
# WAITING FOR DATA
# =========================================================

if not st.session_state.environment_ready:

    st.header(
        "Business Discovery Starts With Your Data"
    )

    st.write(
        "Connect your data from the sidebar. "
        "The system will first understand what information "
        "is available and what kinds of analysis it can support."
    )


# =========================================================
# READY FOR BUSINESS INTAKE
# =========================================================

else:

    st.header(
        "How can I help you today?"
    )

    # =====================================================
    # FIRST STAKEHOLDER REQUEST
    # =====================================================

    if not st.session_state.stakeholder_goal:

        st.write(
            "Tell me what you're trying to understand "
            "about your business."
        )

        stakeholder_goal = st.text_area(
            "Business question or goal",
            placeholder=(
                "Example: I want to understand why "
                "some customers are unhappy."
            ),
            height=120,
        )

        if st.button(
            "Continue",
            type="primary",
        ):

            if not stakeholder_goal.strip():

                st.warning(
                    "Enter what you want to understand."
                )

            else:

                st.session_state.stakeholder_goal = (
                    stakeholder_goal.strip()
                )

                try:

                    with st.spinner(
                        "Reviewing what your data can support..."
                    ):

                        step = get_next_intake_step()

                        st.session_state.current_intake_step = (
                            step
                        )

                        if step[
                            "ready_for_analysis"
                        ]:

                            st.session_state.analysis_ready = True

                            st.session_state.structured_question = (
                                step[
                                    "structured_question"
                                ]
                            )

                    st.rerun()

                except Exception as error:

                    st.error(
                        "The intake engine could not process the request."
                    )

                    st.code(
                        str(error)
                    )


    # =====================================================
    # EXISTING INTAKE
    # =====================================================

    else:

        # -------------------------------------------------
        # ORIGINAL BUSINESS NEED
        # -------------------------------------------------

        with st.chat_message(
            "user"
        ):

            st.write(
                st.session_state.stakeholder_goal
            )

        # -------------------------------------------------
        # PREVIOUS INTAKE TURNS
        # -------------------------------------------------

        for turn in st.session_state.intake_history:

            with st.chat_message(
                "assistant"
            ):

                st.write(
                    turn[
                        "assistant_message"
                    ]
                )

                st.write(
                    f"**{turn['question']}**"
                )

            with st.chat_message(
                "user"
            ):

                answer = turn[
                    "answer"
                ]

                if isinstance(
                    answer,
                    list,
                ):

                    st.write(
                        ", ".join(
                            str(item)
                            for item in answer
                        )
                    )

                else:

                    st.write(
                        str(answer)
                    )


        # =================================================
        # BUSINESS QUESTION DEFINED
        # =================================================

        if st.session_state.analysis_ready:

            current_step = (
                st.session_state.current_intake_step
            )

            if current_step:

                with st.chat_message(
                    "assistant"
                ):

                    st.write(
                        current_step[
                            "assistant_message"
                        ]
                    )

            st.success(
                "Business question confirmed"
            )

            st.subheader(
                "Structured Business Question"
            )

            st.write(
                st.session_state.structured_question
            )

            st.divider()

            # =================================================
            # POST-INTAKE ROUTING — THREE PATHS
            # =================================================

            if not st.session_state.analysis_complete:

                st.subheader(
                    "Your analytical question is ready"
                )

                st.write(
                    "Choose how you want to continue."
                )

                route_col1, route_col2, route_col3 = st.columns(3)

                with route_col1:
                    if st.button(
                        "Analyze Here",
                        type="primary",
                        use_container_width=True,
                        key="post_intake_analyze_here",
                    ):
                        st.session_state.post_intake_path = "analyze"
                        st.rerun()

                with route_col2:
                    if st.button(
                        "Use With AI",
                        use_container_width=True,
                        key="post_intake_use_with_ai",
                    ):
                        st.session_state.post_intake_path = "ai"
                        st.rerun()

                with route_col3:
                    if st.button(
                        "Escalate to Analytics Team",
                        use_container_width=True,
                        key="post_intake_escalate",
                    ):
                        st.session_state.post_intake_path = "analyst"
                        st.rerun()

                selected_path = st.session_state.post_intake_path

                if selected_path == "ai":

                    st.subheader(
                        "Use With AI"
                    )

                    st.write(
                        "Copy the confirmed analytical question into ChatGPT, "
                        "Gemini, Claude, your approved enterprise AI, or another "
                        "analytics tool."
                    )

                    st.code(
                        st.session_state.structured_question,
                        language=None,
                    )

                    st.download_button(
                        "Download Confirmed Analytical Question",
                        data=st.session_state.structured_question,
                        file_name="confirmed_analytical_question.txt",
                        mime="text/plain",
                        key="download_confirmed_question",
                    )

                    render_analyst_handoff(
                        key_suffix="use_with_ai_intake"
                    )

                elif selected_path == "analyst":

                    st.subheader(
                        "Escalate to Analytics Team"
                    )

                    st.write(
                        "The discovery context and confirmed analytical question "
                        "are ready to hand to a human analyst."
                    )

                    render_analyst_handoff(
                        key_suffix="intake_escalation"
                    )

                elif selected_path == "analyze":

                    # =========================================
                    # OUTPUT PREFERENCE
                    # =========================================

                    if not st.session_state.output_ready:

                        with st.chat_message(
                            "assistant"
                        ):

                            st.write(
                                "What would be most useful for you to receive?"
                            )

                        output_preference = st.radio(
                            "Select an output",
                            options=[
                                "Key insights",
                                "Visualization",
                                "Insights + visualization",
                                "Analysis recommendations",
                            ],
                            index=None,
                            key="output_preference_selector",
                        )

                        if st.button(
                            "Run Analysis",
                            type="primary",
                            key="run_analysis_choice",
                        ):

                            if output_preference is None:

                                st.warning(
                                    "Choose what you would like me to provide."
                                )

                            else:

                                st.session_state.output_preference = (
                                    output_preference
                                )

                                st.session_state.output_ready = True

                                try:

                                    with st.status(
                                        "Analyzing connected data...",
                                        expanded=True,
                                    ) as status:

                                        st.write(
                                            "Selecting relevant data..."
                                        )

                                        st.write(
                                            "Running calculations..."
                                        )

                                        result = execute_final_analysis()

                                        st.write(
                                            "Interpreting calculated results..."
                                        )

                                        st.session_state.analysis_result = (
                                            result
                                        )

                                        st.session_state.analysis_complete = True
                                        st.session_state.post_analysis_path = None

                                        status.update(
                                            label="Analysis complete",
                                            state="complete",
                                            expanded=False,
                                        )

                                    st.rerun()

                                except Exception as error:

                                    st.session_state.output_ready = False

                                    st.error(
                                        "The analysis could not be completed."
                                    )

                                    st.code(
                                        str(error)
                                    )

                else:

                    st.info(
                        "Choose one of the three paths above to continue."
                    )


            # =================================================
            # FINAL RESULT
            # =================================================

            elif st.session_state.analysis_complete:

                result = (
                    st.session_state.analysis_result
                )

                with st.chat_message(
                    "user"
                ):

                    st.write(
                        st.session_state.output_preference
                    )

                st.success(
                    "Analysis complete"
                )

                records_analyzed = result.get(
                    "records_analyzed"
                )

                if records_analyzed is not None:

                    st.caption(
                        f"{records_analyzed:,} records analyzed"
                    )

                # ---------------------------------------------
                # SUMMARY
                # ---------------------------------------------

                if (
                    st.session_state.output_preference
                    != "Visualization"
                ):

                    st.subheader(
                        "What I found"
                    )

                    summary = result.get(
                        "summary"
                    )

                    if summary:

                        st.write(
                            summary
                        )

                    # -----------------------------------------
                    # KEY INSIGHTS
                    # -----------------------------------------

                    key_insights = result.get(
                        "key_insights",
                        [],
                    )

                    if key_insights:

                        st.subheader(
                            "Key Insights"
                        )

                        for index, insight in enumerate(
                            key_insights,
                            start=1,
                        ):

                            title, insight_text = normalize_insight(
                                insight,
                                index=index,
                            )

                            if not insight_text:
                                continue

                            st.markdown(
                                f"**{title}**"
                            )

                            st.write(
                                insight_text
                            )


                # ---------------------------------------------
                # VISUALIZATIONS
                # ---------------------------------------------

                if (
                    st.session_state.output_preference
                    in [
                        "Visualization",
                        "Insights + visualization",
                    ]
                ):

                    visualizations = result.get(
                        "visualizations",
                        [],
                    )

                    if visualizations:

                        st.subheader(
                            "Visualizations"
                        )

                        for viz in visualizations:

                            render_visualization(
                                viz
                            )

                    else:

                        st.info(
                            "The analysis completed, but no "
                            "visualization was produced for this request."
                        )


                # ---------------------------------------------
                # CAVEATS
                # ---------------------------------------------

                caveats = result.get(
                    "caveats",
                    [],
                )

                if caveats:

                    unique_caveats = list(
                        dict.fromkeys(
                            str(caveat).strip()
                            for caveat in caveats
                            if str(caveat).strip()
                        )
                    )

                    with st.expander(
                        "Analysis notes"
                    ):

                        for caveat in unique_caveats:

                            st.write(
                                f"• {caveat}"
                            )

                st.divider()

                # ---------------------------------------------
                # POST-ANALYSIS ROUTING — THREE PATHS
                # ---------------------------------------------

                st.subheader(
                    "What would you like to do next?"
                )

                next_col1, next_col2, next_col3 = st.columns(3)

                with next_col1:
                    if st.button(
                        "Continue Analysis",
                        type="primary",
                        use_container_width=True,
                        key="post_analysis_continue",
                    ):
                        st.session_state.post_analysis_path = "continue"
                        st.rerun()

                with next_col2:
                    if st.button(
                        "Use With AI",
                        use_container_width=True,
                        key="post_analysis_use_with_ai",
                    ):
                        st.session_state.post_analysis_path = "ai"
                        st.rerun()

                with next_col3:
                    if st.button(
                        "Escalate to Analytics Team",
                        use_container_width=True,
                        key="post_analysis_escalate",
                    ):
                        st.session_state.post_analysis_path = "analyst"
                        st.rerun()

                next_path = st.session_state.post_analysis_path

                if next_path == "continue":

                    st.subheader(
                        "Continue the Investigation"
                    )

                    st.write(
                        "Tell the analyst what you want to understand next. "
                        "You can react naturally to the findings."
                    )

                    continuation_instruction = st.text_area(
                        "Follow-up",
                        value=st.session_state.continuation_instruction,
                        placeholder=(
                            "Example: Overtime is the obvious factor. "
                            "Let's understand more about what's happening around it."
                        ),
                        key="continuation_instruction_input",
                    )

                    if st.button(
                        "Run Follow-up Analysis",
                        type="primary",
                        key="run_followup_analysis",
                    ):

                        if not continuation_instruction.strip():

                            st.warning(
                                "Tell me what you want to investigate next."
                            )

                        else:

                            st.session_state.continuation_instruction = (
                                continuation_instruction.strip()
                            )

                            prior_result = st.session_state.analysis_result or {}
                            prior_summary = str(
                                prior_result.get("summary", "")
                            ).strip()

                            prior_insights = prior_result.get(
                                "key_insights",
                                [],
                            ) or []

                            insight_lines = []
                            for idx, item in enumerate(prior_insights, start=1):
                                title, body = normalize_insight(
                                    item,
                                    index=idx,
                                )
                                if body:
                                    insight_lines.append(
                                        f"- {title}: {body}"
                                    )

                            continuation_question = f"""
ORIGINAL CONFIRMED ANALYTICAL QUESTION:
{st.session_state.structured_question}

PREVIOUS COMPUTED ANALYSIS SUMMARY:
{prior_summary}

PREVIOUS KEY FINDINGS:
{chr(10).join(insight_lines)}

STAKEHOLDER FOLLOW-UP:
{continuation_instruction.strip()}

Continue the investigation from the stakeholder's follow-up. Use the connected
data to recompute and verify every numeric claim. Treat the previous findings as
context, not as unquestioned truth. Investigate the most decision-relevant next
relationships, interactions, or segments supported by the data and the
stakeholder's request. Do not make causal claims from observational evidence.
""".strip()

                            try:

                                with st.status(
                                    "Continuing the investigation...",
                                    expanded=True,
                                ) as status:

                                    st.write(
                                        "Reviewing the previous findings..."
                                    )

                                    st.write(
                                        "Testing the next useful relationships..."
                                    )

                                    followup_result = execute_final_analysis(
                                        question_override=continuation_question,
                                        requested_output_override=(
                                            st.session_state.output_preference
                                        ),
                                    )

                                    if st.session_state.analysis_result:
                                        st.session_state.analysis_history.append(
                                            st.session_state.analysis_result
                                        )

                                    st.session_state.analysis_result = (
                                        followup_result
                                    )

                                    st.session_state.post_analysis_path = None

                                    status.update(
                                        label="Follow-up analysis complete",
                                        state="complete",
                                        expanded=False,
                                    )

                                st.rerun()

                            except Exception as error:

                                st.error(
                                    "The follow-up analysis could not be completed."
                                )

                                st.code(
                                    str(error)
                                )

                elif next_path == "ai":

                    st.subheader(
                        "Use With AI"
                    )

                    ai_context = build_analyst_handoff_text()

                    st.write(
                        "Take the confirmed question and completed analysis "
                        "context into another AI or analytics tool."
                    )

                    st.code(
                        st.session_state.structured_question,
                        language=None,
                    )

                    st.download_button(
                        "Download Analysis Context for AI",
                        data=ai_context,
                        file_name="analysis_context_for_ai.txt",
                        mime="text/plain",
                        key="download_analysis_context_ai",
                    )

                    with st.expander(
                        "View context to carry forward",
                        expanded=False,
                    ):
                        st.text(
                            ai_context
                        )

                elif next_path == "analyst":

                    st.subheader(
                        "Escalate to Analytics Team"
                    )

                    st.write(
                        "The confirmed question, discovery context, completed "
                        "analysis, findings, and limitations are ready for review "
                        "or deeper work by a human analyst."
                    )

                    render_analyst_handoff(
                        key_suffix="completed_analysis"
                    )

                st.divider()

                if st.button(
                    "Ask Another Question",
                    key="another_question_bottom",
                ):

                    reset_intake()

                    st.rerun()


            # =================================================
            # OUTPUT CHOSEN BUT ANALYSIS NOT COMPLETE
            # =================================================

            else:

                st.warning(
                    "The analysis has not completed."
                )


        # =================================================
        # CURRENT CLARIFICATION STEP
        # =================================================

        else:

            step = (
                st.session_state.current_intake_step
            )

            if step:

                with st.chat_message(
                    "assistant"
                ):

                    st.write(
                        step[
                            "assistant_message"
                        ]
                    )

                    st.markdown(
                        f"**{step['question']}**"
                    )

                input_type = step.get(
                    "input_type"
                )

                # Keep compatible choices multi-select even if an LLM response
                # accidentally labels the widget as radio. The intake engine also
                # enforces this, but the UI treats option_relationship as the
                # authoritative interaction contract.
                option_relationship = step.get("option_relationship")
                if option_relationship == "compatible" and input_type in {"radio", "multiselect"}:
                    input_type = "multiselect"
                elif option_relationship == "mutually_exclusive" and input_type in {"radio", "multiselect"}:
                    input_type = "radio"

                answer = None
                additional_context = ""

                # Give every intake turn its own widget identity.
                # This is especially important now that intake can move
                # through clarification -> restatement -> confirmation.
                intake_turn_key = (
                    f"intake_turn_{len(st.session_state.intake_history)}"
                )


                # =========================================
                # RADIO
                # =========================================

                if input_type == "radio":

                    options = step.get(
                        "options",
                        [],
                    )

                    answer = st.radio(
                        "Select one",
                        options=options,
                        index=None,
                        key=f"{intake_turn_key}_radio",
                    )


                # =========================================
                # MULTISELECT
                # =========================================

                elif input_type == "multiselect":

                    options = step.get(
                        "options",
                        [],
                    )

                    answer = st.multiselect(
                        "Select one or more",
                        options=options,
                        key=f"{intake_turn_key}_multiselect",
                    )


                # =========================================
                # DATE RANGE
                # =========================================

                elif input_type == "date_range":

                    date_min = step.get(
                        "date_min"
                    )

                    date_max = step.get(
                        "date_max"
                    )

                    min_value = (
                        date.fromisoformat(
                            date_min
                        )
                        if date_min
                        else None
                    )

                    max_value = (
                        date.fromisoformat(
                            date_max
                        )
                        if date_max
                        else None
                    )

                    if (
                        min_value
                        and max_value
                    ):

                        answer = st.date_input(
                            "Choose date range",
                            value=(
                                min_value,
                                max_value,
                            ),
                            min_value=min_value,
                            max_value=max_value,
                            key=f"{intake_turn_key}_date_range",
                        )

                    else:

                        answer = st.date_input(
                            "Choose date range",
                            key=f"{intake_turn_key}_date_range",
                        )


                # =========================================
                # TEXT
                # =========================================

                elif input_type == "text":

                    answer = st.text_input(
                        "Your answer",
                        key=f"{intake_turn_key}_text",
                    )


                # =========================================
                # ALWAYS-AVAILABLE FREE-TEXT CONTEXT
                # =========================================

                # Guided choices should help the stakeholder, never trap
                # them inside the options proposed by the AI. Every intake
                # turn therefore allows free-text context, including radio,
                # multiselect, date-range, and confirmation turns.
                if input_type != "text":
                    additional_context = st.text_area(
                        "Add context, change the scope, or type your own answer (optional)",
                        placeholder=(
                            "Add anything the choices do not capture — for example a different "
                            "time period, population, priority, exclusion, or refinement."
                        ),
                        key=f"{intake_turn_key}_additional_context",
                        height=90,
                    )

                # =========================================
                # CONTINUE
                # =========================================

                if st.button(
                    "Continue",
                    type="primary",
                    key=f"{intake_turn_key}_continue",
                ):

                    has_context = bool(
                        str(additional_context).strip()
                    )

                    valid_answer = True

                    if answer is None:
                        valid_answer = has_context

                    elif isinstance(
                        answer,
                        str,
                    ) and not answer.strip():
                        valid_answer = has_context

                    elif isinstance(
                        answer,
                        list,
                    ) and len(answer) == 0:
                        valid_answer = has_context

                    elif isinstance(
                        answer,
                        tuple,
                    ) and len(answer) == 0:
                        valid_answer = has_context


                    if not valid_answer:

                        st.warning(
                            "Choose an option or type your own answer before continuing."
                        )

                    else:

                        if isinstance(
                            answer,
                            tuple,
                        ):

                            stored_answer = [
                                item.isoformat()
                                if hasattr(
                                    item,
                                    "isoformat",
                                )
                                else str(item)
                                for item in answer
                            ]

                        elif isinstance(
                            answer,
                            list,
                        ):

                            stored_answer = [
                                item.isoformat()
                                if hasattr(
                                    item,
                                    "isoformat",
                                )
                                else str(item)
                                for item in answer
                            ]

                        elif hasattr(
                            answer,
                            "isoformat",
                        ):

                            stored_answer = (
                                answer.isoformat()
                            )

                        else:

                            stored_answer = answer

                        # Combine structured choice/date input with any
                        # stakeholder-written context into one authoritative
                        # answer for the next intake reasoning turn.
                        context_text = str(
                            additional_context
                        ).strip()

                        if context_text:
                            if stored_answer is None or stored_answer == "" or stored_answer == []:
                                stored_answer = context_text
                            elif isinstance(stored_answer, list):
                                selected_text = ", ".join(
                                    str(item) for item in stored_answer
                                )
                                stored_answer = (
                                    f"Selected: {selected_text}\n"
                                    f"Additional context: {context_text}"
                                )
                            else:
                                stored_answer = (
                                    f"Selected: {stored_answer}\n"
                                    f"Additional context: {context_text}"
                                )


                        st.session_state.intake_history.append(
                            {
                                "assistant_message": step[
                                    "assistant_message"
                                ],
                                "question": step[
                                    "question"
                                ],
                                "answer": stored_answer,
                                "input_type": input_type,
                                "option_relationship": option_relationship,
                                "intake_state": step.get(
                                    "intake_state",
                                    {},
                                ),
                            }
                        )

                        try:

                            with st.spinner(
                                "Updating the analytical request..."
                            ):

                                next_step = (
                                    get_next_intake_step()
                                )

                                st.session_state.current_intake_step = (
                                    next_step
                                )

                                if next_step[
                                    "ready_for_analysis"
                                ]:

                                    st.session_state.analysis_ready = True

                                    st.session_state.structured_question = (
                                        next_step[
                                            "structured_question"
                                        ]
                                    )

                            st.rerun()

                        except Exception as error:

                            st.error(
                                "The intake engine could not continue."
                            )

                            st.code(
                                str(error)
                            )