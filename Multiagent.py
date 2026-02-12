import streamlit as st
from typing import TypedDict, List
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
import os
import json
import re

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ---------------------
# Define state
# ---------------------
class ProjectState(TypedDict):
    project: str
    summary: str
    tech_stack: str
    team: List[str]

llm = ChatGroq(model="openai/gpt-oss-120b")

students = [
    {"name": "Ananya", "visa": "OPT", "skills": ["React", "Node.js", "MongoDB"]},
    {"name": "Rahul", "visa": "STEM OPT", "skills": ["Python", "AWS", "Redshift"]},
    {"name": "Priya", "visa": "OPT", "skills": ["Flutter", "Firebase"]},
    {"name": "Vikram", "visa": "STEM OPT", "skills": ["React", "AWS"]},
]

# ---------------------
# Utility to safely extract JSON
# ---------------------
def safe_json_extract(text):
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                return {}
    return {}

# ---------------------
# Agent functions
# ---------------------
def project_analyzer(state: ProjectState):
    project = state["project"]
    prompt = f"Summarize this project and identify if it’s web, data, or mobile: {project}"
    summary = llm.invoke(prompt).content
    return {"summary": summary}

def tech_recommender(state: ProjectState):
    summary = state["summary"]
    prompt = f"Suggest a suitable tech stack for this project: {summary}. Keep it short."
    tech_stack = llm.invoke(prompt).content
    return {"tech_stack": tech_stack}

def student_allocator_llm(state: ProjectState):
    tech_stack = state["tech_stack"]
    students_json = json.dumps(students, indent=2)

    prompt = f"""
You are a staffing assistant.

Pick the BEST 2 students based on skill match with the tech stack.

Return STRICT JSON ONLY:

{{
 "selected": [
   {{"name": "Student Name", "visa": "VisaType"}}
 ]
}}

Tech stack:
{tech_stack}

Students:
{students_json}
"""

    resp = llm.invoke(prompt).content
    data = safe_json_extract(resp)

    chosen = []
    roster = {s["name"]: s for s in students}

    for item in data.get("selected", []):
        name = item.get("name")
        if name in roster:
            chosen.append(f"{name} ({roster[name]['visa']})")
        if len(chosen) == 2:
            break

    # fallback if LLM output invalid
    if len(chosen) < 2:
        chosen = ["Generic OPT student", "Generic STEM OPT student"]

    return {"team": chosen}

# ---------------------
# Build LangGraph
# ---------------------
graph = StateGraph(ProjectState)
graph.add_node("analyzer", project_analyzer)
graph.add_node("recommender", tech_recommender)
graph.add_node("allocator", student_allocator_llm)

graph.set_entry_point("analyzer")
graph.add_edge("analyzer", "recommender")
graph.add_edge("recommender", "allocator")

workflow = graph.compile()

# ---------------------
# Streamlit UI
# ---------------------
st.title("Project Analyzer & Student Allocator 🚀")

project_desc = st.text_area("Enter your project description:")

if st.button("Run Analysis"):
    if not project_desc.strip():
        st.error("Please enter a project description.")
    else:
        with st.spinner("Analyzing project..."):
            result = workflow.invoke({"project": project_desc})

        st.success("Analysis Complete!")

        st.subheader("📌 Summary")
        st.write(result["summary"])

        st.subheader("🛠 Recommended Tech Stack")
        st.write(result["tech_stack"])

        st.subheader("👥 Suggested Students")
        st.write(result["team"])
