"""
Step 5: Prompt Engineering — Same Question, Three Experts
============================================================
See how the SAME model gives completely different answers
depending on the system prompt. Run with:

    streamlit run steps/step5_personas.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import streamlit as st
import llm_client

st.set_page_config(page_title="Prompt Engineering", page_icon="🧠", layout="wide")
st.title("🧠 Prompt Engineering: Three Experts")
st.markdown("Send the **same question** to three AI personas and compare.")

PERSONAS = {
    "🏗️ Civil Engineer": (
        "You are an expert civil engineer. Focus on structural analysis, "
        "materials science, and building codes. Be specific and technical."
    ),
    "🤖 Robotics Engineer": (
        "You are an expert robotics engineer. Focus on sensors, actuators, "
        "control systems, and autonomous operation. Think in terms of systems."
    ),
    "🎨 Creative Writer": (
        "You are a creative writing coach. Focus on storytelling, metaphor, "
        "and narrative structure. Be inspiring and imaginative."
    ),
}

question = st.text_input("Your question:", value="How can I build a bridge?")

if st.button("Compare Responses", type="primary"):
    cols = st.columns(3)
    for col, (name, prompt) in zip(cols, PERSONAS.items()):
        with col:
            st.subheader(name)
            with st.spinner("Thinking..."):
                reply = llm_client.chat([
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question},
                ], max_tokens=400)
                st.markdown(reply)

st.info(
    "**Key insight:** The model has NOT changed — only your instructions have. "
    "This is how you build agents with different roles."
)
