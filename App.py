import re
import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq
import json

EMBEDDING_DIM = 64
INDEX_NAME = "hr-policy-handbook"
PARENT_STORE_PATH = "parent_store.json"

def load_parent_store(path):
    with open(path) as f:
        return json.load(f)

HISTORICAL_KEYWORDS = [
    "changed", "change", "history", "historical", "before", "previously",
    "used to", "over the years", "past", "old policy", "prior version",
    "compare", "comparison", "difference between",
]
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        truncate_dim=EMBEDDING_DIM,
    )

@st.cache_resource
def load_pinecone_index():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    return pc.Index(INDEX_NAME)

@st.cache_data
def load_parents():
    return load_parent_store(PARENT_STORE_PATH)

def is_historical_query(query: str) -> bool:
    q = query.lower()
    return any(keyword in q for keyword in HISTORICAL_KEYWORDS)

def latest_version(parents: dict) -> str:
    dates_by_version = {}
    for seg in parents.values():
        v = seg["metadata"]["version"]
        d = seg["metadata"]["effective_date"]
        dates_by_version[v] = d
    return max(dates_by_version, key=lambda v: dates_by_version[v])

def embed_query(model, query: str):
    return model.encode(f"search_query: {query}").tolist()

def retrieve_chunks(index, query_vector, current_only: bool, current_version: str, top_k=5):
    query_filter = {"version": {"$eq": current_version}} if current_only else None
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=query_filter,
    )
    return results["matches"]

def resolve_to_parents(matches, parents: dict):
    seen = set()
    resolved = []
    for m in matches:
        key = f"{m['metadata']['version']}|{m['metadata']['parent_section_number']}"
        if key in seen or key not in parents:
            continue
        seen.add(key)
        resolved.append(parents[key])
    return resolved

def build_context(parent_segments):
    blocks = []
    for seg in parent_segments:
        meta = seg["metadata"]
        blocks.append(
            f"[{meta['section_title']} | version {meta['version']}, "
            f"effective {meta['effective_date']}]\n{seg['body']}"
        )
    return "\n\n".join(blocks)

def generate_answer(client, question: str, context: str, historical: bool):
    instruction = (
            "Answer using only the provided policy excerpts. Each excerpt is labeled "
            "with its version and effective date. "
            + (
                "The user is asking about how the policy changed over time — "
                "explicitly compare the versions and their dates in your answer."
                if historical
                else "Answer according to the most recent policy version only."
            )
    )
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",  # current Groq production model as of Sept 2026;
        # llama-3.3-70b-versatile is deprecated (retires Aug 16, 2026)
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"{instruction}\n\nPolicy excerpts:\n{context}\n\nQuestion: {question}",
        }],
    )
    return response.choices[0].message.content

def main():
    st.title("Test RAG pipeline")
    question = st.text_input("Ask a question about the HR policy:")
    if not question:
        return
    model = load_embedding_model()
    index = load_pinecone_index()
    parents = load_parents()
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    historical = is_historical_query(question)
    current_version = latest_version(parents)
    query_vector = embed_query(model, question)
    matches = retrieve_chunks(
        index, query_vector,
        current_only=not historical,
        current_version=current_version,
    )
    parent_segments = resolve_to_parents(matches, parents)
    if not parent_segments:
        st.warning("No relevant policy sections found.")
        return
    context = build_context(parent_segments)
    answer = generate_answer(groq_client, question, context, historical)
    st.markdown("### Answer")
    st.write(answer)
    with st.expander("Sources used"):
        for seg in parent_segments:
            meta = seg["metadata"]
            st.markdown(f"- **{meta['section_title']}** — v{meta['version']} ({meta['effective_date']})")

if __name__ == "__main__":
    main()