import pymupdf
import re
import statistics
import json

from pinecone import Pinecone

PINECONE_API_KEY = PINECONE_API_KEY

SECTION_NUMBER_PATTERN = re.compile(r"^(\d+)(?:\.(\d+))?\.?\s+\S")

def get_line_spans(pdf_path):
    """Extract each line of text with its dominant font size and bold flag."""
    doc = pymupdf.open(pdf_path)
    lines = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                dominant = max(spans, key=lambda s: len(s["text"]))
                size = round(dominant["size"], 1)
                is_bold = bool(dominant["flags"] & 2 ** 4)
                lines.append({"text": text, "size": size, "bold": is_bold})
    doc.close()
    return lines

def is_heading_line(line, body_size, max_words=8):
    """Shared predicate: short, visually distinct, and numbering-pattern-matched."""
    if len(line["text"].split()) > max_words:
        return False
    stands_out = line["size"] > body_size or line["bold"]
    return stands_out and bool(SECTION_NUMBER_PATTERN.match(line["text"]))


def detect_heading_candidates(lines, max_words=8):
    """Primary filter: short lines whose font size/weight stands out from body text."""
    body_size = statistics.mode([l["size"] for l in lines])
    return [l["text"] for l in lines if is_heading_line(l, body_size, max_words)]

def parse_section_number(text):
    m = SECTION_NUMBER_PATTERN.match(text)
    major = int(m.group(1))
    minor = int(m.group(2)) if m.group(2) else None
    return major, minor

def audit_sequence(headings):
    """Secondary sanity check: flag gaps/out-of-order numbers for manual review."""
    flagged = []
    prev_major, prev_minor = None, None
    for text in headings:
        major, minor = parse_section_number(text)
        if prev_major is not None:
            if minor is not None and prev_minor is not None and major == prev_major:
                if minor != prev_minor + 1:
                    flagged.append((text, f"expected {major}.{prev_minor + 1}"))
            elif minor is None and prev_minor is None:
                if major != prev_major + 1:
                    flagged.append((text, f"expected {prev_major + 1}"))
        prev_major, prev_minor = major, minor
    return flagged

def detect_section_headings(pdf_path):
    lines = get_line_spans(pdf_path)
    candidates = detect_heading_candidates(lines)
    flagged = audit_sequence(candidates)
    return candidates, flagged

def build_document_segments(pdf_path):
    """Single ordered pass: split lines into (number, title, body) segments.
    Reuses the same heading test as detect_heading_candidates so heading
    position and body text stay linked. Anything before the first real
    heading is kept under a 'Preamble' segment instead of being dropped.
    """
    lines = get_line_spans(pdf_path)
    body_size = statistics.mode([l["size"] for l in lines])
    segments = []
    current = {"number": None, "title": "Preamble", "body_lines": []}
    heading_texts_in_order = []
    for line in lines:
        if is_heading_line(line, body_size):
            segments.append(current)  # flush the segment we were building
            major, minor = parse_section_number(line["text"])
            number = f"{major}.{minor}" if minor is not None else str(major)
            current = {"number": number, "title": line["text"], "body_lines": []}
            heading_texts_in_order.append(line["text"])
        else:
            current["body_lines"].append(line["text"])
    segments.append(current)  # flush the final segment after the loop ends
    for seg in segments:
        seg["body"] = " ".join(seg.pop("body_lines"))
    flagged = audit_sequence(heading_texts_in_order)
    return segments, flagged

class HRDocumentCleaner:
    """Reused from the original cleaning pass; strip_headers_and_footers and
    append_subsection_headers_to_bullets are dropped since segmenting by
    font/number already handles what those two were compensating for."""
    def strip_headers_and_footers(self, text: str) -> str:
        text = re.sub(r"HR Policy\s*—\s*v\d+\.\d+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Northfield\s*&\s*Vance\s*Group", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bPage\s*\d+\b", "", text, flags=re.IGNORECASE)
        return text

    def standardize_bullet_points(self, text: str) -> str:
        text = re.sub(r"[●•·■♦]", " ", text)
        return text

    def normalize_whitespace_and_line_breaks(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n\s*\n+", "||PARAGRAPH||", text)
        text = re.sub(r"\n(?!-)", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = text.replace("||PARAGRAPH||", "\n\n")
        return text.strip()

def clean_segments(segments):
    """Run the existing cleaners on each segment's body independently."""
    cleaner = HRDocumentCleaner()
    for seg in segments:
        body = cleaner.strip_headers_and_footers(seg["body"])
        body = cleaner.standardize_bullet_points(body)
        body = cleaner.normalize_whitespace_and_line_breaks(body)
        seg["body"] = body
    return segments

def attach_metadata(segments, source, version, effective_date):
    """Attach document-level metadata plus each segment's own section
    identity, so every segment is self-describing once split off as a
    parent document."""
    for seg in segments:
        seg["metadata"] = {
            "source": source,
            "version": version,
            "effective_date": effective_date,
            "section_number": seg["number"] or "preamble",
            "section_title": seg["title"],
        }
    return segments

def split_into_sentences(text):
    """Simple sentence splitter: cut after ./!/? followed by whitespace."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]

def build_child_chunks(segments, target_words=40):
    """Group each parent segment's sentences into ~target_words-sized
    children. Short parents (already under target_words) become a single
    child unchanged. Each child inherits the parent's metadata plus a
    parent reference and its own chunk index."""
    child_chunks = []
    for seg in segments:
        sentences = split_into_sentences(seg["body"])
        current_sentences, current_word_count, chunk_index = [], 0, 0
        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            if current_sentences and current_word_count + sentence_word_count > target_words:
                child_chunks.append(_make_child(seg, current_sentences, chunk_index))
                chunk_index += 1
                current_sentences, current_word_count = [], 0
            current_sentences.append(sentence)
            current_word_count += sentence_word_count
        if current_sentences:
            child_chunks.append(_make_child(seg, current_sentences, chunk_index))
    return child_chunks

def _make_child(seg, sentences, chunk_index):
    metadata = dict(seg["metadata"])
    metadata["parent_section_number"] = seg["number"] or "preamble"
    metadata["chunk_index"] = chunk_index
    version = str(metadata.get("version") or "unversioned")
    section = str(seg["number"] or "preamble")
    vector_id = f"{version}_{section}_{chunk_index}".replace(" ", "_").replace(".", "-")
    return {"id": vector_id, "text": " ".join(sentences), "metadata": metadata}

def build_parent_store(segments):
    """Key each segment by (version, section_number) so repeated section
    numbers across versions (e.g. '5.1' in both v1 and v2) don't collide."""
    store = {}
    for seg in segments:
        key = f"{seg['metadata']['version']}|{seg['number']}"
        store[key] = seg
    return store

def save_parent_store(store, path):
    with open(path, "w") as f:
        json.dump(store, f, indent=2)

def load_parent_store(path):
    with open(path) as f:
        return json.load(f)


doc_meta = {
    "HR_Policy_v1.0_2023.pdf": {"source": "Northfield_Vance_HR_Handbook", "version": "1.0", "effective_date": "2023-01-15"},
    "HR_Policy_v2.0_2025.pdf": {"source": "Northfield_Vance_HR_Handbook", "version": "2.0", "effective_date": "2025-03-01"},
}
for path in ["HR_Policy_v1.0_2023.pdf", "HR_Policy_v2.0_2025.pdf"]:
    segments, flagged = build_document_segments(path)
    segments = clean_segments(segments)
    segments = attach_metadata(segments, **doc_meta[path])
    print(f"\n=== {path} ===")
    for seg in segments:
        print(seg["metadata"])

from pinecone import Pinecone, ServerlessSpec
pc=Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "hr-policy-handbook"
EMBEDDING_DIM = 64
if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(INDEX_NAME)
from nomic import embed
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
EMBEDDING_DIM = 64
BATCH_SIZE = 50
model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True,
    truncate_dim=EMBEDDING_DIM,
)
index = pc.Index("hr-policy-handbook")
def embed_and_upsert(child_chunks, batch_size=BATCH_SIZE):
    """child_chunks: list of dicts with 'id', 'text', 'metadata' (as produced
    by build_child_chunks). Embeds each chunk's text locally and upserts to
    Pinecone."""
    for i in range(0, len(child_chunks), batch_size):
        batch = child_chunks[i:i + batch_size]
        # "search_document: " prefix replaces task_type="search_document"
        prefixed_texts = [f"search_document: {c['text']}" for c in batch]
        vectors = model.encode(prefixed_texts, normalize_embeddings=True)
        to_upsert = [
            {"id": c["id"], "values": vec.tolist(), "metadata": c["metadata"]}
            for c, vec in zip(batch, vectors)
        ]
        index.upsert(vectors=to_upsert)
        print(f"Upserted {i + len(batch)}/{len(child_chunks)} chunks")
doc_meta = {
    "HR_Policy_v1.0_2023.pdf": {"source": "Northfield_Vance_HR_Handbook", "version": "1.0",
                                "effective_date": "2023-01-15"},
    "HR_Policy_v2.0_2025.pdf": {"source": "Northfield_Vance_HR_Handbook", "version": "2.0",
                                "effective_date": "2025-03-01"},
}
all_segments = []
all_children = []
for path, meta in doc_meta.items():
    segments, flagged = build_document_segments(path)
    segments = clean_segments(segments)
    segments = attach_metadata(segments, **meta)
    all_segments.extend(segments)
    all_children.extend(build_child_chunks(segments))
    if flagged:
        print(f"Flagged for manual review in {path}: {flagged}")
print(f"Total child chunks to embed: {len(all_children)}")
parent_store = build_parent_store(all_segments)
save_parent_store(parent_store, "parent_store.json")
print(f"Saved {len(parent_store)} parents to parent_store.json")
embed_and_upsert(all_children)
model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True,
    truncate_dim=EMBEDDING_DIM,
)
index = pc.Index("hr-policy-handbook")
# stats = index.describe_index_stats()
# print("Index stats:", stats)
# known_id = "2-0_5-1_0"  # v2, section 5.1, first chunk
# fetched = index.fetch(ids=[known_id])
# print("\nFetched vector:", fetched)
# query_text = "how many PTO days do employees get"
# query_vector = model.encode(
#     f"search_query: {query_text}"  # manual task prefix, required for local/direct use
# ).tolist()
# results = index.query(
#     vector=query_vector,
#     top_k=5,
#     include_metadata=True,
# )
# print("\nTop matches for query:", query_text)
# for match in results["matches"]:
#     print(f"  score={match['score']:.4f}  id={match['id']}  "
#           f"section={match['metadata'].get('section_title')}  "
#           f"version={match['metadata'].get('version')}")