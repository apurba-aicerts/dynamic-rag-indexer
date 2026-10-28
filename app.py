from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from indexer import (
    SUPPORTED_SUFFIXES,
    create_collection,
    save_uploaded_files,
    sync_documents,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = APP_DIR / "documents"
DEFAULT_DATA_DIR = APP_DIR / ".rag_index"

st.set_page_config(page_title="Dynamic RAG Indexer", page_icon="D", layout="centered")
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f7f8f3 0%, #eaf2ef 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 900px; padding-top: 2.5rem; }
    h1 { color: #183d37; font-family: Georgia, serif; font-weight: 600; }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid #d8e3dd;
        border-left: 3px solid #d56b50;
        padding: 0.8rem 1rem;
    }
    div.stButton > button[kind="primary"] {
        background: #176957;
        border-color: #176957;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Dynamic RAG Indexer")
st.caption("Local corpus · incremental sync")

source_col, data_col = st.columns(2)
with source_col:
    source_value = st.text_input("Documents folder", value=str(DEFAULT_SOURCE))
with data_col:
    data_value = st.text_input("Index folder", value=str(DEFAULT_DATA_DIR))

uploaded_files = st.file_uploader(
    "Add documents",
    type=[suffix.removeprefix(".") for suffix in sorted(SUPPORTED_SUFFIXES)],
    accept_multiple_files=True,
)


def run_sync(source_dir: Path, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with st.spinner("Scanning documents and updating vectors..."):
        collection = create_collection(data_dir)
        st.session_state["last_sync"] = sync_documents(
            source_dir, data_dir / "manifest.json", collection
        )
    st.session_state["data_dir"] = str(data_dir)


source_dir = Path(source_value).expanduser().resolve()
data_dir = Path(data_value).expanduser().resolve()
upload_col, sync_col = st.columns(2)
if upload_col.button("Upload & sync", type="primary", use_container_width=True):
    if not uploaded_files:
        st.warning("Choose at least one document to upload.")
    else:
        try:
            saved_files = save_uploaded_files(source_dir, uploaded_files)
            run_sync(source_dir, data_dir)
            st.success(f"Uploaded {len(saved_files)} file(s) and synced the index.")
        except Exception as error:
            st.error(f"Upload or sync failed: {error}")

if sync_col.button("Sync folder", use_container_width=True):
    if not source_dir.is_dir():
        st.error(f"Documents folder does not exist: {source_dir}")
    else:
        try:
            run_sync(source_dir, data_dir)
            st.success("Sync complete")
        except Exception as error:
            st.error(f"Sync failed: {error}")

active_data_dir = Path(st.session_state.get("data_dir", str(DEFAULT_DATA_DIR)))
manifest_path = active_data_dir / "manifest.json"
manifest = {"documents": {}}
if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        st.warning("The index manifest could not be read.")

documents = manifest.get("documents", {})
chunk_count = sum(len(document.get("chunks", [])) for document in documents.values())
last_sync = st.session_state.get("last_sync", {})

st.divider()
metric_columns = st.columns(4)
metric_columns[0].metric("Documents", len(documents))
metric_columns[1].metric("Tracked chunks", chunk_count)
metric_columns[2].metric("Added", last_sync.get("added_chunks", 0))
metric_columns[3].metric("Reused", last_sync.get("reused_chunks", 0))

if last_sync:
    st.caption(
        f"Last run: {last_sync.get('indexed', 0)} updated, "
        f"{last_sync.get('skipped', 0)} unchanged, "
        f"{last_sync.get('deleted_chunks', 0)} chunks removed, "
        f"{last_sync.get('deleted_documents', 0)} documents removed."
    )