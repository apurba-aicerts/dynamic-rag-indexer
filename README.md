# Dynamic RAG Indexer POC

A small local proof of concept for incrementally indexing Markdown and text files.
It uses LangChain's `RecursiveCharacterTextSplitter`, SHA-256 chunk fingerprints,
a JSON manifest, and a persistent Chroma collection with local sentence-transformer
embeddings.

## Run

1. Use Python 3.10 or newer and install dependencies:

	```powershell
	python -m pip install -r requirements.txt
	```

2. Upload `.txt`, `.md`, `.pdf`, or `.docx` files in the GUI, or put them under `documents/`.
3. Run the indexer:

	```powershell
	python indexer.py
	```

Or launch the GUI:

```powershell
python -m streamlit run app.py
```

The first run downloads the local embedding model. Chroma data and the manifest
are stored under `.rag_index/`. Run the command again after adding, changing, or
deleting files. Unchanged documents are skipped, unchanged chunks in changed
documents retain their vectors, and chunks from deleted files are removed.

PDF indexing extracts embedded text; scanned/image-only PDFs require OCR and are
not processed by this POC.

Options:

```powershell
python indexer.py --source path\to\documents --data-dir path\to\index
```

Run the unit tests with `python -m unittest discover -s tests`.