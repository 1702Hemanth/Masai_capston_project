from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents():
    documents = []
    ids = []

    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8",errors="replace").strip()

        if not text:
            continue

        ids.append(path.stem)
        documents.append(text)

    return ids, documents


def main():
    print("Loading policy documents...")

    ids, documents = load_documents()

    if len(documents) != 8:
        raise ValueError(
            f"Expected 8 documents, but found {len(documents)}."
        )

    print(f"Documents loaded: {len(documents)}")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    embeddings = model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # Recreate the collection so repeated runs stay deterministic.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Zepto support policy corpus"}
    )

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=[
            {
                "source": doc_id,
                "chunk_id": doc_id
            }
            for doc_id in ids
        ],
    )

    print(f"ChromaDB collection: {COLLECTION_NAME}")
    print(f"Documents indexed: {collection.count()}")

    # Verification query.
    results = collection.query(
        query_embeddings=[embeddings[0]],
        n_results=3,
    )

    print("\nVerification query:")
    for result_id, distance in zip(
        results["ids"][0],
        results["distances"][0],
    ):
        print(f"  {result_id}: distance={distance:.4f}")

    print("\nIngestion completed successfully.")


if __name__ == "__main__":
    main()