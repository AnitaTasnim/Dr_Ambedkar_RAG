import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

CORPUS_DIR = "corpus"  # folder with speech1.txt ... speech6.txt

def build_rag():
    # 1. Load all documents from corpus
    all_docs = []
    for filename in sorted(os.listdir(CORPUS_DIR)):
        if filename.endswith(".txt"):
            path = os.path.join(CORPUS_DIR, filename)
            loader = TextLoader(path)
            docs = loader.load()
            # Attach metadata for source
            for doc in docs:
                doc.metadata["source"] = filename
            all_docs.extend(docs)

    # 2. Chunking
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_docs)

    # 3. Embeddings
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Vector DB
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        persist_directory="db"
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # 5. LLM via Ollama
    llm = Ollama(model="llama3.2")

    # 6. RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa

# --------------------------
# Chat loop
# --------------------------
if __name__ == "__main__":
    qa = build_rag()
    print("\nAmbedkarGPT Ready!\n")

    while True:
        query = input("Ask a question (or type exit): ")

        # Exit condition
        if query.lower().strip() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Empty input guard
        if not query.strip():
            print("Please enter a valid question.\n")
            continue

        # Get answer from RAG pipeline
        answer = qa.invoke(query)

        print("\nAnswer:")
        print(f" {answer['result']}\n")

 
        print("Sources:")
        unique_sources = set()

        for src in answer["source_documents"]:
            unique_sources.add(src.metadata["source"])

        for src in unique_sources:
            print(f" - {src}")

        print("\n--------------------------------------------------\n")
   

 