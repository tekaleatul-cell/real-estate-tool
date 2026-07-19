from uuid import uuid4
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None


def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR),
        )


def process_urls(urls):
    """
    Scrapes data from the given urls and stores it in the vector db.
    :param urls: input urls
    :return: yields status strings as it progresses
    """
    yield "Initializing Components...✅"
    initialize_components()

    yield "Resetting vector store...✅"
    vector_store.reset_collection()

    yield "Loading data...✅"
    loader = WebBaseLoader(urls)
    data = loader.load()

    yield "Splitting text into chunks...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
    )
    docs = text_splitter.split_documents(data)

    yield "Adding chunks to vector database...✅"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done adding docs to vector database...✅"


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def generate_answer(query):
    if vector_store is None:
        raise RuntimeError("Vector database is not initialized")

    retriever = vector_store.as_retriever()
    retrieved_docs = retriever.invoke(query)

    context = _format_docs(retrieved_docs)
    sources = sorted({doc.metadata.get("source") for doc in retrieved_docs if doc.metadata.get("source")})

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})

    return answer, ", ".join(sources)


if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html",
    ]

    # This loop is what was missing before — process_urls() is a generator,
    # so it must be iterated to actually run.
    for status in process_urls(urls):
        print(status)

    answer, sources = generate_answer(
        "Tell me what was the 30 year fixed mortgage rate along with the date?"
    )
    print(f"\nAnswer: {answer}")
    print(f"Sources: {sources}")