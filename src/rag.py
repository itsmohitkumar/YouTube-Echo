import logging
import uuid
from typing import List, Literal
from chromadb import Collection
from langchain.chat_models.base import BaseChatModel
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.helpers import num_tokens_from_string

CHUNK_SIZE_FOR_UNPROCESSED_TRANSCRIPT = 512

# Mapping from chunk size to number of chunks to provide equal input length
CHUNK_SIZE_TO_K_MAPPING = {1024: 3, 512: 5, 256: 10, 128: 20}

RAG_SYSTEM_PROMPT = """You are an expert in answering questions and providing information about a topic.

You are going to receive excerpts from a video transcript as context. Furthermore, a user will provide a question or a topic.
If you receive a question, give a detailed answer. If you receive a topic, tell the user what is said about the topic.
In either case, keep your answer grounded solely in the facts of the context.
If the context does not contain the facts to answer the question, apologize and say that you don't know the answer.
"""

rag_user_prompt = PromptTemplate.from_template(
    """Context: {context}
---
Here is the user's question/topic: {question}
"""
)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        ("user", "{input}"),
    ]
)


def split_text_recursively(
    transcript_text: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 0,
    len_func: Literal["characters", "tokens"] = "characters",
) -> List[Document]:
    """Splits a string recursively by characters or tokens."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len if len_func == "characters" else num_tokens_from_string,
    )
    splits = text_splitter.create_documents([transcript_text])
    logging.info(
        "Split transcript into %d chunks with a provided chunk size of %d tokens.",
        len(splits),
        chunk_size,
    )
    return splits


def format_docs_for_context(docs: List[Document]) -> str:
    """Formats documents for context by joining their contents with separators."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def embed_excerpts(
    collection: Collection, excerpts: List[Document], embeddings: Embeddings
):
    """Embeds excerpts and adds them to the collection if no embeddings exist."""
    if collection.count() <= 0:
        for excerpt in excerpts:
            try:
                response = embeddings.embed_query(excerpt.page_content)
                collection.add(
                    ids=[str(uuid.uuid1())],
                    embeddings=[response],
                    documents=[excerpt.page_content],
                )
                logging.info("Added document to collection with ID: %s", str(uuid.uuid1()))
            except Exception as e:
                logging.error("Failed to embed and add document: %s", str(e))


def find_relevant_documents(query: str, db: Chroma, k: int = 3) -> List[Document]:
    """
    Retrieve relevant documents by performing a similarity search.

    Args:
        query (str): The search query.
        db (Chroma): The database to search in.
        k (int): The number of top relevant documents to retrieve. Default is 3.

    Returns:
        List[Document]: A list of the top k relevant documents.
    """
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(input=query)


def generate_response(question: str, llm: BaseChatModel, relevant_docs: List[Document]) -> str:
    """Generates a response from the LLM based on the question and relevant documents."""
    formatted_input = rag_user_prompt.format(
        question=question, context=format_docs_for_context(relevant_docs)
    )
    rag_chain = rag_prompt | llm | StrOutputParser()
    return rag_chain.invoke({"input": formatted_input})
