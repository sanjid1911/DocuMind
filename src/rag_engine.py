import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DB_PATH = "./db"

# Llama 3.1 8B Instruct
REPO_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def get_rag_chain():
    # 1. Load Vector Database
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 2. Convert retrieved Documents into plain text
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 3. LLM (MUST use conversational for Novita provider)
    llm = HuggingFaceEndpoint(
        repo_id=REPO_ID,
        task="conversational",   # ✅ THIS FIXES YOUR ERROR
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )

    # 4. Prompt (simple chat style, no Llama raw tokens)
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer the question strictly using the context below.
If the answer is not in the context, say "I don't have enough information".

Context:
{context}

Question:
{question}
""")

    # 5. Build RAG chain
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
