import os
from langchain_chroma import Chroma
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DB_PATH = "./db"
REPO_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def get_rag_chain():
    # 1. Vector DB
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 2. Format retrieved docs
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 3. Base endpoint MUST be conversational
    base_llm = HuggingFaceEndpoint(
        repo_id=REPO_ID,
        task="conversational",
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )

    # 4. Wrap it in ChatHuggingFace (THIS is the real fix)
    llm = ChatHuggingFace(llm=base_llm)

    # 5. Prompt
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer strictly using the context below.
If the answer is not in the context, say "I don't have enough information".

Context:
{context}

Question:
{question}
""")

    # 6. Chain
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
