import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DB_PATH = "./db"
REPO_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def get_rag_chain():

    # 1. Embeddings + Vector DB
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 2. Format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 3. Correct LLM setup (FIXED)
    base_llm = HuggingFaceEndpoint(
        repo_id=REPO_ID,
        task="conversational",
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
    )

    llm = ChatHuggingFace(llm=base_llm)

    # 4. Prompt
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
""")

    # 5. Chain
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
