import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DB_PATH = "./db"

# Llama 3.1 8B Model
REPO_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def get_rag_chain():
    # Load Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Define LLM
    llm = HuggingFaceEndpoint(
        repo_id=REPO_ID,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )

    # Llama 3 specific prompt format
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful assistant. Answer the question based strictly on the context below.
    If the answer is not in the context, say "I don't have enough information".
    Context: {context}
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {question}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain