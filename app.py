import re
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

# ------------------Step 1: Load File-----------------------
loader = Docx2txtLoader("demo.docx")
documents = loader.load()

texts = [doc.page_content for doc in documents]

# -------------------Step 2: Split Text---------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=40,
    length_function=len,
    is_separator_regex=False,
)
doc_chunks = text_splitter.create_documents(texts)

# -------------------Step 3: Embeddings---------------------
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=doc_chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)
vectorstore.persist()
print("✅ Embeddings stored in ChromaDB")

# ------------------Step 4: Extract Key Insights-------------------------
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
)

client = Groq(api_key="")
key_insights = []

#------------------Step 5: Process Each chunk and get the insight and store in a txt file.
# Process each chunk to extract key insights using the LLM
for chunk in doc_chunks:
    completion = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {
                "role": "user",
                "content": f"""Extract the key insights from the following text:\n\n"{chunk.page_content}"\n\nReturn only the key insights."""
            }
        ],
        temperature=0.6,
        max_completion_tokens=256,
        top_p=0.95,
        stream=False, 
    )
    
    response_text = completion.choices[0].message.content
    key_insights.append(response_text)


with open('key_insights.txt', 'w') as txt_file:
    for insight in key_insights:
        txt_file.write(f"{insight}\n\n") 

print("✅ Key insights saved to 'key_insights.txt'")
