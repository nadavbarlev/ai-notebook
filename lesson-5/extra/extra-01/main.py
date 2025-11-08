from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# load the PDF file
# returns a list of documents (pages)
loader = PyPDFLoader("./grant-letter.pdf")
docs = loader.load()

# In class, we split documents by sentences using NLTK (sent_tokenize) then tokenize into words.
# Here, we use RecursiveCharacterTextSplitter which splits by character count,
# creating fixed-size chunks (1000 chars) with overlap (200 chars) for context preservation.
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
document_chunks = splitter.split_documents(docs)

# In the class, we used SentenceTransformer to embed the sentences.
# Here, we use OllamaEmbeddings which uses the `nomic-embed-text` model.
# Note: The model must be downloaded manually using `ollama pull nomic-embed-text`
# before running this code, unlike NLTK, which automatically downloads resources when needed.
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# Create a FAISS index from the chunks and embeddings model
vector_store = FAISS.from_documents(document_chunks, embeddings_model)

query = "How much options got granted in the grant letter?"

# Create a retriever from the vector store
retriever = vector_store.as_retriever(k=3)

# In class, we manually embedded the query and used FAISS search to find similar chunks.
# Here, LangChain's retriever abstraction handles embedding and similarity search automatically.
relevant_document_chunks = retriever.invoke(query)

context = "\n\n".join(
    [document_chunk.page_content for document_chunk in relevant_document_chunks]
)

# Build the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a mad cynical fianance consultant"),
        ("user", "Here is the context: {context}"),
        ("user", "Here is the question: {query}"),
    ]
)

# Invoke the LLM
llm = ChatOllama(model="llama3.2:3b")
response = llm.invoke(prompt_template.format(context=context, query=query))

print(response.content)
