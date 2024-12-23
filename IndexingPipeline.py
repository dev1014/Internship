import warnings
import os
from glob import glob
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Pipeline
from haystack.components.converters.txt import TextFileToDocument
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

warnings.filterwarnings('ignore')

# Ensure the API key is set in the environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

# Embedding documents
embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small")
documents = [Document(content="Haystack is an open source AI framework to build full AI applications in Python"),
             Document(content="You can build AI Pipelines by combining Components"),]
embedder.run(documents=documents)

# Initialize a Document Store
document_store = InMemoryDocumentStore()

# Setting up the indexing pipeline
converter = TextFileToDocument()
splitter = DocumentSplitter()
embedder = OpenAIDocumentEmbedder()
writer = DocumentWriter(document_store=document_store)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", converter)
indexing_pipeline.add_component("splitter", splitter)
indexing_pipeline.add_component("embedder", embedder)
indexing_pipeline.add_component("writer", writer)
indexing_pipeline.connect("converter", "splitter")
indexing_pipeline.connect("splitter", "embedder")
indexing_pipeline.connect("embedder", "writer")

# Process all text files in the specified directory
text_files = glob('C:\\Users\\devna\\OneDrive\\Desktop\\New folder\\*.txt')  # Change the path to your directory of the folder with your text files
indexing_pipeline.run({"converter": {"sources": text_files}})

# Retrieve and display documents
filtered_documents = document_store.filter_documents()
for i, doc in enumerate(filtered_documents):
    print("\n--------------\n")
    print(f"DOCUMENT {i}")
    print(doc.content)

# Creating a document search pipeline
query_embedder = OpenAITextEmbedder()
retriever = InMemoryEmbeddingRetriever(document_store=document_store)

document_search = Pipeline()
document_search.add_component("query_embedder", query_embedder)
document_search.add_component("retriever", retriever)
document_search.connect("query_embedder.embedding", "retriever.query_embedding")

# Running a search query
question = "How old was Davinci when he died?"
results = document_search.run({"query_embedder": {"text": question}})
for i, document in enumerate(results["retriever"]["documents"]):
    print("\n--------------\n")
    print(f"DOCUMENT {i}")
    print(document.content)

# Search with top_k parameter
question = "Where was Davinci born?"
results = document_search.run({"query_embedder": {"text": question}, "retriever": {"top_k": 3}})
for i, document in enumerate(results["retriever"]["documents"]):
    print("\n--------------\n")
    print(f"DOCUMENT {i}")
    print(document.content)
