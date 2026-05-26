from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import faiss
import numpy as np
import gradio as gr

# Load embedding model
model = SentenceTransformer('BAAI/bge-base-en')

# Global variables
chunks = []
index = None


# Function to process PDF
def process_pdf(pdf_file):

    global chunks
    global index

    # Read PDF
    reader = PdfReader(pdf_file.name)

    text = ""

    for page in reader.pages:
        extracted_text = page.extract_text()

        if extracted_text:
            text += extracted_text

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(text)

    # Generate embeddings
    embeddings = model.encode(chunks)

    embeddings = np.array(embeddings).astype('float32')

    # Create FAISS vector database
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    # Store embeddings
    index.add(embeddings)

    return f"PDF processed successfully!\nTotal chunks created: {len(chunks)}"


# Semantic search function
def semantic_search(query):

    global chunks
    global index

    # Check if PDF uploaded
    if index is None:
        return "Please upload and process a PDF first."

    # Generate query embedding
    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype('float32')

    # Retrieve top 5 similar chunks
    distances, indices = index.search(query_embedding, 5)

    results = ""

    for i, idx in enumerate(indices[0]):

        results += f"Result {i+1}:\n"
        results += f"{chunks[idx]}\n\n"
        results += f"Distance Score: {distances[0][i]}\n"
        results += "-" * 50 + "\n\n"

    return results


# Custom CSS
custom_css = """
body {
    background: linear-gradient(to right, #8e2de2, #4a00e0);
}

.gradio-container {
    background: transparent !important;
    font-family: Arial, sans-serif;
}

h1 {
    text-align: center;
    color: black;
    font-size: 42px !important;
    font-weight: bold;
}

h2 {
    color: blue;
}

textarea {
    border-radius: 15px !important;
    border: 2px solid #667eea !important;
    background-color: #f8f9ff !important;
}

button {
    background: linear-gradient(to right, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    padding: 12px !important;
}

button:hover {
    transform: scale(1.05);
    transition: 0.3s;
}

footer {
    visibility: hidden;
}
"""

# Gradio Blocks UI
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as app:

    gr.Markdown(
        """
        # AI Powered Semantic Search Engine
        
        Upload PDFs and search documents using NLP, Embeddings and Vector Databases.
        """
    )

    with gr.Tab("Upload PDF"):

        gr.Markdown("## Upload Your PDF")

        pdf_input = gr.File(label="Choose PDF File")

        upload_output = gr.Textbox(
            label="Processing Status",
            lines=3
        )

        upload_button = gr.Button("Process PDF")

        upload_button.click(
            fn=process_pdf,
            inputs=pdf_input,
            outputs=upload_output
        )

    with gr.Tab("Search Documents"):

        gr.Markdown("## Semantic Similarity Search")

        query_input = gr.Textbox(
            label="Enter Your Query",
            placeholder="Example: What is NLP?",
            lines=2
        )

        search_output = gr.Textbox(
            label="Top Matching Results",
            lines=20
        )

        search_button = gr.Button("Search")

        search_button.click(
            fn=semantic_search,
            inputs=query_input,
            outputs=search_output
        )

# Launch App
app.launch()