import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "document/FWCM2023.pdf"
DB_CHROMA_PATH = "vectorstore/db_chroma"


def start():
    pdf = PyPDF2.PdfReader(DATA_PATH)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
    texts = text_splitter.split_text(pdf_text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    Chroma.from_texts(
        texts, embeddings, metadatas=metadatas, persist_directory=DB_CHROMA_PATH
    )


if __name__ == "__main__":
    start()
