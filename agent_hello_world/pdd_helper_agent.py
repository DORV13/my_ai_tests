import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader

load_dotenv()

class PdfLoader:
    """
    A class for loading and processing PDF files using PyPDFLoader.
    """

    def __init__(self):
        self.file_path = os.getenv("PDF_PATH")

    def load_and_process_pdf(self):
        """
        Load and process the PDF file.
        Uses PyPDFLoader to extract documents from the file.

        :return: A list of Documents.
        """
        try:
          loader = PyPDFLoader(self.file_path)
          documents = loader.load()
          return documents
        except Exception as e:
            raise RuntimeError(f"Failed to load and process PDF: {e}")
        

def main():
    pdf_loader = PdfLoader()
    documents = pdf_loader.load_and_process_pdf()
    print("done")

if __name__ == "__main__":
    main()