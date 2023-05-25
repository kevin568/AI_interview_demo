import os
import re

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from pypdf import PdfReader
from langchain.docstore.document import Document
# os.environ["OPENAI_API_KEY"] = ""

# load PDF 
def load_pdf(pdf_filepath):
    with open(pdf_filepath, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)

        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

# use list to interpret data to fit chroma 
# 參考此https://www.shopjkl.com/pages/wintogether，中文一次query最多1000字

def split_paragraph(text, pdf_name, max_length=500):
    text = text.replace('\n', '').replace('\n\n', '')
    text = re.sub(r'\s+', ' ', text)
    #replace space into one single space

    delimiter = '(；|。|！|\!|\.|？|\?)'
    #pair two text together (chinese version)
    sentences = ["".join(x) for x in zip(*[iter(re.split(delimiter, text))]*2)]

    paragraphs = []
    current_paragraph = ""
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= max_length:
            current_paragraph += sentence
            current_length += sentence_length
        else:
            paragraphs.append(current_paragraph.strip())
            current_paragraph = sentence
            current_length = sentence_length

    if current_paragraph.strip():
        paragraphs.append(current_paragraph.strip())

    documents = [Document(page_content=paragraph, meta={'source': pdf_name}) for paragraph in paragraphs]
    
    return documents



def save_embeddings_to_storage(documents):
    # Define the directory to save the embeddings
    save_directory = 'db'
    # Initialize an instance of the OpenAIEmbeddings class
    embeddings_instance = OpenAIEmbeddings()
    # Create a Chroma instance from the documents using the specified embeddings
    vector_database = Chroma.from_documents(documents=documents, embedding=embeddings_instance, persist_directory=save_directory)
    # Persist the embeddings to the defined directory
    vector_database.persist()
    # Clear the vector_database variable
    vector_database = None

def main():
    pdf_name = "2023年重要產業趨勢與科技發展探討.pdf"
    content = load_pdf(pdf_name)
    documents = split_paragraph(content, pdf_name)
    save_embeddings_to_storage(documents)


if __name__ == "__main__":
    main()