# utils/file_readers.py

import fitz  # PyMuPDF
from PyPDF2 import PdfReader
import pandas as pd
import extract_msg

def read_pdf(file):
    file.seek(0)
    pdf_reader = PdfReader(file)
    content = ""
    for page in pdf_reader.pages:
        content += page.extract_text() or ""
    return content

def read_csv(file):
    file.seek(0)
    return pd.read_csv(file)

def read_excel(file):
    file.seek(0)
    return pd.read_excel(file)

def read_msg(file):
    msg = extract_msg.Message(file)
    text = msg.body or ""
    text_filename = f"{msg.date}.txt"

    with open(text_filename, 'w', encoding='utf-8') as f:
        f.write(text)

    attachments = []
    for att in msg.attachments:
        name = att.longFilename or att.shortFilename or f"attachment_{msg.attachments.index(att)}"
        data = att.data
        mode = 'wb' if isinstance(data, bytes) else 'w'
        with open(name, mode, encoding=None if isinstance(data, bytes) else 'utf-8') as f:
            f.write(data if isinstance(data, bytes) else str(data))
        attachments.append(name)

    return text_filename, attachments
