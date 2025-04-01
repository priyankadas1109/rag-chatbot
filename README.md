# 🧠 RAG ChatBot for Structured Taxonomy Extraction

A powerful Python-based Retrieval-Augmented Generation (RAG) chatbot designed to generate structured taxonomies from a wide variety of documents, leveraging Amazon Bedrock's Anthropic Claude 3 model and deployed seamlessly with Docker and Jenkins.

---

## 🚀 Features

- **RAG with Amazon Bedrock Claude 3**: Utilizes `anthropic.claude-3-haiku-20240307-v1:0` LLM via Bedrock for intelligent taxonomy generation.
- **Vector Search with Chroma**: Embeds document chunks and performs semantic similarity search using the Chroma vector database.
- **Multimodal File Support**:
  - PDF
  - CSV
  - Excel
  - Word
  - Text
  - PPT
  - Outlook MSG (email body + attachments)
- **Auto-Zip Creation**: Automatically zips and makes extracted attachments downloadable.
- **Chat Interface**: Streamlit-based UI with persistent chat history.
- **Export Capability**: Outputs structured responses to CSV or Excel format for business use.
- **Dockerized**: Fully containerized using Docker and orchestrated via Docker Compose.
- **CI/CD Ready**: Jenkins pipeline included for auto-building and pushing to AWS ECR.

---

## 📂 Folder Structure

rag-chatbot/ │ ├── app/ │ ├── pdf_bot.py # Streamlit app logic │ ├── chains.py # LLM & embedding chain logic │ └── utils/ │ ├── init.py │ └── file_readers.py # Modular file readers │ ├── lambda/ │ ├── handler.py # AWS Lambda handler │ └── requirements.txt │ ├── jenkins/ │ └── Jenkinsfile # Docker build & deploy pipeline │ ├── requirements.txt # Python dependencies ├── docker-compose.yml ├── pdf_bot.Dockerfile ├── .dockerignore ├── .gitignore └── example.env # Template for environment variables


---

## 🔧 Setup

1. **Install Dependencies**
   - Docker Desktop
   - Python 3.10+
   - AWS CLI (configured)

2. **Clone the Repo**
   ```bash
   git clone https://github.com/priyankadas1109/rag-chatbot.git
   cd rag-chatbot

3. **Create .env from Template**
    cp example.env .env
    # Add your AWS/OpenAI credentials etc.

4. **Run with Docker**
    docker-compose up --build

5. **Access Chat UI**
    Visit: http://localhost:8503


## 🧠 Technologies Used

    Amazon Bedrock (Claude 3 via Anthropic)

    Chroma for vector similarity search

    Streamlit for front-end

    LangChain for prompt orchestration

    Docker & Jenkins for CI/CD automation

    pandas, PyPDF2, fitz, extract-msg, pptx etc. for file parsing


## 🛡️ Environment Security

    .env is excluded via .gitignore

    A sample example.env is provided

    Do not push credentials or access keys to GitHub!


## 📈 Impact

    Enhanced taxonomy accuracy and traceability with LLM-enhanced RAG pipeline.

    Efficient extraction from diverse enterprise formats including PDFs, spreadsheets, and emails.

    Reusable export formats (Excel/CSV) boost downstream business usability.



## ✨ Author

    Priyanka Das
    🎓 MS Business Analytics & AI
    🏆 1st Prize – HPE Generative AI Hackathon
    🔗 LinkedIn: https://www.linkedin.com/in/priyanka-das-04677b22/   
