# 📄 RAG Chatbot with Streamlit, AWS Bedrock, and Neo4j

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Streamlit, Neo4j, and multi-LLM support (AWS Bedrock, OpenAI, Ollama, Google GenAI). Containerized via Docker Compose. Optionally extendable with AWS Lambda and SageMaker for enterprise-grade scalability.

---

## 📦 Features

- 🔍 File-supported QA (PDF, CSV, Excel, MSG)
- 🧠 Multi-embedding support: AWS, OpenAI, Ollama, HuggingFace, Google
- 🧱 Vector search via Neo4j graph DB
- 🧠 LLMs via Bedrock (Claude, Titan), OpenAI, Ollama
- 🐳 Dockerized architecture
- ☁️ Optional serverless backend via Lambda
- 🔁 CI/CD via Jenkins
- 📊 SageMaker-compatible inference backend

---

## 🧪 Usage (Local)

```bash
docker-compose up --build
