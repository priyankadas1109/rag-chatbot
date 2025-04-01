FROM langchain/langchain

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade -r requirements.txt

# Copy app code
COPY app/pdf_bot.py ./pdf_bot.py
COPY app/chains.py ./chains.py
COPY app/utils ./utils

EXPOSE 8503

HEALTHCHECK CMD curl --fail http://localhost:8503/_stcore/health

ENTRYPOINT ["streamlit", "run", "pdf_bot.py", "--server.port=8503", "--server.address=0.0.0.0"]
