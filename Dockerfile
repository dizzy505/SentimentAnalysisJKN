#Gunakan image Python yang ringan
FROM python:3.11

#Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

#Install OS dependencies yang dibutuhin buat compile package Python
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    zlib1g-dev \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#Set working directory
WORKDIR /app

#Copy requirements dan install dependensi
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --default-timeout=100 --retries=10 --progress-bar off -r requirements.txt

#Copy ke container
COPY ./app ./app

#Pindah ke direktori app
WORKDIR /app/app

#Expose port Streamlit
EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.enableCORS=false"]