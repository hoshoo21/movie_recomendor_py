FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .

# RUN pip install --no-cache-dir --default-timeout=1000 --retries 50 --no-cache-dir -r requirements.txt
# RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --default-timeout=1000 --retries 50 --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir fastapi uvicorn sentence-transformers && \
    rm -rf /root/.cache/pip

ARG HF_TOKEN 

    # 2. Turn that ARG into an Environment Variable so Python can see it
ENV HF_TOKEN=${HF_TOKEN}


EXPOSE 8000
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]