FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch==2.4.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py crossencoder.py meta.py download.py LICENSE ./

ARG MODEL_NAME
ENV MODEL_NAME=${MODEL_NAME} ENABLE_CUDA=0
RUN python download.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
