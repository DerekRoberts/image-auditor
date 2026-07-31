FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY image_cull.py .

ENTRYPOINT ["python", "image_cull.py"]
