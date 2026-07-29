FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY audit_realism.py .

ENTRYPOINT ["python", "audit_realism.py"]
