FROM python:3.13-slim

WORKDIR /app

COPY requirement.txt .
# Install dependencies (ignoring cache to keep image small)
RUN pip install --no-cache-dir -r requirement.txt

# Copy source code and env file
COPY src/ src/
COPY .env .env

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]