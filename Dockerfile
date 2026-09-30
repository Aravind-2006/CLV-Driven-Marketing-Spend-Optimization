FROM python:3.13-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and trained model
COPY src/ ./src/
COPY model.json .

# Expose FastAPI default port
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
