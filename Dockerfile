FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY env.py .
COPY inference.py .
COPY app.py .

# Expose port 7860 (HuggingFace default)
EXPOSE 7860

# Environment variables (defaults only for API_BASE_URL and MODEL_NAME)
ENV API_BASE_URL=https://api.openai.com/v1
ENV MODEL_NAME=gpt-4o-mini

# Start FastAPI server
CMD ["python", "app.py"]
