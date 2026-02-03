FROM python:3.10-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy requirements first for better caching
COPY backend/requirements.txt .
COPY backend/uv.lock ./uv.lock

# Install dependencies using uv with --system flag
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the backend code
COPY backend/app ./app
COPY backend/models ./models
COPY backend/templates ./templates
COPY backend/config.yaml .

# Create directories for uploads
RUN mkdir -p /app/data/uploads

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "app.main"]