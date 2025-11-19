FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the files (including model.bin)
COPY . .

EXPOSE 9696

# Start the service with gunicorn
CMD ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
