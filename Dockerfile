FROM python:3.11-slim

WORKDIR /usr/src/app

# Install build dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements files first for caching
COPY requirements.txt .
COPY base-tooling-requirements.txt .

# Install tooling and dependencies without hashes
RUN pip install --no-cache-dir -r base-tooling-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .
COPY utils.py .

# Expose application port
EXPOSE 8080

# Use non-root user is recommended (optional)
# RUN useradd -m appuser
# USER appuser

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
