FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies first (layer cache)
COPY pyproject.toml .
RUN uv pip install --system --no-cache .

# Copy application source
COPY src/ src/

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

CMD ["uvicorn", "src.api.endpoints:app", "--host", "0.0.0.0", "--port", "8080"]
