FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.4.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH=/app/src:/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && pip install --no-cache-dir "poetry==${POETRY_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./
RUN poetry install --only api --no-root --no-interaction --no-ansi

COPY master_config.py ./
COPY interface/__init__.py ./interface/__init__.py
COPY interface/api/ ./interface/api/
COPY src/__init__.py ./src/__init__.py
COPY src/models/__init__.py ./src/models/__init__.py
COPY src/models/inference.py ./src/models/inference.py
COPY src/models/common/ ./src/models/common/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "interface.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
