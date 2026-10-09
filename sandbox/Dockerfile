FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

RUN pip install --no-cache-dir pytest \
    && useradd --create-home --uid 65532 sandbox

USER sandbox
WORKDIR /workspace

