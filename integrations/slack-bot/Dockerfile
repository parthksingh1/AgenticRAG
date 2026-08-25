FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

# Non-root: the process writes nothing, so it should not be able to.
RUN useradd -u 1000 -m bot
USER bot

EXPOSE 3100
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:3100/healthz')"

CMD ["python", "app.py"]
