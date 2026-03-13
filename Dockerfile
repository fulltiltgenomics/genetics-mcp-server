FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y sqlite3 && rm -rf /var/lib/apt/lists/*

RUN pip install uv --upgrade

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY src/ src/

ENV PORT=8000
ENV PYTHONPATH=/app/src

EXPOSE 8000 8080

# default: run chat API; override CMD to run MCP server
CMD ["uvicorn", "genetics_mcp_server.chat_api:app", "--host", "0.0.0.0", "--port", "8000"]
