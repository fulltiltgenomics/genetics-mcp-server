FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y sqlite3 && rm -rf /var/lib/apt/lists/*

# pinned: an unpinned uv is an unversioned input to every resolution below it
RUN pip install uv==0.11.3

# the lock alone is enough to install every dependency, so this layer is copied and run
# before any source: editing src/ then reuses it instead of reinstalling the whole set.
# --frozen fails rather than re-resolving, so the image can never ship a dependency set
# the lockfile does not describe
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --no-dev --no-emit-project -o /tmp/requirements.txt \
    && uv pip install --system -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# only the project itself; --no-deps keeps this step from touching the resolved set above
COPY README.md ./
COPY src/ src/
RUN uv pip install --system --no-deps .

ENV PORT=8000
ENV PYTHONPATH=/app/src

EXPOSE 8000 8080

# default: run chat API; override CMD to run MCP server
CMD ["uvicorn", "genetics_mcp_server.chat_api:app", "--host", "0.0.0.0", "--port", "8000"]
