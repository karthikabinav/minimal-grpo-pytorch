FROM pytorch/pytorch:2.5.1-cpu

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir mcp

ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8000

EXPOSE 8000

CMD ["python", "mcp_server.py"]
