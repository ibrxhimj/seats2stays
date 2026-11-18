# Seats2Stays — the simulator, in one image.   docker build -t seats2stays .   docker run --rm -p 8000:8000 seats2stays
# Three stages: (1) the pinned Python packages, (2) the data bundle (only the files the simulator needs, copied from the
# project's processed tables and checked by running a scenario), (3) the runtime: packages + code + web + bundle, as a
# non-root user. No raw competition data, no Excel files and no passenger- or guest-level tables enter the image.
FROM python:3.12-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY requirements-app.txt /tmp/requirements-app.txt
RUN /opt/venv/bin/pip install -r /tmp/requirements-app.txt

FROM deps AS bundle
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/build/src PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
COPY config.yaml ./
COPY src ./src
COPY data/processed ./data/processed
COPY data/reference ./data/reference
RUN python -m atp.phase4 bundle --out /out/bundle \
 && ATP_ROOT=/out/bundle python -c "from atp.simulator import Simulator; s = Simulator(); r = s.run(s.preset('etihad')); assert r['headline'], r; print('bundle runs:', r['headline']['text'], r['headline']['range_text'])"

FROM python:3.12-slim AS runtime
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/app/src ATP_ROOT=/app/bundle ATP_WEB=/app/web \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8000
RUN useradd --system --uid 10001 --no-create-home app
COPY --from=deps /opt/venv /opt/venv
COPY src /app/src
COPY web /app/web
COPY --from=bundle /out/bundle /app/bundle
USER app
WORKDIR /app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["python", "-m", "atp.api"]
