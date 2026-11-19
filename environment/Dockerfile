# Build from environment/ (the only source tree in the solve image).
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

ENV LANG=C.UTF-8 \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git patch ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /bin/bash dev

COPY requirements.lock /opt/benchmark/requirements.lock
# sqlitedict is distributed as a source archive. Bootstrap its build backend
# from the same lock before installing the hashed runtime dependencies.
RUN grep -Ei '^(setuptools|wheel)==' /opt/benchmark/requirements.lock > /tmp/build-tools.lock \
    && python -m pip install --require-hashes -r /tmp/build-tools.lock \
    && python -m pip install --no-build-isolation --require-hashes --only-binary=:all: \
       --no-binary=sqlitedict -r /opt/benchmark/requirements.lock

COPY --chown=dev:dev workspace/repo /workspace/repo
# Register the source checkout as the lm_eval distribution. The deterministic
# CPU adapter only uses the registry and model API; it never downloads weights.
RUN python -m pip install --no-index --no-deps --no-build-isolation -e /workspace/repo \
    && rm -rf /workspace/repo/lm_eval.egg-info

USER dev
WORKDIR /workspace/repo
RUN git init -q -b main \
    && git add -f . \
    && git -c user.name=benchmark -c user.email=snapshot@example.invalid commit -qm snapshot

WORKDIR /workspace
CMD ["sleep", "infinity"]
