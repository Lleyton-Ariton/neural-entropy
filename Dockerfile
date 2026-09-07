FROM rust:1.90-slim AS builder

RUN apt-get update && apt-get install -y python3-pip \
    && pip3 install --break-system-packages maturin

WORKDIR /golem/work
COPY lsm ./lsm

WORKDIR /golem/work/lsm
RUN maturin build --release -o /golem/work/dist

FROM python:3.13-slim

WORKDIR /golem/work

COPY --from=builder /golem/work/dist/*.whl .

RUN pip install --no-cache-dir *.whl

COPY simulation.py .

CMD ["/bin/bash"]
