# syntax=docker/dockerfile:1.7
# NOTE: We use debian:bookworm-slim here instead of distroless because the binary
# links against LLVM libc++ (and c++abi). This base lets us install the required
# runtime libraries via apt without vendoring .so files. If we later switch to
# statically linking libc++ or to a base that ships libc++, we can revisit.

FROM bitnami/minideb:bookworm

ARG TARGETOS
ARG TARGETARCH
ARG VERSION
ARG VCS_REF
ARG REPO_URL
# For labeling only; the ORT libs are copied in from dist/ by the publish job
ARG ORT_VERSION=1.22.0
ARG ORT_FLAVOR=cpu

# Minimal runtime dependencies: libc++ and libc++abi for LLVM C++ runtime,
# and CA certs for TLS. Keep layer small and clean.
RUN set -eux; \
    apt-get update; \
    install_packages \
      libc++1 \
      libc++abi1 \
      ca-certificates; \
    rm -rf /var/lib/apt/lists/*

# OCI labels
LABEL org.opencontainers.image.title="breeze"
LABEL org.opencontainers.image.description="Semantic code chunking CLI and HTTP/MCP server"
LABEL org.opencontainers.image.source="${REPO_URL}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.vendor="casualjim"
LABEL org.opencontainers.image.licenses="MIT"

# Runtime search path so the dynamically loaded ORT libs are found
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/opt/onnxruntime/lib
# Default data directory and TLS disabled by default
ENV BREEZE_DB_DIR=/data/embeddings.db
ENV BREEZE_SERVER_TLS_DISABLED=true

# Copy prebuilt binary and ONNX Runtime libs into place. The publish job prepares
# the dist layout per arch so this is a pure COPY (no network in Docker build).
# Expected layout in build context:
#   dist/${TARGETOS}-${TARGETARCH}/breeze
#   dist/${TARGETOS}-${TARGETARCH}/ort/lib/...
COPY dist/${TARGETOS}-${TARGETARCH}/breeze /usr/local/bin/breeze
COPY dist/${TARGETOS}-${TARGETARCH}/ort/lib/ /opt/onnxruntime/lib/

# Make sure the binary is executable
RUN chmod +x /usr/local/bin/breeze

# Non-root runtime, data volume, and default ports
RUN useradd -r -u 65532 -g root breezeuser \
 && mkdir -p /data \
 && chown -R 65532:0 /data /usr/local/bin /opt/onnxruntime \
 && chmod -R g=u /data /usr/local/bin /opt/onnxruntime
USER 65532:0

VOLUME ["/data"]
EXPOSE 8528 8529

# Default command
ENTRYPOINT ["/usr/local/bin/breeze", "serve"]
