FROM mcr.microsoft.com/devcontainers/base:bookworm

# Features install packages before the workspace .npmrc is available.
ENV NPM_CONFIG_REGISTRY=https://packagefeedproxy.microsoft.io/npm/ \
    NPM_CONFIG_REPLACE_REGISTRY_HOST=npmjs \
    COREPACK_NPM_REGISTRY=https://packagefeedproxy.microsoft.io/npm/ \
    YARN_NPM_REGISTRY_SERVER=https://packagefeedproxy.microsoft.io/npm/ \
    UV_DEFAULT_INDEX=https://packagefeedproxy.microsoft.io/pypi/simple/ \
    PIP_INDEX_URL=https://packagefeedproxy.microsoft.io/pypi/simple/ \
    UV_CACHE_DIR=/home/vscode/.cache/uv-cfs
