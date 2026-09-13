# Named volumes start root-owned; make them writable by the development user.
sudo chown "$(id -u):$(id -g)" \
    node_modules .venv .npm-cfs .next \
    /home/vscode/.local /home/vscode/.local/share \
    /home/vscode/.local/share/uv /home/vscode/.local/share/uv/python || exit

# Setup Git completion
echo "source /usr/share/bash-completion/completions/git" >> ~/.bashrc

# Install Claude Code
curl -fsSL https://claude.ai/install.sh | bash

# Install Codex
curl -fsSL https://chatgpt.com/codex/install.sh | CODEX_NON_INTERACTIVE=1 sh
