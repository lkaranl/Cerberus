#!/usr/bin/env bash
# ==============================================================================
# Script de Instalação e Ativação Automática do Serviço Cerberus no systemd
# ==============================================================================

set -e

# Cores para saída no terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # Sem cor

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}   Cerberus - Instalação de Serviço Automático (24/7)  ${NC}"
echo -e "${CYAN}======================================================${NC}"

# Detecta usuário real (caso executado com sudo)
REAL_USER="${SUDO_USER:-$USER}"
REAL_GROUP="$(id -gn "$REAL_USER" 2>/dev/null || echo "$REAL_USER")"

# Diretório raiz do repositório onde este script reside
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Localiza o executável do Python (prioriza o ambiente virtual .venv)
if [ -f "$REPO_DIR/.venv/bin/python3" ]; then
    PYTHON_BIN="$REPO_DIR/.venv/bin/python3"
    ENV_PATH="$REPO_DIR/.venv/bin:/usr/local/bin:/usr/bin"
elif [ -f "$REPO_DIR/venv/bin/python3" ]; then
    PYTHON_BIN="$REPO_DIR/venv/bin/python3"
    ENV_PATH="$REPO_DIR/venv/bin:/usr/local/bin:/usr/bin"
else
    PYTHON_BIN="$(which python3)"
    ENV_PATH="/usr/local/bin:/usr/bin"
fi

echo -e "📂 Diretório do projeto: ${YELLOW}$REPO_DIR${NC}"
echo -e "👤 Usuário do serviço:   ${YELLOW}$REAL_USER ($REAL_GROUP)${NC}"
echo -e "🐍 Executável Python:     ${YELLOW}$PYTHON_BIN${NC}"

SERVICE_FILE="/etc/systemd/system/cerberus.service"
TEMP_SERVICE="$(mktemp)"

# Gera o conteúdo do serviço adaptado ao ambiente atual
cat <<EOF > "$TEMP_SERVICE"
[Unit]
Description=Cerberus (Sentinela Vision) - Servico de Monitoramento e IA 24/7
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$REAL_USER
Group=$REAL_GROUP
WorkingDirectory=$REPO_DIR
Environment="PATH=$ENV_PATH"
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=-$REPO_DIR/.env
ExecStart=$PYTHON_BIN -m uvicorn app.main:app --host 0.0.0.0 --port 8095
Restart=always
RestartSec=5s
RuntimeMaxSec=24h
LimitNOFILE=65535
SyslogIdentifier=cerberus-vision
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Atualiza também o arquivo local cerberus.service para referência
cp "$TEMP_SERVICE" "$REPO_DIR/cerberus.service"

echo -e "\n📋 Copiando arquivo de serviço para ${YELLOW}$SERVICE_FILE${NC}..."

# Copia para o systemd (solicita sudo se necessário)
if [ "$EUID" -ne 0 ]; then
    sudo cp "$TEMP_SERVICE" "$SERVICE_FILE"
    sudo chmod 644 "$SERVICE_FILE"
    rm -f "$TEMP_SERVICE"

    echo -e "🔄 Recarregando configurações do systemd..."
    sudo systemctl daemon-reload

    echo -e "⚡ Habilitando inicialização automática no boot..."
    sudo systemctl enable cerberus.service

    echo -e "🚀 Iniciando o serviço Cerberus..."
    sudo systemctl restart cerberus.service
else
    cp "$TEMP_SERVICE" "$SERVICE_FILE"
    chmod 644 "$SERVICE_FILE"
    rm -f "$TEMP_SERVICE"

    echo -e "🔄 Recarregando configurações do systemd..."
    systemctl daemon-reload

    echo -e "⚡ Habilitando inicialização automática no boot..."
    systemctl enable cerberus.service

    echo -e "🚀 Iniciando o serviço Cerberus..."
    systemctl restart cerberus.service
fi

echo -e "⏳ Aguardando verificação de integridade..."
sleep 2.5

# Verifica se o serviço está ativo e rodando sem erros
if systemctl is-active --quiet cerberus; then
    # Detecta o IP real da máquina na rede local
    SERVER_IP=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n 1)
    if [ -z "$SERVER_IP" ]; then
        SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
    if [ -z "$SERVER_IP" ]; then
        SERVER_IP="localhost"
    fi

    echo -e "\n${GREEN}======================================================${NC}"
    echo -e "${GREEN}  ✓ Serviço Cerberus iniciado com sucesso e rodando!  ${NC}"
    echo -e "${GREEN}======================================================${NC}"
    echo -e "🌐 Acesse no navegador:"
    echo -e "   • Monitoramento: ${CYAN}http://${SERVER_IP}:8095${NC}"
    echo -e "   • Dashboard:     ${CYAN}http://${SERVER_IP}:8095/dashboard${NC}\n"
    echo -e "📜 Comandos úteis:"
    echo -e "   • Ver status:    systemctl status cerberus"
    echo -e "   • Acompanhar log: journalctl -u cerberus -f"
    echo -e "   • Parar serviço: sudo systemctl stop cerberus\n"
else
    echo -e "\n${RED}======================================================${NC}"
    echo -e "${RED}  ✗ O serviço falhou ao iniciar. Verifique os logs:   ${NC}"
    echo -e "${RED}======================================================${NC}"
    journalctl -u cerberus -n 25 --no-pager
    exit 1
fi
