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
WORKER_SERVICE_FILE="/etc/systemd/system/cerberus-worker-amd.service"
TEMP_SERVICE="$(mktemp)"
TEMP_WORKER_SERVICE="$(mktemp)"

# 1. Gera o conteúdo do serviço principal (NVIDIA / Web / Câmeras)
cat <<EOF > "$TEMP_SERVICE"
[Unit]
Description=Cerberus (Sentinela Vision) - Servico de Monitoramento e IA 24/7
After=network-online.target cerberus-worker-amd.service go2rtc.service
Wants=network-online.target cerberus-worker-amd.service go2rtc.service

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

# 2. Gera o conteúdo do microsserviço especialista (AMD Radeon RX 5600 XT)
cat <<EOF > "$TEMP_WORKER_SERVICE"
[Unit]
Description=Cerberus AMD Specialist Worker (Radeon RX 5600 XT) - Microsservico de IA por Demanda
After=network.target

[Service]
Type=simple
User=$REAL_USER
Group=$REAL_GROUP
WorkingDirectory=$REPO_DIR
Environment="PATH=$ENV_PATH"
Environment="PYTHONUNBUFFERED=1"
Environment="HSA_OVERRIDE_GFX_VERSION=10.3.0"
ExecStart=$PYTHON_BIN app/worker_amd/server.py
Restart=always
RestartSec=5s
LimitNOFILE=65535
SyslogIdentifier=cerberus-amd
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Atualiza também os arquivos locais para referência
cp "$TEMP_SERVICE" "$REPO_DIR/cerberus.service"
cp "$TEMP_WORKER_SERVICE" "$REPO_DIR/cerberus-worker-amd.service"

echo -e "\n📋 Copiando arquivos de serviço para o systemd..."

# Copia para o systemd (solicita sudo se necessário)
if [ "$EUID" -ne 0 ]; then
    sudo cp "$TEMP_SERVICE" "$SERVICE_FILE"
    sudo chmod 644 "$SERVICE_FILE"
    sudo cp "$TEMP_WORKER_SERVICE" "$WORKER_SERVICE_FILE"
    sudo chmod 644 "$WORKER_SERVICE_FILE"
    rm -f "$TEMP_SERVICE" "$TEMP_WORKER_SERVICE"

    echo -e "🔄 Recarregando configurações do systemd..."
    sudo systemctl daemon-reload

    echo -e "⚡ Habilitando inicialização automática no boot..."
    sudo systemctl enable cerberus-worker-amd.service
    sudo systemctl enable cerberus.service

    echo -e "🚀 Iniciando os serviços Dual-GPU..."
    sudo systemctl restart cerberus-worker-amd.service
    sudo systemctl restart cerberus.service
else
    cp "$TEMP_SERVICE" "$SERVICE_FILE"
    chmod 644 "$SERVICE_FILE"
    cp "$TEMP_WORKER_SERVICE" "$WORKER_SERVICE_FILE"
    chmod 644 "$WORKER_SERVICE_FILE"
    rm -f "$TEMP_SERVICE" "$TEMP_WORKER_SERVICE"

    echo -e "🔄 Recarregando configurações do systemd..."
    systemctl daemon-reload

    echo -e "⚡ Habilitando inicialização automática no boot..."
    systemctl enable cerberus-worker-amd.service
    systemctl enable cerberus.service

    echo -e "🚀 Iniciando os serviços Dual-GPU..."
    systemctl restart cerberus-worker-amd.service
    systemctl restart cerberus.service
fi

echo -e "⏳ Aguardando verificação de integridade..."
sleep 2.5

# Verifica se os serviços estão ativos e rodando sem erros
if systemctl is-active --quiet cerberus; then
    SERVER_IP=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{print $7}' | head -n 1)
    if [ -z "$SERVER_IP" ]; then
        SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
    if [ -z "$SERVER_IP" ]; then
        SERVER_IP="localhost"
    fi

    echo -e "\n${GREEN}======================================================${NC}"
    echo -e "${GREEN}  ✓ Serviços Dual-GPU iniciados com sucesso!           ${NC}"
    echo -e "${GREEN}======================================================${NC}"
    echo -e "🌐 Acesse no navegador:"
    echo -e "   • Monitoramento: ${CYAN}http://${SERVER_IP}:8095${NC}"
    echo -e "   • Dashboard:     ${CYAN}http://${SERVER_IP}:8095/dashboard${NC}"
    echo -e "   • Worker AMD:    ${CYAN}http://127.0.0.1:8001/health${NC}"
    echo -e "   • Gateway go2rtc:${CYAN}http://${SERVER_IP}:1984${NC}\n"
    echo -e "📜 Comandos úteis:"
    echo -e "   • Status Cerberus:   systemctl status cerberus"
    echo -e "   • Status Worker AMD: systemctl status cerberus-worker-amd"
    echo -e "   • Status go2rtc:     systemctl status go2rtc"
    echo -e "   • Logs Cerberus:     journalctl -u cerberus -f"
    echo -e "   • Logs Worker AMD:   journalctl -u cerberus-worker-amd -f"
    echo -e "   • Logs go2rtc:       journalctl -u go2rtc -f\n"
else
    echo -e "\n${RED}======================================================${NC}"
    echo -e "${RED}  ✗ O serviço falhou ao iniciar. Verifique os logs:   ${NC}"
    echo -e "${RED}======================================================${NC}"
    journalctl -u cerberus -n 25 --no-pager
    exit 1
fi
