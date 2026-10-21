#!/usr/bin/env bash
# ==============================================================================
# Script de Instalação e Configuração do Gateway WebRTC go2rtc para o Cerberus
# Deve ser executado EXCLUSIVAMENTE no servidor Nobara Linux 44.
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GO2RTC_VERSION="v1.9.8"
INSTALL_BIN="/usr/local/bin/go2rtc"

echo "============================================================"
echo "🚀 Instalando Gateway WebRTC go2rtc (${GO2RTC_VERSION})..."
echo "============================================================"

# Verifica se está rodando como root ou com sudo disponível
if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
else
    SUDO=""
fi

# 1. Download do binário compilado oficial para Linux x86_64
TMP_BIN="/tmp/go2rtc_linux_amd64"
echo "⬇️ Baixando go2rtc ${GO2RTC_VERSION} do GitHub..."
curl -L -s -o "$TMP_BIN" "https://github.com/AlexxIT/go2rtc/releases/download/${GO2RTC_VERSION}/go2rtc_linux_amd64"

# 2. Instalação em /usr/local/bin
$SUDO mv "$TMP_BIN" "$INSTALL_BIN"
$SUDO chmod +x "$INSTALL_BIN"
echo "✅ Binário instalado com sucesso em ${INSTALL_BIN}"

# 3. Criação do Serviço Systemd
SERVICE_FILE="/etc/systemd/system/go2rtc.service"
echo "⚙️ Configurando serviço systemd em ${SERVICE_FILE}..."

$SUDO bash -c "cat <<EOF > ${SERVICE_FILE}
[Unit]
Description=go2rtc WebRTC Gateway for Cerberus CCTV
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${INSTALL_BIN} -config ${PROJECT_DIR}/go2rtc.yaml
Restart=always
RestartSec=3
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF"

# 4. Habilita e inicializa o serviço
$SUDO systemctl daemon-reload
$SUDO systemctl enable --now go2rtc

echo "============================================================"
echo "🎉 go2rtc configurado e rodando com sucesso!"
echo "📡 API WHEP disponível em: http://localhost:1984"
echo "📊 Verifique os logs com: journalctl -u go2rtc -f"
echo "============================================================"
