# Regras de Operação e Restrições de Ambiente (AGENTS.md)

## ⛔ PROIBIÇÃO ABSOLUTA DE EXECUÇÃO NESTE MAC

1. **Natureza desta Máquina**:
   - Este computador é um MacBook (macOS) utilizado **EXCLUSIVAMENTE** para desenvolvimento, edição de arquivos e versionamento Git.
   - O servidor real de produção/execução é uma máquina dedicada rodando **Nobara Linux 44**, operada pelo próprio usuário via conexão SSH.

2. **Comandos Estritamente Proibidos no Mac**:
   - **NUNCA** execute scripts shell (`./setup_service.sh`, `bash *.sh`, `sh *.sh`, scripts de instalação ou deploy).
   - **NUNCA** execute o projeto ou qualquer runtime Python (`python3 -m uvicorn`, `python3 app/...`, `python app.py`, etc.).
   - **NUNCA** execute comandos de gerenciamento de serviços (`systemctl`, `service`, etc.) que pertencem ao servidor Linux.
   - **NUNCA** execute instaladores de dependências globais ou de runtime (`pip install`, etc.).

3. **O Que é Permitido no Mac**:
   - Operações normais de versionamento Git: `git status`, `git diff`, `git add`, `git commit`.
   - Edição, criação e leitura de código e documentação.

4. **Diretrizes de Comunicação**:
   - Sempre responder em português do Brasil.
   - Commits sempre em português do Brasil.
