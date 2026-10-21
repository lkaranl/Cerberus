# Plano de Implementação — Cerberus Dual-GPU: Worker Especialista na AMD RX 5600 XT

> **Para executores agenticos:** SUB-HABILIDADE OBRIGATÓRIA: Utilize `superpowers:subagent-driven-development` (recomendado) ou `superpowers:executing-plans` para implementar este plano tarefa por tarefa. As etapas utilizam a sintaxe de caixa de seleção (`- [ ]`) para rastreamento.

**Objetivo:** Criar e integrar o **Cerberus AMD Worker**, um microsserviço desacoplado que utiliza o poder bruto da **AMD Radeon RX 5600 XT (6 GB GDDR6)** para tarefas pesadas por demanda (Super-Resolução Facial CodeFormer e Reconhecimento Biométrico), mantendo a **NVIDIA GTX 1660** 100% dedicada e sem engasgos no pipeline contínuo em tempo real (NVDEC + YOLOv8 + ByteTrack).

**Arquitetura:** Arquitetura de microsserviço produtor-consumidor via HTTP assíncrono interno (`127.0.0.1:8001`). O processo principal do Cerberus (NVIDIA CUDA) despacha cortes de rostos e placas via requisição não-bloqueante para o worker isolado da AMD (executando via ONNX Runtime Vulkan ou PyTorch ROCm). Se o worker AMD estiver ocupado ou indisponível, há fallback automático transparente para processamento local na 1660.

**Tech Stack:**
- **Host GPU NVIDIA:** CUDA 12, TensorRT, NVDEC, PyTorch (GTX 1660).
- **Worker GPU AMD:** ONNX Runtime com Vulkan Execution Provider / ROCm (`HSA_OVERRIDE_GFX_VERSION=10.3.0`) na RX 5600 XT.
- **Protocolo IPC:** FastAPI / Uvicorn em porta local `8001` (com payload JSON + Base64 ou binário multipart).
- **Testes:** Scripts de benchmark de hardware e testes de estresse assíncronos.

---

## Restrições Globais

- O processo do Cerberus principal (NVIDIA) e o Worker (AMD) devem rodar em **processos Python separados** para evitar conflito de bibliotecas dinâmicas (`libcuda.so` vs `libhsa-runtime64.so`).
- Nenhuma falha no Worker AMD pode interromper a gravação de vídeo ou o loop contínuo de 20 FPS das câmeras na GTX 1660.
- Comunicação e logs padronizados em Português do Brasil.
- Commits estritamente em Português do Brasil.

---

## Estrutura de Arquivos Planejada

```
app/
├── worker_amd/                    # Novo pacote isolado do Worker AMD
│   ├── __init__.py
│   ├── server.py                  # API FastAPI ultraleve do worker (porta 8001)
│   ├── engine_amd.py              # Gerenciador de inferência ONNX Runtime Vulkan/ROCm
│   └── models/                    # Carregadores dos modelos de super-resolução e face
├── core/
│   └── amd_client.py              # Cliente assíncrono do Cerberus para despachar tarefas
scripts/
├── benchmark_dual_gpu.py          # Teste e benchmark comparativo entre GTX 1660 e RX 5600 XT
└── test_amd_worker_integration.py # Teste de estresse assíncrono de ponta a ponta
```

---

## Checklist de Tarefas de Implementação

### Tarefa 1: Script de Diagnóstico e Validação da AMD RX 5600 XT
**Arquivos:**
- Criar: `scripts/benchmark_dual_gpu.py`

**Interfaces:**
- Produz: Relatório impresso detalhando se Vulkan e ROCm estão ativos na GPU AMD, memória VRAM livre e taxa de TFLOPs medida.

- [x] **Passo 1.1: Criar o script de diagnóstico de hardware `scripts/benchmark_dual_gpu.py`**
  - Detectar a presença do dispositivo Vulkan (AMD RADV Navi 10) e checar suporte a ROCm.
  - Executar multiplicação densa de matrizes (GEMM FP16/FP32) para medir o tempo de resposta em ms.
- [x] **Passo 1.2: Adicionar validação de fallback caso apenas a NVIDIA esteja disponível no ambiente de teste.**
- [x] **Passo 1.3: Commit:** `git commit -m "test: adicionar script de benchmark e deteccao de hardware dual gpu"`

---

### Tarefa 2: Módulo de Inferência ONNX / Vulkan da AMD (`engine_amd.py`)
**Arquivos:**
- Criar: `app/worker_amd/engine_amd.py`
- Criar: `app/worker_amd/__init__.py`

**Interfaces:**
- Produz: Classe `AmdInferenceEngine` com métodos:
  - `restore_face(image_np: np.ndarray) -> np.ndarray`
  - `compute_face_embedding(face_crop: np.ndarray) -> list[float]`
  - `get_device_info() -> dict`

- [ ] **Passo 2.1: Criar classe `AmdInferenceEngine` configurada para direcionar tensores ao adapter Vulkan da AMD.**
- [ ] **Passo 2.2: Implementar fallback gracioso para CPU caso a GPU AMD esteja momentaneamente em uso exclusivo de renderização 3D pelo GNOME.**
- [ ] **Passo 2.3: Commit:** `git commit -m "feat(amd-worker): criar engine de inferencia com aceleracao vulkan para rx 5600 xt"`

---

### Tarefa 3: Servidor Microsserviço do Worker AMD (`server.py`)
**Arquivos:**
- Criar: `app/worker_amd/server.py`

**Interfaces:**
- Produz: Endpoints HTTP locais:
  - `GET /health` -> `{ "status": "online", "gpu": "AMD Radeon RX 5600 XT", "backend": "Vulkan" }`
  - `POST /api/v1/restore-face` -> Recebe imagem base64 e retorna imagem restaurada em alta definição.
  - `POST /api/v1/face-embedding` -> Recebe imagem recortada e retorna vetor biométrico normalizado (512-d).

- [ ] **Passo 3.1: Escrever servidor FastAPI desacoplado ouvindo em `127.0.0.1:8001`.**
- [ ] **Passo 3.2: Implementar fila com controle de concorrência (`asyncio.Semaphore`) para proteger a VRAM da RX 5600 XT contra estouro.**
- [ ] **Passo 3.3: Commit:** `git commit -m "feat(amd-worker): criar servidor fastapi desacoplado para worker amd"`

---

### Tarefa 4: Cliente Assíncrono no Cerberus Principal (`amd_client.py`)
**Arquivos:**
- Criar: `app/core/amd_client.py`
- Modificar: `app/detector/recognizer.py` (ou pipeline biométrico)

**Interfaces:**
- Consumes: Endpoints do Worker AMD na porta 8001.
- Produz: `AmdWorkerClient.dispatch_face_restoration(crop: np.ndarray) -> Optional[np.ndarray]`

- [ ] **Passo 4.1: Criar cliente assíncrono com pool de conexões HTTP `httpx.AsyncClient` com timeout configurável (máx 1.5s).**
- [ ] **Passo 4.2: Implementar política de fallback automático:**
  - Se o Worker AMD responder em <1.5s, usar o resultado de alta fidelidade da RX 5600 XT.
  - Se o Worker AMD estiver desligado ou der timeout, processar localmente na GTX 1660 sem travar a thread de vídeo.
- [ ] **Passo 4.3: Conectar o `AmdWorkerClient` aos gatilhos de corte de rosto do Cerberus.**
- [ ] **Passo 4.4: Commit:** `git commit -m "feat(core): integrar cliente assincrono do cerberus com worker amd"`

---

### Tarefa 5: Script de Teste e Validação Ponta a Ponta
**Arquivos:**
- Criar: `scripts/test_amd_worker_integration.py`

**Interfaces:**
- Produz: Relatório completo de testes de estresse:
  - Latência média por requisição (ms).
  - Verificação de integridade da imagem restaurada.
  - Teste de resiliência a quedas (desligar o worker e verificar se o Cerberus continua operando na 1660).

- [ ] **Passo 5.1: Escrever script de teste ponta a ponta simulando 10 requisições simultâneas de cortes de rosto.**
- [ ] **Passo 5.2: Medir o tempo de resposta e certificar ausência de travamentos no stream principal.**
- [ ] **Passo 5.3: Commit:** `git commit -m "test: adicionar suite de validacao de integracao e resiliencia dual gpu"`

---

## Como Saber se Deu Certo (Critérios de Sucesso dos Testes)

1. **Isolamento de Processo**:
   - `nvidia-smi` mostra apenas o Cerberus principal consumindo ~1.3 GB de VRAM na GTX 1660.
   - `radeontop` (ou `rocm-smi`) mostra a atividade de computação e uso de VRAM na RX 5600 XT apenas quando ocorrem eventos de pessoas/carros.
2. **Impacto Zero no FPS das Câmeras**:
   - O FPS das 3 câmeras permanece estável em **16~20 FPS** mesmo durante a restauração facial pesada.
3. **Qualidade Visual Aumentada**:
   - O recorte biométrico salvo no disco ganha nitidez visual reconstruída pela GPU secundária.
4. **Resiliência a Falhas**:
   - Ao matar o processo do worker AMD (`pkill -f worker_amd`), o Cerberus principal continua rodando e emitindo alertas sem quebrar.
