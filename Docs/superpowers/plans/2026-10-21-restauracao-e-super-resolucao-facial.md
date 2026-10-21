# 📋 Plano de Acompanhamento: Restauração e Super-Resolução Facial SOTA

> **Objetivo**: Superar limitações ópticas de câmeras 2K com grande-angular (rostos distantes de 30-40px em sombras profundas sob marquises) através de pipeline 100% via software, sem alterações nas configurações físicas das câmeras.

---

## 🏗️ Status Geral das Etapas

- [x] **Módulo 1: Realce Fotométrico Adaptativo (P&B + CLAHE & Shadow Lifting)** 🚀 *Concluído - Pronto para Validação*
  - [x] Especificação e desenho técnico do realce fotométrico.
  - [x] Adicionar parâmetros de shadow lifting em `app/core/config.py`.
  - [x] Implementar `enhance_shadow_face` e detecção *Dual-Pass* em `app/detector/recognizer.py`.
  - [x] Criar ferramenta de teste e comparação visual: `scripts/test_shadow_lifting.py`.
  - [ ] Validar ganho de detecção em recortes escuros no Nobara Linux.

- [ ] **Módulo 2: Super-Resolução Facial Neural (AI Face Restoration ONNX FP16)** ⏳ *Pendente*
  - [ ] Selecionar e disponibilizar modelo neural ultraleve de restauração facial (Real-ESRGAN Face Compact / GFPGAN Lite ONNX).
  - [ ] Criar classe `FaceSuperResolver` com aceleração CUDA/ONNX Runtime.
  - [ ] Integrar ampliação 4x condicional para faces com tamanho < 60px.
  - [ ] Criar script de benchmark de latência e VRAM: `scripts/test_face_super_res.py`.

- [ ] **Módulo 3: Integração Completa, Validação & Consenso Temporal** ⏳ *Pendente*
  - [ ] Conectar o pipeline completo (P&B CLAHE -> SCRFD 640 -> Super-Resolução -> ArcFace 512-d).
  - [ ] Ajustar o `TrackerManager` para manter o histórico de assinaturas faciais restauradas.
  - [ ] Executar auditoria biométrica completa com o script `diagnose_face_biometrics.py`.
  - [ ] Homologação em produção no servidor Nobara Linux 44.

---

## 📊 Arquitetura do Fluxo de Dados

```mermaid
graph LR
    A[Recorte Pessoa 2K] --> B{Análise de Sombra}
    B -->|Sombra Profunda| C[Módulo 1: P&B CLAHE + Gamma]
    B -->|Normal| D[Imagem Original]
    C --> E[SCRFD 640x640]
    D --> E
    E --> F{Face < 60px?}
    F -->|Sim| G[Módulo 2: Super-Resolução 4x]
    F -->|Não| H[Alinhamento Nativo]
    G --> I[ArcFace 512-d]
    H --> I
    I --> J[Consenso Temporal Tracker]
```

---

## 📝 Registro de Atualizações
- **2026-10-21**: Criação do plano e inicialização do Módulo 1 (Realce Fotométrico e P&B CLAHE).
