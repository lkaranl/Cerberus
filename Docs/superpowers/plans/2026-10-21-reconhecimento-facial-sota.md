# Plano de Implementação: Reconhecimento Facial SOTA de Alta Precisão (Cerberus)

> **Para executores:** Implementação tarefa por tarefa. Passos utilizam sintaxe de checkbox (`- [ ]`) para acompanhamento.

**Objetivo:** Eliminar falsos positivos e garantir o reconhecimento confiável dos moradores cadastrados através de Galeria Multi-Embedding, Filtro de Nitidez (FQA), detector SCRFD 640x640 e Consenso Temporal de 3 frames no rastreador.

**Arquitetura:** 
1. `app/core/config.py`: Parâmetros calibrados de biometria facial, limiares de decisão e nitidez.
2. `app/detector/recognizer.py`: Motor InsightFace atualizado para SCRFD 640x640 + ArcFace ResNet-50 (`buffalo_l`), galeria multi-vetorial por morador e filtro FQA contra borrões de movimento.
3. `app/detector/tracker.py`: Acumulador de evidências temporais por rastro de pessoa com dupla confirmação (alta certeza instantânea ou consenso de 3 frames).
4. `scripts/diagnose_face_biometrics.py`: Ferramenta de auditoria da base cadastrada e validação biométrica com recortes reais.

**Tech Stack:** Python 3.14 / Nobara Linux, PyTorch, InsightFace (ArcFace ResNet-50 / SCRFD), OpenCV, NumPy, ONNX Runtime CUDA.

## Restrições Globais
- Respostas e commits estritamente em Português do Brasil.
- Nenhuma execução de runtime Python ou instalação de dependências no Mac de desenvolvimento; execuções de teste sempre no servidor Nobara Linux.

---

### Tarefa 1: Calibração de Parâmetros de Biometria Facial (`app/core/config.py`)

**Arquivos:**
- Modificar: `app/core/config.py`

**Interfaces:**
- Produz: `settings.face_confidence_threshold` (0.58), `settings.face_instant_threshold` (0.68), `settings.face_min_size` (48), `settings.face_quality_min_blur` (55.0), `settings.face_temporal_votes` (3), `settings.face_model_name` ("buffalo_l").

- [ ] **Passo 1: Adicionar novos campos de configuração facial com aliases de ambiente**
- [ ] **Passo 2: Validar integridade dos tipos e valores padrão**
- [ ] **Passo 3: Commit no Git**

---

### Tarefa 2: Galeria Multi-Embedding e Filtro FQA no Reconhecedor (`app/detector/recognizer.py`)

**Arquivos:**
- Modificar: `app/detector/recognizer.py`

**Interfaces:**
- Consome: `settings.face_confidence_threshold`, `settings.face_min_size`, `settings.face_quality_min_blur`.
- Produz: `face_recognizer.known_gallery: Dict[str, List[np.ndarray]]`, `face_recognizer.detect_and_identify_person(frame, bbox) -> (name, conf, is_known, face_bbox)`.

- [ ] **Passo 1: Atualizar inicialização do InsightFace para tentar `buffalo_l` com fallback para `buffalo_s` e `det_size=(640, 640)`**
- [ ] **Passo 2: Substituir o centróide médio `mean_emb` pela Galeria Multi-Embedding pura**
- [ ] **Passo 3: Implementar Face Quality Assessment (FQA): tamanho mínimo de 48x48 px, proporção anatômica e filtro de variância Laplaciana (blur gate >= 55.0)**
- [ ] **Passo 4: Implementar busca por similaridade máxima k-NN contra todos os vetores de cada morador**
- [ ] **Passo 5: Commit no Git**

---

### Tarefa 3: Consenso Temporal de Decisão no Rastreador (`app/detector/tracker.py`)

**Arquivos:**
- Modificar: `app/detector/tracker.py`

**Interfaces:**
- Consome: `face_recognizer.detect_and_identify_person()`, `settings.face_instant_threshold`, `settings.face_temporal_votes`.
- Produz: `TrackedEntity.face_evidence_history`, `TrackedEntity.is_known` confirmado com consenso.

- [ ] **Passo 1: Adicionar acumulador de votos `face_evidence_history` em `TrackedEntity`**
- [ ] **Passo 2: Implementar critério de dupla confirmação: instantânea ($\ge 0.68$) ou consenso ($\ge 0.58$ em 3 frames)**
- [ ] **Passo 3: Implementar faixa de incerteza protegida (0.48 a 0.57) para aguardar aproximação e não alertar falso desconhecido**
- [ ] **Passo 4: Commit no Git**

---

### Tarefa 4: Script de Diagnóstico e Auditoria Biométrica (`scripts/diagnose_face_biometrics.py`)

**Arquivos:**
- Criar: `scripts/diagnose_face_biometrics.py`

**Interfaces:**
- Consome: `face_recognizer`, `data/known_faces/`, `data/events/`.
- Produz: Relatório completo de qualidade das fotos cadastradas, matriz de confusão cruzada e teste de assertividade com recortes reais.

- [ ] **Passo 1: Criar script com verificação de qualidade de cada foto de morador**
- [ ] **Passo 2: Calcular matriz de similaridade inter-moradores para checar isolamento**
- [ ] **Passo 3: Testar os recortes reais de `data/events/` e exibir ranking biométrico**
- [ ] **Passo 4: Commit no Git e instruções de execução no Nobara Linux**
