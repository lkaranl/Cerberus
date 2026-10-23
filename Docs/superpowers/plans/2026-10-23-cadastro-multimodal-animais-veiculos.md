# Cadastro Multimodal de Pessoas, Veículos e Animais no Dashboard — Plano de Implementação

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) ou superpowers:executing-plans para implementar este plano tarefa por tarefa. Os passos utilizam a sintaxe de caixas de seleção (`- [ ]`) para rastreamento.

**Goal:** Implementar o cadastro e auto-treinamento de pessoas, animais (pets) e veículos diretamente pelo dashboard do Cerberus, suportando tanto classificação a partir de detecções de câmeras (Auto-Enrollment) quanto upload manual direto de fotos.

**Architecture:** Estender a camada de API FastAPI em [app/api/routes.py](file:///Users/karan/Github/Cerberus/app/api/routes.py) para suportar persistência categorizada nas pastas `data/known_faces/`, `data/known_vehicles/` e `data/known_pets/`, integrando-se aos módulos [recognizer.py](file:///Users/karan/Github/Cerberus/app/detector/recognizer.py) e [visual_matcher.py](file:///Users/karan/Github/Cerberus/app/detector/visual_matcher.py). No frontend ([dashboard.html](file:///Users/karan/Github/Cerberus/app/static/dashboard.html)), criar abas para a esteira de candidatos, modal de classificação contextual e novo modal HUD para upload manual direto.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, SQLite (aiosqlite), Vanilla HTML5/CSS3/JavaScript, Torchvision / MobileNetV3 / InsightFace (integrações existentes).

## Global Constraints

- **Restrição de Ambiente (AGENTS.md)**: Proibição absoluta de execução de runtimes Python ou scripts shell neste Mac. Testes são escritos e inspecionados estaticamente; execução real pertence ao servidor Nobara Linux.
- **Idioma**: Todos os textos de interface, documentação e commits devem ser em português do Brasil.
- **Design HUD Tático**: Manter fidelidade visual aos estilos existentes em `dashboard.html` (cores HSL, glassmorphism, tags com acentos esmeralda/ciano/roxo).
- **Retrocompatibilidade**: Manter as rotas legadas `/api/faces/*` como aliases funcionais para não quebrar integrações existentes.

---

### Task 1: Backend — Endpoints de Candidatos e Inventário Consolidado

**Files:**
- Modify: `app/api/routes.py`
- Modify: `app/storage/database.py`
- Test: `tests/test_entity_registration.py`

**Interfaces:**
- Consumes: `db.get_recent_events(limit, is_known, entity_type)`
- Produces: 
  - `GET /api/candidates?entity_type={person|vehicle|pet}&limit=12`
  - `GET /api/entities/known`

- [x] **Passo 1: Escrever testes unitários em `tests/test_entity_registration.py` para os endpoints de candidatos e lista de conhecidos**

```python
import pytest
from app.api.routes import router

def test_api_candidates_structure():
    # Valida estrutura da rota /api/candidates
    routes = [r.path for r in router.routes]
    assert "/candidates" in routes or "/api/candidates" in routes
```

- [x] **Passo 2: Implementar suporte a filtro por `entity_type` em `database.py` se necessário**
Verificar se `get_recent_events` em [database.py](file:///Users/karan/Github/Cerberus/app/storage/database.py) filtra adequadamente por `entity_type` para `person`, `vehicle` e `pet`.

- [x] **Passo 3: Adicionar `GET /api/candidates` e `GET /api/entities/known` em `routes.py`**
Implementar o endpoint `/api/candidates` aceitando `entity_type: str = "person"`, e `/api/entities/known` consolidando `face_recognizer.known_gallery.keys()`, `visual_matcher.known_vehicles.keys()` e `visual_matcher.known_pets.keys()`.

- [x] **Passo 4: Commit das alterações**
```bash
git add app/api/routes.py app/storage/database.py tests/test_entity_registration.py
git commit -m "feat(api): adiciona endpoints de candidatos por tipo e inventario consolidado de entidades"
```

---

### Task 2: Backend — Endpoint Unificado de Classificação de Eventos (`/api/entities/classify-event`)

**Files:**
- Modify: `app/api/routes.py`
- Test: `tests/test_entity_registration.py`

**Interfaces:**
- Consumes: `db.get_event_by_id`, `db.update_event`, `face_recognizer`, `visual_matcher`
- Produces: `POST /api/entities/classify-event`

- [x] **Passo 1: Criar modelo Pydantic `ClassifyEntityRequest`**
```python
class ClassifyEntityRequest(BaseModel):
    event_id: str
    entity_type: str = "person" # "person", "vehicle", "pet"
    entity_name: str
    license_plate: Optional[str] = None
```

- [x] **Passo 2: Implementar o handler `classify_entity_event`**
- Se `entity_type == "person"`: salva em `data/known_faces/<Nome>/`, atualiza evento e recarrega `face_recognizer`.
- Se `entity_type == "vehicle"`: salva em `data/known_vehicles/<Nome>_[<PLACA>]/` (ou `<Nome>/` se sem placa), atualiza evento com `license_plate` e recarrega `visual_matcher.load_vehicles()`.
- Se `entity_type == "pet"`: salva em `data/known_pets/<Nome>/`, atualiza evento e recarrega `visual_matcher.load_pets()`.
- Manter `@router.post("/faces/classify-event")` chamando a mesma lógica para retrocompatibilidade.

- [x] **Passo 3: Commit das alterações**
```bash
git add app/api/routes.py tests/test_entity_registration.py
git commit -m "feat(api): implementa classificacao de eventos para pessoas, veiculos e pets"
```

---

### Task 3: Backend — Endpoint de Upload Direto com Fotos (`/api/entities/upload`)

**Files:**
- Modify: `app/api/routes.py`
- Test: `tests/test_entity_registration.py`

**Interfaces:**
- Produces: `POST /api/entities/upload` (multipart/form-data)

- [x] **Passo 1: Implementar o endpoint `upload_entity`**
Recebe `entity_type: str = Form(...)`, `entity_name: str = Form(...)`, `license_plate: Optional[str] = Form(None)` e `files: List[UploadFile] = File(...)`.
- Valida extensões (`.jpg`, `.jpeg`, `.png`, `.webp`).
- Cria pastas com segurança.
- Grava bytes das imagens.
- Recarrega galeria correspondente na memória/GPU.

- [x] **Passo 2: Commit das alterações**
```bash
git add app/api/routes.py tests/test_entity_registration.py
git commit -m "feat(api): implementa upload manual direto de fotos para pessoas, veiculos e pets"
```

---

### Task 4: Frontend — Estúdio de Treinamento e Auto-Enrollment com Abas

**Files:**
- Modify: `app/static/dashboard.html`

- [x] **Passo 1: Atualizar o HTML da seção `#face-trainer-section`**
- Adicionar abas de categoria: `[👤 Pessoas]` `[🚗 Veículos]` `[🐾 Pets]`.
- Adicionar botão `[➕ Novo Cadastro por Foto]`.

- [x] **Passo 2: Atualizar scripts JS para gerenciar a aba ativa de candidatos**
- Variável de estado `currentCandidateType = "person"`.
- Atualizar `loadTrainingCandidates()` para passar `entity_type=${currentCandidateType}`.
- Renderizar os cards conforme a categoria ativa com botões rápidos dos itens conhecidos correspondentes (`knownPersonsList`, `knownVehiclesList`, `knownPetsList`).

- [x] **Passo 3: Commit das alterações**
```bash
git add app/static/dashboard.html
git commit -m "feat(dashboard): adiciona abas por categoria na esteira de auto-enrollment"
```

---

### Task 5: Frontend — Modais de Classificação e Cadastro Manual

**Files:**
- Modify: `app/static/dashboard.html`

- [x] **Passo 1: Atualizar o modal existente `#modal-classify`**
- Adicionar campo condicional de placa para veículos.
- Rótulos dinâmicos de título e placeholder conforme a categoria.
- Integração da chamada para `/api/entities/classify-event`.

- [x] **Passo 2: Criar o novo modal `#modal-manual-upload`**
- Seletor de Categoria: Pessoa, Veículo, Pet.
- Campos de Nome e Placa (condicional).
- Área de Drag & Drop para upload de fotos com pré-visualização.
- Chamada para `/api/entities/upload` e recarga dinâmica do dashboard e conhecidos.

- [x] **Passo 3: Commit das alterações**
```bash
git add app/static/dashboard.html
git commit -m "feat(dashboard): cria modal de upload manual direto e adapta modal de classificacao"
```

---

### Task 6: Frontend — Tabela Forense e Modal de Evidência Completa

**Files:**
- Modify: `app/static/dashboard.html`

- [x] **Passo 1: Atualizar `renderEventsTable`**
- Exibir botão `Treinar` contextual para eventos não conhecidos de qualquer entidade (`person`, `vehicle`, `pet`).

- [x] **Passo 2: Atualizar `openForensicModal`**
- Atualizar o botão de cadastro no rodapé do modal forense com base no tipo de entidade (`person`, `vehicle`, `pet`).

- [x] **Passo 3: Commit das alterações**
```bash
git add app/static/dashboard.html
git commit -m "feat(dashboard): habilita treinamento e cadastro contextual no feed forense e modal"
```

---

### Task 7: Verificação Completa e Auditoria Final

**Files:**
- Inspect: `app/api/routes.py`
- Inspect: `app/storage/database.py`
- Inspect: `app/static/dashboard.html`
- Inspect: `tests/test_entity_registration.py`

- [x] **Passo 1: Verificação estática e de sintaxe do código Python e HTML/JS**
- [x] **Passo 2: Validação de cobertura dos testes unitários criados**
- [x] **Passo 3: Auditoria final de integridade e commit final**
