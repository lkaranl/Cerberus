# Especificação de Design: Cadastro Multimodal de Pessoas, Veículos e Animais no Dashboard

- **Data**: 2026-10-23
- **Status**: Aprovado
- **Autor**: Antigravity & Karan
- **Objetivo**: Permitir o cadastro e auto-treinamento tanto de animais (pets) quanto de veículos (carros/motos) diretamente pelo dashboard do Cerberus, combinando classificação rápida a partir de eventos capturados pelas câmeras (Auto-Enrollment) com um formulário de upload manual de fotos externas.

---

## 1. Contexto e Motivação

Atualmente, o dashboard do Cerberus possui um esteira de auto-enrollment e um modal de classificação biométrica voltados exclusivamente para **moradores humanos** (`entity_type="person"` e diretório `data/known_faces/`). 

No entanto, o Cerberus já conta com uma infraestrutura pronta e de alto desempenho no módulo [visual_matcher.py](file:///Users/karan/Github/Cerberus/app/detector/visual_matcher.py) (MobileNetV3 + LPR) para identificar:
- **Veículos**: cadastrados em `data/known_vehicles/<Nome>_[<PLACA>]/` ou `data/known_vehicles/<Nome>/`.
- **Pets (cães e gatos)**: cadastrados em `data/known_pets/<Nome>/`.

A presente funcionalidade integra essa capacidade visual diretamente ao dashboard e à API, permitindo que o usuário:
1. Classifique eventos de veículos e pets não identificados capturados pelas câmeras através da esteira de treinamento superior, da tabela de auditoria e do modal forense.
2. Cadastre novos moradores, veículos e pets a qualquer momento fazendo upload direto de fotos do dispositivo (computador/celular).
3. Atualize os modelos de IA e embeddings em tempo real na GPU/RAM sem necessidade de reiniciar o servidor Linux de produção.

---

## 2. Arquitetura e Endpoints da API (Backend)

Todos os endpoints serão implementados em [app/api/routes.py](file:///Users/karan/Github/Cerberus/app/api/routes.py) com persistência em [app/storage/database.py](file:///Users/karan/Github/Cerberus/app/storage/database.py).

### 2.1. Consulta de Candidatos de Treinamento
- **Endpoint**: `GET /api/candidates?entity_type={person|vehicle|pet}&limit=12`
  - Filtra eventos no SQLite onde `is_known = False` e `entity_type` corresponde ao solicitado.
  - Para `vehicle`, retorna eventos de carros/motos.
  - Para `pet`, retorna eventos de cães/gatos.
  - Para `person`, retorna eventos de pessoas (mantendo o endpoint `/api/faces/candidates` como alias para retrocompatibilidade).

### 2.2. Classificação de Eventos Existentes (Auto-Enrollment)
- **Endpoint**: `POST /api/entities/classify-event`
  - **Payload**:
    ```json
    {
      "event_id": "string",
      "entity_type": "person" | "vehicle" | "pet",
      "entity_name": "string",
      "license_plate": "string" // Opcional, para veículos
    }
    ```
  - **Comportamento por Entidade**:
    1. **`person`**:
       - Copia o recorte de evidência (`crop_path`) para `data/known_faces/<Nome_Pessoa>/cctv_<id>_<timestamp>.jpg`.
       - Atualiza o evento no banco: `person_name = clean_name, is_known = 1`.
       - Dispara `face_recognizer.load_known_faces()`.
    2. **`vehicle`**:
       - Higieniza a placa (se fornecida): `placa = placa.upper().replace("-", "").strip()`.
       - Define o diretório: se houver placa, `data/known_vehicles/<Nome_Veiculo>_[<PLACA>]/`, caso contrário `data/known_vehicles/<Nome_Veiculo>/`.
       - Copia o recorte para o diretório de destino: `cctv_veh_<id>_<timestamp>.jpg`.
       - Atualiza o evento no banco: `person_name = clean_name, license_plate = placa (se informada), is_known = 1`.
       - Dispara `visual_matcher.load_vehicles()`.
    3. **`pet`**:
       - Copia o recorte para `data/known_pets/<Nome_Pet>/cctv_pet_<id>_<timestamp>.jpg`.
       - Atualiza o evento no banco: `person_name = clean_name, is_known = 1`.
       - Dispara `visual_matcher.load_pets()`.
  - Retorna confirmação com status, total de entidades recarregadas e caminho salvo.
  - `/api/faces/classify-event` é mantido como alias compatível que redireciona para esta lógica com `entity_type="person"`.

### 2.3. Cadastro Manual Direto com Upload de Fotos
- **Endpoint**: `POST /api/entities/upload` (Formato `multipart/form-data`)
  - **Parâmetros**:
    - `entity_type`: `"person"` | `"vehicle"` | `"pet"`
    - `entity_name`: string (nome limpo)
    - `license_plate`: string opcional (para veículos)
    - `files`: Lista de `UploadFile` (mínimo 1 arquivo de imagem válido)
  - **Validação**:
    - Extensões aceitas: `.jpg`, `.jpeg`, `.png`, `.webp`.
    - Nome obrigatório e não vazio (rejeita 'desconhecido' ou strings vazias).
  - **Persistência**:
    - Cria a pasta correspondente em `data/known_faces/`, `data/known_vehicles/` ou `data/known_pets/`.
    - Salva cada imagem enviada com nomenclatura segura: `upload_<timestamp>_<index>.<ext>`.
  - **Recarga dos Modelos**:
    - Executa recarga a quente instantânea da respectiva galeria biométrica/visual.

### 2.4. Inventário Consolidado de Conhecidos
- **Endpoint**: `GET /api/entities/known`
  - Retorna:
    ```json
    {
      "persons": ["Karan", "Ingrid"],
      "vehicles": ["Meu Carro [NEC4J83]", "Vizinho frente"],
      "pets": ["Thor", "Felix"]
    }
    ```
  - Permite ao frontend popular chips de autocompletar dinamicamente para as 3 categorias.

---

## 3. Interface do Usuário (Frontend / Dashboard)

Arquivo alvo: [app/static/dashboard.html](file:///Users/karan/Github/Cerberus/app/static/dashboard.html).

### 3.1. Painel Superior de Treinamento
1. **Título e Controles**:
   - Título: `Estúdio de Treinamento & Cadastros (Auto-Enrollment)`.
   - Abas seletoras:
     - `[👤 Pessoas]`
     - `[🚗 Veículos]`
     - `[🐾 Pets]`
   - Botão de destaque `[➕ Novo Cadastro por Foto]` que aciona o modal de upload direto.
2. **Esteira de Candidatos**:
   - Ao trocar de aba, busca os eventos não identificados daquela categoria (`/api/candidates?entity_type=...`).
   - Cards com miniatura, câmera, horário e botões de atalho:
     - Para `person`: chips dos moradores conhecidos + `+ Outro`.
     - Para `vehicle`: chips dos veículos conhecidos + `+ Outro`.
     - Para `pet`: chips dos pets conhecidos + `+ Outro`.

### 3.2. Modal Contextual de Classificação (`#modal-classify`)
- Título adaptado dinamicamente:
  - `🏷️ Classificar Morador`
  - `🏷️ Classificar Veículo`
  - `🏷️ Classificar Pet`
- Exibe o recorte do evento.
- Renderiza chips da categoria ativa.
- Campos de input:
  - Se veículo: Campo "Nome / Modelo" + Campo "Placa Mercosul" (já preenchida se detectada via LPR).
  - Se pessoa: Campo "Nome do Morador".
  - Se pet: Campo "Nome do Pet".
- Botão: `Cadastrar e Ensinar IA`.

### 3.3. Novo Modal de Cadastro Manual (`#modal-manual-upload`)
- Modal HUD tático moderno com:
  - Seletor de Categoria (Pills): `👤 Morador`, `🚗 Veículo`, `🐾 Pet`.
  - Campo de Nome da Entidade.
  - Campo de Placa Mercosul (exibido apenas se categoria for `Veículo`).
  - Zona Drag & Drop para seleção de fotos (múltiplas) com pré-visualização das miniaturas selecionadas.
  - Botão `Salvar e Sincronizar na GPU`.

### 3.4. Auditoria Forense e Modal
- Na tabela de auditoria:
  - Botão `Treinar` disponível para todos os eventos não identificados, adaptando o ícone/texto:
    - `👤 Treinar` para pessoas.
    - `🚗 Treinar` para veículos.
    - `🐾 Treinar` para pets.
- No Modal Forense (`openForensicModal`):
  - Botão no rodapé adapta dinamicamente o rótulo:
    - `🏷️ Cadastrar Face como Morador`
    - `🏷️ Cadastrar Veículo`
    - `🏷️ Cadastrar Pet`

---

## 4. Validações e Tratamento de Erros

1. **Higienização de Nomes e Pastas**:
   - Nomes não podem conter caracteres proibidos em sistemas de arquivos (`/`, `\`, `:`, `*`, `?`, `"`, `<`, `>`, `|`).
   - Tratamento seguro de espaços e acentos.
2. **Validação de Placa**:
   - Conversão para letras maiúsculas, remoção de traços e espaços (ex: `BRA2E19`, `NEC4J83`).
3. **Imagens Corrompidas**:
   - Validação da integridade da imagem enviada antes de salvar no disco.
4. **Retrocompatibilidade**:
   - Nenhuma chamada antiga a `/api/faces/*` quebrará. Todas as rotas preexistentes continuam funcionais como wrappers da nova API unificada.

---

## 5. Estratégia de Testes

1. **Testes Unitários / Integração da API**:
   - Criação de suíte de testes em `tests/test_entity_registration.py` para validar:
     - Classificação de eventos para `person`, `vehicle` e `pet`.
     - Upload multipart de fotos para cada categoria.
     - Criação correta das pastas em `data/known_*/`.
     - Consulta de candidatos por categoria com `/api/candidates`.
     - Listagem de conhecidos em `/api/entities/known`.
2. **Validação Visual do Dashboard**:
   - Inspeção da sintaxe HTML/JS no `dashboard.html`.
