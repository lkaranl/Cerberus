# Especificação de Design: Redesign da Tela Principal — Command Center Tático

> **Data**: 2026-10-21  
> **Status**: Aprovado  
> **Objetivo**: Elevar a experiência do usuário (UX/UI) da tela principal do Cerberus para um padrão visual e operacional de **Centro de Comando Tático (Command Center)**, com glassmorphism premium, HUD tático flutuante em cada câmera, drawer lateral retrátil e timeline inteligente com filtros rápidos de eventos.

---

## 1. Visão Geral e Filosofia de Design

A tela principal do Cerberus é o coração operacional do sistema de vigilância. A nova interface adota uma estética *Cyber-Defense / High-Tech Glassmorphism* inspirada em centros de comando militar e estações espaciais:
- **Paleta de Cores**:
  - Fundo: Dark Theme profundo (`#070a12`, `#0b101c`, `#111827`);
  - Superfícies: Vidro translúcido com desfoque de fundo (`backdrop-filter: blur(20px)`, bordas sutis `rgba(255, 255, 255, 0.08)`);
  - Acentos Semânticos:
    - Ciano Elétrico (`#00f0ff`): Sistema, telemetria e HUD tático;
    - Verde Esmeralda (`#10b981`): Moradores conhecidos confirmados;
    - Carmim / Alerta (`#f43f5e`): Pessoas desconhecidas / invasores em área restrita;
    - Violeta Neon (`#8b5cf6`): Veículos identificados e leitura de placas (LPR);
    - Âmbar Solar (`#f59e0b`): Animais de estimação (Pets).
- **Tipografia**: `Outfit` para títulos e rótulos limpos; `JetBrains Mono` para dados técnicos, contadores de tempo, FPS e placas veiculares.

---

## 2. Componentes da Interface

### 2.1. Header Tático (Top Command Bar)
- **Marca**: Ícone tático de sentinela com radar ativo + título `CERBERUS` com badge `AI SENTINEL`.
- **Cápsulas de Telemetria (Hardware HUD)**:
  - Chip GPU: `NVIDIA GTX 1660 [NVDEC + TRT]` com VRAM alocada em tempo real;
  - Chip Câmeras: `3 Câmeras Online (2K 25 FPS)` com indicador de integridade de stream;
  - Relógio Tático: Horário oficial sincronizado (`HH:MM:SS`) com indicador `● AO VIVO`.
- **Controles Rápidos**:
  - Botão de alternância do Drawer Lateral (Expandir / Recolher para modo 100% tela cheia);
  - Link de navegação para o Dashboard Analítico de Estatísticas (`/dashboard`).

### 2.2. Barra de Navegação & Seleção de Layout
- Seletor de visualização em pílula de vidro:
  - Botão `Grid Mosaico (3 Câmeras)`;
  - Botões individuais para cada câmera com miniatura e indicador de status;
- Botão `Modo Foco / Cinema` (oculta cabeçalhos e foca exclusivamente nas câmeras).

### 2.3. Grid de Câmeras 2K com HUD Tático Flutuante
- Cada card de câmera é envelopado em um container estilizado com bordas reflexivas dinâmicas:
  - **Borda Reativa Dinâmica**:
    - Normal: Borda sutil escura (`rgba(255, 255, 255, 0.08)`);
    - Morador Detectado: Borda suavemente pulsante em **Verde Esmeralda**;
    - Desconhecido Detectado: Borda em pulso de alerta em **Carmim / Vermelho**;
    - Veículo Detectado: Borda em **Violeta Neon**.
  - **HUD Flutuante Superior**:
    - Nome da Câmera (ex: `Câmera do Portão [cam-0]`);
    - Tag técnica de resolução e taxa: `2304x1296 | NVDEC | 25 FPS`;
  - **HUD Flutuante Inferior (Ações Táticas no Hover / Visíveis)**:
    - Botão **Snapshot Instantâneo 2K**: captura o quadro nativo da câmera e baixa em JPEG de alta qualidade com um clique;
    - Botão **Foco Individual (Fullscreen)**: expande a câmera com animação fluida para tela cheia;
    - Badge de Detecção Ativa (mostra a contagem de alvos rastreados na cena em tempo real).

### 2.4. Drawer Lateral Retrátil & Timeline Inteligente de Eventos
- O painel lateral tem suporte a **Recolher / Expandir** com transição suave, permitindo que a tela seja usada como monitor de segurança dedicado (100% vídeo) ou central de auditoria.
- **Filtros Rápidos em Chips**:
  - `Todos` (contador total);
  - `Moradores` (ícone verde de escudo);
  - `Desconhecidos` (ícone vermelho de alerta);
  - `Veículos` (ícone de carro/placa);
  - `Pets` (ícone de pata).
- **Cards de Eventos da Timeline**:
  - Foto do recorte em miniatura nítida com borda de qualidade;
  - Nome do Morador destacado com crachá `[CONFIRMADO]` e porcentagem biométrica (ex: `Karan (79%)`);
  - Tag estilizada de placa Mercosul quando aplicável (`[BRA2E19]`);
  - Horário relativo inteligente (`"há 30 seg"`, `"há 3 min"`) + timestamp absoluto no hover;
  - Botão "Limpar Tela" acessível e contador dinâmico com micro-animação a cada novo evento via WebSocket.

### 2.5. Modal Tático de Inspeção de Evidências
- Visualização lado a lado da foto de evidência recortada (com detalhes ampliados de super-resolução) e o quadro geral 2K anotado.
- Metadados técnicos completos: ID do Rastro, Câmera, Confiança da Face, Tipo de Movimento (Estacionado / Em Trânsito).
- Botões de ação rápida para download das evidências originais.

---

## 3. Arquitetura de Implementação
- **HTML**: Estruturação semântica e limpa em [`app/static/index.html`](file:///Users/karan/Github/Cerberus/app/static/index.html).
- **CSS**: Vanilla CSS moderno em [`app/static/css/style.css`](file:///Users/karan/Github/Cerberus/app/static/css/style.css) com variáveis CSS (Design Tokens), Glassmorphism, CSS Grid responsivo e animações por hardware aceleradas por GPU (`transform`, `opacity`).
- **JavaScript**: Lógica reativa em [`app/static/js/app.js`](file:///Users/karan/Github/Cerberus/app/static/js/app.js) para gerenciar:
  - Estado de recolhimento do Drawer;
  - Filtragem instantânea dos eventos na timeline;
  - Captura e download de snapshots em 2K;
  - Bordas dinâmicas de estado da câmera orientadas por WebSockets.
