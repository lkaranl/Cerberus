# Redesign da Tela Principal — Command Center Tático: Implementation Plan

> **Data**: 2026-10-21  
> **Spec**: [Especificação de Design](file:///Users/karan/Github/Cerberus/Docs/superpowers/specs/2026-10-21-redesign-tela-principal-command-center-design.md)  
> **Objetivo**: Implementar a nova interface visual da tela principal no padrão Command Center com glassmorphism, HUD tático flutuante nas câmeras 2K, drawer lateral retrátil e timeline inteligente com chips de filtragem rápida de eventos.

---

## Estrutura de Arquivos Envolvidos
- Modificar: `app/static/index.html` (Estruturação do layout Command Center, Header tático, HUDs e Drawer).
- Modificar: `app/static/css/style.css` (Design tokens, glassmorphism, HUD tático flutuante, bordas reativas animadas, timeline e drawer retrátil).
- Modificar: `app/static/js/app.js` (Gerenciamento do estado do drawer, filtros rápidos da timeline, captura de snapshots 2K, foco em tela cheia e bordas dinâmicas).

---

## Tarefas de Implementação

### Tarefa 1: Estruturação HTML Semântica do Command Center (`app/static/index.html`)
**Arquivos:**
- Modificar: `app/static/index.html`

**Instruções Passo a Passo:**
1. Atualizar o Header da aplicação com identidade `CERBERUS AI SENTINEL`.
2. Adicionar cápsulas de telemetria de hardware (GPU GTX 1660 VRAM, status das 3 câmeras online 2K, relógio tático sincronizado com pulso `AO VIVO`).
3. Adicionar botão de controle de expansão/recolhimento do Drawer Lateral no header (`btn-toggle-drawer`).
4. Reestruturar a barra de controles de visualização (Pílulas de vidro com botões de Mosaico Grid e Câmeras Individuais).
5. Estruturar o container das câmeras com suporte aos overlays HUD táticos (superior com resolução e FPS, inferior com ações de snapshot e fullscreen).
6. Estruturar o Drawer Lateral com barra de chips de filtro rápido (`Todos`, `Moradores`, `Desconhecidos`, `Veículos`, `Pets`) e botão estilizado `Limpar Tela`.
7. Atualizar o modal tático de evidência para inspeção detalhada lado a lado.

---

### Tarefa 2: Design System Tático, Glassmorphism e HUD em CSS (`app/static/css/style.css`)
**Arquivos:**
- Modificar: `app/static/css/style.css`

**Instruções Passo a Passo:**
1. Expandir os Design Tokens (`:root`) com cores semânticas de alta tecnologia:
   - `--accent-cyan: #00f0ff` (HUD e telemetria);
   - `--accent-emerald: #10b981` (Moradores confirmados);
   - `--accent-rose: #f43f5e` (Desconhecidos / invasores);
   - `--accent-violet: #8b5cf6` (Veículos e placas);
   - `--accent-amber: #f59e0b` (Pets).
2. Criar classes de superfícies glassmorphism ultra-sleek com `backdrop-filter: blur(20px)`.
3. Estilizar as cápsulas de hardware no Header com brilhos de led sutis (`glow-dots`).
4. Estilizar o HUD Tático Flutuante sobre o vídeo de cada câmera:
   - Tag superior translúcida com informações de resolução 2K e FPS;
   - Barra inferior de ações táticas (botões de Snapshot e Fullscreen com micro-animações no hover);
   - Efeitos de borda reativa animada (glow pulsante verde para morador, vermelho para desconhecido e violeta para veículo).
5. Estilizar o Drawer Retrátil Lateral:
   - Transição fluida de largura e visibilidade (`transform`, `opacity`, `transition: 0.3s cubic-bezier`);
   - Modo de recolhimento que expande o Grid de câmeras para 100% da tela útil;
   - Chips de filtro com badges numéricos e estados ativos destacados.
6. Estilizar os cards de evento da timeline com miniaturas nítidas, tags Mercosul para placas e crachás biométricos.
7. Estilizar o modal tático de evidências em tema escuro com visualização lado a lado.

---

### Tarefa 3: Reatividade, Lógica de Filtros e Ferramentas Táticas em JS (`app/static/js/app.js`)
**Arquivos:**
- Modificar: `app/static/js/app.js`

**Instruções Passo a Passo:**
1. Adicionar estado para o drawer lateral (`state.isDrawerCollapsed: false`) e implementar alternância suave com persistência no `localStorage`.
2. Implementar função de **Snapshot 2K Instantâneo**:
   - Cria um elemento `<canvas>` invisível, desenha o frame de vídeo atual da câmera selecionada e dispara o download automático do arquivo nomeado com a data, câmera e ID do rastro.
3. Implementar botão de **Tela Cheia Tática (Fullscreen)** individual por câmera.
4. Implementar sistema de **Filtros Rápidos em Chips na Timeline**:
   - Adicionar listeners nos chips (`all`, `residents`, `unknowns`, `vehicles`, `pets`);
   - Filtrar instantaneamente os cards de evento na DOM sem recarregar a página;
   - Atualizar os contadores de cada categoria dinamicamente.
5. Implementar reatividade visual de borda da câmera quando um evento for recebido via WebSocket:
   - Adiciona classe temporária `camera-card--resident` (verde), `camera-card--alert` (vermelho) ou `camera-card--vehicle` (violeta) por 5 segundos com efeito de pulso suave.
6. Ajustar o modal de inspeção para formatar placas com badge estilizado e exibir métricas de reconhecimento facial calibradas.

---

### Tarefa 4: Validação Visual e Homologação
**Instruções Passo a Passo:**
1. Revisar integridade da sintaxe HTML, CSS e JavaScript.
2. Commitar todas as alterações com mensagens em Português do Brasil.
3. Subir para o GitHub (`git push origin main`).
4. Fornecer instruções para o usuário testar a nova interface diretamente no navegador acessando o servidor Nobara.
