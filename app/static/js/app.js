/**
 * Sentinela Vision - Frontend Application
 * Suporte a WebSockets de baixa latência, renderização Canvas de alta performance,
 * alternância de visualização Grid / Single View e feed de alertas em tempo real.
 */

document.addEventListener("DOMContentLoaded", () => {
  const state = {
    cameras: [],
    currentView: "grid", // "grid" ou camera_id ("cam-0", etc.)
    wsConnections: {}, // camera_id -> WebSocket
    eventsWs: null,
    eventsCount: 0,
    statsInterval: null
  };

  // Elementos do DOM
  const dom = {
    clock: document.getElementById("live-clock"),
    hardwareLabel: document.getElementById("hardware-label"),
    camsActiveLabel: document.getElementById("cams-active-label"),
    videoContainer: document.getElementById("video-container"),
    singleSelectors: document.getElementById("single-selectors"),
    btnGrid: document.getElementById("btn-grid"),
    eventsList: document.getElementById("events-list"),
    eventsEmpty: document.getElementById("events-empty"),
    eventsBadge: document.getElementById("events-count-badge"),
    btnClearScreen: document.getElementById("btn-clear-screen"),
    modal: document.getElementById("event-modal"),
    modalClose: document.getElementById("modal-close"),
    modalCameraTitle: document.getElementById("modal-camera-title"),
    modalTime: document.getElementById("modal-time"),
    modalCropImg: document.getElementById("modal-crop-img"),
    modalFullImg: document.getElementById("modal-full-img"),
    modalCropLoader: document.getElementById("modal-crop-loader"),
    modalFullLoader: document.getElementById("modal-full-loader"),
    modalFullRawLink: document.getElementById("modal-full-raw-link"),
    modalTrackId: document.getElementById("modal-track-id"),
    modalConfidence: document.getElementById("modal-confidence"),
    modalEventId: document.getElementById("modal-event-id")
  };

  // --- Relógio ao Vivo ---
  function startClock() {
    function update() {
      const now = new Date();
      dom.clock.textContent = now.toLocaleTimeString("pt-BR", { hour12: false });
    }
    update();
    setInterval(update, 1000);
  }

  // --- Inicialização ---
  async function init() {
    startClock();
    setupModalEvents();
    setupViewSwitching();
    setupClearScreen();

    await loadSystemStats();
    await loadCameras();
    await loadRecentEvents();

    connectEventsWebSocket();

    // Atualiza status e estatísticas a cada 8 segundos
    state.statsInterval = setInterval(() => {
      loadSystemStats();
      refreshCameraTelemetry();
    }, 8000);
  }

  // --- Carregar Estatísticas do Sistema ---
  async function loadSystemStats() {
    try {
      const res = await fetch("/api/stats");
      if (!res.ok) return;
      const data = await res.json();

      if (data.hardware) {
        if (data.hardware.yolo_tensorrt && data.hardware.facenet_tensorrt) {
          dom.hardwareLabel.textContent = `NVIDIA GTX 1660 [YOLO + FaceNet TRT ⚡] (${data.hardware.allocated_mb} MB)`;
        } else if (data.hardware.tensorrt_active) {
          dom.hardwareLabel.textContent = `NVIDIA GTX 1660 [TensorRT FP16 ⚡] (${data.hardware.allocated_mb} MB)`;
        } else if (data.hardware.type && data.hardware.type.includes("CUDA")) {
          dom.hardwareLabel.textContent = `NVIDIA GTX 1660 (${data.hardware.allocated_mb} MB VRAM)`;
        } else if (data.hardware.type && data.hardware.type.includes("Metal")) {
          dom.hardwareLabel.textContent = "Apple Silicon (MPS)";
        } else {
          dom.hardwareLabel.textContent = data.hardware.device.toUpperCase();
        }
      }

      dom.camsActiveLabel.textContent = `${data.cameras_active || 0} ativas`;
    } catch (err) {
      console.warn("Falha ao obter stats do sistema:", err);
    }
  }

  // --- Carregar Câmeras e Configurar Grid ---
  async function loadCameras() {
    try {
      const res = await fetch("/api/cameras");
      if (!res.ok) return;
      state.cameras = await res.json();

      renderViewButtons();
      renderCameraCards();
      connectAllCameraStreams();
    } catch (err) {
      console.error("Erro ao carregar câmeras:", err);
    }
  }

  // --- Renderizar Botões de Seleção de Câmeras ---
  function renderViewButtons() {
    dom.singleSelectors.innerHTML = "";

    state.cameras.forEach(cam => {
      const btn = document.createElement("button");
      btn.className = "view-btn";
      btn.dataset.view = cam.id;
      btn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M23 7l-7 5 7 5V7z" />
          <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
        </svg>
        ${cam.name}
      `;
      btn.addEventListener("click", () => switchView(cam.id));
      dom.singleSelectors.appendChild(btn);
    });
  }

  // --- Configurar Alternância de Visualização ---
  function setupViewSwitching() {
    dom.btnGrid.addEventListener("click", () => switchView("grid"));
  }

  // --- Limpar Alertas da Tela (Mantém no Banco de Dados) ---
  function setupClearScreen() {
    if (dom.btnClearScreen) {
      dom.btnClearScreen.addEventListener("click", () => {
        dom.eventsList.innerHTML = `
          <div class="events-empty-state" id="events-empty">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>
              <circle cx="9" cy="7" r="4"/>
              <line x1="19" y1="8" x2="19" y2="14"/>
              <line x1="22" y1="11" x2="16" y2="11"/>
            </svg>
            <p>Tela limpa.</p>
            <span>Novas detecções em tempo real continuarão aparecendo aqui.</span>
          </div>
        `;
        state.eventsCount = 0;
        dom.eventsBadge.textContent = "0";
      });
    }
  }

  function switchView(viewId) {
    state.currentView = viewId;

    // Atualiza classes ativas nos botões
    document.querySelectorAll(".view-btn").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.view === viewId);
    });

    if (viewId === "grid") {
      dom.videoContainer.className = "video-grid-container mode-grid";
      document.querySelectorAll(".camera-card").forEach(card => {
        card.style.display = "flex";
      });
    } else {
      dom.videoContainer.className = "video-grid-container mode-single";
      document.querySelectorAll(".camera-card").forEach(card => {
        card.style.display = (card.dataset.camId === viewId) ? "flex" : "none";
      });
    }
  }

  // --- Renderizar Cards de Câmeras ---
  function renderCameraCards() {
    dom.videoContainer.innerHTML = "";

    state.cameras.forEach(cam => {
      const card = document.createElement("div");
      card.className = "camera-card";
      card.dataset.camId = cam.id;

      card.innerHTML = `
        <div class="camera-header">
          <div class="camera-title-wrap">
            <span class="camera-name">${cam.name}</span>
            ${cam.priority ? '<span class="camera-priority-badge">Prioritária</span>' : ''}
          </div>
          <div class="camera-badges">
            <span class="cam-status-pill ${cam.connected ? 'online' : 'offline'}" id="status-pill-${cam.id}">
              <span class="chip-dot ${cam.connected ? 'glow-green' : ''}"></span>
              ${cam.connected ? 'ONLINE' : 'CONECTANDO'}
            </span>
            <span class="cam-fps-pill" id="fps-pill-${cam.id}">0.0 FPS</span>
          </div>
        </div>

        <div class="video-viewport">
          <div class="video-placeholder" id="placeholder-${cam.id}">
            <div class="spinner"></div>
            <span>Conectando ao stream RTSP...</span>
          </div>
          <canvas class="video-canvas" id="canvas-${cam.id}"></canvas>

          <div class="video-overlay-bottom">
            <div class="cam-overlay-tag">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <polyline points="12 6 12 12 16 14"/>
              </svg>
              ${cam.id}
            </div>
            <div class="cam-overlay-tag detection-badge" id="det-tag-${cam.id}" style="display: none;">
              ● Detectado
            </div>
          </div>
        </div>
      `;

      dom.videoContainer.appendChild(card);
    });

    dom.videoContainer.className = "video-grid-container mode-grid";
  }

  // --- Conectar Stream de Câmera via WebSocket Binário ---
  // --- Conectar Stream de Câmera via WebSocket Binário com Pacing Adaptativo ---
  function connectCameraStream(camId) {
    if (state.wsConnections[camId]) {
      try { 
        state.wsConnections[camId].onclose = null;
        state.wsConnections[camId].close(); 
      } catch (e) {}
    }

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/stream/${camId}`;
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "blob";

    const canvas = document.getElementById(`canvas-${camId}`);
    const placeholder = document.getElementById(`placeholder-${camId}`);
    const ctx = canvas ? canvas.getContext("2d") : null;

    let frameCount = 0;
    let fpsTimer = performance.now();
    let firstFrame = true;

    ws.onopen = () => {
      // Solicita frame inicial ao abrir a conexão
      if (ws.readyState === WebSocket.OPEN) {
        ws.send("next");
      }
    };

    ws.onmessage = async (event) => {
      if (!ctx || !(event.data instanceof Blob)) return;

      // Se a aba estiver em segundo plano, não processa nem solicita novos frames
      if (document.hidden) {
        return;
      }

      try {
        const imageBitmap = await createImageBitmap(event.data);
        if (firstFrame) {
          if (placeholder) placeholder.classList.add("hidden");
          canvas.width = imageBitmap.width;
          canvas.height = imageBitmap.height;
          firstFrame = false;
        }

        ctx.drawImage(imageBitmap, 0, 0, canvas.width, canvas.height);
        imageBitmap.close();

        // Sinaliza ao servidor que este frame foi desenhado e solicita o próximo frame em tempo real
        if (ws.readyState === WebSocket.OPEN && !document.hidden) {
          ws.send("next");
        }

        frameCount++;
        const now = performance.now();
        if (now - fpsTimer >= 1000) {
          const fps = (frameCount / ((now - fpsTimer) / 1000)).toFixed(1);
          const fpsEl = document.getElementById(`fps-pill-${camId}`);
          if (fpsEl) fpsEl.textContent = `${fps} FPS`;

          const statusEl = document.getElementById(`status-pill-${camId}`);
          if (statusEl && !statusEl.classList.contains("online")) {
            statusEl.className = "cam-status-pill online";
            statusEl.innerHTML = `<span class="chip-dot glow-green"></span> ONLINE`;
          }

          frameCount = 0;
          fpsTimer = now;
        }
      } catch (err) {
        console.warn(`Erro ao renderizar frame de ${camId}:`, err);
        if (ws.readyState === WebSocket.OPEN && !document.hidden) {
          ws.send("next");
        }
      }
    };

    ws.onclose = () => {
      // Reconecta apenas se a aba estiver ativa em primeiro plano
      if (!document.hidden && state.cameras.some(c => c.id === camId)) {
        const statusEl = document.getElementById(`status-pill-${camId}`);
        if (statusEl) {
          statusEl.className = "cam-status-pill offline";
          statusEl.innerHTML = `<span class="chip-dot"></span> RECONECTANDO`;
        }
        setTimeout(() => {
          if (!document.hidden) connectCameraStream(camId);
        }, 2000);
      }
    };

    state.wsConnections[camId] = ws;
  }

  function disconnectAllCameraStreams() {
    Object.keys(state.wsConnections).forEach(camId => {
      try {
        const ws = state.wsConnections[camId];
        if (ws) {
          ws.onclose = null; // Evita loop de reconexão enquanto estiver oculta
          ws.close();
        }
      } catch (e) {}
    });
    state.wsConnections = {};
  }

  function connectAllCameraStreams() {
    state.cameras.forEach(cam => {
      connectCameraStream(cam.id);
    });
  }

  // --- Sincronização Estrita ao Alternar Abas (Zero Latência) ---
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      // Usuário retornou à página: restabelece conexões para obter o frame do segundo atual
      connectAllCameraStreams();
      loadSystemStats();
      refreshCameraTelemetry();
    } else {
      // Usuário saiu da aba: suspende streams pesados de vídeo para evitar buffering acumulado
      disconnectAllCameraStreams();
    }
  });

  // --- Atualizar Telemetria das Câmeras ---
  async function refreshCameraTelemetry() {
    try {
      const res = await fetch("/api/cameras");
      if (!res.ok) return;
      const cams = await res.json();

      cams.forEach(cam => {
        const statusEl = document.getElementById(`status-pill-${cam.id}`);
        if (statusEl && !cam.connected) {
          statusEl.className = "cam-status-pill offline";
          statusEl.innerHTML = `<span class="chip-dot"></span> OFFLINE`;
        }
      });
    } catch (e) {}
  }

  // --- Carregar Histórico Inicial de Eventos ---
  async function loadRecentEvents() {
    try {
      const res = await fetch("/api/events?limit=30");
      if (!res.ok) return;
      const data = await res.json();

      if (data.events && data.events.length > 0) {
        if (dom.eventsEmpty) dom.eventsEmpty.remove();
        state.eventsCount = data.total || data.events.length;
        dom.eventsBadge.textContent = state.eventsCount;

        data.events.forEach(evt => {
          renderEventCard(evt, false);
        });
      }
    } catch (err) {
      console.warn("Falha ao carregar eventos:", err);
    }
  }

  // --- Conectar Canal de Eventos em Tempo Real ---
  function connectEventsWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/events`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === "PERSON_DETECTED" && payload.data) {
          state.eventsCount++;
          dom.eventsBadge.textContent = state.eventsCount;

          if (dom.eventsEmpty && dom.eventsEmpty.parentNode) {
            dom.eventsEmpty.remove();
          }

          renderEventCard(payload.data, true);
          flashDetectionBadge(payload.data.camera_id);
        }
      } catch (err) {
        console.error("Erro ao processar mensagem de evento:", err);
      }
    };

    ws.onclose = () => {
      setTimeout(connectEventsWebSocket, 3000);
    };

    state.eventsWs = ws;
  }

  function flashDetectionBadge(camId) {
    const tag = document.getElementById(`det-tag-${camId}`);
    if (tag) {
      tag.style.display = "flex";
      setTimeout(() => {
        tag.style.display = "none";
      }, 3500);
    }
  }

  // --- Renderizar Card de Alerta na Sidebar ---
  function renderEventCard(evt, isNew = false) {
    const card = document.createElement("div");
    card.className = "event-card";

    const timeObj = new Date(evt.timestamp);
    const timeFormatted = isNaN(timeObj.getTime())
      ? evt.timestamp
      : timeObj.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });

    const cropUrl = `/api/events/${evt.event_id || evt.id}/crop`;
    const confPct = Math.round((evt.confidence || 0) * 100);

        let typeIcon = "👤";
        let badgeStyle = "background: var(--accent-green-dim); color: var(--accent-green); border: 1px solid rgba(0, 230, 118, 0.4);";

        if (evt.entity_type === "vehicle") {
          typeIcon = "🚗";
          if (evt.is_known) badgeStyle = "background: rgba(0, 240, 255, 0.15); color: var(--accent-cyan); border: 1px solid rgba(0, 240, 255, 0.4);";
        } else if (evt.entity_type === "pet") {
          typeIcon = "🐾";
          if (evt.is_known) badgeStyle = "background: rgba(224, 64, 251, 0.15); color: #e040fb; border: 1px solid rgba(224, 64, 251, 0.4);";
        }

        if (!evt.is_known) {
          badgeStyle = "background: rgba(255, 171, 0, 0.15); color: var(--accent-amber); border: 1px solid rgba(255, 171, 0, 0.4);";
        }

        const plateSnippet = evt.license_plate ? ` [${evt.license_plate}]` : '';

        card.innerHTML = `
          <div class="event-thumb-wrap">
            <img class="event-thumb" src="${cropUrl}" alt="Crop" loading="lazy" onerror="this.src='/static/img/placeholder.png'">
          </div>
          <div class="event-info">
            <div class="event-cam-row">
              <span class="event-cam-name">${evt.camera_name || evt.camera_id}</span>
              <span class="event-time">${timeFormatted}</span>
            </div>
            <div class="event-details-row">
              <span class="event-id-badge" style="${badgeStyle}">
                ${typeIcon} ${evt.person_name}${plateSnippet}
              </span>
              <span class="event-confidence">${confPct}%</span>
            </div>
          </div>
        `;

    // Pré-carrega imagem completa em background ao passar o mouse (hover)
    card.addEventListener("mouseenter", () => {
      prefetchEventImages(evt.event_id || evt.id);
    }, { once: true });

    card.addEventListener("click", () => openModal(evt));

    if (isNew) {
      dom.eventsList.prepend(card);
      // Mantém o DOM do cliente leve removendo elementos antigos que saem da tela
      while (dom.eventsList.children.length > 40) {
        dom.eventsList.removeChild(dom.eventsList.lastChild);
      }
      // Pré-carrega imediatamente o alerta mais recente para abertura instantânea
      prefetchEventImages(evt.event_id || evt.id);
    } else {
      dom.eventsList.appendChild(card);
    }
  }

  // --- Prefetch Inteligente de Imagens ---
  const prefetchedEvents = new Set();
  function prefetchEventImages(eventId) {
    if (!eventId || prefetchedEvents.has(eventId)) return;
    prefetchedEvents.add(eventId);

    // Cria requisição discreta em background para aquecer o cache HTTP do navegador
    const preImg = new Image();
    preImg.src = `/api/events/${eventId}/full`;
  }

  // --- Modal de Evidência ---
  function setupModalEvents() {
    dom.modalClose.addEventListener("click", closeModal);
    dom.modal.addEventListener("click", (e) => {
      if (e.target === dom.modal) closeModal();
    });
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeModal();
    });
  }

  function openModal(evt) {
    const id = evt.event_id || evt.id;
    const timeObj = new Date(evt.timestamp);
    const timeFormatted = isNaN(timeObj.getTime()) ? evt.timestamp : timeObj.toLocaleString("pt-BR");

    dom.modalCameraTitle.textContent = evt.camera_name || evt.camera_id;
    dom.modalTime.textContent = timeFormatted;
    dom.modalTrackId.textContent = `#${evt.tracking_id}`;
    dom.modalConfidence.textContent = `${Math.round((evt.confidence || 0) * 100)}%`;
    dom.modalEventId.textContent = evt.event_id || `ID: ${evt.id}`;

    if (dom.modalFullRawLink) {
      dom.modalFullRawLink.href = `/api/events/${id}/full?raw=true`;
    }

    // 1. Limpa imagens anteriores e mostra loaders para evitar exibir imagem errada
    dom.modalCropImg.classList.remove("loaded");
    dom.modalFullImg.classList.remove("loaded");
    if (dom.modalCropLoader) dom.modalCropLoader.classList.remove("hidden");
    if (dom.modalFullLoader) dom.modalFullLoader.classList.remove("hidden");

    // 2. Handlers do Crop
    dom.modalCropImg.onload = () => {
      dom.modalCropImg.classList.add("loaded");
      if (dom.modalCropLoader) dom.modalCropLoader.classList.add("hidden");
    };
    dom.modalCropImg.onerror = () => {
      if (dom.modalCropLoader) dom.modalCropLoader.classList.add("hidden");
    };

    // 3. Handlers da Cena Completa com Bounding Box
    dom.modalFullImg.onload = () => {
      dom.modalFullImg.classList.add("loaded");
      if (dom.modalFullLoader) dom.modalFullLoader.classList.add("hidden");
    };
    dom.modalFullImg.onerror = () => {
      if (dom.modalFullLoader) dom.modalFullLoader.classList.add("hidden");
    };

    // 4. Inicia o carregamento das imagens
    dom.modalCropImg.src = `/api/events/${id}/crop`;
    dom.modalFullImg.src = `/api/events/${id}/full`;

    // 5. Se já estava no cache HTTP local, ativa imediatamente
    if (dom.modalCropImg.complete && dom.modalCropImg.naturalWidth > 0) {
      dom.modalCropImg.classList.add("loaded");
      if (dom.modalCropLoader) dom.modalCropLoader.classList.add("hidden");
    }
    if (dom.modalFullImg.complete && dom.modalFullImg.naturalWidth > 0) {
      dom.modalFullImg.classList.add("loaded");
      if (dom.modalFullLoader) dom.modalFullLoader.classList.add("hidden");
    }

    dom.modal.classList.add("active");
  }

  function closeModal() {
    dom.modal.classList.remove("active");
    // Libera referência para não manter buffers no DOM
    dom.modalCropImg.src = "";
    dom.modalFullImg.src = "";
    dom.modalCropImg.classList.remove("loaded");
    dom.modalFullImg.classList.remove("loaded");
  }

  init();
});
