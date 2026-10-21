/**
 * Sentinela Vision / Cerberus AI - Frontend Command Center Application
 * 
 * Funcionalidades:
 * - Central de Comando Tático em tela cheia com glassmorphism de alta fidelidade
 * - WebSockets binários de ultrabaixa latência (zero-lag adaptativo)
 * - HUD Tático Flutuante sobre os streams de câmeras 2K (Snapshot instantâneo, Tela Cheia)
 * - Drawer lateral retrátil com histórico persistente no localStorage
 * - Timeline inteligente com filtros rápidos em chips e contadores em tempo real
 * - Bordas reativas dinâmicas nas câmeras conforme tipo de detecção (Morador, Invasor, Veículo)
 * - Modal tático de evidências biométricas lado a lado (Crop e Cena Completa 2K)
 */

document.addEventListener("DOMContentLoaded", () => {
  const state = {
    cameras: [],
    currentView: "grid", // "grid" ou camera_id ("cam-0", etc.)
    wsConnections: {}, // camera_id -> WebSocket
    eventsWs: null,
    eventsCount: 0,
    eventsData: [], // Armazena objetos de eventos recentes para contagem e filtros rápidos
    currentFilter: "all", // "all", "residents", "unknowns", "vehicles", "pets"
    isDrawerCollapsed: false,
    statsInterval: null,
    activeModalEventId: null
  };

  // Elementos do DOM
  const dom = {
    clock: document.getElementById("live-clock"),
    hardwareLabel: document.getElementById("hardware-label"),
    camsActiveLabel: document.getElementById("cams-active-label"),
    mainLayout: document.getElementById("main-layout"),
    btnToggleDrawer: document.getElementById("btn-toggle-drawer"),
    videoContainer: document.getElementById("video-container"),
    singleSelectors: document.getElementById("single-selectors"),
    btnGrid: document.getElementById("btn-grid"),
    filterChips: document.querySelectorAll("#filter-chips .filter-chip"),
    countAll: document.getElementById("count-all"),
    countResidents: document.getElementById("count-residents"),
    countUnknowns: document.getElementById("count-unknowns"),
    countVehicles: document.getElementById("count-vehicles"),
    countPets: document.getElementById("count-pets"),
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

  // --- 1. Relógio Tático ao Vivo ---
  function startClock() {
    function update() {
      const now = new Date();
      if (dom.clock) {
        dom.clock.textContent = now.toLocaleTimeString("pt-BR", { hour12: false });
      }
    }
    update();
    setInterval(update, 1000);
  }

  // --- 2. Controle do Drawer Retrátil Lateral ---
  function setupDrawer() {
    // Recupera estado anterior salvo no navegador
    const savedState = localStorage.getItem("cerberus_drawer_collapsed");
    if (savedState === "true") {
      state.isDrawerCollapsed = true;
      if (dom.mainLayout) dom.mainLayout.classList.add("drawer-collapsed");
    }

    if (dom.btnToggleDrawer) {
      dom.btnToggleDrawer.addEventListener("click", () => {
        state.isDrawerCollapsed = !state.isDrawerCollapsed;
        if (dom.mainLayout) {
          dom.mainLayout.classList.toggle("drawer-collapsed", state.isDrawerCollapsed);
        }
        localStorage.setItem("cerberus_drawer_collapsed", state.isDrawerCollapsed);
      });
    }
  }

  // --- 3. Inicialização Geral ---
  async function init() {
    startClock();
    setupDrawer();
    setupModalEvents();
    setupViewSwitching();
    setupClearScreen();
    setupFilterChips();

    await loadSystemStats();
    await loadCameras();
    await loadRecentEvents();

    connectEventsWebSocket();

    // Atualiza status e telemetria a cada 4 segundos
    state.statsInterval = setInterval(() => {
      loadSystemStats();
      refreshCameraTelemetry();
    }, 4000);
  }

  // --- 4. Carregar Estatísticas do Sistema e GPU ---
  async function loadSystemStats() {
    try {
      const res = await fetch("/api/stats");
      if (!res.ok) return;
      const data = await res.json();

      if (dom.hardwareLabel && data.hardware) {
        if (data.hardware.nvdec_active && data.hardware.yolo_tensorrt && data.hardware.facenet_tensorrt) {
          dom.hardwareLabel.textContent = `GTX 1660 [NVDEC + YOLO + FaceNet TRT ⚡] (${data.hardware.allocated_mb} MB)`;
        } else if (data.hardware.yolo_tensorrt && data.hardware.facenet_tensorrt) {
          dom.hardwareLabel.textContent = `GTX 1660 [YOLO + FaceNet TRT ⚡] (${data.hardware.allocated_mb} MB)`;
        } else if (data.hardware.tensorrt_active) {
          dom.hardwareLabel.textContent = `GTX 1660 [TensorRT FP16 ⚡] (${data.hardware.allocated_mb} MB)`;
        } else if (data.hardware.type && data.hardware.type.includes("CUDA")) {
          dom.hardwareLabel.textContent = `NVIDIA GTX 1660 (${data.hardware.allocated_mb} MB VRAM)`;
        } else if (data.hardware.type && data.hardware.type.includes("Metal")) {
          dom.hardwareLabel.textContent = "Apple Silicon (MPS)";
        } else {
          dom.hardwareLabel.textContent = data.hardware.device.toUpperCase();
        }
      }

      if (dom.camsActiveLabel) {
        dom.camsActiveLabel.textContent = `${data.cameras_active || 0} Ativas (2K 25 FPS)`;
      }
    } catch (err) {
      console.warn("Falha ao obter stats do sistema:", err);
    }
  }

  // --- 5. Carregar Câmeras e Configurar Grid ---
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

  // --- 6. Renderizar Botões de Seleção de Câmeras ---
  function renderViewButtons() {
    if (!dom.singleSelectors) return;
    dom.singleSelectors.innerHTML = "";

    state.cameras.forEach(cam => {
      const btn = document.createElement("button");
      btn.className = "view-pill";
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

  // --- 7. Configurar Alternância de Visualização (Grid vs Single) ---
  function setupViewSwitching() {
    if (dom.btnGrid) {
      dom.btnGrid.addEventListener("click", () => switchView("grid"));
    }
  }

  function switchView(viewId) {
    state.currentView = viewId;

    // Atualiza classes ativas nos pills
    document.querySelectorAll(".view-pill").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.view === viewId);
    });

    if (viewId === "grid") {
      dom.videoContainer.className = "tactical-grid-viewport mode-grid";
      document.querySelectorAll(".camera-card").forEach(card => {
        card.style.display = "flex";
      });
    } else {
      dom.videoContainer.className = "tactical-grid-viewport mode-single";
      document.querySelectorAll(".camera-card").forEach(card => {
        card.style.display = (card.dataset.camId === viewId) ? "flex" : "none";
      });
    }
  }

  // --- 8. Renderizar Cards de Câmeras com HUD Tático ---
  function renderCameraCards() {
    if (!dom.videoContainer) return;
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

        <div class="video-viewport" id="viewport-${cam.id}">
          <div class="video-placeholder" id="placeholder-${cam.id}">
            <div class="spinner"></div>
            <span>Conectando ao stream RTSP 2K...</span>
          </div>
          <canvas class="video-canvas" id="canvas-${cam.id}"></canvas>

          <!-- HUD Tático Superior (Resolução 2K + Aceleração) -->
          <div class="cam-hud-overlay-top">
            <span class="hud-tag-resolution">2K QHD • 25 FPS</span>
            <span class="hud-tag-resolution" style="border-color: rgba(16, 185, 129, 0.4); color: var(--accent-emerald);">NVDEC ⚡</span>
          </div>

          <!-- HUD Tático Inferior (Ações Rápidas no Hover) -->
          <div class="cam-hud-overlay-bottom">
            <div class="hud-actions-left">
              <div class="cam-overlay-tag">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <circle cx="12" cy="12" r="10"/>
                  <polyline points="12 6 12 12 16 14"/>
                </svg>
                ${cam.name || cam.id}
              </div>
              <div class="cam-overlay-tag detection-badge" id="det-tag-${cam.id}" style="display: none;">
                ● Atividade
              </div>
            </div>

            <div class="hud-actions-right">
              <button class="btn-hud-tool" id="btn-snapshot-${cam.id}" title="Capturar frame 2K instantâneo">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
                  <circle cx="12" cy="13" r="4"/>
                </svg>
                Snapshot
              </button>
              <button class="btn-hud-tool" id="btn-fullscreen-${cam.id}" title="Expandir câmera em tela cheia">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <polyline points="15 3 21 3 21 9"/>
                  <polyline points="9 21 3 21 3 15"/>
                  <line x1="21" y1="3" x2="14" y2="10"/>
                  <line x1="3" y1="21" x2="10" y2="14"/>
                </svg>
                Tela Cheia
              </button>
            </div>
          </div>
        </div>
      `;

      dom.videoContainer.appendChild(card);

      // Listeners do HUD da Câmera
      const btnSnap = card.querySelector(`#btn-snapshot-${cam.id}`);
      if (btnSnap) {
        btnSnap.addEventListener("click", (e) => {
          e.stopPropagation();
          takeCameraSnapshot(cam.id, cam.name);
        });
      }

      const btnFull = card.querySelector(`#btn-fullscreen-${cam.id}`);
      if (btnFull) {
        btnFull.addEventListener("click", (e) => {
          e.stopPropagation();
          toggleCameraFullscreen(card);
        });
      }
    });

    dom.videoContainer.className = "tactical-grid-viewport mode-grid";
  }

  // --- 9. Ferramenta de Snapshot 2K Instantâneo ---
  function takeCameraSnapshot(camId, camName) {
    const canvas = document.getElementById(`canvas-${camId}`);
    if (!canvas || canvas.width === 0) {
      console.warn(`Canvas de ${camId} ainda não possui frames renderizados.`);
      return;
    }

    const btn = document.getElementById(`btn-snapshot-${camId}`);
    const origHtml = btn ? btn.innerHTML : "";
    if (btn) btn.innerHTML = `✓ Salvo!`;

    try {
      const dataUrl = canvas.toDataURL("image/jpeg", 0.95);
      const link = document.createElement("a");
      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      const safeName = (camName || camId).toLowerCase().replace(/[^a-z0-9]/g, "_");
      link.download = `cerberus_${safeName}_${timestamp}.jpg`;
      link.href = dataUrl;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err) {
      console.error("Erro ao gerar snapshot 2K:", err);
    }

    if (btn) {
      setTimeout(() => {
        btn.innerHTML = origHtml;
      }, 1500);
    }
  }

  // --- 10. Ferramenta de Tela Cheia Tática (Fullscreen) ---
  function toggleCameraFullscreen(cardElement) {
    if (!document.fullscreenElement) {
      if (cardElement.requestFullscreen) {
        cardElement.requestFullscreen();
      } else if (cardElement.webkitRequestFullscreen) {
        cardElement.webkitRequestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  }

  // --- 11. Decodificador Universal de Frames (Resiliente a Safari / WebKit) ---
  const isWebKit = /^((?!chrome|android).)*safari/i.test(navigator.userAgent) || 
                   (navigator.userAgent.includes("AppleWebKit") && !navigator.userAgent.includes("Chrome"));
  let useImageFallback = isWebKit;

  function decodeFrameSource(blob) {
    const typedBlob = blob.type === "image/jpeg" ? blob : blob.slice(0, blob.size, "image/jpeg");

    if (!useImageFallback && typeof createImageBitmap === "function") {
      return createImageBitmap(typedBlob).catch(() => {
        // Fallback automático para Safari / WebKit em caso de restrição de segurança interna
        useImageFallback = true;
        return decodeViaImageTag(typedBlob);
      });
    }

    return decodeViaImageTag(typedBlob);
  }

  function decodeViaImageTag(blob) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        resolve(img);
      };
      img.onerror = (err) => {
        URL.revokeObjectURL(url);
        reject(err);
      };
      img.src = url;
    });
  }

  // --- 12. Conectar Stream de Câmera via WebSocket Binário ---
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

    ws.onopen = () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send("next");
      }
    };

    ws.onmessage = async (event) => {
      if (!ctx || !(event.data instanceof Blob)) return;

      // Se a aba estiver oculta em segundo plano, não processa o frame
      if (document.hidden) return;

      try {
        const source = await decodeFrameSource(event.data);
        const sWidth = source.naturalWidth || source.width;
        const sHeight = source.naturalHeight || source.height;

        if (canvas.width !== sWidth || canvas.height !== sHeight) {
          canvas.width = sWidth;
          canvas.height = sHeight;
          ctx.imageSmoothingEnabled = true;
          ctx.imageSmoothingQuality = "high";
          if (placeholder) placeholder.classList.add("hidden");
        }

        ctx.drawImage(source, 0, 0, canvas.width, canvas.height);

        // Se for ImageBitmap, libera o recurso explicitamente
        if (typeof source.close === "function") {
          source.close();
        }

        // Solicita próximo frame ao backend de streaming
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
        if (ws.readyState === WebSocket.OPEN && !document.hidden) {
          ws.send("next");
        }
      }
    };

    ws.onclose = () => {
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
          ws.onclose = null;
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

  // --- 12. Sincronização Inteligente de Abas (Zero Buffer Acumulado) ---
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      connectAllCameraStreams();
      loadSystemStats();
      refreshCameraTelemetry();
    } else {
      disconnectAllCameraStreams();
    }
  });

  // --- 13. Atualizar Telemetria das Câmeras ---
  async function refreshCameraTelemetry() {
    try {
      const res = await fetch("/api/cameras");
      if (!res.ok) return;
      const cams = await res.json();

      cams.forEach(cam => {
        const statusEl = document.getElementById(`status-pill-${cam.id}`);
        const fpsEl = document.getElementById(`fps-pill-${cam.id}`);
        if (statusEl) {
          if (cam.connected) {
            if (!statusEl.classList.contains("online")) {
              statusEl.className = "cam-status-pill online";
              statusEl.innerHTML = `<span class="chip-dot glow-green"></span> ONLINE`;
            }
            const ws = state.wsConnections[cam.id];
            if (!document.hidden && (!ws || ws.readyState === WebSocket.CLOSED)) {
              connectCameraStream(cam.id);
            }
          } else {
            statusEl.className = "cam-status-pill offline";
            statusEl.innerHTML = `<span class="chip-dot"></span> OFFLINE`;
            if (fpsEl) fpsEl.textContent = "0.0 FPS";
          }
        }
      });
    } catch (e) {}
  }

  // --- 14. Filtros Rápidos em Chips na Timeline ---
  function setupFilterChips() {
    if (!dom.filterChips) return;
    dom.filterChips.forEach(chip => {
      chip.addEventListener("click", () => {
        const filter = chip.dataset.filter || "all";
        applyTimelineFilter(filter);
      });
    });
  }

  function applyTimelineFilter(filter) {
    state.currentFilter = filter;

    // Atualiza botões
    dom.filterChips.forEach(chip => {
      chip.classList.toggle("active", chip.dataset.filter === filter);
    });

    // Filtra cards existentes na timeline
    const cards = dom.eventsList.querySelectorAll(".event-card");
    cards.forEach(card => {
      const cat = card.dataset.category;
      if (filter === "all" || cat === filter) {
        card.style.display = "flex";
      } else {
        card.style.display = "none";
      }
    });
  }

  function getCategoryForEvent(evt) {
    if (evt.entity_type === "vehicle") return "vehicles";
    if (evt.entity_type === "pet") return "pets";
    if (evt.is_known) return "residents";
    return "unknowns";
  }

  function updateFilterCounters() {
    let all = state.eventsData.length;
    let residents = 0;
    let unknowns = 0;
    let vehicles = 0;
    let pets = 0;

    state.eventsData.forEach(evt => {
      const cat = getCategoryForEvent(evt);
      if (cat === "residents") residents++;
      else if (cat === "unknowns") unknowns++;
      else if (cat === "vehicles") vehicles++;
      else if (cat === "pets") pets++;
    });

    if (dom.countAll) dom.countAll.textContent = all;
    if (dom.countResidents) dom.countResidents.textContent = residents;
    if (dom.countUnknowns) dom.countUnknowns.textContent = unknowns;
    if (dom.countVehicles) dom.countVehicles.textContent = vehicles;
    if (dom.countPets) dom.countPets.textContent = pets;
  }

  // --- 15. Carregar Histórico Inicial de Eventos ---
  async function loadRecentEvents() {
    try {
      const res = await fetch("/api/events?limit=30");
      if (!res.ok) return;
      const data = await res.json();

      if (data.events && data.events.length > 0) {
        if (dom.eventsEmpty) dom.eventsEmpty.remove();
        state.eventsData = data.events;
        state.eventsCount = data.total || data.events.length;
        if (dom.eventsBadge) dom.eventsBadge.textContent = state.eventsCount;

        updateFilterCounters();

        data.events.forEach(evt => {
          renderEventCard(evt, false);
        });

        // Aplica o filtro atual aos cards carregados
        applyTimelineFilter(state.currentFilter);
      }
    } catch (err) {
      console.warn("Falha ao carregar eventos:", err);
    }
  }

  // --- 16. Conectar Canal de Eventos em Tempo Real ---
  function connectEventsWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/events`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === "PERSON_DETECTED" && payload.data) {
          const evt = payload.data;
          state.eventsCount++;
          if (dom.eventsBadge) dom.eventsBadge.textContent = state.eventsCount;

          if (dom.eventsEmpty && dom.eventsEmpty.parentNode) {
            dom.eventsEmpty.remove();
          }

          // Adiciona no topo do histórico
          state.eventsData.unshift(evt);
          if (state.eventsData.length > 50) state.eventsData.pop();
          updateFilterCounters();

          renderEventCard(evt, true);
          flashDetectionBadge(evt.camera_id);
          triggerCameraAlert(evt.camera_id, evt.entity_type, evt.is_known);
        } else if (payload.type === "EVENT_UPDATED" && payload.data) {
          updateEventCard(payload.data);
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

  // --- 17. Reatividade Visual de Borda nas Câmeras (Glow Tático) ---
  function triggerCameraAlert(camId, entityType, isKnown) {
    const card = document.querySelector(`.camera-card[data-cam-id="${camId}"]`);
    if (!card) return;

    let alertClass = "camera-card--alert";
    if (entityType === "vehicle") {
      alertClass = "camera-card--vehicle";
    } else if (isKnown) {
      alertClass = "camera-card--resident";
    }

    card.classList.remove("camera-card--resident", "camera-card--alert", "camera-card--vehicle");
    card.classList.add(alertClass);

    if (card._alertTimeout) clearTimeout(card._alertTimeout);
    card._alertTimeout = setTimeout(() => {
      card.classList.remove(alertClass);
    }, 4500);
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

  // --- 18. Atualizar Card Existente ao Reconhecer Rosto ---
  function updateEventCard(evt) {
    const eventId = evt.event_id || evt.id;
    const card = document.getElementById(`event-card-${eventId}`);
    if (!card) return;

    // Atualiza categoria no dataset
    const newCategory = getCategoryForEvent(evt);
    card.dataset.category = newCategory;

    // Atualiza thumbnail com cache-buster
    const img = card.querySelector(".event-thumb");
    if (img) {
      img.src = `/api/events/${eventId}/crop?t=${Date.now()}`;
    }

    // Atualiza badge de identificação
    const badge = card.querySelector(".event-id-badge");
    if (badge) {
      let typeIcon = "👤";
      let badgeStyle = "background: var(--accent-emerald-dim); color: var(--accent-emerald); border: 1px solid rgba(16, 185, 129, 0.4);";

      if (evt.entity_type === "vehicle") {
        typeIcon = "🚗";
        if (evt.is_known) badgeStyle = "background: rgba(0, 240, 255, 0.15); color: var(--accent-cyan); border: 1px solid rgba(0, 240, 255, 0.4);";
      } else if (evt.entity_type === "pet") {
        typeIcon = "🐾";
        if (evt.is_known) badgeStyle = "background: rgba(224, 64, 251, 0.15); color: #e040fb; border: 1px solid rgba(224, 64, 251, 0.4);";
      }

      if (!evt.is_known) {
        badgeStyle = "background: rgba(244, 63, 94, 0.15); color: var(--accent-rose); border: 1px solid rgba(244, 63, 94, 0.4);";
      }

      badge.style = badgeStyle;
      badge.innerHTML = `${typeIcon} ${evt.person_name}`;
    }

    // Atualiza contadores
    const idx = state.eventsData.findIndex(e => (e.event_id || e.id) === eventId);
    if (idx !== -1) {
      state.eventsData[idx] = evt;
      updateFilterCounters();
    }

    // Atualiza visibilidade com base no filtro ativo
    if (state.currentFilter !== "all" && newCategory !== state.currentFilter) {
      card.style.display = "none";
    } else {
      card.style.display = "flex";
    }

    // Se modal aberto inspecionando este evento, atualiza em tempo real
    if (state.activeModalEventId === eventId) {
      if (dom.modalCropImg) dom.modalCropImg.src = `/api/events/${eventId}/crop?t=${Date.now()}`;
      if (dom.modalFullImg) dom.modalFullImg.src = `/api/events/${eventId}/full?t=${Date.now()}`;
    }
  }

  // --- 19. Renderizar Card de Evento na Timeline Inteligente ---
  function renderEventCard(evt, isNew = false) {
    const eventId = evt.event_id || evt.id;
    const category = getCategoryForEvent(evt);

    const card = document.createElement("div");
    card.className = "event-card";
    card.id = `event-card-${eventId}`;
    card.dataset.category = category;

    // Respeita filtro ativo ao renderizar novo card
    if (state.currentFilter !== "all" && category !== state.currentFilter) {
      card.style.display = "none";
    }

    const timeObj = new Date(evt.timestamp);
    const timeFormatted = isNaN(timeObj.getTime())
      ? evt.timestamp
      : timeObj.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });

    const cropUrl = `/api/events/${eventId}/crop`;
    const confPct = Math.round((evt.confidence || 0) * 100);

    let typeIcon = "👤";
    let badgeStyle = "background: var(--accent-emerald-dim); color: var(--accent-emerald); border: 1px solid rgba(16, 185, 129, 0.4);";

    if (evt.entity_type === "vehicle") {
      typeIcon = "🚗";
      if (evt.is_known) badgeStyle = "background: rgba(0, 240, 255, 0.15); color: var(--accent-cyan); border: 1px solid rgba(0, 240, 255, 0.4);";
    } else if (evt.entity_type === "pet") {
      typeIcon = "🐾";
      if (evt.is_known) badgeStyle = "background: rgba(224, 64, 251, 0.15); color: #e040fb; border: 1px solid rgba(224, 64, 251, 0.4);";
    }

    if (!evt.is_known) {
      badgeStyle = "background: rgba(244, 63, 94, 0.15); color: var(--accent-rose); border: 1px solid rgba(244, 63, 94, 0.4);";
    }

    const plateHtml = evt.license_plate
      ? `<span class="event-plate-tag" style="font-family: var(--font-mono); font-size: 0.68rem; font-weight: 700; background: #000; color: #fff; border: 1px solid #333; padding: 2px 6px; border-radius: 3px; margin-left: 6px;">[${evt.license_plate}]</span>`
      : '';

    card.innerHTML = `
      <div class="event-thumb-wrap">
        <img class="event-thumb" src="${cropUrl}" alt="Evidência" loading="lazy" onerror="this.src='/static/img/placeholder.png'">
      </div>
      <div class="event-info">
        <div class="event-cam-row">
          <span class="event-cam-name">${evt.camera_name || evt.camera_id}</span>
          <span class="event-time mono">${timeFormatted}</span>
        </div>
        <div class="event-details-row">
          <span class="event-id-badge" style="${badgeStyle}">
            ${typeIcon} ${evt.person_name}
          </span>
          ${plateHtml}
          <span class="event-confidence mono">${confPct}%</span>
        </div>
      </div>
    `;

    // Pré-carrega imagem completa em background ao passar o mouse (hover)
    card.addEventListener("mouseenter", () => {
      prefetchEventImages(eventId);
    }, { once: true });

    card.addEventListener("click", () => openModal(evt));

    if (isNew) {
      dom.eventsList.prepend(card);
      while (dom.eventsList.children.length > 45) {
        dom.eventsList.removeChild(dom.eventsList.lastChild);
      }
      prefetchEventImages(eventId);
    } else {
      dom.eventsList.appendChild(card);
    }
  }

  // --- 20. Prefetch Inteligente de Imagens para o Modal ---
  const prefetchedEvents = new Set();
  function prefetchEventImages(eventId) {
    if (!eventId || prefetchedEvents.has(eventId)) return;
    prefetchedEvents.add(eventId);

    const preImg = new Image();
    preImg.src = `/api/events/${eventId}/full`;
  }

  // --- 21. Limpar Tela Tática ---
  function setupClearScreen() {
    if (dom.btnClearScreen) {
      dom.btnClearScreen.addEventListener("click", () => {
        dom.eventsList.innerHTML = `
          <div class="events-empty-state" id="events-empty">
            <div class="empty-radar-circle">
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/>
              </svg>
            </div>
            <p>Tela limpa.</p>
            <span>Novas detecções em tempo real continuarão surgindo aqui.</span>
          </div>
        `;
        state.eventsData = [];
        state.eventsCount = 0;
        if (dom.eventsBadge) dom.eventsBadge.textContent = "0";
        updateFilterCounters();
      });
    }
  }

  // --- 22. Modal Tático de Evidências Biométricas ---
  function setupModalEvents() {
    if (dom.modalClose) dom.modalClose.addEventListener("click", closeModal);
    if (dom.modal) {
      dom.modal.addEventListener("click", (e) => {
        if (e.target === dom.modal) closeModal();
      });
    }
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeModal();
    });
  }

  function openModal(evt) {
    const id = evt.event_id || evt.id;
    const timeObj = new Date(evt.timestamp);
    const timeFormatted = isNaN(timeObj.getTime()) ? evt.timestamp : timeObj.toLocaleString("pt-BR");

    if (dom.modalCameraTitle) dom.modalCameraTitle.textContent = evt.camera_name || evt.camera_id;
    if (dom.modalTime) dom.modalTime.textContent = timeFormatted;
    if (dom.modalTrackId) dom.modalTrackId.textContent = `#${evt.tracking_id}`;
    if (dom.modalConfidence) dom.modalConfidence.textContent = `${Math.round((evt.confidence || 0) * 100)}%`;
    if (dom.modalEventId) dom.modalEventId.textContent = evt.event_id || `ID: ${evt.id}`;

    if (dom.modalFullRawLink) {
      dom.modalFullRawLink.href = `/api/events/${id}/full?raw=true`;
    }

    dom.modalCropImg.classList.remove("loaded");
    dom.modalFullImg.classList.remove("loaded");
    if (dom.modalCropLoader) dom.modalCropLoader.classList.remove("hidden");
    if (dom.modalFullLoader) dom.modalFullLoader.classList.remove("hidden");

    dom.modalCropImg.onload = () => {
      dom.modalCropImg.classList.add("loaded");
      if (dom.modalCropLoader) dom.modalCropLoader.classList.add("hidden");
    };
    dom.modalCropImg.onerror = () => {
      if (dom.modalCropLoader) dom.modalCropLoader.classList.add("hidden");
    };

    dom.modalFullImg.onload = () => {
      dom.modalFullImg.classList.add("loaded");
      if (dom.modalFullLoader) dom.modalFullLoader.classList.add("hidden");
    };
    dom.modalFullImg.onerror = () => {
      if (dom.modalFullLoader) dom.modalFullLoader.classList.add("hidden");
    };

    dom.modalCropImg.src = `/api/events/${id}/crop`;
    dom.modalFullImg.src = `/api/events/${id}/full`;

    if (dom.modalCropImg.complete && dom.modalCropImg.naturalWidth > 0) {
      dom.modalCropImg.classList.add("loaded");
      if (dom.modalCropLoader) dom.modalCropLoader.classList.add("hidden");
    }
    if (dom.modalFullImg.complete && dom.modalFullImg.naturalWidth > 0) {
      dom.modalFullImg.classList.add("loaded");
      if (dom.modalFullLoader) dom.modalFullLoader.classList.add("hidden");
    }

    state.activeModalEventId = id;
    dom.modal.classList.add("active");
  }

  function closeModal() {
    state.activeModalEventId = null;
    if (dom.modal) dom.modal.classList.remove("active");
    if (dom.modalCropImg) {
      dom.modalCropImg.src = "";
      dom.modalCropImg.classList.remove("loaded");
    }
    if (dom.modalFullImg) {
      dom.modalFullImg.src = "";
      dom.modalFullImg.classList.remove("loaded");
    }
  }

  init();
});
