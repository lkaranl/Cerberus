/**
 * Cerberus AI — Clean CCTV 2x2 Application
 * 
 * Focado 100% em desempenho, máxima taxa de quadros e zero poluição visual.
 * Grid 2x2:
 * - Quadrante 1: Câmera 1 (Portão)
 * - Quadrante 2: Câmera 2 (Garagem)
 * - Quadrante 3: Câmera 3 (Área fundo)
 * - Quadrante 4: Relógio Minimalista CFTV + Radar de Presença em Tempo Real
 */

document.addEventListener("DOMContentLoaded", () => {
  const state = {
    cameras: [],
    wsConnections: {},
    eventsWs: null,
    eventsCount: 0,
    statsInterval: null,
    activeModalEventId: null
  };

  const dom = {
    gridContainer: document.getElementById("cctv-grid"),
    quadrantInfo: document.getElementById("quadrant-info"),
    clock: document.getElementById("live-clock"),
    greeting: document.getElementById("clock-greeting"),
    date: document.getElementById("live-date"),
    gpuTag: document.getElementById("cctv-gpu-tag"),
    radarFeed: document.getElementById("radar-events-feed"),
    radarEmpty: document.getElementById("radar-empty"),
    modal: document.getElementById("event-modal"),
    modalClose: document.getElementById("modal-close"),
    modalCameraTitle: document.getElementById("modal-camera-title"),
    modalTime: document.getElementById("modal-time"),
    modalCropImg: document.getElementById("modal-crop-img"),
    modalFullImg: document.getElementById("modal-full-img"),
    modalFullRawLink: document.getElementById("modal-full-raw-link"),
    modalTrackId: document.getElementById("modal-track-id"),
    modalConfidence: document.getElementById("modal-confidence"),
    modalEventId: document.getElementById("modal-event-id")
  };



  // --- 2. Decodificador Universal de Frames (Resiliente a Safari / WebKit) ---
  const isWebKit = /^((?!chrome|android).)*safari/i.test(navigator.userAgent) || 
                   (navigator.userAgent.includes("AppleWebKit") && !navigator.userAgent.includes("Chrome"));
  let useImageFallback = isWebKit;

  const camImagePool = {};

  function decodeFrameSource(camId, blob) {
    const typedBlob = blob.type === "image/jpeg" ? blob : blob.slice(0, blob.size, "image/jpeg");

    if (!useImageFallback && typeof createImageBitmap === "function") {
      return createImageBitmap(typedBlob).catch(() => {
        useImageFallback = true;
        return decodeViaImageTag(camId, typedBlob);
      });
    }

    return decodeViaImageTag(camId, typedBlob);
  }

  function decodeViaImageTag(camId, blob) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(blob);
      if (!camImagePool[camId]) {
        camImagePool[camId] = new Image();
      }
      const img = camImagePool[camId];

      img.onload = () => {
        URL.revokeObjectURL(url);
        resolve(img);
      };

      img.onerror = () => {
        URL.revokeObjectURL(url);
        reject();
      };

      img.src = url;
    });
  }

  // --- 3. Inicialização Geral ---
  async function init() {
    setupModalEvents();

    await loadSystemStats();
    await loadCameras();
    await loadRecentEvents();

    connectEventsWebSocket();

    // Atualiza status e telemetria a cada 5 segundos
    state.statsInterval = setInterval(() => {
      loadSystemStats();
      refreshCameraTelemetry();
    }, 5000);
  }

  // --- 4. Obter Estatísticas de Hardware ---
  async function loadSystemStats() {
    try {
      const res = await fetch("/api/stats");
      if (!res.ok) return;
      const data = await res.json();

      if (dom.gpuTag && data.hardware) {
        if (data.hardware.name) {
          const cleanName = data.hardware.name.replace(/NVIDIA\s+/i, "").replace(/GeForce\s+/i, "");
          const mem = data.hardware.allocated_mb ? ` (${Math.round(data.hardware.allocated_mb)} MB)` : "";
          const tech = data.hardware.nvdec_active ? "NVDEC ⚡" : (data.hardware.tensorrt_active ? "TRT ⚡" : "CUDA ⚡");
          dom.gpuTag.textContent = `${cleanName} • ${tech}${mem}`;
        } else {
          dom.gpuTag.textContent = "GTX 1660 • CUDA ⚡";
        }
      }
    } catch (e) {}
  }

  // --- 5. Carregar e Injetar Câmeras no Grid 2x2 ---
  async function loadCameras() {
    try {
      const res = await fetch("/api/cameras");
      if (!res.ok) return;
      state.cameras = await res.json();

      renderCctvSlots();
      connectAllCameraStreams();
    } catch (e) {
      console.error("Erro ao carregar câmeras:", e);
    }
  }

  function renderCctvSlots() {
    // Remove slots anteriores preservando o 4º quadrante (quadrant-info)
    const existingSlots = dom.gridContainer.querySelectorAll(".cctv-cam-slot");
    existingSlots.forEach(slot => slot.remove());

    state.cameras.forEach((cam) => {
      const slot = document.createElement("div");
      slot.className = "cctv-cam-slot";
      slot.id = `slot-${cam.id}`;
      slot.dataset.camId = cam.id;

      slot.innerHTML = `
        <div class="cctv-placeholder" id="placeholder-${cam.id}">
          <div class="cctv-spinner"></div>
          <span>Conectando...</span>
        </div>

        <canvas class="cctv-canvas" id="canvas-${cam.id}"></canvas>

        <div class="cctv-top-pill ${cam.connected ? '' : 'offline'}" id="top-pill-${cam.id}">
          <span class="cctv-pill-dot"></span>
          <span id="fps-val-${cam.id}">-- FPS</span>
        </div>

        <button class="cctv-fullscreen-btn" id="fs-btn-${cam.id}" title="Tela Cheia">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="15 3 21 3 21 9"/>
            <polyline points="9 21 3 21 3 15"/>
            <line x1="21" y1="3" x2="14" y2="10"/>
            <line x1="3" y1="21" x2="10" y2="14"/>
          </svg>
        </button>
      `;

      // Insere antes do quadrant-info para manter os 3 slots no topo e à esquerda
      dom.gridContainer.insertBefore(slot, dom.quadrantInfo);

      const fsBtn = slot.querySelector(`#fs-btn-${cam.id}`);
      if (fsBtn) {
        fsBtn.addEventListener("click", () => toggleSlotFullscreen(slot));
      }

      slot.addEventListener("dblclick", () => toggleSlotFullscreen(slot));
    });
  }

  function toggleSlotFullscreen(slotElement) {
    const isFull = slotElement.classList.contains("fullscreen-active");
    document.querySelectorAll(".cctv-cam-slot").forEach(s => s.classList.remove("fullscreen-active"));

    if (!isFull) {
      slotElement.classList.add("fullscreen-active");
    }
  }

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      document.querySelectorAll(".cctv-cam-slot").forEach(s => s.classList.remove("fullscreen-active"));
      closeModal();
    }
  });

  // --- 6. Conectar Stream via WebSocket Binário com Pacing e Watchdog ---
  function connectCameraStream(camId) {
    const existingWs = state.wsConnections[camId];
    if (existingWs) {
      // Se já estiver conectado ou em handshake, não fecha para evitar loop
      if (existingWs.readyState === WebSocket.OPEN || existingWs.readyState === WebSocket.CONNECTING) {
        return;
      }
      try {
        existingWs.onclose = null;
        existingWs.close();
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
    let lastFrameTime = performance.now();

    // Watchdog: se não receber frame em 800ms, manda "next" para reativar o pipeline
    const watchdog = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN && !document.hidden) {
        if (performance.now() - lastFrameTime > 800) {
          ws.send("next");
        }
      }
    }, 400);

    ws.onopen = () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send("next");
      }
    };

    ws.onmessage = async (event) => {
      if (!ctx || !(event.data instanceof Blob) || event.data.size === 0) return;
      if (document.hidden) return;

      lastFrameTime = performance.now();

      try {
        const source = await decodeFrameSource(camId, event.data);
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

        if (typeof source.close === "function") {
          source.close();
        }

        // Sinaliza ao servidor que o frame foi renderizado e solicita o próximo
        if (ws.readyState === WebSocket.OPEN && !document.hidden) {
          ws.send("next");
        }

        frameCount++;
        const now = performance.now();
        if (now - fpsTimer >= 1000) {
          const fps = Math.round(frameCount / ((now - fpsTimer) / 1000));
          const fpsEl = document.getElementById(`fps-val-${camId}`);
          if (fpsEl) fpsEl.textContent = `${fps} FPS`;

          const pill = document.getElementById(`top-pill-${camId}`);
          if (pill) pill.classList.remove("offline");

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
      clearInterval(watchdog);
      if (!document.hidden && state.cameras.some(c => c.id === camId)) {
        const pill = document.getElementById(`top-pill-${camId}`);
        if (pill) pill.classList.add("offline");

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

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      connectAllCameraStreams();
      loadSystemStats();
    } else {
      disconnectAllCameraStreams();
    }
  });

  async function refreshCameraTelemetry() {
    try {
      const res = await fetch("/api/cameras");
      if (!res.ok) return;
      const cams = await res.json();

      cams.forEach(cam => {
        const pill = document.getElementById(`top-pill-${cam.id}`);
        if (pill) {
          if (cam.connected) {
            pill.classList.remove("offline");
            const ws = state.wsConnections[cam.id];
            if (!document.hidden && (!ws || ws.readyState === WebSocket.CLOSED)) {
              connectCameraStream(cam.id);
            }
          } else {
            pill.classList.add("offline");
          }
        }
      });
    } catch (e) {}
  }

  // --- 7. Radar de Presença em Tempo Real ---
  async function loadRecentEvents() {
    try {
      const res = await fetch("/api/events?limit=15");
      if (!res.ok) return;
      const data = await res.json();

      if (data.events && data.events.length > 0) {
        if (dom.radarEmpty) dom.radarEmpty.remove();
        data.events.forEach(evt => {
          renderRadarItem(evt, false);
        });
      }
    } catch (e) {}
  }

  function connectEventsWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/events`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === "PERSON_DETECTED" && payload.data) {
          const evt = payload.data;
          if (dom.radarEmpty && dom.radarEmpty.parentNode) {
            dom.radarEmpty.remove();
          }

          renderRadarItem(evt, true);
          flashSlotAlert(evt.camera_id, evt.entity_type, evt.is_known);
        } else if (payload.type === "EVENT_UPDATED" && payload.data) {
          updateRadarItem(payload.data);
        }
      } catch (err) {}
    };

    ws.onclose = () => {
      setTimeout(connectEventsWebSocket, 3000);
    };

    state.eventsWs = ws;
  }

  function flashSlotAlert(camId, entityType, isKnown) {
    const slot = document.getElementById(`slot-${camId}`);
    if (!slot) return;

    let alertCls = "alert-unknown";
    if (entityType === "vehicle") alertCls = "alert-vehicle";
    else if (isKnown) alertCls = "alert-resident";

    slot.classList.remove("alert-resident", "alert-unknown", "alert-vehicle");
    slot.classList.add(alertCls);

    if (slot._alertTimeout) clearTimeout(slot._alertTimeout);
    slot._alertTimeout = setTimeout(() => {
      slot.classList.remove(alertCls);
    }, 4000);
  }

  function renderRadarItem(evt, isNew = false) {
    const eventId = evt.event_id || evt.id;
    const item = document.createElement("div");
    item.className = "radar-item";
    item.id = `radar-item-${eventId}`;

    const timeObj = new Date(evt.timestamp);
    const timeFormatted = isNaN(timeObj.getTime())
      ? evt.timestamp
      : timeObj.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });

    const cropUrl = `/api/events/${eventId}/crop`;

    let badgeClass = "unknown";
    let typeName = evt.person_name || "Desconhecido";

    if (evt.entity_type === "vehicle") {
      badgeClass = "vehicle";
      typeName = evt.license_plate ? `🚗 ${evt.license_plate}` : "🚗 Veículo";
    } else if (evt.is_known) {
      badgeClass = "resident";
    }

    item.innerHTML = `
      <img class="radar-item-thumb" src="${cropUrl}" alt="Crop" loading="lazy" onerror="this.src='/static/img/placeholder.png'">
      <div class="radar-item-details">
        <span class="radar-item-name">${typeName}</span>
        <span class="radar-item-meta">
          <span>${evt.camera_name || evt.camera_id}</span>
          <span>•</span>
          <span>${timeFormatted}</span>
        </span>
      </div>
      <span class="radar-item-badge ${badgeClass}">${badgeClass.toUpperCase()}</span>
    `;

    item.addEventListener("click", () => openModal(evt));

    if (isNew) {
      dom.radarFeed.prepend(item);
      while (dom.radarFeed.children.length > 20) {
        dom.radarFeed.removeChild(dom.radarFeed.lastChild);
      }
    } else {
      dom.radarFeed.appendChild(item);
    }
  }

  function updateRadarItem(evt) {
    const eventId = evt.event_id || evt.id;
    const item = document.getElementById(`radar-item-${eventId}`);
    if (!item) return;

    const img = item.querySelector(".radar-item-thumb");
    if (img) img.src = `/api/events/${eventId}/crop?t=${Date.now()}`;

    const nameEl = item.querySelector(".radar-item-name");
    const badgeEl = item.querySelector(".radar-item-badge");

    let badgeClass = "unknown";
    let typeName = evt.person_name || "Desconhecido";

    if (evt.entity_type === "vehicle") {
      badgeClass = "vehicle";
      typeName = evt.license_plate ? `🚗 ${evt.license_plate}` : "🚗 Veículo";
    } else if (evt.is_known) {
      badgeClass = "resident";
    }

    if (nameEl) nameEl.textContent = typeName;
    if (badgeEl) {
      badgeEl.className = `radar-item-badge ${badgeClass}`;
      badgeEl.textContent = badgeClass.toUpperCase();
    }

    if (state.activeModalEventId === eventId) {
      if (dom.modalCropImg) dom.modalCropImg.src = `/api/events/${eventId}/crop?t=${Date.now()}`;
      if (dom.modalFullImg) dom.modalFullImg.src = `/api/events/${eventId}/full?t=${Date.now()}`;
    }
  }

  // --- 8. Modal de Evidência ---
  function setupModalEvents() {
    if (dom.modalClose) dom.modalClose.addEventListener("click", closeModal);
    if (dom.modal) {
      dom.modal.addEventListener("click", (e) => {
        if (e.target === dom.modal) closeModal();
      });
    }
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

    dom.modalCropImg.src = `/api/events/${id}/crop`;
    dom.modalFullImg.src = `/api/events/${id}/full`;

    state.activeModalEventId = id;
    dom.modal.classList.add("active");
  }

  function closeModal() {
    state.activeModalEventId = null;
    if (dom.modal) dom.modal.classList.remove("active");
    if (dom.modalCropImg) dom.modalCropImg.src = "";
    if (dom.modalFullImg) dom.modalFullImg.src = "";
  }

  init();
});
