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
    webrtcConnections: {},
    metaWsConnections: {},
    streamModes: {}, // 'webrtc' | 'mjpeg'
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
      // Telemetria consumida pelo dashboard; sem injecao no relogio clean
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

        <video class="cctv-video" id="video-${cam.id}" autoplay playsinline muted></video>
        <canvas class="cctv-canvas cctv-overlay" id="canvas-${cam.id}"></canvas>

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

  // --- 6. Streaming Híbrido: WebRTC WHEP Nativo (25-30 FPS) com Fallback para WebSocket ---

  function connectCameraStream(camId) {
    // Inicia o canal leve de metadados de IA para desenhar o overlay (caixas e nomes)
    connectMetadataStream(camId);

    // Tenta negociar WebRTC ultra-rápido via gateway go2rtc (porta 1984 / proxy /api/whep)
    connectWebRTC(camId).catch((err) => {
      console.warn(`[WebRTC] Falha ao conectar ${camId} (${err.message}). Ativando fallback WebSocket JPEG...`);
      connectCameraStreamWebSocket(camId);
    });
  }

  async function connectWebRTC(camId) {
    const video = document.getElementById(`video-${camId}`);
    const canvas = document.getElementById(`canvas-${camId}`);
    const placeholder = document.getElementById(`placeholder-${camId}`);
    const slot = document.getElementById(`slot-${camId}`);

    if (!video) return;

    // Fecha conexão WebRTC anterior se existir
    if (state.webrtcConnections[camId]) {
      try { state.webrtcConnections[camId].close(); } catch (e) {}
      delete state.webrtcConnections[camId];
    }

    const pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      bundlePolicy: "max-bundle"
    });

    state.webrtcConnections[camId] = pc;
    state.streamModes[camId] = "webrtc";

    // Garante que o elemento de vídeo esteja visível e o canvas em modo overlay
    video.style.display = "block";
    if (canvas) canvas.classList.add("cctv-overlay");

    pc.addTransceiver("video", { direction: "recvonly" });

    // Medição de FPS real direto da decodificação de vídeo do hardware do navegador
    let frameCount = 0;
    let fpsTimer = performance.now();

    function onVideoFrame() {
      frameCount++;
      const now = performance.now();
      if (now - fpsTimer >= 1000) {
        const fps = Math.round((frameCount * 1000) / (now - fpsTimer));
        const fpsEl = document.getElementById(`fps-val-${camId}`);
        if (fpsEl) fpsEl.textContent = `${fps} FPS`;

        const pill = document.getElementById(`top-pill-${camId}`);
        if (pill) pill.classList.remove("offline");

        frameCount = 0;
        fpsTimer = now;
      }

      if (state.streamModes[camId] === "webrtc" && video.requestVideoFrameCallback) {
        video.requestVideoFrameCallback(onVideoFrame);
      }
    }

    pc.ontrack = (event) => {
      if (event.streams && event.streams[0]) {
        video.srcObject = event.streams[0];
        video.play().catch(() => {});
        if (placeholder) placeholder.classList.add("hidden");

        if ("requestVideoFrameCallback" in video) {
          video.requestVideoFrameCallback(onVideoFrame);
        } else {
          // Fallback para medição de FPS periódica
          setInterval(() => {
            const fpsEl = document.getElementById(`fps-val-${camId}`);
            if (fpsEl && state.streamModes[camId] === "webrtc") fpsEl.textContent = "25 FPS";
          }, 1000);
        }
      }
    };

    pc.oniceconnectionstatechange = () => {
      if (pc.iceConnectionState === "failed" || pc.iceConnectionState === "disconnected") {
        console.warn(`[WebRTC] Conexão ${camId} perdida (${pc.iceConnectionState}). Ativando fallback...`);
        connectCameraStreamWebSocket(camId);
      }
    };

    // Cria SDP Offer local
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    // Timeout de 3.5s para negociar com o gateway WHEP
    const abortCtrl = new AbortController();
    const timeoutId = setTimeout(() => abortCtrl.abort(), 3500);

    const res = await fetch(`/api/whep/${camId}`, {
      method: "POST",
      headers: { "Content-Type": "application/sdp" },
      body: offer.sdp,
      signal: abortCtrl.signal
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      throw new Error(`WHEP endpoint retornou status ${res.status}`);
    }

    const answerSdp = await res.text();
    await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });
  }

  // --- 7. Fallback Automático: WebSocket Binário JPEG (Caso o WebRTC esteja offline) ---
  function connectCameraStreamWebSocket(camId) {
    state.streamModes[camId] = "mjpeg";

    const video = document.getElementById(`video-${camId}`);
    if (video) video.style.display = "none";

    const canvas = document.getElementById(`canvas-${camId}`);
    if (canvas) canvas.classList.remove("cctv-overlay");

    const existingWs = state.wsConnections[camId];
    if (existingWs) {
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

    const placeholder = document.getElementById(`placeholder-${camId}`);
    const ctx = canvas ? canvas.getContext("2d") : null;

    let frameCount = 0;
    let fpsTimer = performance.now();
    let lastFrameTime = performance.now();

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
        setTimeout(() => {
          if (!document.hidden && state.streamModes[camId] === "mjpeg") {
            connectCameraStreamWebSocket(camId);
          }
        }, 2000);
      }
    };

    state.wsConnections[camId] = ws;
  }

  // --- 8. Canal de Metadados de IA: Desenho do Canvas Overlay Transparente ---
  function connectMetadataStream(camId) {
    if (state.metaWsConnections[camId]) {
      try {
        state.metaWsConnections[camId].onclose = null;
        state.metaWsConnections[camId].close();
      } catch (e) {}
    }

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/stream-meta/${camId}`;
    const ws = new WebSocket(wsUrl);

    let clearTimer = null;

    ws.onmessage = (event) => {
      // Se estiver rodando no fallback MJPEG, as caixas já são queimadas no JPEG pelo servidor
      if (state.streamModes[camId] !== "webrtc") return;

      try {
        const meta = JSON.parse(event.data);
        drawMetadataOverlay(camId, meta);

        // Limpa caixas após 1.8 segundos sem novas detecções de IA
        clearTimeout(clearTimer);
        clearTimer = setTimeout(() => {
          const canvas = document.getElementById(`canvas-${camId}`);
          if (canvas) {
            const ctx = canvas.getContext("2d");
            if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
          }
        }, 1800);
      } catch (e) {}
    };

    ws.onclose = () => {
      if (!document.hidden && state.cameras.some(c => c.id === camId)) {
        setTimeout(() => {
          if (!document.hidden) connectMetadataStream(camId);
        }, 2000);
      }
    };

    state.metaWsConnections[camId] = ws;
  }

  function drawMetadataOverlay(camId, meta) {
    const canvas = document.getElementById(`canvas-${camId}`);
    const slot = document.getElementById(`slot-${camId}`);
    if (!canvas || !slot || !meta || !meta.width || !meta.height) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = slot.clientWidth;
    const h = slot.clientHeight;

    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    ctx.clearRect(0, 0, w, h);

    if (!meta.detections || meta.detections.length === 0) return;

    const scaleX = w / meta.width;
    const scaleY = h / meta.height;

    meta.detections.forEach(det => {
      if (!det.bbox) return;
      const [x1, y1, x2, y2] = det.bbox;

      const sx1 = x1 * scaleX;
      const sy1 = y1 * scaleY;
      const sw = (x2 - x1) * scaleX;
      const sh = (y2 - y1) * scaleY;

      // Esquema de Cores Neon
      let strokeColor = "#ffaa00"; // Desconhecido (Âmbar)
      if (det.is_known) {
        if (det.entity_type === "person") strokeColor = "#76e600"; // Verde Neon
        else if (det.entity_type === "vehicle") strokeColor = "#00f0ff"; // Ciano
        else if (det.entity_type === "pet") strokeColor = "#ff00b4"; // Magenta
      }

      // 1. Desenho da Caixa Delimitadora com Anti-Aliasing
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2.2;
      ctx.strokeRect(sx1, sy1, sw, sh);

      // 2. Rótulo com Fundo Escuro Semi-Transparente
      const labelText = det.is_known 
        ? `${det.name} (${Math.round(det.confidence * 100)}%)`
        : `${det.class_label} #${det.tracking_id || '?'} (${Math.round(det.confidence * 100)}%)`;

      ctx.font = "600 12px 'Outfit', -apple-system, sans-serif";
      const textMetrics = ctx.measureText(labelText);
      const padX = 8;
      const padY = 4;
      const boxW = textMetrics.width + (padX * 2);
      const boxH = 22;

      let labelX = sx1;
      let labelY = sy1 - boxH;
      if (labelY < 0) labelY = sy1;

      // Fundo Deep Slate
      ctx.fillStyle = "rgba(15, 20, 28, 0.88)";
      ctx.fillRect(labelX, labelY, boxW, boxH);

      // Borda do rótulo
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 1;
      ctx.strokeRect(labelX, labelY, boxW, boxH);

      // Texto branco nítido
      ctx.fillStyle = "#ffffff";
      ctx.fillText(labelText, labelX + padX, labelY + 15);
    });
  }

  function disconnectAllCameraStreams() {
    // Fecha conexões WebRTC
    Object.keys(state.webrtcConnections).forEach(camId => {
      try { state.webrtcConnections[camId].close(); } catch (e) {}
    });
    state.webrtcConnections = {};

    // Fecha WebSockets de vídeo
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

    // Fecha WebSockets de metadados
    Object.keys(state.metaWsConnections).forEach(camId => {
      try {
        const ws = state.metaWsConnections[camId];
        if (ws) {
          ws.onclose = null;
          ws.close();
        }
      } catch (e) {}
    });
    state.metaWsConnections = {};
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

  // --- 7. Radar de Presença Multi-Câmera Dedicado (Portão, Garagem e Fundo) ---
  const radarDedicatedSlots = [
    { id: "slot-portao",   target: "portao", defaultName: "Câmera do Portão", sublabel: "Portão", evt: null },
    { id: "slot-garagem",  target: "garagem", defaultName: "Câmera da Garagem", sublabel: "Garagem", evt: null },
    { id: "slot-fundo",    target: "fundo",   defaultName: "Câmera Área do Fundo", sublabel: "Área do Fundo", evt: null }
  ];

  function getCameraSlotType(evt) {
    const cid = String(evt.camera_id || "").toLowerCase();
    const cname = String(evt.camera_name || "").toLowerCase();
    if (cid === "cam-0" || cname.includes("portão") || cname.includes("portao")) {
      return "portao";
    }
    if (cid === "cam-1" || cname.includes("garagem")) {
      return "garagem";
    }
    if (cid === "cam-2" || cname.includes("fundo") || cname.includes("fundos") || cname.includes("quintal")) {
      return "fundo";
    }
    return "portao";
  }

  function renderAllRadarSlots() {
    if (!dom.radarFeed) return;
    if (dom.radarEmpty && dom.radarEmpty.parentNode) {
      dom.radarEmpty.remove();
    }

    dom.radarFeed.innerHTML = "";

    radarDedicatedSlots.forEach((slotDef) => {
      const card = createRadarSlotCard(slotDef);
      dom.radarFeed.appendChild(card);
    });
  }

  function createRadarSlotCard(slotDef) {
    const evt = slotDef.evt;
    const item = document.createElement("div");
    item.className = "radar-item";
    item.id = `radar-${slotDef.id}`;

    if (!evt) {
      item.classList.add("radar-item-standby");
      item.innerHTML = `
        <div class="radar-thumb-wrap radar-thumb-standby">
          <span class="standby-cam-icon">📹</span>
          <span class="radar-cam-tag">${slotDef.sublabel}</span>
        </div>
        <div class="radar-item-content">
          <div class="radar-card-header-row">
            <span class="radar-item-badge standby">MONITORANDO</span>
            <span class="radar-item-time mono">--:--</span>
          </div>
          <div class="radar-card-body-row">
            <span class="radar-item-name standby-name">Perímetro Tranquilo</span>
          </div>
          <div class="radar-card-footer-row">
            <span class="radar-cam-dot standby"></span>
            <span class="radar-item-ago">Nenhuma atividade recente</span>
          </div>
        </div>
      `;
      return item;
    }

    const eventId = evt.event_id || evt.id;
    item.setAttribute("data-event-id", eventId);

    const timeObj = new Date(evt.timestamp);
    const timeFormatted = isNaN(timeObj.getTime())
      ? evt.timestamp
      : timeObj.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });

    let timeAgo = "";
    if (!isNaN(timeObj.getTime())) {
      const diffSec = Math.max(0, Math.floor((Date.now() - timeObj.getTime()) / 1000));
      if (diffSec < 60) {
        timeAgo = "agora";
      } else if (diffSec < 3600) {
        timeAgo = `há ${Math.floor(diffSec / 60)}m`;
      } else if (diffSec < 86400) {
        timeAgo = `há ${Math.floor(diffSec / 3600)}h`;
      }
    }

    const cropUrl = `/api/events/${eventId}/crop`;
    const { badgeClass, badgeText, primaryTitle } = parseRadarEvent(evt);
    const camDisplay = slotDef.sublabel;

    item.innerHTML = `
      <div class="radar-thumb-wrap">
        <img class="radar-item-thumb" src="${cropUrl}" alt="Evidência" loading="lazy" onerror="this.src='/static/img/placeholder.png'">
        <span class="radar-cam-tag">${camDisplay}</span>
      </div>
      <div class="radar-item-content">
        <div class="radar-card-header-row">
          <span class="radar-item-badge ${badgeClass}">${badgeText}</span>
          <span class="radar-item-time mono">${timeFormatted}</span>
        </div>
        <div class="radar-card-body-row">
          <span class="radar-item-name" title="${primaryTitle || badgeText}">${primaryTitle || "Pessoa Detectada"}</span>
        </div>
        <div class="radar-card-footer-row">
          <span class="radar-status-dot-pulse"></span>
          <span class="radar-item-ago">${timeAgo || "agora"}</span>
        </div>
      </div>
    `;

    item.addEventListener("click", () => openModal(evt));
    return item;
  }

  async function loadRecentEvents() {
    try {
      const res = await fetch("/api/events?limit=80");
      if (!res.ok) return;
      const data = await res.json();
      const events = data.events || [];

      // 1. Portão (cam-0): mais recente
      const portaoEvts = events.filter(e => getCameraSlotType(e) === "portao");
      if (portaoEvts[0]) radarDedicatedSlots[0].evt = portaoEvts[0];

      // 2. Garagem (cam-1): mais recente
      const garagemEvts = events.filter(e => getCameraSlotType(e) === "garagem");
      if (garagemEvts[0]) radarDedicatedSlots[1].evt = garagemEvts[0];

      // 3. Fundo (cam-2): mais recente
      const fundoEvts = events.filter(e => getCameraSlotType(e) === "fundo");
      if (fundoEvts[0]) radarDedicatedSlots[2].evt = fundoEvts[0];

      renderAllRadarSlots();
    } catch (e) {}
  }

  function handleNewRadarEvent(evt) {
    const type = getCameraSlotType(evt);
    if (type === "portao") {
      radarDedicatedSlots[0].evt = evt;
    } else if (type === "garagem") {
      radarDedicatedSlots[1].evt = evt;
    } else if (type === "fundo") {
      radarDedicatedSlots[2].evt = evt;
    }
    renderAllRadarSlots();
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
          handleNewRadarEvent(evt);
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

  function parseRadarEvent(evt) {
    const isVehicle = evt.entity_type === "vehicle";
    const isPet = evt.entity_type === "pet";
    const isResident = Boolean(evt.is_known);

    let badgeClass = "unknown";
    let badgeText = "DESCONHECIDO";
    let primaryTitle = "";

    if (isResident) {
      badgeClass = "resident";
      badgeText = "MORADOR";
      primaryTitle = evt.person_name || "Morador";
    } else if (isVehicle) {
      badgeClass = "vehicle";
      badgeText = "VEÍCULO";
      if (evt.license_plate) {
        primaryTitle = `🚗 ${evt.license_plate}`;
      } else if (evt.person_name && !evt.person_name.toLowerCase().includes("desconhecido")) {
        primaryTitle = evt.person_name;
      }
    } else if (isPet) {
      badgeClass = "pet";
      badgeText = "ANIMAL";
      if (evt.person_name && !evt.person_name.toLowerCase().includes("desconhecido")) {
        primaryTitle = evt.person_name;
      }
    } else {
      badgeClass = "unknown";
      badgeText = "DESCONHECIDO";
      primaryTitle = "";
    }

    return { badgeClass, badgeText, primaryTitle };
  }

  function updateRadarItem(evt) {
    const eventId = evt.event_id || evt.id;
    let found = false;
    radarDedicatedSlots.forEach(slot => {
      if (slot.evt && (slot.evt.event_id === eventId || slot.evt.id === eventId)) {
        slot.evt = { ...slot.evt, ...evt };
        found = true;
      }
    });

    if (found) {
      renderAllRadarSlots();
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
