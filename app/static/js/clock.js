/**
 * Cerberus AI — Módulo de Relógio CFTV Estilo Apple Watch Ultra / StandBy
 * 
 * Engenharia de Mostrador Tático de Alta Fidelidade:
 * - 60 Marcadores de Segundos (Bezel Ticks) individuais de precisão suíça/aeronáutica.
 * - Posicionamento geométrico perfeito via SVG Vector Engine nativo.
 * - Iluminação progressiva com marcador de ponta (leading tick com glow esmeralda).
 * - Tipografia monumental inspirada na linguagem de design da Apple (San Francisco / Modular Ultra).
 * - Componente 100% isolado e desacoplado.
 */

class AppleModularClock {
  constructor(containerId = "cctv-clock-widget") {
    this.container = document.getElementById(containerId);
    if (!this.container) return;

    this.diasSemana = ["DOM", "SEG", "TER", "QUA", "QUI", "SEX", "SÁB"];
    this.meses = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"];

    this.width = 330;
    this.height = 175;
    this.tickElements = [];

    this.render();
    this.buildBezelTicks();
    this.start();
  }

  render() {
    this.container.innerHTML = `
      <div class="apple-clock-card">
        
        <!-- Bezel de 60 Segundos em SVG de Alta Precisão -->
        <svg class="apple-bezel-svg" id="apple-bezel-svg" viewBox="0 0 ${this.width} ${this.height}">
          <!-- Trilha invisível usada para calcular as coordenadas exatas dos 60 ticks -->
          <rect 
            id="apple-bezel-guide"
            x="14" y="14" 
            width="${this.width - 28}" 
            height="${this.height - 28}" 
            rx="22" ry="22"
            fill="none" 
            stroke="none"
          />
          <!-- Grupo onde os 60 marcadores de precisão serão injetados -->
          <g id="apple-ticks-group"></g>
        </svg>

        <!-- Conteúdo Interno do Mostrador -->
        <div class="apple-dial-content">
          
          <!-- Topo: Data Compacta & Status do Sistema -->
          <div class="apple-dial-header">
            <span class="apple-dial-date" id="apple-dial-date">QUA, 21 OUT</span>
            <div class="apple-live-status">
              <span class="apple-status-beacon"></span>
              <span class="apple-status-text" id="apple-greeting-tag">TARDE</span>
            </div>
          </div>

          <!-- Centro: Hora Monumental -->
          <div class="apple-time-display">
            <div class="apple-digits-wrap">
              <span class="apple-digit" id="apple-hours">12</span>
              <span class="apple-colon">:</span>
              <span class="apple-digit" id="apple-minutes">45</span>
            </div>
            <span class="apple-seconds-pill mono" id="apple-seconds-pill">26</span>
          </div>

          <!-- Base: Status do Sistema -->
          <div class="apple-dial-footer">
            <div class="apple-sentry-badge">
              <span class="apple-sentry-dot"></span>
              <span class="apple-sentry-text">24/7 ONLINE</span>
            </div>
            <span class="apple-system-label">CERBERUS SENTRY</span>
          </div>

        </div>

      </div>
    `;

    this.elHours = document.getElementById("apple-hours");
    this.elMinutes = document.getElementById("apple-minutes");
    this.elSecPill = document.getElementById("apple-seconds-pill");
    this.elDate = document.getElementById("apple-dial-date");
    this.elGreeting = document.getElementById("apple-greeting-tag");
    this.ticksGroup = document.getElementById("apple-ticks-group");
  }

  buildBezelTicks() {
    const guide = document.getElementById("apple-bezel-guide");
    if (!guide || !this.ticksGroup) return;

    const totalLen = guide.getTotalLength();
    this.tickElements = [];

    // O segundo 0 começa exatamente no topo central
    // Para um retângulo de 330x200 com raio 22, o topo central fica em L = (largura_topo / 2)
    const topCenterOffset = (this.width - 28) / 2;

    for (let i = 0; i < 60; i++) {
      // Distribui 60 pontos equidistantes
      const frac = i / 60;
      let dist = (frac * totalLen) + topCenterOffset;
      if (dist > totalLen) dist -= totalLen;

      const p1 = guide.getPointAtLength(dist);
      const nextDist = dist + 1 > totalLen ? dist + 1 - totalLen : dist + 1;
      const p2 = guide.getPointAtLength(nextDist);

      // Vetor tangente (dx, dy)
      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      const len = Math.sqrt(dx * dx + dy * dy) || 1;

      // Vetor normal apontando para dentro do display (-dy, dx)
      const nx = -dy / len;
      const ny = dx / len;

      // Cada 5 segundos (0, 5, 10, 15...) é um marcador maior (marcador mestre de 7px)
      // Os intermediários têm 4px
      const isMajor = i % 5 === 0;
      const tickLength = isMajor ? 7.5 : 4.5;
      const strokeWidth = isMajor ? 2.0 : 1.2;

      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", p1.x.toFixed(2));
      line.setAttribute("y1", p1.y.toFixed(2));
      line.setAttribute("x2", (p1.x + nx * tickLength).toFixed(2));
      line.setAttribute("y2", (p1.y + ny * tickLength).toFixed(2));
      line.setAttribute("stroke-width", strokeWidth);
      line.setAttribute("stroke-linecap", "round");
      line.setAttribute("class", `apple-tick ${isMajor ? 'major' : 'minor'}`);

      this.ticksGroup.appendChild(line);
      this.tickElements.push(line);
    }
  }

  start() {
    const update = () => {
      const now = new Date();
      const h = now.getHours();
      const m = now.getMinutes();
      const s = now.getSeconds();

      // Horas e Minutos
      if (this.elHours) this.elHours.textContent = String(h).padStart(2, "0");
      if (this.elMinutes) this.elMinutes.textContent = String(m).padStart(2, "0");

      // Segundos digitais no badge
      if (this.elSecPill) this.elSecPill.textContent = String(s).padStart(2, "0");

      // Data estilo Apple Watch
      if (this.elDate) {
        const diaSem = this.diasSemana[now.getDay()];
        const diaNum = now.getDate();
        const mesStr = this.meses[now.getMonth()];
        this.elDate.textContent = `${diaSem}, ${diaNum} ${mesStr}`;
      }

      // Status de período
      if (this.elGreeting) {
        if (h >= 5 && h < 12) this.elGreeting.textContent = "MANHÃ";
        else if (h >= 12 && h < 18) this.elGreeting.textContent = "TARDE";
        else this.elGreeting.textContent = "NOITE";
      }

      // Atualização dos 60 Marcadores de Segundos (Apple Modular Bezel)
      // Ilumina do segundo 0 até o segundo atual 's'
      for (let i = 0; i < 60; i++) {
        const tick = this.tickElements[i];
        if (!tick) continue;

        if (i <= s) {
          tick.classList.add("active");
          if (i === s) {
            tick.classList.add("head");
          } else {
            tick.classList.remove("head");
          }
        } else {
          tick.classList.remove("active", "head");
        }
      }
    };

    update();
    setInterval(update, 1000);
  }
}

// Inicialização autônoma do módulo
document.addEventListener("DOMContentLoaded", () => {
  window.cerberusClock = new AppleModularClock("cctv-clock-widget");
});
