/**
 * Cerberus AI — Módulo de Relógio CFTV Tático com Borda de Segundos
 * 
 * Componente modular desacoplado.
 * Desenha o relógio digital com mostrador de segundos perimétrico (60 ticks animados),
 * saudação inteligente e calendário.
 */

class CctvClock {
  constructor(containerId = "cctv-clock-widget") {
    this.container = document.getElementById(containerId);
    if (!this.container) return;

    this.meses = [
      "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
      "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
    ];

    this.totalTicks = 60;
    this.render();
    this.start();
  }

  render() {
    this.container.innerHTML = `
      <div class="clock-module-wrap">
        <span class="clock-greeting" id="clock-greeting">BOA TARDE</span>
        
        <div class="clock-frame-container">
          <!-- Borda Perimétrica com 60 Ticks de Segundos -->
          <svg class="clock-ticks-svg" viewBox="0 0 280 150" preserveAspectRatio="none">
            <rect 
              id="clock-track"
              x="3" y="3" width="274" height="144" rx="20" ry="20"
              class="clock-rect-track"
            />
            <rect 
              id="clock-progress"
              x="3" y="3" width="274" height="144" rx="20" ry="20"
              class="clock-rect-progress"
            />
          </svg>

          <!-- Ticks Pontilhados Individuais (60 Segundos) -->
          <div class="clock-seconds-ring" id="clock-seconds-ring"></div>

          <!-- Dígitos Principais da Hora -->
          <div class="clock-digits-box">
            <span class="clock-digits" id="clock-digits">--:--</span>
            <span class="clock-seconds-mini mono" id="clock-seconds-mini">:00</span>
          </div>
        </div>

        <span class="clock-date" id="clock-date">-- de -- de ----</span>
        <span class="cctv-gpu-tag" id="cctv-gpu-tag">GPU: NVIDIA GTX 1660</span>
      </div>
    `;

    this.elGreeting = document.getElementById("clock-greeting");
    this.elDigits = document.getElementById("clock-digits");
    this.elSecMini = document.getElementById("clock-seconds-mini");
    this.elDate = document.getElementById("clock-date");
    this.elProgress = document.getElementById("clock-progress");
    this.elRing = document.getElementById("clock-seconds-ring");

    this.generateTicks();
  }

  generateTicks() {
    if (!this.elRing) return;
    this.elRing.innerHTML = "";

    // Mede perímetro do retângulo para dasharray
    if (this.elProgress) {
      const perimeter = (274 * 2) + (144 * 2) - 16; // Aprox perímetro retangular com raio
      this.perimeter = perimeter;
      this.elProgress.style.strokeDasharray = `${perimeter}`;
      this.elProgress.style.strokeDashoffset = `${perimeter}`;
    }
  }

  start() {
    const tick = () => {
      const now = new Date();
      const h = now.getHours();
      const m = now.getMinutes();
      const s = now.getSeconds();

      // Dígitos da hora
      const hStr = String(h).padStart(2, "0");
      const mStr = String(m).padStart(2, "0");
      const sStr = String(s).padStart(2, "0");

      if (this.elDigits) this.elDigits.textContent = `${hStr}:${mStr}`;
      if (this.elSecMini) this.elSecMini.textContent = `:${sStr}`;

      // Saudação contextual
      if (this.elGreeting) {
        if (h >= 5 && h < 12) this.elGreeting.textContent = "BOM DIA";
        else if (h >= 12 && h < 18) this.elGreeting.textContent = "BOA TARDE";
        else this.elGreeting.textContent = "BOA NOITE";
      }

      // Data formatada
      if (this.elDate) {
        this.elDate.textContent = `${now.getDate()} de ${this.meses[now.getMonth()]} de ${now.getFullYear()}`;
      }

      // Progresso perimétrico dos segundos na borda
      if (this.elProgress && this.perimeter) {
        const fraction = (s + 1) / 60;
        const offset = this.perimeter * (1 - fraction);
        this.elProgress.style.strokeDashoffset = `${offset}`;
      }
    };

    tick();
    setInterval(tick, 1000);
  }
}

// Inicialização automática ao carregar o DOM
document.addEventListener("DOMContentLoaded", () => {
  window.cerberusClock = new CctvClock("cctv-clock-widget");
});
