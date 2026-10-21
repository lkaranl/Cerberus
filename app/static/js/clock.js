/**
 * Cerberus AI — Módulo de Relógio CFTV Moderno com Borda de 60 Segundos
 * 
 * Componente 100% modular e desacoplado.
 * Renderiza mostrador digital de alta precisão com 60 ticks perimétricos animados em SVG.
 */

class CctvModernClock {
  constructor(containerId = "cctv-clock-widget") {
    this.container = document.getElementById(containerId);
    if (!this.container) return;

    this.diasSemana = [
      "Domingo", "Segunda-feira", "Terça-feira", "Quarta-feira",
      "Quinta-feira", "Sexta-feira", "Sábado"
    ];

    this.meses = [
      "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
      "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
    ];

    // Dimensões do mostrador retangular
    this.width = 330;
    this.height = 170;
    this.radius = 18;
    this.strokeWidth = 3;

    // Perímetro exato do retângulo com cantos arredondados
    // P = 2*(W - 2R) + 2*(H - 2R) + 2*PI*R
    const innerW = this.width - (2 * this.radius);
    const innerH = this.height - (2 * this.radius);
    this.perimeter = (2 * innerW) + (2 * innerH) + (2 * Math.PI * this.radius);

    // 60 ticks: cada tick tem 6px de traço e o restante de espaçamento
    const tickLen = 5.5;
    const gapLen = (this.perimeter / 60) - tickLen;
    this.dashArrayPattern = `${tickLen.toFixed(3)} ${gapLen.toFixed(3)}`;

    this.render();
    this.start();
  }

  render() {
    this.container.innerHTML = `
      <div class="modern-clock-card">
        
        <!-- Header com Saudação & Ícone Dinâmico -->
        <div class="clock-top-bar">
          <div class="clock-greeting-pill">
            <span class="clock-icon" id="clock-period-icon">🌤️</span>
            <span class="clock-greeting-text" id="clock-greeting-text">BOA TARDE</span>
          </div>
          <span class="clock-seconds-counter mono" id="clock-seconds-counter">00s</span>
        </div>

        <!-- Mostrador Central com Borda de 60 Ticks -->
        <div class="clock-display-viewport">
          <svg class="clock-seconds-svg" viewBox="0 0 ${this.width} ${this.height}">
            <!-- Trilha Base com os 60 ticks apagados -->
            <rect 
              x="${this.strokeWidth / 2}" 
              y="${this.strokeWidth / 2}" 
              width="${this.width - this.strokeWidth}" 
              height="${this.height - this.strokeWidth}" 
              rx="${this.radius}" 
              ry="${this.radius}" 
              class="clock-svg-track"
              style="stroke-dasharray: ${this.dashArrayPattern};"
            />
            <!-- Ticks Ativos iluminados até o segundo atual -->
            <rect 
              id="clock-svg-active"
              x="${this.strokeWidth / 2}" 
              y="${this.strokeWidth / 2}" 
              width="${this.width - this.strokeWidth}" 
              height="${this.height - this.strokeWidth}" 
              rx="${this.radius}" 
              ry="${this.radius}" 
              class="clock-svg-active"
              style="stroke-dasharray: ${this.dashArrayPattern};"
            />
          </svg>

          <!-- Dígitos Grandes e Nítidos -->
          <div class="clock-time-display">
            <span class="time-digit mono" id="clock-hours">12</span>
            <span class="time-colon">:</span>
            <span class="time-digit mono" id="clock-minutes">00</span>
          </div>
        </div>

        <!-- Rodapé com Data por Extenso e GPU -->
        <div class="clock-footer-bar">
          <span class="clock-full-date" id="clock-full-date">Carregando data...</span>
          <div class="clock-gpu-pill" id="cctv-gpu-tag">
            <span class="gpu-indicator-dot"></span>
            <span class="gpu-label-text">GTX 1660 • CUDA ⚡</span>
          </div>
        </div>

      </div>
    `;

    this.elIcon = document.getElementById("clock-period-icon");
    this.elGreeting = document.getElementById("clock-greeting-text");
    this.elSecCounter = document.getElementById("clock-seconds-counter");
    this.elHours = document.getElementById("clock-hours");
    this.elMinutes = document.getElementById("clock-minutes");
    this.elFullDate = document.getElementById("clock-full-date");
    this.elActiveRect = document.getElementById("clock-svg-active");
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

      // Contador de segundos
      if (this.elSecCounter) this.elSecCounter.textContent = `${String(s).padStart(2, "0")}s`;

      // Saudação e Ícone de Período
      if (this.elGreeting && this.elIcon) {
        if (h >= 5 && h < 12) {
          this.elGreeting.textContent = "BOM DIA";
          this.elIcon.textContent = "🌅";
        } else if (h >= 12 && h < 18) {
          this.elGreeting.textContent = "BOA TARDE";
          this.elIcon.textContent = "☀️";
        } else {
          this.elGreeting.textContent = "BOA NOITE";
          this.elIcon.textContent = "🌙";
        }
      }

      // Data formatada com dia da semana
      if (this.elFullDate) {
        const diaSemana = this.diasSemana[now.getDay()];
        const diaMes = now.getDate();
        const mes = this.meses[now.getMonth()];
        const ano = now.getFullYear();
        this.elFullDate.textContent = `${diaSemana}, ${diaMes} de ${mes} de ${ano}`;
      }

      // Progresso dos 60 Ticks na borda SVG
      // A cada segundo (0 a 59), ilumina exatamente a fração proporcional de ticks
      if (this.elActiveRect) {
        const tickCount = s === 0 ? 60 : s; // 0s fecha o ciclo completo
        const progressPerimeter = (tickCount / 60) * this.perimeter;
        this.elActiveRect.style.strokeDashoffset = `${this.perimeter - progressPerimeter}`;
      }
    };

    update();
    setInterval(update, 1000);
  }
}

// Inicialização autônoma do módulo
document.addEventListener("DOMContentLoaded", () => {
  window.cerberusClock = new CctvModernClock("cctv-clock-widget");
});
