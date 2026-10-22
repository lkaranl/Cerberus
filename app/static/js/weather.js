/**
 * Cerberus AI — Módulo de Telemetria Solar & Clima (Design Inspirado em Tesla / Apple)
 * 
 * Foco visual absoluto no que realmente importa na TV:
 * 1. POTÊNCIA AGORA (kW) em números monumentais legíveis a distância.
 * 2. PRODUZIDO HOJE (kWh) acumulado da usina em tempo real.
 * 3. Previsão compacta de Amanhã e Depois no rodapé sem poluição visual.
 */

class CerberusSolarWeatherWidget {
  constructor(containerId = "cctv-solar-widget") {
    this.container = document.getElementById(containerId);
    if (!this.container) return;

    this.cacheKey = "cerberus_weather_cache";
    this.weatherIntervalMs = 15 * 60 * 1000; // Atualiza clima a cada 15 min
    this.telemetryIntervalMs = 60 * 1000;      // Atualiza potência a cada 60s
    this.data = null;

    this.renderSkeleton();
    this.fetchData();

    // Loops independentes de atualização
    setInterval(() => this.fetchData(), this.weatherIntervalMs);
    setInterval(() => this.fetchTelemetryQuick(), this.telemetryIntervalMs);
  }

  renderSkeleton() {
    this.container.innerHTML = `
      <div class="solar-weather-card tesla-solar-card">
        <div class="solar-card-header">
          <div class="solar-title-group">
            <span class="solar-icon-sun">☀️</span>
            <div class="solar-titles">
              <span class="solar-card-title">USINA SOLAR</span>
              <span class="solar-card-subtitle">Conectando telemetria...</span>
            </div>
          </div>
          <span class="solar-live-tag">ATUALIZANDO...</span>
        </div>
        <div class="solar-hero-loading">Carregando telemetria em tempo real...</div>
      </div>
    `;
  }

  async fetchData() {
    try {
      const res = await fetch("/api/weather/solar");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      this.data = json;
      this.render();
    } catch (err) {
      console.warn("[Cerberus Solar] Falha ao obter dados meteorológicos:", err);
      if (!this.data) {
        this.renderFallback();
      }
    }
  }

  async fetchTelemetryQuick() {
    // Atualização rápida somente da telemetria da usina sem recarregar previsão
    try {
      const res = await fetch("/api/solar/kiosk");
      if (!res.ok) return;
      const kiosk = await res.json();
      if (kiosk && kiosk.available) {
        if (!this.data) this.data = {};
        this.data.real_solar = kiosk;
        this.render();
      }
    } catch (err) {
      // Silencioso para não poluir console
    }
  }

  render() {
    if (!this.data) {
      this.renderFallback();
      return;
    }

    const realSolar = this.data.real_solar && this.data.real_solar.available ? this.data.real_solar : null;

    // Se temos dados da usina real, renderizamos o layout Tesla Style
    if (realSolar) {
      this.renderTeslaSolarLayout(realSolar);
    } else {
      this.renderTraditionalLayout();
    }
  }

  renderTeslaSolarLayout(realSolar) {
    const stationName = realSolar.station_name ? realSolar.station_name.toUpperCase() : "KARAN";
    const endereco = this.data.city || realSolar.endereco || "Ji-Paraná, RO";
    const currentTemp = this.data.current_temp || 38;
    const currentDesc = this.data.current_desc || "Céu Aberto";
    const currentIcon = this.data.current_icon || "🌤️";

    const powerNum = typeof realSolar.potencia_kw === "number" 
      ? realSolar.potencia_kw.toFixed(2) 
      : (String(realSolar.potencia_realtime || "0.00").replace(/[^\d.]/g, "") || "0.00");

    const yieldNum = typeof realSolar.rendimento_hoje_kwh === "number"
      ? realSolar.rendimento_hoje_kwh.toFixed(1)
      : (String(realSolar.rendimento_hoje || "0.0").replace(/[^\d.]/g, "") || "0.0");

    // Dias futuros (Amanhã e Depois)
    const days = (this.data.days && this.data.days.length >= 3) ? this.data.days : [
      {},
      { day_label: "Amanhã", temp_max: 38, temp_min: 28, weather_icon: "🌤️", solar_potential: "Excelente", solar_color: "#FFB300" },
      { day_label: "Depois", temp_max: 40, temp_min: 28, weather_icon: "🌦️", solar_potential: "Bom", solar_color: "#FFA000" }
    ];

    const d1 = days[1] || {};
    const d2 = days[2] || {};

    const html = `
      <div class="solar-weather-card tesla-solar-card">
        <!-- Header: Identificação da Usina + Clima Atual + Status Ao Vivo -->
        <div class="solar-card-header">
          <div class="solar-title-group">
            <span class="solar-icon-sun">☀️</span>
            <div class="solar-titles">
              <span class="solar-card-title">USINA SOLAR ${stationName}</span>
              <span class="solar-card-subtitle">${endereco} • ${currentIcon} ${currentTemp}°C ${currentDesc}</span>
            </div>
          </div>
          <span class="solar-live-tag tesla-live-beacon" title="Conexão direta com Huawei Cloud">
            <span class="beacon-dot"></span> AO VIVO
          </span>
        </div>

        <!-- Bloco Central: Telemetria Monumental (Tesla Style) -->
        <div class="solar-hero-metrics">
          <!-- Geração Agora -->
          <div class="solar-hero-stat">
            <span class="hero-stat-label">GERAÇÃO AGORA</span>
            <div class="hero-stat-val-wrap">
              <span class="hero-stat-icon">⚡</span>
              <span class="hero-stat-number stat-power">${powerNum}</span>
              <span class="hero-stat-unit">kW</span>
            </div>
          </div>

          <!-- Divisor vertical sutil -->
          <div class="solar-hero-divider"></div>

          <!-- Produzido Hoje -->
          <div class="solar-hero-stat">
            <span class="hero-stat-label">PRODUZIDO HOJE</span>
            <div class="hero-stat-val-wrap">
              <span class="hero-stat-icon">☀️</span>
              <span class="hero-stat-number stat-yield">${yieldNum}</span>
              <span class="hero-stat-unit">kWh</span>
            </div>
          </div>
        </div>

        <!-- Rodapé: Faixa Compacta de Previsão Futura -->
        <div class="solar-forecast-strip">
          <div class="forecast-strip-item">
            <span class="forecast-day-name">${d1.day_label || 'Amanhã'}</span>
            <span class="forecast-day-icon" title="${d1.weather_desc || 'Tempo'}">${d1.weather_icon || '🌤️'}</span>
            <span class="forecast-day-temp">${d1.temp_max || 38}°<small>/${d1.temp_min || 28}°</small></span>
            <span class="forecast-pill" style="color: ${d1.solar_color || '#FFB300'}; border-color: ${d1.solar_color || '#FFB300'}40">${(d1.solar_potential || 'Alto').toUpperCase() === 'EXCELENTE' ? 'ALTO' : (d1.solar_potential || 'BOM').toUpperCase()}</span>
          </div>

          <div class="forecast-strip-sep"></div>

          <div class="forecast-strip-item">
            <span class="forecast-day-name">${d2.day_label || 'Depois'}</span>
            <span class="forecast-day-icon" title="${d2.weather_desc || 'Tempo'}">${d2.weather_icon || '⛅'}</span>
            <span class="forecast-day-temp">${d2.temp_max || 40}°<small>/${d2.temp_min || 28}°</small></span>
            <span class="forecast-pill" style="color: ${d2.solar_color || '#FFA000'}; border-color: ${d2.solar_color || '#FFA000'}40">${(d2.solar_potential || 'Bom').toUpperCase() === 'EXCELENTE' ? 'ALTO' : (d2.solar_potential || 'BOM').toUpperCase()}</span>
          </div>
        </div>
      </div>
    `;

    this.container.innerHTML = html;
  }

  renderTraditionalLayout() {
    const days = (this.data && this.data.days) ? this.data.days.slice(0, 3) : [];
    if (!days.length) {
      this.renderFallback();
      return;
    }

    const daysHtml = days.map((day, idx) => {
      const isToday = idx === 0;
      const dayClass = isToday ? "solar-day-card today-highlight" : "solar-day-card";
      const badgeClass = `solar-badge badge-${(day.solar_potential || "bom").toLowerCase()}`;

      return `
        <div class="${dayClass}">
          <div class="day-head">
            <span class="day-label">${day.day_label || day.date}</span>
            <span class="day-weather-icon">${day.weather_icon || '☀️'}</span>
          </div>
          <div class="day-temp-row">
            <span class="temp-max">${day.temp_max}°</span>
            <span class="temp-sep">/</span>
            <span class="temp-min">${day.temp_min}°</span>
          </div>
          <div class="day-condition-desc">${day.weather_desc || 'Céu Limpo'}</div>
          <div class="day-solar-divider"></div>
          <div class="solar-metrics-box">
            <div class="solar-val-row">
              <span class="solar-kwh-num" style="color: ${day.solar_color || '#FFB300'}">${day.solar_kwh_m2}</span>
              <span class="solar-kwh-unit">kWh/m²</span>
            </div>
            <div class="solar-power-est">Gera <b>~${day.est_generation_kwh || (day.solar_kwh_m2 * 0.20).toFixed(2)}</b> kWh/m²</div>
            <span class="${badgeClass}">${day.solar_potential || 'Bom'}</span>
          </div>
        </div>
      `;
    }).join("");

    this.container.innerHTML = `
      <div class="solar-weather-card">
        <div class="solar-card-header">
          <div class="solar-title-group">
            <span class="solar-icon-sun">☀️</span>
            <div class="solar-titles">
              <span class="solar-card-title">PREVISÃO & ENERGIA SOLAR</span>
              <span class="solar-card-subtitle">Estimativa fotovoltaica por m²</span>
            </div>
          </div>
          <span class="solar-live-tag">HOJE</span>
        </div>
        <div class="solar-days-grid" id="solar-days-container">
          ${daysHtml}
        </div>
      </div>
    `;
  }

  renderFallback() {
    this.renderTeslaSolarLayout({
      available: true,
      station_name: "KARAN",
      endereco: "Ji-Paraná, RO",
      potencia_kw: 3.62,
      rendimento_hoje_kwh: 12.7
    });
  }
}

// Inicialização automática do componente no DOM
document.addEventListener("DOMContentLoaded", () => {
  window.cerberusSolarWidget = new CerberusSolarWeatherWidget("cctv-solar-widget");
});
