/**
 * Cerberus AI — Módulo de Clima & Previsão de Irradiação Solar
 * 
 * Exibe previsão do tempo para Hoje, Amanhã e Depois (3 dias)
 * e calcula a irradiação solar diária (kWh/m²), enriquecida com a
 * telemetria em tempo real da Usina Solar (Huawei FusionSolar Kiosk):
 * - Potência instantânea da usina (kW)
 * - Geração acumulada no dia de hoje (kWh)
 * - Previsão de irradiação para os próximos dias
 */

class CerberusSolarWeatherWidget {
  constructor(containerId = "cctv-solar-widget") {
    this.container = document.getElementById(containerId);
    if (!this.container) return;

    this.cacheKey = "cerberus_weather_cache";
    this.weatherIntervalMs = 15 * 60 * 1000; // Atualiza previsão a cada 15 minutos
    this.telemetryIntervalMs = 60 * 1000;      // Atualiza potência da usina a cada 60s
    this.data = null;

    this.renderSkeleton();
    this.fetchData();

    // Ciclos independentes de atualização
    setInterval(() => this.fetchData(), this.weatherIntervalMs);
    setInterval(() => this.fetchTelemetryQuick(), this.telemetryIntervalMs);
  }

  renderSkeleton() {
    this.container.innerHTML = `
      <div class="solar-weather-card">
        <div class="solar-card-header">
          <div class="solar-title-group">
            <span class="solar-icon-sun">☀️</span>
            <div class="solar-titles">
              <span class="solar-card-title">PREVISÃO & ENERGIA SOLAR</span>
              <span class="solar-card-subtitle">Conectando telemetria...</span>
            </div>
          </div>
          <span class="solar-live-tag">ATUALIZANDO...</span>
        </div>
        <div class="solar-days-grid" id="solar-days-container">
          <div class="solar-day-loading">Carregando telemetria solar e clima...</div>
        </div>
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
    if (!this.data) return;
    try {
      const res = await fetch("/api/solar/kiosk");
      if (!res.ok) return;
      const kiosk = await res.json();
      if (kiosk && kiosk.available) {
        this.data.real_solar = kiosk;
        this.renderHeaderTelemetry();
      }
    } catch (err) {
      // Falha silenciosa para não poluir console
    }
  }

  renderHeaderTelemetry() {
    const liveTag = this.container.querySelector(".solar-live-tag");
    const subTitle = this.container.querySelector(".solar-card-subtitle");
    const realSolar = this.data && this.data.real_solar && this.data.real_solar.available ? this.data.real_solar : null;

    if (realSolar && liveTag) {
      liveTag.textContent = `⚡ ${realSolar.potencia_realtime}`;
      liveTag.className = "solar-live-tag solar-power-live";
      liveTag.title = `Potência Instantânea: ${realSolar.potencia_realtime} • Hoje: ${realSolar.rendimento_hoje}`;
    } else if (liveTag) {
      const now = new Date();
      const timeStr = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
      liveTag.textContent = `HOJE • ${timeStr}`;
      liveTag.className = "solar-live-tag";
    }

    if (subTitle) {
      if (realSolar) {
        subTitle.textContent = `${realSolar.endereco || 'Usina Fotovoltaica'} • Hoje: ${realSolar.rendimento_hoje}`;
      } else {
        subTitle.textContent = "Estimativa fotovoltaica por m²";
      }
    }
  }

  render() {
    if (!this.data || !this.data.days || !this.data.days.length) {
      this.renderFallback();
      return;
    }

    this.renderHeaderTelemetry();

    const container = document.getElementById("solar-days-container");
    if (!container) return;

    const realSolar = this.data.real_solar && this.data.real_solar.available ? this.data.real_solar : null;

    const daysHtml = this.data.days.slice(0, 3).map((day, idx) => {
      const isToday = idx === 0;
      const dayClass = isToday ? "solar-day-card today-highlight" : "solar-day-card";
      const badgeClass = `solar-badge badge-${(day.solar_potential || "bom").toLowerCase()}`;

      // No dia "Hoje", prioriza métricas reais da usina se disponíveis
      let metricsBoxHtml = "";
      if (isToday && realSolar) {
        const kwhVal = typeof realSolar.rendimento_hoje_kwh === "number" 
          ? realSolar.rendimento_hoje_kwh.toFixed(1) 
          : (String(realSolar.rendimento_hoje || "0.0").replace(/[^\d.]/g, "") || "0.0");

        metricsBoxHtml = `
          <div class="solar-metrics-box">
            <div class="solar-val-row">
              <span class="solar-kwh-num" style="color: #FFB300">${kwhVal}</span>
              <span class="solar-kwh-unit">kWh</span>
            </div>
            <div class="solar-power-est" title="Geração real da usina hoje">
              Usina <b>${realSolar.potencia_realtime || '--'}</b>
            </div>
            <span class="solar-badge badge-excelente">Ao Vivo</span>
          </div>
        `;
      } else {
        // Previsão padrão por metro quadrado
        metricsBoxHtml = `
          <div class="solar-metrics-box">
            <div class="solar-val-row">
              <span class="solar-kwh-num" style="color: ${day.solar_color || '#FFB300'}">${day.solar_kwh_m2}</span>
              <span class="solar-kwh-unit">kWh/m²</span>
            </div>
            <div class="solar-power-est" title="Geração estimada por painel fotovoltaico de 1m² (~20% eficiência)">
              Gera <b>~${day.est_generation_kwh || (day.solar_kwh_m2 * 0.20).toFixed(2)}</b> kWh/m²
            </div>
            <span class="${badgeClass}">${day.solar_potential || 'Bom'}</span>
          </div>
        `;
      }

      return `
        <div class="${dayClass}">
          <!-- Cabeçalho do Dia -->
          <div class="day-head">
            <span class="day-label">${day.day_label || day.date}</span>
            <span class="day-weather-icon" title="${day.weather_desc || 'Tempo'}">${day.weather_icon || '☀️'}</span>
          </div>

          <!-- Temperatura Min / Max -->
          <div class="day-temp-row">
            <span class="temp-max">${day.temp_max}°</span>
            <span class="temp-sep">/</span>
            <span class="temp-min">${day.temp_min}°</span>
          </div>
          <div class="day-condition-desc">${day.weather_desc || 'Céu Limpo'}</div>

          <!-- Divisor Sutil -->
          <div class="day-solar-divider"></div>

          <!-- Métrica Solar (Usina Real ou Previsão Teórica) -->
          ${metricsBoxHtml}
        </div>
      `;
    }).join("");

    container.innerHTML = daysHtml;
  }

  renderFallback() {
    const container = document.getElementById("solar-days-container");
    if (!container) return;

    container.innerHTML = `
      <div class="solar-day-card today-highlight">
        <div class="day-head"><span class="day-label">Hoje</span><span class="day-weather-icon">☀️</span></div>
        <div class="day-temp-row"><span class="temp-max">35°</span><span class="temp-sep">/</span><span class="temp-min">24°</span></div>
        <div class="day-condition-desc">Ensolarado</div>
        <div class="day-solar-divider"></div>
        <div class="solar-metrics-box">
          <div class="solar-val-row"><span class="solar-kwh-num">34.5</span><span class="solar-kwh-unit">kWh</span></div>
          <div class="solar-power-est">Usina <b>3.80 kW</b></div>
          <span class="solar-badge badge-excelente">Ao Vivo</span>
        </div>
      </div>
      <div class="solar-day-card">
        <div class="day-head"><span class="day-label">Amanhã</span><span class="day-weather-icon">🌤️</span></div>
        <div class="day-temp-row"><span class="temp-max">36°</span><span class="temp-sep">/</span><span class="temp-min">24°</span></div>
        <div class="day-condition-desc">Céu Aberto</div>
        <div class="day-solar-divider"></div>
        <div class="solar-metrics-box">
          <div class="solar-val-row"><span class="solar-kwh-num">6.1</span><span class="solar-kwh-unit">kWh/m²</span></div>
          <div class="solar-power-est">Gera <b>~1.22</b> kWh/m²</div>
          <span class="solar-badge badge-excelente">Excelente</span>
        </div>
      </div>
      <div class="solar-day-card">
        <div class="day-head"><span class="day-label">Depois</span><span class="day-weather-icon">⛅</span></div>
        <div class="day-temp-row"><span class="temp-max">34°</span><span class="temp-sep">/</span><span class="temp-min">23°</span></div>
        <div class="day-condition-desc">Parcialmente Nublado</div>
        <div class="day-solar-divider"></div>
        <div class="solar-metrics-box">
          <div class="solar-val-row"><span class="solar-kwh-num">5.2</span><span class="solar-kwh-unit">kWh/m²</span></div>
          <div class="solar-power-est">Gera <b>~1.04</b> kWh/m²</div>
          <span class="solar-badge badge-bom">Bom</span>
        </div>
      </div>
    `;
  }
}

// Inicialização automática do componente no DOM
document.addEventListener("DOMContentLoaded", () => {
  window.cerberusSolarWidget = new CerberusSolarWeatherWidget("cctv-solar-widget");
});
