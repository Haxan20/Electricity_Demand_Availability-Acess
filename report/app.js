// Grid Pulse report -- driven entirely by REPORT_DATA (see data.js), which
// is a real export of the project's engineered dataset + trained model
// metrics. Nothing here is placeholder/fake data.

document.addEventListener('DOMContentLoaded', () => {
  const D = REPORT_DATA;

  // ---------------- Chart.js dark theme defaults ----------------
  Chart.defaults.color = '#9aa4b2';
  Chart.defaults.borderColor = '#262d3a';
  Chart.defaults.font.family = "'Inter', sans-serif";

  const ACCENT = '#f5a623';
  const GREEN = '#3ecf8e';
  const YELLOW = '#e8c547';
  const ORANGE = '#f5a623';
  const RED = '#ef5b5b';

  // ---------------- Count-up stat animation ----------------
  document.querySelectorAll('.stat-value').forEach(el => {
    const target = parseInt(el.dataset.count, 10);
    const suffix = el.dataset.suffix || '';
    const duration = 1200;
    const start = performance.now();
    function tick(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const value = Math.floor(eased * target);
      el.textContent = value.toLocaleString() + suffix;
      if (progress < 1) requestAnimationFrame(tick);
      else el.textContent = target.toLocaleString() + suffix;
    }
    requestAnimationFrame(tick);
  });

  const loadingTag = document.getElementById('loading-tag');
  if (loadingTag) {
    setTimeout(() => {
      loadingTag.textContent = `${D.overview.n_feeders} feeders · ${D.overview.n_addresses.toLocaleString()} addresses loaded`;
    }, 1300);
  }

  // ---------------- 01 Monthly trend ----------------
  const months = D.monthly_trend.map(r => r.MONTH);
  new Chart(document.getElementById('chart-monthly'), {
    type: 'line',
    data: {
      labels: months,
      datasets: [
        {
          label: 'Avg hours/day',
          data: D.monthly_trend.map(r => r.avg_hours),
          borderColor: ACCENT,
          backgroundColor: 'rgba(245,166,35,0.12)',
          fill: true,
          tension: 0.35,
          yAxisID: 'y',
        },
        {
          label: 'Avg shortfall (h)',
          data: D.monthly_trend.map(r => r.avg_shortfall),
          borderColor: RED,
          backgroundColor: 'transparent',
          borderDash: [4, 3],
          tension: 0.35,
          yAxisID: 'y',
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { usePointStyle: true } } },
      scales: {
        y: { title: { display: true, text: 'Hours' }, grid: { color: '#1c222e' } },
        x: { grid: { display: false } },
      },
    },
  });

  // ---------------- 02 Day of week rhythm ----------------
  new Chart(document.getElementById('chart-dow'), {
    type: 'bar',
    data: {
      labels: D.day_of_week_rhythm.map(r => r.DAY_OF_WEEK),
      datasets: [{
        label: 'Avg hours/day',
        data: D.day_of_week_rhythm.map(r => r.avg_hours),
        backgroundColor: ACCENT,
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { min: 0, max: 20, grid: { color: '#1c222e' } },
        x: { grid: { display: false } },
      },
    },
  });

  // ---------------- 03 Reliability distribution ----------------
  const colorCounts = D.reliability_color_counts;
  new Chart(document.getElementById('chart-reliability'), {
    type: 'doughnut',
    data: {
      labels: ['Good (\u226518h)', 'Fair (12-18h)', 'Poor (6-12h)', 'Critical (<6h)'],
      datasets: [{
        data: [colorCounts.green || 0, colorCounts.yellow || 0, colorCounts.orange || 0, colorCounts.red || 0],
        backgroundColor: [GREEN, YELLOW, ORANGE, RED],
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, padding: 14 } } },
    },
  });

  // ---------------- 03 Band distribution ----------------
  const bandEntries = Object.entries(D.band_distribution).sort((a, b) => b[0] - a[0]);
  new Chart(document.getElementById('chart-band'), {
    type: 'bar',
    data: {
      labels: bandEntries.map(([band]) => `${band}h band`),
      datasets: [{
        label: 'Feeders',
        data: bandEntries.map(([, count]) => count),
        backgroundColor: '#5b8def',
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: '#1c222e' } },
        y: { grid: { display: false } },
      },
    },
  });

  // ---------------- 04 Leaflet map ----------------
  const colorHex = { green: GREEN, yellow: YELLOW, orange: ORANGE, red: RED, gray: '#666' };
  const map = L.map('leaflet-map', { scrollWheelZoom: false }).setView([6.55, 3.38], 10);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
    subdomains: 'abcd',
    maxZoom: 19,
  }).addTo(map);

  D.feeders_geo.forEach(f => {
    const marker = L.circleMarker([f.latitude, f.longitude], {
      radius: 5,
      color: colorHex[f.color] || '#666',
      fillColor: colorHex[f.color] || '#666',
      fillOpacity: 0.85,
      weight: 1,
    }).addTo(map);
    marker.bindPopup(
      `<b>${f.FEEDER_NAME}</b><br>` +
      `Avg hours/day: ${f.avg_actual_hours.toFixed(1)}h<br>` +
      `Band: ${f.band.toFixed(0)}h max<br>` +
      `Availability vs band: ${f.avg_availability_pct.toFixed(1)}%<br>` +
      `Addresses: ${f.n_addresses}`
    );
  });

  // ---------------- 05 Model metrics ----------------
  const mm = D.model_metrics;
  function setMetric(id, val, digits = 2) {
    const el = document.getElementById(id);
    if (el) el.textContent = typeof val === 'number' ? val.toFixed(digits) : val;
  }
  setMetric('m-train-mae', mm.train.mae); setMetric('m-train-rmse', mm.train.rmse); setMetric('m-train-r2', mm.train.r2, 3);
  setMetric('m-val-mae', mm.val.mae); setMetric('m-val-rmse', mm.val.rmse); setMetric('m-val-r2', mm.val.r2, 3);
  setMetric('m-test-mae', mm.test.mae); setMetric('m-test-rmse', mm.test.rmse); setMetric('m-test-r2', mm.test.r2, 3);

  // ---------------- 05 Feature importance ----------------
  const imp = D.feature_importance; // [[name, value], ...]
  new Chart(document.getElementById('chart-importance'), {
    type: 'bar',
    data: {
      labels: imp.map(([name]) => name),
      datasets: [{
        label: 'Relative importance',
        data: imp.map(([, val]) => val),
        backgroundColor: ACCENT,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: '#1c222e' } },
        y: { grid: { display: false } },
      },
    },
  });

  // ---------------- 06 Best / worst tables ----------------
  function renderTable(elId, rows) {
    const el = document.getElementById(elId);
    let html = '<thead><tr><th>Feeder</th><th>Avg hrs/day</th><th>Band</th><th>Addresses</th></tr></thead><tbody>';
    rows.forEach(r => {
      html += `<tr><td>${r.FEEDER_NAME}</td><td>${r.avg_actual_hours.toFixed(1)}h</td><td>${r.band.toFixed(0)}h</td><td>${r.n_addresses}</td></tr>`;
    });
    html += '</tbody>';
    el.innerHTML = html;
  }
  renderTable('table-worst', D.worst5);
  renderTable('table-best', D.best5);

  // ---------------- Scroll-spy nav ----------------
  const navLinks = document.querySelectorAll('#side-nav ul a[href^="#"]');
  const sections = Array.from(navLinks).map(a => document.querySelector(a.getAttribute('href')));
  function onScroll() {
    let current = sections[0];
    const scrollPos = window.scrollY + 140;
    sections.forEach(sec => {
      if (sec && sec.offsetTop <= scrollPos) current = sec;
    });
    navLinks.forEach(a => {
      a.classList.toggle('active', a.getAttribute('href') === `#${current.id}`);
    });
  }
  window.addEventListener('scroll', onScroll);
  onScroll();

  // ---------------- Live app link ----------------
  // Point this at wherever the Streamlit app ends up deployed (Streamlit
  // Community Cloud / Hugging Face Spaces / Render). Left as a placeholder
  // since deployment target wasn't specified.
  const liveLink = document.getElementById('live-app-link');
  if (liveLink) liveLink.href = 'https://your-streamlit-app-url-here';
});
