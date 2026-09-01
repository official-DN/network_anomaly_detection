/* main.js — Chart helpers and auto-refresh for Flask dashboard */

const CHART_DEFAULTS = {
  responsive: true,
  plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } },
  scales: {
    x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: '#1a2a4a' } },
    y: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: '#1a2a4a' } }
  }
}

const COLORS = ['#00d4ff','#ff3366','#ffaa00','#8b5cf6','#00ff88','#f97316','#ec4899']

const instances = {}

function makeChart(id, type, labels, datasets, extra = {}) {
  const el = document.getElementById(id)
  if (!el) return
  if (instances[id]) instances[id].destroy()
  const opts = type === 'pie' || type === 'doughnut'
    ? { responsive: true, plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } } }
    : { ...CHART_DEFAULTS, ...extra }
  instances[id] = new Chart(el, { type, data: { labels, datasets }, options: opts })
}

// ── Format helpers ────────────────────────────────────────
const fmt = n => n == null ? '—' : n >= 1e6 ? `${(n/1e6).toFixed(1)}M` : n >= 1e3 ? `${(n/1e3).toFixed(1)}K` : String(n)
const pct = v => v != null ? `${(v * 100).toFixed(1)}%` : '—'

function scoreBar(val) {
  const p = Math.min(100, Math.round((val || 0) * 100))
  return `<div class="score-wrap">
    <div class="score-bar"><div class="score-fill" style="width:${p}%"></div></div>
    <span class="score-pct">${p}%</span>
  </div>`
}

function tag(val) {
  return `<span class="tag tag-${val}">${val}</span>`
}

// ── Auto-refresh current page every 10 seconds ───────────
document.addEventListener('DOMContentLoaded', () => {
  const page = document.body.dataset.page
  if (page && typeof window[`refresh_${page}`] === 'function') {
    window[`refresh_${page}`]()
    setInterval(() => window[`refresh_${page}`](), 10000)
  }
})
