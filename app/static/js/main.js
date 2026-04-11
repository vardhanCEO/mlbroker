/* ── Shared utilities ──────────────────────────────────────────────────── */

const fmt = {
  money: v => {
    if (v == null) return '—';
    const n = parseFloat(v);
    return (n < 0 ? '-$' : '$') + Math.abs(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  },
  pct: v => {
    if (v == null) return '—';
    const n = parseFloat(v);
    return (n >= 0 ? '+' : '') + n.toFixed(2) + '%';
  },
  num: (v, d = 2) => v == null ? '—' : parseFloat(v).toFixed(d),
};

function colorClass(v) {
  const n = parseFloat(v);
  if (n > 0) return 'positive';
  if (n < 0) return 'negative';
  return 'text-muted';
}

function toast(msg, type = 'info') {
  const c = document.getElementById('toast-container');
  if (!c) return;
  const bg = type === 'success' ? '#3fb950' : type === 'danger' ? '#f85149' : '#58a6ff';
  const div = document.createElement('div');
  div.className = 'toast align-items-center show border-0 mb-2';
  div.style.cssText = `background:${bg}18;border:1px solid ${bg}44!important;color:var(--text);max-width:340px`;
  div.innerHTML = `<div class="d-flex"><div class="toast-body">${msg}</div>
    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button></div>`;
  c.appendChild(div);
  setTimeout(() => div.remove(), 4000);
}

/* ── Market clock in sidebar ───────────────────────────────────────────── */
async function updateClock() {
  try {
    const r = await fetch('/api/clock');
    if (!r.ok) return;
    const d = await r.json();
    const badge = document.getElementById('market-status-badge');
    if (!badge) return;
    const dot  = badge.querySelector('.status-dot');
    const text = badge.querySelector('.status-text');
    if (d.is_open) {
      dot.className  = 'status-dot open';
      text.textContent = 'Market Open';
    } else {
      dot.className  = 'status-dot closed';
      const nextOpen = d.next_open ? new Date(d.next_open).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}) : '';
      text.textContent = nextOpen ? `Closed · Opens ${nextOpen}` : 'Market Closed';
    }
  } catch {}
}

/* ── Account in topbar ─────────────────────────────────────────────────── */
async function updateTopbar() {
  try {
    const r = await fetch('/api/account');
    if (!r.ok) return;
    const d = await r.json();
    const eq  = document.getElementById('header-equity');
    const pnl = document.getElementById('header-pnl');
    if (eq)  { eq.textContent = fmt.money(d.equity || d.portfolio_value); }
    if (pnl) {
      const v = parseFloat(d.equity || 0) - parseFloat(d.last_equity || d.equity || 0);
      pnl.textContent  = fmt.money(v);
      pnl.className = 'stat-val ' + colorClass(v);
    }
  } catch {}
}

/* ── Init on every page ────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  updateClock();
  updateTopbar();
  initTickerStrip('ticker-bar');
  setInterval(updateClock,  30_000);
  setInterval(updateTopbar, 60_000);

  // auto-dismiss alerts
  document.querySelectorAll('.alert').forEach(el => setTimeout(() => el.remove(), 5000));
});

/* ── Ticker strip initialiser ──────────────────────────────────────────── */

const _TICKER_SYMS = [
  'BTC/USD','ETH/USD','SOL/USD','BNB/USD',
  'AVAX/USD','DOGE/USD','XRP/USD','LINK/USD',
];

function initTickerStrip(containerId) {
  const wrap = document.getElementById(containerId);
  if (!wrap) return;

  function buildItems() {
    return _TICKER_SYMS.map(s =>
      `<span class="ticker-item" id="tk-${s.replace('/','')}" data-sym="${s}">
        <span class="ticker-sym">${s.replace('/USD','')}</span>
        <span class="ticker-price">—</span>
        <span class="ticker-chg text-muted">—</span>
      </span>`
    ).join('');
  }

  const track = wrap.querySelector('.ticker-track') || document.createElement('div');
  track.className = 'ticker-track';
  track.innerHTML = buildItems() + buildItems();   // duplicate for seamless loop
  wrap.innerHTML  = '';
  wrap.appendChild(track);

  function fetchTickers() {
    fetch('/api/tickers')
      .then(r => r.json())
      .then(list => list.forEach(t => _updateTickerItem(t.symbol, t.price, t.change_pct)))
      .catch(() => {});
  }

  fetchTickers();
  setInterval(fetchTickers, 15_000);
}

function _updateTickerItem(sym, price, pct) {
  document.querySelectorAll(`[id="tk-${sym.replace('/','')}"`)
    .forEach(el => {
      el.querySelector('.ticker-price').textContent = fmt.money(price);
      const chg = el.querySelector('.ticker-chg');
      chg.textContent  = (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%';
      chg.className    = 'ticker-chg ' + (pct >= 0 ? 'positive' : 'negative');
    });
}

/* ── Lightweight Charts helper ─────────────────────────────────────────── */
function makeChart(container, height = 420) {
  if (!window.LightweightCharts) return null;
  const chart = LightweightCharts.createChart(container, {
    width:  container.clientWidth,
    height,
    layout: {
      background:  { color: '#1c2128' },
      textColor:   '#7d8590',
      fontFamily:  'Inter, sans-serif',
      fontSize:    11,
    },
    grid: {
      vertLines: { color: '#21262d' },
      horzLines: { color: '#21262d' },
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#30363d' },
    timeScale:       { borderColor: '#30363d', timeVisible: true, secondsVisible: false },
    handleScroll:    { vertTouchDrag: false },
  });

  // responsive resize
  const ro = new ResizeObserver(() => {
    chart.applyOptions({ width: container.clientWidth });
  });
  ro.observe(container);

  return chart;
}

function addCandles(chart) {
  return chart.addCandlestickSeries({
    upColor:        '#3fb950',
    downColor:      '#f85149',
    borderUpColor:  '#3fb950',
    borderDownColor:'#f85149',
    wickUpColor:    '#3fb950',
    wickDownColor:  '#f85149',
  });
}

function addVolume(chart) {
  const s = chart.addHistogramSeries({
    priceFormat:  { type: 'volume' },
    priceScaleId: 'vol',
  });
  chart.priceScale('vol').applyOptions({
    scaleMargins: { top: 0.75, bottom: 0 },
  });
  return s;
}

function signalMarkers(signals) {
  return signals.map(s => ({
    time:     s.time,
    position: s.action === 'buy' ? 'belowBar' : 'aboveBar',
    color:    s.action === 'buy' ? '#3fb950'  : '#f85149',
    shape:    s.action === 'buy' ? 'arrowUp'  : 'arrowDown',
    text:     s.action.toUpperCase(),
  }));
}
