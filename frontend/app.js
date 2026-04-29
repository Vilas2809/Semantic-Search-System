const API = 'http://localhost:8000';
let devMode = false;
let attrId = 0;

// ── Dev Mode ──────────────────────────────────────────────
document.getElementById('devmode-toggle').addEventListener('click', () => {
  devMode = !devMode;
  document.body.classList.toggle('devmode', devMode);
  const btn  = document.getElementById('devmode-toggle');
  const pill = document.getElementById('devmode-pill');
  btn.classList.toggle('active', devMode);
  pill.textContent = devMode ? 'ON' : 'OFF';
  pill.className   = devMode ? 'pill pill-on' : 'pill pill-off';
});

// ── Tab navigation ────────────────────────────────────────
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
    if (btn.dataset.tab === 'stats') loadStats();
  });
});

// ── Health check ──────────────────────────────────────────
async function checkHealth() {
  const dot   = document.getElementById('status-dot');
  const label = document.getElementById('status-label');
  try {
    const res = await fetch(`${API}/health`);
    if (res.ok) {
      dot.className = 'dot dot-ok';
      label.textContent = 'API online';
    } else throw new Error();
  } catch {
    dot.className = 'dot dot-error';
    label.textContent = 'API offline';
  }
}
checkHealth();
setInterval(checkHealth, 10000);

// ── Pipeline animation ────────────────────────────────────
function setPipelineActive(active) {
  document.getElementById('pipeline')?.classList.toggle('active', active);
}

// ── Utilities ─────────────────────────────────────────────
function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

const STOPS = new Set([
  'is','the','a','an','and','or','but','in','on','at','to','for','of',
  'with','by','what','how','why','when','where','who','which','does',
  'do','did','are','was','were','be','been','have','has','had','will',
  'would','could','should','may','might','can','it','its','this','that',
]);

function queryTerms(q) {
  return [...new Set(
    q.toLowerCase().split(/\W+/)
     .filter(t => t.length > 2 && !STOPS.has(t))
  )].sort((a, b) => b.length - a.length);
}

function highlight(escapedText, query) {
  let t = escapedText;
  for (const term of queryTerms(query)) {
    t = t.replace(new RegExp(`(${term})`, 'gi'), '<mark>$1</mark>');
  }
  return t;
}

// SVG circular match indicator
function matchCircle(pct) {
  const r  = 15;
  const cx = 20, cy = 20;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - Math.min(pct, 100) / 100);
  const color = pct >= 65 ? 'var(--green)' : pct >= 40 ? 'var(--yellow)' : 'var(--red)';
  return `<svg class="match-circle" width="40" height="40" viewBox="0 0 40 40">
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="2.5"/>
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${color}" stroke-width="2.5"
      stroke-dasharray="${circ.toFixed(2)}" stroke-dashoffset="${offset.toFixed(2)}"
      stroke-linecap="round" transform="rotate(-90 ${cx} ${cy})"
      style="transition:stroke-dashoffset 0.9s cubic-bezier(.4,0,.2,1)"/>
    <text x="${cx}" y="${cy}" text-anchor="middle" dominant-baseline="central"
      fill="${color}" font-size="8.5" font-weight="700" font-family="Inter,system-ui">${pct}%</text>
  </svg>`;
}

// Latency bar HTML
function latencyBar(lat) {
  const segs = [
    ['Embed',  lat.embedding_ms],
    ['BM25',   lat.bm25_ms],
    ['Vector', lat.vector_ms],
    ['Fusion', lat.fusion_ms],
    ['Rerank', lat.reranking_ms],
  ];
  const segHtml = segs.map(([label, ms]) =>
    `<span class="lat-seg"><span class="lat-label">${label}</span><span class="lat-val">${ms}ms</span></span>`
  ).join('<span class="lat-div">·</span>');
  return `<span class="lat-header">⚡ Latency</span>${segHtml}<span class="lat-total">Total ${lat.total_ms}ms</span>`;
}

// Skeleton loading
function skeletonResults(n = 3) {
  return Array.from({ length: n }, () => `
    <div class="skeleton-card">
      <div class="skeleton-header">
        <div style="flex:1"><div class="skeleton-line short"></div></div>
        <div class="skeleton-circle"></div>
      </div>
      <div class="skeleton-line"></div>
      <div class="skeleton-line medium"></div>
      <div class="skeleton-line short"></div>
    </div>`).join('');
}

// Build a full result card
function resultCard(r, index, query, delay = 0) {
  const source   = r.metadata?.filename || r.metadata?.doc_id || r.chunk_id;
  const chunkIdx = r.metadata?.chunk_index !== undefined ? `Section ${parseInt(r.metadata.chunk_index) + 1}` : '';
  const docId    = r.metadata?.doc_id || '—';
  const pct      = r.match_pct || 0;
  const tier     = pct >= 65 ? 'high' : pct >= 40 ? 'medium' : 'low';
  const id       = `attr-${attrId++}`;

  const preview = r.content.length > 300 ? r.content.slice(0, 300) + '…' : r.content;
  const previewHtml = highlight(esc(preview), query);
  const fullHtml    = highlight(esc(r.content), query);

  return `
    <div class="result-card ${tier}" style="animation-delay:${delay}ms">
      <div class="result-header">
        <span class="result-rank">#${index}</span>
        <span class="result-source">${esc(source)}</span>
        <span class="result-method-badge">${esc(r.retrieval_method)}</span>
        ${matchCircle(pct)}
      </div>
      <div class="result-body">${previewHtml}</div>

      <button class="attr-toggle" onclick="toggleAttr('${id}', this)">
        <span class="attr-arrow">▶</span>
        Source Attribution
        ${chunkIdx ? `<span style="margin-left:auto;font-size:10.5px;font-family:var(--mono);color:var(--dim)">${esc(chunkIdx)}</span>` : ''}
      </button>
      <div id="${id}" class="attr-panel">
        <div class="attr-inner">
          <div class="attr-meta">
            <span class="attr-meta-item">📄 ${esc(source)}</span>
            ${chunkIdx ? `<span class="attr-meta-item">📌 ${esc(chunkIdx)}</span>` : ''}
            <span class="attr-meta-item">🗂 ${esc(docId)}</span>
          </div>
          <div class="attr-full-text">${fullHtml}</div>
        </div>
      </div>

      <div class="dev-row">
        <span class="dev-kv"><span class="dev-k">rrf_score</span><span class="dev-v">${r.score}</span></span>
        <span class="dev-kv"><span class="dev-k">rerank</span><span class="dev-v">${r.rerank_score ?? '—'}</span></span>
        <span class="dev-kv"><span class="dev-k">match_pct</span><span class="dev-v">${pct}%</span></span>
        <span class="dev-kv"><span class="dev-k">chunk_id</span><span class="dev-v">${esc(r.chunk_id)}</span></span>
      </div>
    </div>`;
}

function toggleAttr(id, btn) {
  const panel = document.getElementById(id);
  panel.classList.toggle('open');
  btn.classList.toggle('open');
}

function renderResults(results, query, container) {
  if (!results?.length) {
    container.innerHTML = `<div class="empty-state">
      <div class="empty-state-icon">◌</div>
      <p>No results found. Try indexing some documents first.</p>
    </div>`;
    return;
  }
  container.innerHTML = `<div class="results-list">${
    results.map((r, i) => resultCard(r, i + 1, query, i * 55)).join('')
  }</div>`;
}

function showLatency(lat, bar) {
  if (!devMode || !lat) return;
  bar.innerHTML = latencyBar(lat);
  bar.style.display = 'flex';
}

// ── SEARCH ────────────────────────────────────────────────
document.getElementById('search-btn').addEventListener('click', runSearch);
document.getElementById('search-input').addEventListener('keydown', e => { if (e.key === 'Enter') runSearch(); });

async function runSearch() {
  const query  = document.getElementById('search-input').value.trim();
  const top_k  = parseInt(document.getElementById('search-topk').value);
  const out    = document.getElementById('search-results');
  const latBar = document.getElementById('search-latency');
  if (!query) return;

  setPipelineActive(true);
  latBar.style.display = 'none';
  out.innerHTML = skeletonResults(top_k);

  try {
    const res  = await fetch(`${API}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k }),
    });
    const data = await res.json();
    showLatency(data.latency, latBar);
    renderResults(data.results, query, out);
  } catch {
    setPipelineActive(false);
    out.innerHTML = `<div class="msg msg-error">Could not reach the API — is the backend running?</div>`;
  }
}

// ── ASK ───────────────────────────────────────────────────
document.getElementById('ask-btn').addEventListener('click', runAsk);
document.getElementById('ask-input').addEventListener('keydown', e => { if (e.key === 'Enter') runAsk(); });

async function runAsk() {
  const query  = document.getElementById('ask-input').value.trim();
  const top_k  = parseInt(document.getElementById('ask-topk').value);
  const out    = document.getElementById('ask-result');
  const latBar = document.getElementById('ask-latency');
  if (!query) return;

  latBar.style.display = 'none';
  out.innerHTML = `<div class="msg msg-loading"><span class="loader"></span>Retrieving context and generating answer with Groq…</div>`;

  try {
    const res  = await fetch(`${API}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k }),
    });
    const data = await res.json();
    showLatency(data.latency, latBar);

    const sourcesHtml = data.sources?.length
      ? `<div class="sources-label">Sources Used (${data.sources.length})</div>
         <div class="results-list">${data.sources.map((r, i) => resultCard(r, i + 1, query, i * 55)).join('')}</div>`
      : '';

    out.innerHTML = `
      <div class="answer-wrapper">
        <div class="answer-header">
          <span class="answer-header-label">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            Answer
          </span>
          <span class="answer-model">${esc(data.model)}</span>
        </div>
        <div class="answer-body">${esc(data.answer)}</div>
      </div>
      ${sourcesHtml}`;
  } catch {
    out.innerHTML = `<div class="msg msg-error">Could not reach the API — is the backend running?</div>`;
  }
}

// ── INDEX — file ──────────────────────────────────────────
const fileInput = document.getElementById('file-input');
const fileDrop  = document.getElementById('file-drop');
const fileName  = document.getElementById('file-name');
const uploadBtn = document.getElementById('upload-btn');

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) {
    fileName.textContent = fileInput.files[0].name;
    uploadBtn.disabled = false;
  }
});
fileDrop.addEventListener('click', () => fileInput.click());
fileDrop.addEventListener('dragover', e => { e.preventDefault(); fileDrop.classList.add('drag-over'); });
fileDrop.addEventListener('dragleave', () => fileDrop.classList.remove('drag-over'));
fileDrop.addEventListener('drop', e => {
  e.preventDefault();
  fileDrop.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) { fileInput.files = e.dataTransfer.files; fileName.textContent = file.name; uploadBtn.disabled = false; }
});

uploadBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  const out  = document.getElementById('upload-result');
  if (!file) return;
  out.innerHTML = `<div class="msg msg-loading"><span class="loader"></span>Indexing into BM25 + vector store…</div>`;
  uploadBtn.disabled = true;
  const form = new FormData();
  form.append('file', file);
  try {
    const res  = await fetch(`${API}/index/file`, { method: 'POST', body: form });
    const data = await res.json();
    out.innerHTML = `<div class="msg msg-success">Indexed <strong>${data.chunks_added}</strong> chunks from <strong>${data.filename}</strong></div>`;
    fileName.textContent = '';
    fileInput.value = '';
  } catch {
    out.innerHTML = `<div class="msg msg-error">Upload failed — is the backend running?</div>`;
  } finally {
    uploadBtn.disabled = false;
  }
});

// ── INDEX — text ──────────────────────────────────────────
document.getElementById('text-btn').addEventListener('click', async () => {
  const text   = document.getElementById('text-input').value.trim();
  const doc_id = document.getElementById('text-id').value.trim() || 'manual';
  const out    = document.getElementById('text-result');
  if (!text) return;
  out.innerHTML = `<div class="msg msg-loading"><span class="loader"></span>Indexing…</div>`;
  try {
    const res  = await fetch(`${API}/index/text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, doc_id }),
    });
    const data = await res.json();
    out.innerHTML = `<div class="msg msg-success">Indexed <strong>${data.chunks_added}</strong> chunks as <strong>${data.doc_id}</strong></div>`;
    document.getElementById('text-input').value = '';
    document.getElementById('text-id').value    = '';
  } catch {
    out.innerHTML = `<div class="msg msg-error">Indexing failed — is the backend running?</div>`;
  }
});

// ── STATS ─────────────────────────────────────────────────
document.getElementById('refresh-stats').addEventListener('click', loadStats);

async function loadStats() {
  const out = document.getElementById('stats-result');
  out.innerHTML = `<div class="msg msg-loading"><span class="loader"></span>Loading…</div>`;
  try {
    const res  = await fetch(`${API}/stats`);
    const data = await res.json();
    const docsHtml = data.documents.length === 0
      ? `<div class="empty-state"><div class="empty-state-icon">📂</div><p>No documents indexed yet.</p></div>`
      : `<ul class="doc-list">${data.documents.map((id, i) => `
          <li class="doc-item" style="animation:slideUp 0.3s ease ${i * 40}ms both">
            <span class="doc-dot"></span>
            <span class="doc-name">${esc(id)}</span>
            <button class="delete-btn" onclick="deleteDoc('${esc(id)}')">Remove</button>
          </li>`).join('')}</ul>`;
    out.innerHTML = `
      <div class="stats-kpi">
        <div class="kpi-card">
          <div class="kpi-val">${data.total_chunks}</div>
          <div class="kpi-label">Total Chunks</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-val">${data.documents.length}</div>
          <div class="kpi-label">Documents</div>
        </div>
      </div>
      <div class="sources-label" style="margin-bottom:14px">Indexed Documents</div>
      ${docsHtml}`;
  } catch {
    out.innerHTML = `<div class="msg msg-error">Could not load stats — is the backend running?</div>`;
  }
}

async function deleteDoc(docId) {
  if (!confirm(`Remove "${docId}" from the index?`)) return;
  try {
    await fetch(`${API}/documents/${encodeURIComponent(docId)}`, { method: 'DELETE' });
    loadStats();
  } catch { alert('Delete failed.'); }
}
