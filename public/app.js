// Hash Table Simulator Web Frontend Logic

const state = {
  datasetSize: 100,
  strategy: 'Separate Chaining',
  sizingMode: 'Auto',
  customSize: 150,
  targetLf: 0.75,
  seed: 42,
  userList: [],
  tableData: null,
  charts: {},
  logs: []
};

// API Base URL - empty string routes to relative /api endpoints
const API_BASE = '';

// DOM Elements
const el = {
  datasetSizeSelect: document.getElementById('dataset-size-select'),
  strategySelect: document.getElementById('strategy-select'),
  sizingModeGroup: document.getElementById('sizing-mode-group'),
  customSizeGroup: document.getElementById('custom-size-group'),
  customTableSize: document.getElementById('custom-table-size'),
  targetLfGroup: document.getElementById('target-lf-group'),
  lfSegmentedGroup: document.getElementById('lf-segmented-group'),
  
  btnGenerateData: document.getElementById('btn-generate-data'),
  btnPopulateTable: document.getElementById('btn-populate-table'),
  btnViewGrid: document.getElementById('btn-view-grid'),
  btnOpenRecLab: document.getElementById('btn-open-rec-lab'),
  btnRunLookup: document.getElementById('btn-run-lookup'),
  btnRunCollision: document.getElementById('btn-run-collision'),
  btnRunNumpy: document.getElementById('btn-run-numpy'),
  btnGenerateCharts: document.getElementById('btn-generate-charts'),
  btnHowItWorks: document.getElementById('btn-how-it-works'),
  btnReset: document.getElementById('btn-reset'),
  themeToggle: document.getElementById('theme-toggle'),

  // Stats
  statUsers: document.getElementById('stat-users'),
  statTableSize: document.getElementById('stat-table-size'),
  statLoadFactor: document.getElementById('stat-load-factor'),
  statCollisions: document.getElementById('stat-collisions'),
  statCollisionRate: document.getElementById('stat-collision-rate'),
  statAvgBucket: document.getElementById('stat-avg-bucket'),
  statMaxBucket: document.getElementById('stat-max-bucket'),
  statMemory: document.getElementById('stat-memory'),

  // Visualizer
  visEmptyState: document.getElementById('vis-empty-state'),
  bucketsGrid: document.getElementById('buckets-grid'),
  visTitle: document.getElementById('vis-title'),
  visSubtitle: document.getElementById('vis-subtitle'),

  // Search & Trace
  searchKeyInput: document.getElementById('search-key-input'),
  animSpeedSelect: document.getElementById('anim-speed-select'),
  btnRunSearch: document.getElementById('btn-run-search'),
  btnPickRandomUser: document.getElementById('btn-pick-random-user'),
  traceKey: document.getElementById('trace-key'),
  traceBucket: document.getElementById('trace-bucket'),
  traceStatus: document.getElementById('trace-status'),
  traceLatency: document.getElementById('trace-latency'),
  traceStepsContainer: document.getElementById('trace-steps-container'),

  // Recommender
  recUserSelect: document.getElementById('rec-user-select'),
  recTopN: document.getElementById('rec-top-n'),
  recAlgoSelect: document.getElementById('rec-algo-select'),
  btnGenerateRecs: document.getElementById('btn-generate-recs'),
  userProfileContent: document.getElementById('user-profile-content'),
  recItemsList: document.getElementById('rec-items-list'),
  recLatencyBadge: document.getElementById('rec-latency-badge'),

  // Charts
  btnRefreshCharts: document.getElementById('btn-refresh-charts'),

  // Console
  consoleBody: document.getElementById('console-body'),
  consoleFilter: document.getElementById('console-filter'),
  btnCopyConsole: document.getElementById('btn-copy-console'),
  btnClearConsole: document.getElementById('btn-clear-console'),
  logCount: document.getElementById('log-count'),

  // Tabs
  tabButtons: document.querySelectorAll('.tab-btn'),
  tabPanes: document.querySelectorAll('.tab-pane')
};

// ── LOGGING SYSTEM ──
function log(msg, type = 'info') {
  const time = new Date().toLocaleTimeString();
  const entry = { time, msg, type };
  state.logs.push(entry);
  renderLogLine(entry);
  el.logCount.innerText = `${state.logs.length} events`;
}

function renderLogLine(entry) {
  const filter = el.consoleFilter.value.trim().toLowerCase();
  if (filter && !entry.msg.toLowerCase().includes(filter)) return;

  const div = document.createElement('div');
  div.className = `log-line ${entry.type}`;
  div.innerText = `[${entry.time}] ${entry.msg}`;
  el.consoleBody.appendChild(div);
  el.consoleBody.scrollTop = el.consoleBody.scrollHeight;
}

function renderAllLogs() {
  el.consoleBody.innerHTML = '';
  state.logs.forEach(renderLogLine);
}

// ── TAB SWITCHING ──
function switchTab(tabId) {
  el.tabButtons.forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });
  el.tabPanes.forEach(pane => {
    pane.classList.toggle('active', pane.id === tabId);
  });
}

// ── API CALLS ──
async function apiCall(endpoint, method = 'GET', body = null) {
  if (window.location.protocol === 'file:') {
    const errorMsg = "Local file:// protocol detected. Please run 'python server.py' and open http://localhost:8000, or view your deployed Vercel URL.";
    log(errorMsg, 'error');
    alert(errorMsg);
    throw new Error(errorMsg);
  }

  try {
    const opts = {
      method,
      headers: { 'Content-Type': 'application/json' }
    };
    if (body) opts.body = JSON.stringify(body);
    
    // Primary attempt
    let resp = await fetch(`${API_BASE}${endpoint}`, opts);
    
    // If 404 and endpoint starts with /api/, try stripping /api/ as fallback
    if (resp.status === 404 && endpoint.startsWith('/api/')) {
      const fallbackEndpoint = endpoint.replace('/api/', '/');
      resp = await fetch(`${API_BASE}${fallbackEndpoint}`, opts);
    }

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status} ${resp.statusText}` }));
      throw new Error(err.detail || err.message || `Server error: ${resp.status}`);
    }
    return await resp.json();
  } catch (err) {
    log(`API Error: ${err.message}`, 'error');
    throw err;
  }
}

// ── ACTIONS ──

// 1. Generate Structured Data
async function handleGenerateData() {
  log(`Generating synthetic user dataset (${state.datasetSize} users, skew=75%)...`, 'info');
  try {
    const res = await apiCall('/api/generate-data', 'POST', {
      num_users: state.datasetSize,
      seed: state.seed
    });
    
    state.userList = res.all_user_ids;
    state.tableData = null; // Invalidate previously built hash table
    updateUserDropdowns(res.all_user_ids);
    
    // Reset table metrics until user populates
    el.statUsers.innerText = res.total_users;
    el.statTableSize.innerText = '-';
    el.statLoadFactor.innerText = '-';
    el.statCollisions.innerText = '-';
    el.statCollisionRate.innerText = '-';
    el.statAvgBucket.innerText = '-';
    el.statMaxBucket.innerText = '-';
    el.statMemory.innerText = '-';

    // Invalidate visualizer view
    el.visEmptyState.style.display = 'flex';
    el.visEmptyState.innerHTML = `
      <div class="empty-state-icon">⚡</div>
      <h3>New Dataset Generated (${res.total_users} Users)</h3>
      <p style="color:var(--accent-amber);">Previous table structures invalidated. Click <strong>"2. Populate Hash Table"</strong> to index this dataset.</p>
    `;
    el.bucketsGrid.style.display = 'none';
    el.bucketsGrid.innerHTML = '';

    el.recItemsList.innerHTML = '<p class="empty-note">Table unpopulated. Click "2. Populate Hash Table" to activate recommendation engine.</p>';

    log(`✔ Generated ${res.total_users} unique user interaction profiles (Seed ${res.seed}).`, 'success');
    log("New dataset generated. Populate the hash table to continue.", 'warning');
  } catch (err) {
    log(`Failed to generate data: ${err.message}`, 'error');
  }
}

// 2. Populate Hash Table
async function handlePopulateTable() {
  log(`Populating Hash Table using ${state.strategy}...`, 'info');
  try {
    const res = await apiCall('/api/populate', 'POST', {
      strategy: state.strategy,
      size_mode: state.sizingMode,
      custom_size: state.customSize,
      target_lf: state.targetLf,
      num_users: state.datasetSize,
      seed: state.seed
    });

    state.tableData = res;
    state.userList = res.user_ids;
    updateUserDropdowns(res.user_ids);
    updateStats(res.stats, res.is_extendible);
    renderVisualizer(res);
    switchTab('tab-visualizer');
    
    log(`✔ Successfully indexed ${res.stats.total_users} users into ${res.strategy}.`, 'success');
    if (res.is_extendible) {
      log(`Extendible Stats: Directory = ${res.stats.directory_size} | Unique Buckets = ${res.stats.num_unique_buckets} | Utilization = ${(res.stats.bucket_utilization * 100).toFixed(1)}% | Splits = ${res.stats.splits_count}`, 'data');
    } else {
      log(`Stats: Table Size = ${res.stats.table_size} | Load Factor = ${res.stats.load_factor} | Collisions = ${res.stats.collisions} (${res.stats.collision_rate_pct}%)`, 'data');
    }
  } catch (err) {
    log(`Population failed: ${err.message}`, 'error');
  }
}

function updateStats(stats, isExtendible = false) {
  el.statUsers.innerText = stats.total_users;

  if (isExtendible) {
    el.statTableSize.innerText = `${stats.directory_size} (Dir)`;
    el.statLoadFactor.innerText = `${(stats.bucket_utilization * 100).toFixed(1)}%`;
    el.statLoadFactor.title = "Bucket Utilization: Records / (Unique Buckets * Capacity)";
    el.statCollisions.innerText = `${stats.splits_count} Splits`;
    el.statCollisionRate.innerText = `${stats.num_unique_buckets} Bkts`;
    el.statAvgBucket.innerText = `Cap ${stats.bucket_capacity}`;
    el.statMaxBucket.innerText = `d = ${stats.global_depth}`;
    el.statMemory.innerText = formatBytes(stats.memory_bytes);
  } else {
    el.statTableSize.innerText = stats.table_size;
    el.statLoadFactor.innerText = typeof stats.load_factor === 'number' ? stats.load_factor.toFixed(2) : stats.load_factor;
    el.statLoadFactor.title = "Load Factor: Records / Table Size";
    el.statCollisions.innerText = stats.collisions;
    el.statCollisionRate.innerText = `${stats.collision_rate_pct}%`;
    el.statAvgBucket.innerText = stats.avg_bucket_len;
    el.statMaxBucket.innerText = stats.max_bucket_len;
    el.statMemory.innerText = formatBytes(stats.memory_bytes);
  }
}

function formatBytes(bytes) {
  if (!bytes || bytes <= 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(2)} MB`;
}

function updateUserDropdowns(userIds) {
  if (!userIds || userIds.length === 0) {
    el.recUserSelect.innerHTML = '<option value="">No users available</option>';
    return;
  }
  el.recUserSelect.innerHTML = '';
  userIds.slice(0, 100).forEach(uid => {
    const opt = document.createElement('option');
    opt.value = uid;
    opt.innerText = `User ${uid}`;
    el.recUserSelect.appendChild(opt);
  });
  if (userIds.length > 0) {
    el.searchKeyInput.value = userIds[0];
  }
}

// ── RENDER BUCKET VISUALIZER ──
function renderVisualizer(data) {
  el.visEmptyState.style.display = 'none';
  el.bucketsGrid.style.display = 'flex';
  el.bucketsGrid.innerHTML = '';

  el.visTitle.innerText = `${data.strategy} — Bucket Inspector`;

  if (data.is_extendible) {
    el.visSubtitle.innerText = `Extendible Directory: Global Depth = ${data.stats.global_depth} (${data.stats.directory_size} slots) ➔ ${data.stats.num_unique_buckets} physical buckets. True bucket utilization: ${(data.stats.bucket_utilization * 100).toFixed(1)}%.`;
    renderExtendibleGrid(data.buckets);
    return;
  }

  el.visSubtitle.innerText = `Displaying bucket layout for ${data.stats.total_users} keys across ${data.stats.table_size} table slots.`;

  data.buckets.forEach(bucket => {
    const row = document.createElement('div');
    row.className = `bucket-row ${bucket.is_collision ? 'has-collision' : ''}`;
    row.id = `bucket-row-${bucket.index}`;

    const idxBadge = document.createElement('div');
    idxBadge.className = 'bucket-idx-badge';
    idxBadge.innerText = `Slot ${bucket.index.toString().padStart(2, '0')}`;
    row.appendChild(idxBadge);

    const arrow = document.createElement('div');
    arrow.className = 'bucket-arrow';
    arrow.innerText = '➔';
    row.appendChild(arrow);

    const nodesContainer = document.createElement('div');
    nodesContainer.className = 'bucket-nodes';

    if (bucket.is_empty) {
      const emptyPill = document.createElement('div');
      emptyPill.className = 'empty-slot-pill';
      emptyPill.innerText = '[ Empty Slot ]';
      nodesContainer.appendChild(emptyPill);
    } else {
      bucket.items.forEach((item, i) => {
        const pill = document.createElement('div');
        pill.className = 'user-node-pill';
        pill.title = `Click to search User ${item.key}: Movies [${item.preview.join(', ')}...]`;
        pill.innerHTML = `
          <span class="pill-user">User ${item.key}</span>
          <span class="pill-count">${item.count} movies</span>
        `;
        pill.addEventListener('click', () => {
          el.searchKeyInput.value = item.key;
          switchTab('tab-search');
          handleSearch();
        });
        nodesContainer.appendChild(pill);

        if (i < bucket.items.length - 1) {
          const conn = document.createElement('span');
          conn.className = 'chain-connector';
          conn.innerText = '➔';
          nodesContainer.appendChild(conn);
        }
      });
    }

    row.appendChild(nodesContainer);
    el.bucketsGrid.appendChild(row);
  });
}

function renderExtendibleGrid(extData) {
  const container = document.createElement('div');
  container.className = 'extendible-view';

  const meta = document.createElement('div');
  meta.className = 'dir-meta';
  meta.innerText = `Extendible Directory Global Depth = ${extData.global_depth} (Directory size: ${extData.directory_size} pointer slots)`;
  container.appendChild(meta);

  extData.directory.forEach(dir => {
    const row = document.createElement('div');
    row.className = 'bucket-row';
    row.id = `bucket-row-${dir.dir_index}`;

    const idxBadge = document.createElement('div');
    idxBadge.className = 'bucket-idx-badge';
    idxBadge.innerText = `[${dir.bin_index}] #${dir.dir_index}`;
    row.appendChild(idxBadge);

    const arrow = document.createElement('div');
    arrow.className = 'bucket-arrow';
    arrow.innerText = '➔';
    row.appendChild(arrow);

    const nodesContainer = document.createElement('div');
    nodesContainer.className = 'bucket-nodes';

    const depthBadge = document.createElement('span');
    depthBadge.style.cssText = 'font-size:11px; color:#d8b4fe; margin-right:8px; font-weight:600;';
    depthBadge.innerText = `Bucket Local Depth ${dir.local_depth} (${dir.items.length}/${dir.capacity}):`;
    nodesContainer.appendChild(depthBadge);

    if (dir.items.length === 0) {
      const emptyPill = document.createElement('div');
      emptyPill.className = 'empty-slot-pill';
      emptyPill.innerText = '[ Empty Bucket ]';
      nodesContainer.appendChild(emptyPill);
    } else {
      dir.items.forEach(item => {
        const pill = document.createElement('div');
        pill.className = 'user-node-pill';
        pill.title = `Click to search User ${item.key}: Movies [${item.preview.join(', ')}...]`;
        pill.innerHTML = `
          <span class="pill-user">User ${item.key}</span>
          <span class="pill-count">${item.count} movies</span>
        `;
        pill.addEventListener('click', () => {
          el.searchKeyInput.value = item.key;
          switchTab('tab-search');
          handleSearch();
        });
        nodesContainer.appendChild(pill);
      });
    }

    row.appendChild(nodesContainer);
    container.appendChild(row);
  });

  el.bucketsGrid.appendChild(container);
}

// ── SEARCH & STEP TRACE ──
async function handleSearch() {
  if (!state.tableData) {
    alert('Please populate the hash table first before searching.');
    return;
  }

  const key = parseInt(el.searchKeyInput.value);
  if (isNaN(key)) {
    alert('Please enter a valid numeric User ID.');
    return;
  }

  log(`Searching for User ID #${key}...`, 'info');
  try {
    const res = await apiCall('/api/search', 'POST', { key });

    el.traceKey.innerText = `#${res.key}`;
    el.traceBucket.innerText = res.hash_val !== null ? `Bucket ${res.hash_val}` : 'N/A';
    el.traceStatus.innerText = res.found ? 'FOUND ✔' : 'NOT FOUND ✖';
    el.traceStatus.style.color = res.found ? 'var(--accent-green)' : 'var(--accent-red)';
    el.traceLatency.innerText = `${res.latency_us} µs`;

    animateTrace(res.trace, res.found, res.value);

    log(res.found 
      ? `Lookup Success: User #${res.key} located in ${res.latency_us} µs (${res.items_count} movies).`
      : `Lookup Complete: User #${res.key} not present in table (${res.latency_us} µs).`,
      res.found ? 'success' : 'warning'
    );
  } catch (err) {
    log(`Search failed: ${err.message}`, 'error');
  }
}

function animateTrace(trace, found, val) {
  el.traceStepsContainer.innerHTML = '';
  const speed = el.animSpeedSelect.value;
  const delay = speed === 'Fast' ? 180 : speed === 'Slow' ? 800 : 400;

  trace.forEach((step, idx) => {
    setTimeout(() => {
      const card = document.createElement('div');
      const isLast = idx === trace.length - 1;
      
      let statusClass = 'probe';
      if (step.matched || (isLast && found)) {
        statusClass = 'success';
      } else if (isLast && !found) {
        statusClass = 'failure';
      }

      const bucketIdx = (step.index !== undefined && step.index !== null) ? step.index 
                      : (step.bucket !== undefined && step.bucket !== null) ? step.bucket 
                      : (step.bucket_idx !== undefined && step.bucket_idx !== null) ? step.bucket_idx 
                      : 'N/A';

      const stepTitle = step.action || `Step ${step.step_num || idx + 1}`;
      const stepDetails = step.details || (step.matched ? 'Key matched record!' : (isLast && !found ? 'Key not found in table.' : 'Inspecting bucket slot...'));

      card.className = `step-card ${statusClass}`;
      card.innerHTML = `
        <span class="step-num">#${step.step_num || idx + 1}</span>
        <div class="step-details">
          <strong>${stepTitle}:</strong> ${bucketIdx !== 'N/A' ? `Slot ${bucketIdx} · ` : ''}<span>${stepDetails}</span>
        </div>
      `;
      el.traceStepsContainer.appendChild(card);
      el.traceStepsContainer.scrollTop = el.traceStepsContainer.scrollHeight;

      // Highlight corresponding row in visualizer
      if (bucketIdx !== 'N/A') {
        const row = document.getElementById(`bucket-row-${bucketIdx}`);
        if (row) {
          document.querySelectorAll('.bucket-row.highlighted').forEach(r => r.classList.remove('highlighted'));
          row.classList.add('highlighted');
        }
      }
    }, idx * delay);
  });
}

// ── RECOMMENDER LAB ──
async function handleGenerateRecs() {
  if (!state.tableData) {
    alert('Please populate the hash table first before generating recommendations.');
    return;
  }

  const uid = parseInt(el.recUserSelect.value);
  if (isNaN(uid)) {
    alert('Please select a valid User ID.');
    return;
  }

  const topN = parseInt(el.recTopN.value);
  const algo = el.recAlgoSelect.value;

  log(`Running ${algo} for User #${uid} (Top-${topN})...`, 'info');
  try {
    const res = await apiCall('/api/recommend', 'POST', {
      user_id: uid,
      top_n: topN,
      algorithm: algo
    });

    el.recLatencyBadge.innerText = `${res.latency_ms} ms`;
    renderUserProfile(res.user_history, uid);
    renderRecommendations(res.recommendations);

    log(`✔ Generated Top-${topN} recommendations for User #${uid} via ${algo} in ${res.latency_ms} ms.`, 'success');
  } catch (err) {
    log(`Recommendation generation failed: ${err.message}`, 'error');
  }
}

function renderUserProfile(history, uid) {
  const dist = history.genre_distribution || {};
  const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;

  let barsHtml = Object.entries(dist).map(([genre, count]) => {
    const pct = Math.round((count / total) * 100);
    return `
      <div class="genre-bar-row">
        <span class="genre-bar-label">${genre}</span>
        <div class="genre-bar-track">
          <div class="genre-bar-fill" style="width: ${pct}%"></div>
        </div>
        <span style="font-family:var(--font-mono); font-size:10px;">${count} (${pct}%)</span>
      </div>
    `;
  }).join('');

  el.userProfileContent.innerHTML = `
    <div class="profile-stats-group">
      <div style="font-size:13px; font-weight:700;">User #${uid} History (${history.movie_ids.length} watched movies)</div>
      <div style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted); line-height:1.6;">
        Movie IDs: [${history.movie_ids.join(', ')}]
      </div>
      <div class="genre-bars">
        <div style="font-size:11px; font-weight:700; color:var(--accent-cyan); margin-bottom:4px;">Genre Skew Analysis:</div>
        ${barsHtml || '<p>No genre history recorded.</p>'}
      </div>
    </div>
  `;
}

function renderRecommendations(items) {
  el.recItemsList.innerHTML = '';
  items.forEach((item, idx) => {
    setTimeout(() => {
      const card = document.createElement('div');
      card.className = 'rec-item-card';
      card.innerHTML = `
        <div class="rec-rank">#${item.rank}</div>
        <div class="rec-info">
          <div class="rec-movie-id">Movie ID #${item.movie_id} <span class="rec-genre-tag">${item.genre_name}</span></div>
          <div class="rec-source">Method: ${item.source}</div>
        </div>
      `;
      el.recItemsList.appendChild(card);
    }, idx * 60);
  });
}

// ── CHARTS & BENCHMARKS ──
async function handleRunBenchmarks() {
  log('Running benchmark suite across sizes & load factors...', 'info');
  switchTab('tab-charts');

  try {
    const [lookupRes, collisionRes, numpyRes] = await Promise.all([
      apiCall('/api/benchmarks/lookup'),
      apiCall('/api/benchmarks/collision'),
      apiCall('/api/benchmarks/numpy')
    ]);

    renderLookupChart(lookupRes);
    renderCollisionChart(collisionRes);
    renderMemoryChart(lookupRes);
    renderNumpyChart(numpyRes);

    log('✔ Benchmarks completed & charts updated.', 'success');
  } catch (err) {
    log(`Benchmark execution failed: ${err.message}`, 'error');
  }
}

function renderLookupChart(data) {
  const ctx = document.getElementById('chart-lookup').getContext('2d');
  if (state.charts.lookup) state.charts.lookup.destroy();

  const labels = data.sizes.map(s => `${s} Users`);
  const chainingData = data.results['Separate Chaining'].map(d => d.avg_lookup_us);
  const probingData = data.results['Linear Probing'].map(d => d.avg_lookup_us);

  state.charts.lookup = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Separate Chaining (µs)',
          data: chainingData,
          borderColor: '#00e5ff',
          backgroundColor: 'rgba(0, 229, 255, 0.1)',
          fill: true,
          tension: 0.3
        },
        {
          label: 'Linear Probing (µs)',
          data: probingData,
          borderColor: '#ffd54f',
          backgroundColor: 'rgba(255, 213, 79, 0.1)',
          fill: true,
          tension: 0.3
        }
      ]
    },
    options: getChartOptions('Avg Lookup Latency (µs)')
  });
}

function renderCollisionChart(data) {
  const ctx = document.getElementById('chart-collision').getContext('2d');
  if (state.charts.collision) state.charts.collision.destroy();

  const labels = data.load_factors.map(lf => `λ = ${lf}`);
  const chainingData = data.results['Separate Chaining'].map(d => d.collision_rate_pct);
  const probingData = data.results['Linear Probing'].map(d => d.collision_rate_pct);

  state.charts.collision = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Separate Chaining (%)',
          data: chainingData,
          backgroundColor: 'rgba(0, 229, 255, 0.7)',
          borderRadius: 4
        },
        {
          label: 'Linear Probing (%)',
          data: probingData,
          backgroundColor: 'rgba(255, 82, 82, 0.7)',
          borderRadius: 4
        }
      ]
    },
    options: getChartOptions('Collision Rate (%)')
  });
}

function renderMemoryChart(data) {
  const ctx = document.getElementById('chart-memory').getContext('2d');
  if (state.charts.memory) state.charts.memory.destroy();

  const labels = data.sizes.map(s => `${s} Users`);
  const chainingData = data.results['Separate Chaining'].map(d => d.memory_bytes);
  const probingData = data.results['Linear Probing'].map(d => d.memory_bytes);

  state.charts.memory = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Separate Chaining (Bytes)',
          data: chainingData,
          borderColor: '#b388ff',
          tension: 0.3
        },
        {
          label: 'Linear Probing (Bytes)',
          data: probingData,
          borderColor: '#00e676',
          tension: 0.3
        }
      ]
    },
    options: getChartOptions('Memory Footprint (Bytes)')
  });
}

function renderNumpyChart(data) {
  const ctx = document.getElementById('chart-numpy').getContext('2d');
  if (state.charts.numpy) state.charts.numpy.destroy();

  state.charts.numpy = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Python List', 'NumPy bincount'],
      datasets: [
        {
          label: 'Execution Time (Seconds)',
          data: [data.python_list_time_sec, data.numpy_time_sec],
          backgroundColor: ['#ff5252', '#00e676'],
          borderRadius: 6
        }
      ]
    },
    options: {
      ...getChartOptions('Time (s)'),
      plugins: {
        ...getChartOptions().plugins,
        title: {
          display: true,
          text: `NumPy Speedup Factor: ${data.speedup}x`,
          color: '#00e676',
          font: { family: 'Outfit', size: 13, weight: 'bold' }
        }
      }
    }
  });
}

function getChartOptions(yAxisLabel = '') {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: '#94a3b8',
          font: { family: 'Outfit', size: 11 }
        }
      }
    },
    scales: {
      x: {
        ticks: { color: '#64748b', font: { family: 'Outfit', size: 10 } },
        grid: { color: 'rgba(255, 255, 255, 0.05)' }
      },
      y: {
        title: {
          display: !!yAxisLabel,
          text: yAxisLabel,
          color: '#94a3b8',
          font: { family: 'Outfit', size: 11 }
        },
        ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } },
        grid: { color: 'rgba(255, 255, 255, 0.05)' }
      }
    }
  };
}

// ── EVENT LISTENERS ──
function setupEventListeners() {
  // Sidebar options
  el.datasetSizeSelect.addEventListener('change', (e) => {
    state.datasetSize = parseInt(e.target.value);
  });

  el.strategySelect.addEventListener('change', (e) => {
    state.strategy = e.target.value;
  });

  el.sizingModeGroup.querySelectorAll('.segment-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      el.sizingModeGroup.querySelectorAll('.segment-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.sizingMode = btn.dataset.mode;
      const isCustom = state.sizingMode === 'Custom';
      el.customSizeGroup.style.display = isCustom ? 'flex' : 'none';
      el.targetLfGroup.style.display = isCustom ? 'none' : 'flex';
    });
  });

  el.customTableSize.addEventListener('change', (e) => {
    state.customSize = parseInt(e.target.value) || 150;
  });

  el.lfSegmentedGroup.querySelectorAll('.segment-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      el.lfSegmentedGroup.querySelectorAll('.segment-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.targetLf = parseFloat(btn.dataset.lf);
    });
  });

  // Action buttons
  el.btnGenerateData.addEventListener('click', handleGenerateData);
  el.btnPopulateTable.addEventListener('click', handlePopulateTable);
  el.btnViewGrid.addEventListener('click', () => switchTab('tab-visualizer'));
  el.btnOpenRecLab.addEventListener('click', () => switchTab('tab-recommender'));

  // Benchmarks
  el.btnRunLookup.addEventListener('click', async () => {
    log('Running Lookup Benchmark...', 'info');
    const res = await apiCall('/api/benchmarks/lookup');
    Object.entries(res.results).forEach(([strat, data]) => {
      log(`--- ${strat} ---`, 'info');
      data.forEach(d => log(`Users: ${d.size} | Latency: ${d.avg_lookup_us} µs | Collisions: ${d.collisions} | Mem: ${d.memory_bytes} B`, 'data'));
    });
  });

  el.btnRunCollision.addEventListener('click', async () => {
    log('Running Collision Experiment...', 'info');
    const res = await apiCall('/api/benchmarks/collision');
    Object.entries(res.results).forEach(([strat, data]) => {
      log(`--- ${strat} ---`, 'info');
      data.forEach(d => log(`Load Factor: ${d.load_factor} | Collision Rate: ${d.collision_rate_pct}% | Latency: ${d.avg_lookup_us} µs`, 'data'));
    });
  });

  el.btnRunNumpy.addEventListener('click', async () => {
    log('Comparing Python List vs NumPy bincount...', 'info');
    const res = await apiCall('/api/benchmarks/numpy');
    log(`Python List: ${res.python_list_time_sec}s | NumPy bincount: ${res.numpy_time_sec}s | Speedup: ${res.speedup}x`, 'success');
  });

  el.btnGenerateCharts.addEventListener('click', handleRunBenchmarks);
  el.btnRefreshCharts.addEventListener('click', handleRunBenchmarks);

  el.btnHowItWorks.addEventListener('click', () => switchTab('tab-edu'));

  // Reset button: syncs with backend /api/reset
  el.btnReset.addEventListener('click', async () => {
    try {
      await apiCall('/api/reset', 'POST');
    } catch (e) {
      console.warn("Backend reset notification failed", e);
    }
    state.userList = [];
    state.tableData = null;
    el.statUsers.innerText = '0';
    el.statTableSize.innerText = '0';
    el.statLoadFactor.innerText = '0.00';
    el.statCollisions.innerText = '0';
    el.statCollisionRate.innerText = '0.0%';
    el.statAvgBucket.innerText = '0.0';
    el.statMaxBucket.innerText = '0';
    el.statMemory.innerText = '0 B';
    el.visEmptyState.style.display = 'flex';
    el.visEmptyState.innerHTML = `
      <div class="empty-state-icon">⚡</div>
      <h3>Hash Table Not Populated Yet</h3>
      <p>Click <strong>"1. Generate Structured Data"</strong> followed by <strong>"2. Populate Hash Table"</strong> to render the bucket array.</p>
    `;
    el.bucketsGrid.style.display = 'none';
    el.bucketsGrid.innerHTML = '';
    el.recUserSelect.innerHTML = '<option value="">Populate table first</option>';
    el.userProfileContent.innerHTML = '<p class="empty-note">Select a user and run recommendation.</p>';
    el.recItemsList.innerHTML = '<p class="empty-note">Recommendations will appear here.</p>';
    el.traceStepsContainer.innerHTML = '<div class="empty-state"><p>Enter a User ID and click <strong>Search & Animate</strong> to trace the hash lookup path.</p></div>';
    el.traceKey.innerText = '-';
    el.traceBucket.innerText = '-';
    el.traceStatus.innerText = '-';
    el.traceLatency.innerText = '-';
    log('Simulator state reset on server and client.', 'warning');
  });

  // Search
  el.btnRunSearch.addEventListener('click', handleSearch);
  el.searchKeyInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSearch();
  });
  el.btnPickRandomUser.addEventListener('click', () => {
    if (!state.tableData || state.userList.length === 0) {
      alert('Please populate the hash table first.');
      return;
    }
    const rand = state.userList[Math.floor(Math.random() * state.userList.length)];
    el.searchKeyInput.value = rand;
    handleSearch();
  });

  // Recommender
  el.btnGenerateRecs.addEventListener('click', handleGenerateRecs);

  // Tabs
  el.tabButtons.forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Console actions
  el.consoleFilter.addEventListener('input', renderAllLogs);
  el.btnClearConsole.addEventListener('click', () => {
    state.logs = [];
    el.consoleBody.innerHTML = '';
    el.logCount.innerText = '0 events';
  });
  el.btnCopyConsole.addEventListener('click', () => {
    const text = state.logs.map(l => `[${l.time}] ${l.msg}`).join('\n');
    navigator.clipboard.writeText(text).then(() => {
      el.btnCopyConsole.innerText = '✓ Copied';
      setTimeout(() => el.btnCopyConsole.innerText = '📋 Copy', 1500);
    });
  });

  // Theme toggle
  el.themeToggle.addEventListener('click', () => {
    const isLight = document.body.classList.toggle('light-theme');
    el.themeToggle.innerText = isLight ? '☀️' : '🌙';
  });
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  log('Hash Table Recommender Simulator ready. Ready to simulate!', 'success');
});
