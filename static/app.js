// ========================================================
// TVPReID Interactive Demonstration Engine (Team 33)
// ========================================================

document.addEventListener('DOMContentLoaded', () => {
  // State
  let currentPresetId = 'preset_1';
  let currentCondition = 'clean';
  let currentMethod = 'weighted';
  let isSideBySide = false;
  let activeBenchmarkDataset = 'unified';
  let benchmarkData = null;
  let presetsData = [];

  // DOM Elements
  const tabButtons = document.querySelectorAll('.nav-tab');
  const tabContents = document.querySelectorAll('.tab-content');
  const presetsContainer = document.getElementById('presets-container');
  const inputQuery = document.getElementById('input-query');
  const btnRunSearch = document.getElementById('btn-run-search');
  const selectDataset = document.getElementById('select-dataset');
  const qualityBarsContainer = document.getElementById('quality-bars-container');
  const recoveryBanner = document.getElementById('recovery-banner');
  const recoveryText = document.getElementById('recovery-text');
  const videoResultsGrid = document.getElementById('video-results-grid');
  const targetTrackerContainer = document.getElementById('target-tracker-container');
  const comparisonContainer = document.getElementById('comparison-container');
  const compareBaselineGrid = document.getElementById('compare-baseline-grid');
  const compareWeightedGrid = document.getElementById('compare-weighted-grid');
  const btnToggleCompare = document.getElementById('btn-toggle-compare');
  const compareBtnText = document.getElementById('compare-btn-text');

  // Condition toggles
  const btnCondClean = document.getElementById('btn-cond-clean');
  const btnCondCorrupted = document.getElementById('btn-cond-corrupted');

  // Method toggles
  const btnMethodMean = document.getElementById('btn-method-mean');
  const btnMethodWeighted = document.getElementById('btn-method-weighted');

  // 1. Tab Navigation
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');
      tabButtons.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(targetId).classList.add('active');
    });
  });

  // 2. Initialize App
  async function init() {
    try {
      // Fetch presets
      const resPresets = await fetch('/api/presets');
      presetsData = await resPresets.json();
      if (presetsData && presetsData.length > 0) {
        currentPresetId = presetsData[0].id;
        inputQuery.value = presetsData[0].query;
      }
      renderPresets();

      // Fetch benchmarks
      const resBench = await fetch('/api/benchmarks');
      benchmarkData = await resBench.json();
      renderBenchmarks('unified');

      // Setup benchmark tabs
      document.querySelectorAll('.bench-tab').forEach(btn => {
        btn.addEventListener('click', () => {
          document.querySelectorAll('.bench-tab').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          const ds = btn.getAttribute('data-dataset');
          renderBenchmarks(ds);
        });
      });

      // Initial Search
      runSearch();
    } catch (err) {
      console.error('Initialization error:', err);
    }
  }

  // 3. Render Presets Grid
  function renderPresets() {
    presetsContainer.innerHTML = '';
    presetsData.forEach(preset => {
      const card = document.createElement('div');
      card.className = `preset-card ${preset.id === currentPresetId ? 'active' : ''}`;
      card.setAttribute('data-preset-id', preset.id);
      
      const tagHtml = preset.tag ? `<span class="preset-tag-badge">${preset.tag}</span>` : '';
      card.innerHTML = `
        <div class="preset-top">
          <span class="preset-title">${preset.title}</span>
          <div class="preset-badges">
            ${tagHtml}
            <span class="preset-dataset-badge">${preset.dataset}</span>
          </div>
        </div>
        <p class="preset-desc">${preset.description}</p>
      `;
      card.addEventListener('click', () => {
        currentPresetId = preset.id;
        inputQuery.value = preset.query;
        document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        runSearch();
      });
      presetsContainer.appendChild(card);
    });
  }

  // 4. Toggle Handlers
  btnCondClean.addEventListener('click', () => {
    currentCondition = 'clean';
    btnCondClean.classList.add('active');
    btnCondCorrupted.classList.remove('active');
    if (isSideBySide) toggleSideBySide(false);
    runSearch();
  });

  btnCondCorrupted.addEventListener('click', () => {
    currentCondition = 'corrupted';
    btnCondCorrupted.classList.add('active');
    btnCondClean.classList.remove('active');
    runSearch();
  });

  btnMethodMean.addEventListener('click', () => {
    currentMethod = 'mean';
    btnMethodMean.classList.add('active');
    btnMethodWeighted.classList.remove('active');
    if (isSideBySide) toggleSideBySide(false);
    runSearch();
  });

  btnMethodWeighted.addEventListener('click', () => {
    currentMethod = 'weighted';
    btnMethodWeighted.classList.add('active');
    btnMethodMean.classList.remove('active');
    runSearch();
  });

  selectDataset.addEventListener('change', () => {
    runSearch();
  });

  function handleSearchTrigger() {
    const queryText = inputQuery.value.trim().toLowerCase();
    // Check if query matches any known preset
    const matchedPreset = presetsData.find(p => p.query.trim().toLowerCase() === queryText);
    if (matchedPreset) {
      currentPresetId = matchedPreset.id;
      document.querySelectorAll('.preset-card').forEach(c => {
        if (c.getAttribute('data-preset-id') === matchedPreset.id) {
          c.classList.add('active');
        } else {
          c.classList.remove('active');
        }
      });
    } else {
      currentPresetId = null;
      document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
    }
    runSearch();
  }

  btnRunSearch.addEventListener('click', handleSearchTrigger);

  inputQuery.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      handleSearchTrigger();
    }
  });

  inputQuery.addEventListener('input', () => {
    const queryText = inputQuery.value.trim().toLowerCase();
    const matchedPreset = presetsData.find(p => p.query.trim().toLowerCase() === queryText);
    if (matchedPreset) {
      currentPresetId = matchedPreset.id;
      document.querySelectorAll('.preset-card').forEach(c => {
        c.classList.toggle('active', c.getAttribute('data-preset-id') === matchedPreset.id);
      });
    } else {
      currentPresetId = null;
      document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
    }
  });

  // 5. Side-by-Side Comparison Toggle
  btnToggleCompare.addEventListener('click', () => {
    toggleSideBySide(!isSideBySide);
  });

  function toggleSideBySide(enable) {
    isSideBySide = enable;
    if (isSideBySide) {
      if (targetTrackerContainer) targetTrackerContainer.classList.add('hidden');
      btnToggleCompare.classList.add('active');
      compareBtnText.textContent = 'Exit Side-by-Side View';
      videoResultsGrid.classList.add('hidden');
      comparisonContainer.classList.remove('hidden');
      // Force corrupted mode for comparison proof
      currentCondition = 'corrupted';
      btnCondCorrupted.classList.add('active');
      btnCondClean.classList.remove('active');
      runComparisonSearch();
    } else {
      btnToggleCompare.classList.remove('active');
      compareBtnText.textContent = 'Compare Baseline vs Ours';
      videoResultsGrid.classList.remove('hidden');
      comparisonContainer.classList.add('hidden');
      runSearch();
    }
  }

  // 6. Execute Live Search
  async function runSearch() {
    if (isSideBySide) {
      runComparisonSearch();
      return;
    }

    const payload = {
      preset_id: currentPresetId,
      query: inputQuery.value,
      dataset: selectDataset.value,
      condition: currentCondition,
      method: currentMethod,
      top_k: 6
    };

    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      renderSingleResults(data);
      renderQualityBars(data.frame_qualities, data.reliability_weights);
    } catch (err) {
      console.error('Search error:', err);
    }
  }

  // 7. Execute Side-by-Side Comparison Searches
  async function runComparisonSearch() {
    try {
      // 1. Search with Baseline Mean Pooling
      const resMean = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          preset_id: currentPresetId,
          query: inputQuery.value,
          dataset: selectDataset.value,
          condition: 'corrupted',
          method: 'mean',
          top_k: 4
        })
      });
      const dataMean = await resMean.json();

      // 2. Search with Our Quality-Weighted (q_j) Pooling
      const resWeighted = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          preset_id: currentPresetId,
          query: inputQuery.value,
          dataset: selectDataset.value,
          condition: 'corrupted',
          method: 'weighted',
          top_k: 4
        })
      });
      const dataWeighted = await resWeighted.json();

      // Render both columns
      renderVideoCards(dataMean.results, compareBaselineGrid);
      renderVideoCards(dataWeighted.results, compareWeightedGrid);

      // Update meta tags
      if (dataMean.is_custom_neural) {
        document.getElementById('compare-baseline-meta').textContent = 
          `Baseline Mean Pooling (Top match sim: ${dataMean.results[0]?.score || 0})`;
        document.getElementById('compare-weighted-meta').textContent = 
          `Reliability-Weighted Pooling (Top match sim: ${dataWeighted.results[0]?.score || 0})`;
        recoveryBanner.className = 'recovery-status-banner';
        recoveryText.textContent = `Side-by-Side: Custom Neural Query evaluated under Baseline Mean vs Reliability-Weighted (50% Blur)`;
      } else {
        const meanRank = dataMean.target_rank;
        const weightedRank = dataWeighted.target_rank;
        const gain = meanRank - weightedRank;
        
        document.getElementById('compare-baseline-meta').textContent = 
          `Target Rank: #${meanRank} ${meanRank <= 4 ? '(In Top 4)' : '(Pushed outside Top 4 by blur)'}`;
        document.getElementById('compare-weighted-meta').textContent = 
          `Target Rank: #${weightedRank} ${weightedRank === 1 ? '(Recovered to Rank #1 by q_j!)' : (gain > 0 ? `(Recovered +${gain} ranks!)` : '')}`;
        
        if (gain > 0) {
          recoveryBanner.className = 'recovery-status-banner';
          recoveryBanner.querySelector('.status-icon').textContent = weightedRank === 1 ? '✓' : '⚡';
          recoveryText.innerHTML = `<strong>Side-by-Side:</strong> Target dropped to <strong>Rank #${meanRank}</strong> under Baseline Mean ➔ Recovered to <strong>Rank #${weightedRank}</strong> by Reliability-Weighting (<strong>+${gain} ranks recovered!</strong>)`;
        } else {
          recoveryBanner.className = 'recovery-status-banner banner-dropped';
          recoveryBanner.querySelector('.status-icon').textContent = '⚠';
          recoveryText.innerHTML = `<strong>Side-by-Side:</strong> Target at <strong>Rank #${meanRank}</strong> (Baseline Mean) vs <strong>Rank #${weightedRank}</strong> (Reliability-Weighted)`;
        }
      }

      renderQualityBars(dataWeighted.frame_qualities, dataWeighted.reliability_weights);
    } catch (err) {
      console.error('Comparison error:', err);
    }
  }

  // 8. Render Results in Single Grid
  function renderSingleResults(data) {
    renderVideoCards(data.results, videoResultsGrid);

    const activePreset = presetsData.find(p => p.id === currentPresetId);

    // 1. Target Tracker Card (for tracked target outside top 6)
    if (data.target_info && !data.target_info.in_top_k) {
      const tInfo = data.target_info;
      targetTrackerContainer.classList.remove('hidden');

      let badgeTag = '';
      let badgeClass = '';
      let summaryText = '';
      let actionBtnHtml = '';

      if (currentCondition === 'clean') {
        badgeTag = `Pristine Gallery Rank #${tInfo.rank}`;
        badgeClass = 'badge-tracker-status-clean';
        summaryText = `In pristine surveillance footage, target identity <strong>${tInfo.id}</strong> (${tInfo.dataset}) stands at <strong>Rank #${tInfo.rank}</strong> out of ${data.total_searched} candidates in the gallery. Switch to 50% Gaussian Blur to test degradation and verify reliability-weighted recovery.`;
        actionBtnHtml = `
          <button class="btn btn-secondary btn-sm" id="btn-quick-blur">
            <span>🔴 Test 50% Gaussian Blur</span>
          </button>
        `;
      } else if (currentMethod === 'mean') {
        badgeTag = `Pushed to Rank #${tInfo.rank} by Blur`;
        badgeClass = 'badge-tracker-status-corrupted';
        summaryText = `Target identity <strong>${tInfo.id}</strong> (${tInfo.dataset}) was pushed down to position <strong>#${tInfo.rank}</strong> under Uniform Mean Pooling because 50% Gaussian blur corrupted its temporal features.`;
        actionBtnHtml = `
          <button class="btn btn-primary btn-sm" id="btn-quick-recover">
            <span>⚡ Switch to Our Reliability (q<sub>j</sub>) to Recover</span>
          </button>
        `;
      } else {
        // currentCondition === 'corrupted' && currentMethod === 'weighted'
        const baselineRank = activePreset ? activePreset.corrupted_rank_mean : tInfo.rank;
        const gain = baselineRank - tInfo.rank;
        if (gain > 0) {
          badgeTag = `Recovered +${gain} Positions (Rank #${tInfo.rank})`;
          badgeClass = 'badge-tracker-status-recovered';
          summaryText = `Quality-weighted pooling (q<sub>j</sub>) suppressed blurred frames, lifting target <strong>${tInfo.id}</strong> from baseline Rank #${baselineRank} up to <strong>Rank #${tInfo.rank}</strong> (<strong>+${gain} ranks gain</strong> across the full ${data.total_searched}-video gallery).`;
        } else {
          badgeTag = `Rank #${tInfo.rank} / ${data.total_searched}`;
          badgeClass = 'badge-tracker-status-corrupted';
          summaryText = `Target <strong>${tInfo.id}</strong> stands at Rank #${tInfo.rank} under quality-weighted pooling.`;
        }
        actionBtnHtml = `
          <button class="btn btn-secondary btn-sm" id="btn-quick-compare">
            <span>⚡ View Side-by-Side Comparison</span>
          </button>
        `;
      }

      targetTrackerContainer.innerHTML = `
        <div class="target-tracker-card ${currentCondition === 'clean' ? 'tracker-clean' : ''}">
          <div class="tracker-left">
            <div class="tracker-badge-strip">
              <span class="badge-tracker-rank">★ Ground-Truth Target: Rank #${tInfo.rank} / ${data.total_searched}</span>
              <span class="badge-tracker-status ${badgeClass}">${badgeTag}</span>
              <span class="badge-tracker-score">Cosine Sim: +${Number(tInfo.score).toFixed(4)}</span>
            </div>
            <div class="tracker-summary">
              ${summaryText}
            </div>
            <div class="tracker-actions">
              ${actionBtnHtml}
            </div>
          </div>
          <div class="tracker-right">
            <div class="tracker-video-box">
              <video src="${tInfo.video_url}" autoplay loop muted playsinline controls preload="metadata"></video>
              <div class="tracker-rank-tag">Rank #${tInfo.rank}</div>
              <div class="tracker-dataset-tag">${tInfo.dataset} &bull; ${tInfo.id}</div>
            </div>
            <p class="tracker-caption">"${tInfo.caption}"</p>
          </div>
        </div>
      `;

      // Event listeners for quick action buttons
      const btnBlur = document.getElementById('btn-quick-blur');
      if (btnBlur) {
        btnBlur.addEventListener('click', () => {
          currentCondition = 'corrupted';
          btnCondCorrupted.classList.add('active');
          btnCondClean.classList.remove('active');
          runSearch();
        });
      }
      const btnRecover = document.getElementById('btn-quick-recover');
      if (btnRecover) {
        btnRecover.addEventListener('click', () => {
          currentMethod = 'weighted';
          btnMethodWeighted.classList.add('active');
          btnMethodMean.classList.remove('active');
          runSearch();
        });
      }
      const btnCompare = document.getElementById('btn-quick-compare');
      if (btnCompare) {
        btnCompare.addEventListener('click', () => {
          toggleSideBySide(true);
        });
      }
    } else {
      targetTrackerContainer.classList.add('hidden');
      targetTrackerContainer.innerHTML = '';
    }

    // 2. Status Banner Logic
    if (data.is_custom_neural) {
      recoveryBanner.className = 'recovery-status-banner';
      recoveryBanner.querySelector('.status-icon').textContent = '⚡';
      const topScore = data.results && data.results.length > 0 ? (data.results[0].score).toFixed(4) : '0';
      const methodLabel = data.method === 'weighted' ? 'Quality-Weighted (q_j)' : 'Baseline Mean-Pooled';
      recoveryText.innerHTML = `<strong>Neural Retrieval (${methodLabel}):</strong> Evaluated across <strong>${data.total_searched || 820} video tracklets</strong> using <strong>bert-base-uncased + MFGF [${data.condition}]</strong> &bull; Top Cosine Sim: <strong>${topScore}</strong>`;
    } else if (data.target_info) {
      const tInfo = data.target_info;
      const tRank = tInfo.rank;
      const tId = tInfo.id;
      const tScore = Number(tInfo.score).toFixed(4);

      if (currentCondition === 'clean') {
        if (tRank === 1) {
          recoveryBanner.className = 'recovery-status-banner';
          recoveryBanner.querySelector('.status-icon').textContent = '✓';
          recoveryText.innerHTML = `<strong>✓ Clean Retrieval:</strong> Ground-Truth Target <strong>${tId}</strong> identified at <strong>Rank #1</strong>! (Cos Sim: +${tScore})`;
        } else if (tInfo.in_top_k) {
          recoveryBanner.className = 'recovery-status-banner';
          recoveryBanner.querySelector('.status-icon').textContent = '✓';
          recoveryText.innerHTML = `<strong>✓ Clean Retrieval:</strong> Target <strong>${tId}</strong> located at <strong>Rank #${tRank}</strong> in Top-6 Grid (Cos Sim: +${tScore})`;
        } else {
          recoveryBanner.className = 'recovery-status-banner banner-info';
          recoveryBanner.querySelector('.status-icon').textContent = 'ℹ';
          recoveryText.innerHTML = `<strong>ℹ Clean Gallery Baseline:</strong> Target <strong>${tId}</strong> stands at <strong>Rank #${tRank} / ${data.total_searched}</strong> under pristine footage (Cos Sim: +${tScore})`;
        }
      } else if (currentMethod === 'mean') {
        recoveryBanner.className = 'recovery-status-banner banner-dropped';
        recoveryBanner.querySelector('.status-icon').textContent = '⚠';
        if (tInfo.in_top_k) {
          recoveryText.innerHTML = `<strong>⚠ Quality Degradation:</strong> Target <strong>${tId}</strong> dropped to <strong>Rank #${tRank}</strong> under 50% Blur (Baseline Mean Failed &bull; Visible in Top-6 Grid)`;
        } else {
          recoveryText.innerHTML = `<strong>⚠ Quality Degradation:</strong> Target <strong>${tId}</strong> dropped to <strong>Rank #${tRank} / ${data.total_searched}</strong> (Pushed outside Top-6 grid by blur &bull; Switch to Our Reliability (q_j) to recover!)`;
        }
      } else {
        // Corrupted + Weighted (q_j)
        const baselineRank = activePreset ? activePreset.corrupted_rank_mean : tRank;
        const gain = baselineRank - tRank;
        if (tRank === 1) {
          recoveryBanner.className = 'recovery-status-banner';
          recoveryBanner.querySelector('.status-icon').textContent = '✓';
          recoveryText.innerHTML = `<strong>✓ Target Recovered:</strong> Person <strong>${tId}</strong> recovered to <strong>Rank #1</strong> by Quality-Weighting (q_j)${gain > 0 ? ` (up from Rank #${baselineRank} under blur mean, +${gain} ranks recovered!)` : ''} (Cos Sim: +${tScore})`;
        } else if (tInfo.in_top_k) {
          recoveryBanner.className = 'recovery-status-banner';
          recoveryBanner.querySelector('.status-icon').textContent = '✓';
          recoveryText.innerHTML = `<strong>✓ Target Recovered:</strong> Person <strong>${tId}</strong> recovered into Top-6 at <strong>Rank #${tRank}</strong> by Quality-Weighting (q_j)${gain > 0 ? ` (up from Rank #${baselineRank}, +${gain} ranks recovered!)` : ''}`;
        } else if (gain > 0) {
          recoveryBanner.className = 'recovery-status-banner';
          recoveryBanner.querySelector('.status-icon').textContent = '⚡';
          recoveryText.innerHTML = `<strong>⚡ Deep Recovery:</strong> Target <strong>${tId}</strong> recovered from Rank #${baselineRank} ➔ <strong>Rank #${tRank}</strong> (+${gain} ranks gained by q_j across ${data.total_searched} videos!)`;
        } else {
          recoveryBanner.className = 'recovery-status-banner banner-dropped';
          recoveryBanner.querySelector('.status-icon').textContent = '⚠';
          recoveryText.innerHTML = `Target <strong>${tId}</strong> stands at <strong>Rank #${tRank} / ${data.total_searched}</strong> (Cos Sim: +${tScore})`;
        }
      }
    } else if (data.target_rank === 1) {
      recoveryBanner.className = 'recovery-status-banner';
      recoveryBanner.querySelector('.status-icon').textContent = '✓';
      recoveryText.textContent = `Target Identified at Rank #1`;
    } else {
      recoveryBanner.className = 'recovery-status-banner banner-dropped';
      recoveryBanner.querySelector('.status-icon').textContent = '⚠';
      recoveryText.textContent = `Target at Rank #${data.target_rank}`;
    }
  }

  // Helper to Render Video Cards
  function renderVideoCards(candidates, container) {
    container.innerHTML = '';
    candidates.forEach(cand => {
      const card = document.createElement('div');
      let targetClass = '';
      let rankBadgeClass = '';
      if (cand.is_target) {
        if (cand.rank === 1) {
          targetClass = 'target-match target-rank-1';
          rankBadgeClass = 'rank-badge-target';
        } else {
          targetClass = 'target-match target-dropped';
          rankBadgeClass = 'rank-badge-target-dropped';
        }
      }
      card.className = `video-card ${targetClass}`.trim();
      
      const rankLabel = cand.badge_label || (cand.is_target && cand.rank === 1 ? '★ Rank #1 (True Target)' : `Rank #${cand.rank}`);

      card.innerHTML = `
        <div class="video-card-top">
          <video src="${cand.video_url}" autoplay loop muted playsinline controls preload="metadata"></video>
          <div class="rank-overlay-badge ${rankBadgeClass}">${rankLabel}</div>
          <div class="dataset-overlay-badge">${cand.dataset} · ${cand.id}</div>
        </div>
        <div class="video-card-bottom">
          <div class="score-row">
            <span class="score-label">Cosine Similarity</span>
            <span class="score-value">${cand.score !== undefined ? (cand.score >= 0 ? '+' : '') + Number(cand.score).toFixed(4) : '+0.0000'}</span>
          </div>
          <div class="score-bar-bg">
            <div class="score-bar-fill" style="width: ${Math.max(5, Math.min(100, (cand.score !== undefined ? cand.score : 0.5) * 100))}%;"></div>
          </div>
          <p class="caption-snippet">"${cand.caption}"</p>
        </div>
      `;
      container.appendChild(card);
    });
  }

  // 9. Render Frame Quality (q_j) Timeline
  function renderQualityBars(qValues, weights) {
    qualityBarsContainer.innerHTML = '';
    qValues.forEach((q, idx) => {
      const col = document.createElement('div');
      col.className = 'frame-bar-col';
      const isHigh = q >= 0.5;
      const heightPercent = Math.round(q * 100);

      col.innerHTML = `
        <div class="bar-track">
          <div class="bar-fill ${isHigh ? 'bar-fill-high' : 'bar-fill-low'}" style="height: ${heightPercent}%;"></div>
        </div>
        <span class="frame-label">F${idx+1}</span>
        <span class="frame-score ${isHigh ? 'score-high' : 'score-low'}">${q.toFixed(2)}</span>
      `;
      qualityBarsContainer.appendChild(col);
    });

    const isCorrupted = qValues.some(q => q < 0.5);
    const expText = document.getElementById('quality-explanation-text');
    if (isCorrupted) {
      expText.innerHTML = `
        <strong>Blur Detected:</strong> Frames 5 to 8 exhibit severe Gaussian blur. The unsupervised <strong>Frame Quality Estimator</strong> downweights their contribution from <strong>~0.82 down to ~0.22</strong> (w<sub>j</sub> down to ${weights[4]}), preventing blurred frames from corrupting the person representation!
      `;
    } else {
      expText.innerHTML = `
        <strong>Pristine Footage:</strong> All 8 frames exhibit sharp edge gradients and balanced illumination (q<sub>j</sub> &approx; <strong>0.78 &ndash; 0.85</strong>). Quality weights are distributed uniformly, preserving natural temporal feature aggregation.
      `;
    }
  }

  // 10. Render Benchmark Analytics Tab
  function renderBenchmarks(datasetKey) {
    if (!benchmarkData || !benchmarkData[datasetKey]) return;
    const data = benchmarkData[datasetKey];

    // KPIs
    document.getElementById('kpi-r1').textContent = `${data.corrupted.weighted.r1.toFixed(2)}%`;
    document.getElementById('kpi-r1-gain').textContent = `+${data.relative_gain_r1}% rel`;
    document.getElementById('kpi-r1-sub').textContent = `Baseline: ${data.corrupted.mean_pool.r1.toFixed(2)}% ± ${data.corrupted.mean_pool.r1_std.toFixed(2)}%`;

    document.getElementById('kpi-r5').textContent = `${data.corrupted.weighted.r5.toFixed(2)}%`;
    document.getElementById('kpi-r5-gain').textContent = `+${data.relative_gain_r5}% rel`;
    document.getElementById('kpi-r5-sub').textContent = `Baseline: ${data.corrupted.mean_pool.r5.toFixed(2)}% ± ${data.corrupted.mean_pool.r5_std.toFixed(2)}%`;

    document.getElementById('kpi-r10').textContent = `${data.corrupted.weighted.r10.toFixed(2)}%`;
    document.getElementById('kpi-r10-gain').textContent = `+${data.relative_gain_r10}% rel`;
    document.getElementById('kpi-r10-sub').textContent = `Baseline: ${data.corrupted.mean_pool.r10.toFixed(2)}% ± ${data.corrupted.mean_pool.r10_std.toFixed(2)}%`;

    document.getElementById('kpi-map').textContent = `${data.corrupted.weighted.map.toFixed(2)}%`;
    document.getElementById('kpi-map-gain').textContent = `+${data.relative_gain_map}% rel`;
    document.getElementById('kpi-map-sub').textContent = `Baseline: ${data.corrupted.mean_pool.map.toFixed(2)}% ± ${data.corrupted.mean_pool.map_std.toFixed(2)}%`;

    document.getElementById('chart-section-title').textContent = `${data.name} — 5-Seed Comparative Retrieval Performance`;

    // Visual Bar Charts
    const chartContainer = document.getElementById('chart-bars-container');
    chartContainer.innerHTML = '';

    const metrics = [
      { key: 'r1', label: 'Rank-1 Retrieval Accuracy' },
      { key: 'r5', label: 'Rank-5 Retrieval Accuracy' },
      { key: 'r10', label: 'Rank-10 Retrieval Accuracy' },
      { key: 'map', label: 'Mean Average Precision (mAP)' }
    ];

    metrics.forEach(m => {
      const meanVal = data.corrupted.mean_pool[m.key];
      const weightVal = data.corrupted.weighted[m.key];
      const maxVal = Math.max(meanVal, weightVal, 1) * 1.35;

      const group = document.createElement('div');
      group.className = 'chart-bar-group';
      group.innerHTML = `
        <div class="chart-label-row">
          <span>${m.label} (Under 50% Gaussian Blur)</span>
          <span style="color: var(--accent-cyan); font-family: var(--font-mono);">+${((weightVal - meanVal) / meanVal * 100).toFixed(1)}% Relative Gain</span>
        </div>
        <div class="chart-bar-pair">
          <div class="chart-single-bar">
            <span class="bar-method-label">Uniform Mean Pool</span>
            <div class="bar-fill-track">
              <div class="bar-fill-colored bar-mean" style="width: ${(meanVal / maxVal * 100).toFixed(1)}%;">
                ${meanVal.toFixed(2)}%
              </div>
            </div>
          </div>
          <div class="chart-single-bar">
            <span class="bar-method-label">Ours (q<sub>j</sub> Weighting)</span>
            <div class="bar-fill-track">
              <div class="bar-fill-colored bar-weighted" style="width: ${(weightVal / maxVal * 100).toFixed(1)}%;">
                ${weightVal.toFixed(2)}%
              </div>
            </div>
          </div>
        </div>
      `;
      chartContainer.appendChild(group);
    });

    // Populate Table
    const tbody = document.getElementById('benchmarks-table-body');
    tbody.innerHTML = `
      <tr>
        <td><strong>Clean Test Data</strong></td>
        <td>Uniform Mean Pool</td>
        <td>${data.clean.mean_pool.r1.toFixed(2)}% ± ${data.clean.mean_pool.r1_std.toFixed(2)}%</td>
        <td>${data.clean.mean_pool.r5.toFixed(2)}% ± ${data.clean.mean_pool.r5_std.toFixed(2)}%</td>
        <td>${data.clean.mean_pool.r10.toFixed(2)}% ± ${data.clean.mean_pool.r10_std.toFixed(2)}%</td>
        <td>${data.clean.mean_pool.map.toFixed(2)}% ± ${data.clean.mean_pool.map_std.toFixed(2)}%</td>
      </tr>
      <tr>
        <td><strong>Clean Test Data</strong></td>
        <td><strong>Ours (q<sub>j</sub> Weighting)</strong></td>
        <td>${data.clean.weighted.r1.toFixed(2)}% ± ${data.clean.weighted.r1_std.toFixed(2)}%</td>
        <td>${data.clean.weighted.r5.toFixed(2)}% ± ${data.clean.weighted.r5_std.toFixed(2)}%</td>
        <td>${data.clean.weighted.r10.toFixed(2)}% ± ${data.clean.weighted.r10_std.toFixed(2)}%</td>
        <td>${data.clean.weighted.map.toFixed(2)}% ± ${data.clean.weighted.map_std.toFixed(2)}%</td>
      </tr>
      <tr>
        <td><strong style="color: #f87171;">Corrupted 50% Blur</strong></td>
        <td>Uniform Mean Pool</td>
        <td>${data.corrupted.mean_pool.r1.toFixed(2)}% ± ${data.corrupted.mean_pool.r1_std.toFixed(2)}%</td>
        <td>${data.corrupted.mean_pool.r5.toFixed(2)}% ± ${data.corrupted.mean_pool.r5_std.toFixed(2)}%</td>
        <td>${data.corrupted.mean_pool.r10.toFixed(2)}% ± ${data.corrupted.mean_pool.r10_std.toFixed(2)}%</td>
        <td>${data.corrupted.mean_pool.map.toFixed(2)}% ± ${data.corrupted.mean_pool.map_std.toFixed(2)}%</td>
      </tr>
      <tr class="highlight-row">
        <td><strong style="color: #34d399;">Corrupted 50% Blur</strong></td>
        <td><strong>Ours (q<sub>j</sub> Weighting)</strong></td>
        <td>${data.corrupted.weighted.r1.toFixed(2)}% ± ${data.corrupted.weighted.r1_std.toFixed(2)}%</td>
        <td>${data.corrupted.weighted.r5.toFixed(2)}% ± ${data.corrupted.weighted.r5_std.toFixed(2)}%</td>
        <td>${data.corrupted.weighted.r10.toFixed(2)}% ± ${data.corrupted.weighted.r10_std.toFixed(2)}%</td>
        <td>${data.corrupted.weighted.map.toFixed(2)}% ± ${data.corrupted.weighted.map_std.toFixed(2)}%</td>
      </tr>
    `;
  }

  // Start app
  init();
});
