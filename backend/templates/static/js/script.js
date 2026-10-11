document.addEventListener('DOMContentLoaded', () => {
  const fileInput = document.getElementById('file-input');
  const uploadArea = document.getElementById('upload-area');
  const fileInfo = document.getElementById('file-info');
  const analysisResults = document.getElementById('analysis-results');
  const resultsContainer = document.getElementById('results-container');
  const selectButton = document.querySelector('.select-file-btn');
  const detectButton = document.querySelector('.detect-btn');
  const summaryCheckbox = document.getElementById('generate-ai-summary');
  const acceptedExtensions = ['.jpg', '.jpeg', '.png', '.webp', '.wav', '.mp3', '.m4a', '.mp4', '.mov', '.webm'];
  const API_KEY_STORAGE = 'fusiontrace.apiKey';
  const TECHNICAL_SOURCES = ['file-inspection', 'image-metadata'];
  const STATUS_LABELS = { ok: 'Ran', disabled: 'Disabled', error: 'Error', skipped: 'Skipped' };

  let currentScanData = null;
  let currentScanId = null;
  let heatmapObjectUrl = null;
  let heatmapDataUrl = null;

  // ── API key (optional; only when the server enables API keys) ─────────────
  const apiKeyPanel = document.getElementById('api-key-panel');
  const apiKeyInput = document.getElementById('api-key-input');
  const apiKeyStatus = document.getElementById('api-key-status');

  function getApiKey() {
    try { return sessionStorage.getItem(API_KEY_STORAGE) || ''; } catch (e) { return ''; }
  }
  function setApiKey(value) {
    try {
      if (value) sessionStorage.setItem(API_KEY_STORAGE, value);
      else sessionStorage.removeItem(API_KEY_STORAGE);
    } catch (e) { /* storage blocked: the key lasts until reload */ }
  }
  function showApiKeyPanel(message) {
    if (!apiKeyPanel) return;
    apiKeyPanel.classList.remove('hidden');
    if (message && apiKeyStatus) apiKeyStatus.textContent = message;
    apiKeyInput?.focus();
  }
  function refreshApiKeyStatus() {
    if (!apiKeyStatus) return;
    apiKeyStatus.textContent = getApiKey()
      ? 'A key is set for this browser tab. It is sent in the X-API-Key header and forgotten when the tab closes.'
      : 'This server requires an API key. It is kept only in this browser tab and sent in the X-API-Key header.';
  }
  document.getElementById('api-key-save')?.addEventListener('click', () => {
    const value = (apiKeyInput?.value || '').trim();
    if (!value) return notify('Enter an API key first.', 'error');
    setApiKey(value);
    apiKeyInput.value = '';
    refreshApiKeyStatus();
    fetchHistory();
  });
  document.getElementById('api-key-clear')?.addEventListener('click', () => {
    setApiKey('');
    refreshApiKeyStatus();
    fetchHistory();
  });
  refreshApiKeyStatus();

  class ApiError extends Error {
    constructor(message, status) { super(message); this.status = status; }
  }

  async function apiFetch(url, options = {}) {
    const headers = new Headers(options.headers || {});
    const key = getApiKey();
    if (key) headers.set('X-API-Key', key);
    const response = await fetch(url, { ...options, headers });
    if (response.status === 401) {
      showApiKeyPanel('This server requires a valid API key. Enter it above and try again.');
      throw new ApiError('A valid API key is required.', 401);
    }
    return response;
  }

  async function errorDetail(response, fallback) {
    try {
      const body = await response.json();
      return typeof body.detail === 'string' ? body.detail : fallback;
    } catch (e) {
      return fallback;
    }
  }

  // ── File selection ─────────────────────────────────────────────────────────
  const clearHeatmap = () => {
    if (heatmapObjectUrl) URL.revokeObjectURL(heatmapObjectUrl);
    heatmapObjectUrl = null;
    heatmapDataUrl = null;
  };

  const reset = () => {
    fileInput.value = '';
    uploadArea.classList.remove('hidden');
    fileInfo.classList.add('hidden');
    analysisResults.classList.add('hidden');
    document.getElementById('feedback-section')?.classList.add('hidden');
    currentScanData = null;
    currentScanId = null;
    clearHeatmap();
  };

  const selectFile = () => fileInput.click();
  const showSelected = () => {
    const file = fileInput.files[0];
    if (!file) return;
    const dot = file.name.lastIndexOf('.');
    const extension = dot >= 0 ? file.name.slice(dot).toLowerCase() : '';
    if (!acceptedExtensions.includes(extension)) return notify('Unsupported media. Use JPG, PNG, WebP, WAV, MP3, M4A, MP4, MOV or WebM.', 'error');
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-meta').textContent = `${formatSize(file.size)} • ${file.type || 'unknown type'}`;
    uploadArea.classList.add('hidden');
    fileInfo.classList.remove('hidden');
  };

  selectButton.addEventListener('click', selectFile);
  fileInput.addEventListener('change', showSelected);
  uploadArea.addEventListener('dragover', event => { event.preventDefault(); uploadArea.classList.add('dragging'); });
  uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragging'));
  uploadArea.addEventListener('drop', event => { event.preventDefault(); uploadArea.classList.remove('dragging'); fileInput.files = event.dataTransfer.files; showSelected(); });
  document.querySelector('.remove-file-btn').addEventListener('click', reset);
  document.querySelector('.try-another-btn').addEventListener('click', reset);

  detectButton.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;
    detectButton.disabled = true;
    detectButton.textContent = 'Submitting…';
    try {
      const form = new FormData();
      form.append('file', file);
      form.append('generate_ai_summary', summaryCheckbox && !summaryCheckbox.disabled && summaryCheckbox.checked ? 'true' : 'false');
      const scanResponse = await apiFetch('/api/scans', { method: 'POST', body: form });
      if (!scanResponse.ok) throw new Error(await errorDetail(scanResponse, 'Unable to start the scan.'));
      const scan = await scanResponse.json();
      currentScanId = scan.id;
      clearHeatmap();
      fileInfo.classList.add('hidden'); analysisResults.classList.remove('hidden');
      showProgress('queued');
      document.getElementById('feedback-section')?.classList.add('hidden');
      fetchHistory();
      await poll(scan.id);
    } catch (error) {
      notify(error.message, 'error');
      if (!currentScanId) fileInfo.classList.remove('hidden');
    } finally { detectButton.disabled = false; detectButton.textContent = 'Detect Now'; }
  });

  function showProgress(status) {
    const message = status === 'queued'
      ? 'Queued — waiting for a free analysis worker. Scans run one after another on this server.'
      : 'Analysis is running. Video and long audio files take longer.';
    resultsContainer.innerHTML = `<p class="file-info-result">${escapeHtml(message)}</p>`;
  }

  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

  async function poll(scanId) {
    const startedAt = Date.now();
    const maxWaitMs = 20 * 60 * 1000;
    let failures = 0;
    let lastStatus = null;
    while (Date.now() - startedAt < maxWaitMs) {
      await sleep(Date.now() - startedAt < 60000 ? 2000 : 5000);
      if (currentScanId !== scanId) return;  // the user moved on
      let scan;
      try {
        const response = await apiFetch(`/api/scans/${scanId}`);
        if (response.status === 404) throw new ApiError('This scan is no longer available.', 404);
        if (!response.ok) throw new Error(`Server error ${response.status}`);
        scan = await response.json();
        failures = 0;
      } catch (err) {
        if (err instanceof ApiError) throw err;
        failures += 1;  // transient network glitch or server restart
        if (failures > 5) throw new Error('Server unreachable. Please wait a moment and check the scan history.');
        continue;
      }
      if (scan.status === 'completed') {
        render(scan);
        fetchHistory();
        if (enrichmentsPending(scan.report)) pollEnrichments(scanId);
        return;
      }
      if (scan.status === 'failed') {
        fetchHistory();
        resultsContainer.innerHTML = `<p class="file-info-result">${escapeHtml(scan.error || 'Analysis failed.')}</p>`;
        return;
      }
      if (scan.status !== lastStatus) { showProgress(scan.status); lastStatus = scan.status; }
    }
    throw new Error('The scan is taking longer than expected. Check the scan history later for the result.');
  }

  function enrichmentsPending(report) {
    return !!report && (report.ai_summary_status === 'pending' || report.heatmap_status === 'pending');
  }

  async function pollEnrichments(scanId) {
    for (let i = 0; i < 24; i++) {
      await sleep(5000);
      if (currentScanId !== scanId) return;
      let scan;
      try {
        const response = await apiFetch(`/api/scans/${scanId}`);
        if (!response.ok) continue;
        scan = await response.json();
      } catch (err) {
        if (err instanceof ApiError) return;
        continue;
      }
      const r = scan.report;
      if (!r) return;
      currentScanData = scan;
      updateSummaryBox(r);
      updateHeatmapPanel(scan);
      if (!enrichmentsPending(r)) return;
    }
  }

  // ── Signal formatting (shared by the page and the PDF) ─────────────────────
  function modelName(item) {
    const model = item.model;
    if (!model) return TECHNICAL_SOURCES.includes(item.source) ? 'Technical file check (no model)' : 'No model ran';
    return model.display_name || model.name || 'Unnamed model';
  }

  function isHeuristic(item) { return !!(item.model && item.model.heuristic); }

  function statusOf(item) { return item.status || 'ok'; }

  function riskText(item) {
    return item.risk !== null && item.risk !== undefined ? `${Math.round(item.risk * 100)}%` : 'N/A';
  }

  function signalOutcome(item) {
    const d = item.details || {};
    if (statusOf(item) !== 'ok') return d.reason || 'This signal did not run.';
    if (item.source === 'video-frame-model' && d.per_frame_scores) {
      let text = `${d.frames_analyzed} frame(s) sampled — average risk ${d.average_risk_score}`;
      if (d.peak_frame) text += `; highest at ${d.peak_frame.timestamp_seconds}s (${d.peak_frame.risk_score})`;
      return text;
    }
    if (d.prediction === 'Fake') return 'Signal leans synthetic / manipulated';
    if (d.prediction === 'Real') return 'No elevated synthetic signal';
    return `${riskText(item)} risk signal`;
  }

  function pdfDetail(item) {
    const d = item.details || {};
    if (item.source === 'file-inspection') {
      return `${d.mime_type || ''} · ${d.size_bytes ? (d.size_bytes / 1024).toFixed(1) + ' KB' : ''} · SHA-256: ${(d.sha256 || '').slice(0, 18)}…`;
    }
    if (item.source === 'image-metadata') {
      return `Format: ${d.format || 'N/A'} · ${d.dimensions?.[0] || 0}×${d.dimensions?.[1] || 0}px · EXIF: ${d.exif_present ? 'Present' : 'Absent'}`;
    }
    if (statusOf(item) !== 'ok') return d.reason || 'Did not run';
    if (item.source === 'video-frame-model') {
      return `${d.frames_analyzed || 0} frames sampled · avg ${d.average_risk_score} · peak ${d.peak_frame?.timestamp_seconds}s (${d.peak_frame?.risk_score})`;
    }
    if (d.prediction) return `${d.prediction}${d.confidence ? ' (' + d.confidence + ')' : ''}`;
    return '';
  }

  // ── Main On-Screen Render Function ─────────────────────────────────────────
  function render(scan) {
    currentScanData = scan;
    const report = scan.report || {};
    const assessment = report.assessment || {};
    const isHigh = assessment.risk_rating === 'high';
    const isMed = assessment.risk_rating === 'medium';

    const evidenceItems = (report.evidence || [])
      .filter(item => !TECHNICAL_SOURCES.includes(item.source))
      .map(item => {
        const status = statusOf(item);
        const statusBadge = status === 'ok'
          ? `<span class="signal-badge signal-risk">risk ${escapeHtml(riskText(item))}</span>`
          : `<span class="signal-badge signal-status-${escapeHtml(status)}">${escapeHtml(STATUS_LABELS[status] || status)}</span>`;
        const heuristicBadge = isHeuristic(item) ? '<span class="signal-badge signal-heuristic" title="Rule-based check, not a trained model">Heuristic</span>' : '';
        return `<li class="signal-item signal-${escapeHtml(status)}">
          <div class="signal-head"><strong>${escapeHtml(item.label)}</strong> ${statusBadge} ${heuristicBadge}</div>
          <div class="signal-model">${isHeuristic(item) ? 'Heuristic' : 'Model'}: ${escapeHtml(modelName(item))}</div>
          <div class="signal-outcome">${escapeHtml(signalOutcome(item))}</div>
        </li>`;
      }).join('');

    resultsContainer.innerHTML = `
      <div class="results-box ${isHigh ? 'bg-error-light' : isMed ? 'bg-warn-light' : 'bg-success-light'}">

        <!-- Verdict Header -->
        <div class="result ${isHigh ? 'fake' : isMed ? 'medium' : 'real'}">
          <span class="result-text">${escapeHtml((assessment.risk_rating || 'UNKNOWN').toUpperCase())} authenticity risk</span>
        </div>
        <p class="file-info-result">Automated assessment for <strong>${escapeHtml(scan.filename)}</strong>. These are risk signals, not proof of manipulation or authenticity.</p>
        <p class="file-info-result"><strong>Recommended action:</strong> ${escapeHtml(assessment.recommended_action || 'Manual review required.')}</p>

        <!-- Score Bar -->
        <div class="confidence-score">
          <div class="score-label">
            <span>Evidence-weighted risk score</span>
            <span class="score-value">${escapeHtml(assessment.risk_score || 0)}%</span>
          </div>
          <div class="progress-bar">
            <div class="progress ${isHigh ? 'fake' : isMed ? 'medium' : 'real'}" style="width:${Number(assessment.risk_score) || 0}%"></div>
          </div>
        </div>

        <!-- Detector Signals -->
        <div class="anomalies">
          <div class="font-medium mb-2">Detector signals</div>
          <ul class="signal-list">${evidenceItems}</ul>
        </div>

        <!-- Explanation heatmap (image scans) -->
        <div id="heatmap-panel" class="heatmap-panel hidden"></div>

        <!-- AI Summary -->
        <div id="ai-summary-slot"></div>

        <div class="action-buttons">
          <button class="btn-primary pdf-btn" id="download-pdf-btn">📄 Download PDF Report</button>
        </div>

        <p class="file-info-result retention-note">Retention: uploaded media and the heatmap are deleted after ${escapeHtml(report.retention_hours || 24)} hours; working files are deleted when the scan finishes.</p>

      </div>`;

    updateSummaryBox(report);
    updateHeatmapPanel(scan);

    document.getElementById('download-pdf-btn')?.addEventListener('click', () => downloadPDFReport(currentScanData));

    const feedbackSection = document.getElementById('feedback-section');
    if (feedbackSection) {
      feedbackSection.classList.remove('hidden');
      document.getElementById('feedback-form').reset();
      document.getElementById('feedback-status').textContent = '';
      const hList = document.getElementById('feedback-history-list');
      const hContainer = document.getElementById('feedback-history-container');
      if (scan.feedback_history && scan.feedback_history.length > 0) {
        hContainer.classList.remove('hidden');
        hList.innerHTML = scan.feedback_history.map(f => `<li><strong>${escapeHtml(f.label.replace('_', ' '))}</strong> (${escapeHtml(new Date(f.created_at).toLocaleString())}): ${escapeHtml(f.notes || 'No notes')}</li>`).join('');
      } else {
        hContainer.classList.add('hidden');
      }
    }
  }

  function updateSummaryBox(report) {
    const slot = document.getElementById('ai-summary-slot');
    if (!slot) return;
    const note = '<span class="ai-summary-note">(summary of detector scores only — not an authenticity verdict)</span>';
    if (report.ai_summary) {
      slot.innerHTML = `<div class="ai-summary-box">
          <div class="ai-summary-label">🤖 AI Forensic Summary ${note}</div>
          <p class="ai-summary-text">${escapeHtml(report.ai_summary)}</p>
        </div>`;
    } else if (report.ai_summary_status === 'pending') {
      slot.innerHTML = `<div class="ai-summary-box ai-summary-loading">
          <div class="ai-summary-label">🤖 AI Forensic Summary</div>
          <p class="ai-summary-text ai-summary-muted">Generating summary…</p>
        </div>`;
    } else if (report.ai_summary_requested) {
      slot.innerHTML = `<div class="ai-summary-box ai-summary-loading">
          <div class="ai-summary-label">🤖 AI Forensic Summary</div>
          <p class="ai-summary-text ai-summary-muted">Summary unavailable. The scan result is unchanged.</p>
        </div>`;
    } else {
      slot.innerHTML = '';
    }
  }

  function heatmapCaption(report) {
    const model = report.heatmap_model;
    const name = model ? (model.display_name || model.name) : 'the image model';
    return `Produced by ${name}. Warm colours mark the regions that most influenced this model's score. It shows where the model looked, not proof of manipulation.`;
  }

  async function updateHeatmapPanel(scan) {
    const panel = document.getElementById('heatmap-panel');
    const report = scan.report || {};
    if (!panel) return;
    const status = report.heatmap_status || (report.heatmap_ready ? 'ready' : 'not_applicable');
    if (status === 'pending') {
      panel.classList.remove('hidden');
      panel.innerHTML = '<div class="heatmap-title">Explanation heatmap</div><p class="option-note">Generating the explanation heatmap…</p>';
      return;
    }
    if (status === 'expired') {
      panel.classList.remove('hidden');
      panel.innerHTML = '<div class="heatmap-title">Explanation heatmap</div><p class="option-note">The heatmap was deleted by the retention policy.</p>';
      return;
    }
    if (status !== 'ready' || !report.heatmap_ready) {
      panel.classList.add('hidden');
      panel.innerHTML = '';
      return;
    }
    if (panel.dataset.scanId === scan.id && heatmapObjectUrl) return;  // already shown
    try {
      const response = await apiFetch(`/api/scans/${scan.id}/heatmap`);
      if (!response.ok) throw new Error('Heatmap not available');
      const blob = await response.blob();
      clearHeatmap();
      heatmapObjectUrl = URL.createObjectURL(blob);
      heatmapDataUrl = await blobToDataUrl(blob);
      panel.dataset.scanId = scan.id;
      panel.classList.remove('hidden');
      panel.innerHTML = `<div class="heatmap-title">Explanation heatmap</div>
        <p class="option-note">${escapeHtml(heatmapCaption(report))}</p>
        <img class="heatmap-image" src="${heatmapObjectUrl}" alt="Explanation heatmap overlaid on the uploaded image">`;
    } catch (err) {
      panel.classList.remove('hidden');
      panel.innerHTML = '<div class="heatmap-title">Explanation heatmap</div><p class="option-note">The heatmap could not be loaded.</p>';
    }
  }

  function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(blob);
    });
  }

  // ── Standalone 1-Page PDF Generator ─────────────────────────────────────────
  function downloadPDFReport(scan) {
    if (!scan || !scan.report) return notify('Report data not loaded yet', 'error');
    const report = scan.report;
    const assessment = report.assessment || {};
    const isHigh = assessment.risk_rating === 'high';
    const isMed = assessment.risk_rating === 'medium';

    const ratingColor = isHigh ? '#dc2626' : isMed ? '#d97706' : '#16a34a';
    const ratingBg = isHigh ? '#fef2f2' : isMed ? '#fffbeb' : '#f0fdf4';
    const ratingBorder = isHigh ? '#fca5a5' : isMed ? '#fde68a' : '#86efac';
    const cell = 'padding: 5px 8px; border: 1px solid #cbd5e1;';

    const evidenceRows = (report.evidence || []).map(item => {
      const status = statusOf(item);
      const model = `${modelName(item)}${isHeuristic(item) ? ' (heuristic)' : ''}`;
      return `<tr>
        <td style="${cell} font-weight: 500; color: #1e293b;">${escapeHtml(item.label)}</td>
        <td style="${cell} color: #475569; font-size: 7.5pt;">${escapeHtml(model)}</td>
        <td style="${cell} color: #475569;">${escapeHtml(STATUS_LABELS[status] || status)}</td>
        <td style="${cell} color: ${ratingColor}; font-weight: 600;">${escapeHtml(riskText(item))}</td>
        <td style="${cell} font-weight: 500; color: #475569;">${escapeHtml(item.weight || 0)}</td>
        <td style="${cell} color: #475569; font-size: 7.5pt; font-weight: 400;">${escapeHtml(pdfDetail(item))}</td>
      </tr>`;
    }).join('');

    const frameItem = (report.evidence || []).find(i => i.source === 'video-frame-model');
    const frameScores = frameItem?.details?.per_frame_scores;
    const frameGridHtml = frameScores ? `
      <div style="margin-top: 10px;">
        <div style="font-size: 9.5pt; font-weight: 600; color: #1e293b; margin-bottom: 4px; border-bottom: 1px solid #e2e8f0; padding-bottom: 2px;">Sampled Video Frame Risk Timeline</div>
        <div style="display: flex; flex-wrap: wrap; gap: 4px;">
          ${frameScores.map(f => `
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 3px; padding: 3px 6px; text-align: center; min-width: 45px;">
              <div style="font-size: 7.5pt; font-weight: 600; color: #334155;">${escapeHtml(f.timestamp_seconds)}s</div>
              <div style="font-size: 7.5pt; color: #64748b;">${escapeHtml(f.risk_score)}</div>
            </div>
          `).join('')}
        </div>
      </div>` : '';

    const heatmapSrc = report.heatmap_ready && heatmapDataUrl && currentScanData?.id === scan.id ? heatmapDataUrl : null;
    const heatmapHtml = heatmapSrc ? `
      <div style="margin-top: 10px; page-break-inside: avoid;">
        <div style="font-size: 9.5pt; font-weight: 600; color: #1e293b; margin-bottom: 2px;">Explanation Heatmap</div>
        <div style="font-size: 7.5pt; color: #64748b; margin-bottom: 4px;">${escapeHtml(heatmapCaption(report))}</div>
        <img src="${heatmapSrc}" style="max-width: 320px; max-height: 200px; border-radius: 4px; border: 1px solid #cbd5e1; display: block;" />
      </div>` : '';

    const feedbackHtml = (scan.feedback_history && scan.feedback_history.length > 0) ? `
      <div style="margin-top: 10px;">
        <div style="font-size: 9.5pt; font-weight: 600; color: #1e293b; margin-bottom: 4px; border-bottom: 1px solid #e2e8f0; padding-bottom: 2px;">Human Review Audit Trail</div>
        <ul style="margin: 0; padding-left: 16px; font-size: 8pt; color: #334155;">
          ${scan.feedback_history.map(f => `<li><strong>${escapeHtml(f.label.replace('_', ' '))}</strong> (${escapeHtml(new Date(f.created_at).toLocaleString())}): ${escapeHtml(f.notes || 'No notes provided')}</li>`).join('')}
        </ul>
      </div>` : '';

    const scanDate = new Date(scan.completed_at || scan.created_at).toLocaleString();

    const reportHtml = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>FusionTrace Forensic Report - ${escapeHtml(scan.id)}</title>
  <style>
    @page { size: A4 portrait; margin: 8mm 12mm; }
    body {
      font-family: Arial, Helvetica, sans-serif;
      color: #0f172a;
      background: #ffffff;
      margin: 0;
      padding: 0;
      line-height: 1.35;
      font-size: 8.5pt;
      -webkit-font-smoothing: antialiased;
    }
    .header { display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid #4f46e5; padding-bottom: 6px; margin-bottom: 10px; }
    .title { font-size: 16pt; font-weight: 700; color: #4f46e5; letter-spacing: -0.3px; margin: 0; }
    .subtitle { font-size: 8pt; color: #64748b; margin-top: 1px; }
    .meta { font-size: 8pt; color: #475569; text-align: right; line-height: 1.3; }
    .verdict-card { background: ${ratingBg}; border: 1px solid ${ratingBorder}; border-radius: 5px; padding: 10px 12px; margin-bottom: 10px; }
    .verdict-badge { display: inline-block; background: ${ratingColor}; color: #ffffff; font-weight: 700; font-size: 9pt; text-transform: uppercase; padding: 2px 7px; border-radius: 3px; letter-spacing: 0.3px; }
    .verdict-text { font-size: 8pt; color: #334155; margin-top: 4px; margin-bottom: 2px; }
    .rec-text { font-size: 8pt; color: #0f172a; font-weight: 600; margin-bottom: 6px; }
    .score-container { margin-top: 4px; }
    .score-label { display: flex; justify-content: space-between; font-size: 8pt; font-weight: 700; color: #1e293b; margin-bottom: 2px; }
    .progress-bar-bg { background: #e2e8f0; height: 6px; border-radius: 3px; overflow: hidden; width: 100%; }
    .progress-bar-fill { background: ${ratingColor}; height: 100%; width: ${Number(assessment.risk_score) || 0}%; }
    .ai-box { background: #eff6ff; border-left: 3.5px solid #3b82f6; border-radius: 4px; padding: 8px 10px; margin-top: 10px; margin-bottom: 10px; }
    .ai-title { font-weight: 700; color: #1e40af; font-size: 8.5pt; margin-bottom: 3px; }
    .ai-body { font-size: 8pt; color: #1e293b; margin: 0; line-height: 1.35; }
    .table-title { font-size: 9.5pt; font-weight: 700; color: #1e293b; margin-top: 10px; margin-bottom: 4px; border-bottom: 1px solid #e2e8f0; padding-bottom: 2px; }
    table { width: 100%; border-collapse: collapse; margin-top: 2px; font-size: 8pt; }
    th { background: #f8fafc; border: 1px solid #cbd5e1; padding: 4px 7px; text-align: left; font-weight: 700; color: #334155; }
    td { font-family: Arial, Helvetica, sans-serif !important; }
    .footer { margin-top: 14px; border-top: 1px solid #e2e8f0; padding-top: 6px; font-size: 7pt; color: #94a3b8; text-align: center; line-height: 1.3; }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <div class="title">FusionTrace</div>
      <div class="subtitle">Media Authenticity Forensic Report</div>
    </div>
    <div class="meta">
      <div><strong>Scan ID:</strong> ${escapeHtml(scan.id)}</div>
      <div><strong>Date:</strong> ${escapeHtml(scanDate)}</div>
      <div><strong>File:</strong> ${escapeHtml(scan.filename)} (${escapeHtml(scan.media_type)})</div>
    </div>
  </div>

  <div class="verdict-card">
    <span class="verdict-badge">${escapeHtml((assessment.risk_rating || 'UNKNOWN').toUpperCase())} AUTHENTICITY RISK</span>
    <div class="verdict-text">Automated risk assessment for <strong>${escapeHtml(scan.filename)}</strong>. These are risk signals, not proof.</div>
    <div class="rec-text">Recommended action: ${escapeHtml(assessment.recommended_action || 'Manual review required.')}</div>
    <div class="score-container">
      <div class="score-label">
        <span>Evidence-Weighted Risk Score</span>
        <span>${escapeHtml(assessment.risk_score || 0)}%</span>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill"></div>
      </div>
    </div>
  </div>

  ${report.ai_summary ? `
  <div class="ai-box">
    <div class="ai-title">🤖 AI Forensic Summary <span style="font-weight:400; font-size:7pt; color:#3b82f6;">(summary of detector scores only — not an authenticity verdict)</span></div>
    <div class="ai-body">${escapeHtml(report.ai_summary)}</div>
  </div>` : ''}

  <div class="table-title">Detector Signals & Evidence Breakdown</div>
  <table>
    <thead>
      <tr>
        <th style="width: 20%;">Signal</th>
        <th style="width: 22%;">Model</th>
        <th style="width: 8%;">Status</th>
        <th style="width: 8%;">Risk</th>
        <th style="width: 7%;">Weight</th>
        <th>Technical Detail</th>
      </tr>
    </thead>
    <tbody>
      ${evidenceRows}
    </tbody>
  </table>
  <div style="font-size: 7pt; color: #64748b; margin-top: 3px;">Signals marked "(heuristic)" are rule-based checks, not trained models. Signals that did not run carry no weight.</div>

  ${frameGridHtml}
  ${heatmapHtml}
  ${feedbackHtml}

  <div class="footer">
    <div>Retention: uploaded media and the heatmap are deleted after ${escapeHtml(report.retention_hours || 24)} hours; working files are deleted when the scan finishes.</div>
    <div>This report is generated by automated detectors and does not constitute proof of manipulation or authenticity.</div>
  </div>

  <script>
    window.onload = function() {
      setTimeout(function() {
        window.print();
      }, 250);
    };
  </script>
</body>
</html>`;

    const printIframe = document.createElement('iframe');
    printIframe.style.position = 'fixed';
    printIframe.style.right = '0';
    printIframe.style.bottom = '0';
    printIframe.style.width = '0';
    printIframe.style.height = '0';
    printIframe.style.border = '0';
    document.body.appendChild(printIframe);

    const doc = printIframe.contentWindow.document;
    doc.open();
    doc.write(reportHtml);
    doc.close();

    setTimeout(() => {
      if (document.body.contains(printIframe)) {
        document.body.removeChild(printIframe);
      }
    }, 10000);
  }

  // ── History & Feedback ──────────────────────────────────────────────────────
  async function fetchHistory() {
    const tbody = document.getElementById('history-tbody');
    if (!tbody) return;
    try {
      const response = await apiFetch('/api/scans');
      if (!response.ok) throw new Error(`Server error ${response.status}`);
      const scans = await response.json();
      if (!scans.length) {
        tbody.innerHTML = '<tr><td colspan="7">No scans yet.</td></tr>';
        return;
      }
      tbody.innerHTML = scans.map(s => {
        const date = new Date(s.created_at).toLocaleString();
        const feedback = s.latest_feedback ? s.latest_feedback.label.replace('_', ' ') : 'Pending';
        const rating = s.risk_rating ? s.risk_rating.toUpperCase() : 'N/A';
        return `<tr>
          <td>${escapeHtml(date)}</td>
          <td>${escapeHtml(s.filename)}</td>
          <td>${escapeHtml(s.media_type)}</td>
          <td>${escapeHtml(s.status)}</td>
          <td>${escapeHtml(rating)}</td>
          <td>${escapeHtml(feedback)}</td>
          <td><button class="btn-outline btn-sm open-scan-btn" data-id="${escapeHtml(s.id)}">Open</button></td>
        </tr>`;
      }).join('');
      document.querySelectorAll('.open-scan-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
          const id = e.target.getAttribute('data-id');
          await openScan(id);
        });
      });
    } catch (e) {
      const message = e instanceof ApiError && e.status === 401 ? 'Enter your API key above to see your scans.' : 'Scan history could not be loaded.';
      tbody.innerHTML = `<tr><td colspan="7">${escapeHtml(message)}</td></tr>`;
    }
  }

  async function openScan(id) {
    try {
      const res = await apiFetch(`/api/scans/${id}`);
      if (!res.ok) throw new Error(await errorDetail(res, 'Failed to load scan'));
      const scan = await res.json();
      reset();
      currentScanId = scan.id;
      uploadArea.classList.add('hidden');
      analysisResults.classList.remove('hidden');
      if (scan.status === 'completed') {
        render(scan);
        if (enrichmentsPending(scan.report)) pollEnrichments(scan.id);
      } else if (scan.status === 'failed') {
        resultsContainer.innerHTML = `<p class="file-info-result">${escapeHtml(scan.error || 'Analysis failed.')}</p>`;
      } else {
        showProgress(scan.status);
        poll(scan.id).catch(err => notify(err.message, 'error'));
      }
      document.getElementById('upload-detect').scrollIntoView({ behavior: 'smooth' });
    } catch (e) { notify(e.message, 'error'); }
  }

  const feedbackForm = document.getElementById('feedback-form');
  if (feedbackForm) {
    feedbackForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (!currentScanData) return;
      const formData = new FormData(feedbackForm);
      const label = formData.get('feedback_label');
      const notes = formData.get('notes');
      if (!label) return notify('Please select a feedback label', 'error');
      const submitBtn = document.getElementById('submit-feedback-btn');
      submitBtn.disabled = true;
      try {
        const res = await apiFetch(`/api/scans/${currentScanData.id}/feedback`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ label, notes })
        });
        if (!res.ok) throw new Error(await errorDetail(res, 'Failed to submit feedback'));
        document.getElementById('feedback-status').innerHTML = '<span style="color: green;">Feedback saved!</span>';
        await openScan(currentScanData.id);
        fetchHistory();
      } catch (err) {
        document.getElementById('feedback-status').innerHTML = `<span style="color: red;">${escapeHtml(err.message)}</span>`;
      } finally { submitBtn.disabled = false; }
    });
  }

  function escapeHtml(value) { const element = document.createElement('div'); element.textContent = String(value); return element.innerHTML; }
  function formatSize(bytes) { return bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1048576).toFixed(1)} MB`; }
  function notify(message, type) { alert(`${type === 'error' ? 'Error: ' : ''}${message}`); }

  fetchHistory();
});
