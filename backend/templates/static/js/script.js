document.addEventListener('DOMContentLoaded', () => {
  const fileInput = document.getElementById('file-input');
  const uploadArea = document.getElementById('upload-area');
  const fileInfo = document.getElementById('file-info');
  const analysisResults = document.getElementById('analysis-results');
  const resultsContainer = document.getElementById('results-container');
  const selectButton = document.querySelector('.select-file-btn');
  const detectButton = document.querySelector('.detect-btn');
  const accepted = ['image/jpeg', 'image/png', 'image/webp', 'audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/mp4', 'video/mp4', 'video/webm', 'video/quicktime'];

  let currentScanData = null;
  let currentScanId = null;

  const reset = () => {
    fileInput.value = '';
    uploadArea.classList.remove('hidden');
    fileInfo.classList.add('hidden');
    analysisResults.classList.add('hidden');
    document.getElementById('feedback-section')?.classList.add('hidden');
    currentScanData = null;
    currentScanId = null;
  };

  const selectFile = () => fileInput.click();
  const showSelected = () => {
    const file = fileInput.files[0];
    if (!file) return;
    if (!accepted.includes(file.type)) return notify('Unsupported media. Use an image, audio file, or MP4/MOV/WebM video.', 'error');
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
      const form = new FormData(); form.append('file', file);
      const scanResponse = await fetch('/api/scans', { method: 'POST', body: form });
      if (!scanResponse.ok) throw new Error((await scanResponse.json()).detail || 'Unable to start scan');
      const scan = await scanResponse.json();
      currentScanId = scan.id;
      fileInfo.classList.add('hidden'); analysisResults.classList.remove('hidden');
      resultsContainer.innerHTML = '<p class="file-info-result">Analysis is running. Video scans can take a little longer.</p>';
      document.getElementById('feedback-section')?.classList.add('hidden');
      await poll(scan.id);
    } catch (error) {
      notify(error.message, 'error');
      fileInfo.classList.remove('hidden');
    } finally { detectButton.disabled = false; detectButton.textContent = 'Detect Now'; }
  });

  async function poll(scanId) {
    for (let attempt = 0; attempt < 120; attempt += 1) {
      await new Promise(resolve => setTimeout(resolve, 2000));
      let response, scan;
      try {
        response = await fetch(`/api/scans/${scanId}`);
        if (!response.ok) throw new Error(`Server error ${response.status}`);
        scan = await response.json();
      } catch (err) {
        // Transient network glitch or server restart — retry silently for up to 3 attempts
        if (attempt > 3) throw new Error('Server unreachable. The model may still be loading — please wait a moment and try again.');
        continue;
      }
      if (scan.status === 'completed') {
        render(scan);
        fetchHistory();
        if (!scan.report?.ai_summary || !scan.report?.heatmap_ready) {
          pollEnrichments(scanId, scan.report?.media_type);
        }
        return;
      }
      if (scan.status === 'failed') throw new Error(scan.error || 'Analysis failed');
    }
    throw new Error('Analysis timed out. Please try a shorter file.');
  }

  async function pollEnrichments(scanId, mediaType) {
    for (let i = 0; i < 6; i++) {
      await new Promise(resolve => setTimeout(resolve, 5000));
      const response = await fetch(`/api/scans/${scanId}`);
      const scan = await response.json();
      const r = scan.report;
      if (!r) return;

      const summaryBox = document.getElementById('ai-summary-content');
      if (summaryBox && r.ai_summary) {
        summaryBox.textContent = r.ai_summary;
        summaryBox.closest('.ai-summary-box')?.classList.remove('ai-summary-loading');
        currentScanData = scan;
      }

      if (mediaType === 'image' && r.heatmap_ready) {
        currentScanData = scan;
      }

      const summaryDone = !!r.ai_summary;
      const heatmapDone = mediaType !== 'image' || r.heatmap_ready;
      if (summaryDone && heatmapDone) return;
    }
  }

  // ── Main On-Screen Render Function ─────────────────────────────────────────
  function render(scan) {
    currentScanData = scan;
    const report = scan.report || {};
    const assessment = report.assessment || {};
    const isHigh = assessment.risk_rating === 'high';
    const isMed = assessment.risk_rating === 'medium';

    const evidenceItems = (report.evidence || [])
      .filter(item => item.source !== 'file-inspection' && item.source !== 'image-metadata')
      .map(item => {
        const d = item.details || {};
        let outcome;
        if (item.source === 'video-frame-model' && d.per_frame_scores) {
          outcome = `${d.frames_analyzed} frame(s) sampled — average risk ${d.average_risk_score}`;
          if (d.peak_frame) outcome += `; highest at ${d.peak_frame.timestamp_seconds}s (${d.peak_frame.risk_score})`;
        } else if (d.prediction) {
          outcome = d.prediction === 'Fake' ? 'Detected as synthetic / manipulated' : 'No synthetic signal detected';
        } else if (d.reason) {
          outcome = d.reason;
        } else {
          outcome = `${Math.round((item.risk || 0) * 100)}% risk signal`;
        }
        return `<li><strong>${escapeHtml(item.label)}:</strong> ${escapeHtml(outcome)}</li>`;
      }).join('');

    const aiSummaryHtml = report.ai_summary
      ? `<div class="ai-summary-box">
           <div class="ai-summary-label">🤖 AI Forensic Summary <span class="ai-summary-note">(summarization of detector signals only — not an authenticity verdict)</span></div>
           <p class="ai-summary-text" id="ai-summary-content">${escapeHtml(report.ai_summary)}</p>
         </div>`
      : `<div class="ai-summary-box ai-summary-loading">
           <div class="ai-summary-label">🤖 AI Forensic Summary</div>
           <p class="ai-summary-text" id="ai-summary-content" style="color:#64748b;font-style:italic;">Generating summary…</p>
         </div>`;

    resultsContainer.innerHTML = `
      <div class="results-box ${isHigh ? 'bg-error-light' : isMed ? 'bg-warn-light' : 'bg-success-light'}">

        <!-- Verdict Header -->
        <div class="result ${isHigh ? 'fake' : isMed ? 'medium' : 'real'}">
          <span class="result-text">${escapeHtml((assessment.risk_rating || 'UNKNOWN').toUpperCase())} authenticity risk</span>
        </div>
        <p class="file-info-result">Automated assessment for <strong>${escapeHtml(scan.filename)}</strong>. This is not proof of manipulation or authenticity.</p>
        <p class="file-info-result"><strong>Recommended action:</strong> ${escapeHtml(assessment.recommended_action || 'Manual review required.')}</p>

        <!-- Score Bar -->
        <div class="confidence-score">
          <div class="score-label">
            <span>Evidence-weighted risk score</span>
            <span class="score-value">${assessment.risk_score || 0}%</span>
          </div>
          <div class="progress-bar">
            <div class="progress ${isHigh ? 'fake' : isMed ? 'medium' : 'real'}" style="width:${assessment.risk_score || 0}%"></div>
          </div>
        </div>

        <!-- Detector Signals -->
        <div class="anomalies">
          <div class="font-medium mb-2">Detector signals</div>
          <ul class="anomalies-list">${evidenceItems}</ul>
        </div>

        <!-- AI Summary (generous spacing) -->
        ${aiSummaryHtml}

        <!-- Action Buttons (no JSON link, rich royal blue PDF button) -->
        <div class="action-buttons">
          <button class="btn-primary pdf-btn" id="download-pdf-btn">📄 Download PDF Report</button>
        </div>

        <!-- Retention Note -->
        <p class="file-info-result retention-note">Retention: uploaded media and artifacts are configured for deletion after ${report.retention_hours || 24} hours.</p>

      </div>`;

    // Wire up PDF button
    const pdfBtn = document.getElementById('download-pdf-btn');
    if (pdfBtn) {
      pdfBtn.addEventListener('click', () => {
        downloadPDFReport(currentScanData);
      });
    }

    // Show feedback section
    const feedbackSection = document.getElementById('feedback-section');
    if (feedbackSection) {
      feedbackSection.classList.remove('hidden');
      document.getElementById('feedback-form').reset();
      document.getElementById('feedback-status').textContent = '';
      const hList = document.getElementById('feedback-history-list');
      const hContainer = document.getElementById('feedback-history-container');
      if (scan.feedback_history && scan.feedback_history.length > 0) {
        hContainer.classList.remove('hidden');
        hList.innerHTML = scan.feedback_history.map(f => `<li><strong>${escapeHtml(f.label.replace('_', ' '))}</strong> (${new Date(f.created_at).toLocaleString()}): ${escapeHtml(f.notes || 'No notes')}</li>`).join('');
      } else {
        hContainer.classList.add('hidden');
      }
    }
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

    // Uniform Arial font styling for table cells to prevent dark/bold letter artifacts
    const evidenceRows = (report.evidence || []).map(item => {
      const d = item.details || {};
      let detail = '';
      if (item.source === 'file-inspection') {
        detail = `${d.mime_type || ''} · ${d.size_bytes ? (d.size_bytes / 1024).toFixed(1) + ' KB' : ''} · SHA-256: ${(d.sha256 || '').slice(0, 18)}…`;
      } else if (item.source === 'image-metadata') {
        detail = `Format: ${d.format || 'N/A'} · ${d.dimensions?.[0] || 0}×${d.dimensions?.[1] || 0}px · EXIF: ${d.exif_present ? 'Present' : 'Absent'}`;
      } else if (item.source === 'video-frame-model') {
        detail = `${d.frames_analyzed || 0} frames sampled · avg ${d.average_risk_score} · peak ${d.peak_frame?.timestamp_seconds}s (${d.peak_frame?.risk_score})`;
      } else if (d.prediction) {
        detail = `${d.prediction}${d.confidence ? ' (' + d.confidence + ')' : ''}`;
      } else if (d.reason) {
        detail = d.reason;
      }
      const riskPct = item.risk !== null && item.risk !== undefined ? `${Math.round(item.risk * 100)}%` : 'N/A';
      return `<tr>
        <td style="padding: 5px 8px; border: 1px solid #cbd5e1; font-weight: 500; color: #1e293b;">${escapeHtml(item.label)}</td>
        <td style="padding: 5px 8px; border: 1px solid #cbd5e1; color: ${ratingColor}; font-weight: 600;">${escapeHtml(riskPct)}</td>
        <td style="padding: 5px 8px; border: 1px solid #cbd5e1; font-weight: 500; color: #475569;">${item.weight || 0}</td>
        <td style="padding: 5px 8px; border: 1px solid #cbd5e1; color: #475569; font-size: 8pt; font-weight: 400;">${escapeHtml(detail)}</td>
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
              <div style="font-size: 7.5pt; font-weight: 600; color: #334155;">${f.timestamp_seconds}s</div>
              <div style="font-size: 7.5pt; color: #64748b;">${f.risk_score}</div>
            </div>
          `).join('')}
        </div>
      </div>` : '';

    const heatmapUrl = report.heatmap_ready ? `/api/scans/${scan.id}/heatmap` : null;
    const heatmapHtml = heatmapUrl ? `
      <div style="margin-top: 10px; page-break-inside: avoid;">
        <div style="font-size: 9.5pt; font-weight: 600; color: #1e293b; margin-bottom: 2px;">Model Attention Heatmap (GradCAM)</div>
        <div style="font-size: 7.5pt; color: #64748b; margin-bottom: 4px;">Red and yellow regions highlight visual areas that most influenced the prediction.</div>
        <img src="${heatmapUrl}" style="max-width: 320px; max-height: 200px; border-radius: 4px; border: 1px solid #cbd5e1; display: block;" />
      </div>` : '';

    const feedbackHtml = (scan.feedback_history && scan.feedback_history.length > 0) ? `
      <div style="margin-top: 10px;">
        <div style="font-size: 9.5pt; font-weight: 600; color: #1e293b; margin-bottom: 4px; border-bottom: 1px solid #e2e8f0; padding-bottom: 2px;">Human Review Audit Trail</div>
        <ul style="margin: 0; padding-left: 16px; font-size: 8pt; color: #334155;">
          ${scan.feedback_history.map(f => `<li><strong>${escapeHtml(f.label.replace('_', ' '))}</strong> (${new Date(f.created_at).toLocaleString()}): ${escapeHtml(f.notes || 'No notes provided')}</li>`).join('')}
        </ul>
      </div>` : '';

    const scanDate = new Date(scan.completed_at || scan.created_at).toLocaleString();

    const reportHtml = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>FusionTrace Forensic Report - ${scan.id}</title>
  <style>
    @page {
      size: A4 portrait;
      margin: 8mm 12mm;
    }
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
    .header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 2px solid #4f46e5;
      padding-bottom: 6px;
      margin-bottom: 10px;
    }
    .title {
      font-size: 16pt;
      font-weight: 700;
      color: #4f46e5;
      letter-spacing: -0.3px;
      margin: 0;
    }
    .subtitle {
      font-size: 8pt;
      color: #64748b;
      margin-top: 1px;
    }
    .meta {
      font-size: 8pt;
      color: #475569;
      text-align: right;
      line-height: 1.3;
    }
    .verdict-card {
      background: ${ratingBg};
      border: 1px solid ${ratingBorder};
      border-radius: 5px;
      padding: 10px 12px;
      margin-bottom: 10px;
    }
    .verdict-badge {
      display: inline-block;
      background: ${ratingColor};
      color: #ffffff;
      font-weight: 700;
      font-size: 9pt;
      text-transform: uppercase;
      padding: 2px 7px;
      border-radius: 3px;
      letter-spacing: 0.3px;
    }
    .verdict-text {
      font-size: 8pt;
      color: #334155;
      margin-top: 4px;
      margin-bottom: 2px;
    }
    .rec-text {
      font-size: 8pt;
      color: #0f172a;
      font-weight: 600;
      margin-bottom: 6px;
    }
    .score-container {
      margin-top: 4px;
    }
    .score-label {
      display: flex;
      justify-content: space-between;
      font-size: 8pt;
      font-weight: 700;
      color: #1e293b;
      margin-bottom: 2px;
    }
    .progress-bar-bg {
      background: #e2e8f0;
      height: 6px;
      border-radius: 3px;
      overflow: hidden;
      width: 100%;
    }
    .progress-bar-fill {
      background: ${ratingColor};
      height: 100%;
      width: ${assessment.risk_score || 0}%;
    }
    .ai-box {
      background: #eff6ff;
      border-left: 3.5px solid #3b82f6;
      border-radius: 4px;
      padding: 8px 10px;
      margin-top: 10px;
      margin-bottom: 10px;
    }
    .ai-title {
      font-weight: 700;
      color: #1e40af;
      font-size: 8.5pt;
      margin-bottom: 3px;
    }
    .ai-body {
      font-size: 8pt;
      color: #1e293b;
      margin: 0;
      line-height: 1.35;
    }
    .table-title {
      font-size: 9.5pt;
      font-weight: 700;
      color: #1e293b;
      margin-top: 10px;
      margin-bottom: 4px;
      border-bottom: 1px solid #e2e8f0;
      padding-bottom: 2px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 2px;
      font-size: 8pt;
    }
    th {
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      padding: 4px 7px;
      text-align: left;
      font-weight: 700;
      color: #334155;
    }
    td {
      font-family: Arial, Helvetica, sans-serif !important;
    }
    .footer {
      margin-top: 14px;
      border-top: 1px solid #e2e8f0;
      padding-top: 6px;
      font-size: 7pt;
      color: #94a3b8;
      text-align: center;
      line-height: 1.3;
    }
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
    <div class="verdict-text">Automated risk assessment for <strong>${escapeHtml(scan.filename)}</strong>.</div>
    <div class="rec-text">Recommended action: ${escapeHtml(assessment.recommended_action || 'Manual review required.')}</div>
    <div class="score-container">
      <div class="score-label">
        <span>Evidence-Weighted Risk Score</span>
        <span>${assessment.risk_score || 0}%</span>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill"></div>
      </div>
    </div>
  </div>

  ${report.ai_summary ? `
  <div class="ai-box">
    <div class="ai-title">🤖 AI Forensic Summary <span style="font-weight:400; font-size:7pt; color:#3b82f6;">(summarization of detector signals only — not an authenticity verdict)</span></div>
    <div class="ai-body">${escapeHtml(report.ai_summary)}</div>
  </div>` : ''}

  <div class="table-title">Detector Signals & Evidence Breakdown</div>
  <table>
    <thead>
      <tr>
        <th style="width: 26%;">Signal Label</th>
        <th style="width: 12%;">Risk Score</th>
        <th style="width: 10%;">Weight</th>
        <th>Technical Detail</th>
      </tr>
    </thead>
    <tbody>
      ${evidenceRows}
    </tbody>
  </table>

  ${frameGridHtml}
  ${heatmapHtml}
  ${feedbackHtml}

  <div class="footer">
    <div>Retention: uploaded media and artifacts are configured for deletion after ${report.retention_hours || 24} hours.</div>
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
      const response = await fetch('/api/scans');
      const scans = await response.json();
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
          <td><button class="btn-outline btn-sm open-scan-btn" data-id="${s.id}">Open</button></td>
        </tr>`;
      }).join('');
      document.querySelectorAll('.open-scan-btn').forEach(btn => {
        btn.addEventListener('click', async (e) => {
          const id = e.target.getAttribute('data-id');
          await openScan(id);
        });
      });
    } catch (e) { console.error('Failed to fetch history', e); }
  }

  async function openScan(id) {
    try {
      const res = await fetch(`/api/scans/${id}`);
      if (!res.ok) throw new Error('Failed to load scan');
      const scan = await res.json();
      reset();
      uploadArea.classList.add('hidden');
      analysisResults.classList.remove('hidden');
      if (scan.status === 'completed') {
        render(scan);
      } else {
        resultsContainer.innerHTML = `<p>Scan status: ${scan.status}</p>`;
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
        const res = await fetch(`/api/scans/${currentScanData.id}/feedback`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ label, notes })
        });
        if (!res.ok) throw new Error((await res.json()).detail || 'Failed to submit feedback');
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
