/* ========== Shared state ========== */
let API_BASE =
  localStorage.getItem("apiBase") ||
  document.getElementById("apiBase").value;
document.getElementById("apiBase").value = API_BASE;

const saveApiBtn = document.getElementById("saveApi");
const modelSelect = document.getElementById("modelSelect");
const autoNote = document.getElementById("autoNote");

let MODELS = [];

saveApiBtn.onclick = () => {
  API_BASE = document.getElementById("apiBase").value;
  localStorage.setItem("apiBase", API_BASE);
  alert("API base saved");
  loadModels();
};

const fileInput = document.getElementById("file");
const runBtn = document.getElementById("runBtn");
const th = document.getElementById("th");
const thv = document.getElementById("thv");
const autoBtn = document.getElementById("autoTh");

fileInput.addEventListener(
  "change",
  () => (runBtn.disabled = !fileInput.files.length)
);
th.addEventListener(
  "input",
  () => (thv.textContent = (+th.value).toFixed(2))
);

let donutChart, rocChart;

/* ========== Load models from backend ========== */
async function loadModels() {
  try {
    modelSelect.innerHTML = '<option value="">Loading…</option>';
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    MODELS = data.models || [];
    modelSelect.innerHTML = "";

    if (MODELS.length === 0) {
      modelSelect.innerHTML = '<option value="">No models</option>';
      return;
    }

    MODELS.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.key;
      opt.textContent = m.name || m.key;
      modelSelect.appendChild(opt);
    });

    const defKey = data.default_model || (MODELS[0] && MODELS[0].key);
    if (defKey) {
      modelSelect.value = defKey;
    }
  } catch (e) {
    console.error("Failed to load models", e);
    if (!modelSelect.options.length) {
      modelSelect.innerHTML = `
        <option value="bot-iot">Bot-IoT (default)</option>
        <option value="iot23">IoT-23</option>
      `;
    }
  }
}
loadModels();

/* ========== Tabs ========== */
const tabButtons = document.querySelectorAll('.nav-item[data-tab]');
tabButtons.forEach((btn) => {
  btn.onclick = () => {
    tabButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    document
      .querySelectorAll("main section")
      .forEach((s) => (s.style.display = "none"));
    document.getElementById("tab-" + btn.dataset.tab).style.display = "block";
    if (btn.dataset.tab === "devices") startAutoRefresh();
  };
});

/* ========== Helpers ========== */
function modelKeyToName(key) {
  if (!key) return "";
  const m = MODELS.find((m) => m.key === key);
  return m ? m.name || m.key : key;
}

function applyAutoInfo(data) {
  const at = data.auto_threshold;
  if (at && typeof at.threshold === "number") {
    autoNote.textContent = `Auto threshold = ${at.threshold.toFixed(
      2
    )} (metric=${(at.metric || "f1").toUpperCase()}, F1=${(
      at.f1 || 0
    ).toFixed(4)}, FPR=${(at.fpr || 0).toFixed(4)})`;
  } else {
    autoNote.textContent = "";
  }
}

/* ========== Dashboard Predict ========== */
runBtn.onclick = async () => {
  try {
    if (!fileInput.files.length) return alert("Choose a CSV first.");
    if (!modelSelect.value) return alert("Select a model first.");

    const fd = new FormData();
    fd.append("file", fileInput.files[0]);
    fd.append("threshold", th.value);
    fd.append("model", modelSelect.value);

    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: fd,
    });
    const text = await res.text();
    if (!res.ok) throw new Error(text);
    const data = JSON.parse(text);
    renderAll(data);
    applyAutoInfo(data);
  } catch (e) {
    console.error(e);
    alert("Predict failed: " + e.message);
  }
};

autoBtn.onclick = async () => {
  try {
    if (!fileInput.files.length) return alert("Choose a CSV first.");
    if (!modelSelect.value) return alert("Select a model first.");

    const fd = new FormData();
    fd.append("file", fileInput.files[0]);
    fd.append("model", modelSelect.value);
    fd.append("auto_threshold", "1"); // ask backend to auto-tune

    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: fd,
    });
    const text = await res.text();
    if (!res.ok) throw new Error(text);
    const data = JSON.parse(text);
    renderAll(data);

    const s = data.summary || {};
    if (typeof s.threshold === "number") {
      th.value = s.threshold;
      thv.textContent = (+s.threshold).toFixed(2);
    }
    applyAutoInfo(data);
  } catch (e) {
    console.error(e);
    alert("Auto-threshold failed: " + e.message);
  }
};

function renderAll(data) {
  const s = data.summary || {};
  const modelName = data.model ? modelKeyToName(data.model) : "";

  document.getElementById("summary").textContent =
    `Total: ${fmtInt(s.total_records)} | Anomalies: ${fmtInt(
      s.anomalies_detected
    )} | Normal: ${fmtInt(s.normal_records)}` +
    (modelName ? ` | Model: ${modelName}` : "");

  const m = data.metrics || {};
  const kpis = document.getElementById("kpis");
  kpis.innerHTML = "";
  const labels = {
    accuracy: "Accuracy",
    precision: "Precision",
    recall_tpr: "Recall (TPR)",
    f1: "F1",
    fpr: "FPR",
    auc: "AUC",
    mse: "MSE",
  };
  Object.entries(labels).forEach(([k, label]) => {
    const v = m[k];
    const val = v == null ? "—" : typeof v === "number" ? v.toFixed(4) : v;
    const div = document.createElement("div");
    div.className = "kpi card";
    div.innerHTML = `<div class="label">${label}</div><div class="value">${val}</div>`;
    kpis.appendChild(div);
  });

  const dist = [s.normal_records || 0, s.anomalies_detected || 0];
  donutChart?.destroy?.();
  donutChart = new Chart(
    document.getElementById("donut").getContext("2d"),
    {
      type: "doughnut",
      data: {
        labels: ["Normal (0)", "Attack (1)"],
        datasets: [{ data: dist }],
      },
      options: {
        plugins: {
          legend: {
            labels: { color: "#cfd8e3" },
          },
        },
      },
    }
  );

  const roc = data.roc;
  const aucBadge = document.getElementById("aucBadge");
  aucBadge.textContent = roc ? `AUC: ${(+roc.auc).toFixed(4)}` : "";

  rocChart?.destroy?.();
  if (
    roc &&
    Array.isArray(roc.fpr) &&
    Array.isArray(roc.tpr) &&
    roc.fpr.length === roc.tpr.length &&
    roc.fpr.length > 1
  ) {
    let pts = roc.tpr
      .map((y, i) => ({ x: Number(roc.fpr[i]), y: Number(y) }))
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    pts = pts.filter(
      (p) => p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1
    );
    const MAX = 500;
    if (pts.length > MAX) {
      const step = Math.ceil(pts.length / MAX);
      pts = pts.filter((_, i) => i % step === 0);
    }

    rocChart = new Chart(
      document.getElementById("roc").getContext("2d"),
      {
        type: "line",
        data: { datasets: [{ label: "ROC", data: pts, parsing: false }] },
        options: {
          animation: false,
          plugins: { legend: { display: true } },
          scales: {
            x: {
              type: "linear",
              min: 0,
              max: 1,
              title: { display: true, text: "FPR" },
            },
            y: {
              min: 0,
              max: 1,
              title: { display: true, text: "TPR" },
            },
          },
          elements: { line: { tension: 0 } },
        },
      }
    );
  }

  const cm = data.confusion;
  drawHeatmap2x2(
    document.getElementById("cm"),
    cm?.matrix || [
      [0, 0],
      [0, 0],
    ],
    cm?.matrix_pct || [
      [0, 0],
      [0, 0],
    ]
  );
  // NEW: text summary under the matrix
  const cmNoteEl = document.getElementById("cmNote");
  if (cm && cm.matrix && cmNoteEl) {
    const mat = cm.matrix;
    const tn = mat?.[0]?.[0] ?? 0;
    const fp = mat?.[0]?.[1] ?? 0;
    const fn = mat?.[1]?.[0] ?? 0;
    const tp = mat?.[1]?.[1] ?? 0;

    const tpr = tp + fn > 0 ? tp / (tp + fn) : 0;    // recall
    const fprVal = fp + tn > 0 ? fp / (fp + tn) : 0; // false positive rate

    cmNoteEl.textContent =
      `TP (detected attacks): ${fmtInt(tp)} · ` +
      `TN (correct normals): ${fmtInt(tn)} · ` +
      `FP (false alarms): ${fmtInt(fp)} · ` +
      `FN (missed attacks): ${fmtInt(fn)} · ` +
      `TPR: ${tpr.toFixed(4)} · FPR: ${fprVal.toFixed(4)}`;
  } else if (cmNoteEl) {
    cmNoteEl.textContent = "Confusion matrix not available for this run.";
  }
}

function drawHeatmap2x2(canvas, m, p) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // matrix layout we expect:
  // m = [[TN, FP],
  //      [FN, TP]]
  const vals = [...(m[0] || [0, 0]), ...(m[1] || [0, 0])];
  const maxVal = Math.max(...vals, 1);

  // layout
  const marginTop = 60;
  const marginLeft = 80;
  const marginRight = 20;
  const marginBottom = 45;

  const cellSize = Math.min(
    (W - marginLeft - marginRight) / 2,
    (H - marginTop - marginBottom) / 2
  );

  const x0 = marginLeft;
  const y0 = marginTop;

  // cell metadata: label + description + color
  const cells = [
    [
      { short: "TN", desc: "True Normal", color: "#2563eb" },   // top-left
      { short: "FP", desc: "False Alarm", color: "#f59e0b" },   // top-right
    ],
    [
      { short: "FN", desc: "Missed Attack", color: "#ef4444" }, // bottom-left
      { short: "TP", desc: "Detected Attack", color: "#22c55e" } // bottom-right
    ],
  ];

  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  // draw 4 cells
  for (let r = 0; r < 2; r++) {
    for (let c = 0; c < 2; c++) {
      const meta = cells[r][c];
      const v = m?.[r]?.[c] ?? 0;
      const pct = (p?.[r]?.[c] ?? 0) * 100;

      const x = x0 + c * cellSize;
      const y = y0 + r * cellSize;

      // background intensity based on value
      const alpha = 0.25 + 0.6 * (v / maxVal);
      const base = meta.color;
      // simple way: use the same color but with alpha via rgba overlay
      ctx.fillStyle = base;
      ctx.globalAlpha = alpha;
      ctx.fillRect(x, y, cellSize, cellSize);
      ctx.globalAlpha = 1;

      ctx.strokeStyle = "#020617";
      ctx.lineWidth = 1.2;
      ctx.strokeRect(x, y, cellSize, cellSize);

      // texts
      ctx.fillStyle = "#e5e7eb";

      // short label (TN/FP/FN/TP) at top of cell
      ctx.font = "11px system-ui";
      ctx.fillText(meta.short, x + cellSize / 2, y + 12);

      // main count
      ctx.font = "18px system-ui";
      ctx.fillText(String(v), x + cellSize / 2, y + cellSize / 2 - 4);

      // percentage under count
      ctx.font = "11px system-ui";
      ctx.fillText(
        pct.toFixed(1) + "%",
        x + cellSize / 2,
        y + cellSize / 2 + 18
      );

      // description at bottom of cell
      ctx.font = "10px system-ui";
      ctx.fillText(
        meta.desc,
        x + cellSize / 2,
        y + cellSize - 10
      );
    }
  }

  // axis labels
  ctx.fillStyle = "#e5e7eb";
  ctx.font = "12px system-ui";
  ctx.textAlign = "center";

  // X-axis main label
  ctx.fillText(
    "Predicted Class",
    x0 + cellSize,
    H - 10
  );

  // X-axis tick labels (top)
  ctx.font = "11px system-ui";
  ctx.fillText("Normal (0)", x0 + cellSize / 2, marginTop - 18);
  ctx.fillText("Attack (1)", x0 + cellSize + cellSize / 2, marginTop - 18);

  // Y-axis main label (rotated)
  ctx.save();
  ctx.translate(20, y0 + cellSize);
  ctx.rotate(-Math.PI / 2);
  ctx.font = "12px system-ui";
  ctx.textAlign = "center";
  ctx.fillText("Actual Class", 0, 0);
  ctx.restore();

  // Y-axis tick labels
  ctx.font = "11px system-ui";
  ctx.textAlign = "right";
  ctx.fillText("Normal (0)", marginLeft - 10, y0 + cellSize / 2);
  ctx.fillText("Attack (1)", marginLeft - 10, y0 + cellSize + cellSize / 2);
}



/* ========== Devices Tab ========== */
const deviceSearch = document.getElementById("deviceSearch");
const deviceStatus = document.getElementById("deviceStatus");
const deviceSort = document.getElementById("deviceSort");
const pageSizeSel = document.getElementById("pageSize");
const autoRefreshCb = document.getElementById("autoRefresh");
const refreshSecInp = document.getElementById("refreshSec");
const refreshBtn = document.getElementById("refreshDevices");
const deviceTableBody = document.querySelector("#deviceTable tbody");
const devicesNote = document.getElementById("devicesNote");
const prevPageBtn = document.getElementById("prevPage");
const nextPageBtn = document.getElementById("nextPage");
const pageInfo = document.getElementById("pageInfo");

let devicesData = null;
let filtered = [];
let currentPage = 1;

refreshBtn.onclick = fetchDevices;
deviceSearch.addEventListener("input", () => {
  currentPage = 1;
  renderDevices();
});
deviceStatus.addEventListener("change", () => {
  currentPage = 1;
  renderDevices();
});
deviceSort.addEventListener("change", () => {
  currentPage = 1;
  renderDevices();
});
pageSizeSel.addEventListener("change", () => {
  currentPage = 1;
  renderDevices();
});
prevPageBtn.addEventListener("click", () => {
  if (currentPage > 1) {
    currentPage--;
    renderDevices();
  }
});
nextPageBtn.addEventListener("click", () => {
  const max = Math.max(1, Math.ceil(filtered.length / pageSize()));
  if (currentPage < max) {
    currentPage++;
    renderDevices();
  }
});

let refreshTimer = null;
function startAutoRefresh() {
  stopAutoRefresh();
  if (!autoRefreshCb.checked) return;
  const sec = Math.max(5, parseInt(refreshSecInp.value || "30", 10));
  refreshTimer = setInterval(() => {
    const devicesTabVisible =
      document.getElementById("tab-devices").style.display !== "none";
    if (devicesTabVisible && fileInput.files.length) fetchDevices(true);
  }, sec * 1000);
}
function stopAutoRefresh() {
  if (refreshTimer) {
    clearInterval(refreshTimer);
    refreshTimer = null;
  }
}
autoRefreshCb.addEventListener("change", startAutoRefresh);
refreshSecInp.addEventListener("change", startAutoRefresh);

async function fetchDevices(silent = false) {
  try {
    if (!fileInput.files.length) return !silent && alert("Choose a CSV first.");
    if (!modelSelect.value) return !silent && alert("Select a model first.");

    const fd = new FormData();
    fd.append("file", fileInput.files[0]);
    fd.append("threshold", th.value);
    fd.append("model", modelSelect.value);

    const res = await fetch(`${API_BASE}/devices?group_by=both`, {
      method: "POST",
      body: fd,
    });
    const text = await res.text();
    if (!res.ok) throw new Error(text);
    const data = JSON.parse(text);

    devicesData = data;
    currentPage = 1;
    renderDevices();
    const modelName = data.model ? modelKeyToName(data.model) : "";
    devicesNote.textContent =
      `Group by: ${data.dev_col} | threshold=${data.threshold}` +
      (modelName ? ` | model=${modelName}` : "") +
      `. Showing ${fmtInt(data.devices.length)} devices.`;
  } catch (e) {
    console.error(e);
    !silent && alert("Devices fetch failed: " + e.message);
  }
}

function renderDevices() {
  if (!devicesData || !Array.isArray(devicesData.devices)) return;

  const q = (deviceSearch.value || "").toLowerCase().trim();
  const status = deviceStatus.value;
  filtered = devicesData.devices.filter((d) => {
    if (status && d.status !== status) return false;
    const attackText = (d.top_attack || "—").toString().toLowerCase();
    const nameText = (d.name || "").toLowerCase();
    return (
      !q ||
      String(d.device).toLowerCase().includes(q) ||
      attackText.includes(q) ||
      nameText.includes(q)
    );
  });

  const sortKey = deviceSort.value;
  filtered.sort((a, b) => {
    const by = (k) =>
      Number((a.totals || {})[k] || 0) -
      Number((b.totals || {})[k] || 0);
    switch (sortKey) {
      case "anomalies_desc":
        return (b.anomalies || 0) - (a.anomalies || 0);
      case "anomaly_rate_desc":
        return (b.anomaly_rate || 0) - (a.anomaly_rate || 0);
      case "bytes_desc":
        return by("bytes") > 0 ? -1 : by("bytes") < 0 ? 1 : 0;
      case "pkts_desc":
        return by("pkts") > 0 ? -1 : by("pkts") < 0 ? 1 : 0;
      case "device_asc":
      default:
        return String(a.device).localeCompare(String(b.device));
    }
  });

  const total = filtered.length;
  const psize = pageSize();
  const maxPage = Math.max(1, Math.ceil(total / psize));
  if (currentPage > maxPage) currentPage = maxPage;
  const start = (currentPage - 1) * psize;
  const pageItems = filtered.slice(start, start + psize);

  const rows = pageItems.map((d, idx) => {
    const badge =
      d.status === "RED"
        ? '<span class="status red">RED</span>'
        : '<span class="status green">GREEN</span>';
    const bytes = prettyBytes(d?.totals?.bytes);
    const pkts = fmtInt(d?.totals?.pkts);
    const mac = d.smac || d.dmac || "—";
    const name = d.name || "—";
    const dtype = d.dtype || "—";
    const vendor = d.vendor || "—";
    const attack = d.top_attack
      ? `<span class="pill">${esc(d.top_attack)}</span>`
      : "—";
    return `
      <tr>
        <td>${fmtInt(start + idx + 1)}</td>
        <td>${esc(d.device)}</td>
        <td>${esc(name)}</td>
        <td>${esc(dtype)}</td>
        <td>${esc(vendor)}</td>
        <td>${esc(mac)}</td>
        <td>${esc(pkts)}</td>
        <td>${esc(bytes)}</td>
        <td>${badge}</td>
        <td>${attack}</td>
        <td>${((d.anomaly_rate || 0) * 100).toFixed(2)}%</td>
        <td>${d.last_seen ? esc(d.last_seen) : "—"}</td>
        <td>${fmtInt(d.total)}</td>
        <td>${fmtInt(d.anomalies)}</td>
      </tr>
    `;
  });
  deviceTableBody.innerHTML =
    rows.join("") ||
    `<tr><td colspan="14" class="muted">No devices match.</td></tr>`;
  pageInfo.textContent = `Page ${currentPage}/${maxPage}`;
  prevPageBtn.disabled = currentPage <= 1;
  nextPageBtn.disabled = currentPage >= maxPage;
}

function pageSize() {
  return Math.max(1, parseInt(pageSizeSel.value || "25", 10));
}

function fmtInt(x) {
  if (x == null || !isFinite(x)) return "0";
  try {
    return Intl.NumberFormat().format(Math.round(x));
  } catch {
    return String(x);
  }
}
function prettyBytes(b) {
  const n = Number(b);
  if (!isFinite(n) || n <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = Math.floor(Math.log(n) / Math.log(1024));
  return (n / Math.pow(1024, i)).toFixed(1) + " " + units[i];
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[m]));
}
