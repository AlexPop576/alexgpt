// Live training page: reads runs/supernova/log.txt through the server every 15 seconds.
//   loss curve (train + validation EN / RO), progress, how the sample texts change, and
//   "What it reads": a random document from the training data.

const TRAIN = { data: null, timer: 0, prompt: 0, source: "", hover: null };
const SOURCE_NAMES = {
  fineweb_edu: "FineWeb-Edu · educational web pages",
  cosmopedia: "Cosmopedia · textbooks written by a bigger AI",
  wiki_en: "Wikipedia (English)",
  soda: "SODA · invented everyday conversations",
  wiki_ro: "Wikipedia (română)",
  fineweb2_ro: "FineWeb-2 · Romanian web pages",
};

function trainingOpen() { return !$("trainingPage").hidden; }

function openTraining(on) {
  $("trainingPage").hidden = !on;
  $("trainingToggle").classList.toggle("active", on);
  clearInterval(TRAIN.timer);
  if (on) {
    loadTraining();
    TRAIN.timer = setInterval(loadTraining, 15000);
    if (!$("peekText").dataset.loaded) loadPeek();
  }
}

async function loadTraining() {
  try {
    const response = await fetch("/api/training");
    TRAIN.data = await response.json();
  } catch (e) {
    $("trStatus").textContent = "Can't reach the server.";
    return;
  }
  drawTraining();
}

function formatTokens(count) {
  if (count >= 1e9) return (count / 1e9).toFixed(2) + "B";
  return Math.round(count / 1e6) + "M";
}

function drawTraining() {
  const data = TRAIN.data;
  if (!data || !data.found) {
    $("trStatus").textContent = "No training log yet (runs/supernova/log.txt).";
    return;
  }
  const ago = data.seconds_since_update;
  const agoText = ago < 90 ? ago + " s ago" : ago < 5400 ? Math.round(ago / 60) + " min ago" : Math.round(ago / 3600) + " h ago";
  $("trStatus").innerHTML = (data.running ? '<span class="tr-dot on"></span>Training now' : data.step >= (data.total_steps || 1) * 0.995 ? '<span class="tr-dot done"></span>Finished' : '<span class="tr-dot"></span>Not running') +
    " · last line " + agoText;

  const tokensPerStep = 131072;
  const total = data.total_steps || 1;
  const done = data.step / total;
  const decayStart = Math.floor(total * 0.85);
  const lastVal = data.val[data.val.length - 1];
  const finish = data.hours_left && data.running ? new Date(Date.now() + data.hours_left * 3600e3) : null;
  const stat = (value, label) => '<div class="tr-stat"><div class="tr-value">' + value + '</div><div class="tr-label">' + label + "</div></div>";
  $("trStats").innerHTML =
    stat(data.step.toLocaleString() + '<span class="tr-of"> / ' + total.toLocaleString() + "</span>", "steps") +
    stat(formatTokens(data.step * tokensPerStep) + '<span class="tr-of"> / ' + formatTokens(total * tokensPerStep) + "</span>", "tokens read") +
    stat(lastVal ? lastVal[1].toFixed(2) + '<span class="tr-of"> · ' + (lastVal[2] ? lastVal[2].toFixed(2) : "–") + "</span>" : "–", "val loss EN · RO") +
    stat(data.speed ? data.speed.toFixed(1) + "K" : "–", "tokens / second") +
    stat(data.hours_left ? "~" + data.hours_left.toFixed(1) + " h" : "–",
      finish ? "left · done " + finish.toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit" }) : "left (while it runs)");

  $("trProgress").innerHTML =
    '<div class="tr-bar"><i style="width:' + (100 * done).toFixed(2) + '%"></i>' +
    '<span class="tr-mark" style="left:' + (100 * 300 / total) + '%" title="warm-up ends"></span>' +
    '<span class="tr-mark decay" style="left:85%"></span></div>' +
    '<div class="tr-bar-labels"><span>' + (100 * done).toFixed(1) + "% done</span>" +
    "<span>step " + decayStart.toLocaleString() + " · stable.pt is saved here: Singularity starts from it</span></div>";

  drawLossChart();
  drawSamples();
}

// ---------------------------------------------------------------- loss chart (one y axis, log scale)
const CHART = { W: 760, H: 280, left: 44, right: 70, top: 14, bottom: 30, low: 2, high: 10 };

function chartX(step) {
  const total = TRAIN.data.total_steps || 1;
  return CHART.left + (CHART.W - CHART.left - CHART.right) * step / total;
}
function chartY(loss) {
  const value = Math.min(CHART.high, Math.max(CHART.low, loss));
  const t = (Math.log(value) - Math.log(CHART.low)) / (Math.log(CHART.high) - Math.log(CHART.low));
  return CHART.H - CHART.bottom - t * (CHART.H - CHART.top - CHART.bottom);
}

function drawLossChart() {
  const data = TRAIN.data;
  const total = data.total_steps;
  let svg = "";
  for (const loss of [2, 3, 4, 6, 8, 10]) {
    const y = chartY(loss);
    svg += '<line x1="' + CHART.left + '" x2="' + (CHART.W - CHART.right) + '" y1="' + y + '" y2="' + y + '" class="grid"/>' +
      '<text x="' + (CHART.left - 8) + '" y="' + (y + 4) + '" class="axis" text-anchor="end">' + loss + "</text>";
  }
  for (let i = 0; i <= 5; i++) {
    const step = Math.round(total * i / 5);
    svg += '<text x="' + chartX(step) + '" y="' + (CHART.H - 8) + '" class="axis" text-anchor="middle">' +
      (step >= 1000 ? (step / 1000).toFixed(step % 1000 ? 1 : 0) + "k" : step) + "</text>";
  }
  const decay = Math.floor(total * 0.85);
  svg += '<line x1="' + chartX(decay) + '" x2="' + chartX(decay) + '" y1="' + CHART.top + '" y2="' + (CHART.H - CHART.bottom) + '" class="decay"/>' +
    '<text x="' + (chartX(decay) - 6) + '" y="' + (CHART.top + 10) + '" class="axis" text-anchor="end">learning rate goes down →</text>';

  const path = (points) => points.map((p, i) => (i ? "L" : "M") + chartX(p[0]).toFixed(1) + " " + chartY(p[1]).toFixed(1)).join(" ");
  // train loss: every 10 steps it is noisy, so a light raw line + a smoothed one on top
  const smooth = [];
  let average = null;
  for (const [step, loss] of data.steps) {
    average = average === null ? loss : 0.85 * average + 0.15 * loss;
    smooth.push([step, average]);
  }
  svg += '<path d="' + path(data.steps.map((s) => [s[0], s[1]])) + '" class="train-raw"/>';
  svg += '<path d="' + path(smooth) + '" class="train"/>';
  const en = data.val.filter((v) => v[1] !== null).map((v) => [v[0], v[1]]);
  const ro = data.val.filter((v) => v[2] !== null).map((v) => [v[0], v[2]]);
  if (en.length) svg += '<path d="' + path(en) + '" class="val en"/>';
  if (ro.length) svg += '<path d="' + path(ro) + '" class="val ro"/>';
  for (const p of en) svg += '<circle cx="' + chartX(p[0]) + '" cy="' + chartY(p[1]) + '" r="3.5" class="dot en"/>';
  for (const p of ro) svg += '<circle cx="' + chartX(p[0]) + '" cy="' + chartY(p[1]) + '" r="3.5" class="dot ro"/>';
  // direct labels at the newest point
  if (en.length) { const p = en[en.length - 1]; svg += '<text x="' + (chartX(p[0]) + 8) + '" y="' + (chartY(p[1]) + 4) + '" class="label">EN ' + p[1].toFixed(2) + "</text>"; }
  if (ro.length) { const p = ro[ro.length - 1]; svg += '<text x="' + (chartX(p[0]) + 8) + '" y="' + (chartY(p[1]) - 6) + '" class="label">RO ' + p[1].toFixed(2) + "</text>"; }
  if (TRAIN.hover) {
    const h = TRAIN.hover;
    svg += '<line x1="' + chartX(h[0]) + '" x2="' + chartX(h[0]) + '" y1="' + CHART.top + '" y2="' + (CHART.H - CHART.bottom) + '" class="cross"/>' +
      '<circle cx="' + chartX(h[0]) + '" cy="' + chartY(h[1]) + '" r="4" class="dot train"/>';
  }
  $("trChart").setAttribute("viewBox", "0 0 " + CHART.W + " " + CHART.H);
  $("trChart").innerHTML = svg;
}

function chartHover(event) {
  const data = TRAIN.data;
  if (!data || !data.steps.length) return;
  const rect = $("trChart").getBoundingClientRect();
  const x = (event.clientX - rect.left) * CHART.W / rect.width;
  const step = (x - CHART.left) / (CHART.W - CHART.left - CHART.right) * data.total_steps;
  let best = data.steps[0];
  for (const s of data.steps) if (Math.abs(s[0] - step) < Math.abs(best[0] - step)) best = s;
  if (step > data.step + data.total_steps * 0.03) { chartLeave(); return; }
  TRAIN.hover = best;
  let val = null;
  for (const v of data.val) if (v[0] <= best[0]) val = v;
  drawLossChart();
  flowTip("<b>step " + best[0].toLocaleString() + "</b><br>train loss " + best[1].toFixed(3) +
    "<br>learning rate " + best[2].toExponential(1) +
    (val ? "<br>val (step " + val[0] + ") EN " + val[1].toFixed(2) + (val[2] ? " · RO " + val[2].toFixed(2) : "") : ""), event);
}
function chartLeave() { TRAIN.hover = null; flowTip(null); if (TRAIN.data) drawLossChart(); }

// ---------------------------------------------------------------- samples
function drawSamples() {
  const data = TRAIN.data;
  const tabs = $("trPrompts");
  tabs.innerHTML = data.prompts.map((p, i) =>
    '<button class="tr-chip' + (i === TRAIN.prompt ? " on" : "") + '" data-prompt="' + i + '">' + escapeHtml(p) + "…</button>").join("");
  const prompt = data.prompts[TRAIN.prompt];
  let html = "";
  for (let i = data.samples.length - 1; i >= 0; i--) {
    const sample = data.samples[i];
    const text = sample.texts[TRAIN.prompt];
    if (!text) continue;
    const continuation = text.startsWith(prompt) ? text.slice(prompt.length) : text;
    const val = data.val[i];
    html += '<div class="tr-sample"><div class="tr-sample-step">step ' + sample.step.toLocaleString() +
      (val ? '<span>loss ' + val[1].toFixed(2) + "</span>" : "") + "</div>" +
      '<div class="tr-sample-text"><b>' + escapeHtml(prompt) + "</b>" + escapeHtml(continuation) + "</div></div>";
  }
  $("trSamples").innerHTML = html || '<div class="tr-muted">The first samples come at step 250.</div>';
}

// ---------------------------------------------------------------- what it reads
async function loadPeek() {
  $("peekText").textContent = "Reading…";
  try {
    const response = await fetch("/api/training/peek" + (TRAIN.source ? "?source=" + TRAIN.source : ""));
    const data = await response.json();
    if (data.error) { $("peekText").textContent = data.error; return; }
    $("peekSource").innerHTML = "<b>" + escapeHtml(SOURCE_NAMES[data.source] || data.source) + "</b> · " +
      Math.round(data.share * 100) + "% of what it reads · " + data.tokens + " tokens" + (data.complete ? "" : " (first 700)");
    $("peekText").textContent = data.text;
    $("peekText").dataset.loaded = "1";
  } catch (e) {
    $("peekText").textContent = "Can't reach the server.";
  }
}

function setupTraining() {
  $("trainingToggle").addEventListener("click", () => openTraining(!trainingOpen()));
  $("trClose").addEventListener("click", () => openTraining(false));
  $("trChart").addEventListener("mousemove", chartHover);
  $("trChart").addEventListener("mouseleave", chartLeave);
  $("trPrompts").addEventListener("click", (event) => {
    const button = event.target.closest("[data-prompt]");
    if (!button) return;
    TRAIN.prompt = parseInt(button.dataset.prompt);
    drawSamples();
  });
  const sources = $("peekSources");
  sources.innerHTML = '<button class="tr-chip on" data-source="">any (like training)</button>' +
    Object.keys(SOURCE_NAMES).map((s) => '<button class="tr-chip" data-source="' + s + '">' + s + "</button>").join("");
  sources.addEventListener("click", (event) => {
    const button = event.target.closest("[data-source]");
    if (!button) return;
    TRAIN.source = button.dataset.source;
    for (const b of sources.children) b.classList.toggle("on", b === button);
    loadPeek();
  });
  $("peekAgain").addEventListener("click", loadPeek);
  document.addEventListener("keydown", (event) => { if (event.key === "Escape" && trainingOpen()) openTraining(false); });
}
