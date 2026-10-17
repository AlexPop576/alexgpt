// AlexGPT interface: talks to server.py and draws what the model does.

// Small line icons, one per model (18x18, drawn with strokes).
const ICONS = {
  Pebble:   '<path d="M3.5 11.2c0-3.6 2.9-6.2 6.3-6.2 3 0 5 1.9 5 4.6 0 3.1-2.6 5.4-6.1 5.4-3 0-5.2-1.4-5.2-3.8Z"/><path d="M7.2 8.3c.7-.6 1.6-.9 2.5-.9"/>',
  Slate:    '<path d="M2.5 7.5L9 4.5l6.5 3L9 10.5Z"/><path d="M2.5 10.5L9 13.5l6.5-3"/>',
  Quartz:   '<path d="M9 1.8l3 3.2v8.2l-3 3-3-3V5Z"/><path d="M6 5l3 1.6L12 5"/><path d="M9 6.6v9.6"/>',
  Prism:    '<path d="M8.5 3.5l5.5 10H3Z"/><path d="M1 10.2l4.6-1.3"/><path d="M11.4 8.6l5-1.6M11.9 9.8l4.8.2M12.4 11l4.3 1.6"/>',
  Sapphire: '<ellipse cx="9" cy="9" rx="5.5" ry="7"/><path d="M9 2L6.2 9 9 16l2.8-7Z"/><path d="M3.5 9h11"/>',
  Diamond:  '<path d="M5 3.5h8l3 4L9 15.5 2 7.5Z"/><path d="M2 7.5h14"/><path d="M5 3.5l2 4 2 8 2-8 2-4"/>',
};

// The full family. Models the server has loaded become selectable; the rest show as "soon".
const FAMILY = [
  { name: "Pebble",   dark: "#b5b3aa", light: "#7a776d", note: "Bigram · sees 1 character" },
  { name: "Slate",    dark: "#9db6d6", light: "#4f6788", note: "MLP · sees 8 characters" },
  { name: "Quartz",   dark: "#e2c4ef", light: "#9a5fb5", note: "Attention" },
  { name: "Prism",    dark: "#c9a7ff", light: "#7b55d6", note: "Transformer" },
  { name: "Sapphire", dark: "#6f9bf0", light: "#2f5fc4", note: "Trained on the GPU" },
  { name: "Diamond",  dark: "#bfe6ff", light: "#3b8fc2", note: "Flagship" },
];

const state = {
  available: {},      // name -> info from the server
  current: null,      // selected model name
  generating: false,
  paused: false,
  controller: null,   // used to stop a generation
  lastStep: null,     // last character step, kept so the network can be redrawn (theme, toggle)
};

const $ = (id) => document.getElementById(id);
const END_MARKER = "<|endoftext|>";

// ---------------------------------------------------------------- helpers

function theme() {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

function familyEntry(name) {
  for (const entry of FAMILY) if (entry.name === name) return entry;
  return { name, dark: "#9db6d6", light: "#4f6788", note: "" };
}

function modelColor(name) {
  return familyEntry(name)[theme()];
}

// returns a <span class="model-icon"> with the model's icon, in its color
function makeIcon(name, size) {
  const span = document.createElement("span");
  span.className = "model-icon";
  span.dataset.model = name;
  const paths = ICONS[name] || ICONS.Diamond;
  span.innerHTML =
    '<svg width="' + size + '" height="' + size + '" viewBox="0 0 18 18" fill="none" stroke="currentColor" ' +
    'stroke-width="1.35" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + paths + "</svg>";
  span.style.setProperty("--icon-color", modelColor(name));
  return span;
}

function recolorIcons() {
  for (const icon of document.querySelectorAll(".model-icon")) {
    if (icon.dataset.model) icon.style.setProperty("--icon-color", modelColor(icon.dataset.model));
  }
}

function visibleChar(char) {
  if (char === " ") return "␣";
  if (char === "\n") return "↵";
  return char;
}

function formatParams(count) {
  if (count >= 1e6) return (count / 1e6).toFixed(1) + "M params";
  if (count >= 1e3) return (count / 1e3).toFixed(1) + "K params";
  return count + " params";
}

// probability -> one of 5 color buckets (p0 = very unsure ... p4 = very sure)
function confidenceClass(prob) {
  if (prob < 0.10) return "p0";
  if (prob < 0.30) return "p1";
  if (prob < 0.60) return "p2";
  if (prob < 0.90) return "p3";
  return "p4";
}

function savePreference(key, value) {
  try { localStorage.setItem(key, value); } catch (e) { /* storage blocked: ignore */ }
}
function loadPreference(key) {
  try { return localStorage.getItem(key); } catch (e) { return null; }
}

function panelOpen() {
  return $("insidePanel").classList.contains("open");
}

// ---------------------------------------------------------------- models

async function loadModels() {
  let list = [];
  try {
    const response = await fetch("/api/models");
    list = await response.json();
  } catch (e) {
    $("modelName").textContent = "Offline";
    $("emptySubtitle").textContent = "Can't reach the server. Start it with: python server.py";
    return;
  }

  for (const info of list) state.available[info.name] = info;

  if (list.length === 0) {
    $("modelName").textContent = "No models";
    $("emptySubtitle").textContent = "No trained models found in checkpoints/.";
    return;
  }

  for (const info of list) {
    let known = false;
    for (const entry of FAMILY) if (entry.name === info.name) known = true;
    if (!known) FAMILY.push({ name: info.name, dark: "#9db6d6", light: "#4f6788", note: info.description });
  }

  const saved = loadPreference("alexgpt-model");
  if (saved && state.available[saved]) {
    selectModel(saved);
  } else {
    selectModel(list[list.length - 1].name);   // newest model by default
  }
}

function buildMenu() {
  const menu = $("modelMenu");
  menu.innerHTML = "";

  const label = document.createElement("div");
  label.className = "menu-label";
  label.textContent = "Models";
  menu.append(label);

  for (const entry of FAMILY) {
    const info = state.available[entry.name];
    const option = document.createElement("button");
    option.type = "button";
    option.className = "model-option";
    option.disabled = !info;

    const text = document.createElement("div");
    text.className = "option-text";
    const name = document.createElement("div");
    name.className = "option-name";
    name.textContent = entry.name;
    const desc = document.createElement("div");
    desc.className = "option-desc";
    desc.textContent = info ? info.description + " · " + formatParams(info.params) : entry.note;
    text.append(name, desc);

    option.append(makeIcon(entry.name, 18), text);

    if (!info) {
      const tag = document.createElement("span");
      tag.className = "option-tag";
      tag.textContent = "Soon";
      option.append(tag);
    } else if (entry.name === state.current) {
      const check = document.createElement("span");
      check.className = "option-check";
      check.innerHTML = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M3 7.2l2.6 2.6L11 4.4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
      option.append(check);
    }

    option.addEventListener("click", () => {
      selectModel(entry.name);
      $("modelPicker").classList.remove("open");
    });
    menu.append(option);
  }
}

function selectModel(name) {
  state.current = name;
  savePreference("alexgpt-model", name);
  const info = state.available[name];

  $("modelName").textContent = name;
  const pickerIcon = makeIcon(name, 16);
  pickerIcon.id = "modelIcon";
  $("modelIcon").replaceWith(pickerIcon);

  const emptyIcon = $("emptyIcon");
  if (emptyIcon) {
    emptyIcon.innerHTML = "";
    emptyIcon.append(makeIcon(name, 46));
  }

  document.documentElement.style.setProperty("--accent", modelColor(name));

  $("contextLabel").textContent = info.context_size + (info.context_size === 1 ? " character" : " characters");
  buildContextChips(info.context_size);
  buildMenu();
  updateSendButton();
  state.lastStep = null;
  netColors = null;                 // the accent color changed
  drawNetwork(null);
}

// ---------------------------------------------------------------- theme

function setTheme(newTheme) {
  document.documentElement.dataset.theme = newTheme;
  savePreference("alexgpt-theme", newTheme);
  recolorIcons();
  if (state.current) document.documentElement.style.setProperty("--accent", modelColor(state.current));
  netColors = null;                 // read the new theme's colors on the next draw
  drawNetwork(state.lastStep);
}

// ---------------------------------------------------------------- inside panel

function buildContextChips(size) {
  const box = $("contextChips");
  box.innerHTML = "";
  for (let i = 0; i < size; i++) {
    const chip = document.createElement("div");
    chip.className = "ctx special";
    chip.textContent = "␣";
    box.append(chip);
  }
}

function drawContext(contextText) {
  const chips = $("contextChips").children;
  for (let i = 0; i < chips.length && i < contextText.length; i++) {
    const char = contextText[i];
    const chip = chips[i];
    chip.textContent = visibleChar(char);
    chip.className = "ctx";
    if (char === " " || char === "\n") chip.classList.add("special");
    if (i === contextText.length - 1) chip.classList.add("newest");
  }
}

function buildBars() {
  const box = $("bars");
  box.innerHTML = "";
  for (let i = 0; i < 5; i++) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = '<span class="bar-label"></span><div class="bar-track"><div class="bar-fill"></div></div><span class="bar-value"></span>';
    box.append(row);
  }
}

function drawBars(top, pickedChar) {
  const rows = $("bars").children;
  for (let i = 0; i < 5; i++) {
    const row = rows[i];
    const [char, prob] = top[i] || ["", 0];
    row.classList.toggle("picked", char !== "" && char === pickedChar);
    row.children[0].textContent = visibleChar(char);
    row.children[1].firstChild.style.transform = "scaleX(" + prob.toFixed(3) + ")";
    row.children[2].textContent = char === "" ? "" : (prob * 100).toFixed(1) + "%";
  }
}

function neuronColor(value) {
  // -1 -> orange, 0 -> neutral, +1 -> blue
  const base = theme() === "light" ? [236, 234, 230] : [36, 39, 45];
  const target = value >= 0 ? [79, 141, 255] : [240, 122, 69];
  const amount = Math.sqrt(Math.min(1, Math.abs(value)));   // sqrt: small values stay visible
  const red = Math.round(base[0] + (target[0] - base[0]) * amount);
  const green = Math.round(base[1] + (target[1] - base[1]) * amount);
  const blue = Math.round(base[2] + (target[2] - base[2]) * amount);
  return "rgb(" + red + "," + green + "," + blue + ")";
}

// ---------------------------------------------------------------- network view
// The whole model drawn as layers, on one canvas. Everything is placed in a fixed
// 900 x 440 drawing space and scaled to the panel width, so dragging the panel wider zooms in.
//
//  input -> embedding -> x @ W1 + b1 -> tanh -> h · W2 + b2 (logits) -> softmax -> output

const NET_W = 900;
const NET_H = 440;
const NET_TOP = 66;
const NET_BOTTOM = 340;
const PER_COLUMN = 20;                 // hidden neurons are drawn 20 per column
const NET = {
  input: 30,
  embedding: 92,  embeddingWidth: 112,
  pre: 268,       preWidth: 135,
  hidden: 448,    hiddenWidth: 135,
  logits: 690,    logitsHalf: 52,
  output: 818,
};

const LAYER_INFO = {
  input:     "<b>Input.</b> The last characters of the text. This is all the model can see.",
  embedding: "<b>Embedding.</b> Each character is replaced by its row from the embedding table. One column of dots = one character's vector. They are glued into one long vector x.",
  pre:       "<b>x @ W1 + b1.</b> A vector of 200 neurons, drawn as 10 columns of 20 only so it fits. Every neuron mixes all 128 embedding numbers with its own weights (one column of W1) and adds its bias. Values can be any size.",
  hidden:    "<b>tanh → h.</b> Same neurons, same positions as on the left, squashed between −1 and 1. Compare the two blocks: big values saturate to full blue or full orange.",
  logits:    "<b>Logits: h · W2 + b2.</b> The matching step. Each character has a template (its column of W2); its score is the dot product between h and that template. Bars to the right are positive scores.",
  output:    "<b>Softmax → probabilities.</b> exp(score / temperature), divided by the total. Only the differences between scores matter: 1.4 points ahead at temperature 0.7 means e^(1.4/0.7) ≈ 7× more likely. The ringed character was drawn at random.",
  attention: "<b>Attention.</b> The newest character compares its query with the key of every character before it (dot product), softmax turns the scores into shares, and it collects a weighted mix of their values. Thicker arc = bigger share. Quartz is untrained for now, so the shares are still random.",
  table:     "<b>Count table.</b> Pebble has no neurons: it reads the row of the previous character from its table of counts.",
  loss:      "<b>Loss, only during training.</b> −log of the probability given to the real next character. When chatting there is no right answer, so there is no loss.",
  idle:      "Hover a layer to see what it does. Turn on Weights to see which connections carry the most signal for the current character.",
};

let netColors = null;
function networkColors() {
  const style = getComputedStyle(document.documentElement);
  netColors = {
    text: style.getPropertyValue("--text").trim(),
    muted: style.getPropertyValue("--text-muted").trim(),
    faint: style.getPropertyValue("--text-faint").trim(),
    line: style.getPropertyValue("--border-strong").trim(),
    surface: style.getPropertyValue("--surface-2").trim(),
    accent: style.getPropertyValue("--accent").trim(),
  };
}

// evenly spread `count` positions between top and bottom
function spread(index, count, top, bottom) {
  if (count <= 1) return (top + bottom) / 2;
  return top + index * (bottom - top) / (count - 1);
}

// position of value number `index` inside a block of dots that is `perColumn` dots tall
function dotPosition(index, perColumn, columns, left, width) {
  const column = Math.floor(index / perColumn);
  const row = index % perColumn;
  const gap = columns > 1 ? width / (columns - 1) : 0;
  return { x: left + column * gap, y: spread(row, perColumn, NET_TOP, NET_BOTTOM) };
}

function edgeColor(value) {
  const alpha = (0.12 + 0.6 * Math.min(1, Math.abs(value))).toFixed(3);
  return value >= 0 ? "rgba(79,141,255," + alpha + ")" : "rgba(240,122,69," + alpha + ")";
}

function circle(context, x, y, radius, fill) {
  context.fillStyle = fill;
  context.beginPath();
  context.arc(x, y, radius, 0, Math.PI * 2);
  context.fill();
}

function line(context, x1, y1, x2, y2) {
  context.beginPath();
  context.moveTo(x1, y1);
  context.lineTo(x2, y2);
  context.stroke();
}

function drawNetwork(step) {
  const canvas = $("network");
  const shownWidth = canvas.clientWidth;
  if (shownWidth === 0) return;                         // panel closed
  // match the canvas pixels to its size on screen, so it stays sharp when the panel grows
  const pixelRatio = window.devicePixelRatio || 1;
  const pixelWidth = Math.round(shownWidth * pixelRatio);
  const pixelHeight = Math.round(pixelWidth * NET_H / NET_W);
  if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
  if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
  const context = canvas.getContext("2d");
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, canvas.width, canvas.height);
  const scale = pixelWidth / NET_W;
  context.setTransform(scale, 0, 0, scale, 0, 0);       // from here on, draw in 900 x 440 units

  const info = state.available[state.current];
  if (!info) return;
  if (!netColors) networkColors();
  const colors = netColors;
  const layers = info.layers || {};
  const isMLP = layers.hidden > 0;
  const showWeights = $("weightsToggle").checked;
  const contextSize = info.context_size;
  const sans = getComputedStyle(document.body).fontFamily;
  const mono = "JetBrains Mono, ui-monospace, Consolas, monospace";
  const top = step ? step.top : [];

  if (layers.attention) {
    drawAttentionView(context, step, info, colors, sans, mono);
    $("networkLabel").textContent = formatParams(info.params);
    return;
  }

  // --- layer names
  context.font = "500 17px " + sans;
  context.fillStyle = colors.faint;
  context.textAlign = "center";
  context.textBaseline = "alphabetic";
  context.fillText("input", NET.input, 26);
  if (isMLP) {
    context.fillText("embedding", NET.embedding + NET.embeddingWidth / 2, 26);
    context.fillText("x @ W1 + b1", NET.pre + NET.preWidth / 2, 26);
    context.fillText("tanh → h", NET.hidden + NET.hiddenWidth / 2, 26);
    context.fillText("logits · h·W2 + b2", NET.logits, 26);
  } else {
    context.fillText("count table", (NET.input + NET.output) / 2, 26);
  }
  context.fillText("softmax", NET.output + 22, 26);

  // --- where every node sits
  const inputY = [];
  for (let i = 0; i < contextSize; i++) inputY.push(spread(i, contextSize, NET_TOP, NET_BOTTOM));
  const outputY = [];
  for (let k = 0; k < 5; k++) outputY.push(spread(k, 5, NET_TOP + 14, NET_BOTTOM - 44));
  const embedDim = layers.embed_dim || 0;
  const hiddenCount = layers.hidden || 0;
  const hiddenColumns = Math.ceil(hiddenCount / PER_COLUMN);
  const embeddingAt = (index) => dotPosition(index, embedDim, contextSize, NET.embedding, NET.embeddingWidth);
  const preAt = (index) => dotPosition(index, PER_COLUMN, hiddenColumns, NET.pre, NET.preWidth);
  const hiddenAt = (index) => dotPosition(index, PER_COLUMN, hiddenColumns, NET.hidden, NET.hiddenWidth);

  // --- lines (drawn first, so the dots sit on top)
  context.lineCap = "round";
  if (isMLP) {
    // input -> its column in the embedding (a lookup in the table, not weights)
    context.lineWidth = 1.2;
    context.strokeStyle = colors.line;
    for (let i = 0; i < contextSize; i++) {
      const first = embeddingAt(i * embedDim);
      line(context, NET.input + 16, inputY[i], first.x - 7, (NET_TOP + NET_BOTTOM) / 2);
    }
    // tanh: each neuron keeps its place, a short arrow between the two blocks says so
    context.strokeStyle = colors.line;
    context.lineWidth = 1.5;
    const middle = (NET_TOP + NET_BOTTOM) / 2;
    line(context, NET.pre + NET.preWidth + 14, middle, NET.hidden - 14, middle);
    line(context, NET.hidden - 20, middle - 5, NET.hidden - 14, middle);
    line(context, NET.hidden - 20, middle + 5, NET.hidden - 14, middle);

    if (showWeights && step && step.edges_in) {
      // embedding -> neurons: the strongest x[i] * W1[i][j]
      for (const [from, to, value] of step.edges_in) {
        const a = embeddingAt(from);
        const b = preAt(to);
        context.strokeStyle = edgeColor(value);
        context.lineWidth = 0.8 + 1.8 * Math.abs(value);
        line(context, a.x, a.y, b.x, b.y);
      }
      // neurons -> each top character's score: the strongest h[j] * W2[j][char]
      for (const [from, to, value] of step.edges_out) {
        const a = hiddenAt(from);
        context.strokeStyle = edgeColor(value);
        context.lineWidth = 0.8 + 1.8 * Math.abs(value);
        line(context, a.x, a.y, NET.logits - NET.logitsHalf - 10, outputY[to]);
      }
    } else if (step) {
      // without weights: one faint line from the neurons to every score
      context.strokeStyle = colors.line;
      context.lineWidth = 1;
      for (let k = 0; k < top.length; k++) line(context, NET.hidden + NET.hiddenWidth + 10, (NET_TOP + NET_BOTTOM) / 2, NET.logits - NET.logitsHalf - 10, outputY[k]);
    }
    // score -> softmax
    if (step) {
      context.strokeStyle = colors.line;
      context.lineWidth = 1;
      for (let k = 0; k < top.length; k++) line(context, NET.logits + NET.logitsHalf + 34, outputY[k], NET.output - 19, outputY[k]);
    }
  } else if (step) {
    // Pebble: the previous character points straight at the probabilities
    context.strokeStyle = colors.accent;
    for (let k = 0; k < top.length; k++) {
      context.globalAlpha = 0.15 + 0.85 * top[k][1];
      context.lineWidth = 1 + 4 * top[k][1];
      line(context, NET.input + 16, inputY[0], NET.output - 19, outputY[k]);
    }
    context.globalAlpha = 1;
  }

  // --- input characters
  context.textAlign = "center";
  context.textBaseline = "middle";
  for (let i = 0; i < contextSize; i++) {
    const char = step ? step.context[i] : " ";
    circle(context, NET.input, inputY[i], 15, colors.surface);
    if (step && i === contextSize - 1) {
      context.strokeStyle = colors.accent;
      context.lineWidth = 2;
      context.beginPath(); context.arc(NET.input, inputY[i], 15, 0, Math.PI * 2); context.stroke();
    }
    context.fillStyle = char === " " || char === "\n" ? colors.faint : colors.text;
    context.font = "500 16px " + mono;
    context.fillText(visibleChar(char || " "), NET.input, inputY[i] + 1);
  }

  if (isMLP) {
    // embedding
    const embedding = step && step.embedding ? step.embedding : null;
    let biggest = 1e-9;
    if (embedding) for (const value of embedding) biggest = Math.max(biggest, Math.abs(value));
    for (let index = 0; index < contextSize * embedDim; index++) {
      const p = embeddingAt(index);
      circle(context, p.x, p.y, 4.5, neuronColor(embedding ? embedding[index] / biggest : 0));
    }
    // before tanh (scaled by its biggest value, so the colors show the shape, not the size)
    const pre = step && step.pre ? step.pre : null;
    let biggestPre = 1e-9;
    if (pre) for (const value of pre) biggestPre = Math.max(biggestPre, Math.abs(value));
    for (let index = 0; index < hiddenCount; index++) {
      const p = preAt(index);
      circle(context, p.x, p.y, 4.8, neuronColor(pre ? pre[index] / biggestPre : 0));
    }
    // after tanh
    const neurons = step && step.neurons ? step.neurons : null;
    for (let index = 0; index < hiddenCount; index++) {
      const p = hiddenAt(index);
      circle(context, p.x, p.y, 4.8, neuronColor(neurons ? neurons[index] : 0));
    }
    if (pre) {
      context.font = "400 13px " + mono;
      context.fillStyle = colors.faint;
      context.fillText(String(contextSize * embedDim), NET.embedding + NET.embeddingWidth / 2, NET_BOTTOM + 24);
      context.fillText(String(hiddenCount), NET.pre + NET.preWidth / 2, NET_BOTTOM + 24);
      context.fillText(hiddenCount + " · −1…1", NET.hidden + NET.hiddenWidth / 2, NET_BOTTOM + 24);
    }

    // logits: one signed bar per top character, centered on zero
    const scores = step && step.scores ? step.scores : null;
    let biggestScore = 1e-9;
    if (scores) for (const value of scores) biggestScore = Math.max(biggestScore, Math.abs(value));
    context.strokeStyle = colors.line;
    context.lineWidth = 1;
    line(context, NET.logits, NET_TOP, NET.logits, NET_BOTTOM - 30);
    for (let k = 0; k < 5; k++) {
      if (!scores || k >= scores.length) continue;
      const length = (scores[k] / biggestScore) * NET.logitsHalf;
      context.fillStyle = scores[k] >= 0 ? "rgba(79,141,255,0.85)" : "rgba(240,122,69,0.85)";
      context.beginPath();
      context.roundRect(Math.min(NET.logits, NET.logits + length), outputY[k] - 7, Math.abs(length), 14, 3);
      context.fill();
      context.font = "400 14px " + mono;
      context.fillStyle = colors.muted;
      context.textAlign = "left";
      context.fillText(scores[k].toFixed(1), NET.logits + NET.logitsHalf + 6, outputY[k] + 1);
      context.textAlign = "center";
    }
  } else {
    context.fillStyle = colors.faint;
    context.font = "400 16px " + sans;
    context.fillText("one row of counts", (NET.input + NET.output) / 2, NET_BOTTOM + 28);
  }

  drawOutputs(context, step, info, colors, sans, mono, outputY);

  $("networkLabel").textContent = formatParams(info.params);
}


// output column (softmax) and the dashed loss box, shared by every model
function drawOutputs(context, step, info, colors, sans, mono, outputY) {
  const top = step ? step.top : [];
  context.textAlign = "center";
  context.textBaseline = "middle";
  // output: the 5 most likely characters after softmax
  for (let k = 0; k < 5; k++) {
    circle(context, NET.output, outputY[k], 17, colors.surface);
    if (!top[k]) continue;
    const [char, prob] = top[k];
    context.globalAlpha = 0.2 + 0.8 * prob;
    circle(context, NET.output, outputY[k], 17, colors.accent);
    context.globalAlpha = 1;
    if (char === step.char) {
      context.strokeStyle = colors.text;
      context.lineWidth = 2;
      context.beginPath(); context.arc(NET.output, outputY[k], 20.5, 0, Math.PI * 2); context.stroke();
    }
    context.fillStyle = prob > 0.5 ? colors.surface : colors.text;
    context.font = "500 16px " + mono;
    context.fillText(visibleChar(char), NET.output, outputY[k] + 1);
    context.textAlign = "left";
    context.fillStyle = colors.muted;
    context.font = "400 15px " + mono;
    context.fillText(Math.round(prob * 100) + "%", NET.output + 28, outputY[k] + 1);
    context.textAlign = "center";
  }
  context.fillStyle = colors.faint;
  context.font = "400 14px " + sans;
  context.fillText("+" + Math.max(0, info.vocab_size - 5) + " more", NET.output + 16, NET_BOTTOM - 6);

  // loss: dashed, because it only exists during training
  context.setLineDash([5, 5]);
  context.strokeStyle = colors.faint;
  context.lineWidth = 1.5;
  context.beginPath(); context.roundRect(NET.output - 50, 384, 128, 34, 10); context.stroke();
  context.setLineDash([]);
  context.fillStyle = colors.faint;
  context.font = "500 15px " + sans;
  context.fillText("loss · training", NET.output + 14, 402);
}

// Quartz: the context as one row of characters; arcs show where the newest character looks
function drawAttentionView(context, step, info, colors, sans, mono) {
  const contextSize = info.context_size;
  const rowY = 250;
  const left = 24;
  const right = 640;
  const gap = (right - left) / (contextSize - 1);
  const weights = step && step.attention ? step.attention : null;
  let biggest = 1e-9;
  if (weights) for (const value of weights) biggest = Math.max(biggest, value);

  context.font = "500 17px " + sans;
  context.fillStyle = colors.faint;
  context.textAlign = "left";
  context.textBaseline = "alphabetic";
  context.fillText("context · where the newest character looks", left - 10, 26);
  context.textAlign = "center";
  context.fillText("softmax", NET.output + 22, 26);

  // arcs from the newest character to every character it can see
  const newestX = right;
  if (weights) {
    context.lineCap = "round";
    for (let s = 0; s < contextSize; s++) {
      const share = weights[s] / biggest;                     // 1 = the position it looks at most
      if (share < 0.02) continue;
      const x = left + s * gap;
      const height = 30 + (newestX - x) * 0.32;
      context.strokeStyle = colors.accent;
      context.globalAlpha = 0.12 + 0.88 * share;
      context.lineWidth = 1 + 7 * share;
      context.beginPath();
      context.moveTo(newestX, rowY - 14);
      context.quadraticCurveTo((newestX + x) / 2, rowY - 14 - height, x, rowY - 14);
      context.stroke();
    }
    context.globalAlpha = 1;
  }

  // the characters, with a small bar under each one: its share of the attention
  context.textBaseline = "middle";
  for (let s = 0; s < contextSize; s++) {
    const x = left + s * gap;
    const char = step ? step.context[s] : " ";
    const newest = s === contextSize - 1;
    circle(context, x, rowY, 10, colors.surface);
    if (newest && step) {
      context.strokeStyle = colors.accent;
      context.lineWidth = 2;
      context.beginPath(); context.arc(x, rowY, 10, 0, Math.PI * 2); context.stroke();
    }
    context.fillStyle = char === " " || char === "\n" ? colors.faint : colors.text;
    context.font = "500 13px " + mono;
    context.fillText(visibleChar(char || " "), x, rowY + 1);
    if (weights) {
      const barHeight = 70 * weights[s] / biggest;
      context.fillStyle = colors.accent;
      context.globalAlpha = 0.85;
      context.fillRect(x - 4, rowY + 18, 8, barHeight);
      context.globalAlpha = 1;
    }
  }
  if (weights) {
    // name the position it looks at most
    let best = 0;
    for (let s = 1; s < contextSize; s++) if (weights[s] > weights[best]) best = s;
    context.font = "400 14px " + mono;
    context.fillStyle = colors.muted;
    context.fillText(Math.round(weights[best] * 100) + "%", left + best * gap, rowY + 106);
  }

  // newest character -> outputs
  const outputY = [];
  for (let k = 0; k < 5; k++) outputY.push(spread(k, 5, NET_TOP + 14, NET_BOTTOM - 44));
  if (step) {
    context.strokeStyle = colors.line;
    context.lineWidth = 1;
    for (let k = 0; k < step.top.length; k++) line(context, newestX + 14, rowY, NET.output - 19, outputY[k]);
  }
  drawOutputs(context, step, info, colors, sans, mono, outputY);
}

// hovering a layer explains it under the drawing
function layerAt(x, y) {
  const info = state.available[state.current];
  const isMLP = info && info.layers && info.layers.hidden > 0;
  if (x > NET.output - 55 && y > 375) return "loss";
  if (info && info.layers && info.layers.attention) return x > NET.output - 40 ? "output" : "attention";
  if (x < 60) return "input";
  if (x > NET.output - 40) return "output";
  if (!isMLP) return "table";
  if (x < NET.pre - 20) return "embedding";
  if (x < NET.hidden - 15) return "pre";
  if (x < NET.logits - NET.logitsHalf - 20) return "hidden";
  return "logits";
}
function showLayerNote(name) { $("layerNote").innerHTML = LAYER_INFO[name]; }

$("network").addEventListener("mousemove", (event) => {
  const canvas = $("network");
  const scale = NET_W / canvas.clientWidth;
  showLayerNote(layerAt(event.offsetX * scale, event.offsetY * scale));
});
$("network").addEventListener("mouseleave", () => showLayerNote("idle"));
showLayerNote("idle");

$("weightsToggle").checked = loadPreference("alexgpt-weights") === "on";
$("weightsToggle").addEventListener("change", () => {
  savePreference("alexgpt-weights", $("weightsToggle").checked ? "on" : "off");
  drawNetwork(state.lastStep);
});

function showStats(count, confidenceSum, startTime, ending) {
  const seconds = (performance.now() - startTime) / 1000;
  $("statChars").textContent = count;
  $("statConfidence").textContent = count ? Math.round((confidenceSum / count) * 100) + "%" : "—";
  $("statSpeed").textContent = seconds > 0.05 ? Math.round(count / seconds) : "—";
  $("statEnding").textContent = ending;
}

// ---------------------------------------------------------------- chat

// The chat follows the new text only while you are at the bottom.
// The moment you scroll up, it stops following; scroll back down and it follows again.
let followBottom = true;
let lastScrollTop = 0;

function onMessagesScroll() {
  const box = $("messages");
  const distanceToBottom = box.scrollHeight - box.scrollTop - box.clientHeight;
  if (box.scrollTop < lastScrollTop - 1) {
    followBottom = false;                                        // you moved up
  } else if (distanceToBottom < 30) {
    followBottom = true;                                         // you came back to the bottom
  }
  lastScrollTop = box.scrollTop;
}

function scrollToBottom(force) {
  const box = $("messages");
  if (force) followBottom = true;
  if (followBottom) {
    box.scrollTop = box.scrollHeight;
    lastScrollTop = box.scrollTop;
  }
}

function addUserMessage(text) {
  const message = document.createElement("div");
  message.className = "message user done";
  const bubble = document.createElement("div");
  bubble.className = "user-bubble";
  bubble.textContent = text;
  message.append(bubble);
  $("messages").append(message);
}

function addBotMessage(prompt) {
  const message = document.createElement("div");
  message.className = "message bot";

  const avatar = document.createElement("div");
  avatar.className = "bot-avatar thinking";
  avatar.append(makeIcon(state.current, 16));

  const body = document.createElement("div");
  body.className = "bot-body";
  const meta = document.createElement("div");
  meta.className = "bot-meta";
  meta.textContent = state.current;
  const text = document.createElement("div");
  text.className = "bot-text";

  // the end of the prompt, faded, so the continuation reads naturally
  const echo = document.createElement("span");
  echo.className = "echo";
  echo.textContent = prompt.length > 28 ? "…" + prompt.slice(-28) : prompt;
  const caret = document.createElement("span");
  caret.className = "caret";
  text.append(echo, caret);

  body.append(meta, text);
  message.append(avatar, body);
  $("messages").append(message);
  return { message, avatar, text, caret };
}

// A letter is "settled" once its fade-in animation is over (~0.5 s after it appeared).
// Settled letters are glued to their neighbour, so the reply never holds thousands of elements:
//   - a sure letter (p4) is appended to the plain text just before it
//   - an unsure letter joins the previous colored piece if it has the same color
// This runs while the reply is still being written, not only at the end.
function settleLetters(spans) {
  for (const span of spans) {
    if (!span.parentNode) continue;                 // already removed
    const letter = span.textContent;
    const level = span.dataset.level;
    const previous = span.previousSibling;

    if (level === "p4") {
      if (previous && previous.nodeType === Node.TEXT_NODE) {
        previous.data += letter;                    // glue to the text before it
        span.remove();
      } else {
        span.replaceWith(document.createTextNode(letter));
      }
      continue;
    }

    const sameColorBefore = previous && previous.nodeType === Node.ELEMENT_NODE &&
      previous.classList.contains("settled") && previous.dataset.level === level;
    if (sameColorBefore) {
      previous.firstChild.data += letter;           // glue to the colored piece before it
      previous.removeAttribute("title");            // a piece of several letters has no single %
      span.remove();
    } else {
      span.className = "tok settled " + level;      // start a new colored piece
    }
  }
}

// ---------------------------------------------------------------- pause / step
// While paused, letters keep arriving from the server but wait in a queue.
// "Next" draws exactly one of them; "Resume" draws the rest at normal speed.
let stepOnce = null;        // set by send(): draws one waiting letter
let resumeDrawing = null;   // set by send(): starts drawing again

function setPaused(on) {
  state.paused = on;
  $("pauseButton").classList.toggle("paused", on);
  $("pauseButton").setAttribute("aria-label", on ? "Resume" : "Pause");
  $("pauseButton").title = on ? "Resume" : "Pause to see every step";
  $("stepButton").hidden = !on;
  if (on) $("chosenNote").textContent = "paused · press → for the next character";
  else if (resumeDrawing) resumeDrawing();
}

function setGenerating(on) {
  state.generating = on;
  $("pauseButton").hidden = !on;
  if (!on) $("stepButton").hidden = true;
  const button = $("sendButton");
  button.classList.toggle("stop", on);
  button.setAttribute("aria-label", on ? "Stop" : "Send");
  updateSendButton();
}

function updateSendButton() {
  $("sendButton").disabled = !state.generating && ($("input").value.trim() === "" || !state.current);
}

async function send(prompt) {
  if (!state.current) return;
  const empty = $("emptyState");
  if (empty) empty.remove();

  addUserMessage(prompt);
  const bot = addBotMessage(prompt);
  scrollToBottom(true);
  setGenerating(true);

  const controller = new AbortController();
  state.controller = controller;
  stepOnce = () => renderFrame(1);
  // after a pause, the letters that piled up are played back at the chosen speed, not all at once
  resumeDrawing = () => {
    if (pending.length <= 1) {
      if (!frameRequested) { frameRequested = true; requestAnimationFrame(() => renderFrame()); }
      return;
    }
    catchingUp = true;
    const playNext = () => {
      if (state.paused || controller.signal.aborted) { catchingUp = false; return; }
      if (pending.length === 0) { catchingUp = false; return; }   // caught up: back to normal drawing
      const delay = parseInt($("speed").value);
      renderFrame(delay === 0 ? 8 : 1);                           // speed "max": 8 letters per frame
      if (delay === 0) requestAnimationFrame(playNext);
      else setTimeout(playNext, delay);
    };
    playNext();
  };

  const startTime = performance.now();
  let count = 0;
  let confidenceSum = 0;
  let ending = "";

  // characters arrive faster than the screen refreshes, so they wait here
  // and get drawn together once per frame (this is what keeps it smooth)
  let pending = [];
  let frameRequested = false;
  let catchingUp = false;     // true while replaying letters that piled up during a pause

  // limit = how many waiting letters to draw (undefined = all of them)
  function renderFrame(limit) {
    frameRequested = false;
    if (pending.length === 0) return;
    if ((state.paused || catchingUp) && limit === undefined) return;   // paused or replaying: letters wait their turn

    let take = pending.length;
    if (limit !== undefined && limit < take) take = limit;
    const drawn = pending.splice(0, take);                  // remove them from the front of the queue

    const fragment = document.createDocumentFragment();
    const batch = [];
    for (const step of drawn) {
      count += 1;
      confidenceSum += step.prob;
      const span = document.createElement("span");
      span.dataset.level = confidenceClass(step.prob);
      span.className = "tok new " + span.dataset.level;
      span.textContent = step.char;
      span.title = (step.prob * 100).toFixed(1) + "%";
      fragment.append(span);
      batch.push(span);
    }
    bot.text.insertBefore(fragment, bot.caret);
    // once the fade-in is over, glue these letters to the text before them
    setTimeout(() => settleLetters(batch), 500);

    const latest = drawn[drawn.length - 1];

    if (panelOpen()) {
      drawContext(latest.context);
      $("chosenChar").textContent = visibleChar(latest.char);
      $("chosenProb").textContent = (latest.prob * 100).toFixed(1) + "%";
      $("chosenNote").textContent = state.paused ? "paused · press → for the next character" : "chance of the letter it picked";
      drawBars(latest.top, latest.char);
      state.lastStep = latest;
      drawNetwork(latest);
      showStats(count, confidenceSum, startTime, "Writing…");
    } else {
      state.lastStep = latest;
    }
    scrollToBottom(false);
  }

  // a letter goes on screen: count it and wait for the next frame
  let held = [];
  function show(step) {
    pending.push(step);
    if (!frameRequested && !state.paused && !catchingUp) {
      frameRequested = true;
      requestAnimationFrame(renderFrame);
    }
  }

  try {
    const lengthValue = parseInt($("length").value);
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: state.current,
        prompt: prompt,
        temperature: parseFloat($("temperature").value),
        max_chars: lengthValue >= 850 ? 0 : lengthValue,   // 0 = auto
        delay_ms: parseInt($("speed").value),
      }),
      signal: controller.signal,
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let end = buffer.indexOf("\n\n");
      while (end >= 0) {
        const line = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        end = buffer.indexOf("\n\n");
        if (!line.startsWith("data: ")) continue;
        const step = JSON.parse(line.slice(6));

        if (step.type === "token") {
          if (count === 0 && held.length === 0) bot.avatar.classList.remove("thinking");
          // letters that could be the start of "<|endoftext|>" wait, so the marker never shows
          let heldText = "";
          for (const heldStep of held) heldText += heldStep.char;
          if (END_MARKER.startsWith(heldText + step.char)) {
            held.push(step);
          } else {
            for (const heldStep of held) show(heldStep);   // false alarm: show them
            held = [];
            if (step.char === END_MARKER[0]) held.push(step);
            else show(step);
          }
        } else if (step.type === "end_of_story") {
          held = [];                                       // the marker: throw it away
          ending = "Ended the story on its own.";
        }
      }
    }
  } catch (error) {
    if (error.name !== "AbortError") ending = "Error: " + error.message;
  }

  for (const heldStep of held) show(heldStep);   // stopped in the middle of a "<": show it

  // the server may be done while letters still wait (paused): keep drawing until the queue is empty
  stepOnce = () => renderFrame(1);
  while (pending.length > 0 && !controller.signal.aborted) {
    if (!state.paused && !catchingUp) renderFrame();
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  stepOnce = null;
  setPaused(false);

  if (ending === "Ended the story on its own." && !controller.signal.aborted) {
    $("chosenChar").textContent = "■";
    $("chosenProb").textContent = "End";
    $("chosenNote").textContent = "the model wrote <|endoftext|>";
  }
  if (controller.signal.aborted) ending = "Stopped by you.";
  if (ending === "") ending = "Reached the length limit.";
  showStats(count, confidenceSum, startTime, ending);

  bot.caret.remove();
  bot.avatar.classList.remove("thinking");
  bot.message.classList.add("done");             // off-screen finished replies are skipped when drawing
  state.controller = null;
  setGenerating(false);
}

// ---------------------------------------------------------------- wiring

function autoGrow() {
  const input = $("input");
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
}

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.generating) {
    state.controller.abort();
    setPaused(false);
    return;
  }
  const text = $("input").value.trim();
  if (!text || !state.current) return;
  $("input").value = "";
  autoGrow();
  send(text);
});

$("input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});
$("input").addEventListener("input", () => { autoGrow(); updateSendButton(); });
$("messages").addEventListener("scroll", onMessagesScroll, { passive: true });

for (const chip of document.querySelectorAll(".chip")) {
  chip.addEventListener("click", () => {
    $("input").value = chip.textContent;
    autoGrow();
    updateSendButton();
    $("composer").requestSubmit();
  });
}

$("modelButton").addEventListener("click", (event) => {
  event.stopPropagation();
  $("modelPicker").classList.toggle("open");
});
document.addEventListener("click", (event) => {
  if (!$("modelPicker").contains(event.target)) $("modelPicker").classList.remove("open");
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") $("modelPicker").classList.remove("open");
  // while paused, the right arrow draws the next character (unless you are typing)
  if (event.key === "ArrowRight" && state.paused && stepOnce && document.activeElement !== $("input")) {
    event.preventDefault();
    stepOnce();
  }
});
$("pauseButton").addEventListener("click", () => setPaused(!state.paused));
$("stepButton").addEventListener("click", () => { if (stepOnce) stepOnce(); });

$("themeToggle").addEventListener("click", () => {
  setTheme(theme() === "light" ? "dark" : "light");
});

$("panelToggle").addEventListener("click", () => {
  const open = $("insidePanel").classList.toggle("open");
  $("panelToggle").classList.toggle("active", open);
  savePreference("alexgpt-panel", open ? "open" : "closed");
  if (open) drawNetwork(state.lastStep);
});
// ---- drag the left edge of the visualisation panel to resize it
const PANEL_MIN = 320;
function setPanelWidth(width) {
  const maximum = Math.max(PANEL_MIN, window.innerWidth - 380);   // always leave room for the chat
  const clamped = Math.round(Math.min(maximum, Math.max(PANEL_MIN, width)));
  document.documentElement.style.setProperty("--panel-width", clamped + "px");
  return clamped;
}
let resizing = false;
$("resizeHandle").addEventListener("pointerdown", (event) => {
  resizing = true;
  $("insidePanel").classList.add("resizing");
  document.body.classList.add("resizing-panel");
  $("resizeHandle").setPointerCapture(event.pointerId);
  event.preventDefault();
});
$("resizeHandle").addEventListener("pointermove", (event) => {
  if (!resizing) return;
  setPanelWidth(window.innerWidth - event.clientX);
  drawNetwork(state.lastStep);                                   // redraw at the new size
});
$("resizeHandle").addEventListener("pointerup", () => {
  if (!resizing) return;
  resizing = false;
  $("insidePanel").classList.remove("resizing");
  document.body.classList.remove("resizing-panel");
  const width = parseInt(getComputedStyle(document.documentElement).getPropertyValue("--panel-width"));
  savePreference("alexgpt-panel-width", String(width));
});
$("resizeHandle").addEventListener("dblclick", () => {             // double click: back to the default size
  setPanelWidth(460);
  savePreference("alexgpt-panel-width", "460");
  drawNetwork(state.lastStep);
});
setPanelWidth(parseInt(loadPreference("alexgpt-panel-width")) || 460);
window.addEventListener("resize", () => drawNetwork(state.lastStep));
$("insidePanel").addEventListener("transitionend", () => drawNetwork(state.lastStep));

if (loadPreference("alexgpt-panel") !== "closed") {
  $("insidePanel").classList.add("open");
  $("panelToggle").classList.add("active");
}

function setConfidenceColors(on) {
  $("confidenceToggle").checked = on;
  $("messages").classList.toggle("show-confidence", on);
  $("confidenceLegend").classList.toggle("visible", on);
  savePreference("alexgpt-confidence", on ? "on" : "off");
}
$("confidenceToggle").addEventListener("change", () => setConfidenceColors($("confidenceToggle").checked));
setConfidenceColors(loadPreference("alexgpt-confidence") !== "off");

function bindSlider(id, format) {
  const slider = $(id);
  const show = () => { $(id + "Value").textContent = format(parseFloat(slider.value)); };
  slider.addEventListener("input", show);
  show();
}
bindSlider("speed", (ms) => (ms === 0 ? "max" : ms + " ms"));
bindSlider("temperature", (t) => t.toFixed(2));
bindSlider("length", (n) => (n >= 850 ? "Auto" : String(n)));

// brand mark: the Diamond icon, in the text color
const brandIcon = makeIcon("Diamond", 18);
brandIcon.removeAttribute("data-model");
brandIcon.style.setProperty("--icon-color", "var(--text)");
$("brandMark").append(brandIcon);

buildBars();
drawBars([], "");
updateSendButton();
loadModels();
