// AlexGPT interface: talks to server.py and draws what the model does.

// Small line icons, one per model (18x18, drawn with strokes).
const ICONS = {
  // a four-point star with a small sparkle next to it
  Spark:       '<path d="M8.4 2.2c.5 3.6 2.6 5.7 6.2 6.2-3.6.5-5.7 2.6-6.2 6.2-.5-3.6-2.6-5.7-6.2-6.2 3.6-.5 5.7-2.6 6.2-6.2Z" fill="currentColor" fill-opacity=".18"/><path d="M14.6 1.6v3M13.1 3.1h3"/>',
  // a cloud of gas curling into itself, with two young stars
  Nebula:      '<path d="M2.6 11.6C2.4 8.1 5.1 5.4 8.5 5.4c2.7 0 4.8 1.8 4.8 4.2 0 1.9-1.5 3.4-3.4 3.4-1.6 0-2.8-1.1-2.8-2.5 0-1.1.9-1.9 1.9-1.9"/><path d="M2.2 14.6c2.3 1.3 5.2 1.6 7.9 1 2.6-.6 4.7-2.2 6-4.3"/><circle cx="14.6" cy="4.2" r=".9" fill="currentColor" stroke="none"/><circle cx="4.2" cy="3.8" r=".6" fill="currentColor" stroke="none"/>',
  // a spinning star with two beams and its tilted orbit
  Pulsar:      '<path d="M9 9L14.8 1.8M9 9L3.2 16.2"/><path d="M11.6 5.8l3.8-1.6-1.1-2.3M6.4 12.2l-3.8 1.6 1.1 2.3" stroke-opacity=".5"/><ellipse cx="9" cy="9" rx="7" ry="2.3" stroke-opacity=".55"/><circle cx="9" cy="9" r="2.2" fill="currentColor" stroke="none"/>',
  // a spiral galaxy with two arms around a bright core
  Galaxy:      '<path d="M9 9c1.5-1.7 4.3-1.4 5.4.6 1.3 2.3-.2 5.2-2.9 5.6"/><path d="M9 9c-1.5 1.7-4.3 1.4-5.4-.6-1.3-2.3.2-5.2 2.9-5.6"/><circle cx="9" cy="9" r="1.5" fill="currentColor" stroke="none"/><circle cx="15.2" cy="3.4" r=".6" fill="currentColor" stroke="none"/><circle cx="2.8" cy="14.6" r=".6" fill="currentColor" stroke="none"/>',
  // a nova: a small hot star with rays, pulling a dotted stream of gas from its bigger companion
  Nova:        '<circle cx="11.6" cy="6.8" r="1.6" fill="currentColor" stroke="none"/><path d="M11.6 2.6v1.4M11.6 9.6v1.4M7.4 6.8h1.4M14.4 6.8h1.4" stroke-opacity=".7"/><circle cx="4.8" cy="12.6" r="2.7" fill="currentColor" fill-opacity=".18"/><path d="M7 11.2c1.7-.7 3-1.9 3.9-3.3" stroke-dasharray="1 1.5"/>',
  // an exploding star: bright core, shock wave, flying rays
  Supernova:   '<circle cx="9" cy="9" r="2.3" fill="currentColor" fill-opacity=".35"/><circle cx="9" cy="9" r="5" stroke-dasharray="1.4 1.9"/><path d="M9 .8v2M9 15.2v2M.8 9h2M15.2 9h2M3.2 3.2l1.4 1.4M13.4 13.4l1.4 1.4M3.2 14.8l1.4-1.4M13.4 4.6l1.4-1.4"/>',
  // a black hole: a dark centre, its bright edge, and the disc of matter crossing in front
  Singularity: '<path d="M1.4 9.4c.4-1.6 3.7-2.9 7.6-2.9s7.2 1.3 7.6 2.9" stroke-opacity=".55"/><circle cx="9" cy="9" r="3.4" style="fill: var(--bg)"/><path d="M1.4 9.4c.4 1.6 3.7 2.9 7.6 2.9s7.2-1.3 7.6-2.9"/>',
};

// The full family. Models the server has loaded become selectable; the rest show as "soon".
const FAMILY = [
  { name: "Spark",       dark: "#f3c969", light: "#a8741a", note: "Bigram · sees 1 character" },
  { name: "Nebula",      dark: "#e9a6dc", light: "#a1408c", note: "MLP · sees 8 characters" },
  { name: "Pulsar",      dark: "#7fd8ea", light: "#1d8199", note: "Transformer · sees 32 characters" },
  { name: "Galaxy",      dark: "#a9a5ff", light: "#5650d4", note: "GPU · BPE · conversations" },
  { name: "Nova",        dark: "#ff8fa3", light: "#c23a55", note: "Supernova before SFT: continues your text" },
  { name: "Supernova",   dark: "#ff9d6e", light: "#cc5326", note: "Answers you, EN + RO" },
  { name: "Singularity Base", dark: "#c9c5ff", light: "#4a3f8f", note: "Singularity before SFT: continues your text" },
  { name: "Singularity", dark: "#e6e3ff", light: "#2c2548", note: "Answers your questions" },
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
  const paths = ICONS[name] || ICONS[name.split(" ")[0]] || ICONS.Singularity;   // "Supernova Chat" -> Supernova's icon
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
  if (char.length > 1) return char.replace(/\n/g, "↵").replace(/^ /, "␣");   // a BPE token like " the"
  return char;
}

// Galaxy and later models chat (BPE tokens, conversation format) instead of continuing your text
function isChatModel(name) {
  const info = state.available[name];
  return !!(info && info.layers && info.layers.chat);
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

  const openedChat = findChat(currentChatId);           // the chat open at start keeps its own model
  const saved = openedChat && openedChat.model ? openedChat.model : loadPreference("alexgpt-model");
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
  // the open chat remembers the model you picked, even before you write in it
  try {
    const openChat = findChat(currentChatId);
    if (openChat && openChat.model !== name) { openChat.model = name; saveChats(); }
  } catch (e) { /* chats not loaded yet */ }
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
  document.documentElement.dataset.model = name;
  document.documentElement.dataset.chat = isChatModel(name) ? "1" : "0";
  $("length").dispatchEvent(new Event("input"));            // the Length label means something else for chat models          // lets the CSS give every model its own animations
  setCosmosModel(name);

  if (isChatModel(name)) {
    $("contextLabel").textContent = info.context_size + " tokens · last 32 shown";
    buildContextChips(32);
  } else {
    $("contextLabel").textContent = info.context_size + (info.context_size === 1 ? " character" : " characters");
    buildContextChips(info.context_size);
  }
  buildMenu();
  updateSendButton();
  state.lastStep = null;
  netColors = null;                 // the accent color changed
  attentionBlock = null;
  flowReset();                      // flow.js: forget the other model's tokens
  // the Weights switch only means something for Nebula's network
  const hasWeights = info.layers && info.layers.hidden > 0 && !info.layers.transformer;
  $("weightsToggle").closest("label").style.visibility = hasWeights ? "visible" : "hidden";
  drawNetwork(null);
}

// ---------------------------------------------------------------- theme

function setTheme(newTheme) {
  document.documentElement.dataset.theme = newTheme;
  savePreference("alexgpt-theme", newTheme);
  recolorIcons();
  if (state.current) document.documentElement.style.setProperty("--accent", modelColor(state.current));
  for (const message of document.querySelectorAll(".message.bot[data-model]"))
    message.style.setProperty("--accent", modelColor(message.dataset.model));
  netColors = null;                // read the new theme's colors on the next draw
  if (state.current) setCosmosModel(state.current);
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
  attention: "<b>Attention.</b> The newest character compares its query with the key of every character before it (dot product), softmax turns the scores into shares, and it collects a weighted mix of their values. Thicker arc = bigger share. ",
  embeddingTr: "<b>Embedding + position.</b> The newest character's vector (64 numbers) when it enters the model: its row from the character table plus the row of its position.",
  block:     "<b>Transformer block.</b> The line on top is the residual stream: the newest character's vector, passed from block to block. Attention collects information from the characters before and adds it to the vector (first +), then the MLP thinks about it and adds its result (second +). The bars show how much each part changed the vector. Click a block to see its attention on top.",
  table:     "<b>Count table.</b> Spark has no neurons: it reads the row of the previous character from its table of counts.",
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
  const chatInfo = state.available[state.current];
  if (chatInfo && chatInfo.layers && chatInfo.layers.chat) {
    renderFlow(step);                       // flow.js: the 3-step view, drawn as HTML
    $("networkLabel").textContent = formatParams(chatInfo.params);
    return;
  }
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

  if (layers.transformer) {
    drawTransformerView(context, step, info, colors, sans, mono);
    $("networkLabel").textContent = formatParams(info.params);
    return;
  }

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
    // Spark: the previous character points straight at the probabilities
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
function drawOutputs(context, step, info, colors, sans, mono, outputY, moreY = NET_BOTTOM - 6, lossY = 384) {
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
  context.fillText("+" + Math.max(0, info.vocab_size - 5) + " more", NET.output + 16, moreY);

  // loss: dashed, because it only exists during training
  context.setLineDash([5, 5]);
  context.strokeStyle = colors.faint;
  context.lineWidth = 1.5;
  context.beginPath(); context.roundRect(NET.output - 50, lossY, 128, 34, 10); context.stroke();
  context.setLineDash([]);
  context.fillStyle = colors.faint;
  context.font = "500 15px " + sans;
  context.fillText("loss · training", NET.output + 14, lossY + 18);
}

// Pulsar: the context as one row of characters; arcs show where the newest character looks
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


// ---------------------------------------------------------------- Pulsar: the whole transformer
// Top: the context, with arcs showing where the newest character looks (one block at a time).
// Bottom: the newest character's vector on its way through the blocks (the "residual stream"):
//   embedding → block 1 (attention adds, MLP adds) → block 2 → block 3 → logits → softmax
let attentionBlock = null;          // which block's attention is drawn on top (null = the last one)

const TV = {
  rowY: 128, left: 24, right: 640,      // the context row
  streamY: 214,                         // the residual stream line
  gridTop: 238,                         // where the dot grids start
  embedX: 30,
  blockLeft: 104, blockWidth: 172,      // block b starts at blockLeft + b * blockWidth
};

// a square-ish grid of colored dots; values are scaled so the biggest one is full color
function dotGrid(context, values, left, top, columns, gap, radius, alreadyScaled) {
  let biggest = 1e-9;
  if (values && !alreadyScaled) for (const value of values) biggest = Math.max(biggest, Math.abs(value));
  const count = values ? values.length : columns * columns;
  for (let index = 0; index < count; index++) {
    const x = left + (index % columns) * gap;
    const y = top + Math.floor(index / columns) * gap;
    const value = values ? (alreadyScaled ? values[index] : values[index] / biggest) : 0;
    circle(context, x, y, radius, neuronColor(value));
  }
}

function vectorSize(values) {
  let total = 0;
  for (const value of values) total += value * value;
  return Math.sqrt(total);
}

function drawTransformerView(context, step, info, colors, sans, mono) {
  const layers = info.layers;
  const contextSize = info.context_size;
  const blockCount = layers.blocks;
  const trace = step && step.trace ? step.trace : null;
  const shownBlock = attentionBlock === null || attentionBlock >= blockCount ? blockCount - 1 : attentionBlock;

  // ---------- top: attention of one block
  context.font = "500 17px " + sans;
  context.fillStyle = colors.faint;
  context.textAlign = "left";
  context.textBaseline = "alphabetic";
  context.fillText("attention · block " + (shownBlock + 1) + " · where the newest character looks", TV.left - 10, 24);
  context.textAlign = "center";

  const gap = (TV.right - TV.left) / (contextSize - 1);
  const weights = trace ? trace.blocks[shownBlock].attention : null;
  let biggest = 1e-9;
  if (weights) for (const value of weights) biggest = Math.max(biggest, value);
  if (weights) {
    context.lineCap = "round";
    context.strokeStyle = colors.accent;
    for (let s = 0; s < contextSize; s++) {
      const share = weights[s] / biggest;
      if (share < 0.03) continue;
      const x = TV.left + s * gap;
      const height = Math.min(88, 14 + (TV.right - x) * 0.15);
      context.globalAlpha = 0.12 + 0.88 * share;
      context.lineWidth = 1 + 6 * share;
      context.beginPath();
      context.moveTo(TV.right, TV.rowY - 12);
      context.quadraticCurveTo((TV.right + x) / 2, TV.rowY - 12 - height, x, TV.rowY - 12);
      context.stroke();
    }
    context.globalAlpha = 1;
  }
  context.textBaseline = "middle";
  for (let s = 0; s < contextSize; s++) {
    const x = TV.left + s * gap;
    const char = step ? step.context[s] : " ";
    circle(context, x, TV.rowY, 9.5, colors.surface);
    if (step && s === contextSize - 1) {
      context.strokeStyle = colors.accent;
      context.lineWidth = 2;
      context.beginPath(); context.arc(x, TV.rowY, 9.5, 0, Math.PI * 2); context.stroke();
    }
    context.fillStyle = char === " " || char === "\n" ? colors.faint : colors.text;
    context.font = "500 12px " + mono;
    context.fillText(visibleChar(char || " "), x, TV.rowY + 1);
    if (weights) {
      context.fillStyle = colors.accent;
      context.globalAlpha = 0.8;
      context.fillRect(x - 3, TV.rowY + 14, 6, 26 * weights[s] / biggest);
      context.globalAlpha = 1;
    }
  }

  // ---------- bottom: the residual stream of the newest character
  const blocksEnd = TV.blockLeft + blockCount * TV.blockWidth;
  context.strokeStyle = colors.line;
  context.lineWidth = 2;
  line(context, TV.embedX, TV.streamY, NET.logits - NET.logitsHalf - 12, TV.streamY);

  // embedding
  dotGrid(context, trace ? trace.embedding : null, TV.embedX, TV.gridTop, 8, 7, 2.6, false);
  context.font = "400 13px " + sans;
  context.fillStyle = colors.faint;
  context.fillText("embedding", TV.embedX + 24, TV.gridTop + 66);

  // how much each part adds to the stream (vector sizes), on one shared scale
  let biggestAdd = 1e-9;
  if (trace) {
    for (const block of trace.blocks) {
      biggestAdd = Math.max(biggestAdd, vectorSize(block.attn_out), vectorSize(block.mlp_out));
    }
  }

  for (let b = 0; b < blockCount; b++) {
    const left = TV.blockLeft + b * TV.blockWidth;
    const block = trace ? trace.blocks[b] : null;
    const attnX = left + 4;
    const mlpX = left + 64;

    // block frame: the shown attention block is highlighted
    context.strokeStyle = b === shownBlock ? colors.accent : colors.line;
    context.lineWidth = b === shownBlock ? 1.6 : 1;
    context.beginPath();
    context.roundRect(left - 8, TV.streamY - 6, TV.blockWidth - 14, 196, 10);
    context.stroke();
    context.fillStyle = b === shownBlock ? colors.text : colors.muted;
    context.font = "500 14px " + sans;
    context.fillText("block " + (b + 1), left + (TV.blockWidth - 30) / 2, TV.streamY - 16 + 0);

    // "+" where attention and the MLP add to the stream
    for (const x of [attnX + 24, mlpX + 41]) {
      circle(context, x, TV.streamY, 7, colors.surface);
      context.strokeStyle = colors.muted; context.lineWidth = 1.2;
      context.beginPath(); context.arc(x, TV.streamY, 7, 0, Math.PI * 2); context.stroke();
      context.fillStyle = colors.text; context.font = "500 12px " + mono;
      context.fillText("+", x, TV.streamY + 1);
    }

    // attention output (what it collected) and MLP neurons
    dotGrid(context, block ? block.attn_out : null, attnX, TV.gridTop, 8, 7, 2.6, false);
    dotGrid(context, block ? block.hidden : null, mlpX, TV.gridTop, 16, 5.4, 2.1, true);
    context.font = "400 13px " + sans;
    context.fillStyle = colors.faint;
    context.fillText("attention", attnX + 24, TV.gridTop + 66);
    context.fillText("MLP · " + layers.hidden, mlpX + 41, TV.gridTop + 96);

    // contribution bars: how much attention / MLP changed the vector
    if (block) {
      const barLeft = attnX - 2;
      const barWidth = 140;
      const parts = [["attn", vectorSize(block.attn_out)], ["mlp", vectorSize(block.mlp_out)]];
      for (let p = 0; p < 2; p++) {
        const y = TV.gridTop + 124 + p * 16;
        context.fillStyle = colors.surface;
        context.fillRect(barLeft + 34, y - 4, barWidth - 34, 8);
        context.fillStyle = colors.accent;
        context.fillRect(barLeft + 34, y - 4, (barWidth - 34) * parts[p][1] / biggestAdd, 8);
        context.textAlign = "left";
        context.fillStyle = colors.muted;
        context.font = "400 12px " + mono;
        context.fillText(parts[p][0], barLeft, y);
        context.textAlign = "center";
      }
    }
  }

  // logits bars + softmax, like Nebula
  const outputY = [];
  for (let k = 0; k < 5; k++) outputY.push(spread(k, 5, TV.gridTop - 6, 366));
  context.font = "500 17px " + sans;
  context.fillStyle = colors.faint;
  context.fillText("logits", NET.logits, TV.streamY - 16);
  const scores = step && step.scores ? step.scores : null;
  if (scores) {
    let biggestScore = 1e-9;
    for (const value of scores) biggestScore = Math.max(biggestScore, Math.abs(value));
    context.strokeStyle = colors.line;
    context.lineWidth = 1;
    line(context, NET.logits, outputY[0] - 12, NET.logits, outputY[4] + 12);
    for (let k = 0; k < scores.length; k++) {
      const length = (scores[k] / biggestScore) * NET.logitsHalf;
      context.fillStyle = scores[k] >= 0 ? "rgba(79,141,255,0.85)" : "rgba(240,122,69,0.85)";
      context.beginPath();
      context.roundRect(Math.min(NET.logits, NET.logits + length), outputY[k] - 6, Math.abs(length), 12, 3);
      context.fill();
      context.font = "400 13px " + mono;
      context.fillStyle = colors.muted;
      context.textAlign = "left";
      context.fillText(scores[k].toFixed(1), NET.logits + NET.logitsHalf + 6, outputY[k] + 1);
      context.textAlign = "center";
      context.strokeStyle = colors.line;
      line(context, NET.logits + NET.logitsHalf + 34, outputY[k], NET.output - 19, outputY[k]);
    }
  }
  drawOutputs(context, step, info, colors, sans, mono, outputY, 392, 404);
  // the "softmax" title for the output column
  context.font = "500 17px " + sans;
  context.fillStyle = colors.faint;
  context.fillText("softmax", NET.output + 22, TV.streamY - 16);
}

// which part of the Pulsar drawing is under the mouse
function transformerPartAt(x, y, info) {
  if (y < TV.rowY + 45) return { name: "attention" };
  if (x > NET.output - 55 && y > 398) return { name: "loss" };
  if (x > NET.output - 40) return { name: "output" };
  if (x > NET.logits - NET.logitsHalf - 20) return { name: "logits" };
  if (x < TV.blockLeft - 12) return { name: "embeddingTr" };
  const block = Math.floor((x - TV.blockLeft + 8) / TV.blockWidth);
  if (block >= 0 && block < info.layers.blocks) return { name: "block", block: block };
  return { name: "idle" };
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
  const info = state.available[state.current];
  if (info && info.layers && info.layers.transformer) {
    showLayerNote(transformerPartAt(event.offsetX * scale, event.offsetY * scale, info).name);
    return;
  }
  showLayerNote(layerAt(event.offsetX * scale, event.offsetY * scale));
});
// Pulsar: click a block to see its attention on top
$("network").addEventListener("click", (event) => {
  const info = state.available[state.current];
  if (!info || !info.layers || !info.layers.transformer) return;
  const scale = NET_W / $("network").clientWidth;
  const part = transformerPartAt(event.offsetX * scale, event.offsetY * scale, info);
  if (part.name === "block") {
    attentionBlock = part.block;
    drawNetwork(state.lastStep);
  }
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
  message.dataset.model = state.current;                  // the reply keeps the color of the model that wrote it
  message.style.setProperty("--accent", modelColor(state.current));

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
  if (isChatModel(state.current)) echo.textContent = "";   // a chat model answers, it does not continue your text
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

// deletes the last `count` characters of a reply that are already on screen
function removeLastCharacters(bot, count) {
  let node = bot.caret.previousSibling;
  while (count > 0 && node && node !== bot.echo) {
    const previous = node.previousSibling;
    const text = node.textContent;
    if (node.classList && node.classList.contains("echo")) break;
    if (text.length <= count) {
      count -= text.length;
      node.remove();
    } else {
      node.textContent = text.slice(0, text.length - count);
      count = 0;
    }
    node = previous;
  }
}

function setGenerating(on) {
  state.generating = on;
  // the model, temperature and length are fixed for a reply; speed can change any time
  $("modelButton").disabled = on;
  $("composer").classList.toggle("writing", on);           // the ring around the message box speeds up
  $("temperature").disabled = on;
  $("length").disabled = on;
  if (on) $("modelPicker").classList.remove("open");
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

// ---------------------------------------------------------------- tools: a status line and a source card
// While a tool runs: "Searching Wikipedia" with a moving shine.  Then: "Searched Wikipedia" (click = what it found).
const TOOL_LABELS = {
  wiki: ["Searching Wikipedia", "Searched Wikipedia", "search"],
  calc: ["Using the calculator", "Used the calculator", "calc"],
  weather: ["Checking the weather", "Checked the weather", "cloud"],
  now: ["Checking the time", "Checked the time", "clock"],
  convert: ["Converting units", "Converted units", "swap"],
  dice: ["Rolling the dice", "Rolled the dice", "die"],
  coin: ["Flipping a coin", "Flipped a coin", "coin"],
  pick: ["Picking one", "Picked one", "list"],
  memory_save: ["Saving to memory", "Saved to memory", "memory"],
  memory_get: ["Checking memory", "Checked memory", "memory"],
  today_in_history: ["Looking up this day in history", "Looked up this day in history", "calendar"],
};
const TOOL_ICONS = {     // small line drawings, 16x16, drawn with the text color
  search: '<circle cx="7" cy="7" r="4.5"/><path d="M10.5 10.5 14 14"/>',
  calc: '<rect x="3" y="1.5" width="10" height="13" rx="2"/><path d="M5.5 4.5h5M5.5 8h1M9.5 8h1M5.5 11h1M9.5 11h1"/>',
  cloud: '<path d="M4.5 12.5h7a3 3 0 0 0 .3-6 4 4 0 0 0-7.6 1A2.5 2.5 0 0 0 4.5 12.5z"/>',
  clock: '<circle cx="8" cy="8" r="6"/><path d="M8 4.5V8l2.5 1.5"/>',
  swap: '<path d="M3 5.5h10M10.5 3 13 5.5 10.5 8M13 10.5H3M5.5 8 3 10.5 5.5 13"/>',
  die: '<rect x="2.5" y="2.5" width="11" height="11" rx="2.5"/><circle cx="5.7" cy="5.7" r=".6"/><circle cx="10.3" cy="10.3" r=".6"/><circle cx="8" cy="8" r=".6"/>',
  coin: '<circle cx="8" cy="8" r="6"/><circle cx="8" cy="8" r="3.5"/>',
  list: '<path d="M6 4.5h7M6 8h7M6 11.5h7M3 4.5h.5M3 8h.5M3 11.5h.5"/>',
  memory: '<path d="M4 2.5h8v11l-4-2.5-4 2.5z"/>',
  calendar: '<rect x="2.5" y="3.5" width="11" height="10" rx="2"/><path d="M2.5 7h11M5.5 2v3M10.5 2v3"/>',
};

function toolIcon(name) {
  return '<svg viewBox="0 0 16 16" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.4" ' +
    'stroke-linecap="round" stroke-linejoin="round">' + (TOOL_ICONS[name] || TOOL_ICONS.search) + "</svg>";
}

function toolWhat(step) {           // "wiki: en: NATO" -> "NATO"
  return step.call.replace(/^[^:]*:\s*/, "").replace(/^(ro|en):\s*/, "").trim();
}

function makeToolStatus(step, running) {
  const labels = TOOL_LABELS[step.tool] || ["Using " + step.tool, "Used " + step.tool, "search"];
  const box = document.createElement("div");
  box.className = "tool-status" + (running ? " running" : "");
  const head = document.createElement("button");
  head.type = "button";
  head.className = "ts-head";
  head.innerHTML = '<span class="ts-icon">' + toolIcon(labels[2]) + '</span><span class="ts-label"></span>' +
    '<span class="ts-what"></span><svg class="ts-chevron" viewBox="0 0 16 16" width="12" height="12" fill="none" ' +
    'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M6 4l4 4-4 4"/></svg>';
  head.querySelector(".ts-label").textContent = running ? labels[0] : labels[1];
  const what = toolWhat(step);
  if (what) head.querySelector(".ts-what").textContent = "“" + (what.length > 48 ? what.slice(0, 47) + "…" : what) + "”";
  const detail = document.createElement("div");
  detail.className = "ts-detail";
  detail.hidden = true;
  head.addEventListener("click", () => {
    if (box.classList.contains("running")) return;
    detail.hidden = !detail.hidden;
    box.classList.toggle("open", !detail.hidden);
  });
  box.append(head, detail);
  box._labels = labels;
  return box;
}

function finishToolStatus(box, step) {
  box.classList.remove("running");
  box.classList.add("done");
  const label = box.querySelector(".ts-label");
  if (step.unused) {
    label.textContent = "Searched Wikipedia · nothing useful found";
    box.classList.add("failed");
  } else if (step.result.startsWith("error")) {
    label.textContent = box._labels[1] + " · didn't work";
    box.classList.add("failed");
  } else {
    label.textContent = box._labels[1];
  }
  const detail = box.querySelector(".ts-detail");
  detail.textContent = "";
  if (step.source) {                                 // Wikipedia: the page as a card + the passage the model read
    detail.classList.add("with-source");
    detail.append(makeSourceCard(step.source));
    const passage = document.createElement("div");
    passage.className = "ts-passage";
    passage.textContent = step.result.replace(/^[^—]*—\s*/, "").replace(/\s*\(https?:\/\/[^)]*\)\s*$/, "");
    detail.append(passage);
  } else {
    detail.textContent = step.result;
  }
}

function makeSourceCard(source) {
  const card = document.createElement("a");
  card.className = "source-card";
  card.href = source.url;
  card.target = "_blank";
  card.rel = "noopener noreferrer";
  let host = "";
  try { host = new URL(source.url).hostname; } catch (error) { host = ""; }
  const icon = document.createElement("span");
  icon.className = "sc-icon";
  const image = document.createElement("img");
  image.alt = "";
  image.src = "https://" + (host || "en.wikipedia.org") + "/static/favicon/wikipedia.ico";   // the site's own favicon
  image.addEventListener("error", () => { image.remove(); icon.textContent = "W"; icon.classList.add("letter"); });
  icon.append(image);
  const words = document.createElement("span");
  words.className = "sc-words";
  const title = document.createElement("span");
  title.className = "sc-title";
  title.textContent = source.title;
  const site = document.createElement("span");
  site.className = "sc-site";
  site.textContent = host || source.site || "";
  words.append(title, site);
  const arrow = document.createElement("span");
  arrow.className = "sc-arrow";
  arrow.innerHTML = '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.6" ' +
    'stroke-linecap="round" stroke-linejoin="round"><path d="M5.5 10.5l5-5M6.5 5.5h4v4"/></svg>';
  card.append(icon, words, arrow);
  return card;
}

function addSourceCards(body, sources) {
  if (!sources || sources.length === 0) return;
  const row = document.createElement("div");
  row.className = "source-row";
  for (const source of sources) row.append(makeSourceCard(source));
  body.append(row);
}

// ---------------------------------------------------------------- markdown
// The model writes markdown ("| a | b |", "**bold**", "- item").  While it writes, the raw text is shown (letter by
// letter, with the confidence colors); when the reply is finished it becomes formatted HTML, and the colors are put
// back on the letters that are still visible.
function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function markdownInline(text) {
  let html = escapeHtml(text);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(^|[^*\w])\*([^*\s][^*]*)\*(?!\w)/g, "$1<em>$2</em>");
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return html;
}

function markdownToHtml(source) {
  const lines = source.replace(/\r/g, "").split("\n");
  const out = [];
  let paragraph = [];
  const flush = () => { if (paragraph.length) { out.push("<p>" + paragraph.map(markdownInline).join("<br>") + "</p>"); paragraph = []; } };
  const cells = (line) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*```/.test(line)) {                                  // ``` code block ```
      flush();
      const code = [];
      i += 1;
      while (i < lines.length && !/^\s*```/.test(lines[i])) { code.push(lines[i]); i += 1; }
      out.push("<pre><code>" + escapeHtml(code.join("\n")) + "</code></pre>");
      i += 1;
      continue;
    }
    if (/^\s*\|.*\|/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {   // a table
      flush();
      const head = cells(line);
      let html = "<div class=\"md-table\"><table><thead><tr>" + head.map((c) => "<th>" + markdownInline(c) + "</th>").join("") + "</tr></thead><tbody>";
      i += 2;
      while (i < lines.length && /^\s*\|/.test(lines[i])) {
        html += "<tr>" + cells(lines[i]).map((c) => "<td>" + markdownInline(c) + "</td>").join("") + "</tr>";
        i += 1;
      }
      out.push(html + "</tbody></table></div>");
      continue;
    }
    const heading = line.match(/^\s*(#{1,4})\s+(.*)$/);
    if (heading) { flush(); out.push("<h" + (heading[1].length + 2) + ">" + markdownInline(heading[2]) + "</h" + (heading[1].length + 2) + ">"); i += 1; continue; }
    if (/^\s*([-*•]|\d+[.)])\s+/.test(line)) {                 // a list
      flush();
      const ordered = /^\s*\d/.test(line);
      const items = [];
      while (i < lines.length && /^\s*([-*•]|\d+[.)])\s+/.test(lines[i])) {
        items.push("<li>" + markdownInline(lines[i].replace(/^\s*([-*•]|\d+[.)])\s+/, "")) + "</li>");
        i += 1;
      }
      out.push((ordered ? "<ol>" : "<ul>") + items.join("") + (ordered ? "</ol>" : "</ul>"));
      continue;
    }
    if (/^\s*(---+|\*\*\*+)\s*$/.test(line)) { flush(); out.push("<hr>"); i += 1; continue; }
    if (line.trim() === "") { flush(); i += 1; continue; }
    paragraph.push(line);
    i += 1;
  }
  flush();
  return out.join("");
}

// pieces = [[text, level], ...] of the reply.  The formatted text is a subsequence of the raw text (markdown only
// removes characters like "|" and "**"), so walking both together gives every visible letter its level again.
function applyMarkdown(container, pieces) {
  const raw = [];
  const levels = [];
  for (const [pieceText, level] of pieces) {
    for (const char of pieceText) { raw.push(char); levels.push(level); }
  }
  container.innerHTML = markdownToHtml(raw.join(""));
  container.classList.add("md");
  let position = 0;
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  while (walker.nextNode()) textNodes.push(walker.currentNode);
  for (const node of textNodes) {
    const fragment = document.createDocumentFragment();
    let runText = "";
    let runLevel = null;
    const close = () => {
      if (!runText) return;
      if (runLevel === "p4" || runLevel === null) fragment.append(document.createTextNode(runText));
      else {
        const span = document.createElement("span");
        span.className = "tok settled " + runLevel;
        span.dataset.level = runLevel;
        span.textContent = runText;
        fragment.append(span);
      }
      runText = "";
    };
    for (const char of node.data) {
      let found = -1;
      for (let look = position; look < Math.min(raw.length, position + 60); look += 1) {
        if (raw[look] === char) { found = look; break; }
      }
      const level = found >= 0 ? levels[found] : "p4";
      if (found >= 0) position = found + 1;
      if (level !== runLevel) { close(); runLevel = level; }
      runText += char;
    }
    close();
    node.replaceWith(fragment);
  }
}

async function send(prompt) {
  if (!state.current) return;
  const chat = ensureChat(prompt);
  const history = [];                              // what was said before in this chat (chat models read it)
  for (const turn of chat.messages) history.push({ role: turn.role, text: turn.text });
  chat.messages.push({ role: "user", text: prompt });
  chat.model = state.current;
  chat.updated = Date.now();
  saveChats();
  renderChatList();
  const empty = $("emptyState");
  if (empty) empty.remove();

  addUserMessage(prompt);
  const bot = addBotMessage(prompt);
  flowReset();
  activeCaret = bot.caret;
  scrollToBottom(true);
  setGenerating(true);

  const controller = new AbortController();
  state.controller = controller;
  stepOnce = () => renderFrame(1);
  resumeDrawing = () => startPump();

  const startTime = performance.now();
  let count = 0;
  let confidenceSum = 0;
  let ending = "";
  const toolSteps = [];          // saved with the reply, so the status lines come back when the chat is opened again
  const sources = [];
  const toolBoxes = {};
  let holdUntil = 0;             // the reply waits a moment, so the "Searching…" animation can be seen

  // The server sends letters as fast as it can; they wait in this queue
  // and are shown at the speed chosen with the Speed slider (which you can move while it writes).
  let pending = [];

  // limit = how many waiting letters to draw (undefined = all of them)
  function renderFrame(limit) {
    if (pending.length === 0) return;

    let take = pending.length;
    if (limit !== undefined && limit < take) take = limit;
    const drawn = pending.splice(0, take);                  // remove them from the front of the queue

    const fragment = document.createDocumentFragment();
    const batch = [];
    let latestModelStep = null;
    for (const step of drawn) {
      if (step.grounded) {                                  // sentences copied from the article, not written by the model
        const span = document.createElement("span");
        span.dataset.level = "p4";
        span.className = "tok new p4 grounded";
        span.textContent = step.char;
        fragment.append(span);
        batch.push(span);
        continue;
      }
      latestModelStep = step;
      flowRecord(step);                                     // flow.js keeps them: ‹ › go back to earlier tokens
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
    emitFromCaret(bot.caret);                               // a little trail of particles behind the new letters
    // once the fade-in is over, glue these letters to the text before them
    setTimeout(() => settleLetters(batch), 500);

    const latest = latestModelStep;

    if (latest === null) {
      scrollToBottom(false);
      return;
    }
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

  // the "pump" takes letters from the queue at the current speed, until the queue is empty
  let pumping = false;
  function startPump() {
    if (pumping) return;
    pumping = true;
    pump();
  }
  function pump() {
    if (state.paused || controller.signal.aborted || pending.length === 0) {
      pumping = false;
      return;
    }
    const wait = holdUntil - performance.now();
    if (wait > 0) {                         // a tool status is still animating: the reply starts after it
      setTimeout(pump, wait);
      return;
    }
    const delay = parseInt($("speed").value);
    if (delay === 0) {
      renderFrame();                        // max speed: everything that arrived, once per frame
      requestAnimationFrame(pump);
    } else {
      renderFrame(1);                       // one letter, then wait `delay` ms
      setTimeout(pump, delay);
    }
  }

  // a letter arrived from the server: put it in the queue
  let held = [];
  function show(step) {
    pending.push(step);
    startPump();
  }

  try {
    const lengthValue = parseInt($("length").value);
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: state.current,
        prompt: prompt,
        history: isChatModel(state.current) ? history : [],
        visualize: panelOpen(),                              // the inside of Galaxy is only sent when you can see it
        temperature: parseFloat($("temperature").value),
        max_chars: lengthValue >= 850 || isChatModel(state.current) ? 0 : lengthValue,   // 0 = auto
        min_tokens: isChatModel(state.current) && lengthValue < 850 ? Math.round(lengthValue / 5) : 0,
        delay_ms: 0,                                         // the page sets the speed, see pump()
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
        } else if (step.type === "tool_start") {
          const box = makeToolStatus(step, true);              // "Searching Wikipedia", with a moving shine
          box._started = performance.now();
          toolBoxes[step.call] = box;
          bot.text.parentNode.insertBefore(box, bot.text);
          holdUntil = Math.max(holdUntil, performance.now() + 900);
          scrollToBottom(false);
        } else if (step.type === "tool") {
          let box = toolBoxes[step.call];
          if (!box) {
            box = makeToolStatus(step, true);
            box._started = performance.now();
            bot.text.parentNode.insertBefore(box, bot.text);
          }
          const shownFor = performance.now() - box._started;
          const finishIn = Math.max(0, 900 - shownFor);       // fast tools (calc) still show their animation briefly
          setTimeout(() => finishToolStatus(box, step), finishIn);
          holdUntil = Math.max(holdUntil, performance.now() + finishIn + 150);
          toolSteps.push({ tool: step.tool, call: step.call, result: step.result, unused: !!step.unused, source: step.source || null });
        } else if (step.type === "source") {
          sources.push({ title: step.title, url: step.url, site: step.site });
        } else if (step.type === "error") {
          ending = "Error: " + step.message;               // e.g. out of GPU memory: the server says what broke
          const note = document.createElement("span");
          note.className = "error-note";
          note.textContent = "⚠ " + step.message;
          bot.text.insertBefore(note, bot.caret);
        } else if (step.type === "end_of_story") {
          held = [];                                       // the marker: throw it away
          if (step.remove) {
            // a damaged marker ("endoftext|>") was already shown or is waiting: take it back
            let toRemove = step.remove;
            while (toRemove > 0 && pending.length > 0) { pending.pop(); toRemove -= 1; }
            removeLastCharacters(bot, toRemove);
            count -= Math.min(count, toRemove);
          }
          ending = isChatModel(state.current) ? "Finished its reply." : "Ended the story on its own.";
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
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  stepOnce = null;
  setPaused(false);

  if ((ending === "Ended the story on its own." || ending === "Finished its reply.") && !controller.signal.aborted) {
    $("chosenChar").textContent = "■";
    $("chosenProb").textContent = "End";
    $("chosenNote").textContent = "the model wrote <|endoftext|>";
  }
  if (controller.signal.aborted) ending = "Stopped by you.";
  if (ending === "") ending = "Reached the length limit.";
  showStats(count, confidenceSum, startTime, ending);

  bot.caret.remove();
  activeCaret = null;
  // save the reply in the chat (without the faded echo of the prompt)
  const errorNotes = [...bot.text.querySelectorAll(".error-note")];
  for (const note of errorNotes) note.remove();          // errors are shown, but not saved as part of the reply
  const echo = bot.text.querySelector(".echo");
  const replyText = bot.text.textContent.slice(echo ? echo.textContent.length : 0);
  // keep the confidence colors: [text, level] pieces (plain text = sure, p4)
  const pieces = [];
  for (const node of bot.text.childNodes) {
    if (node === echo) continue;
    if (node.nodeType === Node.TEXT_NODE) pieces.push([node.data, "p4"]);
    else if (node.dataset && node.dataset.level) pieces.push([node.textContent, node.dataset.level]);
  }
  const savedTurn = { role: "bot", text: replyText, pieces: pieces, model: state.current };
  if (toolSteps.length) savedTurn.tools = toolSteps;
  if (sources.length) savedTurn.sources = sources;
  chat.messages.push(savedTurn);
  applyMarkdown(bot.text, pieces);
  for (const note of errorNotes) bot.text.append(note);
  chat.updated = Date.now();
  saveChats();
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

// suggestion chips (the empty screen can be rebuilt, so listen on the parent)
$("messages").addEventListener("click", (event) => {
  const chip = event.target.closest(".chip");
  if (!chip || state.generating) return;
  $("input").value = chip.textContent;
  autoGrow();
  updateSendButton();
  $("composer").requestSubmit();
});

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
  const show = () => {
    $(id + "Value").textContent = format(parseFloat(slider.value));
    // how far the knob is, in %, so the track left of it can be colored
    const percent = (slider.value - slider.min) / (slider.max - slider.min) * 100;
    slider.style.setProperty("--fill", percent + "%");
  };
  slider.addEventListener("input", show);
  show();
}
bindSlider("speed", (ms) => (ms === 0 ? "max" : ms + " ms"));
bindSlider("temperature", (t) => t.toFixed(2));
// chat models: Auto = the reply ends when the model wants; otherwise = at least n/5 tokens
bindSlider("length", (n) => (n >= 850 ? "Auto" : isChatModel(state.current) ? "≥" + Math.round(n / 5) + " tok" : String(n)));

// brand mark: the Singularity icon, in the text color
const brandIcon = makeIcon("Singularity", 18);
brandIcon.removeAttribute("data-model");
brandIcon.style.setProperty("--icon-color", "var(--text)");
$("brandMark").append(brandIcon);

buildBars();
drawBars([], "");
updateSendButton();
loadModels();

// ================================================================ COSMOS BACKGROUND
// One canvas behind the chat. Every model has its own small universe, drawn faintly
// so it never fights with the text. It pauses when the tab is hidden and stays still
// for people who ask their system for reduced motion.

const cosmos = {
  canvas: null, context: null, width: 0, height: 0, ratio: 1,
  scene: null, model: null, stars: [], last: 0, time: 0,
  reducedMotion: window.matchMedia("(prefers-reduced-motion: reduce)").matches,
};

function hexToRgb(hex) {
  const value = parseInt(hex.replace("#", ""), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}
function rgba(rgb, alpha) {
  return "rgba(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + "," + alpha.toFixed(3) + ")";
}
function cosmosInk() {
  // star color: light dots on dark, dark dots on light
  return theme() === "light" ? [40, 40, 70] : [235, 238, 255];
}
function random(min, max) { return min + Math.random() * (max - min); }

function resizeCosmos() {
  const canvas = cosmos.canvas;
  cosmos.ratio = Math.min(window.devicePixelRatio || 1, 1.5);
  cosmos.width = canvas.clientWidth;
  cosmos.height = canvas.clientHeight;
  canvas.width = Math.round(cosmos.width * cosmos.ratio);
  canvas.height = Math.round(cosmos.height * cosmos.ratio);
  makeStars();
  if (cosmos.scene) cosmos.scene.init();
}

function makeStars() {
  cosmos.stars = [];
  const count = Math.round((cosmos.width * cosmos.height) / 9000);
  for (let i = 0; i < count; i++) {
    cosmos.stars.push({ x: Math.random(), y: Math.random(), size: random(0.4, 1.3),
                        phase: random(0, 6.28), speed: random(0.3, 1.2), base: random(0.15, 0.55) });
  }
}

function drawStars(context, strength) {
  const ink = cosmosInk();
  for (const star of cosmos.stars) {
    const twinkle = 0.6 + 0.4 * Math.sin(cosmos.time * star.speed + star.phase);
    context.fillStyle = rgba(ink, star.base * twinkle * strength);
    context.fillRect(star.x * cosmos.width, star.y * cosmos.height, star.size, star.size);
  }
}

function glow(context, x, y, radius, rgb, alpha) {
  const gradient = context.createRadialGradient(x, y, 0, x, y, radius);
  gradient.addColorStop(0, rgba(rgb, alpha));
  gradient.addColorStop(1, rgba(rgb, 0));
  context.fillStyle = gradient;
  context.fillRect(x - radius, y - radius, radius * 2, radius * 2);
}

// ---------------------------------------------------------------- the six universes
const SCENES = {

  // Spark: warm sparks rising slowly from the bottom, like embers from a fire
  Spark: (color) => {
    let sparks = [];
    return {
      init() {
        sparks = [];
        for (let i = 0; i < 46; i++) sparks.push(newSpark(true));
      },
      draw(context, delta) {
        drawStars(context, 0.5);
        for (const spark of sparks) {
          spark.y -= spark.speed * delta;
          spark.x += Math.sin(cosmos.time * spark.sway + spark.phase) * 0.15;
          const life = spark.y / cosmos.height;
          const flicker = 0.55 + 0.45 * Math.sin(cosmos.time * 7 + spark.phase);
          glow(context, spark.x, spark.y, spark.size * 5, color, 0.16 * flicker * life);
          context.fillStyle = rgba(color, 0.75 * flicker * life);
          context.beginPath(); context.arc(spark.x, spark.y, spark.size, 0, 6.28); context.fill();
          if (spark.y < -10) Object.assign(spark, newSpark(false));
        }
      },
    };
    function newSpark(anywhere) {
      return { x: random(0, cosmos.width), y: anywhere ? random(0, cosmos.height) : cosmos.height + 10,
               speed: random(10, 34), size: random(0.6, 1.8), sway: random(0.4, 1.4), phase: random(0, 6.28) };
    }
  },

  // Nebula: big soft clouds of colored gas drifting past each other
  Nebula: (color) => {
    let clouds = [];
    const second = [color[2], Math.round(color[1] * 0.7), color[0]];   // a second tint, from the same color
    return {
      init() {
        clouds = [];
        const size = Math.max(cosmos.width, cosmos.height);
        for (let i = 0; i < 6; i++) {
          clouds.push({ x: random(0.1, 0.9), y: random(0.05, 0.8), radius: random(0.22, 0.42) * size,
                        speedX: random(0.02, 0.05), speedY: random(0.015, 0.04), phase: random(0, 6.28),
                        tint: i % 2 === 0 ? color : second, alpha: random(0.07, 0.12) });
        }
      },
      draw(context) {
        drawStars(context, 0.8);
        context.globalCompositeOperation = theme() === "light" ? "multiply" : "screen";
        for (const cloud of clouds) {
          const x = (cloud.x + 0.08 * Math.sin(cosmos.time * cloud.speedX + cloud.phase)) * cosmos.width;
          const y = (cloud.y + 0.08 * Math.cos(cosmos.time * cloud.speedY + cloud.phase)) * cosmos.height;
          glow(context, x, y, cloud.radius, cloud.tint, cloud.alpha);
        }
        context.globalCompositeOperation = "source-over";
      },
    };
  },

  // Pulsar: a far-away spinning star; two faint beams turn slowly, a soft ring pulses now and then
  Pulsar: (color) => {
    return {
      init() {},
      draw(context) {
        drawStars(context, 0.8);
        const x = cosmos.width * 0.86, y = cosmos.height * 0.16;
        const angle = cosmos.time * 0.12;                         // one turn every ~50 seconds
        const length = Math.min(cosmos.width, cosmos.height) * 0.55;
        for (const side of [0, Math.PI]) {
          const direction = angle + side;
          const endX = x + Math.cos(direction) * length, endY = y + Math.sin(direction) * length;
          const gradient = context.createLinearGradient(x, y, endX, endY);
          gradient.addColorStop(0, rgba(color, 0.10));
          gradient.addColorStop(1, rgba(color, 0));
          context.fillStyle = gradient;
          context.beginPath();
          context.moveTo(x, y);
          context.lineTo(x + Math.cos(direction - 0.035) * length, y + Math.sin(direction - 0.035) * length);
          context.lineTo(x + Math.cos(direction + 0.035) * length, y + Math.sin(direction + 0.035) * length);
          context.closePath();
          context.fill();
        }
        const progress = (cosmos.time / 4) % 1;                   // one soft ring every 4 seconds
        context.strokeStyle = rgba(color, 0.12 * (1 - progress));
        context.lineWidth = 1;
        context.beginPath(); context.arc(x, y, 6 + progress * 90, 0, 6.28); context.stroke();
        const beat = 0.8 + 0.2 * Math.sin(cosmos.time * 2.5);
        glow(context, x, y, 26, color, 0.22 * beat);
        context.fillStyle = rgba(theme() === "light" ? color : [255, 255, 255], 0.8);
        context.beginPath(); context.arc(x, y, 1.8, 0, 6.28); context.fill();
      },
    };
  },

  // Galaxy: thousands of stars on two spiral arms, turning slowly (inner stars faster)
  Galaxy: (color) => {
    let particles = [];
    return {
      init() {
        particles = [];
        const scale = Math.min(cosmos.width, cosmos.height) * 0.55;
        for (let i = 0; i < 1400; i++) {
          const arm = i % 2;
          const distance = Math.pow(Math.random(), 0.65);
          particles.push({ distance: distance * scale,
                           angle: arm * Math.PI + distance * 5.6 + random(-0.4, 0.4) * (1.25 - distance),
                           size: random(0.6, 1.9), alpha: random(0.35, 1) * (1.15 - distance * 0.8),
                           white: Math.random() < 0.25 });            // some stars are white, most take the model color
        }
      },
      draw(context, delta) {
        drawStars(context, 0.6);
        const x = cosmos.width * 0.78, y = cosmos.height * 0.26;
        glow(context, x, y, Math.min(cosmos.width, cosmos.height) * 0.3, color, 0.22);
        const ink = cosmosInk();
        for (const particle of particles) {
          particle.angle += delta * 0.05 / (0.25 + particle.distance / 300);
          const px = x + Math.cos(particle.angle) * particle.distance;
          const py = y + Math.sin(particle.angle) * particle.distance * 0.5;    // tilted disc
          context.fillStyle = rgba(particle.white ? ink : color, particle.alpha * 0.8);
          context.fillRect(px, py, particle.size, particle.size);
        }
        glow(context, x, y, 30, theme() === "light" ? color : [255, 255, 255], 0.35);
      },
    };
  },

  // Supernova: a star explodes again and again; a shock wave and debris fly outwards
  Supernova: (color) => {
    let debris = [];
    let lastBlast = -10;
    return {
      init() { debris = []; lastBlast = -10; },
      draw(context, delta) {
        drawStars(context, 0.7);
        const x = cosmos.width * 0.8, y = cosmos.height * 0.24;
        const since = cosmos.time - lastBlast;
        if (since > 6) {
          lastBlast = cosmos.time;
          for (let i = 0; i < 70; i++) {
            const angle = random(0, 6.28);
            debris.push({ x, y, speedX: Math.cos(angle) * random(25, 130), speedY: Math.sin(angle) * random(25, 130),
                          life: 1, size: random(0.8, 2.6) });
          }
        }
        const flash = Math.max(0, 1 - since / 0.8);
        glow(context, x, y, 70 + since * 45, color, 0.12 * Math.max(0, 1 - since / 6));   // the glowing remnant grows
        glow(context, x, y, 60 + 160 * flash, color, 0.2 + 0.45 * flash);
        for (let wave = 0; wave < 2; wave++) {
          const radius = since * (110 - wave * 35);
          const fade = Math.max(0, 1 - since / 6);
          context.strokeStyle = rgba(color, 0.3 * fade);
          context.lineWidth = 2 - wave;
          context.beginPath(); context.arc(x, y, radius, 0, 6.28); context.stroke();
        }
        for (const piece of debris) {
          piece.x += piece.speedX * delta; piece.y += piece.speedY * delta;
          piece.life -= delta / 6;
          context.fillStyle = rgba(color, Math.max(0, piece.life) * 0.95);
          context.fillRect(piece.x, piece.y, piece.size, piece.size);
        }
        debris = debris.filter((piece) => piece.life > 0);
        context.fillStyle = rgba([255, 255, 255], 0.85);
        context.beginPath(); context.arc(x, y, 2.4, 0, 6.28); context.fill();
      },
    };
  },

  // Nova: a white dwarf pulls a stream of gas from its companion star; every few seconds the gas flares up
  Nova: (color) => {
    let gas = [];
    const warm = [255, 176, 130];
    return {
      init() { gas = []; },
      draw(context, delta) {
        drawStars(context, 0.75);
        const x = cosmos.width * 0.78, y = cosmos.height * 0.3;
        const angle = cosmos.time * 0.22;
        const companionX = x + Math.cos(angle) * 95, companionY = y + Math.sin(angle) * 36;
        glow(context, companionX, companionY, 40, warm, 0.3);
        context.fillStyle = rgba(warm, 0.9);
        context.beginPath(); context.arc(companionX, companionY, 5, 0, 6.28); context.fill();
        if (Math.random() < delta * 45) gas.push({ progress: 0, bend: random(-14, 14), size: random(0.8, 1.8) });
        for (const piece of gas) {
          piece.progress += delta * 0.42;
          const t = piece.progress;
          // a curved path from the companion to the dwarf, bending around it
          const middleX = (companionX + x) / 2 + piece.bend - Math.sin(angle) * 30;
          const middleY = (companionY + y) / 2 + piece.bend * 0.5 + Math.cos(angle) * 30;
          const px = (1 - t) * (1 - t) * companionX + 2 * (1 - t) * t * middleX + t * t * x;
          const py = (1 - t) * (1 - t) * companionY + 2 * (1 - t) * t * middleY + t * t * y;
          context.fillStyle = rgba(t < 0.5 ? warm : color, 0.75 * Math.sin(Math.PI * Math.min(1, t)));
          context.fillRect(px, py, piece.size, piece.size);
        }
        gas = gas.filter((piece) => piece.progress < 1);
        const cycle = cosmos.time % 9;
        const flare = cycle < 1.4 ? Math.sin(cycle / 1.4 * Math.PI) : 0;    // the gas ignites: a short, bright flash
        glow(context, x, y, 26 + 130 * flare, color, 0.2 + 0.5 * flare);
        context.fillStyle = rgba([255, 255, 255], 0.9);
        context.beginPath(); context.arc(x, y, 2.2, 0, 6.28); context.fill();
      },
    };
  },

  // Singularity: a black hole; hot matter spirals in along a glowing disc and disappears
  Singularity: (color) => {
    let matter = [];
    const hot = [255, 196, 132];                          // the disc glows warm, like hot gas
    return {
      init() {
        matter = [];
        for (let i = 0; i < 420; i++) matter.push(newMatter(true));
      },
      draw(context, delta) {
        drawStars(context, 0.7);
        const x = cosmos.width * 0.78, y = cosmos.height * 0.28;
        const hole = Math.min(cosmos.width, cosmos.height) * 0.05;
        glow(context, x, y, hole * 7, hot, 0.13);
        glow(context, x, y, hole * 4, color, 0.12);
        // move every particle: faster and closer as it falls in
        for (const particle of matter) {
          particle.angle += delta * 1.4 / Math.sqrt(particle.distance / hole);
          particle.distance -= delta * 18 * (hole / particle.distance);
          if (particle.distance < hole * 1.1) Object.assign(particle, newMatter(false));
        }
        // back half of the disc, then the hole, then the front half (so the disc passes in front of it)
        drawDisc(context, x, y, hole, false);
        context.fillStyle = theme() === "light" ? "rgba(18,14,30,0.92)" : "rgba(0,0,0,1)";
        context.beginPath(); context.arc(x, y, hole, 0, 6.28); context.fill();
        context.strokeStyle = rgba(hot, 0.7 + 0.2 * Math.sin(cosmos.time * 2));
        context.lineWidth = 1.6;
        context.shadowColor = rgba(hot, 0.9);
        context.shadowBlur = 14;
        context.beginPath(); context.arc(x, y, hole * 1.06, 0, 6.28); context.stroke();
        context.shadowBlur = 0;
        drawDisc(context, x, y, hole, true);
      },
    };
    function drawDisc(context, x, y, hole, front) {
      for (const particle of matter) {
        const inFront = Math.sin(particle.angle) > 0;
        if (inFront !== front) continue;
        const px = x + Math.cos(particle.angle) * particle.distance;
        const py = y + Math.sin(particle.angle) * particle.distance * 0.28;
        const heat = Math.min(1, (hole * 2.2) / particle.distance);
        context.fillStyle = rgba(heat > 0.5 ? hot : color, (0.3 + 0.65 * heat) * particle.alpha);
        context.fillRect(px, py, particle.size, particle.size);
      }
    }
    function newMatter(anywhere) {
      const hole = Math.min(cosmos.width, cosmos.height) * 0.05;
      return { distance: anywhere ? random(hole * 1.2, hole * 8) : random(hole * 5, hole * 8),
               angle: random(0, 6.28), size: random(0.7, 1.9), alpha: random(0.5, 1) };
    }
  },
};

// ---------------------------------------------------------------- particles from the writing cursor
// Every model leaves a different little trail behind the letters it writes.
// Kept small and short-lived, and emitted at most every `every` seconds, so at high speed
// they stay a small glow at the cursor instead of a stripe over the whole line.
const TRAILS = {
  Spark:       { every: 0.07, speedX: [-6, 6],  speedY: [-24, -10], life: 0.8,  size: [1.0, 1.9], gravity: 0 },     // embers rise
  Nebula:      { every: 0.09, speedX: [-6, 6],  speedY: [-6, 6],    life: 1.0,  size: [2.2, 3.4], gravity: 0, soft: true }, // soft puffs
  Pulsar:      { every: 0.18, speedX: [0, 0],   speedY: [0, 0],     life: 0.8,  size: [1, 1],     gravity: 0, ring: true }, // a small ring
  Galaxy:      { every: 0.09, speedX: [-9, 9],  speedY: [8, 22],    life: 1.0,  size: [1.6, 2.6], gravity: 14, star: true }, // little falling stars
  Nova:        { every: 0.11, speedX: [-10, 10], speedY: [-10, 10], life: 0.8, size: [1.6, 2.6], gravity: 0, soft: true }, // warm glow puffs
  Supernova:   { every: 0.22, count: 6, speedX: [-34, 34], speedY: [-34, 34], life: 0.6, size: [1.1, 2.0], gravity: 0 }, // tiny bursts
  Singularity: { every: 0.08, speedX: [0, 0],   speedY: [0, 0],     life: 0.75, size: [0.8, 1.4], gravity: 0, orbit: true }, // pulled in
};
const sparks = [];
let lastSparkTime = 0;
let activeCaret = null;              // the cursor of the reply being written: it sparkles even between tokens

function emitFromCaret(caret) {
  if (cosmos.reducedMotion || !caret || !caret.isConnected || !cosmos.model) return;
  const trail = TRAILS[cosmos.model] || TRAILS[String(cosmos.model).split(" ")[0]] || TRAILS.Spark;
  const now = performance.now() / 1000;
  if (now - lastSparkTime < trail.every) return;
  lastSparkTime = now;
  const box = caret.getBoundingClientRect();
  if (box.width === 0) return;
  for (let i = 0; i < (trail.count || 1); i++) {
    if (sparks.length > 32) sparks.shift();
    sparks.push({
      x: box.left + box.width / 2, y: box.top + box.height / 2,
      speedX: random(trail.speedX[0], trail.speedX[1]), speedY: random(trail.speedY[0], trail.speedY[1]),
      age: 0, life: trail.life * random(0.7, 1.2), size: random(trail.size[0], trail.size[1]),
      angle: random(0, 6.28), radius: random(6, 11), trail: trail,
    });
  }
}

function drawSparks(context, delta) {
  if (sparks.length === 0) return;
  const color = hexToRgb(modelColor(cosmos.model || "Spark"));
  for (let i = sparks.length - 1; i >= 0; i--) {
    const spark = sparks[i];
    spark.age += delta;
    if (spark.age > spark.life) { sparks.splice(i, 1); continue; }
    const fade = 1 - spark.age / spark.life;
    const trail = spark.trail;
    if (trail.ring) {
      context.strokeStyle = rgba(color, 0.6 * fade);
      context.lineWidth = 1.2;
      context.beginPath(); context.arc(spark.x, spark.y, 2 + (1 - fade) * 10, 0, 6.28); context.stroke();
      continue;
    }
    let x, y;
    if (trail.orbit) {
      spark.angle += delta * 6;
      const radius = spark.radius * fade;                    // spirals into the cursor
      x = spark.x + Math.cos(spark.angle) * radius;
      y = spark.y + Math.sin(spark.angle) * radius * 0.5;
    } else {
      spark.speedY += trail.gravity * delta;
      spark.x += spark.speedX * delta; spark.y += spark.speedY * delta;
      x = spark.x; y = spark.y;
    }
    if (trail.soft) {
      glow(context, x, y, spark.size * 3.5, color, 0.8 * fade);
      context.fillStyle = rgba(color, 0.75 * fade);             // a small bright core so it shows on the purple sky
      context.beginPath(); context.arc(x, y, spark.size * 0.45, 0, 6.28); context.fill();
    } else if (trail.star) {
      context.strokeStyle = rgba(color, 0.85 * fade);
      context.lineWidth = 1;
      const arm = spark.size * 1.4;
      context.beginPath(); context.moveTo(x - arm, y); context.lineTo(x + arm, y); context.moveTo(x, y - arm); context.lineTo(x, y + arm); context.stroke();
    } else {
      context.fillStyle = rgba(color, 0.9 * fade);
      context.beginPath(); context.arc(x, y, spark.size, 0, 6.28); context.fill();
    }
  }
}

function setCosmosModel(name) {
  if (!cosmos.canvas) return;
  const makeScene = SCENES[name] || SCENES[name.split(" ")[0]] || SCENES.Spark;
  const color = hexToRgb(modelColor(name));
  // fade out, switch universe, fade in
  cosmos.canvas.classList.add("switching");
  setTimeout(() => {
    cosmos.model = name;
    cosmos.scene = makeScene(color);
    cosmos.scene.init();
    if (cosmos.reducedMotion) drawCosmosFrame(0);
    cosmos.canvas.classList.remove("switching");
  }, cosmos.scene ? 350 : 0);
}

function drawCosmosFrame(delta) {
  const context = cosmos.context;
  context.setTransform(cosmos.ratio, 0, 0, cosmos.ratio, 0, 0);
  context.clearRect(0, 0, cosmos.width, cosmos.height);
  if (cosmos.scene) cosmos.scene.draw(context, delta);
  if (activeCaret) emitFromCaret(activeCaret);
  drawSparks(context, delta);
}

function cosmosLoop(now) {
  const delta = Math.min(0.05, (now - cosmos.last) / 1000);
  cosmos.last = now;
  if (!document.hidden && !cosmos.reducedMotion) {
    cosmos.time += delta;
    drawCosmosFrame(delta);
  }
  requestAnimationFrame(cosmosLoop);
}

function startCosmos() {
  cosmos.canvas = $("cosmos");
  cosmos.context = cosmos.canvas.getContext("2d");
  resizeCosmos();
  window.addEventListener("resize", resizeCosmos);
  cosmos.last = performance.now();
  requestAnimationFrame(cosmosLoop);
}
startCosmos();


// ---------------------------------------------------------------- saved chats
// Every chat is kept in this browser (localStorage), so it is still there after closing the page.
// chat = { id, title, updated, messages: [{ role: "user" | "bot", text, model }] }
const CHATS_KEY = "alexgpt-chats";
const EMPTY_STATE_HTML = $("emptyState").outerHTML;   // to rebuild the welcome screen for a new chat
let chats = [];
let currentChatId = null;

function loadChats() {
  try { chats = JSON.parse(localStorage.getItem(CHATS_KEY)) || []; } catch (e) { chats = []; }
  // once: the base model is now called Nova, and "Supernova Chat" is now just Supernova
  if (loadPreference("alexgpt-nova-renamed") !== "yes") {
    const renamed = { "Supernova": "Nova", "Supernova Chat": "Supernova" };
    for (const chat of chats) {
      if (renamed[chat.model]) chat.model = renamed[chat.model];
      for (const turn of chat.messages || []) if (renamed[turn.model]) turn.model = renamed[turn.model];
    }
    if (renamed[loadPreference("alexgpt-model")]) savePreference("alexgpt-model", renamed[loadPreference("alexgpt-model")]);
    saveChats();
    savePreference("alexgpt-nova-renamed", "yes");
  }
}
function saveChats() {
  try { localStorage.setItem(CHATS_KEY, JSON.stringify(chats)); } catch (e) { /* storage full or blocked */ }
  savePreference("alexgpt-current-chat", currentChatId || "");
}
function findChat(id) {
  for (const chat of chats) if (chat.id === id) return chat;
  return null;
}

// the chat we are writing in; a new one starts with the first message
function ensureChat(firstMessage) {
  let chat = findChat(currentChatId);
  if (!chat) {
    chat = { id: Date.now().toString(36), title: firstMessage.slice(0, 48), updated: Date.now(), messages: [] };
    chats.push(chat);
    currentChatId = chat.id;
  }
  return chat;
}

function renderChatList() {
  const list = $("chatList");
  list.innerHTML = "";
  const sorted = chats.slice().sort((a, b) => b.updated - a.updated);
  if (sorted.length === 0) {
    const note = document.createElement("div");
    note.className = "empty-note";
    note.textContent = "No chats yet";
    list.append(note);
    return;
  }
  for (const chat of sorted) {
    const row = document.createElement("div");           // a div, not a button: typing a space while renaming must not click it
    row.className = "chat-row" + (chat.id === currentChatId ? " active" : "");
    row.setAttribute("role", "button");
    row.tabIndex = 0;
    const title = document.createElement("span");
    title.className = "chat-title";
    title.textContent = chat.title || "Untitled";
    const remove = document.createElement("span");
    remove.className = "chat-delete";
    remove.title = "Delete chat";
    remove.innerHTML = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 3l6 6M9 3l-6 6" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      if (state.generating) return;
      chats = chats.filter((other) => other.id !== chat.id);
      if (chat.id === currentChatId) openChat(null);
      saveChats();
      renderChatList();
    });
    const rename = document.createElement("span");
    rename.className = "chat-delete";
    rename.title = "Rename chat";
    rename.innerHTML = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 10l.6-2.4L8.2 2a1 1 0 0 1 1.4 0l.4.4a1 1 0 0 1 0 1.4L4.4 9.4Z" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"/></svg>';
    rename.addEventListener("click", (event) => { event.stopPropagation(); startRename(chat, title); });
    title.addEventListener("dblclick", (event) => { event.stopPropagation(); startRename(chat, title); });
    row.append(title, rename, remove);
    row.addEventListener("click", () => { if (!state.generating) openChat(chat.id); });
    list.append(row);
  }
}

// a saved reply, drawn without animation, in the color of the model that wrote it
function addStoredBotMessage(turn) {
  const model = turn.model || state.current;
  const message = document.createElement("div");
  message.className = "message bot done";
  message.dataset.model = model;
  message.style.setProperty("--accent", modelColor(model));
  const avatar = document.createElement("div");
  avatar.className = "bot-avatar";
  avatar.append(makeIcon(model, 16));
  const body = document.createElement("div");
  body.className = "bot-body";
  const meta = document.createElement("div");
  meta.className = "bot-meta";
  meta.textContent = model;
  const text = document.createElement("div");
  text.className = "bot-text";
  if (turn.pieces) {
    for (const [pieceText, level] of turn.pieces) {
      if (level === "p4") { text.append(document.createTextNode(pieceText)); continue; }
      const span = document.createElement("span");
      span.className = "tok settled " + level;
      span.dataset.level = level;
      span.textContent = pieceText;
      text.append(span);
    }
  } else {
    text.textContent = turn.text;
  }
  body.append(meta);
  for (const step of turn.tools || []) {
    const box = makeToolStatus(step, false);
    finishToolStatus(box, step);
    body.append(box);
  }
  body.append(text);
  applyMarkdown(text, turn.pieces || [[turn.text, "p4"]]);
  message.append(avatar, body);
  $("messages").append(message);
}

// id = null opens an empty new chat
function openChat(id) {
  currentChatId = id;
  const box = $("messages");
  box.innerHTML = "";
  const chat = findChat(id);
  // switch to the model this chat was using
  if (chat && chat.model && state.available[chat.model] && chat.model !== state.current) selectModel(chat.model);
  if (!chat) {
    box.innerHTML = EMPTY_STATE_HTML;
    if (state.current) $("emptyIcon").append(makeIcon(state.current, 46));
  } else {
    for (const turn of chat.messages) {
      if (turn.role === "user") addUserMessage(turn.text);
      else addStoredBotMessage(turn);
    }
    scrollToBottom(true);
  }
  savePreference("alexgpt-current-chat", currentChatId || "");
  renderChatList();
}

$("newChatButton").addEventListener("click", () => { if (!state.generating) openChat(null); });
$("chatsToggle").addEventListener("click", () => {
  const open = $("chatsPanel").classList.toggle("open");
  $("chatsToggle").classList.toggle("active", open);
  savePreference("alexgpt-chats-panel", open ? "open" : "closed");
});
if (loadPreference("alexgpt-chats-panel") === "open") {
  $("chatsPanel").classList.add("open");
  $("chatsToggle").classList.add("active");
}

loadChats();
const lastChat = loadPreference("alexgpt-current-chat");
if (lastChat && findChat(lastChat)) openChat(lastChat);
else renderChatList();

// rename in place: the title becomes a text box; Enter or clicking away saves, Escape cancels
function startRename(chat, titleElement) {
  const input = document.createElement("input");
  input.className = "chat-rename";
  input.value = chat.title || "";
  titleElement.replaceWith(input);
  input.focus();
  input.select();
  let done = false;
  function finish(save) {
    if (done) return;
    done = true;
    const newTitle = input.value.trim();
    if (save && newTitle) { chat.title = newTitle; saveChats(); }
    renderChatList();
  }
  input.addEventListener("click", (event) => event.stopPropagation());
  input.addEventListener("keydown", (event) => {
    event.stopPropagation();
    if (event.key === "Enter") finish(true);
    if (event.key === "Escape") finish(false);
  });
  input.addEventListener("blur", () => finish(true));
}
