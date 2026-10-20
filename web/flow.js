// Galaxy / Supernova Visualisation: what happens to the newest token, in 3 steps.
//   1. READ    the tokens it sees, and the empty slot "?" it has to fill
//   2. BLOCKS  the newest token's vector goes down through every block:
//              attention (reads the other tokens) -> MLP (thinks alone) -> a guess "so far" (logit lens)
//   3. OUTPUT  final scores (logits) -> softmax -> temperature -> the dice picks one
// While it writes, the rows light up one after another (top -> bottom), like the vector moving down.
// ‹ › go back to earlier tokens of the reply; "Live" follows the newest one again.

const FLOW = {
  steps: [],        // every step of the current reply that has the inside data
  index: -1,        // -1 = live (the newest), otherwise the step shown
  block: -1,        // the block opened below its row (-1 = the last block)
  head: -1,         // -1 = all heads averaged, otherwise one head
  latest: null,
  frame: 0,
  lastDrawn: null,
};

function flowReset() {
  FLOW.steps = [];
  FLOW.index = -1;
  FLOW.latest = null;
  FLOW.lastDrawn = null;
}

function flowRecord(step) {
  if (!step || !step.galaxy) return;
  FLOW.steps.push(step);
  if (FLOW.steps.length > 600) FLOW.steps.shift();
}

// called for every new token: draws at most once per screen frame (no lag at high speed)
function renderFlow(step) {
  if (step) FLOW.latest = step;
  if (FLOW.frame) return;
  FLOW.frame = requestAnimationFrame(() => { FLOW.frame = 0; drawFlow(); });
}

function flowShown() {
  if (FLOW.index >= 0 && FLOW.steps[FLOW.index]) return FLOW.steps[FLOW.index];
  return FLOW.latest;
}

function escapeHtml(text) {
  return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function tokenLabel(piece, max) {
  let text = visibleChar(piece);
  if (max && text.length > max) text = text.slice(0, max - 1) + "…";
  return escapeHtml(text);
}

function percent(value) {
  if (value >= 0.995) return "100%";
  if (value < 0.01) return value > 0 ? "<1%" : "0%";
  return Math.round(value * 100) + "%";
}

function accentRgb() {
  const value = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  const hex = value.startsWith("#") ? value : "#a9a5ff";
  return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
}

// heads averaged (or one head), for the opened block
function blockWeights(block, head) {
  const rows = head >= 0 && head < block.heads.length ? [block.heads[head]] : block.heads;
  const weights = [];
  for (let c = 0; c < rows[0].length; c++) {
    let sum = 0;
    for (const row of rows) sum += row[c];
    weights.push(sum / rows.length);
  }
  return weights;
}

function drawFlow() {
  const box = $("flow");
  if (!box || box.offsetParent === null) return;          // panel closed or not a chat model
  const step = flowShown();
  const info = state.available[state.current] || {};
  const blockCount = info.layers && info.layers.blocks ? info.layers.blocks : 8;

  if (!step || !step.galaxy) {
    box.innerHTML =
      '<div class="fl-empty"><b>Send a message to watch it think.</b><br>' +
      "For every token you will see 3 steps: what it reads, what each of its " + blockCount +
      " blocks does, and how it picks the next token.</div>";
    return;
  }
  const data = step.galaxy;
  const blocks = data.blocks;
  const live = FLOW.index < 0;
  const position = live ? FLOW.steps.length : FLOW.index + 1;
  const picked = data.picked ? data.picked[0] : step.char;
  const selected = FLOW.block < 0 || FLOW.block >= blocks.length ? blocks.length - 1 : FLOW.block;

  // the first block after which the guess is already the token that got picked (and stays so)
  let decided = -1;
  for (let b = blocks.length - 1; b >= 0; b--) {
    if (blocks[b].lens[0][0] === picked) decided = b; else break;
  }
  let biggestChange = 0.0001;
  for (const block of blocks) biggestChange = Math.max(biggestChange, block.attn, block.mlp);

  let html = "";
  // ---------------- navigation
  html += '<div class="fl-nav">' +
    '<button class="fl-btn" data-act="prev" title="Earlier token" ' + (position <= 1 ? "disabled" : "") + ">‹</button>" +
    '<span class="fl-pos">token <b>' + position + "</b> of " + FLOW.steps.length + "</span>" +
    '<button class="fl-btn" data-act="next" title="Later token" ' + (live ? "disabled" : "") + ">›</button>" +
    '<button class="fl-live' + (live ? " on" : "") + '" data-act="live" title="Follow the newest token">' +
    (live ? "● Live" : "Go live") + "</button></div>";

  // ---------------- 1. read
  html += '<div class="fl-stage" style="--i:0">' +
    '<div class="fl-head"><span class="fl-num">1</span><b>Read</b>' +
    '<span class="fl-sub">the last ' + data.tokens.length + " of the " + (info.context_size || "") + " tokens it can see</span></div>" +
    '<div class="fl-tokens">';
  for (let c = 0; c < data.tokens.length; c++) {
    const newest = c === data.tokens.length - 1;
    html += '<span class="fl-tok' + (newest ? " newest" : "") + '" data-tip="' +
      (newest ? "The newest token. Its vector is the one that travels through the blocks." : "Token " + (c + 1)) +
      '">' + tokenLabel(data.tokens[c], 9) + "</span>";
  }
  html += '<span class="fl-tok slot" data-tip="The token it has to guess. It fills in at the end of step 3.">' +
    '<span class="q">?</span><span class="answer">' + tokenLabel(picked, 9) + "</span></span></div>" +
    '<div class="fl-cap">Its only job: guess what goes in <b>?</b> Everything below is for this one guess.</div></div>';

  // ---------------- 2. blocks
  html += '<div class="fl-stage">' +
    '<div class="fl-head"><span class="fl-num">2</span><b>' + blocks.length + " blocks, top to bottom</b>" +
    '<span class="fl-sub">click a block to open it</span></div>' +
    '<div class="fl-cols"><span>#</span>' +
    '<span data-tip="<b>Attention map.</b> One line per head, one column per token (the same 12 as above). Bright = that head reads that token a lot. Every head looks for something else.">Attention</span>' +
    '<span data-tip="<b>How much it changed.</b> A = what attention added to the vector, M = what the MLP added, compared with the vector\'s size before the block.">Change</span>' +
    '<span data-tip="<b>Guess so far (logit lens).</b> We ask the vector: if you stopped after this block, which token would you say? Big = its top guess, small = guesses 2 and 3.">Guess so far</span></div>';
  for (let b = 0; b < blocks.length; b++) {
    const block = blocks[b];
    const [guess, chance] = block.lens[0];
    const open = b === selected;
    html += '<div class="fl-row' + (open ? " open" : "") + (b === decided ? " decided" : "") +
      '" data-block="' + b + '" style="--i:' + (b + 1) + '">' +
      '<span class="fl-bn">' + (b + 1) + "</span>" +
      '<canvas class="fl-heat" data-block="' + b + '"></canvas>' +
      '<span class="fl-change" data-tip="Block ' + (b + 1) + ": attention changed the vector by " +
      Math.round(block.attn * 100) + "%, the MLP by " + Math.round(block.mlp * 100) + '%.">' +
      '<i class="a" style="width:' + Math.max(4, 100 * Math.sqrt(block.attn / biggestChange)) + '%"></i>' +
      '<i class="m" style="width:' + Math.max(4, 100 * Math.sqrt(block.mlp / biggestChange)) + '%"></i></span>' +
      '<span class="fl-guess" data-tip="If it stopped after block ' + (b + 1) + ", it would say <b>" +
      tokenLabel(guess) + "</b> (" + percent(chance) + "). Then " + tokenLabel(block.lens[1][0]) + " (" +
      percent(block.lens[1][1]) + "), " + tokenLabel(block.lens[2][0]) + " (" + percent(block.lens[2][1]) + ').">' +
      '<b class="' + (guess === picked ? "match" : "") + '">' + tokenLabel(guess, 10) + "</b>" +
      '<span class="fl-pct">' + percent(chance) + "</span>" +
      '<span class="fl-alt">' + tokenLabel(block.lens[1][0], 7) + " · " + tokenLabel(block.lens[2][0], 7) + "</span>" +
      (b === decided ? '<span class="fl-badge" data-tip="From here on, every block already guesses the token that got picked.">decided</span>' : "") +
      "</span></div>";
    if (open) html += flowDetail(block, b, data);
  }
  html += "</div>";

  // ---------------- 3. output
  const output = data.output || [];
  html += '<div class="fl-stage" style="--i:' + (blocks.length + 1) + '">' +
    '<div class="fl-head"><span class="fl-num">3</span><b>Pick a token</b>' +
    '<span class="fl-sub">scores → chances → dice</span></div>' +
    '<div class="fl-out-head"><span>Token</span>' +
    '<span data-tip="<b>Score (logit).</b> The final vector compared with every token\'s vector (dot product). Higher = fits better. +1 point = about 2.7× more likely.">Score</span>' +
    '<span data-tip="<b>Softmax.</b> exp(score) divided by the total: the scores become chances that add up to 100%.">Chance</span>' +
    '<span data-tip="<b>After temperature ' + (data.temperature || "") + ' and the repeat penalty.</b> Low temperature makes the leader stronger. Tokens it already wrote get less. The dice uses these.">Final</span></div>';
  let pickedShown = false;
  for (const [piece, score, chance, final] of output) {
    const isPicked = piece === picked && !pickedShown;
    if (isPicked) pickedShown = true;
    html += flowOutputRow(piece, score, chance, final, isPicked);
  }
  if (!pickedShown && data.picked) {
    html += '<div class="fl-out-gap">…</div>' + flowOutputRow(...data.picked, true);
  }
  const pickedFinal = data.picked ? data.picked[3] : step.prob;
  html += '<div class="fl-cap">The dice picked <b class="fl-picked">' + tokenLabel(picked) + "</b> (" + percent(pickedFinal) +
    " chance). " + (pickedFinal < 0.3 ? "Not its top guess: the dice keeps the text from being boring." : "") + "</div></div>";

  const animate = live && FLOW.lastDrawn !== step && parseInt($("speed").value) >= 60;
  box.innerHTML = html;
  box.classList.toggle("sweep", animate);
  if (animate) {
    const stagger = Math.min(45, parseInt($("speed").value) / (blocks.length + 3));
    box.style.setProperty("--stagger", stagger + "ms");
    box.style.setProperty("--rows", blocks.length + 1);
  }
  FLOW.lastDrawn = step;

  // attention maps: one tiny canvas per block (head = row, token = column)
  const rgb = accentRgb();
  for (const canvas of box.querySelectorAll(".fl-heat")) {
    const block = blocks[parseInt(canvas.dataset.block)];
    drawHeat(canvas, block.heads, rgb);
  }
}

function flowOutputRow(piece, score, chance, final, isPicked) {
  return '<div class="fl-out' + (isPicked ? " picked" : "") + '">' +
    '<span class="fl-out-tok">' + tokenLabel(piece, 12) + "</span>" +
    '<span class="fl-num-cell">' + score.toFixed(1) + "</span>" +
    '<span class="fl-num-cell">' + percent(chance) + "</span>" +
    '<span class="fl-out-bar"><i style="width:' + Math.max(1, final * 62) + '%"></i><em>' + percent(final) + "</em>" +
    (isPicked ? '<span class="fl-dice">← picked</span>' : "") + "</span></div>";
}

// the opened block: arcs of where it looks + what attention and the MLP push toward
function flowDetail(block, index, data) {
  const head = FLOW.head >= block.heads.length ? -1 : FLOW.head;
  const weights = blockWeights(block, head);
  const tokens = data.tokens;
  const n = tokens.length;
  let older = 1;
  for (const w of weights) older -= w;
  older = Math.max(0, older);

  let heads = '<button class="fl-hbtn' + (head < 0 ? " on" : "") + '" data-head="-1">all</button>';
  for (let h = 0; h < block.heads.length; h++) {
    heads += '<button class="fl-hbtn' + (head === h ? " on" : "") + '" data-head="' + h + '">' + (h + 1) + "</button>";
  }

  // arcs: from the newest token (right) back to every token it reads
  const W = 420, left = 40, column = (W - left - 6) / n, base = 96;
  const x = (c) => left + c * column + column / 2;
  let strongest = 0.0001;
  for (let c = 0; c < n - 1; c++) strongest = Math.max(strongest, weights[c]);
  let svg = '<svg class="fl-arcs" viewBox="0 0 ' + W + ' 124" role="img" aria-label="Where block ' + (index + 1) + ' looks">';
  let lastLabel = -1000;
  for (let c = n - 2; c >= 0; c--) {
    const w = weights[c];
    if (w < 0.02) continue;
    const x1 = x(n - 1), x2 = x(c);
    const lift = Math.min(80, 14 + 0.32 * (x1 - x2));
    svg += '<path d="M' + x1 + " " + base + " Q" + (x1 + x2) / 2 + " " + (base - lift * 1.8) + " " + x2 + " " + base +
      '" class="arc" style="stroke-width:' + (1 + 7 * w).toFixed(2) + ";opacity:" + (0.25 + 0.75 * w / strongest).toFixed(2) + '"/>';
    const labelX = (x1 + x2) / 2;
    if (w >= 0.1 && Math.abs(labelX - lastLabel) > 26 && (lastLabel = labelX)) svg += '<text x="' + (x1 + x2) / 2 + '" y="' + (base - lift * 0.9 - 4) + '" class="pct">' + Math.round(w * 100) + "%</text>";
  }
  if (weights[n - 1] >= 0.05) {
    svg += '<circle cx="' + x(n - 1) + '" cy="' + (base - 9) + '" r="7" class="arc" style="stroke-width:' +
      (1 + 5 * weights[n - 1]).toFixed(2) + '"/><text x="' + x(n - 1) + '" y="' + (base - 22) + '" class="pct">' +
      Math.round(weights[n - 1] * 100) + "%</text>";
  }
  for (let c = 0; c < n; c++) {
    const w = weights[c];
    svg += '<rect x="' + (x(c) - column / 2 + 1.5) + '" y="' + (base + 4) + '" width="' + (column - 3) + '" height="20" rx="5" class="chip' +
      (c === n - 1 ? " newest" : "") + '" style="fill-opacity:' + (c === n - 1 ? 1 : 0.15 + 0.85 * Math.min(1, w / strongest)).toFixed(2) + '"/>' +
      '<text x="' + x(c) + '" y="' + (base + 18) + '" class="tok">' + tokenLabel(tokens[c], 5) + "</text>";
  }
  svg += '<text x="4" y="' + (base + 18) + '" class="older">older</text><text x="4" y="' + (base - 2) + '" class="older">' +
    percent(older) + "</text></svg>";

  const pushes = (list) => list.map((piece) => "<code>" + tokenLabel(piece, 12) + "</code>").join(" ");
  return '<div class="fl-detail">' +
    '<div class="fl-detail-title">Block ' + (index + 1) + ' · where it looks <span class="fl-heads">' + heads + "</span></div>" + svg +
    '<div class="fl-cap">The newest token (right) collects information from the ones before it. Thicker arc = takes more. ' +
    "“older” = tokens further back than these 12.</div>" +
    '<div class="fl-push"><span class="tag a">A</span><span><b>Attention</b> reads the other tokens and adds what it found. It pushes toward ' +
    pushes(block.attn_push) + "</span></div>" +
    '<div class="fl-push"><span class="tag m">M</span><span><b>MLP</b> (SwiGLU / GELU) thinks about the vector alone: this is where most facts are stored. It pushes toward ' +
    pushes(block.mlp_push) + "</span></div></div>";
}

function drawHeat(canvas, heads, rgb) {
  const columns = heads[0].length;
  const rows = heads.length;
  canvas.width = columns;
  canvas.height = rows;
  const context = canvas.getContext("2d");
  const image = context.createImageData(columns, rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < columns; c++) {
      const value = Math.sqrt(Math.min(1, heads[r][c]));      // sqrt: small values stay visible
      const index = 4 * (r * columns + c);
      image.data[index] = rgb[0];
      image.data[index + 1] = rgb[1];
      image.data[index + 2] = rgb[2];
      image.data[index + 3] = Math.round(18 + 237 * value);
    }
  }
  context.putImageData(image, 0, 0);
}

// ---------------------------------------------------------------- interaction

function flowTip(html, event) {
  const tip = $("flowTip");
  if (!html) { tip.hidden = true; return; }
  tip.innerHTML = html;
  tip.hidden = false;
  const width = tip.offsetWidth, height = tip.offsetHeight;
  let x = event.clientX + 14, y = event.clientY + 16;
  if (x + width > window.innerWidth - 8) x = event.clientX - width - 14;
  if (y + height > window.innerHeight - 8) y = event.clientY - height - 12;
  tip.style.left = Math.max(8, x) + "px";
  tip.style.top = Math.max(8, y) + "px";
}

function setupFlow() {
  const box = $("flow");
  box.addEventListener("click", (event) => {
    const button = event.target.closest("[data-act]");
    if (button) {
      const act = button.dataset.act;
      if (act === "live") FLOW.index = -1;
      if (act === "prev") FLOW.index = Math.max(0, (FLOW.index < 0 ? FLOW.steps.length - 1 : FLOW.index) - 1);
      if (act === "next" && FLOW.index >= 0) FLOW.index = FLOW.index + 1 >= FLOW.steps.length ? -1 : FLOW.index + 1;
      FLOW.lastDrawn = flowShown();          // no sweep animation when you move by hand
      drawFlow();
      return;
    }
    const headButton = event.target.closest("[data-head]");
    if (headButton) { FLOW.head = parseInt(headButton.dataset.head); FLOW.lastDrawn = flowShown(); drawFlow(); return; }
    const row = event.target.closest(".fl-row");
    if (row) { FLOW.block = parseInt(row.dataset.block); FLOW.lastDrawn = flowShown(); drawFlow(); }
  });
  box.addEventListener("mousemove", (event) => {
    const heat = event.target.closest(".fl-heat");
    const step = flowShown();
    if (heat && step && step.galaxy) {
      const block = step.galaxy.blocks[parseInt(heat.dataset.block)];
      const rect = heat.getBoundingClientRect();
      const head = Math.min(block.heads.length - 1, Math.floor((event.clientY - rect.top) / rect.height * block.heads.length));
      const row = block.heads[head];
      let best = 0;
      for (let c = 1; c < row.length; c++) if (row[c] > row[best]) best = c;
      const column = Math.min(row.length - 1, Math.floor((event.clientX - rect.left) / rect.width * row.length));
      let older = 1;
      for (const value of row) older -= value;
      flowTip("<b>Block " + (parseInt(heat.dataset.block) + 1) + ", head " + (head + 1) + "</b><br>" +
        "reads <code>" + tokenLabel(step.galaxy.tokens[column]) + "</code> " + percent(row[column]) +
        "<br>reads most: <code>" + tokenLabel(step.galaxy.tokens[best]) + "</code> " + percent(row[best]) +
        "<br>tokens further back: " + percent(Math.max(0, older)), event);
      return;
    }
    const target = event.target.closest("[data-tip]");
    flowTip(target ? target.dataset.tip : null, event);
  });
  box.addEventListener("mouseleave", () => flowTip(null));
}
