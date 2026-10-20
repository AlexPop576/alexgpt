// "Show tokens": every message is cut into tokens the way the current model reads it.
// Your messages AND the replies: each colored box is one token (one row of the model's table).
// Galaxy / Supernova cut words into BPE pieces ("␣the", "ing"); Spark, Nebula, Pulsar read single characters.
// The text is sent to the server (/api/tokenize), which uses the model's real tokenizer.

const TOKEN_VIEW = { on: false, queued: false };

function tokenViewTargets() {
  const targets = [];
  for (const message of document.querySelectorAll("#messages .message.done")) {
    const element = message.querySelector(".user-bubble, .bot-text");
    if (element && !element.dataset.tokenized) targets.push(element);
  }
  return targets;
}

async function tokenizeElement(element) {
  element.dataset.tokenized = "pending";
  const echo = element.querySelector(".echo");
  const text = element.textContent.slice(echo ? echo.textContent.length : 0);
  const message = element.closest(".message");
  const model = message.dataset.model && state.available[message.dataset.model] ? message.dataset.model : state.current;
  let tokens;
  try {
    const response = await fetch("/api/tokenize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, text }),
    });
    tokens = (await response.json()).tokens;
  } catch (e) {
    element.dataset.tokenized = "";
    return;
  }
  if (!TOKEN_VIEW.on || !tokens) { element.dataset.tokenized = ""; return; }

  element.savedHtml = element.innerHTML;               // to go back when the view is turned off
  const box = document.createDocumentFragment();
  let known = 0;
  tokens.forEach(([piece, id], index) => {
    const span = document.createElement("span");
    span.className = "bpe c" + (index % 5) + (id < 0 ? " unknown" : "");
    span.title = id < 0 ? "this character is not in " + model + "'s vocabulary: it is dropped" : "token #" + id;
    if (id >= 0) known += 1;
    // make spaces and line breaks visible: they are part of the token
    const leading = piece.match(/^ +/);
    if (leading) {
      const mark = document.createElement("i");
      mark.textContent = "␣".repeat(leading[0].length);
      span.append(mark);
      piece = piece.slice(leading[0].length);
    }
    const parts = piece.split("\n");
    parts.forEach((part, partIndex) => {
      if (partIndex > 0) {
        const mark = document.createElement("i");
        mark.textContent = "↵";
        span.append(mark, document.createElement("br"));
      }
      if (part) span.append(document.createTextNode(part));
    });
    box.append(span);
  });
  const count = document.createElement("div");
  count.className = "bpe-count";
  count.textContent = known + " tokens · " + text.length + " characters · " + model;
  element.innerHTML = "";
  element.append(box, count);
  element.dataset.tokenized = "yes";
}

function refreshTokenView() {
  TOKEN_VIEW.queued = false;
  if (!TOKEN_VIEW.on) return;
  for (const element of tokenViewTargets()) tokenizeElement(element);
}

function queueTokenView() {
  if (TOKEN_VIEW.queued) return;
  TOKEN_VIEW.queued = true;
  setTimeout(refreshTokenView, 60);
}

function clearTokenView() {
  for (const element of document.querySelectorAll("#messages [data-tokenized]")) {
    if (element.savedHtml !== undefined) element.innerHTML = element.savedHtml;
    delete element.savedHtml;
    delete element.dataset.tokenized;
  }
}

function setTokenView(on) {
  TOKEN_VIEW.on = on;
  $("tokensToggle").checked = on;
  document.documentElement.dataset.tokens = on ? "1" : "0";
  savePreference("alexgpt-tokens", on ? "on" : "off");
  clearTokenView();
  if (on) queueTokenView();
}

function setupTokenView() {
  $("tokensToggle").addEventListener("change", () => setTokenView($("tokensToggle").checked));
  // new messages, finished replies, or another chat opened: tokenize what is new
  new MutationObserver(queueTokenView).observe($("messages"), { childList: true, subtree: true, attributes: true, attributeFilter: ["class"] });
  // another model = another tokenizer: your messages are cut again its way
  new MutationObserver(() => { if (TOKEN_VIEW.on) { clearTokenView(); queueTokenView(); } })
    .observe(document.documentElement, { attributes: true, attributeFilter: ["data-model"] });
  setTokenView(loadPreference("alexgpt-tokens") === "on");
}
