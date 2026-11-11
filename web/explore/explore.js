// Explore mode: a 3D walk through what Singularity does with one message.
// The data is REAL: POST /api/explore runs the model on the GPU and returns every decision and number (alexgpt/explore.py);
// fromRecord() turns that record into `demo`, the shape the scene plays.  Without a server, the made-up SCENARIOS
// below are played instead (an offline demo, marked "fake data" at the top).
//
// The scene is one straight line of 8 numbered stations, joined by a rail. The message travels along the rail as a labeled packet:
//
//   01 Router -> 02 Tool -> 03 Tokens -> 04 Embeddings -> 05 16 blocks -> 06 Scores -> 07 Pick one -> 08 Answer
//                              ^                                                          |
//                              +-------------- the new token goes back (the loop) --------+
//
// Each station = an HTML board standing in 3D (CSS3DRenderer: crisp text, normal CSS) + WebGL objects
// (rail, floor plates, the 16 blocks, the wheel, packets). WebGL glows thanks to a bloom pass.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS3DRenderer, CSS3DObject } from "three/addons/renderers/CSS3DRenderer.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

// ---------------------------------------------------------------- colors
const COLORS = { text: "#ecebf7", muted: "#9a98b3", faint: "#62607a", accent: "#c9c5ff", router: "#7fd8ea", attn: "#a9a5ff", mlp: "#ff9d6e", ground: "#6fd3a0", ban: "#ff6a45", wiki: "#e9e3c4" };
const RANK_COLORS = ["#ff9d6e", "#a9a5ff", "#7fd8ea", "#e9a6dc", "#f3c969", "#9be3c4", "#8dbbff", "#ffb0c0"];   // same color for a token's bar and its wheel slice

// ---------------------------------------------------------------- the router's 5 checks and the tools
const GATES = [                       // the same order as SingularityChat.run_tools_inner()
  { key: "followup", icon: "↩", name: "Follow-up?", asks: "A short follow-up like “and in Cluj?”" },
  { key: "rules", icon: "⚙", name: "Fixed rules", asks: "Math, units, time, weather, dice, memory?" },
  { key: "identity", icon: "☺", name: "About me?", asks: "A question about the model itself?" },
  { key: "question", icon: "?", name: "Factual question?", asks: "Who / what / when … → look it up" },
  { key: "search", icon: "⌕", name: "Model decides", asks: "Nothing matched: the model may write <|search|>" },
];
const TOOL_TABS = [
  { key: "wiki", icon: "W", name: "Wikipedia", color: COLORS.wiki },
  { key: "calc", icon: "×÷", name: "Calculator", color: COLORS.router },
  { key: "convert", icon: "km", name: "Units", color: COLORS.router },
  { key: "now", icon: "12h", name: "Time", color: COLORS.router },
  { key: "weather", icon: "°C", name: "Weather", color: COLORS.router },
  { key: "dice", icon: "⚄", name: "Dice", color: COLORS.router },
  { key: "memory", icon: "M+", name: "Memory", color: COLORS.router },
  { key: "identity", icon: "ID", name: "Identity file", color: COLORS.router },
];

// ---------------------------------------------------------------- fake data, one per demo question
const SCENARIOS = {
  wiki: {
    chip: "Wikipedia", prompt: "Who invented the telephone?",
    gates: { followup: ["no", "there is no earlier tool call to repeat"], rules: ["no", "not math, units, time or weather"], identity: ["no", "not a question about the model"], question: ["yes", "“who …?” asks for a fact → Wikipedia"] },
    tool: {
      kind: "wiki", tab: "wiki", call: "wiki: en: Who invented the telephone",
      results: [
        { title: "Invention of the telephone", ok: true, why: "has “invented” + “telephone”" },
        { title: "Telephone", ok: true, why: "relevant, lower score" },
        { title: "Inventor (patent)", ok: false, why: "no “telephone”" },
        { title: "The Telephone (opera)", ok: false, why: "an opera, not the device" },
        { title: "Who (pronoun)", ok: false, why: "only matches “who”" },
      ],
      article: "Invention of the telephone",
      sentences: [
        ["The invention of the telephone was the work of many people.", 0.41],
        ["Alexander Graham Bell was the first to be awarded a patent for the electric telephone in 1876.", 0.82],
        ["Antonio Meucci and Elisha Gray also built early telephones.", 0.58],
        ["Disputes about who was first led to many lawsuits.", 0.47],
      ],
      best: 1,
    },
    context: "Alexander Graham Bell was the first to be awarded a patent for the electric telephone in 1876.",
    userTokens: ["Who", " invented", " the", " telephone", "?"],
    contextTokens: ["Alexander", " Graham", " Bell", " was", " the", " first", " to", " be", " awarded", " a", " patent", " for", " the", " electric", " telephone", " in", " 18", "76", "."],
    focus: [" telephone", " Bell"],        // the token the newest one looks at most: early blocks, late blocks
    generated: [
      { token: "Alexander", top: [["Alexander", 0.41], ["The", 0.17], ["Bell", 0.12], ["Thomas", 0.09, "ban"], ["Antonio", 0.06], ["Elisha", 0.04]] },
      { token: " Graham", top: [[" Graham", 0.93], [" Bell", 0.04], [" G", 0.01]] },
      { token: " Bell", top: [[" Bell", 0.98], [" B", 0.01]] },
    ],
    lens: ["the", "the", " a", "The", "The", " who", "Bell", "Bell", "Bell", "Alexander", "Alexander", "Alexander", "Alexander", "Alexander", "Alexander", "Alexander"],
    rest: " is credited with inventing the first practical telephone.",
    grounded: "Bell was awarded the first U.S. patent for the telephone in 1876.",
    source: "Invention of the telephone",
  },
  calc: {
    chip: "Calc", prompt: "What is 17*23?",
    gates: { followup: ["no", "there is no earlier tool call to repeat"], rules: ["yes", "“17*23” is math → calculator"] },
    tool: { kind: "calc", tab: "calc", call: "calc: 17*23", left: "17", op: "×", right: "23", result: "391" },
    context: "17*23 = 391",
    userTokens: ["What", " is", " 17", "*", "23", "?"],
    contextTokens: ["17", "*", "23", " =", " 391"],
    focus: ["*", " 391"],
    generated: [
      { token: "17", top: [["17", 0.71], ["The", 0.15], ["391", 0.08]] },
      { token: " ×", top: [[" ×", 0.55], [" *", 0.31], [" times", 0.09]] },
      { token: " 23", top: [[" 23", 0.97], [" 2", 0.01]] },
    ],
    lens: ["the", " is", " is", "17", "17", "17", "17", "17", "17", "17", "17", "17", "17", "17", "17", "17"],
    rest: " = 391.",
  },
  followup: {
    chip: "Follow-up", prompt: "And in Cluj?",
    before: { user: "What's the weather in Bucharest?", call: "weather: Bucharest", reply: "In Bucharest it is 9°C and cloudy." },
    gates: { followup: ["yes", "short, starts with “and”, names a place → repeat the last tool"] },
    tool: { kind: "weather", tab: "weather", call: "weather: Cluj", oldPlace: "Bucharest", newPlace: "Cluj", rewritten: "What's the weather in Cluj?", temp: "7°C", sky: "light rain" },
    context: "Cluj: 7°C, light rain",
    userTokens: ["What", "'s", " the", " weather", " in", " Cluj", "?"],
    contextTokens: ["Cl", "uj", ":", " 7", "°C", ",", " light", " rain"],
    focus: [" weather", " rain"],
    generated: [
      { token: "In", top: [["In", 0.62], ["It", 0.21], ["Cluj", 0.1]] },
      { token: " Cluj", top: [[" Cluj", 0.95], [" the", 0.02]] },
      { token: " it", top: [[" it", 0.81], [" there", 0.1]] },
    ],
    lens: ["the", " the", " in", " in", "In", "In", "In", "In", "In", "In", "In", "In", "In", "In", "In", "In"],
    rest: " is 7°C with light rain.",
  },
  identity: {
    chip: "Identity", prompt: "Who made you?",
    gates: { followup: ["no", "there is no earlier tool call"], rules: ["no", "no tool fits"], question: ["no", "it is about me, not about the world"], identity: ["yes", "very close to “Who created you?” in the identity file"] },
    tool: { kind: "identity", tab: "identity", matches: [["Who created you?", 0.91], ["What is your name?", 0.34], ["How many parameters do you have?", 0.22], ["What can you do?", 0.18]], threshold: 0.75 },
    direct: { kind: "identity", text: "I was made by Alex, a computer science student in Cluj, as a project to learn how language models work." },
  },
  abstain: {
    chip: "No source", prompt: "Who is Zorblax Quint?",
    gates: { followup: ["no", "there is no earlier tool call"], rules: ["no", "no tool fits"], identity: ["no", "not a question about the model"], question: ["yes", "“who is …?” asks for a fact → Wikipedia"] },
    tool: { kind: "wiki", tab: "wiki", call: "wiki: en: Zorblax Quint", results: [
      { title: "Quint (band)", ok: false, why: "no “Zorblax”" },
      { title: "Quintus", ok: false, why: "no “Zorblax”" },
      { title: "List of fictional robots", ok: false, why: "no “Zorblax Quint”" },
    ] },
    direct: { kind: "abstain", text: "I couldn't find a source about “Zorblax Quint”, so I won't guess." },
  },
};
let demo = SCENARIOS.wiki;

// ---------------------------------------------------------------- where everything sits
const PX = 0.025;                  // 1 board pixel = 0.025 world units
const RAIL_Y = -14;
const ANCHOR_HEIGHT = 1100;        // every board hangs from its top edge inside an invisible box this tall
const STATIONS = [
  { key: "router", num: "01", name: "Router", color: COLORS.router, x: -150, width: 900, view: { y: 1, dist: 38 } },
  { key: "tool", num: "02", name: "Tool", color: COLORS.wiki, x: -113, width: 1300, view: { y: 0, dist: 44 } },
  { key: "tokens", num: "03", name: "Tokens", color: COLORS.accent, x: -80, width: 1000, view: { y: 3, dist: 38 } },
  { key: "embed", num: "04", name: "Embeddings", color: COLORS.accent, x: -44, width: 1180, view: { y: 2, dist: 44 } },
  { key: "blocks", num: "05", name: "16 transformer blocks", color: COLORS.attn, x: -4, width: 1120, top: 19.5, view: { y: 3, dist: 50 } },
  { key: "logits", num: "06", name: "Scores → chances", color: COLORS.mlp, x: 34, width: 760, view: { y: 4, dist: 34 } },
  { key: "sampling", num: "07", name: "Pick one token", color: COLORS.mlp, x: 62, width: 640, view: { y: 0, dist: 36 } },
  { key: "answer", num: "08", name: "Answer", color: COLORS.ground, x: 94, width: 860, view: { y: 6, dist: 34 } },
];
const station = (key) => STATIONS.find((item) => item.key === key);
// the camera for a station: far enough that the whole board (and its 3D objects) fits between the HUD bars
function stationCamera(key) {
  const s = station(key);
  const top = s.top || 13;
  const boardHeight = (boards[key] ? boards[key].offsetHeight : 800) * PX;
  let bottom = top - boardHeight;
  if (key === "blocks") bottom = MLP_Y - 4;
  if (key === "sampling") bottom = WHEEL_Y - 5.5;
  const halfTan = Math.tan((camera.fov * Math.PI) / 360);
  const wide = window.innerWidth > 900;
  const usableHeight = Math.max(0.4, (window.innerHeight - 250) / window.innerHeight);
  const usableWidth = Math.max(0.45, (window.innerWidth - (wide ? 500 : 40)) / window.innerWidth);
  const fitHeight = (top - bottom + 1) / usableHeight / (2 * halfTan);
  const fitWidth = (s.width * PX + 2) / usableWidth / (2 * halfTan * camera.aspect);
  const distance = Math.max(fitHeight, fitWidth, 18);
  const centerY = (top + bottom) / 2;
  if (key === "blocks") return [[s.x + distance * 0.3, centerY + distance * 0.1, distance * 1.02], [s.x - 1.5, centerY - 1.5, 0]];
  return [[s.x, centerY + distance * 0.08, distance], [s.x, centerY, 0]];
}
const OVERVIEW = [[-28, 40, 175], [-28, -4, 0]];
const LOOP_VIEW = [[2, 16, 128], [2, -6, 0]];
const RAIL_START = -168, RAIL_END = 112;

const BLOCK_GAP = 1.75, STREAM_Y = -9.1, ATTN_Y = -6.3, MLP_Y = -12;
const blockX = (index) => station("blocks").x + (index - 7.5) * BLOCK_GAP;
const WHEEL_Y = -6;

// ---------------------------------------------------------------- three.js basics: WebGL scene + CSS3D scene, one camera
const world = document.getElementById("world");
const canvas = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
const scene = new THREE.Scene();
scene.background = new THREE.Color("#07070c");
scene.fog = new THREE.FogExp2("#07070c", 0.0035);
const cssScene = new THREE.Scene();
const cssRenderer = new CSS3DRenderer();
cssRenderer.domElement.id = "css3d";
world.append(cssRenderer.domElement);

const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 900);
camera.position.set(...OVERVIEW[0]);
const controls = new OrbitControls(camera, world);     // on the wrapper, so dragging over a board also moves the camera
controls.target.set(...OVERVIEW[1]);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxDistance = 280;

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.7, 0.55, 0.3);
composer.addPass(bloom);
composer.addPass(new OutputPass());

function resize() {
  const width = window.innerWidth, height = window.innerHeight;
  renderer.setSize(width, height, false);
  composer.setSize(width, height);
  cssRenderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

// ---------------------------------------------------------------- small helpers
const glowMaterial = (color, opacity = 1) =>
  new THREE.MeshBasicMaterial({ color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false });
const $ = (id) => document.getElementById(id);
const esc = (text) => String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const showTok = (token) => token.replace(/^ /, "␣");
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const lerp = (a, b, t) => a + (b - a) * t;
const vec = (array) => new THREE.Vector3(...array);

// a text label drawn once on a canvas; a sprite always faces the camera, a "flat" label lies on the floor
function labelCanvas(text, options) {
  text = text.split("<|").join("<‌|").split("|>").join("|‌>");      // a zero-width break stops the mono font from drawing <| as a triangle
  const fontSize = 64;
  const family = options.mono ? '"JetBrains Mono", monospace' : "Inter, sans-serif";
  const font = (options.weight || 500) + " " + fontSize + "px " + family;
  const drawing = document.createElement("canvas").getContext("2d");
  drawing.font = font;
  const padX = options.bg ? 30 : 8;
  const width = Math.max(8, Math.ceil(drawing.measureText(text).width) + padX * 2);
  const height = fontSize + (options.bg ? 34 : 16);
  drawing.canvas.width = width;
  drawing.canvas.height = height;
  drawing.font = font;
  if (options.bg) {
    drawing.fillStyle = options.bg;
    drawing.strokeStyle = options.border || "rgba(255,255,255,0.2)";
    drawing.lineWidth = 4;
    drawing.beginPath();
    drawing.roundRect(2, 2, width - 4, height - 4, height / 2 - 2);
    drawing.fill();
    drawing.stroke();
  }
  drawing.fillStyle = options.color || COLORS.text;
  drawing.textBaseline = "middle";
  drawing.fillText(text, padX, height / 2 + 4);
  const texture = new THREE.CanvasTexture(drawing.canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  return { texture, aspect: width / height };
}
function makeLabel(text, options = {}) {
  const { texture, aspect } = labelCanvas(text, options);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false, fog: false }));
  const size = options.size || 0.8;
  sprite.scale.set(size * aspect, size, 1);
  return sprite;
}
function makeFloorLabel(text, options = {}) {
  const { texture, aspect } = labelCanvas(text, options);
  const size = options.size || 1;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(size * aspect, size), new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false }));
  mesh.rotation.x = -Math.PI / 2;
  return mesh;
}

function disposeGroup(group) {
  for (const child of [...group.children]) {
    group.remove(child);
    child.traverse((part) => {
      if (part.geometry) part.geometry.dispose();
      if (part.material) { if (part.material.map) part.material.map.dispose(); part.material.dispose(); }
    });
  }
}
const tagStation = (key, object) => { object.traverse((part) => { part.userData.station = key; }); return object; };

// ---------------------------------------------------------------- time: pause, speed, step, cancel
// Every animation is a "task" that the render loop moves forward. Pausing stops them; "step" finishes the current ones.
const clock = { paused: false, speed: 1, run: 0, finishNow: false };
const tasks = new Set();
const CANCELLED = new Error("cancelled");

function animate(duration, update) {
  const run = clock.run;
  return new Promise((resolve, reject) => tasks.add({ elapsed: 0, duration, update, resolve, reject, run }));
}
const wait = (ms) => animate(ms, () => {});

function moveTasks(deltaMs) {
  for (const task of [...tasks]) {
    if (task.run !== clock.run) { tasks.delete(task); task.reject(CANCELLED); continue; }
    if (clock.finishNow) task.elapsed = task.duration;
    else if (!clock.paused) task.elapsed += deltaMs * clock.speed;
    const t = Math.min(1, task.elapsed / Math.max(1, task.duration));
    task.update(t);
    if (t >= 1) { tasks.delete(task); task.resolve(); }
  }
  clock.finishNow = false;
}

// the camera flies on its own clock (it moves even while paused). "via" bends the path through a wider view.
let flight = null;
function flyTo([position, target], duration = 1400, via = null, byUser = false) {
  flight = {
    elapsed: 0, duration, byUser,
    p0: camera.position.clone(), p2: vec(position), p1: via ? vec(via[0]) : null,
    t0: controls.target.clone(), t2: vec(target), t1: via ? vec(via[1]) : null,
  };
}
const bezier = (a, b, c, t) => new THREE.Vector3().copy(a).multiplyScalar((1 - t) * (1 - t)).addScaledVector(b, 2 * (1 - t) * t).addScaledVector(c, t * t);
function moveCamera(deltaMs) {
  if (!flight) return;
  if (clock.paused && !flight.byUser) return;            // the run's own camera moves pause with it; the user's clicks always move
  flight.elapsed += deltaMs * (flight.byUser ? 1 : Math.max(1, clock.speed));
  const t = ease(Math.min(1, flight.elapsed / flight.duration));
  if (flight.p1) {
    camera.position.copy(bezier(flight.p0, flight.p1, flight.p2, t));
    controls.target.copy(bezier(flight.t0, flight.t1, flight.t2, t));
  } else {
    camera.position.lerpVectors(flight.p0, flight.p2, t);
    controls.target.lerpVectors(flight.t0, flight.t2, t);
  }
  if (t >= 1) flight = null;
}
controls.addEventListener("start", () => { flight = null; });   // the user grabbed the camera: stop flying

// ---------------------------------------------------------------- the fixed scenery
function addStars() {
  const count = 2600;
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const radius = 240 + Math.random() * 260;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = radius * Math.cos(phi);
    positions[i * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({ color: "#6f6c95", size: 0.9, sizeAttenuation: true, fog: false })));
}

function addFloor() {
  const grid = new THREE.GridHelper(700, 140, "#2a2840", "#1a1928");
  grid.material.transparent = true;
  grid.material.opacity = 0.35;
  grid.position.y = RAIL_Y - 1;
  scene.add(grid);
}

function roundedRect(width, depth, radius) {
  const shape = new THREE.Shape();
  const x = -width / 2, y = -depth / 2;
  shape.moveTo(x + radius, y);
  shape.lineTo(x + width - radius, y);
  shape.quadraticCurveTo(x + width, y, x + width, y + radius);
  shape.lineTo(x + width, y + depth - radius);
  shape.quadraticCurveTo(x + width, y + depth, x + width - radius, y + depth);
  shape.lineTo(x + radius, y + depth);
  shape.quadraticCurveTo(x, y + depth, x, y + depth - radius);
  shape.lineTo(x, y + radius);
  shape.quadraticCurveTo(x, y, x + radius, y);
  return shape;
}

// every station stands on a plate with its number written on the floor
function addPlates() {
  for (const s of STATIONS) {
    const width = s.width * PX + 3;
    const shape = roundedRect(width, 9, 1.4);
    const plate = new THREE.Mesh(new THREE.ShapeGeometry(shape, 12), new THREE.MeshBasicMaterial({ color: "#0e0e1a", transparent: true, opacity: 0.92 }));
    plate.rotation.x = -Math.PI / 2;
    plate.position.set(s.x, RAIL_Y - 0.7, 0);
    const edge = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(shape.getPoints(64)), new THREE.LineBasicMaterial({ color: s.color, transparent: true, opacity: 0.55 }));
    edge.rotation.x = -Math.PI / 2;
    edge.position.set(s.x, RAIL_Y - 0.66, 0);
    const label = makeFloorLabel(s.num + "  " + s.name.toUpperCase(), { color: s.color, weight: 600, size: 1.1 });
    label.position.set(s.x, RAIL_Y - 0.64, 3.2);
    s.edge = edge;
    s.floorLabel = label;
    scene.add(tagStation(s.key, plate), edge, tagStation(s.key, label));
  }
}
function dimStation(key, dim) {
  const s = station(key);
  s.edge.material.opacity = dim ? 0.12 : 0.55;
  s.floorLabel.material.opacity = dim ? 0.25 : 1;
}

// the rail: where information travels, left to right, with arrows between stations
let loopCurve, evidenceCurve, evidence;
function addRails() {
  const rail = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, RAIL_END - RAIL_START, 8), glowMaterial("#8f8cff", 0.55));
  rail.rotation.z = Math.PI / 2;
  rail.position.set((RAIL_START + RAIL_END) / 2, RAIL_Y, 0);
  scene.add(rail);
  for (let index = 0; index < STATIONS.length - 1; index++) {
    const arrow = new THREE.Mesh(new THREE.ConeGeometry(0.35, 0.9, 16), glowMaterial("#c9c5ff", 0.9));
    arrow.rotation.z = -Math.PI / 2;
    const left = STATIONS[index], right = STATIONS[index + 1];
    arrow.position.set((left.x + left.width * PX / 2 + right.x - right.width * PX / 2) / 2, RAIL_Y, 0);
    scene.add(arrow);
  }
  // the loop: from "pick one token" back to "tokens", in front of the plates
  const pick = station("sampling").x, tokens = station("tokens").x;
  loopCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(pick, RAIL_Y, 0), new THREE.Vector3(pick - 2, RAIL_Y - 2.4, 5.2), new THREE.Vector3((pick + tokens) / 2, RAIL_Y - 3.4, 6.6),
    new THREE.Vector3(tokens + 2, RAIL_Y - 2.4, 5.2), new THREE.Vector3(tokens, RAIL_Y, 0),
  ]);
  scene.add(new THREE.Mesh(new THREE.TubeGeometry(loopCurve, 160, 0.06, 8, false), glowMaterial(COLORS.accent, 0.4)));
  const loopLabel = makeLabel("↺  each new token goes back to 03 and the loop runs again", { size: 1.1, color: COLORS.accent });
  loopLabel.position.copy(loopCurve.getPointAt(0.5)).add(new THREE.Vector3(0, -1.2, 0));
  scene.add(loopLabel);
  // the evidence arc: copied Wikipedia text goes straight from 02 to 08 (only for grounded answers)
  const from = station("tool").x, to = station("answer").x;
  evidenceCurve = new THREE.QuadraticBezierCurve3(new THREE.Vector3(from, 14.5, -1), new THREE.Vector3((from + to) / 2, 46, -4), new THREE.Vector3(to, 14.5, -1));
  evidence = new THREE.Group();
  evidence.add(new THREE.Mesh(new THREE.TubeGeometry(evidenceCurve, 160, 0.06, 8, false), glowMaterial(COLORS.ground, 0.45)));
  const evidenceLabel = makeLabel("copied text from Wikipedia (grounded)", { size: 1.2, color: COLORS.ground });
  evidenceLabel.position.copy(evidenceCurve.getPointAt(0.5)).add(new THREE.Vector3(0, 1.4, 0));
  evidence.add(evidenceLabel);
  evidence.visible = false;
  scene.add(evidence);
}

// small dots that always flow along the rails, so you can see the direction
const flow = {};
function addFlow() {
  const make = (count, color) => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
    const points = new THREE.Points(geometry, new THREE.PointsMaterial({ color, size: 0.32, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false }));
    points.userData.offsets = Array.from({ length: count }, () => Math.random());
    scene.add(points);
    return points;
  };
  flow.main = make(110, "#d6d3ff");
  flow.loop = make(40, COLORS.accent);
  flow.lanes = make(60, "#ffffff");
}
function moveFlow(seconds) {
  const main = flow.main.geometry.attributes.position;
  flow.main.userData.offsets.forEach((offset, k) => {
    const u = (offset + seconds * 0.02) % 1;
    main.setXYZ(k, lerp(RAIL_START, RAIL_END, u), RAIL_Y + 0.02, 0);
  });
  main.needsUpdate = true;
  const loop = flow.loop.geometry.attributes.position;
  flow.loop.userData.offsets.forEach((offset, k) => {
    const point = loopCurve.getPointAt((offset + seconds * 0.04) % 1);
    loop.setXYZ(k, point.x, point.y, point.z);
  });
  loop.needsUpdate = true;
}

// 16 transformer blocks in 3D. Every token of the message has its own lane (a line running through all blocks);
// the newest token's lane is in front. Each block has two slabs the lanes pass between:
//   up:   attention (violet slab, 12 head spheres inside)   - arcs jump from the newest lane to earlier lanes
//   down: MLP (orange slab, neurons narrow -> wide -> narrow) - the token thinks alone
const blocks = [];
const LANES = 12;                                        // how many token lanes are drawn (the last 12 tokens, like the hooks)
const laneZ = (k) => -5 + k * 0.9;                       // lane k (0 = oldest shown, 11 = newest) sits at this depth
const NEW_Z = laneZ(LANES - 1);
const SLAB_DEPTH = 11;
const NEURON_ROWS = [[6, 0.95], [14, 0], [6, -0.95]];     // [neurons, height offset inside the MLP slab]
const HEAD_COLORS = ["#a9a5ff", "#7fd8ea", "#e9a6dc", "#f3c969", "#8dbbff", "#c9c5ff", "#9be3c4", "#ffb0c0", "#b3a2ff", "#7fc4ea", "#e6c9ff", "#ffd59a"];
function addBlocks() {
  const first = blockX(0);
  const names = [["attention", ATTN_Y, COLORS.attn], ["MLP", MLP_Y, COLORS.mlp]];
  for (const [text, y, color] of names) {
    const label = makeLabel(text, { size: 0.75, color, weight: 600 });
    label.position.set(first - 1.2 - label.scale.x / 2, y, NEW_Z + 1);
    scene.add(tagStation("blocks", label));
  }
  const attnBox = new THREE.BoxGeometry(0.95, 2.4, SLAB_DEPTH);
  const mlpBox = new THREE.BoxGeometry(0.95, 2.9, SLAB_DEPTH);
  const sphere = new THREE.SphereGeometry(1, 12, 12);
  for (let index = 0; index < 16; index++) {
    const x = blockX(index);
    const group = new THREE.Group();
    const attn = new THREE.Mesh(attnBox, glowMaterial(COLORS.attn, 0.05));
    attn.position.set(x, ATTN_Y, 0);
    const attnEdge = new THREE.LineSegments(new THREE.EdgesGeometry(attnBox), new THREE.LineBasicMaterial({ color: COLORS.attn, transparent: true, opacity: 0.35 }));
    attnEdge.position.copy(attn.position);
    const mlp = new THREE.Mesh(mlpBox, glowMaterial(COLORS.mlp, 0.05));
    mlp.position.set(x, MLP_Y, 0);
    const mlpEdge = new THREE.LineSegments(new THREE.EdgesGeometry(mlpBox), new THREE.LineBasicMaterial({ color: COLORS.mlp, transparent: true, opacity: 0.35 }));
    mlpEdge.position.copy(mlp.position);
    // 12 attention heads inside the violet slab
    const heads = new THREE.InstancedMesh(sphere, new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.95 }), 12);
    for (let head = 0; head < 12; head++) {
      heads.setMatrixAt(head, new THREE.Matrix4().compose(new THREE.Vector3(x, ATTN_Y, -4.95 + head * 0.9), new THREE.Quaternion(), new THREE.Vector3(0.2, 0.2, 0.2)));
      heads.setColorAt(head, new THREE.Color(HEAD_COLORS[head]).multiplyScalar(0.25));
    }
    // MLP neurons in 3 rows + the wires between them
    const count = NEURON_ROWS.reduce((sum, row) => sum + row[0], 0);
    const neurons = new THREE.InstancedMesh(sphere, new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.95 }), count);
    const rows = [];
    let n = 0;
    NEURON_ROWS.forEach(([size, offset], rowIndex) => {
      const span = size > 6 ? 9.6 : 6;
      const row = [];
      for (let k = 0; k < size; k++) {
        const position = new THREE.Vector3(x, MLP_Y + offset, -span / 2 + (span * k) / (size - 1));
        neurons.setMatrixAt(n, new THREE.Matrix4().compose(position, new THREE.Quaternion(), new THREE.Vector3(0.12, 0.12, 0.12)));
        neurons.setColorAt(n, new THREE.Color(COLORS.mlp).multiplyScalar(0.2));
        row.push({ index: n, row: rowIndex, position });
        n += 1;
      }
      rows.push(row);
    });
    const wirePoints = [];
    for (let rowIndex = 0; rowIndex < 2; rowIndex++) for (const a of rows[rowIndex]) for (const b of rows[rowIndex + 1]) wirePoints.push(a.position, b.position);
    const wires = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(wirePoints),
      new THREE.LineBasicMaterial({ color: COLORS.mlp, transparent: true, opacity: 0.04, blending: THREE.AdditiveBlending, depthWrite: false }));
    const number = makeLabel(String(index + 1), { size: 0.6, color: COLORS.muted, mono: true });
    number.position.set(x, ATTN_Y + 1.9, -SLAB_DEPTH / 2);
    group.add(attn, attnEdge, mlp, mlpEdge, heads, neurons, wires, number);
    group.userData = { attn, attnEdge, mlp, mlpEdge, heads, neurons, rows, wires };
    blocks.push(group);
    scene.add(tagStation("blocks", group));
  }
}
// strength 0..1 lights a whole slab (and the spheres inside it)
function lightPart(index, part, strength) {
  const block = blocks[index].userData;
  const mesh = part === "attn" ? block.attn : block.mlp;
  const edge = part === "attn" ? block.attnEdge : block.mlpEdge;
  mesh.material.opacity = 0.05 + 0.25 * strength;
  edge.material.opacity = 0.35 + 0.65 * strength;
  if (part === "attn") lightHeads(index, new Array(12).fill(strength));
  else lightNeurons(index, strength * 3);
}
function lightHeads(index, strengths) {
  const { heads } = blocks[index].userData;
  strengths.forEach((value, head) => heads.setColorAt(head, new THREE.Color(HEAD_COLORS[head]).multiplyScalar(0.25 + 1.3 * value)));
  heads.instanceColor.needsUpdate = true;
}
function lightNeurons(index, progress) {                // progress 0..3: how far the signal went (row 1, 2, 3)
  const { neurons, rows, wires } = blocks[index].userData;
  for (const row of rows) for (const spot of row) {
    const on = Math.max(0, Math.min(1, progress - spot.row));
    const flicker = spot.row === 1 ? 0.25 + 0.75 * Math.abs(Math.sin(spot.index * 2.7 + index * 1.3)) : 1;
    neurons.setColorAt(spot.index, new THREE.Color(COLORS.mlp).multiplyScalar(0.2 + 1.3 * on * flicker));
  }
  neurons.instanceColor.needsUpdate = true;
  wires.material.opacity = 0.04 + 0.3 * Math.min(1, progress / 2);
}
function resetBlocks() {
  for (let index = 0; index < 16; index++) { lightPart(index, "attn", 0); lightPart(index, "mlp", 0); }
}

// the token lanes for the current message: a line per token through all blocks, its name and a little vector at the start
let lanes = null;
function buildLanes() {
  if (lanes) { live.remove(lanes); disposeGroup(lanes); }
  lanes = new THREE.Group();
  const windowTokens = attentionWindow();
  const startX = blockX(0) - 4.5, endX = blockX(15) + 3;
  const cell = new THREE.BoxGeometry(0.22, 0.22, 0.22);
  windowTokens.forEach((token, k) => {
    const z = laneZ(k + LANES - windowTokens.length);
    const newest = k === windowTokens.length - 1;
    const radius = newest ? 0.07 : 0.035;
    const line = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, endX - startX, 6), glowMaterial(newest ? COLORS.accent : "#ffffff", newest ? 0.7 : 0.22));
    line.rotation.z = Math.PI / 2;
    line.position.set((startX + endX) / 2, STREAM_Y, z);
    lanes.add(line);
    for (let dim = 0; dim < 6; dim++) {             // 6 of the 768 numbers of its vector, as colored cubes
      const value = vectorValue(token.id, dim);
      const cube = new THREE.Mesh(cell, glowMaterial(value > 0 ? COLORS.mlp : COLORS.attn, 0.25 + 0.6 * Math.abs(value)));
      cube.position.set(startX - 2.6 + dim * 0.28, STREAM_Y, z);
      lanes.add(cube);
    }
    const name = makeLabel(showTok(token.text), { size: 0.5, mono: true, color: newest ? COLORS.accent : token.kind === "special" ? COLORS.faint : COLORS.text });
    name.position.set(startX - 3.2 - name.scale.x / 2, STREAM_Y, z);
    lanes.add(name);
  });
  live.add(tagStation("blocks", lanes));
}

// attention arcs above one block: from the newest lane to each earlier lane; thicker = it takes more from there
function attentionArcs(blockIndex) {
  const group = new THREE.Group();
  const windowTokens = attentionWindow();
  const weights = attentionWeights(windowTokens, blockIndex);
  const x = blockX(blockIndex), top = ATTN_Y + 1.25;
  const offset = LANES - windowTokens.length;
  weights.forEach((weight, k) => {
    if (k === windowTokens.length - 1 || weight < 0.02) return;
    const from = new THREE.Vector3(x, top, NEW_Z);
    const to = new THREE.Vector3(x, top, laneZ(k + offset));
    const height = 0.5 + Math.abs(NEW_Z - to.z) * 0.3;
    const middle = new THREE.Vector3(x, top + height * 2, (from.z + to.z) / 2);
    const curve = new THREE.QuadraticBezierCurve3(from, middle, to);
    group.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 32, 0.025 + 0.14 * weight, 8, false), glowMaterial(COLORS.attn, 0.25 + 0.75 * Math.min(1, weight * 2))));
    const drop = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, top - STREAM_Y, 4), glowMaterial(COLORS.attn, 0.1 + 0.6 * weight));
    drop.position.set(x, (top + STREAM_Y) / 2, to.z);      // where the arc lands, a thin line down to the lane it reads from
    group.add(drop);
  });
  live.add(tagStation("blocks", group));
  return group;
}

// dots flowing along the token lanes
function moveLaneFlow(seconds) {
  if (!flow.lanes) return;
  const attribute = flow.lanes.geometry.attributes.position;
  const startX = blockX(0) - 4.5, endX = blockX(15) + 3;
  const visible = lanes ? LANES : 0;
  flow.lanes.userData.offsets.forEach((offset, k) => {
    const u = (offset + seconds * 0.05) % 1;
    attribute.setXYZ(k, lerp(startX, endX, u), visible ? STREAM_Y : -999, laneZ(k % LANES));
  });
  attribute.needsUpdate = true;
}

// ---------------------------------------------------------------- boards: HTML panels standing in the scene
const boards = {};
function addBoards() {
  for (const s of STATIONS) {
    const anchor = document.createElement("div");
    anchor.className = "anchor";
    anchor.style.width = s.width + "px";
    anchor.style.height = ANCHOR_HEIGHT + "px";
    const board = document.createElement("div");
    board.className = "board";
    board.dataset.station = s.key;
    board.style.setProperty("--c", s.color);
    anchor.append(board);
    const object = new CSS3DObject(anchor);
    anchor.style.pointerEvents = "none";               // only the board itself is clickable
    object.scale.setScalar(PX);
    object.position.set(s.x, (s.top || 13) - (ANCHOR_HEIGHT * PX) / 2, 0);
    cssScene.add(object);
    boards[s.key] = board;
  }
}
const boardHead = (key, sub) => {
  const s = station(key);
  return '<div class="b-head"><span class="b-num">' + s.num + '</span><span class="b-title">' + s.name + '</span></div><div class="b-sub">' + sub + "</div>";
};
const label = (number, text) => '<div class="lbl"><span class="n">' + number + "</span>" + text + "</div>";
const reveal = (id) => $(id) && $(id).classList.add("in");

// the message as tokens: [{text, kind, id}]
const SPECIAL_IDS = { "<|user|>": 16380, "<|context|>": 16382, "<|assistant|>": 16381 };
function fakeId(text) {
  if (SPECIAL_IDS[text]) return SPECIAL_IDS[text];
  let hash = 7;
  for (const char of text) hash = (hash * 31 + char.codePointAt(0)) % 16000;
  return 260 + hash;
}
let tokens = [];
const KIND_CLASS = { special: "special", system: "sys", text: "text", ctx: "ctx", call: "call", answer: "hist", new: "new" };
function buildTokens() {
  if (demo.tokenList) return demo.tokenList.map((token) => ({ text: token.text, id: token.id, kind: KIND_CLASS[token.kind] || "text" }));
  const list = [{ text: "<|user|>", kind: "special" }];
  for (const text of demo.userTokens || []) list.push({ text, kind: "text" });
  if (demo.contextTokens) {
    list.push({ text: "<|context|>", kind: "special" });
    for (const text of demo.contextTokens) list.push({ text, kind: "ctx" });
  }
  list.push({ text: "<|assistant|>", kind: "special" });
  return list.map((token) => ({ ...token, id: fakeId(token.text) }));
}
// the numbers of an embedding row: real ones from the server (scaled to -1..1 for the colors), or made up offline
let vectorScale = 1;
function vectorValue(id, dim) {
  const row = demo.vectors && demo.vectors[String(id)];
  if (row) return Math.max(-1, Math.min(1, row[dim] / vectorScale));
  return Math.sin(id * 0.731 + dim * 2.17) * Math.cos(id * 0.13 + dim);
}
function vectorText(id, dim) {
  const row = demo.vectors && demo.vectors[String(id)];
  const value = row ? row[dim] : vectorValue(id, dim);
  const digits = row ? 3 : 2;
  return (value < 0 ? "−" : "") + Math.abs(value).toFixed(digits).replace(/^0(?=\.)/, row ? "" : "0");
}
function cellsHtml(id) {
  let html = "";
  for (let dim = 0; dim < 6; dim++) {
    const value = vectorValue(id, dim);
    const alpha = (0.15 + 0.75 * Math.abs(value)).toFixed(2);
    const color = value > 0 ? "rgba(255,157,110," + alpha + ")" : "rgba(169,165,255," + alpha + ")";
    html += '<span class="cell" style="background:' + color + '">' + vectorText(id, dim) + "</span>";
  }
  return html;
}

// ---- 01 router
function buildRouterBoard() {
  let message = "";
  if (demo.before) {
    message += '<div class="who">Earlier in this chat</div><div class="bubble prev">' + esc(demo.before.user) + '</div><div class="bubble prev">' + esc(demo.before.reply) + "</div>";
  }
  message += '<div class="who">Your message</div><div class="bubble">' + esc(demo.prompt) + "</div>";
  const checks = GATES.map((gate, index) =>
    '<div class="check" id="check-' + index + '"><span class="c-icon">' + gate.icon + '</span><div><div class="c-name">' + (index + 1) + ". " + gate.name +
    '</div><div class="c-ask">' + esc(gate.asks) + '</div><div class="c-why"></div></div><span class="c-state">waiting</span></div>').join("");
  boards.router.innerHTML = boardHead("router", "Plain code checks your message <b>before the model sees it</b>, one check at a time. The first check that says <b>yes</b> decides what happens next.") +
    '<div class="r-grid"><div>' + message + '</div><div class="checks">' + checks + "</div></div>" +
    '<div class="sec r-result" id="routerResult"></div>';
}
function setCheck(index, state, why) {
  const element = $("check-" + index);
  element.className = "check " + state;
  element.querySelector(".c-state").textContent = { checking: "checking…", yes: "✓ yes", no: "✕ no", skip: "not checked" }[state];
  if (why) element.querySelector(".c-why").textContent = "→ " + why;
}

// ---- 02 tool: shows what the chosen tool does, step by step
function inputBoxHtml() {
  const user = demo.tool.rewritten || demo.prompt;
  return '<div class="input-box"><span class="sp">&lt;|user|&gt;</span>' + esc(user) +
    '<span class="ins" id="insContext"><span class="sp">&lt;|context|&gt;</span>' + esc(demo.context || "") + '</span><span class="sp">&lt;|assistant|&gt;</span></div>';
}
function buildToolBoard() {
  if (demo.real) { buildRealToolBoard(); return; }
  const tool = demo.tool;
  const tab = TOOL_TABS.find((item) => item.key === tool.tab);
  boards.tool.style.setProperty("--c", tab.color);
  const tabs = '<div class="tabs">' + TOOL_TABS.map((item) => '<span class="tab" id="tab-' + item.key + '"><i>' + item.icon + "</i>" + item.name + "</span>").join("") + "</div>";
  let sub = "", body = "", right = "";
  if (tool.kind === "wiki") {
    sub = "<b>Wikipedia</b>: one request finds articles and their text. Code keeps only what matches the question.";
    const results = tool.results.map((result, index) => '<div class="res" id="res-' + index + '"><span class="m">' + (index + 1) + "</span><b>" + esc(result.title) + "</b><small></small></div>").join("");
    body += '<div class="sec" id="tSec1">' + label(1, "The router's call") + '<span class="callbox">' + esc(tool.call) + "</span></div>";
    body += '<div class="sec" id="tSec2">' + label(2, "Search results · the question's words must be in the article") + results + "</div>";
    if (tool.sentences) {
      const sentences = tool.sentences.map(([text, score], index) => '<div class="sent" id="sent-' + index + '"><span class="score"><i data-w="' + Math.round(score * 100) + '"></i></span><span>' + esc(text) + "</span></div>").join("");
      right += '<div class="sec" id="tSec3">' + label(3, "Sentences of “" + esc(tool.article) + "”, scored against the question") + sentences + "</div>";
      right += '<div class="sec" id="tSec4">' + label(4, '<span id="inputLabel">The model\'s input, before</span>') + inputBoxHtml() + "</div>";
    } else {
      right += '<div class="sec" id="tSec3"><div class="stop"><b>No article is about “Zorblax Quint”.</b> A factual question with no source gets a fixed reply instead of a guess:' +
        '<div class="fixed-reply" style="--c:var(--ban)">' + esc(demo.direct.text) + '</div><div class="note">The model is not run: steps 03–07 are skipped.</div></div></div>';
    }
  } else if (tool.kind === "calc") {
    sub = "<b>Calculator</b>: plain code does the math, so the result is always right. The model only puts it into a sentence.";
    body += '<div class="sec" id="tSec1">' + label(1, "The router's call") + '<span class="callbox">' + esc(tool.call) + "</span></div>";
    body += '<div class="sec" id="tSec2">' + label(2, "Python computes it exactly") + '<div class="big-eq">' + tool.left + " " + tool.op + " " + tool.right + ' = <b id="calcResult">' + tool.result + "</b></div></div>";
    right += '<div class="sec" id="tSec4">' + label(3, '<span id="inputLabel">The model\'s input, before</span>') + inputBoxHtml() +
      '<div class="note">A 126M model often gets multiplication wrong. Here it only has to repeat 391.</div></div>';
  } else if (tool.kind === "weather") {
    sub = "<b>Follow-up</b>: “And in Cluj?” makes no sense alone. Code repeats the <b>last tool</b> with the new place.";
    body += '<div class="sec" id="tSec1">' + label(1, "Earlier in this chat") +
      '<div class="chatline">You: <b>' + esc(demo.before.user) + "</b></div>" +
      '<div class="chatline">Tool call: <b class="callbox" style="font-size:18px;padding:2px 10px">' + esc(demo.before.call) + "</b></div>" +
      '<div class="chatline">Singularity: <b>' + esc(demo.before.reply) + "</b></div></div>";
    body += '<div class="sec" id="tSec2">' + label(2, "Now: “" + esc(demo.prompt) + "” → same tool, new place") +
      '<span class="callbox swap" id="swap">weather: <s>' + tool.oldPlace + "</s><b>" + tool.newPlace + "</b></span>" +
      '<div class="note">The question is also rewritten in full: <b style="color:var(--text)">“' + esc(tool.rewritten) + "”</b></div></div>";
    right += '<div class="sec" id="tSec3">' + label(3, "The weather tool's result") + '<div class="weather"><span class="deg">' + tool.temp + "</span><span>" + tool.newPlace + "<br><span style=\"color:var(--muted)\">" + tool.sky + "</span></span></div></div>";
    right += '<div class="sec" id="tSec4">' + label(4, '<span id="inputLabel">The model\'s input, before</span>') + inputBoxHtml() + "</div>";
  } else if (tool.kind === "identity") {
    sub = "<b>Identity file</b>: questions about the model are answered from a small file of true facts, so it can't invent things about itself.";
    const rows = tool.matches.map(([question, score], index) => '<div class="match" id="match-' + index + '"><span>' + esc(question) + '</span><span class="score"><i data-w="' + Math.round(score * 100) + '"></i></span><span class="v">' + score.toFixed(2) + "</span></div>").join("");
    body += '<div class="sec" id="tSec1">' + label(1, "How close is it to the questions in identity_qa.txt?") + rows + "</div>";
    right += '<div class="sec" id="tSec2">' + label(2, "Best match " + tool.matches[0][1] + " ≥ " + tool.threshold + " → its answer is used as it is") +
      '<div class="fixed-reply">' + esc(demo.direct.text) + '</div><div class="note">The model is not run: steps 03–07 are skipped.</div></div>';
  }
  boards.tool.innerHTML = boardHead("tool", sub) + tabs + '<div class="t-cols"><div>' + body + '</div><div>' + right + '</div></div>';
}

// ---- 03 tokens
function tokenHtml(token, index) {
  const lead = token.text.startsWith(" ");
  const body = lead ? token.text.slice(1) : token.text;
  return '<span class="tok ' + token.kind + '" id="tok-' + index + '" style="--h:' + ((index * 47) % 360) + '"><span class="tt">' +
    (lead ? '<span class="ws">␣</span>' : "") + esc(body) + '</span><span class="tid">' + token.id + "</span></span>";
}
// long runs (the system prompt, a long Wikipedia text) are folded into one chip, so the strip stays readable
function stripHtml() {
  let html = "";
  let index = 0;
  while (index < tokens.length) {
    const kind = tokens[index].kind;
    let end = index;
    while (end < tokens.length && tokens[end].kind === kind) end += 1;
    const size = end - index;
    const keepStart = kind === "sys" ? 0 : kind === "ctx" ? 10 : kind === "hist" ? 6 : size;
    const keepEnd = kind === "sys" ? 0 : 2;
    if (size > keepStart + keepEnd + 2) {
      for (let k = index; k < index + keepStart; k++) html += tokenHtml(tokens[k], k);
      const name = kind === "sys" ? "system prompt" : kind === "ctx" ? "more from the tool" : "earlier reply";
      html += '<span class="tok fold ' + kind + '"><span class="tt">… ' + name + '</span><span class="tid">' + (size - keepStart - keepEnd) + " tokens</span></span>";
      for (let k = end - keepEnd; k < end; k++) html += tokenHtml(tokens[k], k);
    } else {
      for (let k = index; k < end; k++) html += tokenHtml(tokens[k], k);
    }
    index = end;
  }
  return html;
}
function buildTokensBoard() {
  boards.tokens.innerHTML = boardHead("tokens", "The model can't read letters. The text is cut into <b>tokens</b>: pieces from a fixed list of 16,384. Every piece has a number, its <b>id</b>. The model only sees the ids.") +
    '<div class="sec in">' + label(1, '<span id="stripLabel">The exact text the model receives</span>') + '<div class="strip" id="strip">' + stripHtml() + "</div></div>" +
    '<div class="sec" id="tokLegend"><div class="legend"><span><i style="background:var(--faint)"></i>markers: who speaks</span><span><i style="background:var(--text)"></i>your message</span>' +
    '<span><i style="background:var(--wiki)"></i>from the tool</span><span><i style="background:var(--router)"></i>the tool call</span><span><i style="background:var(--accent)"></i>written by the model</span></div>' +
    '<div class="note" id="tokCount">' + tokens.length + " tokens · common pieces like “␣the” are one token, rare words are split into several · up to 1024 tokens fit</div></div>";
}
function appendToken(token) {
  tokens.push(token);
  const index = tokens.length - 1;
  $("strip").insertAdjacentHTML("beforeend", tokenHtml(token, index));
  requestAnimationFrame(() => $("tok-" + index) && $("tok-" + index).classList.add("cut"));
  $("tokCount").textContent = tokens.length + " tokens · the violet ones were written by the model, one per loop";
}

// ---- 04 embeddings: id -> row of the table -> vector
let embedShown = [];
function buildEmbedBoard() {
  embedShown = tokens.map((token, index) => ({ ...token, index })).filter((token) => token.kind === "text").slice(0, 5);
  const ids = [...new Set(embedShown.map((token) => token.id))].sort((a, b) => a - b);
  const list = embedShown.map((token, k) => '<div class="e-tok" id="etok-' + k + '"><span>' + esc(showTok(token.text)) + "</span><small>" + token.id + "</small></div>").join("");
  let table = '<div class="e-row"><span class="rid">row 0</span>' + cellsHtml(0) + '</div><div class="e-dots">⋮</div>';
  ids.forEach((id, k) => {
    table += '<div class="e-row" id="erow-' + id + '"><span class="rid">row ' + id + "</span>" + cellsHtml(id) + "</div>";
    table += k < ids.length - 1 ? '<div class="e-dots">⋮</div>' : "";
  });
  table += '<div class="e-dots">⋮</div><div class="e-row"><span class="rid">row 16383</span>' + cellsHtml(16383) + "</div>";
  const vectors = embedShown.map((token, k) => '<div class="e-row vec" id="evec-' + k + '"><span class="rid">' + esc(showTok(token.text)) + "</span>" + cellsHtml(token.id) + "</div>").join("");
  boards.embed.innerHTML = boardHead("embed", "Each token id is a <b>row number</b> in one big table: 16,384 rows × 768 numbers. That row is copied out. The list of numbers is the token's <b>vector</b>: what the model computes with.") +
    '<div class="e-grid"><div>' + label(1, "Token → id") + list + "</div>" +
    "<div>" + label(2, "The embedding table") + '<div class="e-table">' + table + "</div></div>" +
    "<div>" + label(3, "Copied rows = vectors") + vectors + "</div></div>" +
    '<div class="sec note" id="embedNote">+ ' + (tokens.length - embedShown.length) + " more tokens, same lookup → <b style=\"color:var(--text)\">" + tokens.length +
    " vectors</b> go into block 1. Only 6 of the 768 numbers are shown (orange = positive, violet = negative). The rows were learned in training: words used in similar ways get similar rows.</div>";
}

// ---- 05 inside a block
const NET = { x: [40, 180, 320], counts: [6, 12, 6] };
function buildBlocksBoard() {
  let net = "";
  const spot = (column, k) => [NET.x[column], 18 + (210 * (k + 0.5)) / NET.counts[column]];
  for (let column = 0; column < 2; column++) {
    for (let a = 0; a < NET.counts[column]; a++) {
      for (let b = 0; b < NET.counts[column + 1]; b++) {
        const [x1, y1] = spot(column, a), [x2, y2] = spot(column + 1, b);
        net += '<line class="w' + (column + 1) + '" x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '"/>';
      }
    }
  }
  for (let column = 0; column < 3; column++) {
    for (let k = 0; k < NET.counts[column]; k++) {
      const [x, y] = spot(column, k);
      net += '<circle class="n' + column + '" cx="' + x + '" cy="' + y + '" r="' + (column === 1 ? 6 : 8) + '"/>';
    }
  }
  net += '<text x="40" y="256" text-anchor="middle">768</text><text x="180" y="256" text-anchor="middle">3072</text><text x="320" y="256" text-anchor="middle">768</text>';
  let lens = "";
  for (let index = 0; index < 16; index++) lens += '<div id="lens-' + index + '">' + (index + 1) + "<b></b></div>";
  boards.blocks.innerHTML = boardHead("blocks", "Each block <b>adds</b> something to the token vectors (the white line below). <b>Attention</b>: the newest token looks back at earlier tokens. <b>MLP</b>: it thinks about itself alone. The same two steps, 16 times, each block with its own weights.") +
    '<div class="k-head">Inside block <b id="kIndex">1</b> of 16 · for the newest token <b id="kNewest"></b></div>' +
    '<div class="k-grid"><div class="k-part attn" id="kAttn"><div class="k-title attn">Attention <small>· 12 heads</small></div><div class="k-desc">Thicker arc = the newest token takes more from there</div>' +
    '<svg id="kArcs" viewBox="0 0 560 200" width="560" height="200"></svg><div class="k-desc">Each of the 12 heads (rows) looks at different tokens:</div><svg id="kHeat" viewBox="0 0 560 110" width="560" height="110"></svg></div>' +
    '<div class="k-part mlp" id="kMlp"><div class="k-title mlp">MLP</div><div class="k-desc">768 numbers → 3072 → 768, then added back to the vector</div><svg id="kNet" viewBox="0 0 360 270" width="360" height="270">' + net + "</svg></div></div>" +
    '<div class="sec in">' + '<div class="lbl" style="margin-top:4px">If the model stopped after each block, it would say… (logit lens)</div><div class="lens">' + lens + "</div></div>";
}
function attentionWindow() { return tokens.slice(demo.real ? -LANES : -10); }
// where the newest token looks in one block: the real average of the 12 heads (first step), or made up offline
function attentionWeights(windowTokens, blockIndex) {
  const detail = demo.stepDetail && demo.stepDetail.blocks && demo.stepDetail.blocks[blockIndex];
  if (demo.real && detail) {
    const heads = detail.heads;
    const size = windowTokens.length;
    const mean = new Array(size).fill(0);
    for (const row of heads) {
      const shift = size - row.length;
      row.forEach((value, k) => { if (k + shift >= 0) mean[k + shift] += value / heads.length; });
    }
    const total = mean.reduce((sum, value) => sum + value, 0) || 1;
    return mean.map((value) => value / total);
  }
  if (demo.real) return windowTokens.map(() => 1 / windowTokens.length);
  const focusText = demo.focus[blockIndex < 6 ? 0 : 1];
  let focus = -1;
  windowTokens.forEach((token, index) => { if (token.text === focusText && index < windowTokens.length - 1) focus = index; });
  const raw = windowTokens.map((token, index) => 0.25 + Math.abs(Math.sin(index * 7.13 + blockIndex * 0.7)) + (index === focus ? 4 : 0));
  const total = raw.reduce((sum, value) => sum + value, 0);
  return raw.map((value) => value / total);
}
function drawAttention(blockIndex) {
  const windowTokens = attentionWindow();
  const weights = attentionWeights(windowTokens, blockIndex);
  const width = 560 / Math.max(windowTokens.length, 1), base = 158, newest = windowTokens.length - 1;
  const xNew = newest * width + width / 2;
  const ranked = weights.map((w, i) => [w, i]).filter(([, i]) => i !== newest).sort((a, b) => b[0] - a[0]);
  let svg = "";
  weights.forEach((weight, index) => {
    if (index === newest) return;
    const x = index * width + width / 2;
    const height = 16 + (xNew - x) * 0.27;
    svg += '<path d="M' + xNew + "," + base + " Q" + (xNew + x) / 2 + "," + (base - height * 2) + " " + x + "," + base + '" fill="none" stroke="#a9a5ff" stroke-linecap="round" stroke-width="' +
      (1.5 + 13 * weight).toFixed(1) + '" stroke-opacity="' + (0.2 + 0.8 * Math.min(1, weight * 2)).toFixed(2) + '"/>';
  });
  ranked.slice(0, 2).forEach(([weight, index]) => {
    const x = index * width + width / 2;
    const height = 16 + (xNew - x) * 0.27;
    svg += '<text x="' + (xNew + x) / 2 + '" y="' + (base - height - 8) + '" text-anchor="middle" fill="#ecebf7" font-size="15" font-family="JetBrains Mono">' + Math.round(weight * 100) + "%</text>";
  });
  windowTokens.forEach((token, index) => {
    const x = index * width;
    const isNew = index === newest;
    const text = showTok(token.text).replace("<|assistant|>", "<|as|>").replace("<|context|>", "<|ctx|>").replace("<|user|>", "<|usr|>").replace("<|endoftext|>", "<|end|>").replace("<|search|>", "<|srch|>");
    const short = width < 50 ? 5 : 8;
    svg += '<rect x="' + (x + 2) + '" y="' + (base + 6) + '" width="' + (width - 4) + '" height="32" rx="7" fill="' + (isNew ? "rgba(201,197,255,.22)" : "rgba(255,255,255,.06)") + '" stroke="' + (isNew ? "#c9c5ff" : "none") + '" stroke-width="2"/>';
    svg += '<text x="' + (x + width / 2) + '" y="' + (base + 27) + '" text-anchor="middle" fill="' + (token.kind === "special" ? "#8a88a8" : "#ecebf7") + '" font-size="' + (text.length > 5 ? 10 : 13) + '" font-family="JetBrains Mono">' + esc(text.length > short ? text.slice(0, short - 1) + "…" : text) + "</text>";
  });
  $("kArcs").innerHTML = svg;
  let heat = "";
  const detail = demo.stepDetail && demo.stepDetail.blocks && demo.stepDetail.blocks[blockIndex];
  for (let head = 0; head < 12; head++) {
    let raw = windowTokens.map((token, index) => Math.pow(Math.abs(Math.sin(index * 3.3 + head * 1.9 + blockIndex)), 3) + (head % 3 === 0 ? weights[index] * 4 : 0));
    if (demo.real && detail) {                                   // the real row of this head, aligned to the window's end
      const row = detail.heads[head] || [];
      raw = windowTokens.map((token, index) => row[index - (windowTokens.length - row.length)] || 0);
    }
    const max = Math.max(...raw) || 1;
    raw.forEach((value, index) => {
      heat += '<rect x="' + (index * width + 2) + '" y="' + head * 9 + '" width="' + (width - 4) + '" height="7" rx="2" fill="#a9a5ff" fill-opacity="' + (0.06 + 0.9 * value / max).toFixed(2) + '"/>';
    });
  }
  $("kHeat").innerHTML = heat;
  const newestToken = windowTokens[newest];
  $("kNewest").textContent = "“" + showTok(newestToken.text) + "”";
}
function mlpPattern(blockIndex) {
  document.querySelectorAll("#kNet .n1").forEach((node, k) => node.classList.toggle("hot", Math.sin(k * 2.7 + blockIndex * 1.3) > 0.1));
}

// ---- 06 scores
function buildLogitsBoard() {
  boards.logits.innerHTML = boardHead("logits", "The newest token's final vector gets a <b>score for each of the 16,384 tokens</b>. Softmax turns the scores into chances that add up to 100%. The top guesses:") +
    '<div id="bars"></div><div class="rest" id="barsRest"></div><div class="sec" id="guard"></div>';
}
function renderBars(step) {
  const top = demo.generated[step].top;
  const max = top[0][1];
  $("bars").innerHTML = top.map(([token, prob], index) =>
    '<div class="bar-row" id="bar-' + index + '" style="--bc:' + RANK_COLORS[index] + '"><span class="dot"></span><span class="tk">' + esc(showTok(token)) +
    '</span><span class="bar"><i data-w="' + (prob / max) * 100 + '"></i></span><span class="pc">' + Math.round(prob * 100) + "%</span></div>").join("");
  const shown = top.reduce((sum, item) => sum + item[1], 0);
  $("barsRest").textContent = "the other " + (16384 - top.length).toLocaleString("en") + " tokens share the last " + Math.max(1, Math.round((1 - shown) * 100)) + "%";
  requestAnimationFrame(() => requestAnimationFrame(() => document.querySelectorAll("#bars i").forEach((bar) => { bar.style.width = bar.dataset.w + "%"; })));
}

// ---- 07 pick, 08 answer
function buildSamplingBoard() {
  boards.sampling.innerHTML = boardHead("sampling", "A <b>weighted draw</b> from the chances: each slice is as big as its chance. Big slices win often, but not always. Temperature 0.7 keeps it fairly safe.") +
    '<div class="pick" id="pick">waiting for the chances…</div>';
}
function buildAnswerBoard() {
  boards.answer.innerHTML = boardHead("answer", "Singularity writes <b>one token at a time</b>. Each new token is also added to its own input (back to 03) before the next one is chosen.") +
    '<div class="ans" id="ansText"><span class="cur" id="ansCursor"></span></div>' +
    '<div class="sec legend" id="ansLegend"><span><i style="background:var(--text)"></i>written by the model</span><span><i style="background:var(--ground)"></i>copied from Wikipedia</span></div>';
}
function answer(text, kind) {
  if (text.includes("<|")) return;                  // <|endoftext|>: the model chose to stop
  const span = document.createElement("span");
  span.className = kind === "grounded" ? "g" : kind === "fixed" ? "" : "t-new";
  span.textContent = text;
  $("ansText").insertBefore(span, $("ansCursor"));
}

// ---------------------------------------------------------------- the HUD (HTML on top of the scene)
const stageButtons = {};
function buildStageList() {
  for (const s of STATIONS) {
    const button = document.createElement("button");
    button.className = "stage";
    button.style.setProperty("--c", s.color);
    button.innerHTML = '<span class="num">' + s.num + "</span><span>" + s.name + "</span>";
    button.addEventListener("click", () => { flyTo(stationCamera(s.key), 1400, null, true); showInfo(infoKeyFor(s.key)); });
    $("stages").append(button);
    stageButtons[s.key] = button;
  }
}
function buildScenarioChips() {
  for (const key in SCENARIOS) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip glass";
    chip.dataset.key = key;
    chip.innerHTML = "<i>" + SCENARIOS[key].chip + "</i>" + SCENARIOS[key].prompt;
    chip.addEventListener("click", () => run(key));
    chip.title = key === "followup" ? "Runs “" + SCENARIOS[key].before.user + "” first, then the follow-up" : "Run it on the real model";
    $("scenarios").append(chip);
  }
}

// the info panel: folded to its title by default; "More" opens the longer text from content.js
let infoOpen = false;
let shownKey = "router";
function infoKeyFor(stationKey) {
  if (stationKey === "tool") {
    if (demo.tool.kind === "wiki") return demo.direct ? "abstain" : "wikipedia";
    if (demo.tool.kind === "none") return "search";
    if (demo.tool.kind === "who") return "abstain";
    return demo.tool.kind === "identity" ? "identity" : "tools";
  }
  return { blocks: "block", embed: "embeddings" }[stationKey] || stationKey;
}
function showInfo(key) {
  shownKey = key;
  const text = EXPLORE_TEXT[key] || { title: key, short: "", long: "", code: "" };
  const color = key === "mlp" ? COLORS.mlp : key === "attention" || key === "block" ? COLORS.attn : key === "answer" || key === "grounded" ? COLORS.ground : COLORS.accent;
  const info = $("info");
  info.style.setProperty("--c", color);
  info.classList.toggle("open", infoOpen);
  info.innerHTML = '<button class="info-head" id="infoToggle"><span><span class="kicker">About this step</span><h2>' + text.title + '</h2></span><span class="tog">' + (infoOpen ? "Less" : "More") + "</span></button>" +
    '<div class="body"><p>' + text.short + '</p><p class="more">' + text.long + '</p><div class="code">' + (text.code || "") + "</div></div>";
  $("infoToggle").addEventListener("click", () => { infoOpen = !infoOpen; showInfo(shownKey); });
}

// one plain sentence at the top: what is happening right now
let captionCount = 0;
function caption(html) {
  captionCount += 1;
  $("caption").hidden = false;
  $("captionStep").textContent = String(captionCount).padStart(2, "0");
  $("captionText").innerHTML = html;
}

function enterStation(key, moveCamera = true) {
  for (const s of STATIONS) {
    boards[s.key].classList.toggle("active", s.key === key);
    const button = stageButtons[s.key];
    if (button.classList.contains("now")) { button.classList.remove("now"); button.classList.add("done"); }
  }
  stageButtons[key].classList.add("now");
  if (moveCamera) flyTo(stationCamera(key));
  showInfo(infoKeyFor(key));
}
function skipStation(key) {
  stageButtons[key].classList.add("skipped");
  boards[key].classList.add("skipped");
  dimStation(key, true);
}

// ---------------------------------------------------------------- packets: what travels on the rails
const live = new THREE.Group();     // everything made for the current message; cleared on every new run
scene.add(live);
function makePacket(text, color = "#ffffff") {
  const group = new THREE.Group();
  group.add(new THREE.Mesh(new THREE.SphereGeometry(0.45, 24, 24), glowMaterial("#ffffff", 0.95)));
  group.add(new THREE.Mesh(new THREE.SphereGeometry(0.95, 24, 24), glowMaterial(color, 0.3)));
  if (text) {
    const tag = makeLabel(text, { size: 1.05, mono: true, bg: "rgba(20,19,34,0.92)", border: color, color: COLORS.text });
    tag.position.y = 1.7;
    group.add(tag);
  }
  live.add(group);
  return group;
}
function removePacket(packet) { live.remove(packet); disposeGroup(packet); }

// along the main rail from one station to the next; the camera follows through a wider view
async function travel(fromKey, toKey, text, duration = 1700, follow = true) {
  const from = station(fromKey).x, to = station(toKey).x;
  if (follow) {
    const next = station(toKey);
    caption("On the rail: <b>" + esc(text) + "</b> → step " + next.num + " · " + next.name);
    const middle = (from + to) / 2, span = Math.abs(to - from);
    flyTo(stationCamera(toKey), duration + 300, [[middle, -2, span * 0.7 + 30], [middle, RAIL_Y + 6, 0]]);
  }
  const packet = makePacket(text);
  await animate(duration, (t) => packet.position.set(lerp(from, to, ease(t)), RAIL_Y + 0.5, 0));
  removePacket(packet);
}
async function travelCurve(curve, text, duration, color) {
  const packet = makePacket(text, color);
  await animate(duration, (t) => packet.position.copy(curve.getPointAt(ease(t))).add(new THREE.Vector3(0, 0.5, 0)));
  removePacket(packet);
}
const loopBack = { getPointAt: (t) => loopCurve.getPointAt(t) };      // the loop curve already runs from 07 back to 03

// ---------------------------------------------------------------- one run: what happens to one message
function pickScenario(text) {
  const lower = text.toLowerCase();
  if (/[0-9]\s*[*x×+\-/]\s*[0-9]/.test(lower)) return "calc";
  if (/^(and|but|și|dar)\b/.test(lower)) return "followup";
  if (/\b(you|your)\b/.test(lower)) return "identity";
  if (/zorblax/.test(lower)) return "abstain";
  return "wiki";
}

function resetRun() {
  clock.run += 1;
  for (const chip of document.querySelectorAll(".chip")) chip.classList.toggle("on", chip.dataset.key === demo.chipKey);
  disposeGroup(live);
  wheel = null;
  lanes = null;
  resetBlocks();
  tokens = buildTokens();
  vectorScale = 1;
  if (demo.vectors) {
    const values = Object.values(demo.vectors).flat().map(Math.abs).sort((a, b) => a - b);
    vectorScale = values[Math.floor(values.length * 0.95)] || 1;          // 95% of the numbers fit the color scale
  }
  captionCount = 0;
  evidence.visible = false;
  for (const s of STATIONS) {
    stageButtons[s.key].className = "stage";
    boards[s.key].className = "board";
    dimStation(s.key, false);
  }
  buildRouterBoard();
  buildToolBoard();
  buildTokensBoard();
  buildEmbedBoard();
  buildBlocksBoard();
  buildLogitsBoard();
  buildSamplingBoard();
  buildAnswerBoard();
  $("hint").style.opacity = 0;
}

// ---------------------------------------------------------------- real runs: POST /api/explore (alexgpt/explore.py)
let exploreHistory = [];          // Explore's last exchange, so a typed follow-up ("and in Cluj?") works
let lastExchange = null;          // {user, call, reply} of that exchange, shown as "Earlier in this chat"
let loading = 0;                  // a ticket: only the newest message may start playing
const TAB_FOR_TOOL = { wiki: "wiki", calc: "calc", convert: "convert", now: "now", today_in_history: "now", weather: "weather",
                       dice: "dice", coin: "dice", pick: "dice", memory_save: "memory", memory_get: "memory" };

async function fetchRun(prompt, history) {
  const response = await fetch("/api/explore", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, history, temperature: 0.7 }),
  });
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || "HTTP " + response.status);
  return data;
}
const historyOf = (record) => [{ role: "user", text: record.prompt },
  { role: "assistant", text: record.direct ? record.direct.text : (record.answer + (record.grounded ? " " + record.grounded.trim() : "")) }];
const exchangeOf = (record) => ({ user: record.prompt, call: record.tools.length ? record.tools[0].call : "",
  reply: record.direct ? record.direct.text : record.answer });

// one generated token of the record -> the scene's shape: bars = the model's own chances (fact_guard bans marked),
// after = its chances once the bans are applied, wheel = what is left after temperature / top-p
function stepToGenerated(step) {
  const banned = step.banned || [];
  const rows = (step.before_guard || step.model_top || step.top || []).slice(0, 6).map(([token, prob]) => [token, prob, banned.includes(token) ? "ban" : undefined]);
  let after = null;
  if (step.before_guard) {
    after = {};
    for (const [token, prob] of step.model_top || []) after[token] = prob;
  }
  return { token: step.token, top: rows, after, wheel: step.top && step.top.length ? step.top : null };
}

function fromRecord(record, before, chipKey) {
  const gates = {};
  for (const gate of record.gates) gates[gate.gate] = [gate.hit ? "yes" : "no", gate.detail];
  const used = record.tools.find((item) => !item.unused) || record.tools[0] || null;
  let tool;
  if (record.direct && record.direct.kind === "identity") {
    tool = { kind: "identity", tab: "identity", matches: record.direct.matches || [], threshold: 0.35 };
  } else if (!used) {
    tool = { kind: record.direct ? "who" : "none", tab: null, call: "" };
  } else {
    const isFollowup = gates.followup && gates.followup[0] === "yes" && before;
    const kind = used.tool === "wiki" ? "wiki" : isFollowup ? "followup" : used.tool === "calc" ? "calc" : "generic";
    tool = { kind, tab: TAB_FOR_TOOL[used.tool] || null, call: used.call, result: used.result, unused: used.unused, search: used.search,
             others: record.tools.filter((item) => item !== used).map((item) => item.call), fromModel: Boolean(gates.search && gates.search[0] === "yes") };
  }
  const steps = record.steps || [];
  const result = {
    real: true, chipKey, prompt: record.prompt, before, gates, tool,
    context: used && !used.unused ? used.result : "", rewritten: record.rewritten,
    direct: record.direct, tokenList: record.tokens, vectors: record.vectors,
    generated: steps.slice(0, 3).map(stepToGenerated), stepDetail: steps[0] || null,
    lens: steps[0] && steps[0].lens ? steps[0].lens : new Array(16).fill(""),
    rest: steps.slice(3).map((item) => item.token).filter((token) => !token.includes("<|")).join(""),
    grounded: (record.grounded || "").trim(), source: record.source ? record.source.title : null,
  };
  if (!result.direct && !result.generated.length) result.direct = { kind: "abstain", text: "(the model wrote nothing)" };
  return result;
}

// how strongly each of the 12 heads looks at one of the last tokens (its biggest weight there), 0..1
function headStrengths(blockIndex) {
  const detail = demo.stepDetail && demo.stepDetail.blocks && demo.stepDetail.blocks[blockIndex];
  if (!demo.real || !detail) return HEAD_COLORS.map((_, head) => Math.abs(Math.sin(head * 1.7 + blockIndex * 0.9)));
  const peaks = detail.heads.map((row) => Math.max(0, ...row));
  const max = Math.max(...peaks) || 1;
  return peaks.map((value) => value / max);
}

// asks the server; while it works the scene waits. No server (or no GPU): the offline demo with made-up numbers.
async function prepareRun(key) {
  const chip = key ? SCENARIOS[key] : null;
  const prompt = chip ? chip.prompt : $("promptInput").value.trim();
  if (!prompt) return false;
  $("promptInput").value = "";                            // like the chat: the box empties once the message is sent
  $("hint").classList.add("gone");
  const ticket = ++loading;
  clock.run += 1;                                         // stop the animation that is playing
  for (const element of document.querySelectorAll(".chip")) element.classList.toggle("on", element.dataset.key === key);
  const reading = (text) => caption("Singularity is reading <b>“" + esc(text) + "”</b> on the GPU… (the real model; every number is recorded)");
  reading(prompt);
  flyTo(stationCamera("router"), 1200);
  let record = null, before = null;
  try {
    let history = chip ? [] : exploreHistory;
    before = chip ? null : lastExchange;
    if (chip && chip.before) {                            // the follow-up demo needs its earlier question first
      reading(chip.before.user);
      const first = await fetchRun(chip.before.user, []);
      before = exchangeOf(first);
      history = historyOf(first);
      reading(prompt);
    }
    record = await fetchRun(prompt, history);
  } catch (error) {
    console.warn("Explore: no live model, playing the offline demo.", error);
  }
  if (ticket !== loading) return false;                   // another message was sent meanwhile
  if (record) {
    demo = fromRecord(record, before, key);
    exploreHistory = historyOf(record);
    lastExchange = exchangeOf(record);
    $("dataMode").textContent = "";
  } else {
    const fallback = key || pickScenario(prompt);
    demo = { ...SCENARIOS[fallback], chipKey: fallback };
    $("dataMode").textContent = "offline demo";
  }
  return true;
}

// running: the ▶ button becomes ■ (stop).  Stop = the same "cancel" a new message uses (clock.run changes).
let running = false;
function setRunning(on) {
  running = on;
  $("runButton").textContent = on ? "■" : "▶";
  $("runButton").title = on ? "Stop" : "Run";
  $("runButton").classList.toggle("stop", on);
}
function stopRun() {
  loading += 1;                                           // a reply still on its way is ignored
  clock.run += 1;                                         // every animation that is playing ends
  if (clock.paused) { clock.paused = false; $("pauseButton").textContent = "❚❚"; }
  setRunning(false);
  caption("Stopped. Type another message, or pick one above.");
}

async function run(key) {
  const ticket = loading + 1;
  setRunning(true);
  const ready = await prepareRun(key);
  if (!ready) { if (loading <= ticket) setRunning(false); return; }
  resetRun();
  try {
    await routerPhase();
    await travel("router", "tool", demo.tool.call || (demo.tool.kind === "identity" ? "identity check" : "no tool"));
    await toolPhase();
    if (demo.direct) { await directPhase(); finish(); return; }
    await travel("tool", "tokens", "the model's input (text)");
    await tokensPhase();
    await travel("tokens", "embed", tokens.length + " token ids");
    await embedPhase();
    await travel("embed", "blocks", tokens.length + " vectors × 768");
    await blocksPhase();
    await travel("blocks", "logits", "the last token's vector");
    await logitsPhase(0);
    await travel("logits", "sampling", "chances", 1300);
    await samplingPhase(0);
    await loopPhase(0);
    for (let step = 1; step < demo.generated.length; step++) await quickStep(step);
    if (demo.rest) await fastForward();
    if (demo.grounded) await groundedPhase();
    finish();
  } catch (error) {
    if (error !== CANCELLED) throw error;
  }
}

function finish() {
  setRunning(false);
  caption("Done. <b>Click any board</b> to read about it, or ask something else below.");
}

// 01: the checks light up one by one: green = yes, red = no; after a yes, the rest are not checked
async function routerPhase() {
  enterStation("router");
  caption(demo.before
    ? "Earlier in this chat: “" + esc(demo.before.user) + "”. Now: <b>“" + esc(demo.prompt) + "”</b>. The <b>router</b> checks it first."
    : "Your message reaches the <b>router</b>: plain code that checks, in order, whether a tool should help.");
  await wait(demo.before ? 2600 : 1600);
  let hit = -1;
  for (let index = 0; index < GATES.length; index++) {
    const gate = GATES[index];
    const verdict = demo.gates[gate.key];
    if (!verdict) { setCheck(index, "skip"); continue; }        // the code went another way before this check
    setCheck(index, "checking");
    caption("Check " + (index + 1) + " · <b>" + gate.name + "</b> " + esc(gate.asks));
    await wait(1000);
    const yes = verdict[0] === "yes";
    setCheck(index, yes ? "yes" : "no", verdict[1]);
    caption("Check " + (index + 1) + " · <b>" + gate.name + "</b> → " + (yes ? "<b>yes</b>: " : "no: ") + esc(verdict[1]));
    await wait(yes ? 1600 : 1100);
    if (yes) { hit = index; break; }
  }
  if (hit >= 0) for (let index = hit + 1; index < GATES.length; index++) setCheck(index, "skip");
  const tab = TOOL_TABS.find((item) => item.key === demo.tool.tab);
  if (tab) {
    $("routerResult").innerHTML = "→ Next: <b>" + tab.name + "</b>" + (demo.tool.call ? ' · <span class="callbox" style="font-size:18px;padding:3px 12px">' + esc(demo.tool.call) + "</span>" : "");
    caption("A check said <b>yes</b>, so the checks after it are not needed. The message goes to the <b>" + tab.name.toLowerCase() + "</b>.");
  } else if (demo.tool.kind === "who") {
    $("routerResult").innerHTML = "→ Nobody to search for: Singularity asks who you mean.";
    caption("A question about “he / she / it”, but nobody was named: the code asks instead of searching.");
  } else {
    $("routerResult").innerHTML = "→ No tool: the model answers from what it learned.";
    caption("No check said yes and the model did not ask for a search: <b>no tool</b>. The model answers on its own.");
  }
  reveal("routerResult");
  await wait(1700);
}

// 02: the chosen tool, step by step
async function toolPhase() {
  if (demo.real) { await realToolPhase(); return; }
  enterStation("tool");
  const tool = demo.tool;
  $("tab-" + tool.tab).classList.add("on");
  if (tool.kind === "wiki") {
    caption("<b>Wikipedia</b> gets the call. One request returns the top articles and their text.");
    reveal("tSec1");
    await wait(1300);
    reveal("tSec2");
    for (let index = 0; index < tool.results.length; index++) {
      const result = tool.results[index];
      const row = $("res-" + index);
      row.classList.add(result.ok ? "ok" : "bad");
      row.querySelector(".m").textContent = result.ok ? "✓" : "✕";
      row.querySelector("small").textContent = result.why;
      caption((result.ok ? "✓ kept: " : "✕ rejected: ") + "<b>" + esc(result.title) + "</b> (" + esc(result.why) + ")");
      await wait(result.ok ? 1000 : 750);
    }
    if (!tool.sentences) {
      reveal("tSec3");
      caption("Nothing relevant was found. A factual question with no source gets a <b>fixed reply</b>: no guessing.");
      await wait(3000);
      return;
    }
    reveal("tSec3");
    caption("The sentences of the best article are <b>scored</b> by the question's words and by the kind of answer (“who” wants a name).");
    await wait(400);
    document.querySelectorAll("#tSec3 .score i").forEach((bar) => { bar.style.width = bar.dataset.w + "%"; });
    await wait(1600);
    $("sent-" + tool.best).classList.add("best");
    caption("The best sentence (score " + tool.sentences[tool.best][1] + ") will be given to the model.");
    await wait(1800);
    await insertContext("The best sentence is added to the model's input as <b>&lt;|context|&gt;</b>. The model reads it like any other text.");
  } else if (tool.kind === "calc") {
    caption("The <b>calculator</b> gets the call <b>" + esc(tool.call) + "</b>.");
    reveal("tSec1");
    await wait(1300);
    reveal("tSec2");
    caption("Python computes it exactly. No model involved, so it can't be wrong.");
    await wait(900);
    $("calcResult").classList.add("in");
    await wait(1800);
    await insertContext("The result is added to the model's input as <b>&lt;|context|&gt;</b>. The model only has to put it into a sentence.");
  } else if (tool.kind === "weather") {
    caption("Earlier in this chat the <b>weather</b> tool ran for Bucharest. That call was remembered.");
    reveal("tSec1");
    await wait(2400);
    reveal("tSec2");
    caption("“And in Cluj?” is a <b>follow-up</b>: same tool as before, only the place changes.");
    await wait(1200);
    $("swap").classList.add("done");
    await wait(2200);
    reveal("tSec3");
    caption("The weather tool runs for <b>Cluj</b>.");
    await wait(1800);
    await insertContext("The rewritten question and the result go into the model's input. Now the model knows what “And in Cluj?” means.");
  } else if (tool.kind === "identity") {
    caption("A question about the model itself. It is compared with the questions in the <b>identity file</b>.");
    reveal("tSec1");
    await wait(400);
    document.querySelectorAll("#tSec1 .score i").forEach((bar) => { bar.style.width = bar.dataset.w + "%"; });
    await wait(1600);
    $("match-0").classList.add("best");
    reveal("tSec2");
    caption("Close enough (0.91). Its <b>true answer</b> is used as it is, and the model is not run.");
    await wait(2800);
  }
}
async function insertContext(text) {
  reveal("tSec4");
  caption("This is the text the model will get. <b>Before</b>: just your message.");
  await wait(1800);
  $("insContext").classList.add("in");
  $("inputLabel").textContent = "The model's input, after: the tool's result was added";
  caption(text);
  await wait(2600);
}

// ---- 02 tool, for a REAL run: what the tool really did (demo.tool comes from fromRecord)
const clip = (text, size) => { text = String(text || ""); return text.length > size ? text.slice(0, size - 1) + "…" : text; };
function realInputHtml(withTool) {
  const user = demo.rewritten || demo.prompt;
  const call = String(demo.tool.call || "").replace(/^wiki:\s*(?:ro|en):\s*/, "wiki: ");     // the model was trained on "wiki: X"
  let html = '<div class="input-box"><span class="sp">&lt;|system|&gt;… &lt;|user|&gt;</span>' + esc(user);
  if (withTool) {
    html += '<span class="ins" id="insContext"><span class="sp">&lt;|assistant|&gt;&lt;|search|&gt;</span>' + esc(call) +
      '<span class="sp">&lt;|endoftext|&gt;&lt;|context|&gt;</span>' + esc(clip(demo.context, 240)) + "</span>";
  }
  return html + '<span class="sp">&lt;|assistant|&gt;</span></div>';
}
function buildRealToolBoard() {
  const tool = demo.tool;
  const tab = TOOL_TABS.find((item) => item.key === tool.tab);
  boards.tool.style.setProperty("--c", tab ? tab.color : COLORS.muted);
  const tabs = '<div class="tabs">' + TOOL_TABS.map((item) => '<span class="tab" id="tab-' + item.key + '"><i>' + item.icon + "</i>" + item.name + "</span>").join("") + "</div>";
  const others = tool.others && tool.others.length ? '<div class="note">+ also: ' + tool.others.map(esc).join(" · ") + "</div>" : "";
  const callBox = (number) => '<div class="sec" id="tSec1">' + label(number, tool.fromModel ? "The call the model wrote itself" : "The router's call") +
    '<span class="callbox">' + esc(tool.call) + "</span>" + others + "</div>";
  const inputSection = (number, withTool, extra) => '<div class="sec" id="tSec4">' + label(number, '<span id="inputLabel">The model\'s input, before</span>') +
    realInputHtml(withTool) + (extra || "") + "</div>";
  let sub = "", left = "", right = "";
  if (tool.kind === "wiki") {
    const search = tool.search || { results: [], sentences: [], names: [] };
    sub = "<b>Wikipedia</b>: one request returns up to 5 articles. An article is allowed only if it contains every name in your question; the best one is picked and its sentences are scored.";
    left += callBox(1);
    const names = search.names && search.names.length ? "must contain “" + search.names.map(esc).join("”, “") + "”" : "your question names nobody, so all are allowed";
    const rows = (search.results || []).map((result, index) => '<div class="res" id="res-' + index + '"><span class="m">' + (index + 1) + "</span><b>" + esc(result.title) + "</b><small></small></div>").join("");
    left += '<div class="sec" id="tSec2">' + label(2, "Search results · " + names) + (rows || '<div class="note">Wikipedia returned no article for this.</div>') +
      (search.fact ? '<div class="weather" style="margin-top:12px;font-size:19px"><span class="deg" style="font-size:26px">WD</span><span><b>Wikidata</b> (the exact fact, put first): ' +
        esc(search.fact.sentence) + "</span></div>" : "") + "</div>";
    if (search.sentences && search.sentences.length) {
      const best = Math.max(1, ...search.sentences.map((item) => item[1]));
      const sentences = search.sentences.map(([text, points], index) => '<div class="sent" id="sent-' + index + '"><span class="score"><i data-w="' + Math.round((points / best) * 100) +
        '"></i></span><span>' + esc(clip(text, 170)) + "</span></div>").join("");
      right += '<div class="sec" id="tSec3">' + label(3, "Sentences of “" + esc(search.chosen || "") + "”, scored against the question") + sentences + "</div>";
    }
    if (demo.direct) {
      right += '<div class="sec" id="tSec4"><div class="stop"><b>No usable article.</b> A factual question with no source gets a fixed reply instead of a guess:' +
        '<div class="fixed-reply" style="--c:var(--ban)">' + esc(demo.direct.text) + '</div><div class="note">The model is not run: steps 03–07 are skipped.</div></div></div>';
    } else if (tool.unused) {
      right += '<div class="sec" id="tSec4">' + label(4, "Not used") + '<div class="stop">The article does not match the question well enough, so it is <b>not used</b>. The model answers on its own.</div>' + realInputHtml(false) + "</div>";
    } else {
      right += inputSection(4, true);
    }
  } else if (tool.kind === "calc") {
    sub = "<b>Calculator</b>: plain code does the math, so the result is always right. The model only puts it into a sentence.";
    const expression = tool.call.replace(/^[^:]*:\s*/, "");
    const size = expression.length + String(tool.result).length > 16 ? 36 : 56;
    left += callBox(1);
    left += '<div class="sec" id="tSec2">' + label(2, "Python computes it exactly") + '<div class="big-eq" style="font-size:' + size + 'px">' + esc(expression) + ' = <b id="calcResult">' + esc(clip(tool.result, 30)) + "</b></div></div>";
    right += inputSection(3, true, '<div class="note">A 126M model often gets arithmetic wrong. Here it only has to repeat the result.</div>');
  } else if (tool.kind === "followup") {
    sub = "<b>Follow-up</b>: “" + esc(demo.prompt) + "” makes no sense alone. Code repeats the <b>last tool</b> with the new place.";
    left += '<div class="sec" id="tSec1">' + label(1, "Earlier in this chat") +
      '<div class="chatline">You: <b>' + esc(demo.before.user) + "</b></div>" +
      '<div class="chatline">Tool call: <b class="callbox" style="font-size:18px;padding:2px 10px">' + esc(demo.before.call || "–") + "</b></div>" +
      '<div class="chatline">Singularity: <b>' + esc(clip(demo.before.reply, 130)) + "</b></div></div>";
    left += '<div class="sec" id="tSec2">' + label(2, "Now: “" + esc(demo.prompt) + "” → same tool, new place") +
      '<span class="callbox swap" id="swap"><s>' + esc(demo.before.call || "") + "</s><b>" + esc(tool.call) + "</b></span>" +
      (demo.rewritten ? '<div class="note">The question is also rewritten in full: <b style="color:var(--text)">“' + esc(demo.rewritten) + "”</b></div>" : "") + "</div>";
    const degrees = (String(tool.result).match(/-?\d+(?:[.,]\d+)?\s*°C/) || [""])[0];
    right += '<div class="sec" id="tSec3">' + label(3, "The tool's result") + '<div class="weather">' + (degrees ? '<span class="deg">' + esc(degrees) + "</span>" : "") +
      "<span>" + esc(clip(tool.result, 150)) + "</span></div></div>";
    right += inputSection(4, true);
  } else if (tool.kind === "identity") {
    sub = "<b>Identity file</b>: questions about the model are answered from a small file of true facts, so it can't invent things about itself.";
    const rows = tool.matches.map(([question, score], index) => '<div class="match" id="match-' + index + '"><span>' + esc(question) + '</span><span class="score"><i data-w="' +
      Math.round(score * 100) + '"></i></span><span class="v">' + score.toFixed(2) + "</span></div>").join("");
    left += '<div class="sec" id="tSec1">' + label(1, "How close is it to the questions in identity_qa.txt?") + rows + "</div>";
    const best = tool.matches[0] ? tool.matches[0][1].toFixed(2) : "?";
    right += '<div class="sec" id="tSec2">' + label(2, "Best match " + best + " ≥ " + tool.threshold + " → its answer is used as it is") +
      '<div class="fixed-reply">' + esc(demo.direct.text) + '</div><div class="note">The model is not run: steps 03–07 are skipped.</div></div>';
  } else if (tool.kind === "who") {
    sub = "<b>Nobody to search for</b>: the question says “he / she / it”, but no name was given earlier.";
    left += '<div class="sec" id="tSec1"><div class="stop">Searching would be a guess, so Singularity asks:<div class="fixed-reply" style="--c:var(--ban)">' + esc(demo.direct.text) + "</div></div></div>";
  } else if (tool.kind === "none") {
    sub = "<b>No tool</b>: no rule matched and the model's first token was not &lt;|search|&gt;. It answers from what it learned in training, so a small model may get facts wrong here.";
    left += '<div class="sec" id="tSec1"><div class="note" style="margin:0">' + esc((demo.gates.search || demo.gates.question || ["", ""])[1]) + "</div></div>";
    right += '<div class="sec" id="tSec4">' + label(1, "The model's input: just your message") + realInputHtml(false) + "</div>";
  } else {
    sub = "<b>" + (tab ? tab.name : "A tool") + "</b>: plain code runs the tool; the model only puts the result into words.";
    left += callBox(1);
    left += '<div class="sec" id="tSec2">' + label(2, "The tool's result") + '<div class="callbox" style="white-space:normal;display:block">' + esc(clip(tool.result, 220)) + "</div></div>";
    right += inputSection(3, true);
  }
  boards.tool.innerHTML = boardHead("tool", sub) + tabs + '<div class="t-cols"><div>' + left + "</div><div>" + right + "</div></div>";
}

async function realToolPhase() {
  enterStation("tool");
  const tool = demo.tool;
  if (tool.tab) $("tab-" + tool.tab).classList.add("on");
  if (tool.kind === "wiki") {
    const search = tool.search || { results: [], sentences: [] };
    caption("<b>Wikipedia</b> gets the call. One request returns up to 5 articles and their text.");
    reveal("tSec1");
    await wait(1300);
    reveal("tSec2");
    for (let index = 0; index < (search.results || []).length; index++) {
      const result = search.results[index];
      const chosen = result.title === search.chosen;
      const row = $("res-" + index);
      row.classList.add(chosen ? "ok" : result.kept ? "maybe" : "bad");
      row.querySelector(".m").textContent = chosen ? "✓" : result.kept ? "·" : "✕";
      row.querySelector("small").textContent = (chosen ? "chosen · " : "") + result.why;
      caption((chosen ? "✓ chosen: " : result.kept ? "· allowed: " : "✕ rejected: ") + "<b>" + esc(result.title) + "</b> (" + esc(result.why) + ")");
      await wait(chosen ? 1100 : 650);
    }
    if (search.sentences && search.sentences.length) {
      reveal("tSec3");
      caption("The sentences of <b>“" + esc(search.chosen || "") + "”</b> get points for the question's words (and for the kind of answer: “when” wants a year).");
      await wait(400);
      document.querySelectorAll("#tSec3 .score i").forEach((bar) => { bar.style.width = bar.dataset.w + "%"; });
      await wait(1500);
      $("sent-0").classList.add("best");
      await wait(1500);
    }
    if (demo.direct) {
      reveal("tSec4");
      caption("No usable article. A factual question with no source gets a <b>fixed reply</b>: no guessing.");
      await wait(3200);
      return;
    }
    if (tool.unused) {
      reveal("tSec4");
      caption("The article doesn't match well enough: it is <b>not used</b>, and the model answers on its own.");
      await wait(3000);
      return;
    }
    await insertContext("The first sentence and the best ones go into the model's input as <b>&lt;|context|&gt;</b>, after the call. The model reads them like any other text.");
  } else if (tool.kind === "calc") {
    caption("The <b>calculator</b> gets the call <b>" + esc(tool.call) + "</b>.");
    reveal("tSec1");
    await wait(1300);
    reveal("tSec2");
    caption("Python computes it exactly. No model involved, so it can't be wrong.");
    await wait(900);
    $("calcResult").classList.add("in");
    await wait(1800);
    await insertContext("The result is added to the model's input as <b>&lt;|context|&gt;</b>. The model only has to put it into a sentence.");
  } else if (tool.kind === "followup") {
    caption("Earlier in this chat a tool ran: <b>" + esc(demo.before.call || "") + "</b>. That call was remembered.");
    reveal("tSec1");
    await wait(2400);
    reveal("tSec2");
    caption("“" + esc(demo.prompt) + "” is a <b>follow-up</b>: the same tool as before, only the place changes.");
    await wait(1200);
    $("swap").classList.add("done");
    await wait(2200);
    reveal("tSec3");
    caption("The tool runs again: <b>" + esc(tool.call) + "</b>.");
    await wait(1800);
    await insertContext("The rewritten question and the result go into the model's input. Now the model knows what “" + esc(demo.prompt) + "” means.");
  } else if (tool.kind === "identity") {
    caption("A question about the model itself. It is compared with the questions in the <b>identity file</b>.");
    reveal("tSec1");
    await wait(400);
    document.querySelectorAll("#tSec1 .score i").forEach((bar) => { bar.style.width = bar.dataset.w + "%"; });
    await wait(1600);
    if ($("match-0")) $("match-0").classList.add("best");
    reveal("tSec2");
    caption("Close enough (" + (tool.matches[0] ? tool.matches[0][1].toFixed(2) : "?") + "). Its <b>true answer</b> is used as it is, and the model is not run.");
    await wait(3000);
  } else if (tool.kind === "who") {
    caption("Nobody was named, so there is nothing to search for: Singularity asks who you mean.");
    reveal("tSec1");
    await wait(2800);
  } else if (tool.kind === "none") {
    caption("<b>No tool</b> this time: the model answers from what it learned in training.");
    reveal("tSec1");
    await wait(1600);
    reveal("tSec4");
    await wait(2200);
  } else {
    caption("The tool runs: <b>" + esc(tool.call) + "</b>.");
    reveal("tSec1");
    await wait(1300);
    reveal("tSec2");
    await wait(1800);
    await insertContext("The result is added to the model's input as <b>&lt;|context|&gt;</b>. The model only puts it into words.");
  }
}

// identity and "no source": a fixed reply written by code; the model stations are skipped
async function directPhase() {
  for (const key of ["tokens", "embed", "blocks", "logits", "sampling"]) skipStation(key);
  caption("The model is <b>not run</b>: the fixed reply goes straight to the answer.");
  await travel("tool", "answer", "fixed reply", 2600);
  enterStation("answer");
  const fixed = document.createElement("div");
  fixed.className = "fixed-reply";
  fixed.style.setProperty("--c", demo.direct.kind === "identity" ? COLORS.router : COLORS.ban);
  fixed.textContent = demo.direct.text;
  $("ansText").innerHTML = "";
  $("ansText").append(fixed);
  $("ansLegend").innerHTML = '<span class="note" style="margin:0">' + (demo.direct.kind === "identity" ? "Written by code from the identity file, not by the model." : "A fixed text: with no source, the code refuses to let the model guess.") + "</span>";
  reveal("ansLegend");
  await wait(1500);
}

// 03: the text is cut into pieces, each piece gets its id
async function tokensPhase() {
  enterStation("tokens");
  caption("This is the exact text the model receives: your message, the tool's result, and <b>markers</b> for who speaks.");
  await wait(2200);
  caption("The <b>tokenizer</b> cuts it into pieces it knows. Each piece gets its <b>id</b> (its number in the list).");
  $("strip").classList.add("cut");
  $("stripLabel").textContent = "The same text, cut into tokens (the small number = id)";
  const pieces = [...document.querySelectorAll("#strip .tok")];
  const pause = Math.max(25, Math.min(90, 4000 / pieces.length));
  for (const piece of pieces) {
    piece.classList.add("cut");
    await wait(pause);
  }
  reveal("tokLegend");
  await wait(2800);
}

// 04: each id picks a row of the table; the row is copied out
async function embedPhase() {
  enterStation("embed");
  caption("The ids are just labels. Now each id becomes <b>numbers</b> the model can compute with.");
  await wait(1600);
  for (let k = 0; k < embedShown.length; k++) {
    const token = embedShown[k];
    const slow = k < 2;
    $("etok-" + k).classList.add("on");
    if (slow) caption("“" + esc(showTok(token.text)) + "” has id <b>" + token.id + "</b> → look at <b>row " + token.id + "</b> of the table.");
    await wait(slow ? 1100 : 350);
    $("erow-" + token.id).classList.add("on");
    await wait(slow ? 1200 : 400);
    $("evec-" + k).classList.add("in");
    if (slow) caption("That row is <b>copied out</b>: it is the vector of “" + esc(showTok(token.text)) + "” (768 numbers, 6 shown).");
    await wait(slow ? 1500 : 450);
    $("etok-" + k).classList.remove("on");
    $("erow-" + token.id).classList.remove("on");
  }
  reveal("embedNote");
  caption("Every token gets its vector the same way: <b>" + tokens.length + " vectors</b> go into the blocks.");
  await wait(2600);
}

// 05: block 1 explained slowly, then the other 15
async function blocksPhase() {
  buildLanes();
  enterStation("blocks", false);
  const x0 = blockX(0);
  flyTo([[x0 + 11, ATTN_Y + 4, NEW_Z + 17], [x0 - 1.5, STREAM_Y - 0.4, 0]], 1800);     // close to block 1: see the lanes, heads and neurons
  caption("Every token's vector travels on its own <b>lane</b> through the 16 blocks (the newest token in front). Each block <b>adds</b> a correction to it.");
  const dot = makePacket("");
  dot.scale.setScalar(0.6);
  const startX = blockX(0) - 4;
  await animate(1600, (t) => dot.position.set(lerp(startX, blockX(0), ease(t)), STREAM_Y, NEW_Z));
  for (let index = 0; index < 16; index++) {
    const careful = index === 0, medium = index === 1 || index === 2;
    const x = blockX(index);
    $("kIndex").textContent = String(index + 1);
    // attention: up to the violet box and back
    drawAttention(index);
    $("kAttn").classList.add("on");
    $("kMlp").classList.remove("on");
    $("kNet").setAttribute("class", "");
    lightPart(index, "attn", 1);
    lightHeads(index, headStrengths(index));
    const arcs = attentionArcs(index);
    if (careful) {
      showInfo("attention");
      const focus = attentionWindow()[attentionWeights(attentionWindow(), 0).indexOf(Math.max(...attentionWeights(attentionWindow(), 0).slice(0, -1)))];
      caption("<b>Attention</b>: the newest token looks back at earlier tokens. Here it takes most from <b>“" + esc(showTok(focus.text)) + "”</b>.");
    }
    if (careful || medium) await animate(careful ? 1200 : 500, (t) => dot.position.set(x, lerp(STREAM_Y, ATTN_Y, Math.sin(t * Math.PI)), NEW_Z));
    await wait(careful ? 3600 : medium ? 600 : 110);
    live.remove(arcs);
    disposeGroup(arcs);
    // MLP: down to the orange box and back; the network lights up narrow -> wide -> narrow
    $("kAttn").classList.remove("on");
    $("kMlp").classList.add("on");
    mlpPattern(index);
    lightPart(index, "attn", 0.25);
    lightPart(index, "mlp", 0.4);
    lightNeurons(index, 0);
    if (careful) { showInfo("mlp"); caption("<b>MLP</b>: the vector is made 4× wider, transformed, and made narrow again. Much of what the model knows is stored here."); }
    if (careful || medium) {
      const phase = careful ? 800 : 250;
      $("kNet").setAttribute("class", "p1");
      const down = animate(careful ? 1400 : 500, (t) => { dot.position.set(x, lerp(STREAM_Y, MLP_Y, Math.sin(t * Math.PI)), NEW_Z); lightNeurons(index, t * 3.2); });
      await wait(phase);
      $("kNet").setAttribute("class", "p1 p2");
      await wait(phase);
      $("kNet").setAttribute("class", "p1 p2 p3");
      await down;
      await wait(careful ? 1200 : 200);
    } else {
      $("kNet").setAttribute("class", "p1 p2 p3");
      lightNeurons(index, 3);
      await wait(110);
    }
    // logit lens
    const cell = $("lens-" + index);
    cell.querySelector("b").textContent = showTok(demo.lens[index]);
    document.querySelectorAll(".lens .now").forEach((element) => element.classList.remove("now"));
    cell.classList.add("now");
    if (careful) {
      caption("If the model stopped right after this block, it would say <b>“" + esc(showTok(demo.lens[0])) + "”</b> (the logit lens). Watch the guess change, block by block.");
      await wait(2600);
      caption("The other 15 blocks do the same two steps, each with its own weights. The board above shows what happens inside each one.");
      flyTo(stationCamera("blocks"), 1600);
    }
    lightPart(index, "attn", 0.15);
    lightPart(index, "mlp", 0.15);
    if (index < 15) await animate(careful || medium ? 450 : 120, (t) => dot.position.set(lerp(x, blockX(index + 1), t), STREAM_Y, NEW_Z));
  }
  await animate(500, (t) => dot.position.set(lerp(blockX(15), blockX(15) + 4, t), STREAM_Y, NEW_Z));
  removePacket(dot);
  showInfo("block");
  caption("After the 16 blocks the guess is <b>“" + esc(showTok(demo.lens[15] || "")) + "”</b>. Now the last vector is turned into scores.");
  await wait(2000);
  resetBlocks();
}

// 06: scores -> chances; fact_guard bans invented names
async function logitsPhase(step) {
  const generated = demo.generated[step];
  const top = generated.top;
  const banned = top.map((item, index) => [item, index]).filter(([item]) => item[2] === "ban");
  renderBars(step);
  // the fact_guard box is filled now (still invisible) so the camera frames the full board
  const names = banned.map(([item]) => "“" + esc(showTok(item[0])) + "”").join(", ");
  $("guard").innerHTML = banned.length
    ? '<div class="stop"><b>fact_guard</b>: ' + names + (banned.length > 1 ? " are" : " is") + " neither in your question nor in the tool's text, so " + (banned.length > 1 ? "they are" : "it is") + " <b>banned</b>. The model can't invent " + (banned.length > 1 ? "them" : "it") + ".</div>"
    : '<div class="note">fact_guard only checks answers with a Wikipedia article. Here all guesses stay.</div>';
  enterStation("logits");
  caption("The last vector gets a <b>score for every token</b> in the list. Softmax turns them into the model's chances.");
  await wait(2600);
  if (banned.length) {
    for (const [, index] of banned) {
      const row = $("bar-" + index);
      row.classList.add("ban");
      row.querySelector("i").style.width = "0%";
      row.querySelector(".pc").textContent = "banned";
    }
    const keptTotal = 1 - banned.reduce((sum, [item]) => sum + item[1], 0);
    top.forEach((item, index) => {
      if (item[2] === "ban") return;
      const after = generated.after ? generated.after[item[0]] : item[1] / keptTotal;
      $("bar-" + index).querySelector(".pc").textContent = after === undefined ? "–" : Math.round(after * 100) + "%";
    });
    reveal("guard");
    caption("<b>fact_guard</b> bans " + names + ": a name or number that is not in the tool's text can't be written.");
    await wait(3400);
  } else {
    reveal("guard");
    await wait(1600);
  }
}

// 07: the wheel: slices = chances (same colors as the bars)
let wheel = null;
function buildWheel(step) {
  if (wheel) { live.remove(wheel); disposeGroup(wheel); }
  const s = station("sampling");
  const outer = new THREE.Group();
  outer.position.set(s.x, WHEEL_Y, 0);
  outer.scale.setScalar(1.2);
  const disc = new THREE.Group();
  outer.add(disc);
  const generated = demo.generated[step];
  const barTokens = generated.top.map((item) => item[0]);
  const source = generated.wheel || generated.top.filter((item) => item[2] !== "ban");
  const kept = source.map(([token, prob], index) => {
    const bar = barTokens.indexOf(token);
    return { token, prob, color: RANK_COLORS[bar >= 0 ? bar : (barTokens.length + index) % RANK_COLORS.length] };
  });
  const total = kept.reduce((sum, item) => sum + item.prob, 0);
  let angle = 0;
  kept.forEach((item) => {
    const size = (item.prob / total) * Math.PI * 2;
    const slice = new THREE.Mesh(new THREE.RingGeometry(1.5, 4.6, 64, 1, angle, Math.max(0.01, size - 0.04)),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(item.color).multiplyScalar(0.2), transparent: true, opacity: 0.95, side: THREE.DoubleSide }));
    const edge = new THREE.Mesh(new THREE.RingGeometry(4.5, 4.6, 64, 1, angle, Math.max(0.01, size - 0.04)), new THREE.MeshBasicMaterial({ color: new THREE.Color(item.color).multiplyScalar(0.7), side: THREE.DoubleSide }));
    disc.add(slice, edge);
    item.slice = slice;
    if (item.prob / total > 0.07) {
      const text = makeLabel(showTok(item.token), { size: 0.75, mono: true, weight: 600, bg: "rgba(12,12,22,0.9)", border: item.color });
      const middle = angle + size / 2;
      text.position.set(Math.cos(middle) * 3.1, Math.sin(middle) * 3.1, 0.1);
      disc.add(text);
    }
    item.middle = angle + size / 2;
    angle += size;
  });
  const rim = new THREE.Mesh(new THREE.TorusGeometry(4.75, 0.05, 8, 160), glowMaterial("#ffffff", 0.3));
  const hub = new THREE.Mesh(new THREE.CircleGeometry(1.35, 48), new THREE.MeshBasicMaterial({ color: "#12111d" }));
  hub.position.z = 0.05;
  const hubRing = new THREE.Mesh(new THREE.TorusGeometry(1.38, 0.05, 8, 80), glowMaterial("#ffffff", 0.5));
  const pointer = new THREE.Mesh(new THREE.ConeGeometry(0.45, 1.2, 24), glowMaterial("#ffffff", 0.95));
  pointer.position.set(0, 5.5, 0.1);
  pointer.rotation.z = Math.PI;
  outer.add(rim, hub, hubRing, pointer);
  outer.userData = { disc, kept };
  live.add(tagStation("sampling", outer));
  wheel = outer;
  return outer;
}
async function spinWheel(step, duration) {
  const chosen = demo.generated[step].token;
  const { disc, kept } = wheel.userData;
  const slice = kept.find((item) => item.token === chosen) || kept.find((item) => item.token.trim() === chosen.trim()) || kept[0];
  const target = Math.PI / 2 - slice.middle + Math.PI * 2 * (duration > 1500 ? 4 : 2);
  await animate(duration, (t) => { disc.rotation.z = target * (1 - Math.pow(1 - t, 3)); });
  slice.slice.material.color.set(slice.color).multiplyScalar(0.4);            // the drawn slice lights up a little and pops out
  await animate(300, (t) => slice.slice.position.set(Math.cos(slice.middle) * 0.35 * t, Math.sin(slice.middle) * 0.35 * t, 0.05));
  const hubText = makeLabel(showTok(chosen), { size: 0.7, mono: true, weight: 600, color: COLORS.mlp });
  hubText.position.z = 0.2;
  wheel.add(hubText);
}
async function samplingPhase(step) {
  enterStation("sampling");
  buildWheel(step);
  $("pick").innerHTML = "spinning…";
  caption("<b>Sampling</b>: a weighted draw. Temperature and top-p cut the long tail first; each slice is as big as what is left (same colors as the bars).");
  await wait(1000);
  await spinWheel(step, 2800);
  const chosen = demo.generated[step];
  const keptTop = chosen.wheel || chosen.top.filter((item) => item[2] !== "ban");
  const total = keptTop.reduce((sum, item) => sum + item[1], 0) || 1;
  const found = keptTop.find((item) => item[0] === chosen.token) || keptTop.find((item) => item[0].trim() === chosen.token.trim()) || keptTop[0];
  const share = Math.round((found[1] / total) * 100);
  $("pick").innerHTML = "Drawn: <b>" + esc(showTok(chosen.token)) + "</b> (" + share + "% chance)" +
    (chosen.wheel ? '<div class="note" style="margin-top:6px">The wheel holds only what is left after temperature and top-p: ' + keptTop.length + " of the 16,384 tokens.</div>" : "");
  caption("Drawn: <b>“" + esc(showTok(chosen.token)) + "”</b>.");
  await wait(1600);
}

// the drawn token goes two ways: to the answer, and back to 03 (the loop)
async function loopPhase(step) {
  const token = demo.generated[step].token;
  caption("The token goes to the <b>answer</b>, and also <b>back to step 03</b>: it becomes part of the input for the next token.");
  flyTo(LOOP_VIEW, 1800);
  await wait(900);
  await Promise.all([
    travel("sampling", "answer", showTok(token), 1600, false),
    travelCurve(loopBack, showTok(token), 2600, COLORS.accent),
  ]);
  answer(token);
  appendToken({ text: token, kind: "new", id: fakeId(token) });
  await wait(1400);
}

// the next tokens: the same loop, fast, seen from far away
async function quickStep(step) {
  const previous = demo.generated[step - 1].token;
  caption("Token " + (step + 1) + ": the <b>same loop</b> again, now with “" + esc(showTok(previous)) + "” at the end of the input.");
  enterStation("blocks", false);
  await travel("tokens", "blocks", null, 900, false);
  buildLanes();
  await animate(700, (t) => { for (let index = 0; index < 16; index++) { const v = Math.max(0, 1 - Math.abs(t * 16 - index) / 2); lightPart(index, "attn", v); lightPart(index, "mlp", v); } });
  resetBlocks();
  await travel("blocks", "logits", null, 600, false);
  enterStation("logits", false);
  renderBars(step);
  $("guard").classList.remove("in");
  await wait(700);
  await travel("logits", "sampling", null, 500, false);
  enterStation("sampling", false);
  buildWheel(step);
  await spinWheel(step, 900);
  $("pick").innerHTML = "Drawn: <b>" + esc(showTok(demo.generated[step].token)) + "</b>";
  const token = demo.generated[step].token;
  await Promise.all([travel("sampling", "answer", showTok(token), 900, false), travelCurve(loopBack, showTok(token), 1300, COLORS.accent)]);
  answer(token);
  appendToken({ text: token, kind: "new", id: fakeId(token) });
  await wait(500);
}

async function fastForward() {
  enterStation("answer");
  caption("… the same loop, <b>once per token</b> (fast forward), until the model writes &lt;|endoftext|&gt;.");
  for (const word of demo.rest.split(/(?= )/)) {
    await animate(200, (t) => { for (let index = 0; index < 16; index++) lightPart(index, "attn", Math.max(0, 1 - Math.abs(t * 16 - index) / 2)); });
    answer(word);
  }
  resetBlocks();
  await wait(800);
}

// grounded: a sentence copied word for word from the article, straight from 02 to 08
async function groundedPhase() {
  evidence.visible = true;
  caption("<b>Grounded sentences</b>: after the model's first sentence, text is <b>copied word for word</b> from the article.");
  flyTo(OVERVIEW, 1800);
  await wait(1200);
  await travelCurve(evidenceCurve, "“" + demo.grounded.slice(0, 32).trim() + "…”", 2600, COLORS.ground);
  const shown = demo.grounded.length > 280 ? demo.grounded.slice(0, 280).replace(/\s+\S*$/, "") + " …" : demo.grounded;   // the board shows the start
  answer(" " + shown, "grounded");
  const source = document.createElement("span");
  source.className = "src";
  source.textContent = "Wikipedia · " + demo.source;
  $("ansText").insertBefore(source, $("ansCursor"));
  reveal("ansLegend");
  enterStation("answer");                              // after the text is in, so the camera frames the whole answer
  await wait(1500);
}

// ---------------------------------------------------------------- clicking parts of the scene
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let downAt = null;
world.addEventListener("pointerdown", (event) => {
  const board = event.target.closest && event.target.closest(".board");
  downAt = { x: event.clientX, y: event.clientY, board };
}, true);
world.addEventListener("pointerup", (event) => {
  if (!downAt || Math.hypot(event.clientX - downAt.x, event.clientY - downAt.y) > 4) return;   // that was a drag
  let key = downAt.board ? downAt.board.dataset.station : null;
  if (!key) {
    pointer.set((event.clientX / window.innerWidth) * 2 - 1, -(event.clientY / window.innerHeight) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(scene.children, true).find((item) => item.object.userData.station);
    if (hit) key = hit.object.userData.station;
  }
  if (key) { showInfo(infoKeyFor(key)); flyTo(stationCamera(key), 1400, null, true); }
});

// ---------------------------------------------------------------- controls
$("promptForm").addEventListener("submit", (event) => {
  event.preventDefault();
  if (running) stopRun();
  else run(null);
});
$("pauseButton").addEventListener("click", () => {
  clock.paused = !clock.paused;
  $("pauseButton").textContent = clock.paused ? "▶" : "❚❚";
});
$("stepButton").addEventListener("click", () => {
  if (!clock.paused) { clock.paused = true; $("pauseButton").textContent = "▶"; }
  clock.finishNow = true;
});
for (const button of $("speed").querySelectorAll("button")) {
  button.addEventListener("click", () => {
    clock.speed = parseFloat(button.dataset.speed);
    for (const other of $("speed").querySelectorAll("button")) other.classList.toggle("on", other === button);
  });
}
$("overviewButton").addEventListener("click", () => flyTo(OVERVIEW, 1400, null, true));
$("historyButton").addEventListener("click", () => $("history").classList.add("open"));
$("historyClose").addEventListener("click", () => $("history").classList.remove("open"));
window.addEventListener("keydown", (event) => {
  if (event.target.tagName === "INPUT") return;
  if (event.code === "Space") { event.preventDefault(); $("pauseButton").click(); }
  if (event.code === "ArrowRight") $("stepButton").click();
  if (event.code === "Escape") $("history").classList.remove("open");
});

function buildTimeline() {
  const colors = ["#c9c5ff", "#f3c969", "#e9a6dc", "#7fd8ea", "#a9a5ff", "#ff9d6e", "#e6e3ff", "#e6e3ff", "#6fd3a0", "#6fd3a0"];
  EXPLORE_HISTORY.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "tl-item";
    row.style.setProperty("--c", colors[index % colors.length]);
    row.innerHTML = "<b>" + item.name + '</b><span class="when">' + item.when + "</span><p>" + item.text + "</p>";
    $("timeline").append(row);
  });
}

// ---------------------------------------------------------------- start
let last = performance.now();
function frame(now) {
  const delta = Math.min(100, now - last);
  last = now;
  moveTasks(delta);
  moveCamera(delta);
  moveFlow(now / 1000);
  moveLaneFlow(now / 1000);
  controls.update();
  composer.render();
  cssRenderer.render(cssScene, camera);
  requestAnimationFrame(frame);
}

await document.fonts.ready;          // sprite labels are drawn on canvases: wait for Inter / JetBrains Mono
addStars();
addFloor();
addPlates();
addRails();
addFlow();
addBlocks();
addBoards();
buildStageList();
buildScenarioChips();
buildTimeline();
demo = { ...SCENARIOS.wiki, chipKey: "wiki" };
resetRun();
clock.run += 1;                      // nothing is running yet
showInfo("router");
requestAnimationFrame(frame);
window.__explore = { camera, controls, clock, flyTo, run, stationCamera };     // for debugging in the browser console
setTimeout(() => run("wiki"), 900);
