import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const api = async (path, body) => {
  const r = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    toast(msg);
    throw new Error(msg);
  }
  return r.json();
};
let toastTimer;
function toast(msg, ok = false) {
  const t = $('toast');
  t.textContent = msg; t.hidden = false; t.classList.toggle('ok', ok);
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, 5000);
}
const debounce = (fn, ms) => { let h; return (...a) => { clearTimeout(h); h = setTimeout(() => fn(...a), ms); }; };
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fixed = (v, n = 2) => (v == null || !isFinite(v) ? '-' : (+v).toFixed(n));

// ---------------------------------------------------------------- 3D view
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const canvas = $('gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161a);
scene.fog = new THREE.Fog(0x14161a, 25, 60);
const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 200);
camera.position.set(3.2, -3.6, 2.0);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0, 0.9);
controls.maxPolarAngle = Math.PI / 2 - 0.02;
scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x30343c, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(4, -6, 10); sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -8, right: 8, top: 8, bottom: -8, near: 1, far: 40 });
scene.add(sun, sun.target);

const robotGroup = new THREE.Group(); scene.add(robotGroup);
let geomMeshes = [], floorMesh = null, sceneVersion = -1, gridInfo = { size: 24, n: 240 }, hasCadVisuals = false;

const grid = new THREE.GridHelper(80, 80, 0x3a404c, 0x2a2e37);
grid.rotation.x = Math.PI / 2; grid.position.z = 0.002; scene.add(grid);

// lidar point cloud
const lidarGeo = new THREE.BufferGeometry();
lidarGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
const lidarPts = new THREE.Points(lidarGeo, new THREE.PointsMaterial({ color: 0x5cf0a0, size: 0.03 }));
lidarPts.frustumCulled = false; scene.add(lidarPts);

// lidar occupancy map drawn on the floor
let mapTex = null;
const mapMesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1),
  new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false }));
mapMesh.position.z = 0.006; scene.add(mapMesh);

// path, goals, footsteps
const pathLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffd34d }));
pathLine.frustumCulled = false; scene.add(pathLine);
const goalGroup = new THREE.Group(); scene.add(goalGroup);
const goalMat = new THREE.MeshStandardMaterial({ color: 0xffd34d, emissive: 0x6b5200 });
const goalRing = new THREE.RingGeometry(0.16, 0.22, 32);
const arrowGeo = new THREE.ConeGeometry(0.09, 0.3, 12).rotateZ(-Math.PI / 2);   // points along +x
const stepGroup = new THREE.Group(); scene.add(stepGroup);
const stepMats = [new THREE.MeshBasicMaterial({ color: 0x4f9cf9, transparent: true, opacity: 0.55 }),
                  new THREE.MeshBasicMaterial({ color: 0xf2a65c, transparent: true, opacity: 0.55 })];
const dragArrow = new THREE.Mesh(arrowGeo, new THREE.MeshBasicMaterial({ color: 0xffffff }));
dragArrow.visible = false; scene.add(dragArrow);

function b64(str, Type) {
  const bin = atob(str), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Type(u8.buffer);
}
function meshGeometry(m) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(b64(m.vert, Float32Array), 3));
  g.setIndex(new THREE.BufferAttribute(b64(m.face, Uint32Array), 1));
  g.computeVertexNormals();
  return g;
}

function buildScene(s) {
  sceneVersion = s.version; gridInfo = s.grid;
  robotGroup.clear(); geomMeshes = []; floorMesh = null;
  const meshGeos = {};
  for (const [id, m] of Object.entries(s.meshes)) meshGeos[id] = meshGeometry(m);
  // When CAD visual meshes (group 3) exist, show them instead of the robot's collision primitives.
  hasCadVisuals = s.geoms.some(g => g.group === 3 && g.type === 'mesh');
  for (const g of s.geoms) {
    const [a, b, c] = g.size; let geo = null;
    switch (g.type) {
      case 'plane': geo = new THREE.PlaneGeometry(a > 0 ? 2 * a : 80, b > 0 ? 2 * b : 80); break;
      case 'sphere': geo = new THREE.SphereGeometry(a, 28, 18); break;
      case 'capsule': geo = new THREE.CapsuleGeometry(a, 2 * b, 8, 20).rotateX(Math.PI / 2); break;
      case 'cylinder': geo = new THREE.CylinderGeometry(a, a, 2 * b, 28).rotateX(Math.PI / 2); break;
      case 'ellipsoid': geo = new THREE.SphereGeometry(1, 28, 18).scale(a, b, c); break;
      case 'box': geo = new THREE.BoxGeometry(2 * a, 2 * b, 2 * c); break;
      case 'mesh': geo = meshGeos[g.mesh]; break;
    }
    const hiddenPrimitive = hasCadVisuals && g.robot && g.group === 0 && g.type !== 'mesh' && !/foot|finger|pad/.test(g.name);
    let mesh;
    if (!geo || g.rgba[3] === 0 || g.group > 3 || hiddenPrimitive) { mesh = new THREE.Object3D(); }
    else if (g.type === 'plane') {
      mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x20242b, roughness: 1 }));
      mesh.receiveShadow = true; floorMesh = mesh;
    } else {
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]), roughness: 0.55, metalness: 0.1,
        transparent: g.rgba[3] < 1, opacity: g.rgba[3] });
      mesh = new THREE.Mesh(geo, mat); mesh.castShadow = true; mesh.receiveShadow = true;
    }
    robotGroup.add(mesh); geomMeshes.push(mesh);
  }
  mapMesh.scale.set(gridInfo.size, gridInfo.size, 1);
  const i = s.info;
  $('model-info').textContent = `${i.custom ? 'Imported: ' + i.custom + '\n' : ''}mass ${i.mass.toFixed(1)} kg   ` +
    `${i.nu} motors   ${i.nv} DOF   ${i.ngeom} geoms   dt ${(i.timestep * 1000).toFixed(1)} ms` +
    (i.trainable ? '' : '\nNo free-floating base or motors: walking is disabled.');
  const sel = $('c-name'); const cur = sel.value; sel.innerHTML = '';
  for (const n of (s.cameras.length ? s.cameras : ['free'])) sel.add(new Option(n, n));
  if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
  refreshObjects();
}

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.floor(w * renderer.getPixelRatio()) || canvas.height !== Math.floor(h * renderer.getPixelRatio())) {
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
}
let rootPos = new THREE.Vector3(0, 0, 0.9);
function animate() {
  resize();
  if ($('follow').checked) {
    const d = rootPos.clone().sub(controls.target);
    controls.target.add(d); camera.position.add(d);
    sun.position.set(rootPos.x + 4, rootPos.y - 6, 10); sun.target.position.copy(rootPos);
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

// ---------------------------------------------------------------- goals: double-click, or Ctrl+drag for heading
const ray = new THREE.Raycaster();
function floorPoint(e) {
  if (!floorMesh) return null;
  const r = canvas.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  const hit = ray.intersectObject(floorMesh)[0];
  return hit ? hit.point : null;
}
canvas.addEventListener('dblclick', async (e) => {
  const p = floorPoint(e);
  if (p) await api('/api/goal', { x: p.x, y: p.y, append: e.shiftKey });
});
let drag = null;
canvas.addEventListener('pointerdown', (e) => {
  if (!e.ctrlKey || e.button !== 0) return;
  const p = floorPoint(e);
  if (!p) return;
  e.stopImmediatePropagation(); e.preventDefault();   // keep OrbitControls out of it
  drag = { start: p.clone(), shift: e.shiftKey, yaw: null };
  try { canvas.setPointerCapture(e.pointerId); } catch {}   // keep receiving moves outside the canvas
  dragArrow.position.set(p.x, p.y, 0.05); dragArrow.visible = true; dragArrow.rotation.set(0, 0, 0);
}, { capture: true });
canvas.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const p = floorPoint(e); if (!p) return;
  const dx = p.x - drag.start.x, dy = p.y - drag.start.y;
  if (Math.hypot(dx, dy) > 0.15) { drag.yaw = Math.atan2(dy, dx); dragArrow.rotation.set(0, 0, drag.yaw); }
});
canvas.addEventListener('pointerup', async (e) => {
  if (!drag) return;
  const d = drag; drag = null; dragArrow.visible = false;
  await api('/api/goal', { x: d.start.x, y: d.start.y, yaw_deg: d.yaw == null ? null : d.yaw * 180 / Math.PI, append: d.shift });
});

// ---------------------------------------------------------------- live stream
let lastTrainGen = -1, lastNavKey = '', lastStepsKey = '', pilotSeen = 0, pilotRun = null;
function onFrame(f) {
  if (f.v !== sceneVersion) { refreshScene(); return; }
  const n = Math.min(geomMeshes.length, f.pos.length / 3);
  for (let i = 0; i < n; i++) {
    const m = geomMeshes[i];
    m.position.set(f.pos[3 * i], f.pos[3 * i + 1], f.pos[3 * i + 2]);
    m.quaternion.set(f.quat[4 * i + 1], f.quat[4 * i + 2], f.quat[4 * i + 3], f.quat[4 * i]);
  }
  rootPos.set(f.root[0], f.root[1], f.root[2] || 0.9);
  const c = f.controller || {};
  $('r-time').textContent = f.t.toFixed(1);
  $('r-height').textContent = f.root[2].toFixed(2);
  $('r-speed').textContent = f.speed.toFixed(2);
  $('r-ctrl').textContent = (c.name || '-') + (f.policy ? ` (${f.policy})` : '');
  $('play').textContent = f.running ? 'Pause' : 'Play';
  $('play').dataset.running = f.running ? '1' : '';
  if (f.error) toast('Simulation error: ' + f.error);
  $('c-error').textContent = f.camera_error ? 'Camera rendering failed: ' + f.camera_error : '';
  $('s-body').textContent = `root height ${f.root[2].toFixed(3)} m\nposition x ${f.root[0].toFixed(2)}  y ${f.root[1].toFixed(2)}\n` +
    Object.entries(f.touch).map(([k, v]) => `${k.replace(/_/g, ' ')} ${v} N`).join('\n') + `\nfalls ${f.falls}`;
  if ($('ctrl-kind').value !== c.kind && document.activeElement !== $('ctrl-kind')) $('ctrl-kind').value = c.kind || 'auto';

  // navigation overlay
  const nav = f.nav, key = JSON.stringify([nav.path, nav.goals]);
  if (key !== lastNavKey) {
    lastNavKey = key;
    pathLine.geometry.setFromPoints(nav.path.map(p => new THREE.Vector3(p[0], p[1], 0.04)));
    goalGroup.clear();
    for (const g of nav.goals) {
      const ring = new THREE.Mesh(goalRing, goalMat); ring.position.set(g[0], g[1], 0.02); goalGroup.add(ring);
      if (g[2] != null) {
        const a = new THREE.Mesh(arrowGeo, goalMat);
        const yaw = g[2] * Math.PI / 180;
        a.position.set(g[0] + 0.3 * Math.cos(yaw), g[1] + 0.3 * Math.sin(yaw), 0.05); a.rotation.set(0, 0, yaw);
        goalGroup.add(a);
      }
    }
  }
  const steps = $('n-showsteps').checked ? (c.footsteps || []).slice(0, 40) : [];
  const sk = JSON.stringify(steps);
  if (sk !== lastStepsKey) {
    lastStepsKey = sk; stepGroup.clear();
    for (const s of steps) {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(0.22, 0.09), stepMats[s[3] ? 1 : 0]);
      m.position.set(s[0], s[1], 0.012); m.rotation.set(0, 0, s[2]); stepGroup.add(m);
    }
  }
  const err = c.goal_error_m != null ? `  error ${fixed(c.goal_error_m, 3)} m` : '';
  const herr = c.heading_error_deg != null ? ` / ${fixed(c.heading_error_deg, 1)} deg` : '';
  $('n-status').textContent = `controller: ${c.name || '-'} (${c.mode || '-'})${c.message ? ' - ' + c.message : ''}\n` +
    `planner: ${nav.state}, node ${nav.node}, goals queued ${nav.goals.length}${nav.message ? '\n' + nav.message : ''}${err}${herr}`;
  $('hud').textContent = nav.goals.length || nav.state === 'reached' || nav.state === 'failed' || nav.state === 'fallen'
    ? `${nav.state}${nav.message ? ': ' + nav.message : ''}${err}${herr}` : '';
  if (f.arms) onArms(f.arms);
  if (f.train) onTrain(f.train);
  if (f.pilot) onPilot(f.pilot);
}
function onTrain(t) {
  const last = t.last;
  $('t-status').textContent = t.state === 'idle' ? 'idle' :
    `${t.name}: ${t.state}  ${t.kind === 'ppo' || t.kind === 'distill' ? 'iteration' : 'generation'} ${t.generation} / ${t.generations}` +
    (t.best != null ? `\nbest ${fixed(t.best)}` : '') +
    (last && last.sps ? `   ${(last.sps / 1000).toFixed(0)}k steps/s` : '') +
    (t.total_steps ? `\n${(t.total_steps / 1e6).toFixed(1)} M steps simulated` : '') + (t.error ? `\n${t.error}` : '');
  if (t.generation !== lastTrainGen) { lastTrainGen = t.generation; refreshTraining(); }
}
function onArms(a) {
  $('a-status').textContent = Object.entries(a).map(([side, s]) =>
    `${side}: ${s.skill || '-'} ${s.state || ''}${s.error_m != null ? `  error ${fixed(s.error_m, 3)} m` : ''}` +
    `${s.holding ? `  holding ${s.holding}` : ''}${s.hand_xyz ? `\n   hand at ${s.hand_xyz.map(v => fixed(v)).join(', ')}` : ''}${s.message ? '\n   ' + s.message : ''}`).join('\n');
}
function onBinary(buf) {
  const tag = new Uint8Array(buf, 0, 1)[0], body = buf.slice(1);
  if (tag === 1) {
    const pts = new Float32Array(body);
    lidarGeo.setAttribute('position', new THREE.BufferAttribute(pts, 3));
    $('l-info').textContent = `${pts.length / 3} returns per scan`;
  } else if (tag === 2) {
    const img = $('cam'); const url = URL.createObjectURL(new Blob([body], { type: 'image/jpeg' }));
    const old = img.src; img.src = url; img.hidden = false; if (old.startsWith('blob:')) URL.revokeObjectURL(old);
  } else if (tag === 3) {
    const n = new Uint32Array(body.slice(0, 4))[0], cells = new Uint8Array(body, 4);
    const rgba = new Uint8Array(n * n * 4);
    for (let i = 0; i < n * n; i++) {
      const v = cells[i] % 3;               // 0 unknown, 1 free, 2 occupied (lidar)
      if (v === 1) rgba.set([70, 150, 110, 70], 4 * i);
      else if (v === 2) rgba.set([240, 90, 80, 230], 4 * i);
    }
    if (mapTex) mapTex.dispose();
    mapTex = new THREE.DataTexture(rgba, n, n, THREE.RGBAFormat);
    mapTex.magFilter = THREE.NearestFilter; mapTex.needsUpdate = true;
    mapMesh.material.map = mapTex; mapMesh.material.needsUpdate = true;
  }
}
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { $('r-conn').textContent = 'connected'; $('r-conn').className = 'good'; };
  ws.onmessage = (e) => (typeof e.data === 'string' ? onFrame(JSON.parse(e.data)) : onBinary(e.data));
  ws.onclose = () => { $('r-conn').textContent = 'offline'; $('r-conn').className = 'bad'; setTimeout(connect, 1000); };
}

// ---------------------------------------------------------------- model + motors
let refreshing = false;
async function refreshScene() {
  if (refreshing) return; refreshing = true;
  try { applyState(await api('/api/state')); } finally { refreshing = false; }
}
function applyState(s) { buildScene(s.scene); setParams(s.params); buildMotors(s.actuators); }
const fmt = (v, spec) => (spec.max - spec.min < 2 ? v.toFixed(3) : spec.max > 100 ? v.toFixed(0) : v.toFixed(2));
let specs = [];
function buildParams(spec) {
  specs = spec; const root = $('param-groups'); root.innerHTML = '';
  for (const group of [...new Set(spec.map(p => p.group))]) {
    const h = document.createElement('h3'); h.textContent = group; root.append(h);
    for (const p of spec.filter(q => q.group === group)) {
      const row = document.createElement('label'); row.className = 'slider';
      row.innerHTML = `<span>${esc(p.label)}</span><input type="range" id="p-${p.name}" min="${p.min}" max="${p.max}" step="${(p.max - p.min) / 200}"><output></output>`;
      const input = row.querySelector('input'), out = row.querySelector('output');
      input.addEventListener('input', () => { out.textContent = fmt(+input.value, p); sendParams(); });
      root.append(row);
    }
  }
}
function setParams(params) {
  for (const p of specs) {
    const input = $('p-' + p.name); if (!input || params[p.name] == null) continue;
    input.value = params[p.name]; input.nextElementSibling.textContent = fmt(params[p.name], p);
  }
  $('p-arms').checked = params.arms;
  $('p-hands').checked = params.hands ?? true;
}
const sendParams = debounce(async () => {
  const d = { arms: $('p-arms').checked, hands: $('p-hands').checked };
  for (const p of specs) d[p.name] = +$('p-' + p.name).value;
  const r = await api('/api/params', d);
  buildScene(r.scene); buildMotors(r.actuators);
}, 120);

function buildMotors(list) {
  const body = document.querySelector('#motors tbody'); body.innerHTML = '';
  for (const a of list) {
    const tr = document.createElement('tr');
    const num = (field, v, step) => `<td><input type="number" data-f="${field}" value="${+v.toPrecision(4)}" step="${step}" aria-label="${esc(a.name)} ${field}"></td>`;
    tr.innerHTML = `<td title="${esc(a.name)}">${esc(a.name.replace(/_/g, ' '))}</td>` +
      (a.servo ? num('kp', a.kp, 10) + num('kv', a.kv, 1) : num('gear', a.gear, 1) + '<td></td>') +
      num('torque', a.torque, 5) + num('damping', a.damping, 0.1) + num('armature', a.armature, 0.005) +
      `<td><input type="range" data-f="ctrl" min="${a.lo}" max="${a.hi}" step="${(a.hi - a.lo) / 100 || 0.01}" value="${a.ctrl}" aria-label="${esc(a.name)} target"></td>`;
    tr.querySelectorAll('input').forEach(inp => inp.addEventListener(inp.type === 'range' ? 'input' : 'change', () =>
      api('/api/actuator', { index: a.index, field: inp.dataset.f, value: +inp.value })));
    body.append(tr);
  }
}

// ---------------------------------------------------------------- charts
function drawChart(canvasId, hist, series, xLabel = 'generation') {
  const c = $(canvasId), g = c.getContext('2d'), W = c.width, H = c.height, pad = 40;
  g.clearRect(0, 0, W, H);
  g.font = '11px system-ui'; g.fillStyle = '#98a0ad';
  if (!hist.length) { g.fillText('No data yet', pad, H / 2); return; }
  const all = hist.flatMap(h => series.map(([k]) => h[k])).filter(v => v != null && isFinite(v));
  if (!all.length) return;
  let lo = Math.min(...all), hi = Math.max(...all); if (hi - lo < 1e-9) hi = lo + 1;
  const x = (i) => pad + (W - pad - 8) * (hist.length === 1 ? 0.5 : i / (hist.length - 1));
  const y = (v) => H - 20 - (H - 30) * (v - lo) / (hi - lo);
  g.strokeStyle = '#333843'; g.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4; g.beginPath(); g.moveTo(pad, y(v)); g.lineTo(W - 8, y(v)); g.stroke();
    g.fillText(Math.abs(v) >= 1000 ? v.toExponential(1) : v.toFixed(Math.abs(v) < 10 ? 2 : 1), 2, y(v) + 4);
  }
  g.fillText(`${xLabel} ${hist[0].gen ?? hist[0].it ?? 1} to ${hist[hist.length - 1].gen ?? hist[hist.length - 1].it ?? hist.length}`, pad, H - 4);
  for (const [key, col] of series) {
    g.strokeStyle = col; g.lineWidth = 2; g.beginPath(); let started = false;
    hist.forEach((h, i) => { const v = h[key]; if (v == null || !isFinite(v)) return;
      if (started) g.lineTo(x(i), y(v)); else { g.moveTo(x(i), y(v)); started = true; } });
    g.stroke();
  }
}
const TRAIN_SERIES = [['mean', '#98a0ad'], ['center', '#4f9cf9'], ['best', '#4cc38a']];

// ---------------------------------------------------------------- training
async function refreshTraining() {
  const [st, pol] = await Promise.all([api('/api/train/status'), api('/api/policies')]);
  drawChart('chart', st.history || [], TRAIN_SERIES, st.kind === 'ppo' || st.kind === 'distill' ? 'iteration' : 'generation');
  buildPolicies(pol);
}
function buildPolicies(list) {
  const root = $('policies'); root.innerHTML = '';
  if (!list.length) { root.innerHTML = '<p class="hint">None yet. Train one above.</p>'; return; }
  for (const p of list) {
    const d = document.createElement('div'); d.className = 'policy';
    d.innerHTML = `<div><b></b><small>${esc(p.kind || p.algo || '')} ${p.generation != null ? 'gen ' + p.generation : ''}  score ${p.fitness != null ? fixed(p.fitness, 1) : '?'}</small></div><button>Load and use</button>`;
    d.querySelector('b').textContent = p.name;
    d.querySelector('button').addEventListener('click', async () => {
      const r = await api('/api/policy/load', { path: p.path });
      buildScene(r.scene); setParams(r.params); buildMotors(r.actuators);
      toast(`Policy "${r.policy}" loaded. Double-click the floor to send the robot somewhere.`, true);
    });
    root.append(d);
  }
}
function updateTrainForm() {
  const algo = $('t-algo').value, rl = algo === 'ppo' || algo === 'distill';
  document.querySelectorAll('.es-only').forEach(e => e.classList.toggle('hidden', rl));
  $('t-pop-label').textContent = rl ? 'Parallel environments' : 'Population';
  $('t-gen-label').textContent = algo === 'ppo' ? 'Iterations' : algo === 'distill' ? 'DAgger rounds' : 'Generations';
}

// ---------------------------------------------------------------- arms
async function refreshObjects() {
  try {
    const objs = await api('/api/objects');
    const sel = $('a-object'), cur = sel.value; sel.innerHTML = '';
    for (const o of objs.filter(o => o.graspable)) sel.add(new Option(`${o.name} (${fixed(o.distance_m, 1)} m)`, o.name));
    if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
  } catch {}
}

// ---------------------------------------------------------------- AI pilot
function onPilot(p) {
  const cost = p.cost_usd != null ? `  $${(+p.cost_usd).toFixed(3)}` : '';
  const tok = p.input_tokens != null ? `  ${p.input_tokens} in / ${p.output_tokens} out tokens` : '';
  $('pi-status').textContent = `${p.state}${p.turns != null ? `  turn ${p.turns}` : ''}${cost}${tok}${p.error ? '\n' + p.error : ''}`;
  if (p.run !== pilotRun) { pilotRun = p.run; pilotSeen = 0; $('pi-log').innerHTML = ''; }
  if ((p.n_events || 0) > pilotSeen) fetchPilotEvents();
}
let fetchingEvents = false;
async function fetchPilotEvents() {
  if (fetchingEvents) return; fetchingEvents = true;
  try {
    const r = await api(`/api/pilot/events?since=${pilotSeen}`);
    for (const ev of r.events) appendEvent(ev);
    pilotSeen = r.next;
  } finally { fetchingEvents = false; }
}
function appendEvent(ev) {
  const d = document.createElement('div'); d.className = `ev ${ev.kind}`;
  let body = '';
  if (ev.kind === 'tool_call') body = `${esc(ev.name)}(${esc(JSON.stringify(ev.args ?? {}))})`;
  else body = esc(ev.text ?? '');
  d.innerHTML = `<small>${esc(ev.kind.replace('_', ' '))}${ev.name && ev.kind !== 'tool_call' ? ' - ' + esc(ev.name) : ''}</small>${body}`;
  if (ev.image_b64) { const img = document.createElement('img'); img.src = `data:image/jpeg;base64,${ev.image_b64}`; img.alt = 'camera view sent to the model'; d.append(img); }
  $('pi-log').append(d);
  d.scrollIntoView({ block: 'nearest' });
}

// ---------------------------------------------------------------- design: topology viewer
const topo = { renderer: null, scene: null, camera: null, controls: null, mesh: null };
function initTopoView() {
  const c = $('topo-view');
  topo.renderer = new THREE.WebGLRenderer({ canvas: c, antialias: true });
  topo.scene = new THREE.Scene(); topo.scene.background = new THREE.Color(0x242830);
  topo.camera = new THREE.PerspectiveCamera(40, c.width / c.height, 0.001, 50);
  topo.camera.position.set(0.5, -0.5, 0.35);
  topo.controls = new OrbitControls(topo.camera, c);
  topo.scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 1.4));
  const l = new THREE.DirectionalLight(0xffffff, 1.2); l.position.set(1, -1, 2); topo.scene.add(l);
  const loop = () => { topo.controls.update(); topo.renderer.render(topo.scene, topo.camera); requestAnimationFrame(loop); };
  loop();
}
function showTopoMesh(m) {
  if (!topo.scene) initTopoView();
  if (topo.mesh) { topo.scene.remove(topo.mesh); topo.mesh.geometry.dispose(); }
  const geo = meshGeometry(m); geo.computeBoundingSphere();
  topo.mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x9fc4ff, roughness: 0.5, metalness: 0.2, side: THREE.DoubleSide }));
  topo.scene.add(topo.mesh);
  const s = geo.boundingSphere; topo.controls.target.copy(s.center);
  topo.camera.position.copy(s.center).add(new THREE.Vector3(1.2, -1.6, 1.0).multiplyScalar(s.radius * 1.4));
}
let topoTimer = null, lastTopoMesh = null;
async function pollTopo() {
  const st = await api('/api/design/topopt/status');
  const h = st.history || [];
  $('tp-info').textContent = st.state === 'idle' ? 'idle' :
    `${st.link || ''}: ${st.state}  iteration ${st.iteration ?? h.length} / ${st.iterations ?? '?'}` +
    (h.length ? `\ncompliance ${fixed(h[h.length - 1].compliance, 4)}  volume ${fixed(h[h.length - 1].volume, 3)}` : '') +
    (st.mass_saving_pct != null ? `\nmass ${fixed(st.mass_before_kg, 3)} kg -> ${fixed(st.mass_after_kg, 3)} kg (${fixed(st.mass_saving_pct, 1)}% lighter)` : '') +
    (st.error ? `\n${st.error}` : '');
  if (st.mesh && st.mesh_id !== lastTopoMesh) { lastTopoMesh = st.mesh_id; showTopoMesh(st.mesh); }
  if (st.state !== 'running' && topoTimer) { clearInterval(topoTimer); topoTimer = null; }
}
let optTimer = null;
async function pollOpt() {
  const st = await api('/api/design/optimize/status');
  drawChart('o-chart', st.history || [], TRAIN_SERIES);
  $('o-info').textContent = st.state === 'idle' ? 'idle' :
    `${st.state}  generation ${st.generation} / ${st.generations}` + (st.best != null ? `  best score ${fixed(st.best, 3)}` : '') +
    (st.best_params ? '\nbest: ' + Object.entries(st.best_params).map(([k, v]) => `${k} ${fixed(v, 3)}`).join(', ') : '') +
    (st.error ? `\n${st.error}` : '');
  if (st.state !== 'running' && optTimer) { clearInterval(optTimer); optTimer = null; }
}

// ---------------------------------------------------------------- wiring
function wire(s) {
  document.querySelectorAll('#tabs button').forEach(b => b.addEventListener('click', () => {
    document.querySelectorAll('#tabs button, .tab').forEach(e => e.classList.remove('active'));
    b.classList.add('active'); $('tab-' + b.dataset.tab).classList.add('active');
    if (b.dataset.tab === 'arms') refreshObjects();
    if (b.dataset.tab === 'design') { pollTopo(); pollOpt(); }
  }));
  $('play').addEventListener('click', () => api('/api/sim', { action: $('play').dataset.running ? 'pause' : 'play' }));
  $('step').addEventListener('click', () => api('/api/sim', { action: 'step' }));
  $('reset').addEventListener('click', () => api('/api/sim', { action: 'reset' }));
  $('speed').addEventListener('change', (e) => api('/api/sim', { speed: +e.target.value }));
  $('p-arms').addEventListener('change', sendParams);
  $('p-hands').addEventListener('change', sendParams);
  $('defaults').addEventListener('click', async () => {
    const d = { arms: true, hands: true }; for (const p of specs) d[p.name] = p.default;
    const r = await api('/api/params', d); buildScene(r.scene); setParams(r.params); buildMotors(r.actuators);
  });
  $('model-load').addEventListener('click', async () => {
    const r = await api('/api/model/load', { path: $('model-path').value });
    buildScene(r.scene); buildMotors(r.actuators); toast('Model loaded', true);
  });
  $('model-builtin').addEventListener('click', async () => applyState(await api('/api/model/builtin', {})));

  // sensors
  const L = s.lidar;
  $('l-enabled').checked = L.enabled; $('l-h').value = L.h_samples; $('l-c').value = L.channels;
  $('l-vmin').value = L.v_min; $('l-vmax').value = L.v_max; $('l-range').value = L.max_range;
  const sendLidar = () => api('/api/lidar', { enabled: $('l-enabled').checked, h_samples: +$('l-h').value,
    channels: +$('l-c').value, v_min: +$('l-vmin').value, v_max: +$('l-vmax').value, max_range: +$('l-range').value });
  ['l-enabled', 'l-h', 'l-c', 'l-vmin', 'l-vmax', 'l-range'].forEach(id => $(id).addEventListener('change', sendLidar));
  const sendCam = () => { api('/api/camera', { enabled: $('c-enabled').checked, name: $('c-name').value, depth: $('c-depth').checked });
    if (!$('c-enabled').checked) $('cam').hidden = true; };
  ['c-enabled', 'c-name', 'c-depth'].forEach(id => $(id).addEventListener('change', sendCam));

  // ROS 2 bridge
  const showRos = (st) => {
    const topics = Object.entries(st.topics || {}).filter(([, t]) => t.sent || t.received)
      .map(([k, t]) => `${k.padEnd(28)} ${t.sent ? `out ${t.sent}` : `in ${t.received}`}${t.rate_hz ? `  ${fixed(t.rate_hz, 1)} Hz` : ''}`);
    $('ros-status').textContent = `${st.state}${st.connected ? ' (connected)' : ''}${st.url ? '  ' + st.url : ''}` +
      (st.error ? `\n${st.error}` : '') + (topics.length ? '\n' + topics.join('\n') : '') +
      (st.rclpy_available === false && $('ros-mode').value === 'rclpy' ? '\nrclpy is not available in this Python' : '');
  };
  let rosTimer = null;
  const pollRos = async () => { const st = await api('/api/ros/status'); showRos(st);
    if (st.state === 'idle' || st.state === 'stopped') { clearInterval(rosTimer); rosTimer = null; } };
  $('ros-connect').addEventListener('click', async () => {
    showRos(await api('/api/ros/connect', { url: $('ros-url').value, mode: $('ros-mode').value }));
    if (!rosTimer) rosTimer = setInterval(pollRos, 1000);
  });
  $('ros-disconnect').addEventListener('click', async () => showRos(await api('/api/ros/disconnect', {})));

  // walking
  $('ctrl-kind').addEventListener('change', async (e) => {
    const r = await api('/api/controller', { kind: e.target.value });
    if (r.kind !== e.target.value || (e.target.value !== 'auto' && r.name !== e.target.value)) toast(`Using "${r.name}": the selected controller is not available for this robot.`);
  });
  document.querySelectorAll('[data-push]').forEach(b => b.addEventListener('click', () =>
    api('/api/push', { direction: b.dataset.push, force: +$('pu-force').value, duration: +$('pu-dur').value })
      .then(r => toast(r.message, true))));
  $('o-random').addEventListener('click', async () => buildScene((await api('/api/obstacles', { action: 'random', n: +$('o-n').value })).scene));
  $('o-clear').addEventListener('click', async () => buildScene((await api('/api/obstacles', { action: 'clear' })).scene));
  $('n-source').addEventListener('change', (e) => api('/api/nav', { source: e.target.value }));
  $('n-showmap').addEventListener('change', (e) => { mapMesh.visible = e.target.checked; });
  $('n-clear').addEventListener('click', () => api('/api/nav', { action: 'clear' }));
  $('n-resetmap').addEventListener('click', () => api('/api/nav', { action: 'reset_map' }));
  $('n-heading').addEventListener('input', (e) => { $('n-heading-o').textContent = e.target.value;
    api('/api/sim', { heading: +e.target.value * Math.PI / 180 }); });

  // arms
  $('a-scene').addEventListener('click', async () => { buildScene((await api('/api/obstacles', { action: 'manipulation' })).scene); });
  $('a-clear').addEventListener('click', async () => buildScene((await api('/api/obstacles', { action: 'clear' })).scene));
  document.querySelectorAll('#tab-arms [data-skill]').forEach(b => b.addEventListener('click', () =>
    api('/api/arms', { side: $('a-side').value, skill: b.dataset.skill }).then(r => r.message && toast(r.message, true))));
  const target = () => [+$('a-x').value, +$('a-y').value, +$('a-z').value];
  $('a-reach').addEventListener('click', () => api('/api/arms', { side: $('a-side').value, skill: 'reach', target: target() }).then(r => toast(r.message, true)));
  $('a-point').addEventListener('click', () => api('/api/arms', { side: $('a-side').value, skill: 'point_at', target: target() }).then(r => toast(r.message, true)));
  $('a-reach-obj').addEventListener('click', () => api('/api/arms', { side: $('a-side').value, skill: 'reach_object', object: $('a-object').value }).then(r => toast(r.message, true)));

  // pilot
  const updPilot = () => { $('pi-claude').hidden = $('pi-provider').value !== 'claude'; };
  $('pi-provider').addEventListener('change', updPilot); updPilot();
  $('pi-start').addEventListener('click', async () => {
    const key = $('pi-key').value.trim();
    await api('/api/pilot/start', {
      provider: $('pi-provider').value, mission: $('pi-mission').value, model: $('pi-model').value,
      effort: $('pi-effort').value, max_turns: +$('pi-turns').value, max_cost_usd: +$('pi-cost').value,
      images: $('pi-images').checked, fallbacks: $('pi-fallbacks').checked, ...(key ? { api_key: key } : {}) });
    $('pi-key').value = '';
  });
  $('pi-stop').addEventListener('click', () => api('/api/pilot/stop', {}));

  // design
  const cadOpts = () => ({ wall_mm: +$('d-wall').value, density: +$('d-density').value });
  $('d-cad').addEventListener('click', async () => {
    const r = await api('/api/design/cad', cadOpts());
    $('d-cad-info').textContent = `CAD robot ${fixed(r.total_mass_kg, 2)} kg (simple shapes: ${fixed(r.primitive_mass_kg, 2)} kg)\n` +
      r.parts.map(p => `${p.name.padEnd(16)} ${fixed(p.mass_kg, 3)} kg`).join('\n');
  });
  $('d-cad-use').addEventListener('click', async () => { applyState(await api('/api/design/cad/use', cadOpts())); toast('Simulating with CAD geometry and masses', true); });
  for (const g of s.design_genes || []) {
    const l = document.createElement('label'); l.className = 'check';
    l.innerHTML = `<input type="checkbox" value="${esc(g.name)}" ${g.default_on ? 'checked' : ''}> ${esc(g.label)}`;
    $('o-genes').append(l);
  }
  $('tp-start').addEventListener('click', async () => {
    await api('/api/design/topopt/start', { link: $('tp-link').value, nel: +$('tp-nel').value, volfrac: +$('tp-vol').value, iterations: +$('tp-iters').value });
    if (!topoTimer) topoTimer = setInterval(pollTopo, 600);
  });
  $('tp-stop').addEventListener('click', () => api('/api/design/topopt/stop', {}));
  $('tp-apply').addEventListener('click', async () => { applyState(await api('/api/design/topopt/apply', {})); toast('Optimised link is now part of the robot', true); });
  $('o-start').addEventListener('click', async () => {
    const genes = [...document.querySelectorAll('#o-genes input:checked')].map(i => i.value);
    await api('/api/design/optimize/start', { algo: $('o-algo').value, genes, generations: +$('o-gens').value,
      population: +$('o-pop').value, workers: +$('o-workers').value });
    if (!optTimer) optTimer = setInterval(pollOpt, 1000);
  });
  $('o-stop').addEventListener('click', () => api('/api/design/optimize/stop', {}));
  $('o-apply').addEventListener('click', async () => { applyState(await api('/api/design/optimize/apply', {})); toast('Best design applied', true); });

  // training
  const T = s.train_defaults, tf = ['population', 'generations', 'episode_seconds', 'threads', 'sigma', 'lr', 'heading_range'];
  tf.forEach(k => { if (T[k] != null) $('t-' + k).value = T[k]; });
  $('t-algo').addEventListener('change', () => {
    const d = (s.train_presets || {})[$('t-algo').value] || {};
    for (const [k, v] of Object.entries(d)) if ($('t-' + k)) $('t-' + k).value = v;
    updateTrainForm();
  });
  $('t-algo').dispatchEvent(new Event('change'));
  $('t-start-train').addEventListener('click', async () => {
    const d = { algo: $('t-algo').value, name: $('t-name').value, evolve_body: $('t-evolve_body').checked,
      continue_from_loaded: $('t-continue').checked };
    tf.forEach(k => { d[k] = +$('t-' + k).value; });
    await api('/api/train/start', d); toast('Training started', true);
  });
  $('t-stop-train').addEventListener('click', () => api('/api/train/stop', {}));
  $('p-unload').addEventListener('click', () => api('/api/policy/unload', {}));
  $('auto-reset').addEventListener('change', (e) => api('/api/sim', { auto_reset: e.target.checked }));
}

(async function main() {
  const s = await api('/api/state');
  $('ver').textContent = `v${s.version} - ${s.native}`;
  buildParams(s.param_spec); wire(s); applyState(s);
  drawChart('chart', s.train.history || [], TRAIN_SERIES); buildPolicies(s.policies);
  connect(); animate();
})();
