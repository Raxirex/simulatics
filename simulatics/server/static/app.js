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
let geomMeshes = [], floorMesh = null, sceneVersion = -1, gridInfo = { size: 24, n: 240 };

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

// path + goals
const pathLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffd34d }));
pathLine.frustumCulled = false; scene.add(pathLine);
const goalGroup = new THREE.Group(); scene.add(goalGroup);
const goalGeo = new THREE.ConeGeometry(0.14, 0.4, 16).rotateX(-Math.PI / 2);
const goalMat = new THREE.MeshStandardMaterial({ color: 0xffd34d, emissive: 0x6b5200 });

function b64(str, Type) {
  const bin = atob(str), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Type(u8.buffer);
}

function buildScene(s) {
  sceneVersion = s.version; gridInfo = s.grid;
  robotGroup.clear(); geomMeshes = []; floorMesh = null;
  const meshGeos = {};
  for (const [id, m] of Object.entries(s.meshes)) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(b64(m.vert, Float32Array), 3));
    g.setIndex(new THREE.BufferAttribute(b64(m.face, Uint32Array), 1));
    g.computeVertexNormals(); meshGeos[id] = g;
  }
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
    let mesh;
    if (!geo || g.rgba[3] === 0 || g.group > 2) { mesh = new THREE.Object3D(); }
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
    (i.trainable ? '' : '\nNo free-floating base or motors: locomotion training is disabled.');
  const sel = $('c-name'); const cur = sel.value; sel.innerHTML = '';
  for (const n of (s.cameras.length ? s.cameras : ['free'])) sel.add(new Option(n, n));
  if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
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

// double-click the floor to set a navigation goal
const ray = new THREE.Raycaster();
canvas.addEventListener('dblclick', async (e) => {
  if (!floorMesh) return;
  const r = canvas.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  const hit = ray.intersectObject(floorMesh)[0];
  if (hit) await api('/api/goal', { x: hit.point.x, y: hit.point.y, append: e.shiftKey });
});

// ---------------------------------------------------------------- live stream
let lastTrainGen = -1, lastNavKey = '';
function onFrame(f) {
  if (f.v !== sceneVersion) { refreshScene(); return; }
  const n = Math.min(geomMeshes.length, f.pos.length / 3);
  for (let i = 0; i < n; i++) {
    const m = geomMeshes[i];
    m.position.set(f.pos[3 * i], f.pos[3 * i + 1], f.pos[3 * i + 2]);
    m.quaternion.set(f.quat[4 * i + 1], f.quat[4 * i + 2], f.quat[4 * i + 3], f.quat[4 * i]);
  }
  rootPos.set(f.root[0], f.root[1], f.root[2] || 0.9);
  $('r-time').textContent = f.t.toFixed(1);
  $('r-height').textContent = f.root[2].toFixed(2);
  $('r-speed').textContent = f.speed.toFixed(2);
  $('r-policy').textContent = f.policy || 'none';
  $('play').textContent = f.running ? 'Pause' : 'Play';
  $('play').dataset.running = f.running ? '1' : '';
  if (f.error) toast('Simulation error: ' + f.error);
  $('c-error').textContent = f.camera_error ? 'Camera rendering failed: ' + f.camera_error : '';
  $('s-body').textContent = `root height ${f.root[2].toFixed(3)} m\nposition x ${f.root[0].toFixed(2)}  y ${f.root[1].toFixed(2)}\n` +
    Object.entries(f.touch).map(([k, v]) => `${k.replace(/_/g, ' ')} ${v} N`).join('\n') + `\nfalls ${f.falls}`;
  // navigation overlay
  const nav = f.nav, key = JSON.stringify([nav.path, nav.goals]);
  if (key !== lastNavKey) {
    lastNavKey = key;
    pathLine.geometry.setFromPoints(nav.path.map(p => new THREE.Vector3(p[0], p[1], 0.04)));
    goalGroup.clear();
    for (const g of nav.goals) { const m = new THREE.Mesh(goalGeo, goalMat); m.position.set(g[0], g[1], 0.2); goalGroup.add(m); }
  }
  $('n-status').textContent = `behaviour tree node: ${nav.node}\ngoals queued: ${nav.goals.length}` +
    (nav.message ? `\n${nav.message}` : '') + (nav.goals.length && !f.policy ? '\nNo policy loaded: the robot cannot walk to the goal yet.' : '');
  if (f.train) onTrain(f.train);
}
function onTrain(t) {
  const last = t.last;
  $('t-status').textContent = t.state === 'idle' ? 'idle' :
    `${t.name}: ${t.state}  generation ${t.generation} / ${t.generations}` +
    (t.best != null ? `\nbest fitness ${t.best.toFixed(2)}` : '') +
    (last ? `   ${(last.sps / 1000).toFixed(0)}k physics steps/s` : '') +
    `\n${(t.total_steps / 1e6).toFixed(1)} M steps simulated` + (t.error ? `\n${t.error}` : '');
  if (t.generation !== lastTrainGen) { lastTrainGen = t.generation; refreshTraining(); }
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

// ---------------------------------------------------------------- panels
let refreshing = false;
async function refreshScene() {
  if (refreshing) return; refreshing = true;
  try { applyState(await api('/api/state')); } finally { refreshing = false; }
}
function applyState(s) {
  buildScene(s.scene); setParams(s.params); buildMotors(s.actuators);
}
const fmt = (v, spec) => (spec.max - spec.min < 2 ? v.toFixed(3) : spec.max > 100 ? v.toFixed(0) : v.toFixed(2));
let specs = [];
function buildParams(spec) {
  specs = spec; const root = $('param-groups'); root.innerHTML = '';
  for (const group of [...new Set(spec.map(p => p.group))]) {
    const h = document.createElement('h3'); h.textContent = group; root.append(h);
    for (const p of spec.filter(q => q.group === group)) {
      const row = document.createElement('label'); row.className = 'slider';
      row.innerHTML = `<span>${p.label}</span><input type="range" id="p-${p.name}" min="${p.min}" max="${p.max}" step="${(p.max - p.min) / 200}"><output></output>`;
      const input = row.querySelector('input'), out = row.querySelector('output');
      input.addEventListener('input', () => { out.textContent = fmt(+input.value, p); sendParams(); });
      root.append(row);
    }
  }
}
function setParams(params) {
  for (const p of specs) {
    const input = $('p-' + p.name); if (!input) continue;
    input.value = params[p.name]; input.nextElementSibling.textContent = fmt(params[p.name], p);
  }
  $('p-arms').checked = params.arms;
}
const sendParams = debounce(async () => {
  const d = { arms: $('p-arms').checked };
  for (const p of specs) d[p.name] = +$('p-' + p.name).value;
  const r = await api('/api/params', d);
  buildScene(r.scene); buildMotors(r.actuators);
}, 120);

function buildMotors(list) {
  const body = document.querySelector('#motors tbody'); body.innerHTML = '';
  for (const a of list) {
    const tr = document.createElement('tr');
    const num = (field, v, step) => `<td><input type="number" data-f="${field}" value="${+v.toPrecision(4)}" step="${step}" aria-label="${a.name} ${field}"></td>`;
    tr.innerHTML = `<td title="${a.name}">${a.name.replace(/_/g, ' ')}</td>` +
      (a.servo ? num('kp', a.kp, 10) + num('kv', a.kv, 1) : num('gear', a.gear, 1) + '<td></td>') +
      num('torque', a.torque, 5) + num('damping', a.damping, 0.1) + num('armature', a.armature, 0.005) +
      `<td><input type="range" data-f="ctrl" min="${a.lo}" max="${a.hi}" step="${(a.hi - a.lo) / 100 || 0.01}" value="${a.ctrl}" aria-label="${a.name} target"></td>`;
    tr.querySelectorAll('input').forEach(inp => inp.addEventListener(inp.type === 'range' ? 'input' : 'change', () =>
      api('/api/actuator', { index: a.index, field: inp.dataset.f, value: +inp.value })));
    body.append(tr);
  }
}

function drawChart(hist) {
  const c = $('chart'), g = c.getContext('2d'), W = c.width, H = c.height, pad = 34;
  g.clearRect(0, 0, W, H);
  g.font = '11px system-ui'; g.fillStyle = '#98a0ad';
  if (!hist.length) { g.fillText('No training data yet', pad, H / 2); return; }
  const all = hist.flatMap(h => [h.best, h.mean, h.center]);
  let lo = Math.min(...all), hi = Math.max(...all); if (hi - lo < 1e-6) hi = lo + 1;
  const x = (i) => pad + (W - pad - 8) * (hist.length === 1 ? 0.5 : i / (hist.length - 1));
  const y = (v) => H - 20 - (H - 30) * (v - lo) / (hi - lo);
  g.strokeStyle = '#333843'; g.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4; g.beginPath(); g.moveTo(pad, y(v)); g.lineTo(W - 8, y(v)); g.stroke();
    g.fillText(v.toFixed(1), 2, y(v) + 4);
  }
  g.fillText(`generation ${hist[0].gen} to ${hist[hist.length - 1].gen}`, pad, H - 4);
  for (const [key, col] of [['mean', '#98a0ad'], ['center', '#4f9cf9'], ['best', '#4cc38a']]) {
    g.strokeStyle = col; g.lineWidth = 2; g.beginPath();
    hist.forEach((h, i) => (i ? g.lineTo(x(i), y(h[key])) : g.moveTo(x(i), y(h[key]))));
    g.stroke();
  }
}
async function refreshTraining() {
  const [st, pol] = await Promise.all([api('/api/train/status'), api('/api/policies')]);
  drawChart(st.history); buildPolicies(pol);
}
function buildPolicies(list) {
  const root = $('policies'); root.innerHTML = '';
  if (!list.length) { root.innerHTML = '<p class="hint">None yet. Train one above, or run the bundled pretrained walker if present.</p>'; return; }
  for (const p of list) {
    const d = document.createElement('div'); d.className = 'policy';
    d.innerHTML = `<div><b></b><small>${p.algo || ''} gen ${p.generation ?? '?'}  fitness ${p.fitness != null ? p.fitness.toFixed(1) : '?'}</small></div><button>Load and play</button>`;
    d.querySelector('b').textContent = p.name;
    d.querySelector('button').addEventListener('click', async () => {
      const r = await api('/api/policy/load', { path: p.path });
      buildScene(r.scene); setParams(r.params); buildMotors(r.actuators);
      await api('/api/sim', { action: 'play' });
      toast(`Policy "${r.policy}" loaded`, true);
    });
    root.append(d);
  }
}

// ---------------------------------------------------------------- wiring
function wire(s) {
  document.querySelectorAll('#tabs button').forEach(b => b.addEventListener('click', () => {
    document.querySelectorAll('#tabs button, .tab').forEach(e => e.classList.remove('active'));
    b.classList.add('active'); $('tab-' + b.dataset.tab).classList.add('active');
  }));
  $('play').addEventListener('click', () => api('/api/sim', { action: $('play').dataset.running ? 'pause' : 'play' }));
  $('step').addEventListener('click', () => api('/api/sim', { action: 'step' }));
  $('reset').addEventListener('click', () => api('/api/sim', { action: 'reset' }));
  $('speed').addEventListener('change', (e) => api('/api/sim', { speed: +e.target.value }));
  $('p-arms').addEventListener('change', sendParams);
  $('defaults').addEventListener('click', async () => {
    const d = { arms: true }; for (const p of specs) d[p.name] = p.default;
    const r = await api('/api/params', d); buildScene(r.scene); setParams(r.params); buildMotors(r.actuators);
  });
  $('model-load').addEventListener('click', async () => {
    const r = await api('/api/model/load', { path: $('model-path').value });
    buildScene(r.scene); buildMotors(r.actuators); toast('Model loaded', true);
  });
  $('model-builtin').addEventListener('click', async () => applyState({ ...(await api('/api/model/builtin', {})) }));

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

  // planning
  $('o-random').addEventListener('click', async () => buildScene((await api('/api/obstacles', { action: 'random', n: +$('o-n').value })).scene));
  $('o-clear').addEventListener('click', async () => buildScene((await api('/api/obstacles', { action: 'clear' })).scene));
  $('n-source').addEventListener('change', (e) => api('/api/nav', { source: e.target.value }));
  $('n-showmap').addEventListener('change', (e) => { mapMesh.visible = e.target.checked; });
  $('n-clear').addEventListener('click', () => api('/api/nav', { action: 'clear' }));
  $('n-resetmap').addEventListener('click', () => api('/api/nav', { action: 'reset_map' }));
  $('n-heading').addEventListener('input', (e) => { $('n-heading-o').textContent = e.target.value;
    api('/api/sim', { heading: +e.target.value * Math.PI / 180 }); });

  // training
  const T = s.train_defaults, tf = ['population', 'generations', 'episode_seconds', 'threads', 'sigma', 'lr', 'heading_range'];
  tf.forEach(k => { $('t-' + k).value = T[k]; });
  $('t-start').addEventListener('click', async () => {
    const d = { algo: $('t-algo').value, name: $('t-name').value, evolve_body: $('t-evolve_body').checked,
      continue_from_loaded: $('t-continue').checked };
    tf.forEach(k => { d[k] = +$('t-' + k).value; });
    await api('/api/train/start', d); toast('Training started', true);
  });
  $('t-stop').addEventListener('click', () => api('/api/train/stop', {}));
  $('p-unload').addEventListener('click', () => api('/api/policy/unload', {}));
  $('auto-reset').addEventListener('change', (e) => api('/api/sim', { auto_reset: e.target.checked }));
}

(async function main() {
  const s = await api('/api/state');
  $('ver').textContent = `v${s.version} - ${s.native}`;
  buildParams(s.param_spec); wire(s); applyState(s);
  drawChart(s.train.history); buildPolicies(s.policies);
  connect(); animate();
})();
