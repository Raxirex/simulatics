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

// SLAM: ghost arrow at the estimated pose, uncertainty ellipse, and three trails
const slamGroup = new THREE.Group(); scene.add(slamGroup);
const slamArrow = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.5, 16).rotateZ(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ color: 0xc58cff, transparent: true, opacity: 0.75 }));
const slamEllipse = new THREE.Mesh(new THREE.RingGeometry(0.95, 1.0, 48),
  new THREE.MeshBasicMaterial({ color: 0xc58cff, transparent: true, opacity: 0.8, side: THREE.DoubleSide }));
const trail = (color) => { const l = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color }));
  l.frustumCulled = false; return l; };
const slamTraj = trail(0xc58cff), odomTraj = trail(0xf2695c), trueTraj = trail(0x4cc38a);
slamGroup.add(slamArrow, slamEllipse, slamTraj, odomTraj, trueTraj);
slamGroup.visible = false;
let slamScans = -1;
function onSlam(s, truePts) {
  slamGroup.visible = !!s && $('n-showslam').checked && !ws.on;
  if (!s) { $('n-slam').textContent = ''; return; }
  const [x, y, yaw] = s.pose;
  slamArrow.position.set(x, y, 0.06); slamArrow.rotation.set(0, 0, yaw);
  const [a, b, ang] = s.ellipse || [0.02, 0.02, 0];
  slamEllipse.position.set(x, y, 0.03); slamEllipse.rotation.set(0, 0, ang || 0);
  slamEllipse.scale.set(Math.max(0.03, 3 * a), Math.max(0.03, 3 * b), 1);   // 3-sigma
  if (s.scans !== slamScans) {
    slamScans = s.scans;
    const pts = (arr, z) => (arr || []).map(p => new THREE.Vector3(p[0], p[1], z));
    slamTraj.geometry.setFromPoints(pts(s.traj, 0.035));
    odomTraj.geometry.setFromPoints(pts(s.odom_traj, 0.03));
    trueTraj.geometry.setFromPoints(pts(truePts, 0.025));
  }
  $('n-slam').textContent = `SLAM ${s.state}  score ${fixed(s.score, 2)}  sigma ${fixed(100 * a, 1)} cm / ${fixed(s.sigma_yaw_deg, 1)} deg` +
    `  ${s.scans} scans, ${fixed(s.match_ms, 1)} ms\npurple: SLAM estimate   red: odometry only   green: true path`;
}

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
  hasCadVisuals = !s.info.custom && s.geoms.some(g => g.group === 3 && g.type === 'mesh');
  const hideCollision = !!s.info.custom;   // Menagerie convention: visuals in group 2, collision in group 3
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
    if (!geo || g.rgba[3] === 0 || g.group > 3 || hiddenPrimitive || (hideCollision && g.group === 3 && g.robot)) { mesh = new THREE.Object3D(); }
    else if (g.type === 'plane') {
      mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x20242b, roughness: 1 }));
      mesh.receiveShadow = true; floorMesh = mesh;
    } else {
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]), roughness: 0.55, metalness: 0.1,
        transparent: g.rgba[3] < 1, opacity: g.rgba[3] });
      mesh = new THREE.Mesh(geo, mat); mesh.castShadow = true; mesh.receiveShadow = true;
    }
    mesh.userData = { index: geomMeshes.length, robot: g.robot, name: g.name,
                      acc: g.name && g.name.startsWith('acc_') ? g.name.slice(4) : null };
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
  if (ws.on) highlightAccessory();
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
  const sk = JSON.stringify([steps, c.footsteps_z]);
  if (sk !== lastStepsKey) {
    lastStepsKey = sk; stepGroup.clear();
    const zs = c.footsteps_z || [];
    steps.forEach((s, i) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(0.22, 0.09), stepMats[s[3] ? 1 : 0]);
      m.position.set(s[0], s[1], (zs[i] || 0) + 0.012); m.rotation.set(0, 0, s[2]); stepGroup.add(m);
    });
  }
  const err = c.goal_error_m != null ? `  error ${fixed(c.goal_error_m, 3)} m` : '';
  const herr = c.heading_error_deg != null ? ` / ${fixed(c.heading_error_deg, 1)} deg` : '';
  $('n-status').textContent = `controller: ${c.name || '-'} (${c.mode || '-'})${c.message ? ' - ' + c.message : ''}\n` +
    `planner: ${nav.state}, node ${nav.node}, goals queued ${nav.goals.length}${nav.message ? '\n' + nav.message : ''}${err}${herr}`;
  $('hud').textContent = nav.goals.length || nav.state === 'reached' || nav.state === 'failed' || nav.state === 'fallen'
    ? `${nav.state}${nav.message ? ': ' + nav.message : ''}${err}${herr}` : '';
  onSlam(f.slam, f.slam_true);
  if (f.arms) onArms(f.arms);
  if (f.train) onTrain(f.train);
  if (f.pilot) onPilot(f.pilot);
}
function onTrain(t) {
  const last = t.last;
  $('t-status').textContent = t.state === 'idle' ? 'idle' :
    `${t.name}: ${t.state}  ${['ppo', 'distill', 'imitation'].includes(t.kind) ? 'iteration' : 'generation'} ${t.generation} / ${t.generations}` +
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
  drawChart('chart', st.history || [], TRAIN_SERIES, ['ppo', 'distill', 'imitation'].includes(st.kind) ? 'iteration' : 'generation');
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
  const algo = $('t-algo').value, rl = algo === 'ppo' || algo === 'distill' || algo === 'imitation';
  document.querySelectorAll('.es-only').forEach(e => e.classList.toggle('hidden', rl));
  $('t-pop-label').textContent = rl ? 'Parallel environments' : 'Population';
  $('t-gen-label').textContent = algo === 'ppo' || algo === 'imitation' ? 'Iterations'
    : algo === 'distill' ? 'DAgger rounds' : 'Generations';
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
    if (b.dataset.tab === 'workshop') enterWorkshop(); else if (ws.on) exitWorkshop();
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

  // robot library
  const showRobots = async () => {
    const r = await api('/api/robots'); const sel = $('rb-name'); sel.innerHTML = '';
    sel.add(new Option('Built-in parametric humanoid (walks)', 'builtin'));
    for (const x of r.robots) sel.add(new Option(`${x.label} - ${x.license}${x.downloaded ? '' : ` (download ~${x.size_mb} MB)`}`, x.name));
    sel.value = r.current;
    const x = r.robots.find(y => y.name === r.current);
    $('rb-info').textContent = x ? `Loaded: ${x.label}. Licence ${x.license}; files in robots/${x.name}/ (LICENSE, SOURCE.txt).` : 'Loaded: built-in humanoid.';
  };
  showRobots().catch(() => {});
  $('rb-load').addEventListener('click', async () => {
    const name = $('rb-name').value;
    $('rb-info').textContent = name === 'builtin' ? 'loading...' : 'loading (downloading first if needed, this can take a minute)...';
    const r = await api('/api/robots/load', { name });
    applyState(r); $('rb-stand').checked = !!r.support_stand; toast(r.message, true); showRobots();
    if (ws.on) { renderAccessories(r); setTimeout(() => { setView('front'); buildBackdrop(); }, 300); }
  });
  $('rb-stand').addEventListener('change', async (e) => { const r = await api('/api/robots/support', { on: e.target.checked }); applyState(r); });
  $('ws-deflidar').addEventListener('change', async (e) => {
    const r = await api('/api/accessories/default_lidar', { on: e.target.checked }); applyState(r); renderAccessories(r);
  });

  // workshop
  document.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));
  $('ws-grid').addEventListener('change', (e) => { if (ws.backdrop) ws.backdrop.visible = e.target.checked; });

  // terrain courses
  $('tc-build').addEventListener('click', async () => {
    const r = await api('/api/obstacles', { action: 'terrain', kind: $('tc-kind').value });
    buildScene(r.scene);
    await api('/api/goal', { x: r.goal[0], y: r.goal[1], yaw_deg: 0 });
    await api('/api/sim', { action: 'play' });
    toast(`Walking across the ${$('tc-kind').selectedOptions[0].text.toLowerCase()} to (${fixed(r.goal[0], 1)}, ${fixed(r.goal[1], 1)})`, true);
  });

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
      continue_from_loaded: $('t-continue').checked, motion: $('m-clip').value };
    tf.forEach(k => { d[k] = +$('t-' + k).value; });
    await api('/api/train/start', d); toast('Training started', true);
  });
  $('t-stop-train').addEventListener('click', () => api('/api/train/stop', {}));

  // motions (imitation)
  const loadMotions = async (select) => {
    const list = await api('/api/motions');
    const sel = $('m-clip'), cur = select || sel.value; sel.innerHTML = '';
    for (const m of list) sel.add(new Option(`${m.name} (${fixed(m.duration_s, 1)} s${m.library ? '' : ', imported'})`, m.name));
    if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
  };
  loadMotions().catch(() => {});
  const play = (mode) => api('/api/motions/play', { motion: $('m-clip').value, mode }).then(r => toast(r.message, true));
  $('m-preview').addEventListener('click', () => play('preview'));
  $('m-pd').addEventListener('click', () => play('pd'));
  $('m-import').addEventListener('click', async () => {
    $('m-info').textContent = 'retargeting...';
    const r = await api('/api/motions/import', { path: $('m-bvh').value });
    const rep = r.report || {};
    $('m-info').textContent = `imported "${r.name}": key-point error mean ${fixed(100 * (rep.key_error_mean_m ?? NaN), 1)} cm, ` +
      `max ${fixed(100 * (rep.key_error_max_m ?? NaN), 1)} cm; leg scale ${fixed(rep.leg_scale, 2)}`;
    await loadMotions(r.name);
  });
  $('p-unload').addEventListener('click', () => api('/api/policy/unload', {}));
  $('auto-reset').addEventListener('change', (e) => api('/api/sim', { auto_reset: e.target.checked }));
}


// ---------------------------------------------------------------- workshop: mount accessories
const ws = { on: false, saved: null, sel: null, items: [], types: {}, move: null, backdrop: null };
const wsGhost = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 0.14, 0xffd34d, 0.05, 0.035);
wsGhost.visible = false; scene.add(wsGhost);
const ACC_COLORS = { lidar: '#26bf73', camera: '#2e3342', payload: '#d19a4d' };

function textSprite(text, color = '#c9d1dc') {
  const c = document.createElement('canvas'); c.width = 256; c.height = 64;
  const g = c.getContext('2d'); g.font = '28px system-ui'; g.fillStyle = color; g.textBaseline = 'middle';
  g.fillText(text, 6, 32);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false, transparent: true }));
  sp.scale.set(0.32, 0.08, 1); return sp;
}
function buildBackdrop() {
  if (ws.backdrop) scene.remove(ws.backdrop);
  const grp = new THREE.Group(), cx = rootPos.x - 0.8, cy = rootPos.y;
  // vertical grid behind the robot in the y-z plane: 10 cm cells, 2.4 m wide, 0 to 2.4 m high
  const grid = new THREE.GridHelper(2.4, 24, 0x5a6372, 0x353b46);
  grid.rotation.z = Math.PI / 2; grid.position.set(cx, cy, 1.2); grp.add(grid);
  const line = (a, b, color) => grp.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]),
    new THREE.LineBasicMaterial({ color })));
  line(new THREE.Vector3(cx + 0.002, cy, 0), new THREE.Vector3(cx + 0.002, cy, 2.4), 0x4f9cf9);       // centre line (y = robot)
  line(new THREE.Vector3(cx + 0.002, cy - 1.2, 0.002), new THREE.Vector3(cx + 0.002, cy + 1.2, 0.002), 0x4f9cf9);
  for (let z = 0.2; z <= 2.4001; z += 0.2) {                     // height labels on both edges
    for (const side of [-1, 1]) {
      const t = textSprite(`z ${z.toFixed(1)} m`); t.position.set(cx + 0.01, cy + side * 1.05, z + 0.03); grp.add(t);
    }
  }
  for (let y = -1.2; y <= 1.2001; y += 0.4) {                    // side (horizontal) labels along the floor
    const t = textSprite(`y ${(cy + y).toFixed(1)}`); t.position.set(cx + 0.01, cy + y, -0.06); grp.add(t);
  }
  grp.visible = $('ws-grid').checked; ws.backdrop = grp; scene.add(grp);
}
function setView(kind) {
  const h = Math.max(0.25, rootPos.z), t = new THREE.Vector3(rootPos.x, rootPos.y, 0.75 * h), d = Math.max(1.1, 2.5 * h);
  const off = { front: [d, 0, 0.2], side: [0, -d, 0.2], back: [-d, 0, 0.2], top: [0.01, 0, d] }[kind] || [d, 0, 0.2];
  controls.target.copy(t); camera.position.set(t.x + off[0], t.y + off[1], t.z + off[2]); controls.update();
}
async function enterWorkshop() {
  if (!ws.on) {
    ws.on = true;
    ws.saved = { pos: camera.position.clone(), target: controls.target.clone(), follow: $('follow').checked };
    $('follow').checked = false;
  }
  renderAccessories(await api('/api/workshop', { enter: true }));
  setTimeout(() => { setView('front'); buildBackdrop(); }, 150);   // after the reset pose reaches the view
}
function exitWorkshop() {
  ws.on = false; wsGhost.visible = false; ws.move = null;
  if (ws.backdrop) ws.backdrop.visible = false;
  if (ws.saved) { camera.position.copy(ws.saved.pos); controls.target.copy(ws.saved.target); $('follow').checked = ws.saved.follow; }
  highlightAccessory(null);
}
function robotHit(e, includeAccessories = false) {
  const r = canvas.getBoundingClientRect();
  ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
  const targets = geomMeshes.filter(m => m.isMesh && m.userData.robot && (includeAccessories || !m.userData.acc));
  return ray.intersectObjects(targets, false)[0] || null;
}
function hitNormal(hit) { return hit.face.normal.clone().transformDirection(hit.object.matrixWorld).normalize(); }
function showGhost(hit) {
  if (!hit) { wsGhost.visible = false; return; }
  wsGhost.position.copy(hit.point); wsGhost.setDirection(hitNormal(hit)); wsGhost.visible = true;
}
function hoverText(hit) {
  if (!hit) return 'Point at the robot to read coordinates.';
  const p = hit.point;
  return `x ${fixed(p.x, 3)}  y ${fixed(p.y, 3)}  z ${fixed(p.z, 3)} m  on ${hit.object.userData.name || 'robot'}`;
}
async function placeAt(hit, type, id) {
  const n = hitNormal(hit);
  const r = await api('/api/accessories/place', { type, id, geom: hit.object.userData.index,
    point: [hit.point.x, hit.point.y, hit.point.z], normal: [n.x, n.y, n.z], snap: +$('ws-snap').value });
  ws.sel = r.placed.id;
  applyState(r); renderAccessories(r);
  toast(`${ws.types[r.placed.type]?.label || r.placed.type} mounted on ${r.placed.body}`, true);
}
function highlightAccessory() {
  for (const m of geomMeshes) {
    if (!m.isMesh || !m.userData.acc) continue;
    m.material.emissive = new THREE.Color(m.userData.acc === ws.sel && ws.on ? 0x3366aa : 0x000000);
  }
}
function renderAccessories(info) {
  ws.items = info.items || []; ws.types = info.types || ws.types;
  const pal = $('ws-palette');
  if (!pal.children.length) {
    for (const [k, t] of Object.entries(ws.types)) {
      const d = document.createElement('div'); d.className = 'pal'; d.draggable = true; d.title = t.hint;
      d.innerHTML = `<span class="sw" style="background:${ACC_COLORS[k] || '#888'}"></span><b>${esc(t.label)}</b><small>drag onto the robot</small>`;
      d.addEventListener('dragstart', (e) => { e.dataTransfer.setData('text/plain', 'acc:' + k); e.dataTransfer.effectAllowed = 'copy'; ws.dragType = k; });
      d.addEventListener('dragend', () => { ws.dragType = null; wsGhost.visible = false; });
      pal.append(d);
    }
  }
  const list = $('ws-list'); list.innerHTML = '';
  if (!info.editable) list.innerHTML = '<p class="hint">Accessories can be mounted on the built-in robot only.</p>';
  for (const a of ws.items) {
    const row = document.createElement('div'); row.className = 'acc' + (a.id === ws.sel ? ' sel' : '');
    const cm = (v) => (+v * 100).toFixed(1);
    row.innerHTML = `<b>${esc(a.label)}</b> <small>on ${esc(a.body)}${a.world ? ` - world ${a.world.map(v => fixed(v, 2)).join(', ')}` : ''}</small>
      <div class="grid3"><label>x cm <input type="number" step="0.5" data-k="0" value="${cm(a.pos[0])}"></label>
      <label>y cm <input type="number" step="0.5" data-k="1" value="${cm(a.pos[1])}"></label>
      <label>z cm <input type="number" step="0.5" data-k="2" value="${cm(a.pos[2])}"></label></div>
      <div class="row">${a.type === 'camera' ? `<label>Field of view <input type="number" data-f="fovy" min="10" max="160" value="${a.fovy}"></label>` : ''}
      <label>Mass kg <input type="number" data-f="mass" step="0.05" min="0.001" value="${a.mass}"></label>
      <button data-del>Remove</button></div>`;
    row.addEventListener('click', (e) => { if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'BUTTON') { ws.sel = a.id; renderAccessories(info); highlightAccessory(); } });
    row.querySelectorAll('input[data-k]').forEach(inp => inp.addEventListener('change', async () => {
      const pos = [...row.querySelectorAll('input[data-k]')].map(i => +i.value / 100);
      const r = await api('/api/accessories/update', { id: a.id, pos }); applyState(r); renderAccessories(r);
    }));
    row.querySelectorAll('input[data-f]').forEach(inp => inp.addEventListener('change', async () => {
      const r = await api('/api/accessories/update', { id: a.id, [inp.dataset.f]: +inp.value }); applyState(r); renderAccessories(r);
    }));
    row.querySelector('[data-del]').addEventListener('click', async () => {
      const r = await api('/api/accessories/delete', { id: a.id }); if (ws.sel === a.id) ws.sel = null;
      applyState(r); renderAccessories(r);
    });
    list.append(row);
  }
  $('ws-warn').textContent = (info.warnings || []).join('\n');
  if (info.default_lidar != null) $('ws-deflidar').checked = info.default_lidar;
  $('ws-deflidar').disabled = !!info.robot && info.robot !== 'builtin';
  $('ws-lidarnote').textContent = info.lidar_mounted === false ? 'No lidar mounted: drag the Lidar from the palette onto the robot to add one.' : '';
  highlightAccessory();
}
// drop from the palette
canvas.addEventListener('dragover', (e) => {
  if (!ws.on) return;
  e.preventDefault(); e.dataTransfer.dropEffect = 'copy';
  const hit = robotHit(e); showGhost(hit); $('ws-hover').textContent = hoverText(hit);
});
canvas.addEventListener('drop', async (e) => {
  if (!ws.on) return;
  e.preventDefault(); wsGhost.visible = false;
  const data = e.dataTransfer.getData('text/plain') || ('acc:' + (ws.dragType || ''));
  const hit = robotHit(e);
  if (!data.startsWith('acc:') || !hit) { toast('Drop accessories onto the robot'); return; }
  await placeAt(hit, data.slice(4));
});
// hover read-out; drag a mounted accessory to move it (captured before OrbitControls)
canvas.addEventListener('pointerdown', (e) => {
  if (!ws.on || e.button !== 0 || e.ctrlKey) return;
  const hit = robotHit(e, true);
  if (!hit || !hit.object.userData.acc) return;
  e.stopImmediatePropagation(); e.preventDefault();
  ws.move = { id: hit.object.userData.acc, x: e.clientX, y: e.clientY, moved: false, hit: null };
  ws.sel = ws.move.id; highlightAccessory(); renderAccessories({ items: ws.items, types: ws.types, editable: true });
  try { canvas.setPointerCapture(e.pointerId); } catch {}
}, { capture: true });
canvas.addEventListener('pointermove', (e) => {
  if (!ws.on) return;
  if (ws.move) {
    if (Math.hypot(e.clientX - ws.move.x, e.clientY - ws.move.y) > 4) ws.move.moved = true;
    const hit = robotHit(e); ws.move.hit = hit; showGhost(hit); $('ws-hover').textContent = hoverText(hit);
    return;
  }
  $('ws-hover').textContent = hoverText(robotHit(e, true));
});
canvas.addEventListener('pointerup', async () => {
  if (!ws.on || !ws.move) return;
  const m = ws.move; ws.move = null; wsGhost.visible = false;
  if (m.moved && m.hit) await placeAt(m.hit, null, m.id);
});

(async function main() {
  const s = await api('/api/state');
  $('ver').textContent = `v${s.version} - ${s.native}`;
  buildParams(s.param_spec); wire(s); applyState(s);
  drawChart('chart', s.train.history || [], TRAIN_SERIES); buildPolicies(s.policies);
  connect(); animate();
})();
