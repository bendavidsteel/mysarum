import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const PROGRAMS = ['Sparkles', 'Excitable', 'Reaction-diffusion', 'Cyclic', 'Neural CA'];
const MAPS = ['twilight', 'magma', 'inferno', 'viridis', 'ember'];
const GROUPS = [
    ['General', /^(?!mood_|comet_|ex_|gs_|cy_|nca_)/, -1],
    ['Sparkles', /^(mood_|comet_)/, 0],
    ['Excitable', /^ex_/, 1],
    ['Reaction-diffusion', /^gs_/, 2],
    ['Cyclic', /^cy_/, 3],
    ['Neural CA', /^nca_/, 4],
];
const $ = id => document.getElementById(id);

// ---- the firmware, compiled to wasm ----

let sim = null;       // { e, info, index }
let stamp = '';
const touched = new Set();  // settings changed here since params.h was last loaded
const ui = {};              // name -> { label, inputs[] }

function cstr(ptr) {
    const bytes = new Uint8Array(sim.e.memory.buffer, ptr);
    let end = 0;
    while (bytes[end]) end++;
    return new TextDecoder().decode(bytes.subarray(0, end));
}

async function load() {
    const res = await fetch('lantern.wasm?' + Date.now());
    if (!res.ok) throw new Error(await res.text());
    const stub = new Proxy({}, { get: () => () => 0 });
    const { instance } = await WebAssembly.instantiate(await res.arrayBuffer(), { wasi_snapshot_preview1: stub });
    const e = instance.exports;
    e._initialize();
    const previous = sim;
    sim = { e, info: [], index: {} };
    for (let i = 0; i < e.sim_param_count(); i++) {
        const name = cstr(e.sim_param_name(i));
        sim.index[name] = i;
        sim.info.push({ name, isInt: !!e.sim_param_is_int(i), lo: e.sim_param_lo(i), hi: e.sim_param_hi(i),
                        boot: e.sim_param_get(i) });
    }
    // Carry over what was changed here, unless params.h now says the same.
    for (const name of [...touched]) {
        const i = sim.index[name];
        if (i === undefined || !previous) { touched.delete(name); continue; }
        const v = previous.e.sim_param_get(previous.index[name]);
        if (v === sim.info[i].boot) touched.delete(name);
        else e.sim_param_set(i, v);
    }
    e.sim_reset((Math.random() * 2 ** 32) >>> 0);
    setupScene();
    buildSettings();
    $('error').style.display = 'none';
}

const get = name => sim.e.sim_param_get(sim.index[name]);

function set(name, value, { send = true } = {}) {
    const i = sim.index[name];
    sim.e.sim_param_set(i, value);
    const v = get(name);
    if (v === sim.info[i].boot) touched.delete(name);
    else touched.add(name);
    refresh(name);
    if (send) queueSend(name, v);
}

// ---- settings panel ----

function fmt(v, info) {
    if (info.isInt) return String(v);
    return String(+v.toPrecision(4));
}

function buildSettings() {
    const open = new Set([...document.querySelectorAll('#params details[open]')].map(d => d.dataset.group));
    const root = $('params');
    root.innerHTML = '';
    for (const key in ui) delete ui[key];
    for (const [title, pattern, program] of GROUPS) {
        const details = document.createElement('details');
        details.dataset.group = title;
        details.dataset.program = program;
        details.open = open.has(title) || program === get('program') || (program === -1 && !open.size);
        details.innerHTML = `<summary>${title}<span class="count"></span></summary>`;
        for (const info of sim.info) {
            if (!pattern.test(info.name) || info.name === 'program') continue;
            details.append(settingRow(info));
        }
        root.append(details);
    }
    for (const info of sim.info) refresh(info.name);
    buildPrograms();
}

function settingRow(info) {
    const row = document.createElement('div');
    row.className = 'row';
    const label = document.createElement('label');
    label.textContent = info.name.replace(/^(mood|ex|gs|cy|nca)_/, '');
    label.title = `${info.name}: ${info.lo} to ${info.hi}, ${info.isInt ? 'whole numbers' : 'decimal'}`;
    label.ondblclick = () => set(info.name, info.boot);
    row.append(label);
    const inputs = [];
    if (/_map$|^map_override$/.test(info.name)) {
        const select = document.createElement('select');
        if (info.lo < 0) select.add(new Option("program's own", -1));
        MAPS.forEach((m, k) => select.add(new Option(m, k)));
        select.onchange = () => set(info.name, +select.value);
        inputs.push(select);
    } else if (info.isInt && info.lo === 0 && info.hi === 1) {
        const box = document.createElement('input');
        box.type = 'checkbox';
        box.onchange = () => set(info.name, box.checked ? 1 : 0);
        inputs.push(box, document.createElement('span'));
    } else {
        const range = document.createElement('input');
        range.type = 'range';
        range.min = info.lo, range.max = info.hi;
        range.step = info.isInt ? 1 : (info.hi - info.lo) / 1000;
        range.oninput = () => set(info.name, +range.value);
        const num = document.createElement('input');
        num.type = 'number';
        num.step = 'any';
        num.onchange = () => set(info.name, +num.value);
        inputs.push(range, num);
    }
    row.append(...inputs);
    ui[info.name] = { label, inputs };
    return row;
}

function refresh(name) {
    if (name === 'program') return buildPrograms();
    const el = ui[name];
    if (!el) return;
    const info = sim.info[sim.index[name]], v = get(name);
    for (const input of el.inputs) {
        if (input.type === 'checkbox') input.checked = !!v;
        else if (input.type === 'range') input.value = v;
        else if (input.type === 'number' && document.activeElement !== input) input.value = fmt(v, info);
        else if (input.tagName === 'SELECT') input.value = v;
    }
    el.label.classList.toggle('changed', touched.has(name));
    for (const details of document.querySelectorAll('#params details')) {
        const n = details.querySelectorAll('label.changed').length;
        details.querySelector('.count').textContent = n ? `${n} changed` : '';
    }
}

function buildPrograms() {
    const root = $('programs');
    root.innerHTML = '';
    PROGRAMS.forEach((title, k) => {
        const b = document.createElement('button');
        b.textContent = title.split(/[ -]/)[0];
        b.title = title;
        b.className = get('program') === k ? 'on' : '';
        b.onclick = () => {
            set('program', k, { send: false });
            send(`p ${k}`);
            for (const d of document.querySelectorAll('#params details'))
                if (+d.dataset.program >= 0) d.open = +d.dataset.program === k;
        };
        root.append(b);
    });
}

// ---- 3D view ----

const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(devicePixelRatio);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07080a);
const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
camera.position.set(0, 2, 11.5);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.autoRotateSpeed = 0.6;

let leds = null, glow = null, paper = null, positions = null, nodes = 0;
const ledColor = new THREE.Color();

function setupScene() {
    if (leds) return;
    nodes = sim.e.sim_nodes();
    positions = new Float32Array(sim.e.memory.buffer, sim.e.sim_positions(), nodes * 3).slice();
    let radius = 0;
    for (let i = 0; i < nodes; i++) radius = Math.max(radius, Math.hypot(...positions.subarray(3 * i, 3 * i + 3)));

    leds = new THREE.InstancedMesh(new THREE.SphereGeometry(0.06, 12, 8), new THREE.MeshBasicMaterial(), nodes);
    const m = new THREE.Matrix4();
    for (let i = 0; i < nodes; i++) {
        leds.setMatrixAt(i, m.makeTranslation(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]));
        leds.setColorAt(i, ledColor.setRGB(0, 0, 0));
    }
    scene.add(leds);

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(nodes * 3), 3));
    glow = new THREE.Points(g, new THREE.ShaderMaterial({
        uniforms: { strength: { value: 1 }, scale: { value: 1 } },
        vertexShader: `
            uniform float scale; attribute vec3 color; varying vec3 vColor;
            void main() {
                vColor = color;
                vec4 mv = modelViewMatrix * vec4(position, 1.0);
                gl_PointSize = scale * 1.6 / -mv.z;
                gl_Position = projectionMatrix * mv;
            }`,
        fragmentShader: `
            uniform float strength; varying vec3 vColor;
            void main() {
                float r2 = dot(gl_PointCoord - 0.5, gl_PointCoord - 0.5) * 4.0;
                gl_FragColor = vec4(vColor * strength * exp(-r2 * 5.0) * 0.6, 1.0);
                #include <colorspace_fragment>
            }`,
        blending: THREE.AdditiveBlending, depthWrite: false, transparent: true,
    }));
    scene.add(glow);

    // Tissue paper just outside the LEDs, lit from within: each LED adds light falling off with distance.
    paper = new THREE.Mesh(new THREE.SphereGeometry(radius * 1.08, 96, 64), new THREE.ShaderMaterial({
        uniforms: {
            ledPos: { value: Array.from({ length: nodes }, (_, i) => new THREE.Vector3().fromArray(positions, 3 * i)) },
            ledCol: { value: Array.from({ length: nodes }, () => new THREE.Vector3()) },
            strength: { value: 0.55 },
        },
        vertexShader: `
            varying vec3 vPos;
            void main() { vPos = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
        fragmentShader: `
            #define N ${nodes}
            uniform vec3 ledPos[N]; uniform vec3 ledCol[N]; uniform float strength; varying vec3 vPos;
            void main() {
                vec3 light = vec3(0.0);
                for (int i = 0; i < N; i++) {
                    vec3 d = vPos - ledPos[i];
                    light += ledCol[i] / (0.12 + dot(d, d) * 3.0);
                }
                light *= strength * 0.08;
                gl_FragColor = vec4(light / (1.0 + light) + vec3(0.002), 1.0);
                #include <colorspace_fragment>
            }`,
        side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true,
    }));
    scene.add(paper);
}

function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.floor(w * devicePixelRatio) || canvas.height !== Math.floor(h * devicePixelRatio)) {
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    }
}

const strip = $('strip').getContext('2d');
const stripImage = strip.createImageData(45, 2);

function draw(frame) {
    const colors = glow.geometry.attributes.color;
    const glowOn = +$('glow').value, paperOn = +$('paper').value;
    for (let i = 0; i < nodes; i++) {
        // LED brightness is linear in the PWM value.
        const r = frame[3 * i] / 255, g = frame[3 * i + 1] / 255, b = frame[3 * i + 2] / 255;
        leds.setColorAt(i, ledColor.setRGB(r + 0.006, g + 0.006, b + 0.006));
        colors.setXYZ(i, r, g, b);
        paper.material.uniforms.ledCol.value[i].set(r, g, b);
        // 45 pixels per row, so row 0 is output 0 and row 1 output 1.
        for (let k = 0; k < 3; k++) stripImage.data[4 * i + k] = 255 * Math.pow(frame[3 * i + k] / 255, 1 / 2.2);
        stripImage.data[4 * i + 3] = 255;
    }
    leds.instanceColor.needsUpdate = true;
    colors.needsUpdate = true;
    glow.material.uniforms.strength.value = glowOn;
    glow.material.uniforms.scale.value = canvas.height;
    glow.visible = glowOn > 0;
    paper.material.uniforms.strength.value = paperOn;
    paper.visible = paperOn > 0;
    strip.putImageData(stripImage, 0, 0);
}

// ---- main loop ----

let paused = false, last = performance.now(), passes = 0, lastReadout = 0;

function linkSlider(id) {
    const range = $(id), num = $(id + '-n');
    range.oninput = () => (num.value = range.value);
    num.onchange = () => (range.value = num.value);
}
['speed', 'loop', 'lag'].forEach(linkSlider);

function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(now - last, 1000);
    last = now;
    if (!sim || !leds) return;
    resize();
    if (!paused) passes += sim.e.sim_advance(dt * +$('speed').value, +$('loop').value, +$('lag').value);
    else sim.e.sim_advance(0, +$('loop').value, +$('lag').value);
    draw(new Uint8Array(sim.e.memory.buffer, sim.e.sim_frame(), nodes * 3));
    controls.autoRotate = $('spin').checked;
    controls.update();
    renderer.render(scene, camera);

    if (now - lastReadout > 250) {
        const ma = sim.e.sim_milliamps(), cap = get('max_milliamps');
        $('ma').textContent = ma.toFixed(0);
        $('ma-cap').textContent = cap;
        $('bright').textContent = sim.e.sim_brightness();
        $('ma-bar').style.width = `${Math.min(100, 100 * ma / cap)}%`;
        $('ma-bar').style.background = sim.e.sim_brightness() < get('brightness') ? 'var(--accent)' : 'var(--good)';
        $('status').textContent = cstr(sim.e.sim_status());
        const t = sim.e.sim_clock() / 1000;
        $('readout').textContent = `${PROGRAMS[get('program')]}\nsimulated ${Math.floor(t / 60)}:${(t % 60).toFixed(0).padStart(2, '0')}` +
            `   ${(passes * 1000 / (now - lastReadout)).toFixed(0)} loop passes/s`;
        passes = 0, lastReadout = now;
    }
}
requestAnimationFrame(frame);

$('pause').onclick = () => {
    paused = !paused;
    $('pause').textContent = paused ? 'Play' : 'Pause';
    $('pause').classList.toggle('on', paused);
};
$('restart').onclick = () => sim.e.sim_reset((Math.random() * 2 ** 32) >>> 0);

// ---- reloading when the firmware sources change ----

async function poll() {
    try {
        const s = await (await fetch('stamp')).text();
        if (s !== stamp) {
            stamp = s;
            await load();
            if (touched.size) log(`reloaded the firmware code, kept ${touched.size} unsaved settings`);
        }
    } catch (err) {
        $('error').textContent = `Couldn't build the firmware for the simulator:\n\n${err.message}`;
        $('error').style.display = 'block';
    }
    setTimeout(poll, 1000);
}
poll();

// ---- saving and flashing ----

function log(line) {
    const el = $('log');
    el.textContent += line + '\n';
    el.scrollTop = el.scrollHeight;
}

async function save() {
    const values = {};
    for (const name of touched) values[name] = get(name);
    if (!Object.keys(values).length) return log('nothing to save: every setting matches params.h'), true;
    const res = await fetch('params', { method: 'POST', body: JSON.stringify(values) });
    const { missing } = await res.json();
    log(`saved ${Object.keys(values).length - missing.length} settings to params.h` +
        (missing.length ? `; not found: ${missing.join(', ')}` : ''));
    return !missing.length;
}

$('save').onclick = save;
$('flash').onclick = async () => {
    const button = $('flash');
    button.disabled = true;
    try {
        if (!(await save())) return;
        if (port) await disconnect();
        log('building and flashing with PlatformIO…');
        const res = await fetch('flash', { method: 'POST' });
        const text = await res.text();
        log(res.ok ? 'flashed' : 'flash failed:\n' + text.split('\n').slice(-15).join('\n'));
    } finally {
        button.disabled = false;
    }
};

// ---- the real lantern over Web Serial ----

let port = null, writer = null, reader = null, writeDone = null, readDone = null;
const pending = new Map();
let sendTimer = null;

function send(line) {
    if (writer) writer.write(line + '\n').catch(err => log(`serial: ${err.message}`));
}

function queueSend(name, value) {
    if (!writer || !$('live').checked) return;
    pending.set(name, value);
    sendTimer ??= setTimeout(() => {
        for (const [n, v] of pending) send(`k ${n} ${v}`);
        pending.clear();
        sendTimer = null;
    }, 40);
}

function setConnected(on) {
    $('connect').textContent = on ? 'Disconnect' : 'Connect';
    $('connect').classList.toggle('on', on);
    $('measure').disabled = $('send-all').disabled = !on;
}

async function connect() {
    if (!('serial' in navigator)) return log('Web Serial needs Chrome or Edge');
    port = await navigator.serial.requestPort();
    await port.open({ baudRate: 115200 });
    const encoder = new TextEncoderStream();
    writeDone = encoder.readable.pipeTo(port.writable).catch(() => {});
    writer = encoder.writable.getWriter();
    const decoder = new TextDecoderStream();
    readDone = port.readable.pipeTo(decoder.writable).catch(() => {});
    reader = decoder.readable.getReader();
    setConnected(true);
    log('connected');
    readLines();
}

async function readLines() {
    let buffer = '';
    try {
        for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += value;
            const lines = buffer.split('\n');
            buffer = lines.pop();
            for (const raw of lines) {
                const line = raw.trim();
                if (!line || line === 'ok') continue;
                const loop = line.match(/^loop ([\d.]+) ms/);
                if (loop) {
                    $('loop').value = $('loop-n').value = (+loop[1]).toFixed(1);
                    log(`lantern main loop: ${loop[1]} ms; set the simulator to match`);
                } else {
                    log(`lantern: ${line}`);
                }
            }
        }
    } catch (err) {
        log(`serial: ${err.message}`);
        disconnect();
    }
}

async function disconnect() {
    const p = port;
    port = null;
    try {
        await reader?.cancel();
        await readDone;
        await writer?.close();
        await writeDone;
        await p?.close();
    } catch { /* already gone */ }
    writer = reader = null;
    setConnected(false);
    log('disconnected');
}

$('connect').onclick = () => (port ? disconnect() : connect().catch(err => log(`serial: ${err.message}`)));
$('measure').onclick = () => send('i');
$('send-all').onclick = () => {
    for (const info of sim.info) if (info.name !== 'program') send(`k ${info.name} ${get(info.name)}`);
    send(`p ${get('program')}`);
    log(`sent ${sim.info.length} settings`);
};
