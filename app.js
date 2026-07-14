/* OtoScope PoC — 録音から埋め込み・可視化・再生分析まで全てクライアントサイドで完結する。
   モデル(YAMNet)もこのアプリ自身から配信し、外部への通信は一切行わない。
   毎測定を YAMNet(1024次元) と DSP特徴(80次元) の両方で埋め込み、
   音声(16kHz Int16)・埋め込み・帯域マーク・メモを IndexedDB に保持する。 */
'use strict';
const $ = id => document.getElementById(id);
const COLORS = ['#4fc3f7', '#ffb74d', '#81c784', '#e57373', '#ba68c8', '#fff176', '#4db6ac', '#f06292'];
const APP_VERSION = '0.0.2';
const SR = 16000, N_FFT = 1024, HOP = 512, N_MEL = 40;
const MELCNN_T = 96, MELCNN_DIM = 512;      // MelCNN(実験): 固定長メル画像 → 512次元
const AC = window.AudioContext || window.webkitAudioContext;

/* ---------- 状態: シリーズはlocalStorage、測定と音声はIndexedDB ---------- */
let CFG = JSON.parse(localStorage.getItem('otoscope-cfg') || 'null') ||
  { series: [{ id: 1, name: 'デフォルト' }], nextSid: 2 };
if (!CFG.device) CFG.device = { name: '', mic: '' };   // 旧データ互換
const saveCfg = () => localStorage.setItem('otoscope-cfg', JSON.stringify(CFG));

function guessDeviceName() {                 // ブラウザからクライアント個体をざっくり推定
  const ud = navigator.userAgentData;
  if (ud && ud.platform) {
    const b = (ud.brands || []).map(x => x.brand).find(n => !/Not.?A.?Brand|Chromium/i.test(n));
    return ud.platform + (b ? ` (${b})` : '');
  }
  const ua = navigator.userAgent;
  const os = /iPhone/.test(ua) ? 'iPhone' : /iPad/.test(ua) ? 'iPad' : /Android/.test(ua) ? 'Android'
    : /Windows/.test(ua) ? 'Windows PC' : /Mac OS X/.test(ua) ? 'Mac' : /Linux/.test(ua) ? 'Linux' : 'デバイス';
  const br = /Edg\//.test(ua) ? 'Edge' : /Chrome\//.test(ua) ? 'Chrome' : /Firefox\//.test(ua) ? 'Firefox'
    : /Safari\//.test(ua) ? 'Safari' : '';
  return os + (br ? ` (${br})` : '');
}
const seriesById = sid => CFG.series.find(s => s.id === sid) || {};
function seriesCondText(sid) {
  const s = seriesById(sid), p = [];
  if (s.location) p.push('📍' + s.location);
  if (s.target) p.push('🎯' + s.target);
  if (s.position) p.push('📐' + s.position);
  if (s.note) p.push('📝' + s.note);
  return p.join(' ／ ');
}
const seriesName = sid => (CFG.series.find(s => s.id === sid) || { name: '?' }).name;
const seriesColor = sid => COLORS[CFG.series.findIndex(s => s.id === sid) % COLORS.length];

let MEAS = [];                              // 全測定（音声以外）をメモリに常駐
let db;
const idbOpen = () => new Promise((res, rej) => {
  const q = indexedDB.open('otoscope', 2);
  q.onupgradeneeded = () => {
    const d = q.result;
    if (!d.objectStoreNames.contains('meas')) d.createObjectStore('meas', { keyPath: 't' });
    if (!d.objectStoreNames.contains('audio')) d.createObjectStore('audio', { keyPath: 't' });
    if (!d.objectStoreNames.contains('models')) d.createObjectStore('models', { keyPath: 'id' });
  };
  q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error);
});
const tx = (store, mode, fn) => new Promise((res, rej) => {
  const t = db.transaction(store, mode), r = fn(t.objectStore(store));
  t.oncomplete = () => res(r && r.result); t.onerror = () => rej(t.error);
});
const dbPut = (s, v) => tx(s, 'readwrite', o => o.put(v));
const dbDel = (s, k) => tx(s, 'readwrite', o => o.delete(k));
const dbClear = s => tx(s, 'readwrite', o => o.clear());
const dbGet = (s, k) => tx(s, 'readonly', o => o.get(k));
const dbAll = s => tx(s, 'readonly', o => o.getAll());

/* ---------- DSP: FFT / メル尺度 / メルスペクトログラム ---------- */
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j |= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const j = i + k + len / 2;
        const tr = re[j] * cr - im[j] * ci, ti = re[j] * ci + im[j] * cr;
        re[j] = re[i + k] - tr; im[j] = im[i + k] - ti;
        re[i + k] += tr; im[i + k] += ti;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}

const mel = f => 2595 * Math.log10(1 + f / 700), imel = m => 700 * (10 ** (m / 2595) - 1);
const MEL0 = mel(50), MEL1 = mel(SR / 2);
const HANN = Float32Array.from({ length: N_FFT }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N_FFT));
const BANK = Array.from({ length: N_MEL }, (_, i) => {
  const p = j => imel(MEL0 + (MEL1 - MEL0) * j / (N_MEL + 1)) / SR * N_FFT;
  const f = new Float32Array(N_FFT / 2 + 1), [a, b, c] = [p(i), p(i + 1), p(i + 2)];
  for (let k = Math.ceil(a); k <= Math.min(c, N_FFT / 2); k++)
    f[k] = k < b ? (k - a) / (b - a) : (c - k) / (c - b);
  return f;
});

function melSpec(x) {                       // x: Float32Array @16kHz → フレーム毎の40帯域 log-mel
  const frames = [];
  for (let s = 0; s + N_FFT <= x.length; s += HOP) {
    const re = new Float32Array(N_FFT), im = new Float32Array(N_FFT);
    for (let i = 0; i < N_FFT; i++) re[i] = x[s + i] * HANN[i];
    fft(re, im);
    const m = new Float32Array(N_MEL);
    for (let j = 0; j < N_MEL; j++) {
      let a = 0; const f = BANK[j];
      for (let k = 0; k <= N_FFT / 2; k++) a += f[k] * (re[k] * re[k] + im[k] * im[k]);
      m[j] = Math.log10(a + 1e-8);
    }
    frames.push(m);
  }
  return frames;
}

/* 複数の周波数帯をSTFTマスクで消す(keep=false)／その帯域だけ残す(keep=true) */
function bandsFilter(x, bands, keep) {
  const H = N_FFT / 4, y = new Float32Array(x.length), norm = new Float32Array(x.length);
  const inAny = new Uint8Array(N_FFT / 2 + 1);
  for (const [lo, hi] of bands) {
    const kLo = Math.round(lo / (SR / 2) * (N_FFT / 2)), kHi = Math.round(hi / (SR / 2) * (N_FFT / 2));
    for (let k = Math.max(0, kLo); k <= Math.min(N_FFT / 2, kHi); k++) inAny[k] = 1;
  }
  for (let s = 0; s + N_FFT <= x.length; s += H) {
    const re = new Float32Array(N_FFT), im = new Float32Array(N_FFT);
    for (let i = 0; i < N_FFT; i++) re[i] = x[s + i] * HANN[i];
    fft(re, im);
    for (let k = 0; k <= N_FFT / 2; k++) {
      if (!!inAny[k] !== keep) {
        re[k] = im[k] = 0;
        if (k > 0 && k < N_FFT / 2) { re[N_FFT - k] = im[N_FFT - k] = 0; }
      }
    }
    for (let i = 0; i < N_FFT; i++) im[i] = -im[i];   // 共役で逆FFT
    fft(re, im);
    for (let i = 0; i < N_FFT; i++) {
      y[s + i] += re[i] / N_FFT * HANN[i];
      norm[s + i] += HANN[i] * HANN[i];
    }
  }
  for (let i = 0; i < x.length; i++) y[i] = norm[i] > 0.15 ? y[i] / norm[i] : 0;  // 端は窓の重なりが薄く発散するため捨てる
  return y;
}

/* ---------- 埋め込み（毎測定で全モデルを計算。ここに追加すれば拡張できる） ---------- */
const l2 = v => { const n = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1; return v.map(x => +(x / n).toFixed(5)); };

function dspEmbed(frames) {                 // 各メル帯域の平均+標準偏差 = 80次元（物理特徴）
  const n = frames.length, v = [];
  for (let j = 0; j < N_MEL; j++) {
    let mu = 0, sd = 0;
    for (const f of frames) mu += f[j] / n;
    for (const f of frames) sd += (f[j] - mu) ** 2 / n;
    v.push(mu, Math.sqrt(sd));
  }
  return l2(v);
}

let yamnet = null, classMap = null, melcnn = null;
function modelStatusText() {
  const parts = [];
  if (yamnet) parts.push('YAMNet');
  parts.push('DSP');
  if (melcnn) parts.push('MelCNN(実験)');
  return parts.join(' + ') + ' で記録中（完全ローカル）';
}
async function initYamnet() {               // モデルは自アプリ配信（models/）— 外部通信なし
  try {
    if (typeof tf === 'undefined') throw new Error('tfjs未読込');
    yamnet = await tf.loadGraphModel('models/yamnet/model.json');
    fetch('models/yamnet/yamnet_class_map.csv')
      .then(r => r.text())
      .then(t => { classMap = t.trim().split(/\r?\n/).slice(1).map(l => l.split(',').slice(2).join(',').replace(/"/g, '')); })
      .catch(() => { });
  } catch (e) {
    yamnet = null;
  }
  $('modelStatus').textContent = modelStatusText();
}

/* MelCNN(実験): メルスペクトログラム画像を固定重みの軽量CNNで512次元へ。
   学習はしていない（決定論的ランダム畳み込み特徴）。シード固定なので端末/セッション間で再現し、
   DSP・YAMNetとは別の非線形な指紋になる。将来は学習済み重みに差し替え可能な構造。 */
function initMelCnn() {
  try {
    if (typeof tf === 'undefined') throw new Error('tfjs未読込');
    const inp = tf.input({ shape: [MELCNN_T, N_MEL, 1] });
    let x = inp;
    for (const ch of [16, 32, 64, 256]) {
      x = tf.layers.conv2d({
        filters: ch, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu',
        useBias: false, kernelInitializer: tf.initializers.glorotNormal({ seed: 1234 })
      }).apply(x);
    }
    const avg = tf.layers.globalAveragePooling2d({}).apply(x);   // 256
    const mx = tf.layers.globalMaxPooling2d({}).apply(x);        // 256
    const cat = tf.layers.concatenate().apply([avg, mx]);        // 512
    melcnn = tf.model({ inputs: inp, outputs: cat });
  } catch (e) {
    melcnn = null;
  }
  $('modelStatus').textContent = modelStatusText();
}

function melcnnEmbed(frames) {               // frames: [T][N_MEL] log-mel → 512次元(L2)
  if (!melcnn || !frames.length) return null;
  return tf.tidy(() => {
    const T = frames.length;
    const buf = new Float32Array(T * N_MEL);
    for (let t = 0; t < T; t++) { const f = frames[t]; for (let j = 0; j < N_MEL; j++) buf[t * N_MEL + j] = f[j]; }
    let x = tf.tensor(buf, [1, T, N_MEL, 1]);
    const mo = tf.moments(x);                                     // 画像ごとに標準化
    x = x.sub(mo.mean).div(mo.variance.sqrt().add(1e-6));
    x = tf.image.resizeBilinear(x, [MELCNN_T, N_MEL]);            // 時間方向を固定長に
    const out = melcnn.predict(x);
    return l2(Array.from(out.dataSync()));
  });
}

async function yamnetEmbed(x) {
  const out = yamnet.predict(tf.tensor1d(x));
  const ts = Array.isArray(out) ? out : [out];
  let scores, emb;                          // 出力テンソルは次元数で判別（521=クラス, 1024=埋め込み）
  for (const t of ts) { const d = t.shape[t.shape.length - 1]; if (d === 521) scores = t; if (d === 1024) emb = t; }
  const e = Array.from(await emb.mean(0).data());
  const sc = await scores.mean(0).data();
  ts.forEach(t => t.dispose());
  const top = [...sc.keys()].sort((a, b) => sc[b] - sc[a]).slice(0, 3)
    .map(i => ({ name: classMap ? classMap[i] : 'class#' + i, p: sc[i] }));
  return { emb: l2(e), top };
}

/* ---------- 録音（生サンプル取得 → 16kHzへ変換。適用された実条件も記録） ---------- */
let lastCond = null;
function record(sec, raw) {
  const cons = raw
    ? { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
    : true;
  return navigator.mediaDevices.getUserMedia({ audio: cons }).then(stream => new Promise(res => {
    const st = stream.getAudioTracks()[0].getSettings();
    lastCond = { ec: !!st.echoCancellation, ns: !!st.noiseSuppression, agc: !!st.autoGainControl };
    const ctx = new AC();
    const src = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    const chunks = []; let n = 0; const need = Math.round(sec * ctx.sampleRate);
    proc.onaudioprocess = e => {
      const d = e.inputBuffer.getChannelData(0);
      chunks.push(new Float32Array(d)); n += d.length;
      let peak = 0; for (const v of d) peak = Math.max(peak, Math.abs(v));
      $('levelBar').style.width = Math.min(100, peak * 130) + '%';
      $('countdown').textContent = 'あと ' + Math.max(0, (need - n) / ctx.sampleRate).toFixed(1) + ' 秒';
      if (n >= need) {
        proc.disconnect(); src.disconnect();
        stream.getTracks().forEach(t => t.stop()); ctx.close();
        const all = new Float32Array(n); let o = 0;
        for (const c of chunks) { all.set(c, o); o += c.length; }
        res({ data: all.subarray(0, need), sr: ctx.sampleRate });
      }
    };
    src.connect(proc); proc.connect(ctx.destination);   // procの出力は無音なのでハウリングしない
  }));
}

function resample(x, sr, target) {
  if (sr === target) return x;
  const n = Math.floor(x.length * target / sr), y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = i * sr / target, k = Math.floor(p), f = p - k;
    y[i] = x[k] + (x[Math.min(k + 1, x.length - 1)] - x[k]) * f;
  }
  return y;
}

/* ---------- 類似度・品質チェック・横断レコメンド ---------- */
const cos = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };  // 埋め込みはL2正規化済み
const judge = s => s >= 0.90 ? '🟢 いつも通り' : s >= 0.75 ? '🟡 やや違う' : '🔴 かなり違う';
const primarySim = m => m.sim.yamnet != null ? m.sim.yamnet : m.sim.dsp;

function simTo(prev, model, emb) {          // 同シリーズ過去平均とのコサイン類似度
  const past = prev.filter(m => m.emb[model]);
  if (!past.length) return null;
  const c = new Array(emb.length).fill(0);
  past.forEach(m => m.emb[model].forEach((x, i) => c[i] += x / past.length));
  const n = Math.sqrt(c.reduce((a, x) => a + x * x, 0)) || 1;
  return cos(emb, c.map(x => x / n));
}

function qualityHints(rec, prev) {
  const h = [];
  if (rec.peak > 0.98) h.push('音が割れています（クリッピング）。少し離れて測り直しを');
  if (rec.rmsDb < -55) h.push('ほぼ無音です。マイクを指で塞いでいないか・遠すぎないか・権限を確認');
  if (prev.length >= 3) {
    const avg = prev.reduce((a, m) => a + m.rmsDb, 0) / prev.length;
    if (rec.rmsDb < avg - 12) h.push(`いつもより約${Math.round(avg - rec.rmsDb)}dB静かです。距離・向き・マイク塞ぎを確認（音が取れていないかも）`);
    if (rec.rmsDb > avg + 12) h.push('いつもよりかなり大きい音です。距離が近すぎるかも');
  }
  const pc = prev.length && prev[prev.length - 1].cond;
  if (pc && rec.cond && (pc.agc !== rec.cond.agc || pc.ns !== rec.cond.ns || pc.ec !== rec.cond.ec))
    h.push('録音条件（AGC等）が同シリーズの前回と異なります。判定の基準が変わるため条件を揃えて');
  return h;
}

/* ラベル: rec.label={status:'normal'|'abnormal', text}。旧データの rec.note は異常メモとして扱う */
function labelOf(m) {
  if (m.label && m.label.status) return m.label.status;
  if (m.note && m.note.text) return 'abnormal';
  return null;
}
function labelText(m) {
  if (m.label) return m.label.text || '';
  if (m.note) return m.note.text || '';
  return '';
}
const embModel = rec => rec.emb.yamnet ? 'yamnet' : (rec.emb.melcnn ? 'melcnn' : 'dsp');

function similarNotes(rec) {                // 異常ラベル付きレコードとの横断照合（全シリーズ）
  const model = embModel(rec), e = rec.emb[model];
  return MEAS.filter(m => m.t !== rec.t && labelOf(m) === 'abnormal' && labelText(m) && m.emb[model])
    .map(m => ({ m, s: cos(e, m.emb[model]) }))
    .filter(o => o.s > 0.85).sort((a, b) => b.s - a.s).slice(0, 3);
}

/* ---------- 描画共通: スペクトログラム＋軸＋帯域オーバレイ ---------- */
const specColor = u => `hsl(${240 - 240 * u},85%,${8 + 52 * u}%)`;
const yOfHz = (hz, H) => H * (1 - (mel(hz) - MEL0) / (MEL1 - MEL0));
const hzOfY = (y, H) => imel(MEL0 + (MEL1 - MEL0) * (1 - y / H));
const HZ_TICKS = [100, 200, 500, 1000, 2000, 4000, 8000];
const hzLabel = hz => hz >= 1000 ? (hz / 1000) + 'k' : '' + hz;

function renderSpec(c, W, H, frames, i0, i1, opts = {}) {
  let lo = 1e9, hi = -1e9;
  frames.forEach(f => f.forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v); }));
  c.fillStyle = '#0d1620'; c.fillRect(0, 0, W, H);
  const n = i1 - i0, w = W / n, h = H / N_MEL;
  for (let t = 0; t < n; t++) for (let j = 0; j < N_MEL; j++) {
    c.fillStyle = specColor((frames[i0 + t][j] - lo) / (hi - lo || 1));
    c.fillRect(t * w, H - (j + 1) * h, w + 1, h + 1);
  }
  if (opts.axes) {
    c.font = '10px sans-serif'; c.textBaseline = 'middle';
    for (const hz of HZ_TICKS) {
      const y = yOfHz(hz, H);
      c.strokeStyle = 'rgba(255,255,255,.25)'; c.beginPath(); c.moveTo(0, y); c.lineTo(6, y); c.stroke();
      c.fillStyle = '#cfe0ee'; c.shadowColor = '#000'; c.shadowBlur = 3;
      c.fillText(hzLabel(hz) + 'Hz', 8, y); c.shadowBlur = 0;
    }
    const t0 = i0 * HOP / SR, t1 = i1 * HOP / SR, span = t1 - t0;
    const step = span > 6 ? 2 : span > 2.5 ? 1 : span > 1 ? 0.5 : 0.1;
    c.textBaseline = 'alphabetic';
    for (let s = Math.ceil(t0 / step) * step; s <= t1; s += step) {
      const x = (s - t0) / span * W;
      c.strokeStyle = 'rgba(255,255,255,.25)'; c.beginPath(); c.moveTo(x, H); c.lineTo(x, H - 6); c.stroke();
      c.fillStyle = '#cfe0ee'; c.shadowColor = '#000'; c.shadowBlur = 3;
      c.fillText(s.toFixed(step < 1 ? 1 : 0) + 's', x + 3, H - 8); c.shadowBlur = 0;
    }
  }
  for (const b of opts.bands || []) {
    const y1 = yOfHz(b.hi, H), y2 = yOfHz(b.lo, H);
    c.fillStyle = b.on ? 'rgba(255,60,60,.22)' : 'rgba(160,160,160,.13)';
    c.fillRect(0, y1, W, y2 - y1);
    c.strokeStyle = b.on ? '#ff5252' : '#889';
    c.setLineDash(b.temp ? [5, 4] : []);
    c.strokeRect(0.5, y1, W - 1, y2 - y1);
    c.setLineDash([]);
  }
}

function drawWave(x) {
  const cv = $('waveCv'), c = cv.getContext('2d');
  cv.width = cv.clientWidth;
  c.fillStyle = '#0d1620'; c.fillRect(0, 0, cv.width, cv.height);
  c.strokeStyle = '#4fc3f7'; c.beginPath();
  for (let px = 0; px < cv.width; px++) {
    const i = Math.floor(px * x.length / cv.width);
    c[px ? 'lineTo' : 'moveTo'](px, cv.height / 2 * (1 - x[i]));
  }
  c.stroke();
}

/* ---------- 可視化共通: 選択連動・カメラ（ズーム/パン）・ジェスチャ ---------- */
let SEL_T = null;                           // カラーマップ ↔ 埋込マップ連動選択
let MAPPTS = [];
let MAPCAM = { z: 1, ox: 0, oy: 0 };        // 埋め込み図の表示変換
let STACKCAM = { rowH: 12, f0: 0, f1: 1, yOff: 0 }; // 行高・周波数ズーム・縦パン

function setSelection(t, { redraw = true } = {}) {
  SEL_T = t;
  const btn = $('selClear');
  if (btn) btn.hidden = !t;
  if (!redraw) return;
  drawStack(); drawMap();
  const m = t && MEAS.find(x => x.t === t);
  if (m) {
    $('stackInfo').textContent =
      `選択中: ${new Date(m.t).toLocaleString('ja-JP')} ／ ${seriesName(m.sid)}`
      + (m.note ? ` ／ 📝 ${m.note.text}` : '')
      + '（再タップで解除／ダブルクリックで詳細）';
  }
}

function attachCanvasNav(cv, {
  onPan, onZoom, onTap, onDbl, canInteract = () => true, cursor = 'grab'
}) {
  const ptrs = new Map();
  let pinch0 = null, pan0 = null, moved = false, tapT = 0, tapX = 0, tapY = 0;
  const pos = e => {
    const r = cv.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top, r };
  };
  cv.style.touchAction = 'none';
  cv.addEventListener('pointerdown', e => {
    if (!canInteract()) return;
    cv.setPointerCapture(e.pointerId);
    const p = pos(e);
    ptrs.set(e.pointerId, p);
    moved = false;
    if (ptrs.size === 2) {
      const [a, b] = [...ptrs.values()];
      pinch0 = { dist: Math.hypot(a.x - b.x, a.y - b.y), cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2 };
      pan0 = null;
    } else if (ptrs.size === 1) {
      pan0 = { x: p.x, y: p.y };
      pinch0 = null;
      cv.style.cursor = 'grabbing';
    }
  });
  cv.addEventListener('pointermove', e => {
    if (!ptrs.has(e.pointerId) || !canInteract()) return;
    const p = pos(e);
    const prev = ptrs.get(e.pointerId);
    if (Math.hypot(p.x - prev.x, p.y - prev.y) > 4) moved = true;
    ptrs.set(e.pointerId, p);
    if (ptrs.size === 2 && pinch0) {
      const [a, b] = [...ptrs.values()];
      const dist = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch0.dist > 1) onZoom(dist / pinch0.dist, (a.x + b.x) / 2, (a.y + b.y) / 2, p.r);
      pinch0 = { dist, cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2 };
    } else if (ptrs.size === 1 && pan0) {
      onPan(p.x - pan0.x, p.y - pan0.y, p.r);
      pan0 = { x: p.x, y: p.y };
    }
  });
  const end = e => {
    if (!ptrs.has(e.pointerId)) return;
    const p = pos(e);
    ptrs.delete(e.pointerId);
    if (ptrs.size < 2) pinch0 = null;
    if (ptrs.size === 1) pan0 = [...ptrs.values()][0];
    if (ptrs.size === 0) {
      cv.style.cursor = cursor;
      if (!moved && canInteract()) {
        const now = Date.now();
        if (now - tapT < 320 && Math.hypot(p.x - tapX, p.y - tapY) < 18) {
          onDbl && onDbl(p.x, p.y, p.r);
          tapT = 0;
        } else {
          onTap && onTap(p.x, p.y, p.r);
          tapT = now; tapX = p.x; tapY = p.y;
        }
      }
      pan0 = null;
    }
  };
  cv.addEventListener('pointerup', end);
  cv.addEventListener('pointercancel', end);
  cv.addEventListener('wheel', e => {
    if (!canInteract()) return;
    e.preventDefault();
    const p = pos(e);
    onZoom(e.deltaY < 0 ? 1.2 : 1 / 1.2, p.x, p.y, p.r);
  }, { passive: false });
}

function hitMap(px, py) {
  let best = null, bd = 16;
  for (const p of MAPPTS) {
    const d = Math.hypot(p.x - px, p.y - py);
    if (d < bd) { bd = d; best = p; }
  }
  return best;
}

function drawMarker(c, px, py, m, { selected, latest }) {
  const col = seriesColor(m.sid);
  const lab = labelOf(m);
  const r = selected ? 8 : 5.5;
  c.lineWidth = selected ? 2.5 : (latest ? 2 : 1.2);
  c.strokeStyle = selected ? '#ffd54f' : (latest ? '#fff' : 'rgba(0,0,0,.5)');
  if (lab === 'abnormal') {
    // 異常: 赤縁の菱形＋!
    c.fillStyle = col;
    c.beginPath();
    c.moveTo(px, py - r - 1); c.lineTo(px + r + 1, py);
    c.lineTo(px, py + r + 1); c.lineTo(px - r - 1, py); c.closePath();
    c.fill();
    if (!selected) c.strokeStyle = '#ff5252';
    c.stroke();
    c.fillStyle = '#fff'; c.font = 'bold 9px sans-serif';
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillText('!', px, py + 0.5);
    c.textAlign = 'start'; c.textBaseline = 'alphabetic';
  } else if (lab === 'normal') {
    // 正常: 塗りつぶし円
    c.fillStyle = col;
    c.beginPath(); c.arc(px, py, r, 0, Math.PI * 2); c.fill(); c.stroke();
  } else {
    // 未ラベル: 中空円
    c.fillStyle = 'rgba(180,200,220,.15)';
    c.beginPath(); c.arc(px, py, r, 0, Math.PI * 2); c.fill();
    if (!selected) c.strokeStyle = col;
    c.setLineDash([2, 2]); c.stroke(); c.setLineDash([]);
  }
  if (latest && !selected) {
    c.strokeStyle = '#fff'; c.lineWidth = 1.5;
    c.beginPath(); c.arc(px, py, r + 4, 0, Math.PI * 2); c.stroke();
  }
}

/* ---------- 埋め込みマップ（PCA・ズーム/パン・選択連動） ---------- */
const mulberry32 = a => () => {              // 決定論的乱数（PCA初期化を固定して向きを安定化）
  a |= 0; a = a + 0x6D2B79F5 | 0;
  let t = Math.imul(a ^ a >>> 15, 1 | a);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
  return ((t ^ t >>> 14) >>> 0) / 4294967296;
};
function pca2(vs) {                          // 上位2主成分へ射影（べき乗法・決定論的）
  const n = vs.length, d = vs[0].length, mu = new Float64Array(d);
  vs.forEach(v => { for (let i = 0; i < d; i++) mu[i] += v[i] / n; });
  const X = vs.map(v => { const r = new Float64Array(d); for (let i = 0; i < d; i++) r[i] = v[i] - mu[i]; return r; });
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const comps = [];
  for (let c = 0; c < 2; c++) {
    const rnd = mulberry32(1234 + c * 7919);
    let v = Float64Array.from({ length: d }, () => rnd() - 0.5);
    for (let it = 0; it < 40; it++) {
      const u = X.map(x => dot(x, v)), w = new Float64Array(d);
      X.forEach((x, i) => { for (let j = 0; j < d; j++) w[j] += u[i] * x[j]; });
      for (const p of comps) { const pj = dot(w, p); for (let j = 0; j < d; j++) w[j] -= pj * p[j]; }
      const nn = Math.sqrt(dot(w, w));
      if (nn < 1e-12) break;
      for (let j = 0; j < d; j++) w[j] /= nn;
      v = w;
    }
    comps.push(v);
  }
  return X.map(x => [dot(x, comps[0]), dot(x, comps[1])]);
}

let MAPCACHE = { key: '', pts: null };       // PCA結果をキャッシュ（選択変更では再計算せず位置維持）
const mapKey = (space, ms) => `${space}|${ms.length}|${ms.length ? ms[ms.length - 1].t : 0}`;
function drawMap() {
  const cv = $('mapCv'), c = cv.getContext('2d');
  cv.width = cv.clientWidth;
  cv.height = Math.max(280, cv.clientHeight || 360);
  c.fillStyle = '#0d1620'; c.fillRect(0, 0, cv.width, cv.height);
  const space = document.querySelector('[name=space]:checked').value;
  const ms = MEAS.filter(m => m.emb[space]);
  MAPPTS = [];
  if (ms.length < 2) {
    $('legend').innerHTML = '';
    c.fillStyle = '#8aa'; c.font = '13px sans-serif';
    c.fillText(`測定が2件たまると表示されます（${space}: 現在${ms.length}件）`, 12, 30);
    return;
  }
  const key = mapKey(space, ms);
  let pts;
  if (MAPCACHE.key === key && MAPCACHE.pts) pts = MAPCACHE.pts;
  else { pts = pca2(ms.map(m => m.emb[space])); MAPCACHE = { key, pts }; }
  let [x0, x1, y0, y1] = [1e9, -1e9, 1e9, -1e9];
  pts.forEach(([x, y]) => { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); });
  const pad = 28, W = cv.width, H = cv.height;
  const fitX = x => pad + (x - x0) / (x1 - x0 || 1) * (W - 2 * pad);
  const fitY = y => pad + (y - y0) / (y1 - y0 || 1) * (H - 2 * pad);
  const cx = W / 2, cy = H / 2, z = MAPCAM.z;
  const sx = x => (fitX(x) - cx) * z + cx + MAPCAM.ox;
  const sy = y => (fitY(y) - cy) * z + cy + MAPCAM.oy;
  const latest = ms.reduce((a, b) => a.t > b.t ? a : b);
  // メモありを後ろに描画して前面に出す
  const order = ms.map((m, i) => i).sort((a, b) => (!!ms[a].note) - (!!ms[b].note));
  order.forEach(i => {
    const m = ms[i], px = sx(pts[i][0]), py = sy(pts[i][1]);
    MAPPTS.push({ x: px, y: py, t: m.t });
    drawMarker(c, px, py, m, { selected: SEL_T === m.t, latest: m === latest });
  });
  const used = [...new Set(ms.map(m => m.sid))];
  $('legend').innerHTML = used.map(sid =>
    `<span class="chip"><i style="background:${seriesColor(sid)}"></i>${seriesName(sid)}</span>`
  ).join('')
    + '<span class="chip">●正常</span><span class="chip">◆異常</span><span class="chip">○未</span><span class="chip">◎最新</span>'
    + (SEL_T ? '<span class="chip sel-chip">選択中あり</span>' : '');
}

/* ---------- カラーマップ（音紋縦積み・選択連動・ズーム） ---------- */
const melAvgOf = m => m.melAvg || m.emb.dsp.filter((_, i) => i % 2 === 0);   // 旧データはDSP埋め込みから代用
function drawStack() {
  const cv = $('stackCv'), c = cv.getContext('2d');
  const wrap = $('stackWrap') || cv.parentElement;
  const W = wrap.clientWidth, axisH = 18, L = 8;
  const rowH = Math.max(6, Math.min(28, STACKCAM.rowH));
  STACKCAM.rowH = rowH;
  const f0 = STACKCAM.f0, f1 = STACKCAM.f1, fSpan = Math.max(0.05, f1 - f0);
  cv.width = W;
  cv.height = Math.max(40, MEAS.length * rowH + axisH);
  c.fillStyle = '#0d1620'; c.fillRect(0, 0, cv.width, cv.height);
  if (!MEAS.length) {
    c.fillStyle = '#8aa'; c.font = '13px sans-serif';
    c.fillText('測定するとここに1行ずつ音紋が積まれます', 12, 25);
    return;
  }
  let lo = 1e9, hi = -1e9;
  MEAS.forEach(m => melAvgOf(m).forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v); }));
  const plotW = W - L - 20;
  MEAS.forEach((m, r) => {
    const row = melAvgOf(m), y = r * rowH;
    const selected = SEL_T === m.t;
    c.fillStyle = seriesColor(m.sid); c.fillRect(0, y, L - 2, rowH - 1);
    for (let j = 0; j < N_MEL; j++) {
      const u = j / N_MEL;
      if (u < f0 || u > f1) continue;
      const x = L + (u - f0) / fSpan * plotW;
      const w = plotW / (N_MEL * fSpan) + 1;
      c.fillStyle = specColor((row[j] - lo) / (hi - lo || 1));
      c.fillRect(x, y, w, rowH - 1);
    }
    const lab = labelOf(m);
    if (lab === 'abnormal') {
      c.fillStyle = '#ff5252';
      c.beginPath();
      c.moveTo(W - 14, y + 1);
      c.lineTo(W - 2, y + rowH / 2);
      c.lineTo(W - 14, y + rowH - 1);
      c.closePath();
      c.fill();
    } else if (lab === 'normal') {
      c.fillStyle = '#66bb6a';
      c.fillRect(W - 12, y + rowH / 2 - 3, 6, 6);
    }
    if (selected) {
      c.strokeStyle = '#ffd54f'; c.lineWidth = 2;
      c.strokeRect(1, y + 0.5, W - 2, rowH - 1);
    }
  });
  c.font = '10px sans-serif'; c.fillStyle = '#cfe0ee';
  for (const hz of HZ_TICKS) {
    const u = (mel(hz) - MEL0) / (MEL1 - MEL0);
    if (u < f0 || u > f1) continue;
    const x = L + (u - f0) / fSpan * plotW;
    c.fillText(hzLabel(hz), x - 6, cv.height - 5);
  }
}

function refreshViz() {
  if ($('viewAnalyze').hidden || $('tab-viz').hidden) return;
  drawStack(); drawMap();
}

function refreshAll() {
  renderLog(); updateStorage();
  refreshViz();
}

/* ---------- 評価レポート（線形プローブ: 凍結埋め込み＋ロジスティック回帰＋交差検証） ---------- */
function trainLogReg(X, y, { lambda = 1, iters = 250, lr = 0.5 }) {
  const n = X.length, d = X[0].length;
  const w = new Float64Array(d); let b = 0;
  for (let it = 0; it < iters; it++) {
    const gw = new Float64Array(d); let gb = 0;
    for (let i = 0; i < n; i++) {
      const xi = X[i]; let z = b;
      for (let j = 0; j < d; j++) z += w[j] * xi[j];
      const p = 1 / (1 + Math.exp(-z)), e = p - y[i];
      for (let j = 0; j < d; j++) gw[j] += e * xi[j];
      gb += e;
    }
    for (let j = 0; j < d; j++) w[j] -= lr * (gw[j] / n + lambda * w[j] / n);
    b -= lr * (gb / n);
  }
  return { w, b };
}
const predictLogReg = (m, x) => { let z = m.b; for (let j = 0; j < x.length; j++) z += m.w[j] * x[j]; return 1 / (1 + Math.exp(-z)); };

function standardizer(X, idx) {               // 学習foldから列ごとの平均/標準偏差
  const d = X[0].length, mean = new Float64Array(d), std = new Float64Array(d);
  for (const i of idx) { const xi = X[i]; for (let j = 0; j < d; j++) mean[j] += xi[j]; }
  for (let j = 0; j < d; j++) mean[j] /= idx.length;
  for (const i of idx) { const xi = X[i]; for (let j = 0; j < d; j++) { const dv = xi[j] - mean[j]; std[j] += dv * dv; } }
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j] / idx.length) || 1;
  return { mean, std, fn: v => v.map((x, j) => (x - mean[j]) / std[j]) };
}
const applyStd = (mean, std, v) => v.map((x, j) => (x - mean[j]) / (std[j] || 1));

function cvScores(X, y, opts) {                // LOO(小)/5-fold(大) の交差検証スコア
  const n = X.length, scores = new Array(n).fill(0.5);
  const loo = n <= 40;
  const folds = loo
    ? X.map((_, i) => [i])
    : (() => { const f = Array.from({ length: 5 }, () => []); [...X.keys()].sort((a, b) => y[a] - y[b]).forEach((idx, k) => f[k % 5].push(idx)); return f; })();
  for (const te of folds) {
    const teSet = new Set(te), tr = [];
    for (let i = 0; i < n; i++) if (!teSet.has(i)) tr.push(i);
    if (!tr.length) continue;
    const st = standardizer(X, tr);
    const model = trainLogReg(tr.map(i => st.fn(X[i])), tr.map(i => y[i]), opts);
    for (const i of te) scores[i] = predictLogReg(model, st.fn(X[i]));
  }
  return { scores, cv: loo ? 'Leave-One-Out' : '5-fold' };
}

function auc(scores, y) {                       // Mann-Whitney（タイ平均ランク）
  const n = scores.length, np = y.reduce((a, v) => a + v, 0), nn = n - np;
  if (!np || !nn) return NaN;
  const order = [...Array(n).keys()].sort((a, b) => scores[a] - scores[b]);
  const ranks = new Array(n); let i = 0;
  while (i < n) { let j = i; while (j < n && scores[order[j]] === scores[order[i]]) j++; const r = (i + j + 1) / 2; for (let k = i; k < j; k++) ranks[order[k]] = r; i = j; }
  let sumPos = 0; for (let k = 0; k < n; k++) if (y[k]) sumPos += ranks[k];
  return (sumPos - np * (np + 1) / 2) / (np * nn);
}

function confusionAt(scores, y, thr) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  scores.forEach((s, i) => { const pred = s >= thr ? 1 : 0; if (pred && y[i]) tp++; else if (pred && !y[i]) fp++; else if (!pred && y[i]) fn++; else tn++; });
  const prec = tp + fp ? tp / (tp + fp) : 0, rec = tp + fn ? tp / (tp + fn) : 0;
  const f1 = prec + rec ? 2 * prec * rec / (prec + rec) : 0, acc = (tp + tn) / (tp + fp + tn + fn || 1);
  return { thr, tp, fp, tn, fn, prec, rec, f1, acc };
}
function bestThreshold(scores, y) {
  const cand = [...new Set(scores)].sort((a, b) => a - b);
  let best = confusionAt(scores, y, 0.5);
  for (const t of cand) { const m = confusionAt(scores, y, t); if (m.f1 > best.f1) best = m; }
  return best;
}

let REPORT = null;
const SPACES = [['yamnet', 'YAMNet'], ['dsp', 'DSP'], ['melcnn', 'MelCNN']];

function renderReportSeries() {
  const sel = $('repSeries'), cur = sel.value;
  sel.innerHTML = '<option value="all">すべてのシリーズ</option>'
    + CFG.series.map(s => `<option value="${s.id}">${s.name}</option>`).join('');
  if (cur) sel.value = cur;
}

function runReport() {
  const sidSel = $('repSeries').value, sid = sidSel === 'all' ? 'all' : +sidSel;
  const lambda = { weak: 0.1, medium: 1, strong: 5 }[$('repReg').value] || 1;
  const opts = { lambda, iters: 250, lr: 0.5 };
  const results = [];
  for (const [sp, name] of SPACES) {
    const ms = MEAS.filter(m => (sid === 'all' || m.sid === sid) && m.emb[sp] && (labelOf(m) === 'normal' || labelOf(m) === 'abnormal'));
    const nPos = ms.filter(m => labelOf(m) === 'abnormal').length, nNeg = ms.length - nPos;
    if (ms.length < 6 || nPos < 2 || nNeg < 2) { results.push({ sp, name, nPos, nNeg, insufficient: true }); continue; }
    const X = ms.map(m => m.emb[sp]), y = ms.map(m => labelOf(m) === 'abnormal' ? 1 : 0);
    const { scores, cv } = cvScores(X, y, opts);
    results.push({ sp, name, ms, scores, y, nPos, nNeg, cv, auc: auc(scores, y), best: bestThreshold(scores, y) });
  }
  results.sort((a, b) => (b.auc || -1) - (a.auc || -1));
  REPORT = { sid, results };
  renderReport();
  const first = results.find(r => !r.insufficient);
  if (first) renderReportDetail(first.sp);
}

function renderReport() {
  const ok = REPORT.results.filter(r => !r.insufficient);
  const pct = v => isNaN(v) ? '—' : (v * 100).toFixed(1) + '%';
  const f3 = v => isNaN(v) ? '—' : v.toFixed(3);
  if (!ok.length) {
    $('repSummary').innerHTML = '<div class="warn">⚠ 評価には各クラス2件以上・合計6件以上の「正常/異常」ラベルが必要です。詳細画面でラベルを付けてください。</div>';
    $('repTable').innerHTML = ''; $('repDetail').innerHTML = '';
    return;
  }
  $('repSummary').innerHTML = `交差検証: <b>${ok[0].cv}</b> ／ AUCの高い埋め込みほど、このデータで正常・異常が分離できています（行クリックで詳細）。`;
  $('repTable').innerHTML = `<table class="rep-table">
    <thead><tr><th>埋め込み</th><th>件数(正常/異常)</th><th>AUC</th><th>F1</th><th>適合率</th><th>再現率</th><th>正解率</th></tr></thead>
    <tbody>${REPORT.results.map(r => r.insufficient
    ? `<tr class="dim"><td>${r.name}</td><td>${r.nNeg}/${r.nPos}</td><td colspan="5">データ不足</td></tr>`
    : `<tr data-sp="${r.sp}"><td><b>${r.name}</b></td><td>${r.nNeg}/${r.nPos}</td><td>${f3(r.auc)}</td><td>${f3(r.best.f1)}</td><td>${pct(r.best.prec)}</td><td>${pct(r.best.rec)}</td><td>${pct(r.best.acc)}</td></tr>`).join('')}
    </tbody></table>`;
  $('repTable').querySelectorAll('tr[data-sp]').forEach(tr => tr.onclick = () => renderReportDetail(tr.dataset.sp));
}

function renderReportDetail(sp) {
  const r = REPORT.results.find(x => x.sp === sp);
  if (!r || r.insufficient) { $('repDetail').innerHTML = ''; return; }
  $('repTable').querySelectorAll('tr[data-sp]').forEach(tr => tr.classList.toggle('best', tr.dataset.sp === sp));
  const b = r.best, thr = b.thr;
  const mis = r.ms.map((m, i) => ({ m, s: r.scores[i], y: r.y[i], pred: r.scores[i] >= thr ? 1 : 0 }))
    .filter(o => o.pred !== o.y).sort((a, z) => Math.abs(z.s - thr) - Math.abs(a.s - thr));
  const pct = v => (v * 100).toFixed(1) + '%';
  $('repDetail').innerHTML = `
    <h3>${r.name} の詳細（しきい値 ${thr.toFixed(2)}・F1最大で自動選択）</h3>
    <table class="cm">
      <tr><td></td><td class="hd">予測: 異常</td><td class="hd">予測: 正常</td></tr>
      <tr><td class="hd">実際: 異常</td><td class="tp">${b.tp}<small>（正しく異常）</small></td><td class="fn">${b.fn}<small>（見逃し）</small></td></tr>
      <tr><td class="hd">実際: 正常</td><td class="fp">${b.fp}<small>（誤検知）</small></td><td class="tn">${b.tn}<small>（正しく正常）</small></td></tr>
    </table>
    <p class="hint">AUC ${isNaN(r.auc) ? '—' : r.auc.toFixed(3)} ／ 適合率 ${pct(b.prec)} ／ 再現率 ${pct(b.rec)} ／ F1 ${b.f1.toFixed(3)}</p>
    <h3>誤判定レコード（${mis.length}件・クリックで詳細）</h3>
    ${mis.length ? mis.map(o => `<div class="reco mis-item" data-t="${o.m.t}">${o.pred ? '🔴誤検知' : '🟠見逃し'} ／ ${new Date(o.m.t).toLocaleString('ja-JP')} ／ ${seriesName(o.m.sid)} ／ スコア ${o.s.toFixed(2)}${labelText(o.m) ? ' ／ 「' + labelText(o.m) + '」' : ''}</div>`).join('') : '<p class="hint">なし（この空間・しきい値では全て正解）</p>'}`;
  $('repDetail').querySelectorAll('.mis-item').forEach(el => el.onclick = () => openDetail(+el.dataset.t));
  const rr = $('repReg').value, rsid = $('repSeries').value;
  $('repDetail').insertAdjacentHTML('beforeend',
    `<div class="inline" style="margin-top:10px"><button id="repSave" class="mini" data-sp="${sp}">＋ このモデルを保存</button> <span id="repSaveMsg" class="hint"></span></div>`);
  $('repSave').onclick = () => saveModelFromReport(sp, rr, rsid);
}

/* ---------- モデルレジストリ（学習済み判定器を保存・録音時に自動判定） ---------- */
let MODELS = [];
const REG_LAMBDA = { weak: 0.1, medium: 1, strong: 5 };
const spaceName = s => ({ yamnet: 'YAMNet', dsp: 'DSP', melcnn: 'MelCNN' }[s] || s);
const regLabel = r => ({ weak: '弱', medium: '中', strong: '強' }[r] || r);
const labeledFor = (space, sid) =>
  MEAS.filter(m => (sid === 'all' || m.sid === sid) && m.emb[space] && (labelOf(m) === 'normal' || labelOf(m) === 'abnormal'));

function buildModel(space, reg, sid, name) {
  const lambda = REG_LAMBDA[reg] || 1;
  const ms = labeledFor(space, sid);
  const nPos = ms.filter(m => labelOf(m) === 'abnormal').length, nNeg = ms.length - nPos;
  if (ms.length < 6 || nPos < 2 || nNeg < 2) return null;
  const X = ms.map(m => m.emb[space]), y = ms.map(m => labelOf(m) === 'abnormal' ? 1 : 0);
  const st = standardizer(X, [...X.keys()]);
  const full = trainLogReg(X.map(st.fn), y, { lambda, iters: 400, lr: 0.5 });
  const { scores, cv } = cvScores(X, y, { lambda, iters: 250, lr: 0.5 });
  const best = bestThreshold(scores, y);
  const r4 = a => Array.from(a, x => +x.toFixed(4));
  return {
    id: Date.now(), name, space, reg, lambda, sid,
    thr: +best.thr.toFixed(3), w: r4(full.w), b: +full.b.toFixed(4), mean: r4(st.mean), std: r4(st.std),
    metrics: { auc: auc(scores, y), f1: best.f1, prec: best.prec, rec: best.rec, acc: best.acc, n: ms.length, nPos, nNeg, cv },
    createdAt: Date.now(), active: true
  };
}

async function saveModelFromReport(space, reg, sidSel) {
  const sid = sidSel === 'all' ? 'all' : +sidSel;
  const def = `${sid === 'all' ? '全体' : seriesName(sid)} / ${spaceName(space)} ${regLabel(reg)}`;
  const name = prompt('モデル名', def);
  if (name === null) return;
  const model = buildModel(space, reg, sid, name.trim() || def);
  if (!model) { $('repSaveMsg').textContent = 'ラベル不足で保存できません'; return; }
  MODELS.push(model);
  await dbPut('models', model);
  $('repSaveMsg').textContent = '✅ 保存しました（モデルタブで管理）';
  setTimeout(() => { const el = $('repSaveMsg'); if (el) el.textContent = ''; }, 2000);
}

const judgeWithModel = (model, emb) => {
  const score = predictLogReg({ w: model.w, b: model.b }, applyStd(model.mean, model.std, emb));
  return { score, pred: score >= model.thr ? 'abnormal' : 'normal' };
};
const modelsFor = (sid, emb) => MODELS.filter(m => (m.sid === 'all' || m.sid === sid) && emb[m.space]);

function judgeRecord(sid, emb) {              // 録音直後: 有効な適用可能モデルで判定
  return modelsFor(sid, emb).filter(m => m.active)
    .map(m => { const j = judgeWithModel(m, emb[m.space]); return { id: m.id, name: m.name, pred: j.pred, score: +j.score.toFixed(3) }; });
}
const judgeHTML = judge => (judge && judge.length)
  ? '<b>モデル判定:</b> ' + judge.map(j =>
    `<span class="judge ${j.pred}">${j.pred === 'abnormal' ? '🔴 異常' : '🟢 正常'} ${j.score.toFixed(2)}</span> <small>${j.name}</small>`).join(' ／ ')
  : '';

function renderModels() {
  const fmt = v => isNaN(v) ? '—' : v.toFixed(3);
  $('modelList').innerHTML = MODELS.length
    ? MODELS.map(m => `<div class="model-card${m.active ? ' on' : ''}" data-id="${m.id}">
        <div class="inline wrap">
          <b>${m.name}</b>
          <span class="chip">${spaceName(m.space)}</span>
          <span class="chip">正則化:${regLabel(m.reg)}</span>
          <span class="chip">対象:${m.sid === 'all' ? '全体' : seriesName(m.sid)}</span>
        </div>
        <p class="hint">AUC ${fmt(m.metrics.auc)} ／ F1 ${fmt(m.metrics.f1)} ／ 学習${m.metrics.n}件(正常${m.metrics.nNeg}/異常${m.metrics.nPos}) ／ しきい値 ${m.thr.toFixed(2)} ／ ${new Date(m.createdAt).toLocaleDateString('ja-JP')}</p>
        <div class="inline wrap">
          <label class="check"><input type="checkbox" class="mdl-active" ${m.active ? 'checked' : ''}> 録音時に使う</label>
          <button class="mini mdl-edit">⚙ 編集/再学習</button>
          <button class="mini mdl-rename">✏ 改名</button>
          <button class="mini danger mdl-del">✕ 削除</button>
        </div></div>`).join('')
    : '<p class="hint">まだモデルがありません。「🧪 評価」タブで空間・正則化・対象を選び「＋ このモデルを保存」で作成してください。</p>';
  $('modelList').querySelectorAll('.model-card').forEach(card => {
    const id = +card.dataset.id, m = MODELS.find(x => x.id === id);
    card.querySelector('.mdl-active').onchange = async e => { m.active = e.target.checked; await dbPut('models', m); card.classList.toggle('on', m.active); };
    card.querySelector('.mdl-rename').onclick = async () => { const n = prompt('モデル名', m.name); if (n) { m.name = n.trim() || m.name; await dbPut('models', m); renderModels(); } };
    card.querySelector('.mdl-del').onclick = async () => { if (!confirm(`「${m.name}」を削除しますか？`)) return; MODELS = MODELS.filter(x => x.id !== id); await dbDel('models', id); renderModels(); };
    card.querySelector('.mdl-edit').onclick = () => openModelEditor(m);
  });
}

/* ---------- モデル編集/再学習ダイアログ（データセット・空間・正則化を選び直せる） ---------- */
let EDIT_ID = null, PREVIEW_MODEL = null;
const previewText = m => `学習件数 ${m.metrics.n}（正常${m.metrics.nNeg}/異常${m.metrics.nPos}） ／ AUC ${isNaN(m.metrics.auc) ? '—' : m.metrics.auc.toFixed(3)} ／ F1 ${m.metrics.f1.toFixed(3)} ／ しきい値 ${m.thr.toFixed(2)}（${m.metrics.cv}）`;
function fillMdSeries() {
  $('mdSeries').innerHTML = '<option value="all">すべてのシリーズ</option>'
    + CFG.series.map(s => `<option value="${s.id}">${s.name}</option>`).join('');
}
function openModelEditor(model) {
  EDIT_ID = model ? model.id : null; PREVIEW_MODEL = null;
  $('modelDlgTitle').textContent = model ? '⚙ モデル編集・再学習' : '＋ 新規モデル';
  fillMdSeries();
  $('mdName').value = model ? model.name : '';
  $('mdSpace').value = model ? model.space : 'yamnet';
  $('mdReg').value = model ? model.reg : 'medium';
  $('mdSeries').value = model ? String(model.sid) : 'all';
  $('mdPreview').innerHTML = model ? '現在: ' + previewText(model) + '<br>設定を変えたら「学習してプレビュー」→「保存」' : '設定を選んで「学習してプレビュー」';
  $('modelDlg').showModal();
}
function mdPreview() {
  const space = $('mdSpace').value, reg = $('mdReg').value;
  const sidSel = $('mdSeries').value, sid = sidSel === 'all' ? 'all' : +sidSel;
  const name = $('mdName').value.trim() || `${sid === 'all' ? '全体' : seriesName(sid)} / ${spaceName(space)} ${regLabel(reg)}`;
  const m = buildModel(space, reg, sid, name);
  if (!m) { $('mdPreview').innerHTML = '<span class="warn">この設定・ラベルでは学習できません（各クラス2件以上・合計6件以上が必要）</span>'; PREVIEW_MODEL = null; return; }
  if (EDIT_ID) m.id = EDIT_ID;
  PREVIEW_MODEL = m;
  $('mdPreview').innerHTML = '結果: ' + previewText(m);
}
async function mdSaveModel() {
  if (!PREVIEW_MODEL) { mdPreview(); if (!PREVIEW_MODEL) return; }
  const m = PREVIEW_MODEL;
  m.name = $('mdName').value.trim() || m.name;
  if (EDIT_ID) {
    const idx = MODELS.findIndex(x => x.id === EDIT_ID);
    m.id = EDIT_ID; m.active = idx >= 0 ? MODELS[idx].active : true;
    if (idx >= 0) MODELS[idx] = m; else MODELS.push(m);
  } else MODELS.push(m);
  await dbPut('models', m);
  PREVIEW_MODEL = null;
  $('modelDlg').close(); renderModels();
}

async function reinferAll() {                 // 推論専用: 全レコードを現在の有効モデルで再判定
  if (!MEAS.length) { $('mdlMsg').textContent = 'レコードがありません'; return; }
  $('mdlMsg').textContent = '再判定中…';
  let n = 0;
  for (const m of MEAS) { m.judge = judgeRecord(m.sid, m.emb); await dbPut('meas', m); n++; }
  $('mdlMsg').textContent = `✅ ${n}件を再判定しました`;
  refreshAll();
  setTimeout(() => { const e = $('mdlMsg'); if (e) e.textContent = ''; }, 2500);
}

/* ---------- 判定バナー・トレンド・モデル比較 ---------- */
function statusOf(rec) {
  const lab = labelOf(rec);
  if (lab) return lab;
  if (rec.judge && rec.judge.length) return rec.judge.some(j => j.pred === 'abnormal') ? 'abnormal' : 'normal';
  return null;
}
function renderVerdict(rec, ps) {
  const el = $('verdict'); el.hidden = false;
  const j = rec.judge || [];
  if (j.length) {
    const abn = j.filter(o => o.pred === 'abnormal');
    const disagree = abn.length > 0 && abn.length < j.length;
    el.className = 'verdict ' + (abn.length ? 'red' : 'green');
    el.innerHTML = `<div class="v-main">${abn.length ? '🔴 異常の可能性' : '🟢 正常'}</div>`
      + `<div class="v-sub">${j.map(o => `${o.pred === 'abnormal' ? '🔴' : '🟢'}${o.name} ${o.score.toFixed(2)}`).join(' ／ ')}</div>`
      + (disagree ? '<div class="v-warn">⚠ モデル間で判定が割れています（要確認）</div>' : '');
  } else if (ps == null) {
    el.className = 'verdict gray';
    el.innerHTML = '<div class="v-main">📌 初回測定</div><div class="v-sub">基準データとして登録しました</div>';
  } else {
    const cls = ps >= 0.90 ? 'green' : ps >= 0.75 ? 'yellow' : 'red';
    const txt = ps >= 0.90 ? '🟢 いつも通り' : ps >= 0.75 ? '🟡 やや違う' : '🔴 かなり違う';
    el.className = 'verdict ' + cls;
    el.innerHTML = `<div class="v-main">${txt}</div><div class="v-sub">過去平均との類似度 ${ps.toFixed(3)} ／ モデル未保存（評価タブで作成すると判定します）</div>`;
  }
}
function renderTrend(sid) {
  const el = $('trendStrip');
  const ms = MEAS.filter(m => m.sid === sid).slice(-20);
  if (ms.length < 2) { el.innerHTML = ''; return; }
  el.innerHTML = '<span class="trend-label">最近の推移</span>' + ms.map(m => {
    const s = statusOf(m), c = s === 'abnormal' ? 'red' : s === 'normal' ? 'green' : 'gray';
    return `<i class="tdot ${c}" data-t="${m.t}" title="${new Date(m.t).toLocaleString('ja-JP')}${s ? '：' + (s === 'abnormal' ? '異常' : '正常') : ''}"></i>`;
  }).join('');
  el.querySelectorAll('.tdot').forEach(d => d.onclick = () => openDetail(+d.dataset.t));
}
function renderCompare(rec) {
  const el = $('dlgCompare');
  const apps = MODELS.filter(m => (m.sid === 'all' || m.sid === rec.sid) && rec.emb[m.space]);
  if (!apps.length) { el.innerHTML = ''; return; }
  const rows = apps.map(m => { const r = judgeWithModel(m, rec.emb[m.space]); return { m, ...r }; });
  const disagree = new Set(rows.map(r => r.pred)).size > 1;
  el.innerHTML = `<h3>モデル比較${disagree ? ' <span class="v-warn">⚠ 判定が割れています</span>' : ''}</h3>`
    + `<table class="rep-table"><thead><tr><th>モデル</th><th>空間</th><th>判定</th><th>スコア</th><th>しきい値</th><th>使用</th></tr></thead><tbody>`
    + rows.map(r => `<tr><td>${r.m.name}</td><td>${spaceName(r.m.space)}</td><td class="${r.pred}">${r.pred === 'abnormal' ? '🔴異常' : '🟢正常'}</td><td>${r.score.toFixed(2)}</td><td>${r.m.thr.toFixed(2)}</td><td>${r.m.active ? '✓' : '—'}</td></tr>`).join('')
    + '</tbody></table>';
}

function renderSeries() {
  $('seriesSel').innerHTML = CFG.series.map(s => `<option value="${s.id}">${s.name}</option>`).join('');
  updateCurChips();
}
function updateCurChips() {
  const sid = +$('seriesSel').value;
  $('curSeries').textContent = seriesName(sid);
  $('curDur').textContent = $('durSel').value;
  const cond = seriesCondText(sid);
  $('seriesCond').textContent = cond || '（条件未設定：「✏条件」で場所・対象・位置関係を登録できます）';
}

function renderLog() {
  const tb = $('logTable').tBodies[0];
  tb.innerHTML = [...MEAS].reverse().map(m => `<tr data-t="${m.t}">
    <td>${new Date(m.t).toLocaleString('ja-JP', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })}</td>
    <td><i class="dot" style="background:${seriesColor(m.sid)}"></i>${seriesName(m.sid)}</td>
    <td>${m.top ? m.top.map(t => t.name).join(' / ') : '—'}</td>
    <td>${primarySim(m) == null ? '基準' : primarySim(m).toFixed(3)}</td>
    <td title="${m.note ? m.note.text : ''}">${m.note ? '📝' : ''}</td>
    <td><button class="mini danger" data-del="${m.t}">✕</button></td></tr>`).join('');
  tb.querySelectorAll('tr').forEach(tr => tr.onclick = () => openDetail(+tr.dataset.t));
  tb.querySelectorAll('[data-del]').forEach(b => b.onclick = async e => {
    e.stopPropagation();
    MEAS = MEAS.filter(m => m.t !== +b.dataset.del);
    await dbDel('meas', +b.dataset.del); await dbDel('audio', +b.dataset.del);
    refreshAll();
  });
}

async function updateStorage() {
  try {
    const e = await navigator.storage.estimate();
    $('storageInfo').textContent = `${MEAS.length}件 / ${(e.usage / 1048576).toFixed(1)}MB使用`;
  } catch { $('storageInfo').textContent = MEAS.length + '件'; }
}

/* ---------- 測定パイプライン ---------- */
async function handleSamples(x) {           // x: Float32Array @16kHz
  const sid = +$('seriesSel').value;
  const frames = melSpec(x);
  const emb = { dsp: dspEmbed(frames) };
  let top = null;
  if (yamnet) { const r = await yamnetEmbed(x); emb.yamnet = r.emb; top = r.top; }
  if (melcnn) { const me = melcnnEmbed(frames); if (me) emb.melcnn = me; }
  const melAvg = Array.from({ length: N_MEL }, (_, j) =>
    +(frames.reduce((a, f) => a + f[j], 0) / frames.length).toFixed(2));
  let rms = 0, peak = 0;
  for (const v of x) { rms += v * v; peak = Math.max(peak, Math.abs(v)); }
  const rmsDb = 20 * Math.log10(Math.sqrt(rms / x.length) + 1e-9);
  const prev = MEAS.filter(m => m.sid === sid);
  const sim = { dsp: simTo(prev, 'dsp', emb.dsp) };
  if (emb.yamnet) sim.yamnet = simTo(prev, 'yamnet', emb.yamnet);
  if (emb.melcnn) sim.melcnn = simTo(prev, 'melcnn', emb.melcnn);
  const judge2 = judgeRecord(sid, emb);      // 保存済みモデルによる自動判定
  const rec = {
    t: Date.now(), sid, rmsDb: +rmsDb.toFixed(1), peak: +peak.toFixed(3),
    cond: lastCond, device: { ...CFG.device }, emb, top, sim, melAvg, note: null, bands: [], judge: judge2
  };
  MEAS.push(rec);
  await dbPut('meas', rec);
  await dbPut('audio', { t: rec.t, pcm: Int16Array.from(x, v => Math.max(-1, Math.min(1, v)) * 32767).buffer });

  const ps = primarySim(rec);
  $('result').hidden = false;
  renderVerdict(rec, ps);
  renderTrend(sid);
  $('resultActions').hidden = false;
  $('toDetail').onclick = () => openDetail(rec.t);
  $('toLabel').onclick = () => openDetail(rec.t);
  $('similarity').innerHTML = (ps == null
    ? '📌 このシリーズで初回の測定です。基準データとして登録しました。'
    : `${judge(ps)}（過去平均との類似度 <b>${ps.toFixed(3)}</b>${sim.yamnet != null && sim.dsp != null ? ` ／ YAMNet ${sim.yamnet.toFixed(3)}・DSP ${sim.dsp.toFixed(3)}` : ''}）`)
    + `<br><small>入力レベル: ${rmsDb.toFixed(1)} dBFS（AGCの影響で絶対値は参考程度）</small>`;
  $('modelJudge').innerHTML = judge2.length ? judgeHTML(judge2)
    : (MODELS.length ? '' : '<small class="hint">保存モデルなし — ラベルを貯めて「評価」タブでモデルを作ると、ここに自動判定が出ます</small>');
  $('quality').innerHTML = qualityHints(rec, prev).map(h => `<div class="warn">⚠ ${h}</div>`).join('');
  $('reco').innerHTML = similarNotes(rec).map(o =>
    `<div class="reco" onclick="openDetail(${o.m.t})">💡 シリーズ『${seriesName(o.m.sid)}』の異常ラベル「${labelText(o.m)}」(${new Date(o.m.t).toLocaleDateString('ja-JP')}) に似ています（類似度 ${o.s.toFixed(2)}）</div>`).join('');
  $('topClasses').innerHTML = top
    ? 'YAMNetの音クラス判定: ' + top.map(t => `<b>${t.name}</b> (${t.p.toFixed(2)})`).join(' / ')
    : '';
  drawWave(x);
  const sc = $('specCv'); sc.width = sc.clientWidth;
  renderSpec(sc.getContext('2d'), sc.width, sc.height, frames, 0, frames.length, { axes: true });
  $('openLatest').onclick = () => openDetail(rec.t);
  refreshAll();
}

$('recBtn').onclick = async () => {
  $('recBtn').disabled = true; $('live').hidden = false;
  try {
    const { data, sr } = await record(+$('durSel').value, $('rawChk').checked);
    await handleSamples(resample(data, sr, SR));
  } catch (e) {
    alert('録音できませんでした: ' + e.message +
      '\n・マイク権限を許可してください\n・スマホの場合は https でアクセスしてください（README参照）');
  }
  $('recBtn').disabled = false; $('live').hidden = true;
};

/* ---------- 再生エンジン（再生位置バー・一時停止/再開対応） ---------- */
let PB = null;                              // {kind, x, offset, playing, ctx, src, startedAt}
const pbPos = () => PB ? PB.offset + (PB.playing ? PB.ctx.currentTime - PB.startedAt : 0) : 0;
function pbStop() {
  if (!PB) return;
  if (PB.playing) { PB.src.onended = null; try { PB.src.stop(); } catch { } PB.ctx.close(); }
  PB = null; updatePlayBtns(); blit();
}
function pbPause() {
  PB.offset = pbPos(); PB.src.onended = null;
  try { PB.src.stop(); } catch { } PB.ctx.close(); PB.playing = false;
  updatePlayBtns();
}
function pbResume() {
  const ctx = new AC({ sampleRate: SR });
  const b = ctx.createBuffer(1, PB.x.length, SR); b.copyToChannel(PB.x, 0);
  const s = ctx.createBufferSource(); s.buffer = b; s.connect(ctx.destination);
  s.start(0, PB.offset);
  PB.ctx = ctx; PB.src = s; PB.startedAt = ctx.currentTime; PB.playing = true;
  s.onended = pbStop;
  updatePlayBtns();
  (function loop() { if (PB && PB.playing) { blit(); requestAnimationFrame(loop); } })();
}
function playToggle(kind) {
  if (PB && PB.kind === kind) { PB.playing ? pbPause() : pbResume(); return; }
  pbStop();
  const bands = activeBands();
  const x = kind === 'orig' ? D.x : bandsFilter(D.x, bands, kind === 'only');
  PB = { kind, x, offset: 0, playing: false };
  pbResume();
}
function updatePlayBtns() {
  const lbl = { orig: '原音', mute: '帯域ミュート', only: '帯域のみ' };
  for (const k of ['orig', 'mute', 'only']) {
    const b = $('play' + k[0].toUpperCase() + k.slice(1));
    b.textContent = (PB && PB.kind === k && PB.playing ? '⏸ ' : '▶ ') + lbl[k];
  }
  const has = D && D.x;
  $('playOrig').disabled = !has;
  $('playMute').disabled = $('playOnly').disabled = !(has && activeBands().length);
}

/* ---------- 詳細モーダル: ズーム/パン・複数帯域・メモ ---------- */
let D = null;                               // {rec, x, frames, view:[0,1], mode, tempBand}
const OFF = document.createElement('canvas');   // 再生バー高速描画用のキャッシュ
const activeBands = () => D ? D.rec.bands.filter(b => b.on).map(b => [b.lo, b.hi]) : [];

async function openDetail(t) {
  pbStop();
  const rec = MEAS.find(m => m.t === t);
  rec.bands = rec.bands || [];
  const a = await dbGet('audio', t);
  const x = a ? Float32Array.from(new Int16Array(a.pcm), v => v / 32767) : null;
  D = { rec, x, frames: x ? melSpec(x) : null, view: [0, 1], mode: 'view', tempBand: null };
  $('dlgTitle').textContent = `${new Date(rec.t).toLocaleString('ja-JP')} — ${seriesName(rec.sid)}`;
  const ps = primarySim(rec);
  const simTxt = ['yamnet', 'dsp', 'melcnn'].filter(k => rec.sim && rec.sim[k] != null)
    .map(k => `${k === 'yamnet' ? 'YAMNet' : k === 'dsp' ? 'DSP' : 'MelCNN'} ${rec.sim[k].toFixed(3)}`).join(' / ') || '—';
  const devTxt = rec.device && (rec.device.name || rec.device.mic)
    ? `📱${rec.device.name || '—'}${rec.device.mic ? ' / 🎤' + rec.device.mic : ''}` : '';
  const condTxt = seriesCondText(rec.sid);
  $('dlgInfo').innerHTML =
    `類似度: ${ps == null ? '基準' : ps.toFixed(3)}（${simTxt}） ／ `
    + `入力 ${rec.rmsDb} dBFS ／ 録音条件 ${rec.cond ? `AGC:${rec.cond.agc ? 'ON' : 'OFF'} NS:${rec.cond.ns ? 'ON' : 'OFF'} EC:${rec.cond.ec ? 'ON' : 'OFF'}` : '不明'}`
    + (devTxt ? `<br>${devTxt}` : '')
    + (condTxt ? `<br>${condTxt}` : '')
    + (rec.top ? `<br>YAMNet判定: ${rec.top.map(o => `${o.name} (${o.p.toFixed(2)})`).join(' / ')}` : '')
    + (rec.judge && rec.judge.length ? `<br>${judgeHTML(rec.judge)}` : '')
    + (x ? '' : '<br>⚠ 音声データなし（インポート由来のレコード）— 再生・帯域分析は不可');
  // ラベル: 既存 → （未設定なら）直前のラベル付きレコードから継承
  let lblStatus = rec.label?.status || (rec.note?.text ? 'abnormal' : null);
  let lblTextV = rec.label ? (rec.label.text || '') : (rec.note?.text || '');
  const idx = MEAS.findIndex(m => m.t === rec.t);
  if (!rec.label && !rec.note) {
    for (let k = idx - 1; k >= 0; k--) {
      if (MEAS[k].label && MEAS[k].label.status) { lblStatus = MEAS[k].label.status; lblTextV = MEAS[k].label.text || ''; break; }
    }
  }
  setLabelForm(lblStatus, lblTextV);
  $('recPos').textContent = `${idx + 1} / ${MEAS.length}`;
  $('prevRec').disabled = idx <= 0;
  $('nextRec').disabled = idx >= MEAS.length - 1;
  renderCompare(rec);
  setMode('view'); renderBands();
  renderSimilar(rec);
  $('detailDlg').showModal();
  redraw();                                 // showModal後でないとcanvas幅が0で描画されない
}
window.openDetail = openDetail;

function renderSimilar(rec) {
  const sims = similarNotes(rec);
  $('dlgSimilar').innerHTML = sims.length
    ? '<h3>💡 似ている異常ラベル（シリーズ横断）</h3>' + sims.map(o =>
      `<div class="reco" data-t="${o.m.t}">『${seriesName(o.m.sid)}』${new Date(o.m.t).toLocaleDateString('ja-JP')}「${labelText(o.m)}」（類似度 ${o.s.toFixed(2)}）</div>`).join('')
    : '';
  $('dlgSimilar').querySelectorAll('[data-t]').forEach(el => el.onclick = () => openDetail(+el.dataset.t));
}

/* ---------- ラベル付与（正常/異常＋テキスト）・前後レコードナビ ---------- */
function updateLabelButtons() {
  $('lblNormal').classList.toggle('on', D.labelStatus === 'normal');
  $('lblAbnormal').classList.toggle('on', D.labelStatus === 'abnormal');
  $('labelBox').classList.toggle('is-abnormal', D.labelStatus === 'abnormal');
}
function setLabelForm(status, text) {
  D.labelStatus = status || null;
  $('noteText').value = text || '';
  updateLabelButtons();
}
function setLabelStatus(s) {
  D.labelStatus = (D.labelStatus === s ? null : s);   // 同じボタン再押下で未ラベルに戻す
  updateLabelButtons();
}
async function commitLabel() {
  if (!D) return;
  const text = $('noteText').value.trim();
  const status = D.labelStatus;
  D.rec.label = status ? { status, text } : (text ? { status: 'abnormal', text } : null);
  if (D.rec.label) D.rec.note = null;       // 旧noteはラベルへ一本化
  await dbPut('meas', D.rec);
}
async function gotoRec(dir) {
  if (!D) return;
  const idx = MEAS.findIndex(m => m.t === D.rec.t);
  const ni = idx + dir;
  if (ni < 0 || ni >= MEAS.length) return;
  await commitLabel();                      // 矢印移動時は現在のラベルを自動確定（連続ラベル付け）
  refreshAll();
  openDetail(MEAS[ni].t);
}

function redraw() {
  const cv = $('bigSpec');
  OFF.width = cv.width = cv.clientWidth || OFF.width; OFF.height = cv.height;
  if (!D || !D.frames) { blit(); return; }
  const n = D.frames.length;
  const i0 = Math.max(0, Math.floor(D.view[0] * n)), i1 = Math.min(n, Math.max(i0 + 2, Math.ceil(D.view[1] * n)));
  const bands = [...D.rec.bands, ...(D.tempBand ? [{ ...D.tempBand, on: true, temp: true }] : [])];
  renderSpec(OFF.getContext('2d'), OFF.width, OFF.height, D.frames, i0, i1, { axes: true, bands });
  blit();
}
function blit() {                           // キャッシュ描画＋再生位置バー
  const cv = $('bigSpec'), c = cv.getContext('2d');
  c.drawImage(OFF, 0, 0);
  if (PB && D && D.frames) {
    const frac = (pbPos() * SR / HOP / D.frames.length - D.view[0]) / (D.view[1] - D.view[0]);
    if (frac >= 0 && frac <= 1) {
      c.strokeStyle = '#fff'; c.lineWidth = 2;
      c.beginPath(); c.moveTo(frac * cv.width, 0); c.lineTo(frac * cv.width, cv.height); c.stroke();
    }
  }
}

function setMode(m) {
  D.mode = m;
  $('modeView').classList.toggle('on', m === 'view');
  $('modeBand').classList.toggle('on', m === 'band');
}

function renderBands() {
  $('bandList').innerHTML = D.rec.bands.length
    ? D.rec.bands.map((b, i) =>
      `<span class="chip band${b.on ? '' : ' offb'}">
        <button class="tgl" data-i="${i}" title="有効/無効">${b.on ? '🔴' : '⚪'}</button>
        ${Math.round(b.lo)}–${Math.round(b.hi)} Hz
        <button class="del" data-i="${i}" title="削除">✕</button></span>`).join('')
    : '<span class="hint">帯域未選択 — 「帯域選択」モードで縦（周波数方向）にドラッグして追加</span>';
  $('bandList').querySelectorAll('.tgl').forEach(b => b.onclick = () => {
    D.rec.bands[+b.dataset.i].on = !D.rec.bands[+b.dataset.i].on;
    dbPut('meas', D.rec); renderBands(); redraw(); updatePlayBtns();
  });
  $('bandList').querySelectorAll('.del').forEach(b => b.onclick = () => {
    D.rec.bands.splice(+b.dataset.i, 1);
    dbPut('meas', D.rec); renderBands(); redraw(); updatePlayBtns();
  });
  updatePlayBtns();
}

const cvB = $('bigSpec');
let drag = null;
const specPtrs = new Map();
let specPinch0 = null;
cvB.onpointerdown = e => {
  if (!D || !D.frames) return;
  cvB.setPointerCapture(e.pointerId);
  const r = cvB.getBoundingClientRect();
  const p = { x: e.clientX - r.left, y: e.clientY - r.top };
  specPtrs.set(e.pointerId, p);
  if (specPtrs.size === 2) {
    const [a, b] = [...specPtrs.values()];
    specPinch0 = { dist: Math.hypot(a.x - b.x, a.y - b.y), view: [...D.view] };
    drag = null;
    return;
  }
  drag = { x: p.x, y: p.y, view: [...D.view], moved: false };
  if (D.mode === 'view') cvB.style.cursor = 'grabbing';
};
cvB.onpointermove = e => {
  if (!D || !D.frames) return;
  const r = cvB.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  if (specPtrs.has(e.pointerId)) specPtrs.set(e.pointerId, { x: px, y: py });
  if (specPtrs.size === 2 && specPinch0) {
    const [a, b] = [...specPtrs.values()];
    const dist = Math.hypot(a.x - b.x, a.y - b.y);
    const z = specPinch0.dist / Math.max(1, dist); // ピンチアウトで拡大
    const span0 = specPinch0.view[1] - specPinch0.view[0];
    const ns = Math.max(0.02, Math.min(1, span0 * z));
    const mid = (a.x + b.x) / 2 / r.width;
    const cx = specPinch0.view[0] + mid * span0;
    let v0 = Math.max(0, Math.min(1 - ns, cx - mid * ns));
    D.view = [v0, v0 + ns];
    redraw();
    return;
  }
  const span = D.view[1] - D.view[0];
  const sec = (D.view[0] + px / r.width * span) * D.frames.length * HOP / SR;
  $('readout').textContent = `⏱ ${sec.toFixed(2)} 秒 ／ 🎵 約 ${Math.round(hzOfY(py, r.height))} Hz`;
  if (!drag) return;
  if (Math.abs(px - drag.x) + Math.abs(py - drag.y) > 6) drag.moved = true;
  if (D.mode === 'view') {
    const dx = (px - drag.x) / r.width * span;
    let v0 = Math.max(0, Math.min(1 - span, drag.view[0] - dx));
    D.view = [v0, v0 + span];
  } else {
    const [a, b] = [hzOfY(drag.y, r.height), hzOfY(py, r.height)].sort((u, v) => u - v);
    D.tempBand = { lo: a, hi: b };
  }
  redraw();
};
cvB.onpointerup = e => {
  specPtrs.delete(e.pointerId);
  if (specPtrs.size < 2) specPinch0 = null;
  cvB.style.cursor = 'crosshair';
  if (specPtrs.size) return;
  if (D && D.mode === 'band' && D.tempBand && drag && drag.moved) {
    D.rec.bands.push({ ...D.tempBand, on: true });
    dbPut('meas', D.rec); renderBands();
    D.tempBand = null; redraw();
    playToggle('mute');                     // 帯域選択したらすぐ試聴
  } else if (D) { D.tempBand = null; redraw(); }
  drag = null;
};
cvB.onpointercancel = () => { specPtrs.clear(); specPinch0 = null; drag = null; };
cvB.ondblclick = () => { if (D) { D.view = [0, 1]; redraw(); } };   // ダブルクリックで全体表示
cvB.addEventListener('wheel', e => {
  if (!D || !D.frames) return;
  e.preventDefault();
  const r = cvB.getBoundingClientRect(), fx = (e.clientX - r.left) / r.width;
  const span = D.view[1] - D.view[0], z = e.deltaY < 0 ? 1 / 1.3 : 1.3;
  const ns = Math.max(0.02, Math.min(1, span * z));
  const cx = D.view[0] + fx * span;
  let v0 = Math.max(0, Math.min(1 - ns, cx - fx * ns));
  D.view = [v0, v0 + ns];
  redraw();
}, { passive: false });

$('modeView').onclick = () => setMode('view');
$('modeBand').onclick = () => setMode('band');
$('zoomReset').onclick = () => { D.view = [0, 1]; redraw(); };
$('playOrig').onclick = () => playToggle('orig');
$('playMute').onclick = () => playToggle('mute');
$('playOnly').onclick = () => playToggle('only');
$('lblNormal').onclick = () => setLabelStatus('normal');
$('lblAbnormal').onclick = () => setLabelStatus('abnormal');
$('prevRec').onclick = () => gotoRec(-1);
$('nextRec').onclick = () => gotoRec(1);
$('noteSave').onclick = async () => {
  await commitLabel();
  $('noteSave').textContent = '✅ 保存しました';
  setTimeout(() => $('noteSave').textContent = '💾 ラベル保存', 1200);
  renderSimilar(D.rec);
  refreshAll();
};
$('detailDlg').addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA') return;       // メモ入力中は矢印キーで移動しない
  if (e.key === 'ArrowLeft') { e.preventDefault(); gotoRec(-1); }
  if (e.key === 'ArrowRight') { e.preventDefault(); gotoRec(1); }
});
$('dlgClose').onclick = () => { pbStop(); $('detailDlg').close(); };

/* ---------- 画面切替（測定/解析・タブ・設定ドロワー） ---------- */
function showView(v) {
  $('viewMeasure').hidden = v !== 'measure';
  $('viewAnalyze').hidden = v !== 'analyze';
  $('navMeasure').classList.toggle('on', v === 'measure');
  $('navAnalyze').classList.toggle('on', v === 'analyze');
  if (v === 'analyze') showTab(document.querySelector('.tabs .on')?.dataset.tab || 'viz');
}
function showTab(t) {
  ['viz', 'report', 'models', 'list'].forEach(k => $('tab-' + k).hidden = k !== t);
  document.querySelectorAll('.tabs [data-tab]').forEach(b => b.classList.toggle('on', b.dataset.tab === t));
  if (t === 'viz') refreshViz();
  if (t === 'report') renderReportSeries();
  if (t === 'models') renderModels();
}
$('navMeasure').onclick = () => showView('measure');
$('navAnalyze').onclick = () => showView('analyze');
document.querySelectorAll('.tabs [data-tab]').forEach(b => b.onclick = () => showTab(b.dataset.tab));

/* ---------- 可視化: ズーム/パン/選択ハンドラ ---------- */
function mapZoomAt(factor, mx, my, r) {
  const z0 = MAPCAM.z, z1 = Math.max(0.6, Math.min(12, z0 * factor));
  const cx = (r?.width || $('mapCv').width) / 2, cy = (r?.height || $('mapCv').height) / 2;
  // カーソル位置を固定してズーム
  MAPCAM.ox = mx - ((mx - MAPCAM.ox - cx) * (z1 / z0) + cx);
  MAPCAM.oy = my - ((my - MAPCAM.oy - cy) * (z1 / z0) + cy);
  MAPCAM.z = z1;
  drawMap();
}
function mapPan(dx, dy) {
  MAPCAM.ox += dx; MAPCAM.oy += dy;
  drawMap();
}
function stackZoomAt(factor, mx, my) {
  const L = 8, plotW = $('stackCv').width - L - 20;
  const u = Math.max(0, Math.min(1, STACKCAM.f0 + (mx - L) / plotW * (STACKCAM.f1 - STACKCAM.f0)));
  const span0 = STACKCAM.f1 - STACKCAM.f0;
  const span1 = Math.max(0.08, Math.min(1, span0 / factor));
  STACKCAM.f0 = Math.max(0, Math.min(1 - span1, u - (u - STACKCAM.f0) * (span1 / span0)));
  STACKCAM.f1 = STACKCAM.f0 + span1;
  STACKCAM.rowH = Math.max(6, Math.min(28, STACKCAM.rowH * (factor > 1 ? 1.08 : 1 / 1.08)));
  drawStack();
}
function stackPan(dx, dy) {
  const span = STACKCAM.f1 - STACKCAM.f0;
  const plotW = Math.max(1, $('stackCv').width - 28);
  const du = -dx / plotW * span;
  let f0 = Math.max(0, Math.min(1 - span, STACKCAM.f0 + du));
  STACKCAM.f0 = f0; STACKCAM.f1 = f0 + span;
  const wrap = $('stackWrap');
  if (wrap) wrap.scrollTop = Math.max(0, wrap.scrollTop - dy);
  drawStack();
}

attachCanvasNav($('mapCv'), {
  onPan: (dx, dy) => mapPan(dx, dy),
  onZoom: (f, x, y, r) => mapZoomAt(f, x, y, r),
  onTap: (x, y) => {
    const hit = hitMap(x, y);
    if (!hit) return;
    setSelection(SEL_T === hit.t ? null : hit.t);
  },
  onDbl: (x, y) => {
    const hit = hitMap(x, y);
    if (hit) openDetail(hit.t);
  }
});

attachCanvasNav($('stackCv'), {
  onPan: (dx, dy) => stackPan(dx, dy),
  onZoom: (f, x, y) => stackZoomAt(f, x, y),
  onTap: (x, y) => {
    const row = Math.floor(y / STACKCAM.rowH);
    const m = MEAS[row];
    if (!m) return;
    setSelection(SEL_T === m.t ? null : m.t);
  },
  onDbl: (x, y) => {
    const row = Math.floor(y / STACKCAM.rowH);
    if (MEAS[row]) openDetail(MEAS[row].t);
  }
});

$('stackCv').addEventListener('pointermove', e => {
  if (e.buttons) return;
  const r = $('stackCv').getBoundingClientRect();
  const y = e.clientY - r.top, x = e.clientX - r.left;
  const row = Math.floor(y / STACKCAM.rowH);
  const m = MEAS[row];
  if (!m || SEL_T) return;
  const L = 8, plotW = $('stackCv').width - L - 20;
  const u = STACKCAM.f0 + (x - L) / plotW * (STACKCAM.f1 - STACKCAM.f0);
  const hz = u >= 0 && u <= 1 ? Math.round(imel(MEL0 + (MEL1 - MEL0) * u)) : null;
  $('stackInfo').textContent = `${new Date(m.t).toLocaleString('ja-JP')} ／ ${seriesName(m.sid)}`
    + (hz ? ` ／ 🎵 約 ${hz} Hz` : '') + (m.note ? ` ／ 📝 ${m.note.text}` : '')
    + '（タップで選択／ダブルクリックで詳細）';
});

$('mapZoomIn').onclick = () => mapZoomAt(1.25, $('mapCv').width / 2, $('mapCv').height / 2, $('mapCv').getBoundingClientRect());
$('mapZoomOut').onclick = () => mapZoomAt(1 / 1.25, $('mapCv').width / 2, $('mapCv').height / 2, $('mapCv').getBoundingClientRect());
$('mapZoomReset').onclick = () => { MAPCAM = { z: 1, ox: 0, oy: 0 }; drawMap(); };
$('stackZoomIn').onclick = () => stackZoomAt(1.2, $('stackCv').width / 2, 0);
$('stackZoomOut').onclick = () => stackZoomAt(1 / 1.2, $('stackCv').width / 2, 0);
$('stackZoomReset').onclick = () => { STACKCAM = { rowH: 12, f0: 0, f1: 1, yOff: 0 }; drawStack(); };
$('selClear').onclick = () => setSelection(null);

const openDrawer = () => { $('drawer').classList.add('open'); $('backdrop').hidden = false; };
const closeDrawer = () => { $('drawer').classList.remove('open'); $('backdrop').hidden = true; };
$('drawerBtn').onclick = openDrawer;
$('condBtn').onclick = openDrawer;
$('drawerClose').onclick = closeDrawer;
$('backdrop').onclick = closeDrawer;
$('seriesSel').onchange = updateCurChips;
$('durSel').onchange = updateCurChips;

/* ---------- シリーズ・エクスポート・まなび ---------- */
let SERIES_EDIT_ID = null;
function openSeriesDialog(sid) {              // sid=null なら新規
  SERIES_EDIT_ID = sid;
  const s = sid != null ? seriesById(sid) : {};
  $('seriesDlgTitle').textContent = sid != null ? '✏ シリーズ条件の編集' : '＋ 新規シリーズ';
  $('sfName').value = s.name || '';
  $('sfLoc').value = s.location || '';
  $('sfTarget').value = s.target || '';
  $('sfPos').value = s.position || '';
  $('sfNote').value = s.note || '';
  $('seriesDlg').showModal();
  $('sfName').focus();
}
function saveSeriesDialog() {
  const name = $('sfName').value.trim();
  if (!name) { $('sfName').focus(); return; }
  const fields = {
    name, location: $('sfLoc').value.trim(), target: $('sfTarget').value.trim(),
    position: $('sfPos').value.trim(), note: $('sfNote').value.trim()
  };
  if (SERIES_EDIT_ID != null) {
    Object.assign(seriesById(SERIES_EDIT_ID), fields);
  } else {
    const id = CFG.nextSid++;
    CFG.series.push({ id, ...fields });
    saveCfg(); renderSeries(); $('seriesSel').value = id;
  }
  saveCfg(); renderSeries();
  if (SERIES_EDIT_ID != null) $('seriesSel').value = SERIES_EDIT_ID;
  updateCurChips();
  $('seriesDlg').close();
}
$('addSeries').onclick = () => openSeriesDialog(null);
$('editSeries').onclick = () => openSeriesDialog(+$('seriesSel').value);
$('sfSave').onclick = saveSeriesDialog;
$('seriesDlgClose').onclick = () => $('seriesDlg').close();

$('devName').value = CFG.device.name || '';
$('devMic').value = CFG.device.mic || '';
$('devName').oninput = () => { CFG.device.name = $('devName').value; saveCfg(); };
$('devMic').oninput = () => { CFG.device.mic = $('devMic').value; saveCfg(); };
$('devAuto').onclick = () => { $('devName').value = guessDeviceName(); CFG.device.name = $('devName').value; saveCfg(); };
$('exportBtn').onclick = () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify({ ...CFG, meas: MEAS })], { type: 'application/json' }));
  a.download = 'otoscope-' + new Date().toISOString().slice(0, 10) + '.json';
  a.click();
};
$('importFile').onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  if (!confirm('現在のデータをインポート内容で置き換えます（音声はJSONに含まれないため消えます）。よろしいですか？')) return;
  const j = JSON.parse(await f.text());
  CFG = { series: j.series, nextSid: j.nextSid, device: j.device || { name: '', mic: '' } }; saveCfg();
  MEAS = j.meas || [];
  await dbClear('meas'); await dbClear('audio');
  for (const m of MEAS) { m.judge = judgeRecord(m.sid, m.emb); await dbPut('meas', m); }
  $('devName').value = CFG.device.name || ''; $('devMic').value = CFG.device.mic || '';
  renderSeries(); refreshAll();
};
$('clearBtn').onclick = async () => {
  if (!confirm('全ての測定データ・音声・シリーズを削除します。よろしいですか？')) return;
  CFG = { series: [{ id: 1, name: 'デフォルト' }], nextSid: 2, device: CFG.device }; saveCfg();
  MEAS = [];
  await dbClear('meas'); await dbClear('audio');
  renderSeries(); refreshAll(); $('result').hidden = true;
};
$('learnBtn').onclick = () => $('learnDlg').showModal();
$('learnClose').onclick = () => $('learnDlg').close();
document.querySelectorAll('[name=space]').forEach(r => r.onchange = () => { MAPCAM = { z: 1, ox: 0, oy: 0 }; drawMap(); });

$('repRun').onclick = () => {
  $('repSummary').textContent = '計算中…';
  $('repTable').innerHTML = ''; $('repDetail').innerHTML = '';
  setTimeout(runReport, 30);                 // 先に「計算中」を描画してから実行
};
$('repReg').onchange = () => { if (REPORT) runReport(); };
$('repSeries').onchange = () => { if (REPORT) runReport(); };

$('mdlNew').onclick = () => openModelEditor(null);
$('mdlReinfer').onclick = reinferAll;
$('mdPreviewBtn').onclick = mdPreview;
$('mdSave').onclick = mdSaveModel;
$('modelDlgClose').onclick = () => $('modelDlg').close();

/* ---------- 起動 ---------- */
$('appVer').textContent = 'v' + APP_VERSION;
$('footVer').textContent = 'v' + APP_VERSION;

(async () => {
  db = await idbOpen();
  MEAS = (await dbAll('meas')).sort((a, b) => a.t - b.t);
  MODELS = (await dbAll('models')).sort((a, b) => a.createdAt - b.createdAt);
  renderSeries(); refreshAll();
  initYamnet();
  initMelCnn();
})();
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js');
window._test = handleSamples;               // マイク無し環境の動作確認用フック（開発用）
