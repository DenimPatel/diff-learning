// diff-learning: static single-page app.
// Course content comes from course.json (built by tools/build.py); Python runs in worker.js (Pyodide).

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// Browser storage is a convenience only: every access may throw (private mode, blocked storage).
const store = {
  get(key, fallback) { try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch (e) { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* ignore */ } },
  del(key) { try { localStorage.removeItem(key); } catch (e) { /* ignore */ } },
};

// Display preferences: one object in localStorage, applied as classes and CSS variables on <html>.
// navW/labW are null until the user drags a pane, so the CSS defaults (and breakpoints) apply.
const DEFAULT_PREFS = {
  navHidden: false, labHidden: false, navW: null, labW: null, focus: false,
  ctxLines: 3, diffWrap: false, codeSize: 12.5, contentWidth: 'normal', proseSize: 'm',
  smooth: 0.9,
};
const PREF_CLASSES = { navHidden: 'nav-hidden', labHidden: 'lab-hidden', focus: 'focus', diffWrap: 'diff-wrap' };
const CONTENT_WIDTHS = { narrow: '720px', normal: '920px', wide: '1200px', full: 'none' };
const PROSE_SIZES = { s: '14px', m: '15px', l: '17px' };
const DIFF_PREFS = ['ctxLines'];
const LAYOUT_PREFS = ['navHidden', 'labHidden', 'navW', 'labW', 'focus', 'codeSize', 'contentWidth'];

function loadPrefs() {
  const saved = store.get('dl-prefs', null);
  const prefs = { ...DEFAULT_PREFS, ...(saved && typeof saved === 'object' ? saved : {}) };
  if (!saved) prefs.smooth = store.get('dl-smooth', DEFAULT_PREFS.smooth); // older key
  return prefs;
}

const MAX_RUNS = 8; // one per categorical color slot
const S = {
  course: null, byId: {}, order: [],
  view: 'lesson', lessonId: null,
  compare: '', diffMode: 'changes',
  knobValues: {},           // source key -> {name: value}
  runs: [], nextRunId: 1, running: null,
  metric: 'loss', logY: false, smooth: 0.9, showTable: false,
  worker: null, workerReady: false,
  editor: null, pgMode: 'edit', pgKnobs: [],
  done: new Set(store.get('dl-done', [])),
  prefs: loadPrefs(),
};

function applyPrefs() {
  const root = document.documentElement, p = S.prefs;
  for (const [key, cls] of Object.entries(PREF_CLASSES)) root.classList.toggle(cls, !!p[key]);
  const vars = {
    '--nav-w': p.navW && p.navW + 'px',
    '--lab-w': p.labW && p.labW + 'px',
    '--code-size': p.codeSize + 'px',
    '--content-width': CONTENT_WIDTHS[p.contentWidth],
    '--prose-size': PROSE_SIZES[p.proseSize],
  };
  for (const [k, v] of Object.entries(vars)) v ? root.style.setProperty(k, v) : root.style.removeProperty(k);
  syncPrefControls();
}

function setPrefs(changes) {
  Object.assign(S.prefs, changes);
  store.set('dl-prefs', S.prefs);
  applyPrefs();
  const keys = Object.keys(changes);
  if (keys.some(k => DIFF_PREFS.includes(k))) rerenderDiffs();
  if (keys.some(k => LAYOUT_PREFS.includes(k))) layoutChanged();
}
function setPref(key, value) { setPrefs({ [key]: value }); }

// reflect prefs in whatever controls show them
function syncPrefControls() {
  document.querySelectorAll('[data-pref]').forEach(el => {
    const v = S.prefs[el.dataset.pref];
    if (el.type === 'checkbox') el.checked = !!v;
    else if (el.classList.contains('seg-btn')) el.classList.toggle('active', String(v) === el.dataset.value);
    else el.value = v;
  });
  syncPaneToggles();
}

// on phones the lesson list is a drawer; elsewhere it's a pane the user can hide
const isDrawer = () => matchMedia('(max-width: 760px)').matches;
function toggleNav() {
  if (isDrawer()) { document.body.classList.toggle('nav-open'); syncPaneToggles(); }
  else setPref('navHidden', !S.prefs.navHidden);
}
function toggleLab() { setPref('labHidden', !S.prefs.labHidden); }
function syncPaneToggles() {
  const navOpen = isDrawer() ? document.body.classList.contains('nav-open') : !S.prefs.navHidden && !S.prefs.focus;
  const labOpen = !S.prefs.labHidden && !S.prefs.focus;
  const nav = $('#nav-toggle'), lab = $('#lab-toggle');
  nav.setAttribute('aria-expanded', String(navOpen));
  nav.title = `${navOpen ? 'Hide' : 'Show'} lessons  [`;
  lab.setAttribute('aria-expanded', String(labOpen));
  lab.title = `${labOpen ? 'Hide' : 'Show'} the lab  ]`;
}

// typing in a field or the editor: single-key shortcuts must not fire
const isTyping = (e) => !!e.target.closest?.('input, select, textarea, [contenteditable], .CodeMirror');

// after a pane changes size: CodeMirror measures itself only when told, Chart.js follows its container
function layoutChanged() {
  requestAnimationFrame(() => { if (S.editor) S.editor.refresh(); if (chart) chart.resize(); });
}

function rerenderDiffs() {
  if (!S.lessonId) return;
  if (S.view === 'lesson') renderLessonDiff();
  else if (S.pgMode === 'diff') setPgMode('diff');
}

// ---------------------------------------------------------------- boot
async function boot() {
  try {
    S.course = await (await fetch('course.json')).json();
  } catch (e) {
    $('#main').innerHTML = '<p style="padding:24px">Could not load <code>course.json</code>. Build the site with <code>python tools/build.py</code> and serve <code>dist/</code>.</p>';
    return;
  }
  for (const l of S.course.lessons) { S.byId[l.id] = l; S.order.push(l.id); }
  S.smooth = S.prefs.smooth;
  $('#smooth').value = S.smooth;
  applyPrefs();
  buildNav();
  bindUi();
  window.addEventListener('hashchange', route);
  route();
}

function lessonNum(l) { return l.id.split('-')[0]; }
function lessonShort(l) { return `${lessonNum(l)} · ${l.title}`; }
function sourceKey() { return S.view === 'playground' ? 'pg:' + S.lessonId : S.lessonId; }

// ---------------------------------------------------------------- routing
function route() {
  const m = location.hash.match(/^#\/(lesson|playground)\/([\w-]+)/);
  if (!m || !S.byId[m[2]]) {
    const last = store.get('dl-last', S.order[0]);
    location.replace('#/lesson/' + (S.byId[last] ? last : S.order[0]));
    return;
  }
  const [, view, id] = m;
  const lessonChanged = id !== S.lessonId;
  S.view = view;
  S.lessonId = id;
  store.set('dl-last', id);
  document.body.classList.remove('nav-open');
  syncPaneToggles();

  $('#tab-lesson').href = '#/lesson/' + id;
  $('#tab-playground').href = '#/playground/' + id;
  $('#tab-lesson').classList.toggle('active', view === 'lesson');
  $('#tab-playground').classList.toggle('active', view === 'playground');
  $('#lesson-view').hidden = view !== 'lesson';
  $('#playground-view').hidden = view !== 'playground';
  document.querySelectorAll('.nav-item').forEach(a => a.classList.toggle('active', a.dataset.id === id));

  if (view === 'lesson') {
    if (lessonChanged) S.compare = '';
    renderLesson();
  } else {
    renderPlayground();
  }
  renderLab();
  if (lessonChanged) $('#main').scrollTop = 0;
}

// ---------------------------------------------------------------- nav
function buildNav() {
  const nav = $('#nav');
  nav.innerHTML = S.course.parts.map(part => {
    const items = S.course.lessons.filter(l => l.part === part.id).map(l => `
      <a class="nav-item${l.variant ? ' variant' : ''}" href="#/lesson/${l.id}" data-id="${l.id}" title="${esc(l.summary)}">
        <span class="num">${esc(lessonNum(l))}</span><span>${esc(l.title)}</span>
        <span class="done" aria-label="ran">${S.done.has(l.id) ? '✓' : ''}</span>
      </a>`).join('');
    return `<div class="part"><div class="part-title">${esc(part.title)}</div>
      <div class="part-summary">${esc(part.summary || '')}</div>${items}</div>`;
  }).join('');
}

function markDone(id) {
  if (S.done.has(id)) return;
  S.done.add(id);
  store.set('dl-done', [...S.done]);
  const el = document.querySelector(`.nav-item[data-id="${id}"] .done`);
  if (el) el.textContent = '✓';
}

// ---------------------------------------------------------------- lesson view
function renderLesson() {
  const l = S.byId[S.lessonId];
  const part = S.course.parts.find(p => p.id === l.part);
  const parent = l.parent && S.byId[l.parent];
  document.title = `${l.title} · diff-learning`;
  $('#crumb').textContent = `${part.title} · lesson ${lessonNum(l)}${l.variant ? ' · branch' : ''}`;
  $('#lesson-title').textContent = l.title;
  $('#lesson-summary').textContent = l.summary;
  const meta = [];
  if (parent) meta.push(`builds on <a href="#/lesson/${parent.id}">${esc(lessonShort(parent))}</a>`);
  const children = S.course.lessons.filter(c => c.parent === l.id);
  if (children.length) meta.push(`leads to ${children.map(c => `<a href="#/lesson/${c.id}">${esc(lessonNum(c))}</a>`).join(', ')}`);
  meta.push(...l.tags.map(t => `<span class="chip">${esc(t)}</span>`));
  if (l.updated) meta.push(`updated ${esc(l.updated)}`);
  $('#lesson-meta').innerHTML = meta.join('<span aria-hidden="true">·</span>');

  const body = $('#lesson-body');
  body.innerHTML = marked.parse(l.md);
  body.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
  body.querySelectorAll('a[href^="http"]').forEach(a => { a.target = '_blank'; a.rel = 'noopener'; });

  $('#lesson-refs').innerHTML = l.references.length ? `<h3>References</h3><ul>${l.references.map(r =>
    `<li><a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.title)}</a></li>`).join('')}</ul>` : '';

  // compare-with options: previous lesson (default), the empty file, any other lesson
  const sel = $('#compare-select');
  const opts = [`<option value="">${parent ? 'previous: ' + esc(lessonShort(parent)) : 'nothing (empty file)'}</option>`];
  if (parent) opts.push('<option value="__empty__">nothing (empty file)</option>');
  for (const p of S.course.parts) {
    opts.push(`<optgroup label="${esc(p.title)}">` + S.course.lessons.filter(o => o.part === p.id && o.id !== l.id)
      .map(o => `<option value="${o.id}">${esc(lessonShort(o))}</option>`).join('') + '</optgroup>');
  }
  sel.innerHTML = opts.join('');
  sel.value = S.compare;
  renderLessonDiff();

  // pager: course order
  const i = S.order.indexOf(l.id);
  const prev = S.byId[S.order[i - 1]], next = S.byId[S.order[i + 1]];
  $('#pager').innerHTML =
    (prev ? `<a class="prev" href="#/lesson/${prev.id}"><small>← previous</small>${esc(lessonShort(prev))}</a>` : '<span></span>') +
    (next ? `<a class="next" href="#/lesson/${next.id}"><small>next →</small>${esc(lessonShort(next))}</a>` : '');
}

function compareBase() {
  const l = S.byId[S.lessonId];
  if (S.compare === '__empty__') return '';
  if (S.compare) return S.byId[S.compare].code;
  return l.parent ? S.byId[l.parent].code : '';
}

function renderLessonDiff() {
  renderDiff($('#diff'), compareBase(), S.byId[S.lessonId].code, S.diffMode, $('#diff-stats'));
}

// ---------------------------------------------------------------- diff rendering
const hlCache = new Map();
function highlightLines(code) {
  if (hlCache.has(code)) return hlCache.get(code);
  const html = hljs.highlight(code, { language: 'python', ignoreIllegals: true }).value;
  // split into lines, closing and re-opening spans that cross a newline (e.g. docstrings)
  const lines = []; const stack = []; let cur = '';
  const re = /(<span[^>]*>)|(<\/span>)|(\n)|([^<\n]+)/g; let m;
  while ((m = re.exec(html))) {
    if (m[1]) { stack.push(m[1]); cur += m[1]; }
    else if (m[2]) { stack.pop(); cur += m[2]; }
    else if (m[3]) { lines.push(cur + '</span>'.repeat(stack.length)); cur = stack.join(''); }
    else cur += m[4];
  }
  lines.push(cur);
  if (hlCache.size > 200) hlCache.clear();
  hlCache.set(code, lines);
  return lines;
}

function diffRows(oldCode, newCode) {
  const oldHl = highlightLines(oldCode), newHl = highlightLines(newCode);
  const rows = []; let o = 0, n = 0;
  for (const part of Diff.diffLines(oldCode, newCode)) {
    const count = part.value.endsWith('\n') ? part.value.split('\n').length - 1 : part.value.split('\n').length;
    for (let k = 0; k < count; k++) {
      if (part.added) rows.push({ type: 'add', newNo: ++n, html: newHl[n - 1] });
      else if (part.removed) rows.push({ type: 'del', oldNo: ++o, html: oldHl[o - 1] });
      else rows.push({ type: 'ctx', oldNo: ++o, newNo: ++n, html: newHl[n - 1] });
    }
  }
  return rows;
}

function rowHtml(r) {
  const mark = r.type === 'add' ? '+' : r.type === 'del' ? '−' : ' ';
  return `<tr class="${r.type}"><td class="ln">${r.oldNo || ''}</td><td class="ln">${r.newNo || ''}</td><td class="mk">${mark}</td><td>${r.html || ' '}</td></tr>`;
}

function renderDiff(el, oldCode, newCode, mode, statsEl) {
  const rows = diffRows(oldCode, newCode);
  const adds = rows.filter(r => r.type === 'add').length, dels = rows.filter(r => r.type === 'del').length;
  if (statsEl) statsEl.innerHTML = `<span class="plus">+${adds}</span> <span class="minus">−${dels}</span>`;
  if (!rows.length) { el.innerHTML = '<div class="empty">An empty file.</div>'; return; }
  if (mode === 'changes' && !adds && !dels) { el.innerHTML = '<div class="empty">No changes: the code is identical.</div>'; return; }
  const CTX = 3;
  const out = [];
  let i = 0;
  while (i < rows.length) {
    if (mode !== 'changes' || rows[i].type !== 'ctx') { out.push(rowHtml(rows[i++])); continue; }
    let j = i;
    while (j < rows.length && rows[j].type === 'ctx') j++;
    const run = rows.slice(i, j);
    const keepHead = i === 0 ? 0 : CTX, keepTail = j === rows.length ? 0 : CTX;
    if (run.length > keepHead + keepTail + 2) {
      const hidden = run.slice(keepHead, run.length - keepTail);
      out.push(...run.slice(0, keepHead).map(rowHtml));
      out.push(`</tbody><tbody class="fold-body"><tr class="fold" tabindex="0" role="button"><td colspan="4">⋯ ${hidden.length} unchanged lines (show)</td></tr></tbody>` +
        `<tbody hidden>${hidden.map(rowHtml).join('')}</tbody><tbody>`);
      out.push(...run.slice(run.length - keepTail).map(rowHtml));
    } else {
      out.push(...run.map(rowHtml));
    }
    i = j;
  }
  el.innerHTML = `<table><tbody>${out.join('')}</tbody></table>`;
  el.querySelectorAll('tr.fold').forEach(tr => {
    const open = () => { const body = tr.parentElement; body.nextElementSibling.hidden = false; body.remove(); };
    tr.addEventListener('click', open);
    tr.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
  });
}

// ---------------------------------------------------------------- playground
function draftKey(id) { return 'dl-draft:' + id; }

function ensureEditor() {
  if (S.editor) return;
  S.editor = CodeMirror.fromTextArea($('#pg-editor'), {
    mode: 'python', lineNumbers: true, indentUnit: 4, tabSize: 4, indentWithTabs: false, lineWrapping: false,
    extraKeys: {
      Tab: cm => cm.somethingSelected() ? cm.indentSelection('add') : cm.replaceSelection('    '),
      'Shift-Tab': cm => cm.indentSelection('subtract'),
      'Ctrl-Enter': () => toggleRun(), 'Cmd-Enter': () => toggleRun(),
    },
  });
  let timer;
  S.editor.on('change', () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      const code = S.editor.getValue();
      const lesson = S.byId[S.lessonId];
      if (code === lesson.code) store.del(draftKey(lesson.id)); else store.set(draftKey(lesson.id), code);
      updatePgStats();
      const knobs = Params.parse(code);
      if (JSON.stringify(knobs) !== JSON.stringify(S.pgKnobs)) { S.pgKnobs = knobs; renderKnobs(); }
    }, 300);
  });
}

function renderPlayground() {
  ensureEditor();
  const l = S.byId[S.lessonId];
  document.title = `Playground · ${l.title} · diff-learning`;
  $('#pg-base').innerHTML = S.course.parts.map(p => `<optgroup label="${esc(p.title)}">` +
    S.course.lessons.filter(o => o.part === p.id).map(o => `<option value="${o.id}">${esc(lessonShort(o))}</option>`).join('') + '</optgroup>').join('');
  $('#pg-base').value = l.id;
  const code = store.get(draftKey(l.id), l.code);
  if (S.editor.getValue() !== code || S.editorLesson !== l.id) S.editor.setValue(code);
  S.editorLesson = l.id;
  S.pgKnobs = Params.parse(code);
  setPgMode(S.pgMode);
  setTimeout(() => S.editor.refresh(), 0);
}

function setPgMode(mode) {
  S.pgMode = mode;
  document.querySelectorAll('[data-pg]').forEach(b => b.classList.toggle('active', b.dataset.pg === mode));
  $('#pg-editor-wrap').hidden = mode !== 'edit';
  $('#pg-diff').hidden = mode !== 'diff';
  if (mode === 'diff') renderDiff($('#pg-diff'), S.byId[S.lessonId].code, S.editor.getValue(), 'changes', null);
  else S.editor.refresh();
  updatePgStats();
}

function updatePgStats() {
  const rows = diffRows(S.byId[S.lessonId].code, S.editor.getValue());
  const adds = rows.filter(r => r.type === 'add').length, dels = rows.filter(r => r.type === 'del').length;
  $('#pg-stats').innerHTML = adds || dels ? `your changes: <span class="plus">+${adds}</span> <span class="minus">−${dels}</span>` : 'unchanged';
}

// ---------------------------------------------------------------- lab: knobs
function currentKnobs() { return S.view === 'playground' ? S.pgKnobs : S.byId[S.lessonId].params; }
function knobValues() {
  const key = sourceKey();
  if (!S.knobValues[key]) S.knobValues[key] = {};
  return S.knobValues[key];
}
function overrides() {
  const vals = knobValues();
  const out = {};
  for (const k of currentKnobs()) if (k.name in vals && vals[k.name] !== k.value) out[k.name] = vals[k.name];
  return out;
}

function renderLab() {
  const l = S.byId[S.lessonId];
  $('#lab-source').textContent = S.view === 'playground' ? `running your playground code (from ${lessonNum(l)})` : `running lesson ${lessonShort(l)}`;
  renderKnobs();
  const exps = S.view === 'lesson' ? l.experiments : [];
  $('#experiments').innerHTML = exps.map((e, i) => `<button class="exp-btn" data-exp="${i}" title="${esc(e.description || '')}">▶ ${esc(e.name)}</button>`).join('');
  renderRuns();
}

function fmt(v, k) {
  if (k.type === 'int') return String(Math.round(v));
  if (typeof v !== 'number') return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1);
  return String(+v.toPrecision(3));
}

function renderKnobs() {
  const knobs = currentKnobs();
  const vals = knobValues();
  const box = $('#knobs');
  if (!knobs.length) {
    box.innerHTML = `<div class="none">No knobs in this ${S.view === 'playground' ? 'code' : 'lesson'}. ${S.view === 'playground' ? 'Tag an assignment with <code># @param</code> to add one.' : 'Just press Run.'}</div>`;
    return;
  }
  box.innerHTML = knobs.map(k => {
    const v = k.name in vals ? vals[k.name] : k.value;
    const changed = v !== k.value ? ' changed' : '';
    let control;
    if (k.type === 'bool') {
      control = `<input type="checkbox" data-knob="${k.name}" ${v ? 'checked' : ''}><span></span>`;
    } else if (k.options) {
      control = `<select data-knob="${k.name}">${k.options.map(o => `<option value="${esc(o)}" ${o === v ? 'selected' : ''}>${esc(o)}</option>`).join('')}</select><span></span>`;
    } else if (k.type === 'str') {
      control = `<input class="val" style="width:100%" data-knob="${k.name}" value="${esc(v)}"><span></span>`;
    } else if (k.min !== undefined && k.max !== undefined) {
      control = `<input type="range" min="0" max="1000" step="1" value="${toSlider(k, v)}" data-knob="${k.name}" data-slider aria-label="${k.name}">` +
        `<input class="val" data-knob="${k.name}" value="${fmt(v, k)}" aria-label="${k.name} value">`;
    } else {
      control = `<span></span><input class="val" data-knob="${k.name}" value="${fmt(v, k)}">`;
    }
    return `<div class="knob${changed}"><label title="${esc(k.name)}">${esc(k.name)}</label>${control}${k.help ? `<div class="help">${esc(k.help)}</div>` : ''}</div>`;
  }).join('');
}

function toSlider(k, v) {
  const t = k.log ? (Math.log(Math.max(v, k.min)) - Math.log(k.min)) / (Math.log(k.max) - Math.log(k.min)) : (v - k.min) / (k.max - k.min);
  return Math.round(Math.min(1, Math.max(0, t)) * 1000);
}
function fromSlider(k, pos) {
  const t = pos / 1000;
  let v = k.log ? Math.exp(Math.log(k.min) + t * (Math.log(k.max) - Math.log(k.min))) : k.min + t * (k.max - k.min);
  if (k.type === 'int') return Math.round(v);
  if (k.step) return +(Math.round(v / k.step) * k.step).toFixed(10);
  return +v.toPrecision(2);
}

function onKnobInput(e) {
  const el = e.target;
  const name = el.dataset.knob;
  if (!name) return;
  const k = currentKnobs().find(x => x.name === name);
  let v;
  if (k.type === 'bool') v = el.checked;
  else if (k.options) v = k.options.find(o => String(o) === el.value);
  else if (k.type === 'str') v = el.value;
  else if (el.dataset.slider !== undefined) v = fromSlider(k, +el.value);
  else { v = Number(el.value); if (!isFinite(v)) return; if (k.type === 'int') v = Math.round(v); }
  knobValues()[name] = v;
  const row = el.closest('.knob');
  row.classList.toggle('changed', v !== k.value);
  if (el.dataset.slider !== undefined) row.querySelector('input.val').value = fmt(v, k);
  else if (e.type === 'change' && row.querySelector('[data-slider]')) row.querySelector('[data-slider]').value = toSlider(k, v);
}

// ---------------------------------------------------------------- lab: running code
function ensureWorker() {
  if (S.worker) return S.worker;
  S.worker = new Worker('worker.js');
  S.workerReady = false;
  S.worker.onmessage = onWorkerMessage;
  S.worker.onerror = (e) => { setStatus('Python worker failed: ' + (e.message || 'see console'), true); finishRun({ error: e.message || 'worker error' }); };
  return S.worker;
}

function runLabel() {
  const l = S.byId[S.lessonId];
  const ov = overrides();
  const knobs = Object.entries(ov).map(([k, v]) => `${k}=${typeof v === 'number' ? fmt(v, { type: Number.isInteger(v) ? 'int' : 'float' }) : v}`).join(', ');
  return `${S.view === 'playground' ? 'playground ' : ''}${lessonNum(l)} ${l.title}${knobs ? ' · ' + knobs : ''}`;
}

function toggleRun() {
  if (S.running) { stopRun(); return; }
  const l = S.byId[S.lessonId];
  const base = S.view === 'playground' ? S.editor.getValue() : l.code;
  const code = Params.apply(base, overrides(), currentKnobs());

  // keep at most MAX_RUNS; drop the oldest finished one
  if (S.runs.length >= MAX_RUNS) S.runs.splice(S.runs.findIndex(r => r !== S.running), 1);
  const used = new Set(S.runs.map(r => r.slot));
  let slot = 1; while (used.has(slot)) slot++;
  const run = { id: S.nextRunId++, slot, label: runLabel(), lessonId: l.id, points: {}, total: 0, visible: true, status: 'running', t0: performance.now() };
  S.runs.push(run);
  S.running = run;
  consoleClear();
  consoleAppend(`▶ ${run.label}\n`, 'sys');
  setRunning(true);
  setStatus(S.workerReady ? 'Running…' : 'Starting Python (first run downloads ~10 MB)…');
  ensureWorker().postMessage({ type: 'run', id: run.id, code });
  renderRuns();
}

function stopRun() {
  if (!S.running) return;
  S.worker.terminate(); // the only way to interrupt Python here; a new worker starts on the next run
  S.worker = null;
  S.workerReady = false;
  consoleAppend('\n■ stopped\n', 'sys');
  finishRun({ stopped: true });
  setStatus('Stopped. (Python reloads on the next run.)');
}

function finishRun({ error, stopped, seconds }) {
  const run = S.running;
  if (!run) return;
  run.status = error ? 'error' : stopped ? 'stopped' : 'done';
  S.running = null;
  setRunning(false);
  if (!error && !stopped) {
    markDone(run.lessonId);
    setStatus(`Done in ${seconds.toFixed(1)}s`);
    $('#progress-bar').style.width = '100%';
  }
  renderRuns();
  scheduleChart();
}

function onWorkerMessage(e) {
  const msg = e.data;
  if (msg.type === 'status') { if (S.running) setStatus(msg.text); return; }
  if (msg.type === 'ready') { S.workerReady = true; return; }
  const run = S.running;
  if (!run || msg.id !== run.id) return;
  if (msg.type === 'out') {
    consoleAppend(msg.text + '\n', msg.stream === 'stderr' ? 'err' : null);
    if (msg.stream === 'stdout') ingestMetrics(run, msg.text);
  } else if (msg.type === 'done') {
    if (msg.error) {
      consoleAppend(trimTraceback(msg.error) + '\n', 'err');
      setStatus('Error: ' + lastLine(msg.error), true);
    }
    finishRun({ error: msg.error, seconds: msg.seconds });
  }
}

function trimTraceback(err) {
  const lines = err.trim().split('\n');
  const i = lines.findIndex(l => l.includes('File "code.py"'));
  return i >= 0 ? ['Traceback (most recent call last):', ...lines.slice(i)].join('\n') : err;
}
function lastLine(s) { const l = s.trim().split('\n'); return l[l.length - 1]; }

// "step 12 / 500 | loss 2.3456 | lr 0.01" -> {step: 12, loss: 2.3456, lr: 0.01}
function parseMetrics(line) {
  if (!line.includes('|')) return null;
  const out = {};
  for (const seg of line.split('|')) {
    const m = seg.trim().match(/^([A-Za-z_]\w*)\s*[:=]?\s*(-?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?|nan|inf)/i);
    if (m) out[m[1]] = parseFloat(m[2].toLowerCase() === 'inf' ? 'Infinity' : m[2]);
  }
  if (!('step' in out)) return null;
  const total = line.match(/step\s+\d+\s*\/\s*(\d+)/);
  return { metrics: out, total: total ? +total[1] : 0 };
}

function ingestMetrics(run, text) {
  let changed = false;
  for (const line of text.split('\n')) {
    const p = parseMetrics(line);
    if (!p) continue;
    const { step, ...rest } = p.metrics;
    for (const [k, v] of Object.entries(rest)) (run.points[k] ||= []).push([step, v]);
    if (p.total) {
      run.total = p.total;
      $('#progress-bar').style.width = `${Math.min(100, 100 * step / p.total)}%`;
      const secs = (performance.now() - run.t0) / 1000;
      setStatus(`Running · step ${step} / ${p.total} · ${secs.toFixed(0)}s`);
    }
    changed = true;
  }
  if (changed) scheduleChart();
}

function setRunning(on) {
  const btn = $('#run-btn');
  btn.classList.toggle('stop', on);
  btn.querySelector('span').textContent = on ? 'Stop' : 'Run';
  btn.querySelector('svg').innerHTML = on ? '<rect x="6" y="6" width="12" height="12" rx="1" fill="currentColor"/>' : '<path d="M7 4v16l13-8z" fill="currentColor"/>';
  if (on) $('#progress-bar').style.width = '0';
}
function setStatus(text, isErr) { const s = $('#status'); s.textContent = text; s.title = text; s.classList.toggle('err', !!isErr); }

// ---------------------------------------------------------------- console
let consoleBuf = [], consoleTimer = null;
function consoleClear() { consoleBuf = []; $('#console').textContent = ''; }
function consoleAppend(text, cls) {
  consoleBuf.push([text, cls]);
  if (!consoleTimer) consoleTimer = requestAnimationFrame(flushConsole);
}
function flushConsole() {
  consoleTimer = null;
  const el = $('#console');
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  const frag = document.createDocumentFragment();
  for (const [text, cls] of consoleBuf) {
    if (cls) { const s = document.createElement('span'); s.className = cls; s.textContent = text; frag.appendChild(s); }
    else frag.appendChild(document.createTextNode(text));
  }
  consoleBuf = [];
  el.appendChild(frag);
  while (el.childNodes.length > 4000) el.removeChild(el.firstChild);
  if (atBottom) el.scrollTop = el.scrollHeight;
}

// ---------------------------------------------------------------- chart
let chart = null, chartTimer = null, chartLast = 0;
function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function withAlpha(hex, a) { const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; }

function smoothed(points, a) {
  let s = 0, w = 0; // exponential moving average, debiased (like TensorBoard)
  return points.map(([x, y]) => { s = a * s + (1 - a) * y; w = a * w + (1 - a); return [x, s / w]; });
}
function thin(points, max = 1500) {
  if (points.length <= max) return points;
  const k = Math.ceil(points.length / max);
  return points.filter((_, i) => i % k === 0 || i === points.length - 1);
}

function scheduleChart() {
  if (chartTimer) return;
  const wait = Math.max(0, 200 - (performance.now() - chartLast));
  chartTimer = setTimeout(() => requestAnimationFrame(() => { chartTimer = null; chartLast = performance.now(); drawChart(); }), wait);
}

function metricsAvailable() {
  const set = new Set();
  S.runs.forEach(r => Object.keys(r.points).forEach(k => set.add(k)));
  return [...set];
}

function drawChart() {
  const metrics = metricsAvailable();
  const sel = $('#metric-select');
  if (!metrics.includes(S.metric) && metrics.length) S.metric = metrics.includes('loss') ? 'loss' : metrics[0];
  const optsHtml = metrics.map(m => `<option ${m === S.metric ? 'selected' : ''}>${esc(m)}</option>`).join('');
  if (sel.innerHTML !== optsHtml) sel.innerHTML = optsHtml;
  const hasData = S.runs.some(r => r.points[S.metric]?.length);
  $('#chart-empty').hidden = hasData;

  const text2 = cssVar('--text-3'), grid = cssVar('--grid'), surface = cssVar('--surface');
  const datasets = [];
  for (const r of S.runs) {
    const pts = r.points[S.metric];
    if (!r.visible || !pts || !pts.length) continue;
    const color = cssVar(`--series-${r.slot}`);
    const raw = thin(pts);
    if (S.smooth > 0 && pts.length > 5) {
      datasets.push({ label: r.label + ' (raw)', data: raw.map(([x, y]) => ({ x, y })), borderColor: withAlpha(color, 0.22), borderWidth: 1, pointRadius: 0, raw: true });
      datasets.push({ label: r.label, data: thin(smoothed(pts, S.smooth)).map(([x, y]) => ({ x, y })), borderColor: color, borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, pointHoverBorderColor: surface, pointHoverBorderWidth: 2, pointHoverBackgroundColor: color });
    } else {
      datasets.push({ label: r.label, data: raw.map(([x, y]) => ({ x, y })), borderColor: color, borderWidth: 2, pointRadius: pts.length < 40 ? 2 : 0, pointHoverRadius: 4, pointHoverBackgroundColor: color, pointBackgroundColor: color });
    }
  }
  const yType = S.logY ? 'logarithmic' : 'linear';
  if (!chart) {
    chart = new Chart($('#chart'), {
      type: 'line',
      data: { datasets },
      options: {
        animation: false, parsing: false, normalized: true, maintainAspectRatio: false,
        interaction: { mode: 'nearest', axis: 'x', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            filter: item => !item.dataset.raw,
            callbacks: {
              title: items => items.length ? `step ${items[0].parsed.x}` : '',
              label: item => ` ${item.dataset.label}: ${fmtY(item.parsed.y)}`,
            },
          },
        },
        scales: {
          x: { type: 'linear', title: { display: true, text: 'step', color: text2 }, ticks: { color: text2, maxTicksLimit: 6 }, grid: { color: grid } },
          y: { type: yType, ticks: { color: text2, maxTicksLimit: 6, callback: v => fmtY(v) }, grid: { color: grid } },
        },
      },
    });
  } else {
    chart.data.datasets = datasets;
    chart.options.scales.y.type = yType;
    Object.assign(chart.options.scales.x.ticks, { color: text2 }); chart.options.scales.x.grid.color = grid;
    chart.options.scales.x.title.color = text2;
    Object.assign(chart.options.scales.y.ticks, { color: text2 }); chart.options.scales.y.grid.color = grid;
    chart.update('none');
  }
  if (S.showTable) renderTable();
  updateRunFinals();
}
function fmtY(v) { const a = Math.abs(v); return a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(1) : String(+v.toPrecision(4)); }

function finalValue(r) {
  const pts = r.points[S.metric];
  if (!pts || !pts.length) return null;
  const sm = S.smooth > 0 && pts.length > 5 ? smoothed(pts, S.smooth) : pts;
  return sm[sm.length - 1][1];
}

function renderRuns() {
  $('#runs').innerHTML = S.runs.map(r => `
    <li class="${r.visible ? '' : 'hidden-run'}" data-run="${r.id}">
      <span class="swatch" style="background:var(--series-${r.slot})"></span>
      <span class="label" title="${esc(r.label)} (click to show/hide)">${esc(r.label)}</span>
      <span class="final" data-final="${r.id}"></span>
      <span class="st">${r.status === 'running' ? '…' : r.status === 'error' ? '⚠' : r.status === 'stopped' ? '■' : ''}</span>
      <button class="x" data-del="${r.id}" aria-label="remove run">×</button>
    </li>`).join('');
  updateRunFinals();
}
function updateRunFinals() {
  for (const r of S.runs) {
    const el = document.querySelector(`[data-final="${r.id}"]`);
    if (el) { const v = finalValue(r); el.textContent = v === null ? '' : fmtY(v); el.title = `last ${S.smooth > 0 ? 'smoothed ' : ''}${S.metric}`; }
  }
}
function renderTable() {
  const rows = S.runs.filter(r => r.points[S.metric]?.length).map(r => {
    const pts = r.points[S.metric];
    const min = Math.min(...pts.map(p => p[1]));
    return `<tr><td>${esc(r.label)}</td><td>${pts[pts.length - 1][0]}</td><td>${fmtY(pts[pts.length - 1][1])}</td><td>${fmtY(finalValue(r))}</td><td>${fmtY(min)}</td></tr>`;
  });
  $('#run-table').innerHTML = rows.length ? `<table><thead><tr><th>run</th><th>steps</th><th>last</th><th>last (smoothed)</th><th>min</th></tr></thead><tbody>${rows.join('')}</tbody></table>` : '<p class="hint">No data yet.</p>';
}

// ---------------------------------------------------------------- misc actions
function download(filename, text) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/x-python' }));
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function toggleTheme() {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('dl-theme', root.dataset.theme); } catch (e) { /* ignore */ }
  drawChart();
}

// value from a [data-pref] control, typed like the default (numbers stay numbers)
function prefValue(key, raw) {
  if (typeof raw === 'boolean') return raw;
  const def = DEFAULT_PREFS[key];
  return (typeof def === 'number' || def === null) && raw !== '' && isFinite(raw) ? Number(raw) : raw;
}

function bindPrefControls() {
  document.addEventListener('click', e => {
    const b = e.target.closest('button[data-pref]');
    if (b) setPref(b.dataset.pref, prefValue(b.dataset.pref, b.dataset.value));
  });
  document.addEventListener('change', e => {
    const el = e.target.closest('input[data-pref], select[data-pref]');
    if (el) setPref(el.dataset.pref, prefValue(el.dataset.pref, el.type === 'checkbox' ? el.checked : el.value));
  });
}

function bindUi() {
  bindPrefControls();
  $('#theme-toggle').addEventListener('click', toggleTheme);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => drawChart());
  $('#nav-toggle').addEventListener('click', toggleNav);
  $('#lab-toggle').addEventListener('click', toggleLab);
  matchMedia('(max-width: 760px)').addEventListener('change', () => { document.body.classList.remove('nav-open'); syncPaneToggles(); });

  $('#compare-select').addEventListener('change', e => { S.compare = e.target.value; renderLessonDiff(); });
  document.querySelectorAll('[data-mode]').forEach(b => b.addEventListener('click', () => {
    S.diffMode = b.dataset.mode;
    document.querySelectorAll('[data-mode]').forEach(x => x.classList.toggle('active', x === b));
    renderLessonDiff();
  }));
  $('#copy-code').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(S.byId[S.lessonId].code); $('#copy-code').textContent = 'Copied'; }
    catch (e) { $('#copy-code').textContent = 'Copy failed'; }
    setTimeout(() => { $('#copy-code').textContent = 'Copy'; }, 1500);
  });
  $('#download-code').addEventListener('click', () => download(`${S.lessonId}.py`, S.byId[S.lessonId].code));
  $('#open-playground').addEventListener('click', () => { location.hash = '#/playground/' + S.lessonId; });

  $('#pg-base').addEventListener('change', e => { location.hash = '#/playground/' + e.target.value; });
  $('#pg-reset').addEventListener('click', () => {
    if (!confirm('Discard your edits and restore the lesson code?')) return;
    store.del(draftKey(S.lessonId));
    S.editor.setValue(S.byId[S.lessonId].code);
    setPgMode('edit');
  });
  document.querySelectorAll('[data-pg]').forEach(b => b.addEventListener('click', () => setPgMode(b.dataset.pg)));

  const knobs = $('#knobs');
  knobs.addEventListener('input', onKnobInput);
  knobs.addEventListener('change', onKnobInput);
  $('#reset-knobs').addEventListener('click', () => { S.knobValues[sourceKey()] = {}; renderKnobs(); });
  $('#experiments').addEventListener('click', e => {
    const b = e.target.closest('[data-exp]');
    if (!b || S.running) return;
    const ex = S.byId[S.lessonId].experiments[+b.dataset.exp];
    S.knobValues[sourceKey()] = { ...(ex.params || {}) };
    renderKnobs();
    toggleRun();
  });
  $('#run-btn').addEventListener('click', toggleRun);
  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !e.target.closest('.CodeMirror')) { e.preventDefault(); toggleRun(); return; }
    if (e.ctrlKey || e.metaKey || e.altKey || isTyping(e)) return;
    const shortcut = { '[': toggleNav, ']': toggleLab }[e.key];
    if (shortcut) { e.preventDefault(); shortcut(); }
  });

  $('#metric-select').addEventListener('change', e => { S.metric = e.target.value; drawChart(); });
  $('#smooth').addEventListener('input', e => { S.smooth = +e.target.value; setPref('smooth', S.smooth); drawChart(); });
  $('#logy').addEventListener('change', e => { S.logY = e.target.checked; drawChart(); });
  $('#table-toggle').addEventListener('click', () => {
    S.showTable = !S.showTable;
    $('#table-toggle').setAttribute('aria-pressed', String(S.showTable));
    $('#run-table').hidden = !S.showTable;
    drawChart();
  });
  $('#clear-runs').addEventListener('click', () => { S.runs = S.runs.filter(r => r === S.running); renderRuns(); drawChart(); });
  $('#runs').addEventListener('click', e => {
    const del = e.target.closest('[data-del]');
    if (del) {
      const id = +del.dataset.del;
      if (S.running && S.running.id === id) stopRun();
      S.runs = S.runs.filter(r => r.id !== id);
      renderRuns(); drawChart();
      return;
    }
    const li = e.target.closest('[data-run]');
    if (li && e.target.classList.contains('label')) {
      const r = S.runs.find(x => x.id === +li.dataset.run);
      r.visible = !r.visible;
      renderRuns(); drawChart();
    }
  });
  $('#clear-console').addEventListener('click', consoleClear);
}

boot();
