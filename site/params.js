// Knobs: `# @param` annotations on plain Python assignments (Colab-style).
// JavaScript twin of tools/params.py; keep the two in sync.
//
//   learning_rate = 0.01  # @param {"min": 0.0001, "max": 0.1, "log": true} step size of the optimizer

const Params = (() => {
  const PARAM_RE = /^(\s*)([A-Za-z_]\w*)\s*=\s*(.+?)\s*#\s*@param\s*(\{[^}]*\})?\s*(.*)$/;

  function parseLiteral(text) {
    const t = text.trim();
    if (t === 'True') return { value: true, type: 'bool' };
    if (t === 'False') return { value: false, type: 'bool' };
    if (/^-?\d+$/.test(t)) return { value: parseInt(t, 10), type: 'int' };
    if (/^-?(\d+\.\d*|\.\d+|\d+)(e[-+]?\d+)?$/i.test(t)) return { value: parseFloat(t), type: 'float' };
    const s = t.match(/^(['"])(.*)\1$/);
    if (s) return { value: s[2], type: 'str' };
    return null;
  }

  function parse(code) {
    const knobs = [];
    code.split('\n').forEach((line, i) => {
      const m = line.match(PARAM_RE);
      if (!m) return;
      const lit = parseLiteral(m[3]);
      if (!lit) return;
      let opts = {};
      if (m[4]) { try { opts = JSON.parse(m[4]); } catch (e) { opts = {}; } }
      knobs.push({ name: m[2], line: i, value: lit.value, type: lit.type, help: m[5].trim(), ...opts });
    });
    return knobs;
  }

  function toLiteral(value, type) {
    if (type === 'bool') return value ? 'True' : 'False';
    if (type === 'int') return String(Math.round(Number(value)));
    if (type === 'float') {
      const s = String(Number(value));
      return /[.eE]/.test(s) || !isFinite(Number(value)) ? s : s + '.0';
    }
    return "'" + String(value).replace(/\\/g, '\\\\').replace(/'/g, "\\'") + "'";
  }

  // Rewrite the literal of each annotated assignment named in `overrides`.
  function apply(code, overrides, knobs) {
    const types = Object.fromEntries((knobs || parse(code)).map(k => [k.name, k.type]));
    return code.split('\n').map(line => {
      const m = line.match(PARAM_RE);
      if (!m || !(m[2] in overrides)) return line;
      const start = m[1].length + line.slice(m[1].length).indexOf(m[3], m[2].length);
      return line.slice(0, start) + toLiteral(overrides[m[2]], types[m[2]]) + line.slice(start + m[3].length);
    }).join('\n');
  }

  return { parse, apply, toLiteral };
})();

if (typeof module !== 'undefined') module.exports = Params;
