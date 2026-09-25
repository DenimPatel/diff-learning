// Diff view helpers: pure functions (no DOM, no libraries) so node can test them.
//
//   splitLines(html)        highlight.js output -> one self-contained HTML string per line
//   pairRows(rows)          unified rows -> side-by-side pairs (removed lines next to what replaced them)
//   wordRanges(a, b)        the changed character ranges inside a pair of similar lines
//   markRanges(html, rs)    wrap those ranges in <mark> inside syntax-highlighted HTML

const DiffView = (() => {
  // Split highlighted HTML into lines, closing and re-opening spans that cross a newline (e.g. docstrings).
  function splitLines(html) {
    const lines = []; const stack = []; let cur = '';
    const re = /(<span[^>]*>)|(<\/span>)|(\n)|([^<\n]+)/g; let m;
    while ((m = re.exec(html))) {
      if (m[1]) { stack.push(m[1]); cur += m[1]; }
      else if (m[2]) { stack.pop(); cur += m[2]; }
      else if (m[3]) { lines.push(cur + '</span>'.repeat(stack.length)); cur = stack.join(''); }
      else cur += m[4];
    }
    lines.push(cur);
    return lines;
  }

  // rows: [{type: 'ctx'|'del'|'add', ...}] in unified order.
  // Each run of removed lines is zipped with the run of added lines right after it.
  function pairRows(rows) {
    const out = [];
    let i = 0;
    while (i < rows.length) {
      if (rows[i].type === 'ctx') { out.push({ type: 'ctx', left: rows[i], right: rows[i] }); i++; continue; }
      const dels = [], adds = [];
      while (i < rows.length && rows[i].type === 'del') dels.push(rows[i++]);
      while (i < rows.length && rows[i].type === 'add') adds.push(rows[i++]);
      for (let k = 0; k < Math.max(dels.length, adds.length); k++) {
        out.push({ type: 'change', left: dels[k] || null, right: adds[k] || null });
      }
    }
    return out;
  }

  // identifiers and numbers stay whole; each space run and punctuation mark is its own token
  const tokenize = (s) => s.match(/[\p{L}\p{N}_]+|\s+|[^\p{L}\p{N}_\s]/gu) || [];

  // Character ranges [start, end) that differ between two lines, on each side, via an LCS of tokens.
  // Returns null when the lines are too different for word highlights to help (a rewritten line).
  function wordRanges(a, b) {
    const A = tokenize(a), B = tokenize(b);
    if (!A.length || !B.length || A.length * B.length > 40000) return null;
    const n = A.length, m = B.length;
    const L = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
    for (let i = n - 1; i >= 0; i--) {
      for (let j = m - 1; j >= 0; j--) L[i][j] = A[i] === B[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
    }
    const keepA = new Array(n).fill(false), keepB = new Array(m).fill(false);
    let same = 0;
    for (let i = 0, j = 0; i < n && j < m;) {
      if (A[i] === B[j]) { keepA[i] = keepB[j] = true; if (A[i].trim()) same += A[i].length; i++; j++; }
      else if (L[i + 1][j] >= L[i][j + 1]) i++;
      else j++;
    }
    const visible = (s) => s.replace(/\s+/g, '').length;
    if (same < 0.4 * Math.min(visible(a), visible(b))) return null;
    const ranges = (toks, keep) => {
      const out = []; let pos = 0;
      toks.forEach((t, k) => {
        const end = pos + t.length;
        if (!keep[k] && t.trim()) {
          const last = out[out.length - 1];
          // join changes separated only by whitespace into one highlight
          if (last && !toks.slice(last.k + 1, k).some((x, q) => keep[last.k + 1 + q] && x.trim())) { last[1] = end; last.k = k; }
          else out.push(Object.assign([pos, end], { k }));
        }
        pos = end;
      });
      return out.map(([s, e]) => [s, e]);
    };
    return { old: ranges(A, keepA), new: ranges(B, keepB) };
  }

  // Wrap the character ranges of a line's text in <mark class="cls"> inside its highlighted HTML.
  // Offsets count decoded characters (an entity like &lt; is one). Marks stay innermost: they close
  // before every span tag and reopen after it, so syntax colors and nesting survive.
  function markRanges(html, ranges, cls = 'wd') {
    if (!ranges || !ranges.length) return html;
    const open = `<mark class="${cls}">`, close = '</mark>';
    let out = '', pos = 0, r = 0, inMark = false;
    const inRange = (p) => { while (r < ranges.length && p >= ranges[r][1]) r++; return r < ranges.length && p >= ranges[r][0]; };
    const emit = (text, len) => {
      const want = inRange(pos);
      if (want !== inMark) { out += want ? open : close; inMark = want; }
      out += text; pos += len;
    };
    const re = /(<[^>]*>)|(&(?:#\d+|#x[\da-f]+|\w+);)|([^<&]+)/gi; let m;
    while ((m = re.exec(html))) {
      if (m[1]) out += inMark ? close + m[1] + open : m[1];
      else if (m[2]) emit(m[2], 1);
      else for (const ch of m[3]) emit(ch, ch.length);
    }
    if (inMark) out += close;
    return out.split(open + close).join(''); // drop empty marks left around adjacent tags
  }

  return { splitLines, pairRows, wordRanges, markRanges, tokenize };
})();

if (typeof module !== 'undefined') module.exports = DiffView;
