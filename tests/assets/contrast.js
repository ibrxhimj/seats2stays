() => {
  const parse = c => { const m = c.match(/rgba?\(([^)]+)\)/); if (!m) return null; const p = m[1].split(/[ ,/]+/).filter(Boolean).map(Number); return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1]; };
  const lum = ([r, g, b]) => { const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }; return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
  const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };
  const bgOf = el => { const stack = []; for (let e = el; e && e.nodeType === 1; e = e.parentElement) { const c = parse(getComputedStyle(e).backgroundColor); if (c && c[3] > 0) { stack.push(c); if (c[3] >= 0.99) break; } } let base = [255, 255, 255]; for (const c of stack.reverse()) base = base.map((v, i) => v * (1 - c[3]) + c[i] * c[3]); return base; };
  const out = [], seen = new Set();
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const t = walker.currentNode; if (!t.textContent.trim()) continue;
    const el = t.parentElement; if (!el || seen.has(el)) continue; seen.add(el);
    const r = el.getBoundingClientRect(); if (!r.width || !r.height) continue;
    const cs = getComputedStyle(el); if (cs.visibility === "hidden" || +cs.opacity === 0) continue;
    let col, bg, size = parseFloat(cs.fontSize), weight = +cs.fontWeight;
    if (el instanceof SVGElement) { col = parse(cs.fill) || parse(cs.color); bg = bgOf(el.ownerSVGElement?.parentElement || el.parentElement); }
    else { col = parse(cs.color); bg = bgOf(el); }
    if (!col) continue;
    const a = col[3] * (+cs.opacity || 1); const fg = col.slice(0, 3).map((v, i) => v * a + bg[i] * (1 - a));
    const cr = ratio(fg, bg), large = size >= 24 || (size >= 18.66 && weight >= 600);
    out.push({ text: t.textContent.trim().slice(0, 40), cls: (el.id ? "#" + el.id : "") + "." + (el.getAttribute("class") || "") , tag: el.tagName, size, weight, cr: +cr.toFixed(2), large });
  }
  return out;
}
