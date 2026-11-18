// The seat stream: seats → passengers → local passengers → hotel check-ins → guest nights.
// Shape is exact: every slice is to scale with the others (the ratios are the model's). Size shows magnitude: in the
// live lever the whole stream grows with the change (square-root scale, so a small change is still readable), which
// never alters the proportions. Slices that leave turn down as exits (empty seats, passengers connecting onwards, local
// passengers with no extra hotel stay); what is left checks in and widens by the nights each check-in stays.
// Built once, then updated in place: SVG for the bands, a canvas for the dots, HTML for the exit notes.
const d3 = window.d3;
const N_DOTS = 700;

function cubic(p0, p1, p2, p3, t) {
  const u = 1 - t;
  return [u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0],
          u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1]];
}

function rand(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}

const WORDS = [
  { of: "of seats", gain: "fly empty", loss: "were empty", n: t => `${t} seats` },
  { of: "of passengers", gain: "connect onwards", loss: "were connecting onwards", n: t => `${t}, transfer and transit` },
  { of: "of inbound passengers", gain: "add no hotel stay", loss: "were not hotel guests", n: t => `${t}: residents, family visits, other lodging` },
];

export class Stream {
  constructor(root) {
    this.root = root;
    this.svg = d3.select(root).append("svg").attr("class", "stream-svg").attr("role", "img")
      .attr("aria-label", "From seats to guest nights");
    this.canvas = root.appendChild(Object.assign(document.createElement("canvas"), { className: "stream-dots" }));
    this.notes = root.appendChild(Object.assign(document.createElement("div"), { className: "stream-notes" }));
    this.msg = root.appendChild(Object.assign(document.createElement("p"), { className: "stream-msg" }));
    const defs = this.svg.append("defs");
    const fade = defs.append("linearGradient").attr("id", "s-fade").attr("x1", 0).attr("x2", 0).attr("y1", 0).attr("y2", 1);
    fade.append("stop").attr("offset", "0%").attr("stop-color", "#A9B8C0").attr("stop-opacity", 0.30);
    fade.append("stop").attr("offset", "100%").attr("stop-color", "#A9B8C0").attr("stop-opacity", 0.02);
    const hatch = defs.append("pattern").attr("id", "s-hatch").attr("patternUnits", "userSpaceOnUse")
      .attr("width", 7).attr("height", 7).attr("patternTransform", "rotate(45)");
    hatch.append("line").attr("x1", 0).attr("y1", 0).attr("x2", 0).attr("y2", 7).attr("stroke", "#EEF3F5").attr("stroke-opacity", 0.35).attr("stroke-width", 2);
    this.gExits = this.svg.append("g");
    this.band = this.svg.append("path");
    this.nights = this.svg.append("path");
    this.hatchPath = this.svg.append("path").attr("fill", "url(#s-hatch)");
    this.gGates = this.svg.append("g");
    this.gHeads = this.svg.append("g");
    const R = rand(20261005);
    this.particles = d3.range(N_DOTS).map(() => ({ r: R(), u: R(), v: R() }));
    this.H = 372;
  }

  render(chain, { scale = 1, outside = false } = {}) {
    const W = this.root.clientWidth, Hs = this.H;
    if (!W) return;
    this.svg.attr("viewBox", `0 0 ${W} ${Hs}`);
    const css = getComputedStyle(document.documentElement);
    const loss = chain.direction === "loss";
    const flow = css.getPropertyValue(loss ? "--loss-d" : "--gain-d").trim();
    const st = chain.stages, steps = chain.steps;
    const v = st.map(s => Math.abs(s.value));
    const ckf = Math.abs(st[3].from_flights ?? st[3].value);
    const ref = Math.max(v[0], v[1], v[2], 1e-9);
    const y0 = 108, Hmax = 150, yEnd = 250, padL = 34, padR = 40;
    const H = Hmax * Math.max(0, Math.min(1, scale));
    const gap = (W - padL - padR) / 4;
    const xs = d3.range(5).map(i => padL + i * gap);
    const hasFlow = chain.has_flight_change && v[0] > 0;
    const h = hasFlow ? [v[0], v[1], v[2], ckf].map(x => H * x / ref) : [0, 0, 0, 0];
    h.push(hasFlow && ckf > 0 ? H * v[4] / ref : 0);

    // ---- headings (four stages; the fifth, guest nights, is the outcome column)
    const heads = st.slice(0, 4).map((s, i) => ({ i, x: xs[i], label: s.label, value: s.text, ratio: s.ratio_text || "" }));
    const g = this.gHeads.selectAll("g.head").data(heads, d => d.i).join(en => {
      const t = en.append("g").attr("class", "head");
      t.append("text").attr("class", "s-label").attr("y", 30);
      t.append("text").attr("class", "s-value").attr("y", 63);
      t.append("text").attr("class", "s-ratio").attr("y", 84);
      return t;
    });
    g.attr("transform", d => `translate(${d.x},0)`).attr("opacity", d => (hasFlow || d.i === 3 ? 1 : 0.4));
    g.select(".s-label").text(d => d.label);
    g.select(".s-value").text(d => d.value);
    g.select(".s-ratio").text(d => d.ratio);

    // ---- no flight changes (demand or stay-length only, or nothing): say so instead of leaving the stream empty
    const flat = Math.abs(st[3].value) > 0.5 ? "No flights change in this scenario, so no seats flow through. The effect starts at the hotel: check-ins change for other reasons."
      : Math.abs(st[4].value) > 0.5 ? "No flights or check-ins change in this scenario: guests stay longer or shorter than usual."
      : "Nothing changes: this is the normal year.";
    this.msg.textContent = hasFlow ? "" : flat;
    this.msg.style.cssText = hasFlow ? "display:none" : `left:${padL}px;top:${y0 + 40}px;width:${Math.min(3 * gap, W - padL - padR)}px`;

    // ---- exits
    const exits = [];
    for (let k = 0; k < 3; k++) {
      const s = steps[k], w = h[k] - h[k + 1];
      if (!hasFlow || !s || s.kind !== "exit" || w < 0.6) continue;
      const yt = y0 + h[k + 1], yb = y0 + h[k];
      const xe = xs[k] + Math.max(gap * 0.5, w / 2 + 28);
      exits.push({ k, s, w, xe,
        outer: [[xs[k], yt], [xe + w / 2, yt], [xe + w / 2, yt + 0.25 * (yEnd - yt)], [xe + w / 2, yEnd]],
        inner: [[xs[k], yb], [xe - w / 2, yb], [xe - w / 2, yb + 0.25 * (yEnd - yb)], [xe - w / 2, yEnd]] });
    }
    this.gExits.selectAll("path").data(exits, d => d.k).join("path").attr("fill", "url(#s-fade)")
      .attr("d", e => { const o = e.outer, i = e.inner; return `M${o[0]}C${o[1]} ${o[2]} ${o[3]}L${i[3]}C${i[2]} ${i[1]} ${i[0]}Z`; });

    // ---- main band and nights
    const top = y0, n4 = h[4];
    let bandD = "";
    if (hasFlow) {
      const b = [`M${xs[0]},${top}`, `L${xs[3]},${top}`, `C${xs[3] + gap * 0.5},${top} ${xs[3] + gap * 0.5},${top} ${xs[4]},${top}`,
                 `L${xs[4]},${top + n4}`, `C${xs[3] + gap * 0.55},${top + n4} ${xs[3] + gap * 0.45},${top + h[3]} ${xs[3]},${top + h[3]}`];
      for (let k = 2; k >= 0; k--) b.push(`L${xs[k + 1]},${top + h[k + 1]}`, `L${xs[k]},${top + h[k + 1]}`, `L${xs[k]},${top + h[k]}`);
      b.push("Z"); bandD = b.join(" ");
    }
    const nightsD = hasFlow && ckf > 0 ? `M${xs[3]},${top} L${xs[4]},${top} L${xs[4]},${top + n4}
      C${xs[3] + gap * 0.55},${top + n4} ${xs[3] + gap * 0.45},${top + h[3]} ${xs[3]},${top + h[3]} Z` : "";
    this.band.attr("d", bandD).attr("fill", flow).attr("fill-opacity", 0.30);
    this.nights.attr("d", nightsD).attr("fill", flow).attr("fill-opacity", 0.42);
    this.hatchPath.attr("d", outside ? bandD : "");

    // ---- gates
    this.gGates.selectAll("line").data(d3.range(4)).join("line").attr("class", "gate")
      .attr("x1", k => xs[k]).attr("x2", k => xs[k]).attr("y1", y0 - 12).attr("y2", k => y0 + Math.max(h[k], 2) + 12);

    // ---- exit notes (HTML over the SVG, wrapped inside each exit's column)
    const notes = exits.map((e, i) => {
      const left = e.xe - e.w / 2, next = exits[i + 1] ? exits[i + 1].xe - exits[i + 1].w / 2 : xs[4] + 10;
      const wd = WORDS[e.k];
      return { left, width: Math.max(120, Math.min(next - left - 14, gap + 20)),
               html: `<b>${d3.format(".0%")(e.s.share ?? 0).replace("100%", "99.9%")}<small>${wd.of}</small></b>${loss ? wd.loss : wd.gain}<span>${wd.n(e.s.text)}</span>` };
    });
    while (this.notes.children.length < notes.length) this.notes.appendChild(Object.assign(document.createElement("div"), { className: "x-note" }));
    [...this.notes.children].forEach((el, i) => {
      const n = notes[i];
      el.style.display = n ? "" : "none";
      if (!n) return;
      el.style.left = `${n.left}px`; el.style.top = `${yEnd + 10}px`; el.style.width = `${n.width}px`;
      if (el._html !== n.html) { el.innerHTML = n.html; el._html = n.html; }
    });

    // ---- dots: a fixed set of particles, each with a stable place inside whichever region it falls in
    const regions = [];
    if (hasFlow) {
      for (let k = 0; k < 3; k++) regions.push({ kind: "band", area: (xs[k + 1] - xs[k]) * h[k + 1], k });
      for (const e of exits) regions.push({ kind: "exit", area: ((e.xe - xs[e.k]) + (yEnd - y0)) * e.w * 0.8, e });
      if (ckf > 0) regions.push({ kind: "night", area: gap * (h[3] + n4) / 2 * 1.3 });
    }
    const total = d3.sum(regions, r => r.area) || 1;
    let acc = 0; for (const r of regions) { r.lo = acc; acc += r.area / total; r.hi = acc; }
    const visible = Math.round(N_DOTS * Math.max(0.15, Math.min(1, scale)));
    const dpr = window.devicePixelRatio || 1, ctx = this.canvas.getContext("2d");
    if (this.canvas.width !== Math.round(W * dpr) || this.canvas.height !== Math.round(Hs * dpr)) {
      this.canvas.width = Math.round(W * dpr); this.canvas.height = Math.round(Hs * dpr);
      this.canvas.style.width = `${W}px`; this.canvas.style.height = `${Hs}px`;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, Hs);
    for (let n = 0; n < (regions.length ? visible : 0); n++) {
      const p = this.particles[n];
      const reg = regions.find(r => p.r >= r.lo && p.r < r.hi) || regions[regions.length - 1];
      let x, y, a = 0.8, col = "#EEF3F5", rad = 1.9;
      if (reg.kind === "band") { x = xs[reg.k] + p.u * (xs[reg.k + 1] - xs[reg.k]); y = top + p.v * h[reg.k + 1]; }
      else if (reg.kind === "night") {
        const t = p.u; x = xs[3] + t * gap; y = top + p.v * (h[3] + (n4 - h[3]) * (3 * t * t - 2 * t * t * t)); a = 0.95;
      } else {
        const t = Math.pow(p.u, 1.25), e = reg.e, i0 = cubic(...e.inner, t), o0 = cubic(...e.outer, t);
        x = i0[0] + (o0[0] - i0[0]) * p.v; y = i0[1] + (o0[1] - i0[1]) * p.v; a = 0.85 * (1 - 0.8 * t); col = "#A9B8C0"; rad = 1.6;
      }
      ctx.globalAlpha = a; ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, rad, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }
}

export function drawRange(svgEl, lo, mid, hi, loss) {
  const W = svgEl.getBoundingClientRect().width, H = 30;
  const svg = d3.select(svgEl).attr("viewBox", `0 0 ${W} ${H}`);
  svg.selectAll("*").remove();
  if (!(Math.abs(hi - lo) > 0)) return;
  const ext = loss ? [Math.min(lo, mid, hi) * 1.08, 0] : [0, Math.max(lo, mid, hi) * 1.08];
  const x = d3.scaleLinear().domain(ext).range([2, W - 2]);
  const col = getComputedStyle(document.documentElement).getPropertyValue(loss ? "--loss-d" : "--gain-d").trim();
  svg.append("line").attr("x1", 2).attr("x2", W - 2).attr("y1", 15).attr("y2", 15).attr("stroke", "#3A4A55").attr("stroke-width", 4).attr("stroke-linecap", "round");
  svg.append("line").attr("x1", x(lo)).attr("x2", x(hi)).attr("y1", 15).attr("y2", 15).attr("stroke", col).attr("stroke-width", 6).attr("stroke-linecap", "round");
  svg.append("circle").attr("cx", x(mid)).attr("cy", 15).attr("r", 7).attr("fill", "#EEF3F5").attr("stroke", "#1F2B33").attr("stroke-width", 2);
  svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 6).attr("y2", 24).attr("stroke", "#A9B8C0");
}
