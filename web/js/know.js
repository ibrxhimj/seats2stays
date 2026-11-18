// "How we know" (Tab 3). Everything is the frozen back-test and Phase 3 evidence, served by /api/know; nothing is re-run here.
//   1. the headline and four numbers (rebuilt vs reported, against "same as last year", the lever check, the range check);
//   2. the back-test itself: reported guest nights against the rebuilt ones, with the 80% range, by country and period;
//      and how the error changes with months ahead (Step 5.7);
//   3. the country catch: flights count the country a plane left from, hotels the guest's passport;
//   4. March 2026: what flights alone would have done, by world region;
//   5. the Sobol check: all levers moved together, and which uncertainty dominates a result;
//   6. every assumption of the chain, and the limits.
const d3 = window.d3;
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const MINUS = "−";
const sig3 = x => { if (!x) return 0; const m = 10 ** (Math.floor(Math.log10(Math.abs(x))) - 2); return Math.round(x / m) * m; };
const n3 = x => (x < 0 ? MINUS : "") + Math.abs(Math.round(sig3(x))).toLocaleString("en-GB");
const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const mon = ym => `${MON[+ym.slice(5) - 1]} ${ym.slice(0, 4)}`;
const monS = ym => `${MON[+ym.slice(5) - 1]} ${ym.slice(2, 4)}`;
const times = v => `×${v.toFixed(2)}`;

export function createKnow({ root, call }) {
  let data = null, built = false;
  const view = { key: "ALL", fold: "A" };
  const $ = id => document.getElementById(id);

  // ------------------------------------------------------------------ 2. the back-test chart
  function drawBacktest() {
    const S = data.series.find(s => s.key === view.key), F = data.folds.find(f => f.id === view.fold);
    const rows = S.rows[view.fold], months = F.months, host = $("bt");
    const W = host.getBoundingClientRect().width; if (!W) return;
    const H = 330, m = { l: 74, r: 18, t: 14, b: 32 };
    const lo = d3.min(rows, r => Math.min(r[0], r[2])), hi = d3.max(rows, r => Math.max(r[0], r[3]));
    const y = d3.scaleLinear().domain([Math.max(0, lo * 0.85), hi * 1.06]).nice().range([H - m.b, m.t]);
    const x = d3.scalePoint().domain(months).range([m.l + 24, W - m.r - 24]);
    const svg = d3.select(host).select("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("height", H);
    svg.selectAll("*").remove();
    for (const t of y.ticks(5)) {
      svg.append("line").attr("class", "grid").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(t)).attr("y2", y(t));
      svg.append("text").attr("class", "tick").attr("x", m.l - 10).attr("y", y(t) + 4).attr("text-anchor", "end").text(d3.format(".3~s")(t));
    }
    months.forEach(mo => svg.append("text").attr("class", "tick").attr("x", x(mo)).attr("y", H - 8).attr("text-anchor", "middle").text(monS(mo)));
    const pts = (i) => rows.map((r, k) => [x(months[k]), y(r[i])]);
    svg.append("path").attr("class", "band").attr("d", d3.area().x((_, k) => x(months[k])).y0(r => y(r[2])).y1(r => y(r[3]))(rows));
    svg.append("path").attr("class", "rebuilt").attr("d", d3.line()(pts(1)));
    svg.append("path").attr("class", "actual").attr("d", d3.line()(pts(0)));
    rows.forEach((r, k) => {
      const inside = r[0] >= r[2] && r[0] <= r[3];
      svg.append("circle").attr("class", "dot rebuilt").attr("cx", x(months[k])).attr("cy", y(r[1])).attr("r", 4);
      svg.append("circle").attr("class", inside ? "dot actual" : "dot actual out").attr("cx", x(months[k])).attr("cy", y(r[0])).attr("r", 5.5);
    });
    const ins = rows.filter(r => r[0] >= r[2] && r[0] <= r[3]).length, e = S.error[data.folds.indexOf(F)];
    $("bt-read").innerHTML = `<b>${esc(S.label)}, ${esc(F.label)}:</b> the rebuilt month is off by <b>${e.toFixed(1)}%</b> on average (WMAPE), `
      + `and the reported month fell inside the 80% range in <b>${ins} of ${rows.length}</b> months. The model was trained to ${esc(F.trained_to)} and never saw these months.`
      + (S.range_note ? ` <span class="note">${esc(S.range_note)}</span>` : "");
    svg.attr("aria-label", `${S.label}, ${F.label}. Reported guest nights by month: ${rows.map((r, k) => `${mon(months[k])} ${n3(r[0])} against ${n3(r[1])} rebuilt`).join("; ")}.`);
    $("bt-table").innerHTML = `<thead><tr><th>Month</th><th>Reported</th><th>Rebuilt</th><th>80% range</th></tr></thead><tbody>${rows.map((r, k) =>
      `<tr><th>${mon(months[k])}</th><td>${n3(r[0])}</td><td>${n3(r[1])}</td><td>${n3(r[2])} to ${n3(r[3])}</td></tr>`).join("")}</tbody>`;
  }

  function buildBacktest() {
    const sel = $("bt-market");
    sel.innerHTML = data.series.map(s => `<option value="${s.key}">${esc(s.label)}${s.flights === false ? " (no flights in the data)" : ""}</option>`).join("");
    sel.addEventListener("change", () => { view.key = sel.value; drawBacktest(); });
    const seg = $("bt-fold");
    seg.innerHTML = data.folds.map(f => `<button type="button" role="radio" data-f="${f.id}" aria-checked="${f.id === view.fold}">${esc(f.label)}</button>`).join("");
    seg.addEventListener("click", e => {
      const b = e.target.closest("button[data-f]"); if (!b) return;
      view.fold = b.dataset.f; seg.querySelectorAll("button").forEach(x => x.setAttribute("aria-checked", String(x === b))); drawBacktest();
    });
  }

  // ------------------------------------------------------------------ 2b. error by months ahead
  function drawAhead() {
    const A = data.ahead, C = A.chart, host = $("ah"); if (!host) return;
    const W = host.getBoundingClientRect().width; if (!W) return;
    const H = 270, m = { l: 52, r: 18, t: 14, b: 34 };
    const all = ["A", "B"].flatMap(f => [...C.country[f], ...C.total[f]]);
    const y = d3.scaleLinear().domain([0, Math.ceil(d3.max(all) / 5) * 5]).range([H - m.b, m.t]);
    const x = d3.scalePoint().domain(C.leads).range([m.l + 20, W - m.r - 20]);
    const svg = d3.select(host).select("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("height", H);
    svg.selectAll("*").remove();
    for (const t of y.ticks(5)) {
      svg.append("line").attr("class", "grid").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(t)).attr("y2", y(t));
      svg.append("text").attr("class", "tick").attr("x", m.l - 10).attr("y", y(t) + 4).attr("text-anchor", "end").text(`${t}%`);
    }
    C.leads.forEach(h => svg.append("text").attr("class", "tick").attr("x", x(h)).attr("y", H - 10).attr("text-anchor", "middle").text(h));
    const line = (vals) => d3.line()(vals.map((v, k) => [x(C.leads[k]), y(v)]));
    for (const f of ["A", "B"]) {
      svg.append("path").attr("class", `ah-line ${f.toLowerCase()} tot`).attr("d", line(C.total[f]));
      svg.append("path").attr("class", `ah-line ${f.toLowerCase()}`).attr("d", line(C.country[f]));
      C.total[f].forEach((v, k) => svg.append("circle").attr("class", `ah-dot ${f.toLowerCase()} tot`).attr("cx", x(C.leads[k])).attr("cy", y(v)).attr("r", 3.5));
      C.country[f].forEach((v, k) => svg.append("circle").attr("class", `ah-dot ${f.toLowerCase()}`).attr("cx", x(C.leads[k])).attr("cy", y(v)).attr("r", 4.5));
    }
    const lab = f => data.folds.find(z => z.id === f).label;
    svg.attr("aria-label", `Error by months ahead. One country at a time, ${lab("A")}: ${C.country.A.map((v, k) => `month ${k + 1} ${v}%`).join(", ")}. ${lab("B")}: ${C.country.B.map((v, k) => `month ${k + 1} ${v}%`).join(", ")}. International total, ${lab("A")}: ${C.total.A.map((v, k) => `month ${k + 1} ${v}%`).join(", ")}. ${lab("B")}: ${C.total.B.map((v, k) => `month ${k + 1} ${v}%`).join(", ")}.`);
  }

  function buildAhead() {
    const A = data.ahead, C = A.chart, lab = f => data.folds.find(z => z.id === f).label;
    $("ahead").innerHTML = `<p class="lead-s">${esc(A.lead)}</p>
      <div class="legend4"><span><i class="ah-line a"></i>One country at a time, ${esc(lab("A"))}</span><span><i class="ah-line b"></i>One country at a time, ${esc(lab("B"))}</span>
        <span><i class="ah-line a tot"></i>International total, ${esc(lab("A"))}</span><span><i class="ah-line b tot"></i>International total, ${esc(lab("B"))}</span></div>
      <div class="bt-wrap" id="ah"><svg role="img"></svg></div>
      <p class="method ah-x">Months after the model’s data ended (1 to 7). Gap between rebuilt and reported guest nights, in %: the average gap over the countries (solid), and for the international total (dashed).</p>
      <ul class="pts">${A.points.map(p => `<li>${esc(p)}</li>`).join("")}</ul>
      <p class="method">${esc(A.method)}</p>
      <details class="tbl"><summary>Show the numbers</summary><table id="ah-table"><thead><tr><th>Months ahead</th><th>Month, ${esc(lab("A"))}</th><th>Month, ${esc(lab("B"))}</th>
        <th>Countries, A</th><th>Countries, B</th><th>Total, A</th><th>Total, B</th><th>Inside the 80% range, A</th><th>Inside the 80% range, B</th></tr></thead><tbody>${C.leads.map((h, k) =>
        `<tr><th>${h}</th><td>${mon(C.months.A[k])}</td><td>${mon(C.months.B[k])}</td><td>${C.country.A[k]}%</td><td>${C.country.B[k]}%</td><td>${C.total.A[k]}%</td><td>${C.total.B[k]}%</td><td>${C.coverage.A[k]}%</td><td>${C.coverage.B[k]}%</td></tr>`).join("")}</tbody></table></details>`;
  }

  // ------------------------------------------------------------------ 3. the country catch
  function buildCatch() {
    const C = data.catch, top = Math.max(...C.rows.flatMap(r => [r.national_day, r.eid]));
    const bar = (v) => `<span class="trk"><i class="${v >= 1 ? "up" : "dn"}" style="width:${(Math.abs(v - 1) / (top - 1)) * 50}%;${v >= 1 ? "left:" : "right:"}50%"></i><u></u></span><b>${times(v)}</b>`;
    $("catch").innerHTML = `<p class="lead-s">${esc(C.lead)}</p>
      <div class="cg" role="table" aria-label="Check-ins in the holiday month against a normal month, by passport">
        <div role="row" class="ch"><span role="columnheader">Passport</span><span role="columnheader">National Day month (UAE residents’ holiday)</span><span role="columnheader">Eid month (Gulf-wide holiday)</span></div>
        ${C.rows.map(r => `<div role="row"><span role="rowheader">${esc(r.label)}</span><span role="cell" class="v">${bar(r.national_day)}</span><span role="cell" class="v">${bar(r.eid)}</span></div>`).join("")}
      </div>
      <p class="method">Bars show check-ins against a normal month: right of the centre line is more, left is fewer.</p>
      <ul class="pts">${C.points.map(p => `<li>${esc(p)}</li>`).join("")}</ul>`;
  }

  // ------------------------------------------------------------------ 4. March 2026
  function buildMarch() {
    const M = data.march;
    $("march").innerHTML = `<p class="lead-s">${esc(M.lead)}</p>
      <div class="mg" role="table" aria-label="March 2026 check-ins against a normal March, by world region">
        <div role="row" class="mh"><span role="columnheader">World region</span><span role="columnheader">Normal March check-ins</span><span role="columnheader">Reported March 2026 against normal; what flights alone would have given</span></div>
        ${M.rows.map(r => {
          const f = r.flights, px = v => `${Math.min(100, v * 100)}%`;
          return `<div role="row"><span role="rowheader">${esc(r.region)}<small>${r.flight_markets} of ${r.markets} ${r.markets === 1 ? "market has" : "markets have"} flights in the data</small></span>
            <span role="cell" class="v">${n3(r.normal_checkins)}</span>
            <span role="cell" class="v"><span class="trk2" aria-hidden="true"><u style="left:100%"></u>${f ? `<i class="fl" style="left:${px(f[0])};width:${Math.max(1.2, (f[1] - f[0]) * 100)}%"></i>` : ""}<em class="ob" style="left:${px(r.observed)}"></em></span>
              <b>${Math.round(r.observed * 100)}%</b><span class="fo">${f ? `flights alone: ${Math.round(f[0] * 100)}% to ${Math.round(f[1] * 100)}%` : "no flight data to compare"}</span></span></div>`;
        }).join("")}
      </div>
      <p class="legend2"><span><em class="ob"></em>Reported March 2026</span><span><i class="fl"></i>What flights alone would have given</span><span><u></u>A normal March</span></p>`;
  }

  // ------------------------------------------------------------------ 5. the Sobol check
  function buildSobol() {
    const S = data.sobol, pc = v => `${Math.round(v * 100)}%`;
    const bar = r => `<span class="sb" aria-hidden="true"><i style="width:${r.alone * 100}%"></i><u style="width:${r.total * 100}%"></u></span>`;
    $("sobol").innerHTML = `<p class="lead-s">${esc(S.lead)}</p>
      <div class="sg" role="table" aria-label="Share of the spread in guest nights each lever explains, all levers moving together">
        <div role="row" class="sh"><span role="columnheader">Lever</span><span role="columnheader">Share of the spread in guest nights</span></div>
        ${S.rows.map(r => `<div role="row"><span role="rowheader">${esc(r.label)}</span><span role="cell" class="v">${bar(r)}<b>${pc(r.total)}</b></span></div>`).join("")}
      </div>
      <p class="method">${esc(S.method)}</p>
      <ul class="pts">${S.points.map(p => `<li>${esc(p)}</li>`).join("")}</ul>`;
  }

  // ------------------------------------------------------------------ 6. assumptions and limits
  function buildRest() {
    $("assume").innerHTML = `<thead><tr><th>Step</th><th>What we assume</th><th>Value</th><th>Where it comes from</th><th>You can change it</th></tr></thead><tbody>${
      data.assumptions.map(a => `<tr><th>${esc(a.step)}</th><td>${esc(a.what)}</td><td class="val">${esc(a.value)}</td><td>${esc(a.source)}</td><td>${esc(a.change)}</td></tr>`).join("")}</tbody>`;
    $("limits").innerHTML = data.limits.map(g => `<h3 class="sub-h">${esc(g.group)}</h3><ul class="limits">${
      g.items.map(l => `<li><b>${esc(l.title)}</b> ${esc(l.text)}</li>`).join("")}</ul>`).join("");
  }

  function build() {
    root.innerHTML = `
      <p class="lead" id="kn-lead"></p>
      <div class="stats" id="kn-stats"></div>
      <p class="method" id="kn-periods"></p>
      <section class="mv-sec" aria-labelledby="kn-h-bt">
        <div class="h"><h2 id="kn-h-bt">Reported against rebuilt, month by month</h2>
          <span class="hint">Months the model never saw, in two separate periods</span></div>
        <div class="bt-ctl"><label>Show <select id="bt-market" aria-label="Country"></select></label>
          <span class="seg" role="radiogroup" aria-label="Period" id="bt-fold"></span></div>
        <div class="bt-wrap" id="bt"><svg role="img"></svg></div>
        <div class="legend3"><span><i class="a"></i>Reported by hotels</span><span><i class="r"></i>Rebuilt from flights and normal stays</span><span><i class="b"></i>80% range</span></div>
        <p class="heat-read" id="bt-read" aria-live="polite"></p>
        <details class="tbl"><summary>Show the numbers</summary><table id="bt-table"></table></details>
      </section>
      <section class="mv-sec" aria-labelledby="kn-h-ah"><div class="h"><h2 id="kn-h-ah">Does the error grow the further ahead?</h2>
        <span class="hint">Months 1 to 7 of each held-out period</span></div><div id="ahead"></div></section>
      <section class="mv-sec" aria-labelledby="kn-h-catch"><div class="h"><h2 id="kn-h-catch">Flights and guests do not count the same people</h2></div><div id="catch"></div></section>
      <section class="mv-sec" aria-labelledby="kn-h-mar"><div class="h"><h2 id="kn-h-mar">March 2026: the one month flights cannot explain</h2></div><div id="march"></div></section>
      <section class="mv-sec" aria-labelledby="kn-h-sob"><div class="h"><h2 id="kn-h-sob">All levers at once</h2>
          <span class="hint">A check on the ranking in “What moves guests”, and on the ranges</span></div><div id="sobol"></div></section>
      <section class="mv-sec" aria-labelledby="kn-h-as"><div class="h"><h2 id="kn-h-as">Every assumption, and where to change it</h2></div><div class="tbl-wrap"><table class="assume" id="assume"></table></div></section>
      <section class="mv-sec" aria-labelledby="kn-h-lim"><div class="h"><h2 id="kn-h-lim">What this does not do</h2></div><div id="limits"></div></section>`;
    $("kn-lead").textContent = data.lead;
    $("kn-stats").innerHTML = data.stats.map(s => `<div class="stat" data-id="${s.id}"><span class="t">${esc(s.title)}</span><b class="big">${esc(s.big)}</b><span class="vs">${esc(s.versus)}</span><span class="n">${esc(s.note)}</span></div>`).join("");
    $("kn-periods").textContent = `Pairs of numbers are the two held-out periods, in order: ${data.folds.map(f => f.label).join(" · ")}. Error is WMAPE: the total gap divided by the total reported.`;
    buildBacktest(); buildAhead(); buildCatch(); buildMarch(); buildSobol(); buildRest();
    built = true;
  }

  return {
    async show() {
      if (!built) {
        root.setAttribute("aria-busy", "true");
        try { data = await call("/api/know"); }
        catch (e) { root.innerHTML = `<p class="lead">This could not be loaded: ${esc(e.message)}</p>`; return; }
        finally { root.removeAttribute("aria-busy"); }
        build();
      }
      drawBacktest(); drawAhead();
    },
    resize() { if (built && !root.hidden) { drawBacktest(); drawAhead(); } },
  };
}
