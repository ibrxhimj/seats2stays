// "What moves guests" (Tab 2). Four answers, every number from the server:
//   1. which lever moves hotel guest nights most — each lever moved 10% up and down over the year, with its 80% range
//      (the Phase 3 sensitivity table, /api/moves);
//   2. where and when one extra seat is worth the most — guest nights per 1,000 seats for one extra daily narrowbody,
//      market by month; a click loads that exact case into the simulator, whose result equals the cell (tested);
//   3. how many flights a target needs — the engine inverted (/api/reverse), as a fill-in sentence;
//   4. four planner rules, each with the numbers it rests on.
import { ANY, monthLabel, lastDay, windowText, newRouteSpec } from "./forms.js";

const d3 = window.d3;
const $$ = (root, sel) => root.querySelector(sel);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const MINUS = "−";
const sig3 = x => { if (!x) return 0; const m = 10 ** (Math.floor(Math.log10(Math.abs(x))) - 2); return Math.round(x / m) * m; };
const num = x => Math.abs(Math.round(x)).toLocaleString("en-GB");
const n3 = x => (x < 0 ? MINUS : "") + num(sig3(x));                      // an estimate, 3 significant figures
const signed3 = x => (x > 0 ? "+" : x < 0 ? MINUS : "") + num(sig3(x));
const flights = x => (x < 0.5 ? "under 1" : Math.round(x).toLocaleString("en-GB"));
const cellText = v => (v >= 1000 ? num(sig3(v)) : String(Math.round(v)));
const longMonth = ym => new Date(`${ym}-01T00:00:00Z`).toLocaleString("en-GB", { month: "long", year: "numeric", timeZone: "UTC" });

// seven steps so that every cell's number is readable (dark text on the lighter four, white on the darker three)
const BINS = [100, 200, 300, 450, 650, 900], SHADES = ["#EDF3F6", "#D3E5EC", "#AECFDB", "#86B8CB", "#4F9AB5", "#1D7C9C", "#0A5873"], WHITE_FROM = 5;
const shade = v => BINS.filter(b => v >= b).length;
const SHOWN = 10;                                                         // markets with flights shown before "Show all"
const DAGGERS = ["†", "‡", "§"];

export function createMoves({ root, meta, call, load }) {
  const byKey = Object.fromEntries(meta.markets.map(m => [m.key, m]));
  let data = null, built = false, seq = 0, timer = 0;
  const $ = id => document.getElementById(id);

  // ------------------------------------------------------------------ 1. the levers
  function leverSide(s) {
    const dir = s.change_pct > 0 ? "+10%" : `${MINUS}10%`;
    return { dir, text: `${signed3(s.d_guest_nights)} guest nights (80% range ${signed3(s.lo)} to ${signed3(s.hi)})` };
  }

  function drawLevers() {
    const L = data.levers, rows = L.rows, host = $("tor");
    const reach = d3.max(rows.flatMap(r => [r.down, r.up].flatMap(s => [Math.abs(s.lo), Math.abs(s.hi)]))) * 1.6;   // room for the end labels
    host.innerHTML = rows.map(r => `<div class="tor-row">
        <div class="tor-label"><b>${esc(r.name)}</b><span>${esc(r.what)}</span><em>1% more: ${r.opposite ? MINUS : "+"}${r.elasticity.toFixed(2)}% guest nights</em></div>
        <svg class="tor-chart" role="img" aria-label="${esc(`${r.name}. If ${leverSide(r.down).dir}: ${leverSide(r.down).text}. If ${leverSide(r.up).dir}: ${leverSide(r.up).text}.`)}"></svg>
      </div>`).join("") + `<div class="tor-row axis"><div></div><svg class="tor-chart" aria-hidden="true"></svg></div>`;
    const svgs = [...host.querySelectorAll("svg.tor-chart")];
    const W = svgs[0].getBoundingClientRect().width;
    if (!W) return;
    const x = d3.scaleLinear().domain([-reach, reach]).range([8, W - 8]), x0 = x(0), ticks = x.ticks(7).filter(t => Math.abs(t) < reach);
    const fmt = t => (t === 0 ? "0" : (t < 0 ? MINUS : "") + d3.format(".2~s")(Math.abs(t)));
    const css = getComputedStyle(document.documentElement);
    const marks = []; rows.forEach(r => [r.down, r.up].forEach(s => { if (s.note) marks.push({ row: r, side: s, sym: DAGGERS[marks.length] }); }));
    const H = 58, mid = 22, bh = 26;
    svgs.forEach((el, i) => {
      const svg = d3.select(el).attr("viewBox", `0 0 ${W} ${i === rows.length ? 30 : H}`).attr("height", i === rows.length ? 30 : H);
      svg.selectAll("*").remove();
      for (const t of ticks) {
        if (i < rows.length) svg.append("line").attr("class", t === 0 ? "zero" : "grid").attr("x1", x(t)).attr("x2", x(t)).attr("y1", 0).attr("y2", H);
        else svg.append("text").attr("class", "tick").attr("x", x(t)).attr("y", 18).attr("text-anchor", "middle").text(fmt(t));
      }
      if (i === rows.length) { svg.append("line").attr("class", "zero").attr("x1", x0).attr("x2", x0).attr("y1", 0).attr("y2", 6); return; }
      const r = rows[i];
      for (const s of [r.down, r.up]) {
        const gain = s.d_guest_nights >= 0, col = css.getPropertyValue(gain ? "--gain" : "--loss").trim();
        const a = Math.min(x0, x(s.d_guest_nights)), b = Math.max(x0, x(s.d_guest_nights));
        svg.append("rect").attr("x", a).attr("y", mid - bh / 2).attr("width", Math.max(1, b - a)).attr("height", bh).attr("fill", col).attr("rx", 2);
        svg.append("line").attr("class", "whisker").attr("x1", x(s.lo)).attr("x2", x(s.hi)).attr("y1", mid).attr("y2", mid);
        for (const v of [s.lo, s.hi]) svg.append("line").attr("class", "whisker").attr("x1", x(v)).attr("x2", x(v)).attr("y1", mid - 6).attr("y2", mid + 6);
        const end = gain ? x(s.hi) + 9 : x(s.lo) - 9, anchor = gain ? "start" : "end";
        const mk = marks.find(m => m.side === s);
        const t = svg.append("text").attr("class", "val").attr("x", end).attr("y", mid + 2).attr("text-anchor", anchor);
        t.append("tspan").text(signed3(s.d_guest_nights) + (mk ? ` ${mk.sym}` : ""));
        svg.append("text").attr("class", "sub").attr("x", end).attr("y", mid + 20).attr("text-anchor", anchor).text(`if ${s.change_pct > 0 ? "+10%" : `${MINUS}10%`}`);
      }
    });
    $("tor-notes").innerHTML = marks.map(m => `<li><span>${m.sym}</span> ${esc(m.row.name)}, ${m.side.change_pct > 0 ? "+10%" : `${MINUS}10%`}: ${esc(m.side.note)}</li>`).join("");
  }

  // ------------------------------------------------------------------ 2. where and when a seat is worth the most
  function heatHtml() {
    const G = data.grid, seats = G.seats_added, wsum = seats.reduce((a, b) => a + b, 0);
    const yearAvg = G.season.reduce((a, v, i) => a + v * seats[i], 0) / wsum;
    const head = G.month_labels.map((l, i) => `<th scope="col">${l.slice(0, 3)}${i === 0 || l.startsWith("Jan") ? `<small>${l.slice(4)}</small>` : ""}</th>`).join("");
    const cell = (v, i, flag, m) => `<td><button type="button" class="cell s${shade(v)}${flag ? " flag" : ""}" data-m="${m}" data-i="${i}" tabindex="-1">${cellText(v)}</button></td>`;
    const avg = `<tr class="avg"><th scope="row">All markets with flights</th>${G.season.map((v, i) => `<td><span class="cell s${shade(v)}" data-avg="${i}">${cellText(v)}</span></td>`).join("")}<td class="yr" data-avg-year>${cellText(yearAvg)}</td></tr>`;
    let n = 0, lastFlights = true, body = "";
    for (const m of G.rows) {
      if (lastFlights && !m.flights) { body += `<tr class="grp"><th colspan="14" scope="colgroup">No flights from these countries in the data: the hotel response is borrowed from other markets</th></tr>`; lastFlights = false; }
      const extra = n >= SHOWN;
      body += `<tr data-m="${m.market}" class="${extra ? "extra" : ""}"><th scope="row">${esc(m.label)}${m.year[3] === 1 ? `<abbr class="dag" title="${esc(G.rules[0])}">†</abbr>` : ""}</th>${
        m.cells.map((c, i) => cell(c[0], i, c[4] > 0 && c[4] < 4, m.market)).join("")}<td class="yr" data-year="${m.market}">${cellText(m.year[0])}</td></tr>`;
      n++;
    }
    return `<table class="heat"><thead><tr><th scope="col" class="corner">Country</th>${head}<th scope="col" class="yr">Year</th></tr></thead><tbody>${avg}${body}</tbody></table>`;
  }

  function heatRead(el) {
    const G = data.grid, def = "Point at a cell to read it. Click one to load it into the simulator.";
    if (!el) return def;
    if (el.dataset.avg !== undefined) return `<b>All markets with flights, ${longMonth(G.months[+el.dataset.avg])}</b>: ${cellText(G.season[+el.dataset.avg])} guest nights per 1,000 seats on average.`;
    if (el.matches("[data-avg-year]")) return `<b>All markets with flights, ${esc(data.span)}</b>: ${el.textContent} guest nights per 1,000 seats on average.`;
    if (el.dataset.year) {
      const m = G.rows.find(r => r.market === el.dataset.year), y = m.year;
      return `<b>${esc(m.label)}, ${esc(data.span)}</b>: ${cellText(y[0])} guest nights per 1,000 seats across the twelve months (80% range ${cellText(y[1])} to ${cellText(y[2])}).`;
    }
    const m = G.rows.find(r => r.market === el.dataset.m), i = +el.dataset.i, c = m.cells[i], rule = G.rules[c[4] - 1];
    return `<b>${esc(m.label)}, ${longMonth(G.months[i])}</b>: one extra daily narrowbody (${num(G.seats_added[i])} seats) brings about ${n3(c[3])} guest nights `
      + `(80% range ${n3(c[1])} to ${n3(c[2])}), ${cellText(c[0])} per 1,000 seats.${rule ? ` <em>${esc(rule)}</em>` : ""}`;
  }

  function buildHeat() {
    const G = data.grid, host = $("heat");
    host.innerHTML = heatHtml();
    const read = $("heat-read"); read.innerHTML = heatRead(null);
    const labels = ["under 100", "100", "200", "300", "450", "650", "900 +"];
    $("heat-legend").innerHTML = `<span class="scale"><b>Guest nights per 1,000 seats</b>${labels.map((t, i) => `<i class="sw s${i}">${t}</i>`).join("")}</span>`
      + `<span><i class="k-flag"></i>Corner mark: extended beyond the tested range, or capped by local passengers; point at the cell to see which</span>`
      + `<span class="dagger">${esc(G.dagger)}</span>`;
    const first = host.querySelector(".cell[data-m]"); if (first) first.tabIndex = 0;
    const set = el => { read.innerHTML = heatRead(el); };
    host.addEventListener("mouseover", e => { const t = e.target.closest(".cell[data-m], td[data-year], .cell[data-avg], [data-avg-year]"); if (t) set(t); });
    host.addEventListener("mouseleave", () => { if (!host.contains(document.activeElement)) set(null); });
    host.addEventListener("focusin", e => {
      const b = e.target.closest(".cell[data-m]"); if (!b) return;
      host.querySelectorAll('.cell[tabindex="0"]').forEach(x => (x.tabIndex = -1)); b.tabIndex = 0; set(b);
    });
    host.addEventListener("focusout", e => { if (!host.contains(e.relatedTarget)) set(null); });
    host.addEventListener("keydown", e => {                                           // arrow keys move between cells (one tab stop for the grid)
      const b = e.target.closest(".cell[data-m]"); if (!b || !e.key.startsWith("Arrow")) return;
      e.preventDefault();
      const rows = [...host.querySelectorAll("tbody tr[data-m]")].filter(r => r.offsetParent !== null);
      const r = rows.indexOf(b.closest("tr")), c = +b.dataset.i;
      const nr = Math.min(rows.length - 1, Math.max(0, r + (e.key === "ArrowDown") - (e.key === "ArrowUp")));
      const nc = Math.min(11, Math.max(0, c + (e.key === "ArrowRight") - (e.key === "ArrowLeft")));
      rows[nr].querySelector(`.cell[data-i="${nc}"]`)?.focus();
    });
    host.addEventListener("click", async e => {
      const b = e.target.closest(".cell[data-m]"); if (!b || b.classList.contains("busy")) return;
      const m = byKey[b.dataset.m], ym = G.months[+b.dataset.i];
      b.classList.add("busy");
      try { await load(newRouteSpec({ market: m.key, city: m.place, airline: ANY, fpw: 7, aircraft: "narrowbody", from: ym, to: ym })); }
      finally { b.classList.remove("busy"); }
    });
    const more = $("heat-more"), total = G.rows.length, extra = host.querySelectorAll("tr.extra").length;
    const setMore = open => {
      host.classList.toggle("open", open); more.setAttribute("aria-expanded", String(open));
      more.textContent = open ? "Show fewer markets" : `Show all ${total} markets`;
    };
    more.hidden = !extra; setMore(false);
    more.addEventListener("click", () => setMore(!host.classList.contains("open")));
  }

  // ------------------------------------------------------------------ 3. how many flights does a target need?
  const seasons = meta.seasons;
  const months = (() => {
    const out = []; let [y, m] = [+meta.horizon[0].slice(0, 4), +meta.horizon[0].slice(5, 7)];
    for (;;) {
      const ym = `${y}-${String(m).padStart(2, "0")}`; out.push(ym);
      if (ym === meta.horizon[1].slice(0, 7)) return out;
      if (++m > 12) { m = 1; y++; }
    }
  })();
  const windowOf = v => {
    const s = seasons.find(x => x.id === v);
    return s ? { start: s.window[0], end: s.window[1], from: s.window[0].slice(0, 7), to: s.window[1].slice(0, 7) }
             : { start: `${v}-01`, end: lastDay(v), from: v, to: v };
  };
  const AIR = { today: "on today's routes", narrowbody: "on a new narrowbody route", widebody: "on a new widebody route" };
  let last = null;                                                                  // the answer on screen, for "Add this to the simulator"

  function buildReverse() {
    const mk = meta.markets.filter(m => m.key !== "DOMESTIC").sort((a, b) => a.label.localeCompare(b.label));
    $("rv-market").innerHTML = mk.map(m => `<option value="${m.key}">${esc(m.label)}</option>`).join("");
    $("rv-market").value = "UNITED KINGDOM";
    $("rv-window").innerHTML = `<optgroup label="Season">${seasons.map(s => `<option value="${s.id}">${esc(s.label)}: ${esc(s.span)}</option>`).join("")}</optgroup>`
      + `<optgroup label="A single month">${months.map(m => `<option value="${m}">${esc(monthLabel(m))}</option>`).join("")}</optgroup>`;
    $("rv-window").value = "winter";
    $("rv-air").innerHTML = Object.entries(AIR).map(([k, t]) => `<option value="${k}">${t}</option>`).join("");
    const sync = () => {                                                            // a country with no flights can only get a new route
      const flies = byKey[$("rv-market").value].has_flight_data;
      $("rv-air").querySelector('[value="today"]').disabled = !flies;
      if (!flies && $("rv-air").value === "today") $("rv-air").value = "narrowbody";
    };
    const target = $("rv-target");
    const parse = () => { const x = Number(target.value.replace(/[,\s]/g, "")); return Number.isFinite(x) && x > 0 ? x : null; };   // 0, negatives, text: no
    const stale = () => { last = null; $("rv-add").disabled = true; };                  // never add an answer to a question that has changed
    const go = () => { sync(); stale(); clearTimeout(timer); timer = setTimeout(ask, 220); };
    target.addEventListener("input", go);
    target.addEventListener("blur", () => { const x = parse(); if (x >= 1 && x < 1e12) target.value = Math.round(x).toLocaleString("en-GB"); });
    for (const id of ["rv-market", "rv-window", "rv-air"]) $(id).addEventListener("change", () => { sync(); stale(); clearTimeout(timer); ask(); });
    $("rv-add").addEventListener("click", async () => {
      if (!last) return;
      const b = $("rv-add"); b.disabled = true;
      try { await load(last.spec); } finally { b.disabled = !last; }
    });
    sync();
    async function ask() {
      const t = parse(), n = ++seq;
      target.setAttribute("aria-invalid", String(!t));
      if (!t) { answer({ error: "Enter a number of extra guest nights above 0." }); return; }
      stale(); $("rv").setAttribute("aria-busy", "true");
      const w = windowOf($("rv-window").value), market = $("rv-market").value, aircraft = $("rv-air").value;
      try {
        const r = await call("/api/reverse", { market, start: w.start, end: w.end, target: t, aircraft });
        if (n === seq) answer({ r, w, market });
      } catch (e) { if (n === seq) answer({ error: e.message }); }
    }
    ask();
  }

  function answer({ r, w, market, error }) {
    const box = $("rv"), det = $("rv-detail"), nEl = $("rv-n");
    box.removeAttribute("aria-busy"); last = null; det.classList.toggle("warn", !!error);
    $("rv-add").disabled = true; $("rv-add").textContent = "Add to the simulator";
    $("rv-about").hidden = false; $("rv-fl").textContent = "flights";
    if (error) { nEl.textContent = "…"; det.textContent = error; $("rv-more").textContent = ""; return; }
    const f = r.flights_per_week, m = byKey[market], today = r.aircraft === "today";
    if (r.aircraft in AIR && $("rv-air").value !== r.aircraft) $("rv-air").value = r.aircraft;               // the server's own choice shows
    $("rv-more").textContent = today ? "more " : "";
    if (f == null) {                                                               // beyond what the simulator takes: say so, with the most it would win
      nEl.textContent = `more than ${flights(r.flights_per_week_max)}`; $("rv-about").hidden = true;
      det.classList.add("warn"); det.textContent = r.text; return;
    }
    const [a, b] = r.flights_per_week_range, nWhole = Math.max(1, Math.round(f));
    nEl.textContent = flights(f); $("rv-about").hidden = f < 0.5; $("rv-fl").textContent = Math.round(f) <= 1 ? "flight" : "flights";
    const range = a != null ? `80% range ${flights(a)} to ${b != null ? flights(b) : `more than ${flights(r.flights_per_week_max)}`}` : "80% range out of reach";
    const cur = r.current_flights_per_week;
    const where = cur >= 0.5 ? `${m.label} has about ${flights(cur)} flights a week in these months` : `There are no flights from ${m.label} in these months`;
    const seats = `about ${n3(r.seats_added)} extra seats in all`;
    const beyond = r.outside_tested_range ? ` Beyond the range the model was tested on (up to ${flights(r.tested_max_flights)} more flights a week): extended in a straight line, not estimated.` : "";
    const more = 100 * f / cur;
    det.textContent = (today ? `${range}. ${where}, so that is ${more < 1 ? "under 1" : Math.round(more)}% more seats (${seats}).`
      : `${range}. ${where}${cur >= 0.5 ? "; this adds" : ": this adds"} ${seats} (${r.seats_per_flight.toFixed(0)} seats a flight).`) + beyond;
    let spec;
    if (today) {
      const pct = 100 * nWhole / cur;
      spec = { changes: [{ type: "route", markets: market, start: w.start, end: w.end, seats_pct: pct }],
               label: `${m.label} flights: +${nWhole} a week`, sub: `${pct > 0 ? "+" : ""}${pct.toFixed(1)}% of seats, ${windowText(w.from, w.to)}` };
    } else spec = newRouteSpec({ market, city: m.place, airline: ANY, fpw: nWhole, aircraft: r.aircraft, from: w.from, to: w.to });
    last = { spec }; $("rv-add").disabled = false;
    $("rv-add").textContent = `Add ${nWhole} ${nWhole === 1 ? "flight" : "flights"} a week to the simulator`;
  }

  // ------------------------------------------------------------------ the page
  function build() {
    root.innerHTML = `
      <p class="lead" id="mv-lead"></p>
      <section class="mv-sec" aria-labelledby="mv-h-levers">
        <div class="h"><h2 id="mv-h-levers">Which lever moves guest nights most?</h2><span class="hint">Each lever moved 10% up and down for the year, <span id="mv-span"></span></span></div>
        <div class="tor" id="tor"></div>
        <p class="method" id="tor-scope"></p>
        <ul class="tor-notes" id="tor-notes"></ul>
      </section>
      <section class="mv-sec" aria-labelledby="mv-h-heat">
        <div class="h"><h2 id="mv-h-heat">Where and when a seat is worth the most</h2><span class="hint">One extra daily narrowbody (${Math.round(data.grid.seats_per_flight)} seats a flight), one month at a time</span></div>
        <div class="heat-wrap" id="heat"></div>
        <p class="heat-read" id="heat-read" aria-live="polite"></p>
        <div class="legend" id="heat-legend"></div>
        <button type="button" class="link" id="heat-more" aria-expanded="false"></button>
      </section>
      <section class="mv-sec" aria-labelledby="mv-h-rev" id="rv">
        <div class="h"><h2 id="mv-h-rev">How many flights does a target need?</h2><span class="hint">The simulator run backwards</span></div>
        <p class="rv-sentence">To win <input id="rv-target" type="text" inputmode="numeric" value="10,000" maxlength="14" aria-label="Extra guest nights wanted" size="9"> more guest nights from
          <select id="rv-market" aria-label="Country"></select> in <select id="rv-window" aria-label="When"></select>, Abu Dhabi needs
          <span id="rv-about">about </span><b class="rv-n" id="rv-n">…</b> <span id="rv-more"></span><span id="rv-fl">flights</span> a week <select id="rv-air" aria-label="Kind of flights"></select>.</p>
        <p class="rv-detail" id="rv-detail" aria-live="polite"></p>
        <button type="button" class="pin" id="rv-add" disabled>Add to the simulator</button>
      </section>
      <section class="mv-sec" aria-labelledby="mv-h-rules">
        <div class="h"><h2 id="mv-h-rules">What a planner should do differently</h2></div>
        <ol class="rules" id="rules"></ol>
      </section>`;
    $("mv-lead").textContent = data.lead;
    $("mv-span").textContent = data.span;
    $("tor-scope").textContent = data.levers.scope;
    $("rules").innerHTML = data.rules.map(r => `<li><b>${esc(r.title)}</b><span>${esc(r.evidence)}</span></li>`).join("");
    buildHeat(); buildReverse();
    built = true;
  }

  return {
    async show() {
      if (!built) {
        root.setAttribute("aria-busy", "true");
        try { data = await call("/api/moves"); }
        catch (e) { root.innerHTML = `<p class="lead">This could not be loaded: ${esc(e.message)}</p>`; return; }
        finally { root.removeAttribute("aria-busy"); }
        build();
      }
      drawLevers();
    },
    resize() { if (built && !root.hidden) drawLevers(); },
  };
}
