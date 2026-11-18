// Seats2Stays — Tab 1: presets, the change strips (each change on its own, plus the scenario total), the live seats
// lever, the four add-a-change forms, scenario A for comparison, the by-month and by-day views, and downloads.
// Every number comes from the server: a full engine run, or — while dragging — the lever curve, whose every position
// equals a full run (tested). Dragging makes no network requests.
import { Stream, drawRange } from "./stream.js";
import { WorldMap, fillList } from "./where.js";
import { drawMonths, drawDaily } from "./when.js";
import { createForms } from "./forms.js";
import { createMoves } from "./moves.js";
import { createKnow } from "./know.js";

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const wholeFlights = x => (x <= 0 ? "0" : x < 0.5 ? "under 1" : Math.round(x).toLocaleString("en-GB"));
const signed = (x, d = 0) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toLocaleString("en-GB", { maximumFractionDigits: d });
const sig3 = x => { if (!x) return 0; const m = 10 ** (Math.floor(Math.log10(Math.abs(x))) - 2); return Math.round(x / m) * m; };
const plain = m => String(m).replace(/^change \d+ \([^)]*\):\s*/, "").replace(/^(transfer_share|load_factor|new_route|route): /, "").replace(/\bnew_route\b/g, "a new route");

async function call(url, body) {
  const r = await fetch(url, body === undefined ? {} : { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || j.detail || `request failed (${r.status})`);
  return j;
}

const state = { meta: null, shown: null, groups: [], view: null, results: new Map(), curves: new Map(), pinned: null, when: "month",
                dailyDrawn: false, lever: { market: "UNITED KINGDOM", season: "winter" }, curve: null, gid: 0, pending: null,
                response: "central" };
window.__frames = [];                                                       // render times while dragging (tests)

const keyOf = changes => JSON.stringify(changes);
function result(changes) {                                                  // full engine run, cached by its changes and the response
  const k = `${state.response}|${keyOf(changes)}`;
  if (!state.results.has(k)) {
    const p = call("/api/run", { changes, response: state.response });
    p.catch(() => state.results.delete(k));
    state.results.set(k, p);
  }
  return state.results.get(k);
}

// ---------------------------------------------------------------- set-up
const [meta, land] = await Promise.all([call("/api/meta"), call("vendor/land-110m.json")]);
state.meta = meta;
const flightKeys = new Set(meta.markets.filter(m => m.has_flight_data).map(m => m.key));
const labelOf = Object.fromEntries(meta.markets.map(m => [m.key, m.label]));
const stream = new Stream($("stream"));
const pickMarket = k => (flightKeys.has(k) ? setLever(k, state.lever.season)
  : forms.open({ kind: "new", values: { market: k, city: meta.markets.find(m => m.key === k).place } }));
const map = new WorldMap($("map"), land, meta, pickMarket);
await document.fonts.ready;

const f = meta.facts;
$("trust-range").textContent = `${f.backtest_A_monthly_total.value.toFixed(1)}–${f.backtest_B_monthly_total.value.toFixed(1)}%`;
document.querySelector(".trust").title = `Error (WMAPE) on two held-out periods: all international guest nights ${f.backtest_A_monthly_total.value.toFixed(1)}% and ${f.backtest_B_monthly_total.value.toFixed(1)}%; one nationality at a time about ${Math.round((f.backtest_A_monthly_market.value + f.backtest_B_monthly_market.value) / 2)}%, which is why each country's figure carries a wider range.`;
$("horizon-note").textContent = "Compared with the normal year, March 2026 to February 2027";
$("lever-market").innerHTML = meta.markets.filter(m => m.has_flight_data).sort((a, b) => a.label.localeCompare(b.label))
  .map(m => `<option value="${m.key}">${esc(m.label)}</option>`).join("");
$("lever-season").innerHTML = meta.seasons.map(s => `<button type="button" data-season="${s.id}" title="${esc(s.span)}">${esc(s.label)}</button>`).join("");
$("presets").innerHTML = meta.presets.map(p => `<button class="preset" data-preset="${p.id}" aria-pressed="false"><b>${esc(p.label)}</b><span>${esc(p.blurb)}</span></button>`).join("");

// ---------------------------------------------------------------- rendering
function months(res) {                                                      // full result, or a lever step + its curve
  if (res.by_month[0]?.label) return res.by_month;
  return state.curve.months.map((m, i) => ({ ...m, ...res.by_month[i] }));
}

function leverScale(res) {
  const cv = state.curve, steps = cv.steps;
  const max = Math.max(Math.abs(steps[0].chain.stages[0].value), Math.abs(steps[steps.length - 1].chain.stages[0].value)) || 1;
  return Math.max(0.12, Math.sqrt(Math.abs(res.chain.stages[0].value) / max));
}

function show(res, { live = false, sweep = false, lever = null, label = state.shown?.label } = {}) {
  const t0 = performance.now();
  state.shown = { res, live, lever, label };
  $("list").parentElement.style.minHeight = "";
  $("viewing").textContent = withResponse(label || "Normal year");             // what the result below is for
  const loss = res.chain.direction === "loss";
  const lead = $("lead"); lead.classList.toggle("loss", loss);
  lead.innerHTML = res.sentence.map(s => s.strong ? `<strong>${esc(s.text)}</strong>` : esc(s.text)).join("") + (res.note ? ` ${esc(res.note)}` : "");
  stream.render(res.chain, { scale: lever ? leverScale(res) : 1, outside: !!lever?.outside });
  const h = res.headline;
  $("big").textContent = h.text;
  const ck = res.chain.stages[3], gn = res.chain.stages[4];
  $("via").textContent = ck.ratio_text && gn.ratio_text
    ? `from ${ck.text.replace(/^[+−]/, "")} ${loss ? "fewer " : ""}check-ins × ${gn.ratio_text.replace(" per check-in", " each")}` : "";
  drawRange($("range"), h.lo, h.d_guest_nights, h.hi, loss);
  $("range-text").textContent = h.d_guest_nights ? `80% range ${h.range_text}` + (res.response && res.response.id !== "central" ? ` · at the ${res.response.label.toLowerCase()} check-in response` : "") : "";
  $("occ").textContent = h.occupancy ? h.occupancy.text.replace("≈", "About") : "";
  const why = lever?.because;
  $("why").hidden = !why; $("why").innerHTML = why ? `<b>Why</b> ${esc(why)}` : "";
  const p = res.chain.per_100_seats;
  $("per100").innerHTML = p ? esc(p.text).replace(/(\d[\d.,]*)/g, "<b>$1</b>") : "";
  map.update(res.by_market, lever ? state.lever.market : null);
  fillList($("list"), res.by_market, pickMarket, flightKeys);
  drawWhen(res, { sweep });
  renderCompare(res);
  if (!live) {
    const notes = res.notes || [];
    $("warnings").innerHTML = notes.filter(n => n.level === "warning")
      .map(n => `<div class="warning"><span class="glyph" aria-hidden="true">!</span><p>${esc(n.text)}</p></div>`).join("");
    const info = notes.filter(n => n.level !== "warning");
    $("notes-box").hidden = !info.length;
    $("notes-summary").textContent = `About this result: ${info.length} ${info.length === 1 ? "note" : "notes"} on how it was calculated`;
    $("notes").innerHTML = info.map(n => `<li>${esc(n.text)}</li>`).join("");
  }
  if (live) window.__frames.push(performance.now() - t0);
}

const monthFrom = new Date(meta.beyond_validation_from + "-01").toLocaleString("en-GB", { month: "short", year: "numeric" });
function legendHtml(day, loss) {
  if (day) return `<span><i class="k-line base"></i>Normal year</span><span><i class="k-line ${loss ? "loss" : ""}"></i>Your scenario</span><span><i class="k-dash"></i>Highest ever recorded</span>`;
  return `<span><i class="k-bar ${loss ? "loss" : ""}"></i>Change in guest nights</span><span><i class="k-range"></i>80% range</span>` +
    (state.pinned ? `<span><i class="k-tick"></i>Scenario A</span>` : "") +
    `<span><i class="k-hatch"></i>From ${monthFrom}: more than seven months beyond the data</span>`;
}

// "When they stay": the change by month (with scenario A as ticks), or by day against the highest level ever recorded.
// The by-day numbers come with a confirmed run: while dragging, the last confirmed day chart stays and says so.
function drawWhen(res, { sweep = false } = {}) {
  const day = state.when === "day", loss = res.chain.direction === "loss", live = state.shown.live;
  $("legend").innerHTML = legendHtml(day, loss);
  $("months").setAttribute("aria-label", day ? "Hotel guests by day against the highest level ever recorded" : "Change in guest nights by month");
  if (!day) { state.dailyDrawn = false; drawMonths($("months"), months(res), { sweep, ghost: state.pinned?.months || null }); $("when-note").hidden = true; return; }
  if (res.daily) { drawDaily($("months"), res.daily, loss); state.dailyDrawn = true; }
  else if (!state.dailyDrawn) $("months").replaceChildren();
  $("when-note").hidden = !(live && !res.daily);
}

// ---------------------------------------------------------------- scenario A and the comparison
function renderCompare(res) {
  const A = state.pinned, box = $("compare");
  box.hidden = !A;
  if (!A) return;
  const h = res.headline, st = res.chain.stages, now = withResponse(state.shown.label || "Now");
  const row = (name, a, b, d, sub) => `<tr><th scope="row">${name}</th><td>${esc(a)}${sub ? `<small>${esc(sub[0])}</small>` : ""}</td><td>${esc(b)}${sub ? `<small>${esc(sub[1])}</small>` : ""}</td><td class="d">${d}</td></tr>`;
  const diff = (x, est) => { const v = est ? sig3(x) : Math.round(x); return v === 0 ? `<span class="same">same</span>` : `<b class="${v < 0 ? "loss" : "gain"}">${signed(v)}</b>`; };
  box.innerHTML = `<div class="cmp-head"><h2>Compared with A</h2><button type="button" class="link" data-unpin>Unpin A</button></div>
    <table><thead><tr><th></th><th>A<small>${esc(A.label)}</small></th><th>Now<small>${esc(now)}</small></th><th>Now − A</th></tr></thead><tbody>
    ${row("Guest nights", A.headline.text, h.text, diff(h.d_guest_nights - A.headline.d_guest_nights, true), [`80% range ${A.headline.range_text}`, `80% range ${h.range_text}`])}
    ${row("Hotel check-ins", A.stages[3].text, st[3].text, diff(st[3].value - A.stages[3].value, true))}
    ${row("Inbound passengers", A.stages[2].text, st[2].text, diff(st[2].value - A.stages[2].value, false))}
    ${row("Seats", A.stages[0].text, st[0].text, diff(st[0].value - A.stages[0].value, false))}</tbody></table>`;
}

function renderPinned() {
  const A = state.pinned;
  $("pinned").innerHTML = A ? `<div class="pinned"><span class="tag">A</span><div><b>${esc(A.label)}</b><span>${esc(A.headline.text)} guest nights</span></div><button type="button" class="x" data-unpin aria-label="Unpin A">×</button></div>` : "";
  $("pin").textContent = A ? "Replace A with this" : "Pin as A to compare";
}

function pin() {
  const sh = state.shown; if (!sh) return;
  const h = sh.res.headline;
  state.pinned = { label: withResponse(sh.label || "Normal year"), headline: { text: h.text, d_guest_nights: h.d_guest_nights, range_text: h.range_text },
    stages: sh.res.chain.stages.map(x => ({ text: x.text, value: x.value })), months: months(sh.res).map(m => ({ month: m.month, d_guest_nights: m.d_guest_nights })) };
  renderPinned(); show(sh.res, { live: sh.live, lever: sh.lever });
}
function unpin() { state.pinned = null; renderPinned(); const sh = state.shown; if (sh) show(sh.res, { live: sh.live, lever: sh.lever }); }
$("pin").addEventListener("click", pin);
document.addEventListener("click", e => { if (e.target.closest("[data-unpin]")) unpin(); });
$("when-view").addEventListener("click", e => {
  const b = e.target.closest("[data-view]"); if (!b) return;
  state.when = b.dataset.view;
  for (const x of $("when-view").children) x.setAttribute("aria-pressed", String(x === b));
  if (state.shown) drawWhen(state.shown.res);
  if (state.when === "day" && state.shown?.preview) previewDays();
});

function showError(e) {
  $("lead").classList.remove("loss");
  $("lead").textContent = `This could not be calculated: ${plain(e.message)}`;
}

// ---------------------------------------------------------------- the rail: presets, changes, total
const totalChanges = () => state.groups.flatMap(g => g.changes);

function renderRail() {
  const only = state.groups.length === 1 ? state.groups[0] : null;
  for (const b of $("presets").children) b.setAttribute("aria-pressed", String(!!only && only.preset === b.dataset.preset));
  $("clear").hidden = !state.groups.length;
  const rows = state.groups.map(g => `
    <div class="strip ${g.loss ? "loss" : ""} ${g.assumption ? "demand" : ""} ${g.form ? "form" : ""} ${state.view === g.id ? "active" : ""}" data-gid="${g.id}">
      <span class="edge"></span>
      <button class="what" data-view="${g.id}">${esc(g.label)}</button>${g.form ? `<button class="edit" data-edit="${g.id}" aria-label="Edit ${esc(g.label)}">Edit</button>` : ""}
      <div class="when">${esc(g.sub)}</div>
      <div class="val">${esc(g.value || "…")}</div>
      <button class="x" data-remove="${g.id}" aria-label="Remove ${esc(g.label)}">×</button>
    </div>`);
  if (state.groups.length > 1)
    rows.push(`<button class="total ${state.view === "total" ? "active" : ""}" data-view="total"><span>Scenario total</span>${state.totalValue ? `<b>${esc(state.totalValue)}</b>` : `<i>Calculate</i>`}</button>`);
  $("strips").innerHTML = rows.join("") || `<p class="empty">No changes yet. Pick a preset, drag the flights above, or add a change.</p>`;
}

async function showView() {
  renderRail();
  const g = state.groups.find(x => x.id === state.view);
  const view = state.view, current = () => (view === "total" ? totalChanges() : g ? g.changes : []);
  const changes = current();
  const busy = setTimeout(() => $("tab-simulator").classList.add("busy"), 150);   // only a long run (a big scenario) shows it
  try {
    const res = await result(changes).finally(() => { clearTimeout(busy); $("tab-simulator").classList.remove("busy"); });
    if (state.view !== view || keyOf(current()) !== keyOf(changes)) return;   // superseded while it ran (a drag, a click)
    if (g) { g.value = res.headline.text; g.loss = res.headline.d_guest_nights < 0; }
    else if (view === "total") state.totalValue = res.headline.text;
    renderRail();
    show(res, { sweep: true, label: g ? g.label : view === "total" ? "Scenario total" : "Normal year",
                lever: g?.lever ? { because: g.lever.because, outside: g.lever.outside } : null });
  } catch (e) { showError(e); }
}

async function loadPreset(pid) {
  const p = await call(`/api/presets/${pid}`);
  const meta_p = meta.presets.find(x => x.id === pid);
  const g = { id: ++state.gid, preset: pid, label: meta_p.label, sub: `${p.changes.length} ${p.changes.length === 1 ? "change" : "changes"}`, changes: p.changes,
              form: p.changes.length === 1 ? forms.fromChange(p.changes[0]) : null };
  state.groups = [g]; state.view = g.id; state.totalValue = null;
  await showView();
  syncSlider();
}

$("presets").addEventListener("click", e => { const b = e.target.closest("[data-preset]"); if (b) loadPreset(b.dataset.preset).catch(showError); });
$("strips").addEventListener("click", e => {
  const ed = e.target.closest("[data-edit]");
  if (ed) { const g = state.groups.find(x => x.id === +ed.dataset.edit); if (g?.form) forms.open({ ...g.form, id: g.id }); return; }
  const rm = e.target.closest("[data-remove]");
  if (rm) {
    const id = +rm.dataset.remove;
    state.groups = state.groups.filter(g => g.id !== id); state.totalValue = null;
    if (state.view === id || (state.view === "total" && state.groups.length < 2)) state.view = state.groups[0]?.id ?? null;
    showView(); syncSlider(); return;
  }
  const v = e.target.closest("[data-view]");
  if (v) {
    state.view = v.dataset.view === "total" ? "total" : +v.dataset.view;
    const g = state.groups.find(x => x.id === state.view);
    if (g?.lever) setLever(g.lever.market, g.lever.season, { keepView: true }); else if (g) followLever(g.changes);
    showView();
  }
});

// ---------------------------------------------------------------- the check-in response (the chain's one estimated step)
// Every result is shown at the central estimate unless the planner picks an end of its validated 80% interval. The
// 80% range is always the model's, taken around the result shown; the live lever and every change are re-run at it.
const RESP = Object.fromEntries(meta.responses.map(r => [r.id, r]));
const upTen = e => `+${((1.1 ** e - 1) * 100).toFixed(1)}%`;
function withResponse(label) { return state.response === "central" ? label : `${label} (${RESP[state.response].label.toLowerCase()} response)`; }
function renderResponse() {
  $("response").innerHTML = ["low", "central", "high"].map(id => `<button type="button" data-resp="${id}" aria-pressed="${id === state.response}">${esc(RESP[id].label)}</button>`).join("");
  const r = RESP[state.response];
  $("resp-note").innerHTML = `How strongly check-ins follow inbound passengers: 10% more passengers, <b>${upTen(r.eps)}</b> check-ins (${esc(r.text)}`
    + (state.response === "central" ? `; Low and High are the ends of that range, ${upTen(RESP.low.eps)} and ${upTen(RESP.high.eps)}).` : ").");
}
renderResponse();
$("response").addEventListener("click", e => {
  const b = e.target.closest("[data-resp]"); if (!b || b.dataset.resp === state.response) return;
  state.response = b.dataset.resp; renderResponse();
  state.totalValue = null;
  for (const g of state.groups) {                                           // every strip's figure at the new response
    g.value = null;
    result(g.changes).then(res => { g.value = res.headline.text; g.loss = res.headline.d_guest_nights < 0; renderRail(); }).catch(() => {});
  }
  showView(); setLever(state.lever.market, state.lever.season, { keepView: true });
});

$("clear").addEventListener("click", () => { state.groups = []; state.view = null; state.totalValue = null; showView(); syncSlider(); });
$("add-change").addEventListener("click", () => forms.open());

// A change (from a form, the heat map or the reverse sum) is run first (the server refuses what it cannot honour, in words the
// dialog shows); then it becomes a strip of its own, selected. Editing replaces the strip's change.
async function addSpec(spec) {
  let res;
  try { res = await result(spec.changes); } catch (e) { throw new Error(plain(e.message)); }
  let g = spec.id != null && state.groups.find(x => x.id === spec.id);
  if (!g) { g = { id: ++state.gid }; state.groups.push(g); }
  delete g.preset;
  Object.assign(g, { changes: spec.changes, label: spec.label, sub: spec.sub, form: spec.kind ? { kind: spec.kind, values: spec.values } : null, assumption: !!spec.assumption,
                     value: res.headline.text, loss: res.headline.d_guest_nights < 0 });
  state.view = g.id; state.totalValue = null;
  renderRail(); show(res, { sweep: true, label: g.label });
  followLever(spec.changes);
}

// The lever follows what is on screen: when the change shown is about one country that has flights, the lever points at that
// country and at the season holding the change's months, so dragging adjusts that country and not the one it was last left on.
// (A drag adds a change of its own: today's flights from that country; the change shown stays in the rail, and Edit changes it.)
const seasonFor = (a, b) => (meta.seasons.find(s => s.id !== "year" && a >= s.window[0] && b <= s.window[1]) || meta.seasons.find(s => s.id === "year")).id;
function followLever(changes) {
  const c = changes?.length === 1 ? changes[0] : null, m = c && (c.market ?? c.markets);
  if (typeof m === "string" && flightKeys.has(m) && c.start && c.end && (m !== state.lever.market || seasonFor(c.start, c.end) !== state.lever.season))
    setLever(m, seasonFor(c.start, c.end), { keepView: true });
}
const forms = createForms(meta, addSpec);
const moves = createMoves({ root: $("tab-moves"), meta, call,
  load: async spec => { tab("simulator"); try { await addSpec(spec); } catch (e) { showError(e); } } });

const know = createKnow({ root: $("tab-know"), call });

// ---------------------------------------------------------------- the live lever
const slider = $("lever-slider");
const leverGroup = (m = state.lever.market, s = state.lever.season) => state.groups.find(g => g.lever && g.lever.market === m && g.lever.season === s);

async function curve(market, season) {
  const k = `${market}|${season}|${state.response}`;
  if (!state.curves.has(k)) {
    const p = call(`/api/lever?market=${encodeURIComponent(market)}&season=${season}&response=${state.response}`);
    p.catch(() => state.curves.delete(k));
    state.curves.set(k, p);
  }
  return state.curves.get(k);
}

function readout(i) {
  const cv = state.curve, st = cv.steps[i];
  const change = st.flights_change === 0 ? "drag to add this as a new change"
    : `${signed(Math.round(st.flights_change))} a week, ${signed(st.seats_pct)}% of seats`;
  $("lever-read").innerHTML = `<b>${wholeFlights(st.flights_per_week)}</b> ${Math.round(st.flights_per_week) === 1 ? "flight" : "flights"} a week
    <span>today ${wholeFlights(cv.current_flights_per_week)}; ${change}</span>${st.outside_tested_range ? `<em>Beyond the tested range: extended, not estimated</em>` : ""}`;
}

function paintTrack() {
  const cv = state.curve, n = cv.steps.length - 1, [lo, hi] = cv.tested_range_flights;
  const fc = cv.steps.map(s => s.flights_change);
  const a = fc.findIndex(x => x >= lo), b = fc.length - 1 - [...fc].reverse().findIndex(x => x <= hi);
  const pos = i => (n ? (i / n) * 100 : 0);
  const tr = $("lever-track");
  tr.style.setProperty("--a", `${pos(Math.max(0, a))}%`); tr.style.setProperty("--b", `${pos(b)}%`);
  $("today-tick").style.left = `calc(${pos(cv.today_index)}% + ${(0.5 - pos(cv.today_index) / 100) * 22}px)`;
}

async function setLever(market, season, { keepView = false } = {}) {
  state.lever = { market, season };
  $("lever-market").value = market;
  for (const b of $("lever-season").children) b.setAttribute("aria-pressed", String(b.dataset.season === season));
  slider.disabled = true;
  try {
    const cv = await curve(market, season);
    if (state.lever.market !== market || state.lever.season !== season || cv.response !== state.response) return;
    state.curve = cv;
    slider.max = String(cv.steps.length - 1);
    const g = leverGroup();
    slider.value = String(g ? g.lever.index : cv.today_index);
    paintTrack(); readout(+slider.value);
    slider.disabled = false;
    if (!keepView) {                       // the planner picked a country or season: the screen follows it at once
      if (g) { if (state.view !== g.id) { state.view = g.id; showView(); } }
      else previewLever(cv);
    }
  } catch (e) { $("lever-read").textContent = e.message; }
}

// A country or season picked with no change of its own yet: show it at today's flights (no change), titled for it, so the
// title, the chart and the map always match the lever. The rail keeps its changes; none is selected until a drag adds one.
function previewLever(cv) {
  const st = cv.steps[cv.today_index];
  state.view = null; state.totalValue = null;
  renderRail();
  const before = document.documentElement.scrollHeight;
  show(st, { live: true, label: `${cv.label} flights, ${cv.season_label.toLowerCase()}`, lever: { because: st.because, outside: st.outside_tested_range } });
  $("warnings").innerHTML = ""; $("notes-box").hidden = true;          // notes belong to the result that was shown before
  const lost = before - document.documentElement.scrollHeight;         // the page keeps its height: a click in the list never moves it
  const row = $("list").parentElement;
  if (lost > 0) row.style.minHeight = `${row.offsetHeight + lost}px`;
  state.shown.preview = true;
  if (state.when === "day") previewDays();
}
function previewDays() {                                               // by day, no change is the normal year's days
  const shown = state.shown;
  result([]).then(r => {
    if (state.shown !== shown || !r.daily) return;
    shown.res = { ...shown.res, daily: r.daily }; shown.live = false; shown.preview = false;
    if (state.when === "day") drawWhen(shown.res);
  }).catch(() => {});
}

function syncSlider() { if (state.curve) setLever(state.lever.market, state.lever.season, { keepView: true }); }

let frame = 0;
function liveStep() {
  frame = 0;
  const cv = state.curve, i = +slider.value, st = cv.steps[i];
  readout(i);
  let g = leverGroup();
  if (!g && st.flights_change !== 0) {
    g = { id: ++state.gid, lever: { market: cv.market, season: cv.season }, changes: [] };
    state.groups.push(g); state.totalValue = null;
  }
  if (g) {
    g.lever.index = i; g.lever.because = st.because; g.lever.outside = st.outside_tested_range;
    g.changes = [{ ...cv.change, seats_pct: st.seats_pct }];
    g.label = `${cv.label} flights, ${cv.season_label.toLowerCase()}`;
    g.sub = st.flights_change === 0 ? "no change" : `${signed(Math.round(st.flights_change))} a week (${signed(st.seats_pct)}% of seats), ${cv.season_span}`;
    g.value = st.headline.text; g.loss = st.headline.d_guest_nights < 0;
    state.view = g.id; state.totalValue = null;
  }
  renderRail();
  show(st, { live: true, label: g?.label || state.shown?.label, lever: { because: st.because, outside: st.outside_tested_range } });
}

slider.addEventListener("input", () => { if (!frame) frame = requestAnimationFrame(liveStep); });
slider.addEventListener("change", () => {                                   // release: confirm with a full engine run
  clearTimeout(state.pending);
  state.pending = setTimeout(async () => {
    const g = leverGroup(); if (!g) return;
    const st = state.curve.steps[+slider.value];
    if (st.flights_change === 0) {                                          // back at today: the change is gone
      state.groups = state.groups.filter(x => x !== g);
      state.view = state.groups[0]?.id ?? null; state.totalValue = null;
      showView(); return;
    }
    const changes = g.changes;
    try {
      const res = await result(changes);
      if (keyOf(g.changes) !== keyOf(changes) || state.view !== g.id) return;
      g.value = res.headline.text; renderRail();
      show(res, { sweep: true, label: g.label, lever: { because: g.lever.because, outside: g.lever.outside } });
    } catch (e) { showError(e); }
  }, 180);
});
$("lever-market").addEventListener("change", e => setLever(e.target.value, state.lever.season));
$("lever-season").addEventListener("click", e => { const b = e.target.closest("[data-season]"); if (b) setLever(state.lever.market, b.dataset.season); });

// ---------------------------------------------------------------- downloads: what is on screen, as a table and as a scenario
const viewChanges = () => (state.view === "total" ? totalChanges() : state.groups.find(g => g.id === state.view)?.changes || []);
const slug = t => t.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "").slice(0, 50) || "scenario";
function saveAs(blob, name) {
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: name });
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
$("dl-json").addEventListener("click", () => {
  const name = state.shown?.label || "scenario";
  saveAs(new Blob([JSON.stringify({ name, response: state.response, changes: viewChanges() }, null, 2)], { type: "application/json" }), `${slug(name)}_scenario.json`);
});
$("dl-csv").addEventListener("click", async e => {
  const b = e.currentTarget, t = b.textContent, name = state.shown?.label || "scenario"; b.disabled = true; b.textContent = "Preparing…";
  try {
    const r = await fetch("/api/export", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name, response: state.response, changes: viewChanges() }) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `request failed (${r.status})`);
    saveAs(await r.blob(), `${slug(name)}_by_market_and_month.csv`);
  } catch (x) { showError(x); } finally { b.disabled = false; b.textContent = t; }
});

// ---------------------------------------------------------------- tabs
function tab(name) {
  const changed = document.querySelector(".app").dataset.tab !== name;
  document.querySelector(".app").dataset.tab = name;
  for (const a of document.querySelectorAll(".tabs a")) a.dataset.tab === name ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current");
  for (const id of ["simulator", "moves", "know"]) $(`tab-${id}`).hidden = id !== name;
  history.replaceState(null, "", `#${name}`); if (changed) window.scrollTo(0, 0);    // a new tab opens at its top; the same tab stays put
  if (name === "simulator" && state.shown) show(state.shown.res, { live: state.shown.live, lever: state.shown.lever });   // charts measure a visible page
  if (name === "moves") moves.show();
  if (name === "know") know.show();
}
// only the tab links: the page wrapper also carries data-tab (for styling), and a click anywhere must not switch tabs
document.querySelectorAll("a[data-tab]").forEach(a => a.addEventListener("click", e => { e.preventDefault(); tab(a.dataset.tab); }));

let resizeT;
window.addEventListener("resize", () => {                                  // redraw what is on screen; no new request
  clearTimeout(resizeT);
  resizeT = setTimeout(() => { state.dailyDrawn = false; !$("tab-simulator").hidden && state.shown && show(state.shown.res, { live: state.shown.live, lever: state.shown.lever }); moves.resize(); know.resize(); }, 150);
});

// ---------------------------------------------------------------- start: a real result, and the lever ready to drag
await Promise.all([loadPreset("etihad"), setLever(state.lever.market, state.lever.season, { keepView: true })]);
{ const h = location.hash.slice(1); if (h === "moves" || h === "know") tab(h); }
document.body.dataset.ready = "1";
