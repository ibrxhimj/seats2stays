// "Add a change": four small forms in one dialog, each building exactly the change the validated scenario engine takes.
//   Change a route        one existing route: flights a week, or plane size            → type "route"
//   Open a new route      country, city, airline, flights a week, plane, optional mix  → type "new_route"
//   Planes and connections how full planes are / how many passengers connect onwards    → "load_factor" / "transfer_share"
//   Demand and stays      hotel demand not caused by flights / how long guests stay     → "demand" / "stay_length"
// A form never calculates anything: it returns the change; the server runs it (and refuses what it cannot honour,
// in words the dialog shows). `submit` is the page's callback; while it runs the dialog stays open.
const $$ = (root, sel) => root.querySelector(sel);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const MN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export const monthLabel = ym => `${MN[+ym.slice(5, 7) - 1]} ${ym.slice(0, 4)}`;
export const lastDay = ym => { const [y, m] = ym.split("-").map(Number); return `${ym}-${String(new Date(Date.UTC(y, m, 0)).getUTCDate()).padStart(2, "0")}`; };
const signed = x => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toLocaleString("en-GB", { maximumFractionDigits: 2 });
export const shortAirline = a => a.replace(/ (Airways|Airlines|Airline)$/, "");
export const windowText = (a, b) => (a === b ? monthLabel(a) : `${monthLabel(a)} to ${monthLabel(b)}`);
const opts = (list, sel) => list.map(o => `<option value="${esc(o.v)}"${o.v === sel ? " selected" : ""}>${esc(o.t)}</option>`).join("");
const radios = (name, list, sel) => `<div class="seg" role="radiogroup">${list.map(o =>
  `<label><input type="radio" name="${name}" value="${o.v}"${o.v === sel ? " checked" : ""}><span>${esc(o.t)}</span></label>`).join("")}</div>`;
export const ANY = "generic";                                                    // no named airline: the market's own passenger mix
const field = (label, html, id) => `<div class="fd-field"><label${id ? ` for="${id}"` : ""}>${label}</label>${html}</div>`;

function monthSpan(months) {                                               // [11,12,1,2] → "Nov to Feb"
  const s = new Set(months);
  if (s.size === 12) return "all year";
  const runs = [];
  for (let m = 1; m <= 12; m++) if (s.has(m) && !s.has((m + 10) % 12 + 1)) {
    let e = m; while (s.has(e % 12 + 1)) e = e % 12 + 1;
    runs.push(e === m ? MN[m - 1] : `${MN[m - 1]} to ${MN[e - 1]}`);
  }
  return runs.join(", ");
}

// The one-route change, in the shape the new-route form and every shortcut (the heat map, the reverse sum) share:
// v = { market, city, airline, fpw, aircraft, from, to [, load_factor, local_share] }, months as "2026-11".
export function newRouteSpec(v) {
  const ch = { type: "new_route", market: v.market, city: v.city, airline: v.airline, start: `${v.from}-01`, end: lastDay(v.to),
               flights_per_week: v.fpw, aircraft_class: v.aircraft };
  if (v.load_factor !== undefined) ch.load_factor = v.load_factor;
  if (v.local_share !== undefined) ch.local_share = v.local_share;
  return { kind: "new", changes: [ch], values: v, label: `New route: ${v.city} (${v.airline === ANY ? "any airline" : shortAirline(v.airline)})`,
           sub: `${v.fpw} ${v.fpw === 1 ? "flight" : "flights"} a week, ${v.aircraft}, ${windowText(v.from, v.to)}` };
}

export function createForms(meta, submit) {
  const hm = [];                                                           // the twelve months of the horizon
  for (let [y, m] = [+meta.horizon[0].slice(0, 4), +meta.horizon[0].slice(5, 7)]; ; m++) {
    if (m > 12) { m = 1; y++; }
    const ym = `${y}-${String(m).padStart(2, "0")}`; hm.push({ v: ym, t: monthLabel(ym), moy: m });
    if (ym === meta.horizon[1].slice(0, 7)) break;
  }
  const byKey = Object.fromEntries(meta.markets.map(m => [m.key, m]));
  const flightMk = meta.markets.filter(m => m.has_flight_data).sort((a, b) => a.label.localeCompare(b.label));
  const allMk = meta.markets.filter(m => m.key !== "DOMESTIC").sort((a, b) => a.label.localeCompare(b.label));
  const airline = Object.fromEntries(meta.airlines.map(a => [a.name, a]));
  const pctText = x => `${Math.round(x * 100)}%`;
  const winter = ["2026-11", "2027-02"];

  const windowHtml = (v, span) => `<div class="fd-field"><label id="fd-when-l">When</label><div class="fd-when" role="group" aria-labelledby="fd-when-l">
      <select name="from" aria-label="From month">${opts(hm, v.from || span[0])}</select><span>to</span>
      <select name="to" aria-label="To month">${opts(hm, v.to || span[1])}</select></div></div>`;
  const window_ = root => {
    const from = $$(root, "[name=from]").value, to = $$(root, "[name=to]").value;
    if (to < from) throw new Error("The last month is before the first month.");
    return { from, to, start: `${from}-01`, end: lastDay(to) };
  };
  const num = (root, name, label, { min, max, nonzero = false }) => {
    const raw = $$(root, `[name=${name}]`).value.trim(), x = Number(raw);
    if (raw === "" || !Number.isFinite(x)) throw new Error(`Enter a number for ${label}.`);
    if (nonzero && x === 0) throw new Error(`The change in ${label} cannot be 0.`);
    if (x < min || x > max) throw new Error(`${label[0].toUpperCase()}${label.slice(1)} must be between ${min} and ${max}.`);
    return x;
  };
  const scopeHtml = (name, v, { all, countries }) => `<select name="${name}" id="fd-${name}">${all.map(g => `<option value="${esc(g.v)}"${g.v === v ? " selected" : ""}>${esc(g.t)}</option>`).join("")}
      <optgroup label="Region">${opts(meta.regions.map(r => ({ v: r, t: r })), v)}</optgroup>
      <optgroup label="Country">${opts(countries.map(m => ({ v: m.key, t: m.label })), v)}</optgroup></select>`;
  const scopeText = (root, name) => { const s = $$(root, `[name=${name}]`); return s.options[s.selectedIndex].text; };

  // ------------------------------------------------------------ the four forms
  const kinds = {
    route: {
      name: "Change a route",
      render(root, v) {
        v = { market: "UNITED KINGDOM", mode: "flights", ...v };
        root.innerHTML = field("Country", `<select name="market" id="fd-market">${opts(flightMk.map(m => ({ v: m.key, t: m.label })), v.market)}</select>`, "fd-market") +
          field("Route", `<select name="route" id="fd-route"></select>`, "fd-route") + `<p class="fd-hint" id="fd-hint"></p>` +
          field("Change", `<div class="fd-inline">${radios("mode", [{ v: "flights", t: "Flights a week" }, { v: "plane", t: "Seats per plane" }], v.mode)}
            <input name="value" type="number" step="1" inputmode="numeric" aria-label="New value"></div>`) + windowHtml(v, winter);
        const routes = () => byKey[$$(root, "[name=market]").value].routes;
        const cur = () => routes()[+$$(root, "[name=route]").value];
        const mode = () => $$(root, "[name=mode]:checked").value;
        const fillValue = () => { const r = cur(); $$(root, "[name=value]").value = Math.round(mode() === "flights" ? r.flights_per_week : r.seats_per_flight); };
        const hint = () => { const r = cur();
          $$(root, "#fd-hint").textContent = `Today: ${Math.round(r.flights_per_week)} flights a week, ${Math.round(r.seats_per_flight)} seats a plane; flies ${monthSpan(r.months)}. Set flights to 0 to cancel the route.`; };
        const span = () => { const fl = hm.filter(m => cur().months.includes(m.moy)); return fl.length === 12 || !fl.length ? winter : [fl[0].v, fl[fl.length - 1].v]; };
        const fillRoutes = (pick) => {
          const rs = routes(); const i = pick ? Math.max(0, rs.findIndex(r => r.city === pick.city && r.airline === pick.airline)) : 0;
          $$(root, "[name=route]").innerHTML = rs.map((r, k) => `<option value="${k}"${k === i ? " selected" : ""}>${esc(r.city)} · ${esc(shortAirline(r.airline))}${r.months.length < 12 ? ` (${monthSpan(r.months)})` : ""}</option>`).join("");
        };
        const syncWindow = () => { const [a, b] = span(); $$(root, "[name=from]").value = a; $$(root, "[name=to]").value = b; };
        fillRoutes(v); hint();
        if (v.value !== undefined) $$(root, "[name=value]").value = v.value; else fillValue();
        if (v.from === undefined) syncWindow();
        $$(root, "[name=market]").onchange = () => { fillRoutes(); hint(); fillValue(); syncWindow(); };
        $$(root, "[name=route]").onchange = () => { hint(); fillValue(); syncWindow(); };
        root.querySelectorAll("[name=mode]").forEach(r => r.onchange = fillValue);
      },
      build(root) {
        const market = $$(root, "[name=market]").value, r = byKey[market].routes[+$$(root, "[name=route]").value], mode = $$(root, "[name=mode]:checked").value;
        const w = window_(root), n = num(root, "value", mode === "flights" ? "flights a week" : "seats per plane", { min: mode === "flights" ? 0 : 1, max: mode === "flights" ? 200 : 600 });
        if (n === Math.round(mode === "flights" ? r.flights_per_week : r.seats_per_flight))
          throw new Error(`That is what this route has today (${n} ${mode === "flights" ? "flights a week" : "seats per plane"}). Enter a different number.`);
        const ch = { type: "route", markets: market, city: r.city, airline: r.airline, [mode === "flights" ? "flights_per_week" : "seats_per_flight"]: n, start: w.start, end: w.end };
        const who = `${r.city} (${shortAirline(r.airline)})`;
        return { kind: "route", changes: [ch], values: { market, city: r.city, airline: r.airline, mode, value: n, from: w.from, to: w.to },
          label: mode === "flights" ? (n === 0 ? `${who} cancelled` : `${who}: ${n} ${n === 1 ? "flight" : "flights"} a week`) : `${who}: ${n}-seat planes`,
          sub: `was ${Math.round(mode === "flights" ? r.flights_per_week : r.seats_per_flight)} ${mode === "flights" ? "a week" : "seats"}, ${windowText(w.from, w.to)}` };
      },
    },

    new: {
      name: "Open a new route",
      render(root, v) {
        v = { market: "MEXICO", city: byKey.MEXICO?.place, airline: "Etihad Airways", fpw: 2, aircraft: "widebody", ...v };
        const air = [...meta.airlines.map(a => ({ v: a.name, t: a.name })), { v: ANY, t: "Any airline (the country's usual passenger mix)" }];
        root.innerHTML = field("Country", `<select name="market" id="fd-market">${opts(allMk.map(m => ({ v: m.key, t: m.label + (m.has_flight_data ? "" : " (no flights in the data)") })), v.market)}</select>`, "fd-market") +
          field("City", `<input name="city" id="fd-city" type="text" value="${esc(v.city || "")}" autocomplete="off">`, "fd-city") +
          field("Airline", `<select name="airline" id="fd-airline">${opts(air, v.airline)}</select>`, "fd-airline") + `<p class="fd-hint" id="fd-hint"></p>` +
          field("Flights a week", `<input name="fpw" id="fd-fpw" type="number" min="1" max="100" step="1" value="${v.fpw}" inputmode="numeric">`, "fd-fpw") +
          field("Plane", radios("aircraft", Object.keys(meta.aircraft).map(k => ({ v: k, t: `${k[0].toUpperCase()}${k.slice(1)}, ${Math.round(meta.aircraft[k])} seats` })), v.aircraft)) +
          windowHtml(v, winter) +
          `<details class="fd-mix"${v.load_factor !== undefined || v.local_share !== undefined ? " open" : ""}><summary>Passenger mix: set it yourself</summary>
            <div class="fd-two">${field("Seats filled, %", `<input name="load_factor" type="number" min="1" max="100" step="1" inputmode="numeric" value="${v.load_factor !== undefined ? Math.round(v.load_factor * 100) : ""}">`)}
            ${field("End their trip in Abu Dhabi, %", `<input name="local_share" type="number" min="1" max="100" step="1" inputmode="numeric" value="${v.local_share !== undefined ? Math.round(v.local_share * 100) : ""}">`)}</div></details>`;
        const hint = () => {
          const m = byKey[$$(root, "[name=market]").value], a = airline[$$(root, "[name=airline]").value];
          const mix = a ? `${a.name} fills about ${pctText(a.load_factor)} of seats and ${pctText(a.local_share)} of its passengers end their trip in Abu Dhabi.` : "";
          $$(root, "#fd-hint").textContent = (m.has_flight_data ? (a ? `Left blank, the passenger mix comes from ${a.name}'s own routes. ${mix}`
              : `Left blank, the passenger mix is ${m.label}'s own: how full its planes are and how many of its passengers end their trip here today.`)
            : `${m.label} has no flights in the data, so the hotel response is borrowed from other markets (about ${Number(meta.facts.new_route_per_seat.value.toPrecision(2))} check-ins per added seat); it is shown in the notes after adding.`);
          const lf = $$(root, "[name=load_factor]"), ls = $$(root, "[name=local_share]");
          lf.placeholder = a ? String(Math.round(a.load_factor * 100)) : ""; ls.placeholder = a ? String(Math.round(a.local_share * 100)) : "";
        };
        hint();
        const city = $$(root, "[name=city]"); city.oninput = () => { city.dataset.touched = "1"; };
        $$(root, "[name=market]").onchange = e => { if (!city.dataset.touched) city.value = byKey[e.target.value].place; hint(); };
        $$(root, "[name=airline]").onchange = hint;
      },
      build(root) {
        const market = $$(root, "[name=market]").value, city = $$(root, "[name=city]").value.trim(), ac = $$(root, "[name=aircraft]:checked").value;
        if (!city) throw new Error("Enter the city the new route flies from.");
        const air = $$(root, "[name=airline]").value, w = window_(root), n = num(root, "fpw", "flights a week", { min: 1, max: 100 });
        const values = { market, city, airline: air, fpw: n, aircraft: ac, from: w.from, to: w.to };
        for (const [name, key] of [["load_factor", "load_factor"], ["local_share", "local_share"]]) {
          if ($$(root, `[name=${name}]`).value.trim() !== "") values[key] = num(root, name, name === "load_factor" ? "seats filled" : "share ending their trip", { min: 1, max: 100 }) / 100;
        }
        return newRouteSpec(values);
      },
    },

    mix: {
      name: "Planes and connections",
      render(root, v) {
        const given = v.value !== undefined;
        v = { mode: "lf", scope: "flight markets", ...v }; if (!given) v.value = v.mode === "lf" ? 5 : -5;
        root.innerHTML = field("What changes", radios("mode", [{ v: "lf", t: "How full planes are" }, { v: "ts", t: "Passengers connecting onwards" }], v.mode)) +
          field("Markets", scopeHtml("scope", v.scope, { all: [{ v: "flight markets", t: "All markets with flights" }], countries: flightMk }), "fd-scope") +
          field(`<span id="fd-vlabel"></span>`, `<input name="value" id="fd-value" type="number" step="any" inputmode="decimal" value="${v.value}">`, "fd-value") +
          `<p class="fd-hint" id="fd-hint"></p>` + windowHtml(v, winter);
        const val = $$(root, "[name=value]"); val.dataset.touched = given ? "1" : ""; val.oninput = () => { val.dataset.touched = "1"; };
        const sync = () => {
          const lf = $$(root, "[name=mode]:checked").value === "lf";
          $$(root, "#fd-vlabel").textContent = lf ? "Change in passengers per seat, %" : "Change in the share connecting onwards, points";
          $$(root, "#fd-hint").textContent = lf ? `Planes are ${pctText(meta.network.load_factor)} full on average; a route cannot fill beyond its seats (capped).`
            : `On average ${pctText(meta.network.transfer_share)} of passengers connect onwards at Abu Dhabi. A negative number means fewer connect and more stay.`;
          if (!val.dataset.touched) val.value = lf ? 5 : -5;
        };
        sync(); root.querySelectorAll("[name=mode]").forEach(r => r.onchange = sync);
      },
      build(root) {
        const mode = $$(root, "[name=mode]:checked").value, scope = $$(root, "[name=scope]").value, w = window_(root);
        const x = num(root, "value", mode === "lf" ? "passengers per seat" : "connecting share", { min: mode === "lf" ? -99 : -100, max: 100, nonzero: true });
        const where = scopeText(root, "scope");
        const ch = { type: mode === "lf" ? "load_factor" : "transfer_share", markets: scope, [mode === "lf" ? "pct" : "points"]: x, start: w.start, end: w.end };
        return { kind: "mix", changes: [ch], values: { mode, scope, value: x, from: w.from, to: w.to },
          label: mode === "lf" ? `${x > 0 ? "Fuller" : "Emptier"} planes: ${signed(x)}% passengers per seat` : `${x < 0 ? "Fewer" : "More"} connecting passengers: ${signed(x)} points`,
          sub: `${where}, ${windowText(w.from, w.to)}` };
      },
    },

    demand: {
      name: "Demand and stays",
      render(root, v) {
        v = { mode: "demand", scope: "all", value: 10, ...v };
        root.innerHTML = field("What changes", radios("mode", [{ v: "demand", t: "Demand" }, { v: "stay", t: "Length of stay" }], v.mode)) +
          field("Guests", scopeHtml("scope", v.scope, { all: [{ v: "all", t: "All guests" }, { v: "international", t: "International guests" }, { v: "DOMESTIC", t: "UAE residents" }], countries: allMk }), "fd-scope") +
          field(`<span id="fd-vlabel"></span>`, `<input name="value" id="fd-value" type="number" step="any" inputmode="decimal" value="${v.value}">`, "fd-value") +
          `<p class="fd-hint" id="fd-hint"></p>` + windowHtml(v, winter);
        const sync = () => {
          const d = $$(root, "[name=mode]:checked").value === "demand";
          $$(root, "#fd-vlabel").textContent = d ? "Change in hotel check-ins not caused by flights, %" : "Change in the average stay, %";
          $$(root, "#fd-hint").textContent = d ? "For causes other than flights: travel advisories, visas, events, regional conditions. An assumption, not estimated from the flight data."
            : "Applies to guests checking in during these months. An assumption, not estimated from the flight data.";
        };
        sync(); root.querySelectorAll("[name=mode]").forEach(r => r.onchange = sync);
      },
      build(root) {
        const mode = $$(root, "[name=mode]:checked").value, scope = $$(root, "[name=scope]").value, w = window_(root);
        const x = num(root, "value", mode === "demand" ? "demand" : "stay length", { min: -99, max: 500, nonzero: true }), where = scopeText(root, "scope");
        return { kind: "demand", assumption: true, changes: [{ type: mode === "demand" ? "demand" : "stay_length", markets: scope, pct: x, start: w.start, end: w.end }],
          values: { mode, scope, value: x, from: w.from, to: w.to },
          label: mode === "demand" ? `Demand ${signed(x)}%: ${where}` : `Stays ${x > 0 ? "longer" : "shorter"} by ${Math.abs(x)}%: ${where}`,
          sub: `${mode === "demand" ? "Not caused by flights" : "Length of stay"}, ${windowText(w.from, w.to)}` };
      },
    },
  };

  // ------------------------------------------------------------ the dialog
  const dlg = document.createElement("dialog");
  dlg.id = "form-dialog"; dlg.setAttribute("aria-labelledby", "fd-title");
  dlg.innerHTML = `<form id="fd-form" novalidate>
      <header><h2 id="fd-title">Add a change</h2><button type="button" class="fd-x" aria-label="Close">×</button></header>
      <div class="fd-kinds" id="fd-kinds">${Object.entries(kinds).map(([k, o]) => `<label><input type="radio" name="kind" value="${k}"><span>${o.name}</span></label>`).join("")}</div>
      <div class="fd-body" id="fd-body"></div>
      <p class="fd-error" id="fd-error" role="alert"></p>
      <footer><button type="button" class="fd-cancel">Cancel</button><button type="submit" class="fd-add">Add change</button></footer></form>`;
  document.body.appendChild(dlg);
  const form = $$(dlg, "form"), body = $$(dlg, "#fd-body"), err = $$(dlg, "#fd-error");
  let kind = "route", editId = null;

  const show = (k, v) => { kind = k; err.textContent = ""; kinds[k].render(body, v || {}); dlg.querySelector(`[name=kind][value=${k}]`).checked = true; };
  const close = () => dlg.close();
  $$(dlg, ".fd-x").onclick = $$(dlg, ".fd-cancel").onclick = close;
  dlg.addEventListener("click", e => { if (e.target === dlg) close(); });                    // click outside
  dlg.querySelectorAll("[name=kind]").forEach(r => r.onchange = () => show(r.value));
  form.addEventListener("submit", async e => {
    e.preventDefault();
    const btn = $$(dlg, ".fd-add"); err.textContent = "";
    let spec;
    try { spec = kinds[kind].build(body); } catch (x) { err.textContent = x.message; return; }
    btn.disabled = true; const t = btn.textContent; btn.textContent = "Calculating…";
    try { await submit({ ...spec, id: editId }); close(); }
    catch (x) { err.textContent = x.message; }
    finally { btn.disabled = false; btn.textContent = t; }
  });

  return {
    open(initial = {}) {                                           // initial: { kind, values, id } to edit, or { kind, values } to prefill
      editId = initial.id ?? null;
      $$(dlg, "#fd-title").textContent = editId ? "Edit this change" : "Add a change";
      $$(dlg, "#fd-kinds").hidden = !!editId;
      $$(dlg, ".fd-add").textContent = editId ? "Save change" : "Add change";
      show(initial.kind || "route", initial.values);
      dlg.showModal();
    },
    // the form values of a preset's single new-route change, so "Edit" works on it too
    fromChange(c) {
      if (c.type !== "new_route" || !c.aircraft_class || c.aircraft_assumed) return null;
      const v = { market: c.market, city: c.city, airline: c.airline, fpw: c.flights_per_week, aircraft: c.aircraft_class, from: c.start.slice(0, 7), to: c.end.slice(0, 7) };
      if (c.load_factor !== undefined) v.load_factor = c.load_factor;
      if (c.local_share !== undefined) v.local_share = c.local_share;
      return { kind: "new", values: v };
    },
  };
}
