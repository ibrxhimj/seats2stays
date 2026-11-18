// "Where the guests come from": the world as seen from Abu Dhabi (azimuthal equidistant: distance from the centre is
// flying distance), each nationality at its largest city; a line from Abu Dhabi to every market whose guests change,
// sized by the change in guest nights. Guests are counted by nationality, not by where their flight left from.
// The land and rings are drawn once; markers and labels are redrawn on each update. Clicking a market sends it to the
// seats lever (it has flights) or opens the new-route form (it has none).
const d3 = window.d3, topojson = window.topojson;
const AUH = [54.38, 24.45];

export class WorldMap {
  constructor(svgEl, land, meta, onPick) {
    this.svg = d3.select(svgEl); this.land = land; this.meta = meta; this.onPick = onPick; this.size = 0;
  }

  _base(S) {
    this.size = S;
    this.svg.attr("viewBox", `0 0 ${S} ${S}`).selectAll("*").remove();
    this.proj = d3.geoAzimuthalEquidistant().rotate([-AUH[0], -AUH[1]]).clipAngle(132).fitExtent([[4, 4], [S - 4, S - 4]], { type: "Sphere" });
    const path = d3.geoPath(this.proj);
    this.svg.append("path").attr("class", "sphere").attr("d", path({ type: "Sphere" }));
    this.svg.append("path").attr("class", "land").attr("d", path(topojson.feature(this.land, this.land.objects.land)));
    this.ringBoxes = [];
    for (const km of [3000, 7000, 12000]) {
      this.svg.append("path").attr("class", "ring").attr("d", path(d3.geoCircle().center(AUH).radius(km / 111.195)()));
      if (km < 5000) continue;
      const r = this.proj([AUH[0], AUH[1] - km / 111.195]);
      const t = this.svg.append("text").attr("class", "ring-label").attr("x", r[0] + 5).attr("y", r[1] - 4).text(`${d3.format(",")(km)} km`);
      const bb = t.node().getBBox();
      this.ringBoxes.push([bb.x, bb.y, bb.x + bb.width, bb.y + bb.height]);
    }
    this.layer = this.svg.append("g");
    this.pts = this.meta.markets.filter(m => m.key !== "DOMESTIC").map(m => ({ ...m, xy: this.proj([m.lon, m.lat]) }));
    this.hub = this.proj(AUH);
  }

  update(byMarket, focus) {
    const S = this.svg.node().getBoundingClientRect().width;
    if (!S) return;
    if (Math.abs(S - this.size) > 0.5) this._base(S);
    const css = getComputedStyle(document.documentElement);
    const col = c => css.getPropertyValue(c < 0 ? "--loss" : "--gain").trim();
    const by = new Map(byMarket.map(m => [m.market, m]));
    const max = d3.max(byMarket, m => Math.abs(m.d_guest_nights)) || 1;
    const r = d3.scaleSqrt().domain([0, max]).range([0, 17]), w = d3.scaleSqrt().domain([0, max]).range([0.8, 4]);
    const pts = this.pts.map(p => ({ ...p, d: by.get(p.key) }));
    const changed = pts.filter(p => p.d).sort((a, b) => Math.abs(b.d.d_guest_nights) - Math.abs(a.d.d_guest_nights));
    const L = this.layer; L.selectAll("*").remove();
    const hub = this.hub;
    L.append("g").selectAll("line").data(changed).join("line").attr("class", "route")
      .attr("x1", hub[0]).attr("y1", hub[1]).attr("x2", d => d.xy[0]).attr("y2", d => d.xy[1])
      .attr("stroke", d => col(d.d.d_guest_nights)).attr("stroke-opacity", 0.55).attr("stroke-width", d => w(Math.abs(d.d.d_guest_nights)));
    const idle = L.append("g").selectAll("circle").data(pts.filter(p => !p.d)).join("circle").attr("class", "idle")
      .attr("cx", d => d.xy[0]).attr("cy", d => d.xy[1]).attr("r", d => d.key === focus ? 4.5 : 2.2);
    const mk = L.append("g").selectAll("circle").data(changed).join("circle").attr("class", "market")
      .attr("cx", d => d.xy[0]).attr("cy", d => d.xy[1]).attr("r", d => Math.max(4, r(Math.abs(d.d.d_guest_nights))))
      .attr("fill", d => col(d.d.d_guest_nights));
    for (const sel of [idle, mk]) {
      sel.filter(d => d.key !== "DOMESTIC").classed("pickable", true).on("click", (ev, d) => this.onPick(d.key));
      sel.append("title").text(d => (d.d ? `${d.label}: ${d.d.text} guest nights` : `${d.label}: no change`) + (d.key === "DOMESTIC" ? "" : d.has_flight_data ? ". Click to change its flights" : ". Click to open a new route"));
    }
    L.append("circle").attr("class", "hub").attr("cx", hub[0]).attr("cy", hub[1]).attr("r", 4.5);
    const boxes = [...this.ringBoxes];                                // market labels never cover the distance labels
    const place = (text, x, y, rad) => {
      const el = L.append("text").attr("class", "m-label").text(text);
      const tw = el.node().getComputedTextLength(), th = 14;
      for (const side of [1, -1]) {
        const lx = side > 0 ? x + rad + 5 : x - rad - 5 - tw, b = [lx - 2, y - th / 2 - 2, lx + tw + 2, y + th / 2 + 2];
        const clash = boxes.some(o => b[0] < o[2] && b[2] > o[0] && b[1] < o[3] && b[3] > o[1]);
        if (!clash && b[0] > 2 && b[2] < S - 2) { el.attr("x", lx).attr("y", y + 4.5); boxes.push(b); return true; }
      }
      el.remove(); return false;
    };
    place("Abu Dhabi", hub[0], hub[1] + 13, 0);
    for (const d of changed) boxes.push([d.xy[0] - 6, d.xy[1] - 6, d.xy[0] + 6, d.xy[1] + 6]);
    let n = 0;
    for (const d of changed) { if (n >= 3) break; if (place(d.label, d.xy[0], d.xy[1], Math.max(4, r(Math.abs(d.d.d_guest_nights))))) n++; }
  }
}

export function fillList(ul, byMarket, onPick, flightKeys, n = 7) {
  const max = d3.max(byMarket, m => Math.abs(m.d_guest_nights)) || 1;
  ul.innerHTML = "";
  if (!byMarket.length) { ul.innerHTML = `<li class="more">No change in any market.</li>`; return; }
  for (const m of byMarket.slice(0, n)) {
    const li = document.createElement("li");
    const pick = m.market !== "DOMESTIC";
    li.innerHTML = `<span class="name">${m.label}</span><span class="v">${m.text}</span>
      <span class="bar ${m.d_guest_nights < 0 ? "loss" : ""}"><i style="width:${(100 * Math.abs(m.d_guest_nights) / max).toFixed(1)}%"></i></span>`;
    if (pick) { li.classList.add("pickable"); li.tabIndex = 0; li.title = flightKeys.has(m.market) ? `Change ${m.label}'s flights` : `Open a new route from ${m.label}`;
                li.onclick = () => onPick(m.market); li.onkeydown = e => { if (e.key === "Enter") onPick(m.market); }; }
    ul.appendChild(li);
  }
  if (byMarket.length > n) {
    const li = document.createElement("li"); li.className = "more";
    li.textContent = `and ${byMarket.length - n} more ${byMarket.length - n === 1 ? "market" : "markets"}`;
    ul.appendChild(li);
  }
}
