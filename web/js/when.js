// "When they stay": the change in guest nights by month with its 80% range; months more than seven months beyond the
// data are hatched (the normal year is less certain there). After a confirmed result the bars grow in once, left to
// right (0.6 s), so the eye follows the change into the calendar; never while dragging, never with reduced motion.
// A pinned scenario A is drawn as a short dark tick on each month's bar. The by-day view (drawDaily) shows the hotel
// load against the highest level ever recorded, and the change in guests each night.
const d3 = window.d3;
const still = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export function drawMonths(svgEl, months, { sweep = false, ghost = null } = {}) {
  const W = svgEl.getBoundingClientRect().width, H = 300, m = { t: 14, r: 6, b: 42, l: 68 };
  if (!W) return;
  const svg = d3.select(svgEl).attr("viewBox", `0 0 ${W} ${H}`);
  svg.selectAll("*").remove();
  const css = getComputedStyle(document.documentElement);
  const gv = ghost ? ghost.map(g => g.d_guest_nights) : [];
  let lo = Math.min(d3.min(months, d => Math.min(0, d.lo, d.d_guest_nights)), ...gv), hi = Math.max(d3.max(months, d => Math.max(0, d.hi, d.d_guest_nights)), ...gv);
  if (lo === 0 && hi === 0) hi = 1000;
  const y = d3.scaleLinear().domain([lo, hi]).nice(5);                 // left margin fixed (room for "−600,000"), so bars never shift sideways
  const big = Math.max(Math.abs(lo), Math.abs(hi)) >= 1e6;
  const fmt = t => (big ? d3.format(".2~s")(t).replace("G", "B") : d3.format(",")(t)).replace("-", "−");
  y.range([H - m.b, m.t]);
  const x = d3.scaleBand().domain(months.map(d => d.month)).range([m.l, W - m.r]).padding(0.28);
  const pat = svg.append("defs").append("pattern").attr("id", "hatch").attr("patternUnits", "userSpaceOnUse").attr("width", 5).attr("height", 5)
    .attr("patternTransform", "rotate(45)");
  pat.append("line").attr("x1", 0).attr("y1", 0).attr("x2", 0).attr("y2", 5).attr("stroke", "#F8FAFB").attr("stroke-width", 1.6).attr("stroke-opacity", 0.75);
  for (const t of y.ticks(5)) {
    svg.append("line").attr("class", t === 0 ? "zero" : "grid").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(t)).attr("y2", y(t));
    svg.append("text").attr("class", "tick").attr("x", m.l - 8).attr("y", y(t) + 4).attr("text-anchor", "end").text(fmt(t));
  }
  const bw = Math.min(24, x.bandwidth()), off = (x.bandwidth() - bw) / 2, r = 4;
  const bar = (d, v = d.d_guest_nights) => {                        // 4px rounded data end, square at the baseline
    const x0 = x(d.month) + off, y0 = y(0), y1 = y(v), up = y1 < y0, hgt = Math.abs(y1 - y0), rr = Math.min(r, hgt);
    if (hgt < 0.01) return `M${x0},${y0}H${x0 + bw}`;
    return up ? `M${x0},${y0}V${y1 + rr}Q${x0},${y1} ${x0 + rr},${y1}H${x0 + bw - rr}Q${x0 + bw},${y1} ${x0 + bw},${y1 + rr}V${y0}Z`
              : `M${x0},${y0}V${y1 - rr}Q${x0},${y1} ${x0 + rr},${y1}H${x0 + bw - rr}Q${x0 + bw},${y1} ${x0 + bw},${y1 - rr}V${y0}Z`;
  };
  const shown = months.filter(d => Math.abs(d.d_guest_nights) >= 0.5);
  const anim = sweep && !still();
  const grow = sel => anim ? sel.attr("d", d => bar(d, 0)).transition().delay((d, i) => i * 30).duration(260).ease(d3.easeCubicOut).attr("d", d => bar(d)) : sel.attr("d", d => bar(d));
  grow(svg.append("g").selectAll("path").data(shown).join("path")
    .attr("fill", d => css.getPropertyValue(d.d_guest_nights < 0 ? "--loss" : "--gain").trim()));
  grow(svg.append("g").selectAll("path").data(shown.filter(d => d.beyond_validation)).join("path").attr("fill", "url(#hatch)"));
  const wh = svg.append("g").selectAll("line").data(shown).join("line").attr("class", "whisker")
    .attr("x1", d => x(d.month) + x.bandwidth() / 2).attr("x2", d => x(d.month) + x.bandwidth() / 2)
    .attr("y1", d => y(d.lo)).attr("y2", d => y(d.hi));
  if (anim) wh.attr("opacity", 0).transition().delay((d, i) => 260 + i * 30).duration(120).attr("opacity", 1);
  svg.append("g").selectAll("rect").data(shown).join("rect").attr("x", d => x(d.month)).attr("width", x.bandwidth())
    .attr("y", m.t).attr("height", H - m.b - m.t).attr("fill", "transparent")
    .append("title").text(d => `${d.label}: ${d3.format("+,")(Math.round(d.d_guest_nights)).replace("-", "−")} guest nights (80% range ${d3.format("+,")(Math.round(d.lo)).replace("-", "−")} to ${d3.format("+,")(Math.round(d.hi)).replace("-", "−")})`);
  if (ghost) {                                                       // scenario A: a dark tick across each month's bar
    const g = svg.append("g").selectAll("g").data(ghost.filter(d => x(d.month) !== undefined)).join("g");
    g.append("line").attr("x1", d => x(d.month) + off - 3).attr("x2", d => x(d.month) + off + bw + 3).attr("y1", d => y(d.d_guest_nights)).attr("y2", d => y(d.d_guest_nights))
      .attr("stroke", "#F8FAFB").attr("stroke-width", 5).attr("stroke-linecap", "round");
    g.append("line").attr("x1", d => x(d.month) + off - 3).attr("x2", d => x(d.month) + off + bw + 3).attr("y1", d => y(d.d_guest_nights)).attr("y2", d => y(d.d_guest_nights))
      .attr("stroke", "#15222A").attr("stroke-width", 2.5).attr("stroke-linecap", "round");
  }
  const ticks = svg.append("g").selectAll("text.tick-m").data(months).join("text").attr("class", "tick tick-m")
    .attr("x", d => x(d.month) + x.bandwidth() / 2).attr("y", H - 24).attr("text-anchor", "middle");
  ticks.append("tspan").text(d => d.label.split(" ")[0]);
  ticks.filter((d, i) => i === 0 || d.label.startsWith("Jan")).append("tspan")
    .attr("x", d => x(d.month) + x.bandwidth() / 2).attr("dy", 16).attr("font-weight", 600).text(d => d.label.split(" ")[1]);
}

// By day: the hotel guests each night (normal year, your scenario, the highest ever recorded) and, below, the change.
// Hovering reads one night. The scenario's daily numbers come with a confirmed run, not while dragging.
export function drawDaily(svgEl, daily, loss) {
  const W = svgEl.getBoundingClientRect().width, H = 300, m = { t: 14, r: 6, b: 34, l: 68 }, topH = 140, botT = m.t + topH + 30, botH = 78;
  if (!W) return;
  const svg = d3.select(svgEl).attr("viewBox", `0 0 ${W} ${H}`); svg.selectAll("*").remove();
  const css = getComputedStyle(document.documentElement), col = css.getPropertyValue(loss ? "--loss" : "--gain").trim();
  const dates = daily.dates.map(s => new Date(s)), base = daily.baseline, scen = daily.scenario, change = scen.map((v, i) => v - base[i]);
  const x = d3.scaleUtc().domain([dates[0], dates[dates.length - 1]]).range([m.l, W - m.r]);
  const hi = Math.max(daily.peak, d3.max(scen), d3.max(base)), lo = Math.min(d3.min(scen), d3.min(base));
  const y = d3.scaleLinear().domain([Math.floor(lo * 0.92 / 5000) * 5000, hi * 1.04]).nice(3).range([m.t + topH, m.t]);
  const cl = Math.max(1, d3.max(change.map(Math.abs))), z = d3.scaleLinear().domain([-cl, cl]).nice(2).range([botT + botH, botT]);
  const fmt = v => d3.format(",")(v).replace("-", "−");
  for (const t of y.ticks(3)) {
    svg.append("line").attr("class", "grid").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(t)).attr("y2", y(t));
    svg.append("text").attr("class", "tick").attr("x", m.l - 8).attr("y", y(t) + 4).attr("text-anchor", "end").text(fmt(t));
  }
  const line = (arr, c, w) => svg.append("path").attr("fill", "none").attr("stroke", c).attr("stroke-width", w).attr("stroke-linejoin", "round")
    .attr("d", d3.line().x((_, i) => x(dates[i])).y(v => y(v))(arr));
  line(base, "#8796A0", 1.4); line(scen, col, 2);
  svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(daily.peak)).attr("y2", y(daily.peak)).attr("stroke", "#15222A").attr("stroke-width", 1.4).attr("stroke-dasharray", "6 4");
  svg.append("text").attr("class", "tick rec").attr("x", W - m.r).attr("y", y(daily.peak) - 6).attr("text-anchor", "end")
    .text(`Highest ever recorded: ${fmt(daily.peak)} (${d3.utcFormat("%-d %b %Y")(new Date(daily.peak_date))})`);
  for (const t of z.ticks(2)) {
    svg.append("line").attr("class", t === 0 ? "zero" : "grid").attr("x1", m.l).attr("x2", W - m.r).attr("y1", z(t)).attr("y2", z(t));
    svg.append("text").attr("class", "tick").attr("x", m.l - 8).attr("y", z(t) + 4).attr("text-anchor", "end").text(fmt(t));
  }
  const bw = Math.max(1, (W - m.l - m.r) / dates.length - 0.3);
  svg.append("g").selectAll("rect").data(change).join("rect").attr("x", (_, i) => x(dates[i]) - bw / 2).attr("width", bw)
    .attr("y", v => (v >= 0 ? z(v) : z(0))).attr("height", v => Math.abs(z(v) - z(0)))
    .attr("fill", v => css.getPropertyValue(v < 0 ? "--loss" : "--gain").trim());
  svg.append("text").attr("class", "tick").attr("x", m.l).attr("y", botT - 8).text("Change in guests each night");
  const lab = svg.append("text").attr("class", "tick").attr("x", m.l).attr("y", m.t - 3).text("Guests in hotels each night");
  const ticks = d3.utcMonths(dates[0], d3.utcDay.offset(dates[dates.length - 1], 1));
  const tk = svg.append("g").selectAll("text").data(ticks).join("text").attr("class", "tick").attr("x", d => x(d) + 2).attr("y", H - 16);
  tk.append("tspan").text(d => d3.utcFormat("%b")(d));
  tk.filter((d, i) => i === 0 || d.getUTCMonth() === 0).append("tspan").attr("x", d => x(d) + 2).attr("dy", 14).attr("font-weight", 600).text(d => d.getUTCFullYear());
  // hover: one night
  const rule = svg.append("line").attr("y1", m.t).attr("y2", botT + botH).attr("stroke", "#15222A").attr("stroke-opacity", 0.5).attr("opacity", 0);
  const read = svg.append("text").attr("class", "tick read").attr("x", m.l).attr("y", m.t - 3).attr("opacity", 0);
  svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.l - m.r).attr("height", botT + botH - m.t).attr("fill", "transparent")
    .on("mousemove", ev => {
      const i = Math.max(0, Math.min(dates.length - 1, d3.bisectCenter(dates, x.invert(d3.pointer(ev)[0]))));
      rule.attr("x1", x(dates[i])).attr("x2", x(dates[i])).attr("opacity", 1); lab.attr("opacity", 0);
      const c = Math.round(change[i]);
      read.attr("opacity", 1).text(`${d3.utcFormat("%a %-d %b %Y")(dates[i])}: ${fmt(Math.round(scen[i]))} guests (${c === 0 ? "no change" : (c > 0 ? "+" : "−") + fmt(Math.abs(c))} vs normal year)`);
    }).on("mouseleave", () => { rule.attr("opacity", 0); read.attr("opacity", 0); lab.attr("opacity", 1); });
}
