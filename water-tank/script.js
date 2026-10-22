"use strict";

/** Water above each column: min(tallest to the left, tallest to the right) - height, floored at 0. */
function waterPerColumn(h) {
  const n = h.length;
  const left = new Array(n), right = new Array(n);
  let m = 0;
  for (let i = 0; i < n; i++) { m = Math.max(m, h[i]); left[i] = m; }
  m = 0;
  for (let i = n - 1; i >= 0; i--) { m = Math.max(m, h[i]); right[i] = m; }
  return h.map((v, i) => Math.max(0, Math.min(left[i], right[i]) - v));
}

function parseHeights(text) {
  const parts = text.replace(/[\[\]]/g, "").split(",").map(s => s.trim()).filter(s => s !== "");
  if (!parts.length) throw new Error("Enter at least one height.");
  return parts.map(p => {
    if (!/^\d+$/.test(p)) throw new Error(`"${p}" is not a whole number of 0 or more.`);
    return Number(p);
  });
}

const NS = "http://www.w3.org/2000/svg";
function rect(x, y, size, cls) {
  const r = document.createElementNS(NS, "rect");
  r.setAttribute("x", x); r.setAttribute("y", y);
  r.setAttribute("width", size); r.setAttribute("height", size);
  r.setAttribute("class", cls);
  return r;
}

function render(heights) {
  const water = waterPerColumn(heights);
  const rows = Math.max(1, ...heights.map((v, i) => v + water[i]));
  const cell = 36, w = heights.length * cell, hgt = rows * cell;
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${w} ${hgt}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `Tank with ${water.reduce((a, b) => a + b, 0)} units of water`);
  const style = document.createElementNS(NS, "style");
  style.textContent = ".b{fill:var(--block)}.w{fill:var(--water)}.g{fill:none;stroke:var(--grid);stroke-width:1}";
  svg.appendChild(style);
  heights.forEach((v, i) => {
    for (let r = 0; r < rows; r++) {
      const y = hgt - (r + 1) * cell; // r = 0 is the bottom row
      const cls = r < v ? "b" : r < v + water[i] ? "w" : "g";
      svg.appendChild(rect(i * cell, y, cell, cls));
      if (cls !== "g") {
        svg.appendChild(rect(i * cell, y, cell, "g")); // grid outline on filled cells
      }
    }
  });
  document.getElementById("chart").replaceChildren(svg);
  document.getElementById("total").textContent = water.reduce((a, b) => a + b, 0);
}

if (typeof document !== "undefined") {
  document.getElementById("form").addEventListener("submit", e => {
    e.preventDefault();
    const err = document.getElementById("error");
    try {
      render(parseHeights(document.getElementById("heights").value));
      err.textContent = "";
    } catch (ex) {
      err.textContent = ex.message;
    }
  });

  render(parseHeights(document.getElementById("heights").value));
}

if (typeof module !== "undefined") module.exports = { waterPerColumn, parseHeights };