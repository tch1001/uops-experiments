const fs = require("fs");

const XML_PATH = "instructions.xml";
const PCORE_CSV = "research_cpu0_pcore.csv";
const ECORE_CSV = "research_cpu16_ecore.csv";

const refs = {
  "ADD_R64_I32": "ADD_R64_I32",
  "ADD_R64_0": "ADD_R64_0",
  "ADD_R64_R64": "ADD_01_R64_R64",
  "IMUL_R64_R64_I32": "IMUL_R64_R64_I8",
  "SHL_R64_1": "SHL_R64_1",
  "ROR_R64_7": "ROR_R64_I8",
  "POPCNT_R64_R64": "POPCNT_R64_R64",
  "LZCNT_R64_R64": "LZCNT_R64_R64",
  "CRC32_R64_R64": "CRC32_R64_R64",
  "ADDSD_XMM_XMM": "ADDSD_XMM_XMM",
  "MULSD_XMM_XMM": "MULSD_XMM_XMM",
  "DIVSD_XMM_XMM": "DIVSD_XMM_XMM",
  "SQRTSD_XMM_XMM": "SQRTSD_XMM_XMM",
  "MOV_R64_M64": "MOV_R64_M64",
};

const latencyChoice = {
  "ADD_R64_I32": [[1, 1]],
  "ADD_R64_0": [[1, 1]],
  "ADD_R64_R64": [[1, 1]],
  "IMUL_R64_R64_I32": [[2, 1]],
  "SHL_R64_1": [[1, 1]],
  "ROR_R64_7": [[1, 1]],
  "POPCNT_R64_R64": [[2, 1]],
  "LZCNT_R64_R64": [[2, 1]],
  "CRC32_R64_R64": [[1, 1], [2, 1]],
  "ADDSD_XMM_XMM": [[1, 1]],
  "MULSD_XMM_XMM": [[1, 1]],
  "DIVSD_XMM_XMM": [[1, 1]],
  "SQRTSD_XMM_XMM": [[2, 1]],
  "MOV_R64_M64": [[2, 1]],
};

function attrs(s) {
  const out = {};
  for (const m of s.matchAll(/([\w.-]+)="([^"]*)"/g)) out[m[1]] = m[2];
  return out;
}

function csvParseLine(line) {
  const out = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (quoted) {
      if (c === '"' && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else if (c === '"') {
        quoted = false;
      } else {
        cur += c;
      }
    } else if (c === '"') {
      quoted = true;
    } else if (c === ",") {
      out.push(cur);
      cur = "";
    } else {
      cur += c;
    }
  }
  out.push(cur);
  return out;
}

function readCsv(path) {
  const lines = fs.readFileSync(path, "utf8").trim().split(/\r?\n/);
  const header = csvParseLine(lines[0]);
  return lines.slice(1).map((line) => {
    const fields = csvParseLine(line);
    const row = {};
    header.forEach((h, i) => row[h] = fields[i]);
    row.median_primary = Number(row.median_primary);
    row.min_primary = Number(row.min_primary);
    row.p90_primary = Number(row.p90_primary);
    return row;
  });
}

function localKey(name) {
  for (const key of Object.keys(refs).sort((a, b) => b.length - a.length)) {
    if (name.includes(key)) return key;
  }
  return null;
}

function median(xs) {
  const ys = xs.slice().sort((a, b) => a - b);
  const n = ys.length;
  return n % 2 ? ys[(n - 1) / 2] : (ys[n / 2 - 1] + ys[n / 2]) / 2;
}

function instructionBlock(xml, urlStem) {
  const needle = `url="uops.info/html-instr/${urlStem}.html"`;
  const pos = xml.indexOf(needle);
  if (pos < 0) return null;
  const start = xml.lastIndexOf("<instruction ", pos);
  const end = xml.indexOf("</instruction>", pos);
  if (start < 0 || end < 0) return null;
  return xml.slice(start, end + "</instruction>".length);
}

function archMeasurement(block, arch) {
  if (!block) return null;
  const archRe = new RegExp(`<architecture name="${arch}">([\\s\\S]*?)</architecture>`);
  const archMatch = block.match(archRe);
  if (!archMatch) return null;
  const body = archMatch[1];
  const measMatch = body.match(/<measurement\b([^>]*)>([\s\S]*?)<\/measurement>/);
  if (!measMatch) return null;
  const a = attrs(measMatch[1]);
  const latencies = [];
  for (const m of measMatch[2].matchAll(/<latency\b([^/>]*)\/>/g)) {
    const la = attrs(m[1]);
    latencies.push({
      start: Number(la.start_op),
      target: Number(la.target_op),
      cycles: latencyCycles(la),
      raw: la.cycles ?? rangeText(la.min_cycles, la.max_cycles) ?? la.cycles_addr ?? rangeText(la.min_cycles_addr, la.max_cycles_addr),
    });
  }
  return {
    TP_unrolled: num(a.TP_unrolled),
    TP_loop: num(a.TP_loop),
    TP_ports: num(a.TP_ports),
    uops: num(a.uops),
    ports: a.ports || "",
    latencies,
  };
}

function num(x) {
  if (x === undefined) return null;
  const n = Number(String(x).replace(/[^\d.]/g, ""));
  return Number.isFinite(n) ? n : null;
}

function rangeText(lo, hi) {
  if (lo === undefined || hi === undefined) return null;
  return `${lo}-${hi}`;
}

function mid(lo, hi) {
  const a = num(lo);
  const b = num(hi);
  if (a == null || b == null) return null;
  return (a + b) / 2;
}

function latencyCycles(attrs) {
  return num(attrs.cycles)
    ?? mid(attrs.min_cycles, attrs.max_cycles)
    ?? num(attrs.min_cycles)
    ?? num(attrs.max_cycles)
    ?? num(attrs.cycles_addr)
    ?? mid(attrs.min_cycles_addr, attrs.max_cycles_addr)
    ?? num(attrs.min_cycles_addr)
    ?? num(attrs.max_cycles_addr);
}

function chooseLatency(key, measurement) {
  if (!measurement) return null;
  const pairs = latencyChoice[key] || [];
  const values = [];
  for (const [start, target] of pairs) {
    const l = measurement.latencies.find((x) => x.start === start && x.target === target);
    if (l && Number.isFinite(l.cycles)) values.push(l.cycles);
  }
  if (!values.length) return null;
  return Math.max(...values);
}

function chooseThroughput(measurement) {
  if (!measurement) return null;
  return measurement.TP_unrolled ?? measurement.TP_loop ?? measurement.TP_ports;
}

function addComparisons(rows, arch, scale, xml) {
  const out = [];
  for (const row of rows) {
    const key = localKey(row.benchmark);
    if (!key || !refs[key]) continue;
    const block = instructionBlock(xml, refs[key]);
    const m = archMeasurement(block, arch);
    const measuredCycles = row.median_primary / scale;
    const refValue = row.kind === "latency" ? chooseLatency(key, m) : chooseThroughput(m);
    out.push({
      kind: row.kind,
      benchmark: row.benchmark,
      uops_stem: refs[key],
      measured_primary: row.median_primary,
      measured_cycles: measuredCycles,
      uops_ref: refValue,
      delta: refValue == null ? null : measuredCycles - refValue,
      uops: m?.uops ?? null,
      ports: m?.ports ?? "",
      tp_unrolled: m?.TP_unrolled ?? null,
      tp_loop: m?.TP_loop ?? null,
      tp_ports: m?.TP_ports ?? null,
    });
  }
  return out;
}

function writeCsv(path, rows) {
  const header = ["kind", "benchmark", "uops_stem", "measured_primary", "measured_cycles", "uops_ref", "delta", "uops", "ports", "tp_unrolled", "tp_loop", "tp_ports"];
  const lines = [header.join(",")];
  for (const r of rows) {
    lines.push(header.map((h) => {
      const v = r[h];
      if (v == null) return "";
      if (typeof v === "number") return Number.isFinite(v) ? v.toFixed(6) : "";
      return `"${String(v).replace(/"/g, '""')}"`;
    }).join(","));
  }
  fs.writeFileSync(path, lines.join("\n") + "\n");
}

function mkMarkdown(pRows, eRows, pScale, eScale) {
  const selected = ["ADD_R64_R64", "IMUL_R64_R64_I32", "POPCNT_R64_R64", "ADDSD_XMM_XMM", "MULSD_XMM_XMM", "DIVSD_XMM_XMM", "SQRTSD_XMM_XMM"];
  const lines = [];
  lines.push("# Extended uops.info-style comparison");
  lines.push("");
  lines.push("uops.info current XML date: 2026-03-29. Direct XML search found no Raptor Lake entry, so this compares the local Raptor Lake-HX P-core against Alder Lake-P and the local E-core against Alder Lake-E.");
  lines.push("");
  lines.push(`Normalization: P-core ${pScale.toFixed(6)} primary units ~= 1 cycle; E-core ${eScale.toFixed(6)} primary units ~= 1 cycle. The scale uses the median of local one-cycle dependency probes: ADD reg/reg, SHL r64,1, and ROR r64,7.`);
  lines.push("");
  lines.push("When uops.info reports latency as a min/max range rather than a single cycle count, the reference table below shows the midpoint. That mainly affects divide and square-root instructions.");
  lines.push("");
  lines.push("## Selected Latency Comparison");
  lines.push("");
  lines.push("| benchmark | local P est | ADL-P ref | local E est | ADL-E ref |");
  lines.push("| --- | ---: | ---: | ---: | ---: |");
  for (const key of selected) {
    const p = pRows.find((r) => r.kind === "latency" && r.benchmark.includes(key));
    const e = eRows.find((r) => r.kind === "latency" && r.benchmark.includes(key));
    if (!p || !e) continue;
    lines.push(`| ${key} | ${fmt(p.measured_cycles)} | ${fmt(p.uops_ref)} | ${fmt(e.measured_cycles)} | ${fmt(e.uops_ref)} |`);
  }
  lines.push("");
  lines.push("## Selected Throughput Comparison");
  lines.push("");
  lines.push("| benchmark | local P est | ADL-P ref | local E est | ADL-E ref |");
  lines.push("| --- | ---: | ---: | ---: | ---: |");
  for (const key of selected) {
    const p = pRows.find((r) => r.kind === "throughput" && r.benchmark.includes(key));
    const e = eRows.find((r) => r.kind === "throughput" && r.benchmark.includes(key));
    if (!p || !e) continue;
    lines.push(`| ${key} | ${fmt(p.measured_cycles)} | ${fmt(p.uops_ref)} | ${fmt(e.measured_cycles)} | ${fmt(e.uops_ref)} |`);
  }
  lines.push("");
  lines.push("Full machine-readable comparison is in `comparison_pcore_adlp.csv` and `comparison_ecore_adle.csv`.");
  lines.push("");
  lines.push("Important limitation: local values are normalized from Windows timing units because user-mode RDPMC is blocked. uops.info values are hardware-counter-backed measurements, so close agreement is more meaningful than small deltas.");
  lines.push("");
  return lines.join("\n");
}

function fmt(x) {
  return x == null ? "" : x.toFixed(2);
}

const xml = fs.readFileSync(XML_PATH, "utf8");
const pRowsRaw = readCsv(PCORE_CSV);
const eRowsRaw = readCsv(ECORE_CSV);

function scaleFrom(rows) {
  const probes = [
    "ADD_R64_R64 dest->dest",
    "SHL_R64_1 dest->dest",
    "ROR_R64_7 dest->dest",
  ];
  const values = probes.map((name) => rows.find((r) => r.kind === "latency" && r.benchmark.includes(name))?.median_primary).filter(Number.isFinite);
  return median(values);
}

const pScale = scaleFrom(pRowsRaw);
const eScale = scaleFrom(eRowsRaw);
const pRows = addComparisons(pRowsRaw, "ADL-P", pScale, xml);
const eRows = addComparisons(eRowsRaw, "ADL-E", eScale, xml);

writeCsv("comparison_pcore_adlp.csv", pRows);
writeCsv("comparison_ecore_adle.csv", eRows);
fs.writeFileSync("EXTENDED_RESULTS.md", mkMarkdown(pRows, eRows, pScale, eScale));

console.log(`p_scale=${pScale}`);
console.log(`e_scale=${eScale}`);
console.log(`p_rows=${pRows.length}`);
console.log(`e_rows=${eRows.length}`);
