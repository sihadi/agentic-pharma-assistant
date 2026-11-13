"use strict";
/* PharmaLens front-end — vanilla JS, no dependencies.
 * Streams POST /ask/stream (Server-Sent Events) to animate the agent pipeline
 * live, then renders the cited answer, grounding metrics, evidence, critic
 * review, plan/tools and trace. History is kept in localStorage. */

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const store = {
  get(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage full or blocked */ } },
};
const HISTORY_KEY = "pharmalens.history";

// ------------------------------------------------------------------ static content
const ICONS = {
  plan: '<path d="M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01"/>',
  retrieve: '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',
  tools: '<path d="M14.7 6.3a4 4 0 00-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 005.4-5.4l-2.5 2.5-2.4-.6-.6-2.4z"/>',
  draft: '<path d="M12 20h9M16.5 3.5a2.1 2.1 0 013 3L7 19l-4 1 1-4z"/>',
  critic: '<path d="M12 3l8 4v5c0 5-3.4 8.4-8 9-4.6-.6-8-4-8-9V7z"/><path d="M9 12l2 2 4-4"/>',
  answer: '<path d="M4 12l5 5L20 6"/>',
};
const STEPS = [
  ["plan", "Plan"], ["retrieve", "Retrieve"], ["tools", "Tools"],
  ["draft", "Draft"], ["critic", "Critic"], ["answer", "Answer"],
];
const EXAMPLES = [
  { kind: "Lookup", color: "#2563eb", icon: ICONS.retrieve,
    q: "What are the main contraindications for warfarin?" },
  { kind: "Comparison", color: "#7c3aed", icon: '<path d="M8 3v18M16 3v18M3 8h5M16 16h5"/>',
    q: "Compare the contraindications of warfarin and apixaban. What conditions are shared and which are unique to each drug?" },
  { kind: "Interaction", color: "#db2777", icon: '<circle cx="9" cy="12" r="5"/><circle cx="15" cy="12" r="5"/>',
    q: "A patient currently on warfarin is prescribed sertraline. What is the risk of bleeding and what monitoring is recommended?" },
  { kind: "Threshold", color: "#d97706", icon: '<path d="M3 17l6-6 4 4 8-8"/><path d="M14 7h7v7"/>',
    q: "A patient with an eGFR of 25 mL/min is being considered for metformin. Is metformin contraindicated at this level of renal function, and what is the threshold?" },
  { kind: "Brand names", color: "#0d9488", icon: '<path d="M20 12l-8 8-9-9V3h8z"/><circle cx="7.5" cy="7.5" r="1"/>',
    q: "Can a patient on Coumadin take Advil for a headache? What do the labels say about bleeding?" },
  { kind: "Out of scope", color: "#64748b", icon: '<circle cx="12" cy="12" r="9"/><path d="M5.6 5.6l12.8 12.8"/>',
    q: "What are the contraindications of lisinopril?" },
];
const STATUS = {
  answered: ["ok", "Answered"],
  answered_with_caveats: ["warn", "Answered with caveats"],
  insufficient_evidence: ["warn", "Insufficient evidence"],
  out_of_scope: ["bad", "Out of scope"],
};
const CLAIM_STATUS = { supported: "ok", ambiguous: "warn", unsupported: "bad", misattributed: "bad" };

// ------------------------------------------------------------------ helpers
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const svg = (paths) => `<svg viewBox="0 0 24 24">${paths}</svg>`;
const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
const secs = (ms) => `${(ms / 1000).toFixed(1)} s`;
const ago = (ts) => {
  const m = Math.round((Date.now() - ts) / 60000);
  return m < 1 ? "just now" : m < 60 ? `${m} min ago` : m < 1440 ? `${Math.round(m / 60)} h ago` : new Date(ts).toLocaleDateString();
};
const fmtArgs = (a) => Object.entries(a || {}).filter(([, v]) => v !== null && v !== undefined && v !== "")
  .map(([k, v]) => `${k}=${Array.isArray(v) ? v.join("+") : v}`).join(", ");

function toast(message) {
  const t = $("#toast");
  t.textContent = message;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { t.hidden = true; }, 2200);
}

// Small, safe Markdown subset (HTML is escaped first).
function inline(text) {
  return esc(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*(?!\s)(.+?)\*(?!\*)/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[(\d+)\]/g, '<a class="cite" href="#ev-$1" data-ref="$1">$1</a>');
}
function markdown(md) {
  const lines = md.replace(/\r/g, "").split("\n");
  const out = [];
  const isList = (l) => /^\s*([-*+]|\d+[.)])\s+/.test(l);
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    let m;
    if ((m = line.match(/^(#{1,4})\s+(.*)/))) { out.push(`<h4>${inline(m[2])}</h4>`); i++; continue; }
    if (line.trim().startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(lines[i++].trim());
      const cells = (r) => r.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const [head, ...rest] = rows.filter((r) => !/^\|?[\s:\-|]+\|?$/.test(r));
      out.push(`<table><thead><tr>${cells(head).map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${
        rest.map((r) => `<tr>${cells(r).map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
      continue;
    }
    if (isList(line)) {
      const ordered = /^\s*\d+[.)]/.test(line);
      const items = [];
      while (i < lines.length && isList(lines[i])) items.push(`<li>${inline(lines[i++].replace(/^\s*([-*+]|\d+[.)])\s+/, ""))}</li>`);
      out.push(ordered ? `<ol>${items.join("")}</ol>` : `<ul>${items.join("")}</ul>`);
      continue;
    }
    if (line.startsWith(">")) {
      const q = [];
      while (i < lines.length && lines[i].startsWith(">")) q.push(lines[i++].replace(/^>\s?/, ""));
      out.push(`<blockquote>${inline(q.join(" "))}</blockquote>`);
      continue;
    }
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|\||>)/.test(lines[i]) && !isList(lines[i])) para.push(lines[i++]);
    out.push(`<p>${inline(para.join(" "))}</p>`);
  }
  return out.join("");
}

// ------------------------------------------------------------------ theme & layout
$("#theme").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("pharmalens.theme", next); } catch { /* ignore */ }
});
const openSidebar = (open) => { $("#sidebar").classList.toggle("open", open); $("#scrim").hidden = !open; };
$("#menu").addEventListener("click", () => openSidebar(true));
$("#scrim").addEventListener("click", () => openSidebar(false));

function showWelcome() {
  $("#welcome").hidden = false;
  $("#run").hidden = true;
  $("#crumb").textContent = "New research";
  $$(".history button").forEach((b) => b.classList.remove("active"));
  $("#question").value = "";
  autoresize();
  $("#question").focus();
  openSidebar(false);
}
$("#new-question").addEventListener("click", showWelcome);

// ------------------------------------------------------------------ health & corpus
async function pollHealth() {
  const dot = $("#status-dot"), text = $("#status-text");
  try {
    const h = await (await fetch("/health")).json();
    if (h.status === "ok") {
      dot.className = "dot ok";
      text.textContent = `Ready · ${h.corpus.documents} documents · ${h.corpus.chunks} passages`;
      $("#model-line").textContent = `${h.config.chat_model} · ${h.config.embedding_model}`;
      loadCorpus();
      return;
    }
    dot.className = h.status === "starting" ? "dot warn" : "dot bad";
    text.textContent = h.status === "starting" ? "Building search index…" : "Degraded";
    $("#model-line").textContent = h.error || "";
  } catch {
    dot.className = "dot bad";
    text.textContent = "API unreachable";
  }
  setTimeout(pollHealth, 2500);
}

async function loadCorpus() {
  try {
    const c = await (await fetch("/corpus")).json();
    const drugs = new Map();
    c.documents.filter((d) => d.drug).forEach((d) => {
      const entry = drugs.get(d.drug) || { aliases: new Set(), files: [] };
      d.aliases.forEach((a) => entry.aliases.add(a));
      entry.files.push(d.source_file);
      drugs.set(d.drug, entry);
    });
    $("#drugs").innerHTML = [...drugs.keys()].sort().map((d) => {
      const e = drugs.get(d);
      const title = `${e.files.join(" · ")}${e.aliases.size ? `\nAlso known as: ${[...e.aliases].slice(0, 6).join(", ")}` : ""}`;
      return `<button class="drug" type="button" data-drug="${esc(d)}" title="${esc(title)}">${esc(d)}</button>`;
    }).join("");
    $("#corpus-note").textContent = `${c.totals.documents} documents · ${c.totals.chunks} passages · ${c.cleaning_rules.length} cleaning rules`;
  } catch { /* coverage panel is optional */ }
}
$("#drugs").addEventListener("click", (e) => {
  const b = e.target.closest(".drug");
  if (!b) return;
  setQuestion(`What are the main contraindications and warnings for ${b.dataset.drug}?`);
  openSidebar(false);
});

// ------------------------------------------------------------------ examples
$("#examples").innerHTML = EXAMPLES.map((ex, i) => `
  <button class="ex-card" type="button" data-i="${i}">
    <span class="ex-kind" style="color:${ex.color}"><i style="background:${ex.color}1f">${svg(ex.icon)}</i>${esc(ex.kind)}</span>
    <span class="ex-q">${esc(ex.q)}</span>
  </button>`).join("");
$("#examples").addEventListener("click", (e) => {
  const card = e.target.closest(".ex-card");
  if (card) ask(EXAMPLES[+card.dataset.i].q);
});

// ------------------------------------------------------------------ composer
const textarea = $("#question");
function autoresize() { textarea.style.height = "auto"; textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`; }
function setQuestion(q) { textarea.value = q; autoresize(); textarea.focus(); textarea.setSelectionRange(q.length, q.length); }
textarea.addEventListener("input", autoresize);
textarea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#ask-form").requestSubmit(); }
});
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement !== textarea) { e.preventDefault(); textarea.focus(); }
});
$("#ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = textarea.value.trim();
  if (q.length < 3) { toast("Please type a question (3+ characters)."); return; }
  ask(q);
});

// ------------------------------------------------------------------ pipeline view
let timing = {};
let runStarted = 0;
let timerHandle = null;

function resetPipeline() {
  timing = {};
  $("#stepper").innerHTML = STEPS.map(([key, label]) => `
    <li class="step" data-step="${key}">
      <div class="step-icon">${svg(ICONS[key])}</div>
      <div class="step-name">${label}</div>
      <div class="step-detail">waiting</div>
      <div class="step-time"></div>
    </li>`).join("");
  $("#activity").innerHTML = "";
  $("#pipeline-card").classList.remove("collapsed");
}
function step(key, state, detail, elapsed) {
  const li = $(`.step[data-step="${key}"]`);
  if (!li) return;
  li.classList.remove("active", "done", "revise");
  if (state) li.classList.add(state);
  if (detail !== undefined) $(".step-detail", li).textContent = detail;
  if (state === "active" && elapsed != null) timing[key] = elapsed;
  if (state === "done" && elapsed != null && timing[key] != null) $(".step-time", li).textContent = secs(elapsed - timing[key]);
  if (state === null) $(".step-time", li).textContent = "";
}
function activity(text, elapsed) {
  const box = $("#activity");
  const row = document.createElement("div");
  row.innerHTML = `<time>+${((elapsed ?? 0) / 1000).toFixed(1)}s</time><span>${esc(text)}</span>`;
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
}

function onStep(e) {
  const t = e.elapsed_ms;
  switch (e.event) {
    case "request_received": step("plan", "active", "decomposing", t); break;
    case "plan_created":
      step("plan", "done", `${(e.intent || "").replaceAll("_", " ")} · ${(e.sub_questions || []).length} sub-q`, t);
      (e.sub_questions || []).forEach((s, i) => activity(`plan ${i + 1}: ${s.question}`, t));
      step("retrieve", "active", "hybrid search", t);
      break;
    case "retrieval_completed":
      step("retrieve", "done", `${e.unique_chunks} passages`, t);
      activity(`retrieve: ${e.queries.length} queries → ${e.unique_chunks} unique passages (FAISS + BM25, RRF)`, t);
      step("tools", "active", "agent deciding", t);
      break;
    case "tool_selected":
      step("tools", "active", e.calls.map((c) => c.tool.replaceAll("_", " ")).join(", "));
      e.calls.forEach((c) => activity(`agent → ${c.tool}(${fmtArgs(c.arguments)})`, t));
      break;
    case "tool_executed":
      activity(`   ${e.tool}: ${e.status}${e.evidence_ids ? ` · ${e.evidence_ids.length} passages` : ""}`, t);
      break;
    case "agent_ready": case "agent_stopped":
      step("tools", "done", undefined, t);
      if (e.note) activity(`agent: ${e.note}`, t);
      step("draft", "active", "writing with citations", t);
      break;
    case "draft_created":
      step("draft", "done", `v${e.version} · ${e.coverage} coverage`, t);
      step("critic", "active", "verifying claims", t);
      break;
    case "critic_completed":
      step("critic", "done", `${e.verdict} · ${pct(e.confidence)}`, t);
      activity(`critic: ${e.verdict}${e.vetoes?.length ? ` (audit veto: ${e.vetoes.join(", ")})` : ""} · ${(e.flagged_claims || []).length} flagged claim(s)`, t);
      step("answer", "active", "finalizing", t);
      break;
    case "revision_requested":
      ["draft", "critic", "answer"].forEach((k) => step(k, null, "waiting"));
      step("tools", "revise", `revision ${e.revision}`);
      step("tools", "active", `revision ${e.revision}`, t);
      activity(`revision ${e.revision}: ${e.instructions}`, t);
      break;
    case "answer_emitted": step("answer", "done", (e.answer_status || "").replaceAll("_", " "), t); break;
  }
}

// ------------------------------------------------------------------ ask
async function streamAsk(question) {
  const res = await fetch("/ask/stream", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }),
  });
  if (!res.ok || !res.body) {
    const payload = await res.json().catch(() => ({}));
    const detail = Array.isArray(payload.detail) ? payload.detail.map((d) => d.msg).join("; ") : payload.detail;
    throw new Error(detail || `Request failed (HTTP ${res.status})`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "", result = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const event = (block.match(/^event: (.*)$/m) || [])[1];
      const data = JSON.parse((block.match(/^data: (.*)$/m) || [, "null"])[1]);
      if (event === "step") onStep(data);
      else if (event === "result") result = data;
      else if (event === "error") throw new Error(`${data.detail} — trace ${data.trace_id}`);
    }
  }
  if (!result) throw new Error("The stream ended without a result.");
  return result;
}

function showRun(question, meta) {
  $("#welcome").hidden = true;
  $("#run").hidden = false;
  $("#q-text").textContent = question;
  $("#q-meta").textContent = meta;
  $("#crumb").textContent = question;
  $("#error").hidden = true;
  $("#content").scrollTop = 0;
}

async function ask(question) {
  const button = $("#submit");
  if (button.disabled) return;
  showRun(question, "Researching the local corpus…");
  resetPipeline();
  $("#report").hidden = true;
  $("#skeleton").hidden = false;
  textarea.value = "";
  autoresize();
  button.disabled = true;
  button.classList.add("loading");
  runStarted = performance.now();
  timerHandle = setInterval(() => { $("#timer").textContent = secs(performance.now() - runStarted); }, 100);
  try {
    const result = await streamAsk(question);
    $("#q-meta").textContent = `Answered in ${secs(result.usage.duration_ms)} · trace ${result.trace_id.slice(0, 8)}`;
    renderReport(result);
    saveHistory(question, result);
    $("#pipeline-card").classList.add("collapsed");
  } catch (err) {
    $("#error").textContent = err.message || String(err);
    $("#error").hidden = false;
    $("#q-meta").textContent = "Failed";
    toast("The request failed");
  } finally {
    clearInterval(timerHandle);
    $("#skeleton").hidden = true;
    button.disabled = false;
    button.classList.remove("loading");
  }
}

// ------------------------------------------------------------------ report
let current = null;

function renderReport(r) {
  current = r;
  $("#report").hidden = false;
  const [cls, label] = STATUS[r.status] || ["muted", r.status];
  $("#status-badge").className = `badge ${cls}`;
  $("#status-badge").textContent = label;
  $("#verdict-badge").textContent = `Critic · ${r.review.verdict}`;
  $("#answer").innerHTML = markdown(r.answer_body);
  $("#disclaimer").textContent = r.disclaimer;

  const files = [...new Map(r.citations.map((c) => [c.source_file, c.source_type])).entries()];
  $("#sources").innerHTML = files.length
    ? `<span>Sources</span>${files.map(([f, t]) => `<span class="src-chip"><b class="${esc(t)}">${esc(t)}</b>${esc(f)}</span>`).join("")}`
    : "<span>No source cited</span>";

  // grounding ring + metrics
  const conf = r.review.confidence;
  const ring = $("#ring");
  ring.style.strokeDashoffset = conf == null ? 213.6 : 213.6 * (1 - conf);
  ring.style.stroke = conf == null ? "var(--line)" : conf >= 0.85 ? "var(--ok)" : conf >= 0.6 ? "var(--warn)" : "var(--bad)";
  $("#ring-value").textContent = pct(conf);
  const audit = r.review.audit || {};
  const numeric = audit.numeric_issues;
  const rows = [
    ["Claims verified", r.review.claims_checked != null ? `${r.review.claims_supported}/${r.review.claims_checked}` : "–",
      r.review.claims_checked && r.review.claims_supported === r.review.claims_checked ? "good" : "warn"],
    ["Citation coverage", pct(audit.citation_coverage), audit.citation_coverage >= 0.9 ? "good" : "warn"],
    ["Numeric check", numeric ? (numeric.length ? `${numeric.length} issue(s)` : "passed") : "–", numeric?.length ? "bad" : "good"],
    ["Revisions", r.review.revisions ?? 0, ""],
    null,
    ["Latency", secs(r.usage.duration_ms), ""],
    ["LLM calls", r.usage.llm_calls, ""],
    ["Tokens", (r.usage.prompt_tokens + r.usage.completion_tokens).toLocaleString(), ""],
    ["Passages cited", r.citations.length, ""],
  ];
  $("#metrics").innerHTML = rows.map((row) => row ? `<dt>${row[0]}</dt><dd class="${row[2]}">${esc(row[1])}</dd>` : '<div class="sep"></div>').join("");

  renderEvidence(r);
  renderReview(r);
  renderPlan(r);
  renderTrace(r);
  selectTab("evidence");
}

function renderEvidence(r) {
  $("#count-evidence").textContent = r.citations.length || "";
  $("#evidence").innerHTML = r.citations.length ? r.citations.map((c) => `
    <li class="ev" id="ev-${c.ref}">
      <div class="ev-head">
        <span class="ev-ref">${c.ref}</span>
        <span class="ev-file">${esc(c.source_file)}</span>
        <span class="ev-tag">${esc(c.source_type)}</span>
        <span class="ev-tag">${esc(c.canonical_section.replaceAll("_", " "))}</span>
        <span class="ev-id">${esc(c.chunk_id)}</span>
      </div>
      <p class="ev-section">${esc(c.section)}</p>
      <div class="ev-text">${esc(c.excerpt)}</div>
      <button class="ev-more" type="button">Show full passage</button>
    </li>`).join("") : '<p class="empty-note">No passage was cited for this answer.</p>';
}

function renderReview(r) {
  const rv = r.review, a = rv.audit || {};
  const flagged = rv.flagged_claims || [];
  $("#count-review").textContent = flagged.length || "";
  if (rv.verdict === "N/A") {
    $("#review").innerHTML = '<p class="empty-note">No review: the question was out of the corpus scope, so no claim was produced.</p>' + limitationsHtml(r);
    return;
  }
  const check = (label, value, cls) => `<div class="check ${cls}"><span>${label}</span><strong>${esc(value)}</strong></div>`;
  $("#review").innerHTML = `
    <div class="check-grid">
      ${check("Final verdict", rv.verdict, rv.verdict === "APPROVE" ? "good" : "warn")}
      ${check("LLM critic verdict", rv.llm_verdict, rv.llm_verdict === "APPROVE" ? "good" : "warn")}
      ${check("Citation coverage", pct(a.citation_coverage), a.citation_coverage >= 0.9 ? "good" : "warn")}
      ${check("Numbers grounded", a.numeric_issues?.length ? `${a.numeric_issues.length} issue(s)` : "all", a.numeric_issues?.length ? "bad" : "good")}
      ${check("Invalid citations", a.invalid_citations?.length ?? 0, a.invalid_citations?.length ? "bad" : "good")}
      ${check("Personal advice", rv.individualized_advice ? "detected" : "none", rv.individualized_advice ? "bad" : "good")}
    </div>
    ${rv.vetoes?.length ? `<div class="panel-title">Deterministic audit veto</div><p class="empty-note">${esc(rv.vetoes.join(" · "))}</p>` : ""}
    <div class="panel-title">Flagged claims</div>
    ${flagged.length ? flagged.map((f) => `
      <div class="claim"><span class="badge ${CLAIM_STATUS[f.status] || "muted"}">${esc(f.status)}</span>
        <p>${esc(f.claim)}</p><small>${esc(f.note)}</small></div>`).join("")
      : `<p class="empty-note">All ${rv.claims_checked ?? ""} claims were verified as supported by the cited passages.</p>`}
    ${a.uncited_statements?.length ? `<div class="panel-title">Statements without citation</div><ul class="plan-list">${a.uncited_statements.map((s) => `<li>${esc(s)}</li>`).join("")}</ul>` : ""}
    ${(numeric => numeric?.length ? `<div class="panel-title">Numbers not found in cited passages</div><ul class="plan-list">${numeric.map((n) => `<li><code>${esc(n.number)}</code> — ${esc(n.sentence)}</li>`).join("")}</ul>` : "")(a.numeric_issues)}
    ${limitationsHtml(r)}`;
}
const limitationsHtml = (r) => r.limitations.length
  ? `<div class="panel-title">Limitations</div><ul class="plan-list">${r.limitations.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>` : "";

function renderPlan(r) {
  const p = r.plan;
  $("#plan").innerHTML = `
    <div class="panel-title">Plan · ${esc((p.intent || "").replaceAll("_", " "))}</div>
    ${p.sub_questions?.length ? `<ol class="plan-list">${p.sub_questions.map((s) => `<li>${esc(s.question)}
      <small> — ${esc(s.drug || "general")}${s.sections?.length ? ` · ${esc(s.sections.join(", "))}` : ""}</small></li>`).join("")}</ol>`
      : '<p class="empty-note">No retrieval plan (out of scope).</p>'}
    ${p.synthesis ? `<p class="empty-note"><strong>Synthesis:</strong> ${esc(p.synthesis)}</p>` : ""}
    ${p.unknown_drugs?.length ? `<p class="empty-note"><strong>Not in corpus:</strong> ${esc(p.unknown_drugs.join(", "))}</p>` : ""}
    <div class="panel-title">Tools chosen by the agent</div>
    ${r.tools_used.length ? r.tools_used.map((t) => `
      <div class="tool-row"><code>${esc(t.tool)}</code><span class="args">${esc(fmtArgs(t.arguments))}</span>
      <span class="badge ${t.status === "ok" ? "ok" : t.status === "error" ? "bad" : "warn"}">${esc(t.status.replaceAll("_", " "))}</span></div>`).join("")
      : '<p class="empty-note">No tool call was needed.</p>'}`;
}

function summarize(e) {
  switch (e.event) {
    case "request_received": return e.question;
    case "plan_created": return `${e.intent} · ${(e.sub_questions || []).length} sub-questions · drugs: ${(e.drugs || []).join(", ") || "–"}`;
    case "retrieval_completed": return `${e.queries.length} queries → ${e.unique_chunks} unique passages`;
    case "tool_selected": return e.calls.map((c) => `${c.tool}(${fmtArgs(c.arguments)})`).join(" · ");
    case "tool_executed": return `${e.tool}: ${e.status}${e.evidence_ids ? ` · ${e.evidence_ids.length} passages` : ""}`;
    case "agent_ready": case "agent_stopped": return e.note || e.reason || "";
    case "draft_created": return `version ${e.version} · coverage ${e.coverage} · cites ${(e.cited || []).join(", ")}`;
    case "critic_completed": return `${e.verdict} (LLM ${e.llm_verdict}) · confidence ${pct(e.confidence)}`;
    case "revision_requested": return e.instructions;
    case "answer_emitted": return `${e.answer_status} · ${e.llm_calls} LLM calls · ${(e.prompt_tokens || 0) + (e.completion_tokens || 0)} tokens`;
    default: return e.error || "";
  }
}
async function renderTrace(r) {
  const box = $("#trace");
  box.innerHTML = '<p class="empty-note">Loading trace…</p>';
  try {
    const res = await fetch(`/trace/${encodeURIComponent(r.trace_id)}`);
    if (!res.ok) throw new Error();
    const t = await res.json();
    box.innerHTML = `
      <ol class="timeline">${t.events.map((e) => `
        <li class="${e.status === "error" ? "err" : ""}"><span class="t-ev">${esc(e.event)}</span><span class="t-time">+${secs(e.elapsed_ms)}${e.duration_ms ? ` · ${secs(e.duration_ms)}` : ""}</span>
        <span class="t-sum">${esc(summarize(e))}</span></li>`).join("")}</ol>
      <details class="raw"><summary>Raw trace JSON (${t.events.length} events, ${t.llm_calls.length} LLM calls)</summary><pre>${esc(JSON.stringify(t, null, 2))}</pre></details>
      <p class="empty-note" style="margin-top:.7rem">Trace id <code>${esc(r.trace_id)}</code> · <a href="/trace/${encodeURIComponent(r.trace_id)}" target="_blank" rel="noopener">open JSON</a></p>`;
  } catch {
    box.innerHTML = `<p class="empty-note">Trace ${esc(r.trace_id)} is no longer available on the server.</p>`;
  }
}

// ------------------------------------------------------------------ tabs, citations, actions
function selectTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".tab-panel").forEach((p) => { p.hidden = p.dataset.panel !== name; });
}
$(".tabs").addEventListener("click", (e) => { const t = e.target.closest(".tab"); if (t) selectTab(t.dataset.tab); });
$("#evidence").addEventListener("click", (e) => {
  const b = e.target.closest(".ev-more");
  if (!b) return;
  const ev = b.closest(".ev");
  ev.classList.toggle("open");
  b.textContent = ev.classList.contains("open") ? "Collapse" : "Show full passage";
});

const tooltip = $("#tooltip");
document.addEventListener("mouseover", (e) => {
  const a = e.target.closest("a.cite");
  if (!a || !current) return;
  const c = current.citations.find((x) => String(x.ref) === a.dataset.ref);
  if (!c) return;
  tooltip.innerHTML = `<b>[${c.ref}] ${esc(c.source_file)}</b><em>${esc(c.section)}</em>${esc(c.excerpt.slice(0, 260))}${c.excerpt.length > 260 ? "…" : ""}`;
  tooltip.hidden = false;
  const rect = a.getBoundingClientRect();
  const left = Math.min(Math.max(8, rect.left - 20), innerWidth - 376);
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${rect.bottom + 8 + 160 > innerHeight ? rect.top - tooltip.offsetHeight - 8 : rect.bottom + 8}px`;
});
document.addEventListener("mouseout", (e) => { if (e.target.closest("a.cite")) tooltip.hidden = true; });
document.addEventListener("click", (e) => {
  const a = e.target.closest("a.cite");
  if (!a) return;
  e.preventDefault();
  tooltip.hidden = true;
  selectTab("evidence");
  const ev = document.getElementById(`ev-${a.dataset.ref}`);
  if (!ev) return;
  ev.scrollIntoView({ behavior: "smooth", block: "center" });
  ev.classList.add("flash", "open");
  setTimeout(() => ev.classList.remove("flash"), 1600);
});

$("#copy").addEventListener("click", async () => {
  if (!current) return;
  try { await navigator.clipboard.writeText(current.answer); toast("Answer copied with sources and disclaimer"); }
  catch { toast("Copy is not available in this browser"); }
});
$("#download").addEventListener("click", () => {
  if (!current) return;
  const blob = new Blob([JSON.stringify(current, null, 2)], { type: "application/json" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `pharmalens-${current.trace_id.slice(0, 8)}.json` });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

// ------------------------------------------------------------------ history
function saveHistory(question, result) {
  const items = store.get(HISTORY_KEY, []).filter((h) => h.question !== question);
  items.unshift({ id: result.trace_id, question, ts: Date.now(), result });
  store.set(HISTORY_KEY, items.slice(0, 12));
  renderHistory(result.trace_id);
}
function renderHistory(activeId) {
  const items = store.get(HISTORY_KEY, []);
  $("#history-empty").hidden = items.length > 0;
  $("#clear-history").hidden = items.length === 0;
  $("#history").innerHTML = items.map((h) => {
    const [cls] = STATUS[h.result.status] || ["muted"];
    const conf = h.result.review?.confidence;
    return `<li><button type="button" data-id="${esc(h.id)}" class="${h.id === activeId ? "active" : ""}">
      <span class="h-dot ${cls}"></span><span class="h-q">${esc(h.question)}</span>
      <span class="h-meta">${ago(h.ts)}${conf != null ? ` · ${pct(conf)}` : ""}</span></button></li>`;
  }).join("");
}
$("#history").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  const h = store.get(HISTORY_KEY, []).find((x) => x.id === b.dataset.id);
  if (!h) return;
  showRun(h.question, `Saved result · ${ago(h.ts)} · trace ${h.id.slice(0, 8)}`);
  resetPipeline();
  STEPS.forEach(([k]) => step(k, "done", "completed"));
  $("#pipeline-card").classList.add("collapsed");
  $("#timer").textContent = secs(h.result.usage.duration_ms);
  renderReport(h.result);
  renderHistory(h.id);
  openSidebar(false);
});
$("#clear-history").addEventListener("click", () => { store.set(HISTORY_KEY, []); renderHistory(); toast("History cleared"); });

// ------------------------------------------------------------------ boot
renderHistory();
pollHealth();
textarea.focus();
