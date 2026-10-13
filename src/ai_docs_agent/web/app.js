"use strict";
const $ = (selector) => document.querySelector(selector);
const el = (tag, className, text) => { const node = document.createElement(tag); if (className) node.className = className; if (text !== undefined) node.textContent = text; return node; };
let health, activeJob, jobTimer, busy = false, messageId = 0;
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : body.errors?.map(e => e.message).join("；") || "操作失败，请检查输入或服务状态。");
  return body;
}
const jsonOptions = body => ({method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
function toast(text) { $("#toast").textContent = text; $("#toast").hidden = false; clearTimeout(toast.timer); toast.timer = setTimeout(() => $("#toast").hidden = true, 4500); }
async function refresh() {
  try {
    health = await api("/api/health");
    const index = health.index, provider = health.providers;
    $("#doc-count").textContent = index.document_count;
    $("#welcome-docs").textContent = index.document_count;
    $("#welcome-chunks").textContent = index.chunk_count;
    $("#index-status").textContent = `${index.document_count} 份文档 · ${index.chunk_count} 个片段\n${index.built_at ? "索引更新：" + new Date(index.built_at).toLocaleString("zh-CN") : "上传文档以开始"}`;
    $("#model-status").textContent = provider.llm_configured ? "模型已配置" : "原文摘录模式";
    $("#answer-note").textContent = provider.llm_configured ? "回答使用检索证据生成，请通过引用核对。Shift + Enter 换行。" : "模型尚未配置，当前展示检索原文。可在左侧设置中启用生成回答。";
    const {documents} = await api("/api/documents");
    const list = $("#document-list"); list.replaceChildren();
    for (const doc of documents) {
      const item = el("div", "document-item"), info = el("div", "document-info");
      item.append(el("span", "file-type", doc.name.split(".").pop().toUpperCase()));
      const link = el("a", "", doc.name); link.href = `/api/documents/${encodeURIComponent(doc.name)}/file`; link.target = "_blank"; link.rel = "noopener"; link.title = doc.name;
      info.append(link, el("small", "", doc.indexed ? `${doc.chunks} 个片段 · 已索引` : "等待索引"));
      const remove = el("button", "delete-doc", "×"); remove.setAttribute("aria-label", `删除 ${doc.name}`);
      remove.onclick = async () => { if (!confirm(`删除文档 ${doc.name} 并更新索引？`)) return; try { const response = await api(`/api/documents/${encodeURIComponent(doc.name)}`, {method:"DELETE"}); watchJob(response.job); await refresh(); } catch(error) { toast(error.message); } };
      item.append(info, remove); list.append(item);
    }
    if (health.job && ["queued", "running"].includes(health.job.status)) watchJob(health.job);
  } catch(error) { $("#model-status").textContent = "服务连接失败"; toast(error.message); }
}
function watchJob(job) {
  $("#job-message").textContent = job.message;
  $("#job-message").classList.toggle("error", job.status === "failed");
  if (["done", "failed"].includes(job.status)) { activeJob = null; clearInterval(jobTimer); $("#rebuild").disabled = false; return; }
  if (activeJob === job.id) return;
  activeJob = job.id; $("#rebuild").disabled = true; clearInterval(jobTimer);
  jobTimer = setInterval(async () => {
    try {
      const latest = await api(`/api/jobs/${activeJob}`);
      $("#job-message").textContent = latest.message;
      if (["done", "failed"].includes(latest.status)) { watchJob(latest); await refresh(); }
    } catch(error) { clearInterval(jobTimer); activeJob = null; $("#rebuild").disabled = false; toast(error.message); }
  }, 1500);
}
async function upload(files) {
  if (!files.length) return;
  const body = new FormData(); for (const file of files) body.append("files", file);
  $("#job-message").textContent = "正在上传和检查文档…";
  try { const response = await api("/api/upload", {method:"POST", body}); if (response.errors.length) toast(response.errors.map(e => `${e.name}：${e.message}`).join("；")); watchJob(response.job); await refresh(); }
  catch(error) { $("#job-message").textContent = error.message; $("#job-message").classList.add("error"); }
}
$("#file-input").onchange = async event => { await upload(event.target.files); event.target.value = ""; };
const zone = $("#upload-zone");
for (const name of ["dragenter", "dragover"]) zone.addEventListener(name, event => { event.preventDefault(); zone.classList.add("dragging"); });
for (const name of ["dragleave", "drop"]) zone.addEventListener(name, event => { event.preventDefault(); zone.classList.remove("dragging"); });
zone.addEventListener("drop", event => upload(event.dataTransfer.files));
$("#rebuild").onclick = async () => { try { watchJob(await api("/api/ingest", jsonOptions({chunk_size:350,overlap:60}))); } catch(error) { toast(error.message); } };
function scrollBottom() { $("#conversation").scrollTop = $("#conversation").scrollHeight; }
function renderAnswer(result, id) {
  const message = el("article", "message answer-message");
  const header = el("div", "answer-header");
  const labels = {generated:"模型生成回答", extractive:"原文摘录", extractive_fallback:"原文摘录 · 模型调用失败", no_evidence:"缺少证据", insufficient_evidence:"证据不足", needs_configuration:"模型待配置"};
  header.append(el("span","answer-avatar","A"), el("strong","","AI Docs Agent"), el("span","mode-badge", labels[result.mode] || result.mode)); message.append(header);
  const text = el("div", "answer-text");
  for (const part of result.answer.split(/(\[\d+\])/g)) {
    const reference = part.match(/^\[(\d+)\]$/);
    if (reference && result.citations.some(c => c.number === Number(reference[1]))) {
      const button = el("button", "citation-link", part);
      button.onclick = () => { const details = document.getElementById(`sources-${id}`); details.open = true; document.getElementById(`source-${id}-${reference[1]}`).scrollIntoView({behavior:"smooth", block:"center"}); };
      text.append(button);
    } else text.append(document.createTextNode(part));
  }
  message.append(text);
  for (const warning of result.warnings || []) message.append(el("div", "warning", warning));
  message.append(el("div", "answer-meta", `${result.method.toUpperCase()} · ${result.route === "web" ? "联网来源" : "本地知识库"} · ${(result.latency_ms / 1000).toFixed(2)} 秒`));
  if (result.citations.length) {
    const details = el("details", "retrieval-details"); details.id = `sources-${id}`;
    details.append(el("summary", "", `参考来源与检索片段 · ${result.citations.length}`)); const cards = el("div", "sources");
    for (const source of result.citations) {
      const card = el("div", "source-card"); card.id = `source-${id}-${source.number}`;
      const title = el("div", "source-title"), link = el("a", "", `[${source.number}] ${source.source}`);
      link.href = source.source_url || `/api/documents/${encodeURIComponent(source.source)}/file`; link.target = "_blank"; link.rel = "noopener noreferrer";
      title.append(link, el("span", "", source.rerank_score !== undefined ? `重排序 ${source.rerank_score.toFixed(3)}` : `检索 ${source.score.toFixed(3)}`));
      card.append(title, el("div", "source-location", [source.section, source.page ? `第 ${source.page} 页` : source.line_start ? `第 ${source.line_start}–${source.line_end} 行` : "网页摘录", `字符 ${source.start}–${source.end}`].filter(Boolean).join(" · ")), el("div", "source-text", source.text)); cards.append(card);
    }
    details.append(cards); message.append(details);
  }
  const trace = el("details", "trace-details"); trace.append(el("summary", "", `查看处理过程 · ${result.trace.length} 个步骤`)); const steps = el("div", "trace-list");
  for (const step of result.trace) { const row = el("div", "trace-step"); row.append(el("strong", "", step.node), el("span", "", step.detail), el("small", "", `${step.latency_ms.toFixed(0)} ms`)); steps.append(row); }
  trace.append(steps); message.append(trace); return message;
}
$("#question-form").onsubmit = async event => {
  event.preventDefault(); const question = $("#question").value.trim(); if (!question || busy) return;
  busy = true; $("#ask").disabled = true; $("#question").value = ""; $("#welcome").hidden = true;
  const id = ++messageId, user = el("div", "message user-message"); user.append(el("div", "user-bubble", question)); $("#conversation").append(user);
  const pending = el("div", "message loading", "正在检索文档并核对证据…"); $("#conversation").append(pending); scrollBottom();
  try {
    const result = await api("/api/query", jsonOptions({question, method:$("#method").value, route:$("#route").value, k:Number($("#top-k").value), candidates:20, answer_mode:"auto"}));
    pending.replaceWith(renderAnswer(result, id));
  } catch(error) { pending.className = "message error-message"; pending.textContent = error.message; }
  finally { busy = false; $("#ask").disabled = false; scrollBottom(); $("#question").focus(); }
};
$("#question").addEventListener("keydown", event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); $("#question-form").requestSubmit(); } });
for (const button of document.querySelectorAll("[data-question]")) button.onclick = () => { $("#question").value = button.dataset.question; $("#question").focus(); };
for (const button of document.querySelectorAll("[data-view]")) button.onclick = async () => { for (const nav of document.querySelectorAll("[data-view]")) nav.classList.toggle("active", nav === button); $("#chat-view").hidden = button.dataset.view !== "chat"; $("#evaluation-view").hidden = button.dataset.view !== "evaluation"; if (button.dataset.view === "evaluation") await loadEvaluation(); };
async function loadEvaluation() {
  const target = $("#evaluation-content"); target.replaceChildren(el("p","loading","正在读取评测结果…"));
  try {
    const report = await api("/api/evaluation"); target.replaceChildren();
    if (report.status === "not_run") { target.append(el("p", "eval-notes", "还没有评测结果。按项目 README 中的评测步骤运行后，这里会显示实测指标。")); return; }
    const cards = el("div", "eval-cards");
    for (const [value,label] of [[report.questions,"标注问题"],[report.k,"检索 Top-K"],[report.repeats,"每题重复测量"]]) { const card = el("div","eval-card"); card.append(el("strong","",String(value)), document.createTextNode(label)); cards.append(card); }
    target.append(cards); const wrap = el("div","table-wrap"), table = el("table"), head = el("thead"), row = el("tr");
    for (const title of ["检索方式","Hit@5","Recall@5","MRR@5","nDCG@5","P50 / ms","P95 / ms"]) row.append(el("th","",title)); head.append(row); table.append(head);
    const body = el("tbody"), names = {dense:"向量检索",bm25:"BM25",hybrid:"混合检索 · RRF",rerank:"混合 + BGE 重排序"};
    for (const [method, metrics] of Object.entries(report.metrics || {})) { const tr = el("tr"); const values = [names[method] || method, ...["hit","recall","mrr","ndcg"].map(key => metrics[key].toFixed(3)), metrics.p50_ms.toFixed(1), metrics.p95_ms.toFixed(1)]; for (const value of values) tr.append(el("td","",value)); body.append(tr); }
    table.append(body); wrap.append(table); target.append(wrap);
    target.append(el("p","eval-notes",report.notes));
    if (report.index_revision !== health?.index.revision) target.append(el("div","warning","评测使用的索引与当前索引不同。文档变更后请重新运行评测。"));
    if (report.errors?.length) target.append(el("div","warning",report.errors.join("；")));
  } catch(error) { target.replaceChildren(el("div","error-message",error.message)); }
}
$("#open-settings").onclick = async () => { try { const settings = await api("/api/settings"); $("#llm-base").value = settings.llm_base_url; $("#llm-model").value = settings.llm_model; $("#answer-mode").value = settings.answer_mode; $("#llm-key-status").textContent = settings.llm_configured ? "已配置" : "未配置"; $("#tavily-key-status").textContent = settings.search_configured ? "已配置" : "未配置"; $("#settings-message").textContent = ""; $("#settings-dialog").showModal(); } catch(error) { toast(error.message); } };
$("#close-settings").onclick = () => $("#settings-dialog").close();
$("#settings-form").onsubmit = async event => { event.preventDefault(); try { await api("/api/settings", jsonOptions({llm_base_url:$("#llm-base").value, llm_model:$("#llm-model").value, llm_key:$("#llm-key").value, tavily_key:$("#tavily-key").value, answer_mode:$("#answer-mode").value})); $("#llm-key").value = ""; $("#tavily-key").value = ""; $("#settings-dialog").close(); toast("配置已保存在本地。下一次提问将使用新设置。"); await refresh(); } catch(error) { $("#settings-message").textContent = error.message; } };
refresh();
