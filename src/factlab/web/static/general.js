"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const state = { busy: false, bootstrapped: false, hasKey: false, maxChars: 30000, defaults: {}, examples: [], result: null, preview: null, report: null, accepted: new Set(), usage: null, usageModel: null, demo: false, jsonEdited: false };
  const statuses = { entailed: "Следует", refuted: "Опровергнуто", undetermined: "Не определяется", unknown: "Решатель не установил", inconsistent: "Противоречивые посылки" };
  function node(tag, className, text) { const el = document.createElement(tag); if (className) el.className = className; if (text !== undefined && text !== null) el.textContent = String(text); return el; }
  function clear(el) { el.replaceChildren(); }
  function show(el, visible = true) { el.hidden = !visible; }
  function fmt(value) { return Number(value || 0).toLocaleString("ru-RU"); }
  function empty(el, message) { el.append(node("p", "inline-empty", message)); }
  function resetError() { show($("g-error"), false); }
  function error(err) {
    $("g-error-title").textContent = err.code === "timeout" ? "Не удалось дождаться ответа" : "Не удалось выполнить запрос";
    $("g-error-message").textContent = err.message || "Ошибка обработки запроса.";
    $("g-error-raw").textContent = err.raw || ""; show($("g-error-details"), Boolean(err.raw)); show($("g-error"));
    $("g-error").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  function ready() { return state.hasKey && Boolean($("g-model").value.trim()); }
  function setBusy(busy, label) {
    state.busy = busy;
    for (const id of ["g-source", "g-model", "g-prompt", "g-json", "g-reset-prompt"]) $(id).disabled = busy;
    $("g-extract").disabled = busy || !ready(); $("g-demo").disabled = busy || !state.examples.length;
    $("g-example").disabled = busy || !state.examples.length; updateSolveAvailability();
    $("g-download").disabled = busy || !state.result;
    $("g-accept-definitions").disabled = busy || !(state.preview?.proposals || []).some((proposal) => proposal.origin === "definition");
    for (const checkbox of $("g-proposals").querySelectorAll("input")) checkbox.disabled = busy;
    show($("g-progress"), busy); $("g-progress-label").textContent = label || "Обрабатываем…";
    document.querySelector(".general-formal-panel").setAttribute("aria-busy", String(busy));
  }
  function updateSolveAvailability() { $("g-solve").disabled = state.busy || !$("g-json").value.trim(); }
  function updateReady() {
    const configured = ready(), status = $("g-connection");
    status.classList.toggle("ready", configured); status.classList.toggle("not-ready", !configured);
    status.lastElementChild.textContent = configured ? "LLM настроена" : state.hasKey ? "Укажите модель" : "Режим без LLM";
    $("g-config-notice").textContent = state.hasKey ? "Ключ API настроен. Укажите модель в настройках, чтобы формализовать свой текст." : "Ключ API ещё не настроен. Готовые примеры и локальная проверка доступны без LLM.";
    show($("g-config-notice"), !configured); $("g-extract").disabled = state.busy || !configured;
  }
  function updateCount() { $("g-char-count").textContent = `${fmt($("g-source").value.length)} / ${fmt(state.maxChars)}`; document.querySelector(".editor-meta").classList.toggle("over-limit", $("g-source").value.length > state.maxChars); }
  function markStale(jsonChanged = false) {
    if (jsonChanged && state.result) { state.jsonEdited = true; show($("g-formal-stale")); }
    if (state.report) { show($("g-stale")); $("g-result-panel").classList.add("is-stale"); }
  }
  async function api(path, payload) {
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), payload === undefined ? 15000 : 120000);
    try {
      const options = { method: payload === undefined ? "GET" : "POST", headers: { Accept: "application/json" }, signal: controller.signal };
      if (payload !== undefined) { options.headers["Content-Type"] = "application/json"; options.body = JSON.stringify(payload); }
      const response = await fetch(path, options); let result;
      try { result = await response.json(); } catch (_) { throw { code: "invalid_response", message: "Сервер вернул ответ, который не удалось прочитать." }; }
      if (!response.ok) throw { code: result.error?.code, message: result.error?.message || (typeof result.detail === "string" ? result.detail : "Проверьте исходный текст и JSON формализации."), raw: result.error?.raw_output, usage: result.usage };
      return result;
    } catch (err) {
      if (err.name === "AbortError") throw { code: "timeout", message: "Время ожидания ответа истекло. Автоматического повторного запроса нет. При обрыве связи провайдер мог уже обработать вызов LLM." };
      if (err instanceof TypeError) throw { code: "network", message: "Нет связи с локальным сервером приложения. Проверьте, что FastAPI работает." };
      throw err;
    } finally { clearTimeout(timer); }
  }
  function validateText() {
    const text = $("g-source").value;
    if (!text.trim()) { error({ message: "Добавьте исходный текст или запустите готовый пример." }); $("g-source").focus(); return null; }
    if (text.length > state.maxChars) { error({ message: `Допустимо до ${fmt(state.maxChars)} символов исходного текста.` }); return null; }
    return text;
  }
  function evidence(container, values) {
    const quotes = typeof values === "string" ? [values] : Array.isArray(values) ? values : [];
    for (const quote of quotes) container.append(node("p", "general-evidence", "«" + quote + "»"));
  }
  function formulaCard(item, query = false) {
    const card = node("article", "general-formula-card"), top = node("div", "general-formula-top");
    top.append(node("span", "general-formula-id", item.id));
    if (query && item.text) top.append(node("span", "general-query-text", item.text));
    card.append(top, node("div", "general-formula", item.formula_text || "Формула не указана"));
    if (item.explanation) card.append(node("p", "general-formula-explanation", item.explanation));
    evidence(card, item.evidence); return card;
  }
  function renderVocabulary(preview) {
    const target = $("g-vocabulary"); clear(target);
    const groups = [
      ["Названные объекты", preview.entities || [], (item) => item.label || item.id, (item) => item.id],
      ["Отношения", preview.predicates || [], (item) => item.label || item.id, (item) => `${item.id} · ${(item.arg_sorts || []).length} арг.`]
    ];
    for (const [title, items, label, detail] of groups) {
      if (!items.length) continue;
      const section = node("div", "general-vocab-section"), list = node("div", "general-vocab-items"); section.append(node("h4", "", title));
      for (const item of items) { const chip = node("span", "general-vocab-item"); chip.append(node("span", "", label(item)), node("code", "", detail(item))); list.append(chip); }
      section.append(list); target.append(section);
    }
    $("g-vocab-count").textContent = `${fmt((preview.entities || []).length)} объектов · ${fmt((preview.predicates || []).length)} отношений`;
    if (!target.childElementCount) empty(target, "В формализации пока нет словаря.");
  }
  function updateAcceptLabel() {
    const definitions = (state.preview?.proposals || []).filter((item) => item.origin === "definition");
    $("g-accept-definitions").textContent = definitions.length && definitions.every((item) => state.accepted.has(item.id)) ? "Снять выбор определений" : "Выбрать все определения";
    $("g-accept-definitions").disabled = state.busy || !definitions.length;
  }
  function renderProposals(proposals) {
    const target = $("g-proposals"); clear(target);
    for (const [index, proposal] of proposals.entries()) {
      const card = node("article", "general-formula-card general-proposal-card"), checkbox = node("input"), content = node("div", "general-proposal-content");
      checkbox.type = "checkbox"; checkbox.id = `g-proposal-${index}`; checkbox.checked = state.accepted.has(proposal.id); card.classList.toggle("is-accepted", checkbox.checked);
      const label = node("label"), top = node("div", "general-formula-top"); label.htmlFor = checkbox.id;
      top.append(node("span", "general-formula-id", proposal.id), node("span", "general-origin", proposal.origin === "definition" ? "Определение" : "Допущение"));
      label.append(top, node("div", "general-formula", proposal.formula_text || "Формула не указана")); content.append(label);
      if (proposal.explanation) content.append(node("p", "general-formula-explanation", proposal.explanation));
      checkbox.addEventListener("change", () => { if (checkbox.checked) state.accepted.add(proposal.id); else state.accepted.delete(proposal.id); card.classList.toggle("is-accepted", checkbox.checked); updateAcceptLabel(); markStale(); });
      card.append(checkbox, content); target.append(card);
    }
    if (!proposals.length) empty(target, "Дополнительных определений и допущений не предложено.");
    updateAcceptLabel();
  }
  function renderFormalization(preview) {
    renderVocabulary(preview);
    const premises = preview.premises || [], proposals = preview.proposals || [], queries = preview.queries || [];
    $("g-premise-count").textContent = fmt(premises.length); $("g-proposal-count").textContent = fmt(proposals.length); $("g-query-count").textContent = fmt(queries.length);
    clear($("g-premises")); for (const premise of premises) $("g-premises").append(formulaCard(premise));
    if (!premises.length) empty($("g-premises"), "Исходных посылок пока нет.");
    renderProposals(proposals);
    clear($("g-query-preview")); for (const query of queries) $("g-query-preview").append(formulaCard(query, true));
    show($("g-query-preview-section"), queries.length > 0);
    const unparsed = state.result?.bundle?.unparsed || preview.unparsed || [];
    show($("g-preview-unparsed-section"), unparsed.length > 0); $("g-preview-unparsed-count").textContent = fmt(unparsed.length); clear($("g-preview-unparsed"));
    for (const item of unparsed) { const card = node("article", "unparsed-card"); card.append(node("p", "unparsed-text", "«" + item.text + "»"), node("p", "unparsed-reason", item.reason)); $("g-preview-unparsed").append(card); }
  }
  function supportMap(report) { return new Map([...(report.premises || []), ...(report.proposals || []).filter((item) => item.accepted)].map((item) => [item.id, item])); }
  function appendSupport(container, ids, map, title = "Основания") {
    if (!Array.isArray(ids) || !ids.length) return;
    const details = node("details", "general-support"), list = node("div", "general-support-list"); details.append(node("summary", "", `${title} · ${fmt(ids.length)}`));
    for (const id of ids) {
      const item = map.get(id), block = node("div", "general-support-item");
      block.append(node("span", "general-formula-id", id));
      if (item) { block.append(node("div", "general-formula", item.formula_text || "")); if (item.explanation) block.append(node("p", "general-formula-explanation", item.explanation)); evidence(block, item.evidence); }
      else block.append(node("p", "general-formula-explanation", "Подробности основания — в полном JSON-отчёте."));
      list.append(block);
    }
    details.append(list); container.append(details);
  }
  function factCard(fact, map) {
    const card = node("article", "general-fact-card"), heading = node("div", "general-query-heading");
    heading.append(node("span", "general-fact-expression" + (fact.negated ? " is-negative" : ""), fact.display || (fact.negated ? "¬" : "") + fact.predicate + "(" + (fact.args || []).join(", ") + ")"), node("span", "general-fact-origin" + (fact.origin === "derived" ? " is-derived" : ""), fact.origin === "derived" ? "Новый факт" : "Из текста"));
    card.append(heading); appendSupport(card, fact.support, map, fact.origin === "derived" ? "Основания вывода" : "Источник факта"); return card;
  }
  function renderReport(report) {
    show($("g-result-panel"), Boolean(report)); if (!report) return;
    show($("g-stale"), false); $("g-result-panel").classList.remove("is-stale");
    $("g-result-mode").textContent = state.demo ? "Демо · без LLM" : "Локальный вывод";
    const map = supportMap(report), consistency = $("g-consistency"); clear(consistency);
    consistency.className = "general-consistency" + (report.consistency === "unsat" ? " is-inconsistent" : report.consistency === "unknown" ? " is-unknown" : "");
    if (report.consistency === "unsat") {
      consistency.append(node("strong", "", "Посылки противоречат друг другу"), node("p", "", "Исправьте исходные утверждения или выбранные определения. Из этой базы не выдаём произвольные утверждения как истинные."));
      appendSupport(consistency, report.consistency_support, map, "Конфликтующие основания");
    } else if (report.consistency === "unknown") {
      consistency.append(node("strong", "", "Совместимость посылок не установлена"), node("p", "", "Решатель не завершил проверку за отведённое время. Это не означает ни совместимость, ни противоречие."));
      if (report.reason_unknown) consistency.append(node("p", "general-formula-explanation", report.reason_unknown));
    } else consistency.append(node("strong", "", "Посылки совместимы"), node("p", "", "В показанной формализации противоречий не обнаружено."));
    const stats = $("g-statistics"); clear(stats);
    for (const [label, value] of [["посылок", report.statistics?.premises], ["определений и допущений принято", report.statistics?.accepted_proposals], ["проверок решателя", report.statistics?.solver_checks]]) {
      if (value != null) { const item = node("span", "", label + ": "); item.append(node("strong", "", fmt(value))); stats.append(item); }
    }
    if (report.statistics?.elapsed_ms != null) stats.append(node("span", "", `${fmt(report.statistics.elapsed_ms)} мс`));
    const queries = $("g-queries"); clear(queries);
    show($("g-queries-section"), (report.queries || []).length > 0); $("g-result-query-count").textContent = fmt((report.queries || []).length);
    for (const query of report.queries || []) {
      const card = node("article", "general-query-card"), heading = node("div", "general-query-heading");
      heading.append(node("h4", "general-query-text", query.text || query.id), node("span", "general-query-status " + query.status, statuses[query.status] || query.status));
      card.append(heading, node("div", "general-formula", query.formula_text || ""));
      if (query.reason_unknown) card.append(node("p", "general-formula-explanation", query.reason_unknown));
      if (query.status === "undetermined") card.append(node("p", "general-unsolved-note", "Посылки допускают как истинность, так и ложность этого утверждения."));
      appendSupport(card, query.support, map); queries.append(card);
    }
    const target = $("g-facts"); clear(target);
    const canShowFacts = report.consistency === "sat", facts = canShowFacts ? report.facts || [] : [];
    const derived = facts.filter((fact) => fact.origin === "derived"), extracted = facts.filter((fact) => fact.origin === "extracted");
    const partial = Boolean(report.statistics?.truncated || report.statistics?.deadline_reached || report.statistics?.named_atoms_unknown);
    $("g-fact-count").textContent = canShowFacts ? fmt(derived.length) : "—";
    $("g-facts-description").textContent = report.consistency === "unsat" ? "Сначала устраните противоречие в посылках." : !canShowFacts ? "Поиск фактов не выполнен: сначала нужно установить совместимость посылок." : `Найдено ${fmt(derived.length)} новых фактов о названных объектах. Исходные факты показаны отдельно ниже.`;
    if (canShowFacts && partial) {
      let note = ` Список частичный: проверено ${fmt(report.statistics?.named_atoms_checked)} из ${fmt(report.statistics?.named_atoms_total)} возможных атомов.`;
      if (report.statistics?.named_atoms_unknown) note += ` Для ${fmt(report.statistics.named_atoms_unknown)} кандидатов решатель не установил результат.`;
      if (report.statistics?.deadline_reached) note += " Достигнут лимит времени.";
      $("g-facts-description").append(node("span", "general-partial-note", note + " Непроверенные кандидаты могут дать дополнительные следствия."));
    }
    for (const fact of derived) target.append(factCard(fact, map));
    if (!derived.length) empty(target, report.consistency === "unsat" ? "Факты не показываются для противоречивых посылок." : !canShowFacts ? "Это не означает, что новых фактов нет: вывод пока недоступен." : partial ? "В проверенной части новых фактов не найдено. Поиск завершён не полностью." : "Среди проверенных комбинаций предикатов и именованных объектов дополнительных фактов нет.");
    const extractedTarget = $("g-extracted-facts"); clear(extractedTarget); show($("g-extracted-section"), extracted.length > 0); $("g-extracted-count").textContent = fmt(extracted.length);
    for (const fact of extracted) extractedTarget.append(factCard(fact, map));
    renderUnparsed(report.unparsed || []); $("g-report-explanation").textContent = report.explanation || "";
  }
  function renderUnparsed(items) {
    const target = $("g-unparsed"); clear(target); show($("g-unparsed-section"), items.length > 0); $("g-unparsed-count").textContent = fmt(items.length);
    for (const item of items) { const card = node("article", "unparsed-card"); card.append(node("p", "unparsed-text", "«" + item.text + "»"), node("p", "unparsed-reason", item.reason)); target.append(card); }
  }
  function renderUsage() {
    const footer = $("g-usage"); clear(footer); const usage = state.usage;
    if (state.demo) footer.append(node("span", "", "Демонстрация · платных вызовов нет"));
    else if (state.usageModel) footer.append(node("span", "", "Формализация: " + state.usageModel));
    if (usage) {
      for (const [field, label] of [["prompt_tokens", "Вход"], ["completion_tokens", "Выход"], ["total_tokens", "Всего"]]) if (usage[field] != null) footer.append(node("span", "", `${label}: ${fmt(usage[field])} токенов`));
      const reasoning = usage.completion_tokens_details?.reasoning_tokens ?? usage.reasoning_tokens;
      if (reasoning != null) footer.append(node("span", "", `В том числе рассуждения: ${fmt(reasoning)}`));
    }
    show(footer, footer.childElementCount > 0);
  }
  function renderResult(result, kind) {
    state.result = result; state.report = result.report || null;
    state.preview = result.preview || { premises: result.report?.premises || [], proposals: result.report?.proposals || [], queries: result.report?.queries || [], sorts: result.bundle?.sorts || [], predicates: result.bundle?.predicates || [], entities: result.bundle?.entities || [] };
    if (kind === "extract" || kind === "demo") state.accepted.clear();
    if (state.report?.proposals) state.accepted = new Set(state.report.proposals.filter((item) => item.accepted).map((item) => item.id));
    if (kind === "extract" || kind === "demo") { state.usage = result.usage || null; state.usageModel = kind === "extract" ? result.model || $("g-model").value.trim() : null; state.demo = Boolean(result.demo); }
    if (typeof result.source_text === "string") $("g-source").value = result.source_text;
    $("g-json").value = typeof result.raw_json === "string" ? result.raw_json : JSON.stringify(result.bundle, null, 2);
    state.jsonEdited = false; show($("g-formal-stale"), false); show($("g-empty"), false); show($("g-formal-content"));
    $("g-step-formal").className = "pipeline-step is-done"; $("g-step-solve").className = "pipeline-step " + (state.report ? "is-done" : "is-active");
    renderFormalization(state.preview); renderReport(state.report); renderUsage(); updateCount();
  }
  async function run(kind) {
    if (state.busy) return; resetError(); let payload;
    if (kind === "demo") payload = { example_id: $("g-example").value };
    else {
      const text = validateText(); if (text === null) return;
      if (kind === "extract") {
        payload = { text }; const model = $("g-model").value.trim(); if (model) payload.model = model;
        if ($("g-prompt").value && $("g-prompt").value !== state.defaults.prompt_template) payload.prompt_template = $("g-prompt").value;
      } else {
        const bundleJson = $("g-json").value.trim();
        try { JSON.parse(bundleJson); } catch (_) { error({ message: "В JSON формализации синтаксическая ошибка. Проверьте кавычки, запятые и скобки." }); $("g-json").closest("details").open = true; $("g-json").focus(); return; }
        payload = { text, bundle_json: bundleJson, accepted_proposal_ids: [...state.accepted] };
      }
    }
    markStale(); setBusy(true, kind === "extract" ? "Модель извлекает факты и посылки. Это единственный платный вызов…" : kind === "demo" ? "Загружаем пример и находим новые факты локально…" : "Решатель проверяет посылки и сам ищет новые факты…");
    try {
      const result = await api("/api/general/" + (kind === "solve" ? "analyze" : kind), payload); renderResult(result, kind);
      if (kind === "solve" && state.report) $("g-result-panel").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      if (err.raw) { $("g-json").value = String(err.raw); $("g-json").closest("details").open = true; markStale(true); }
      if (err.usage) { state.usage = err.usage; state.usageModel = $("g-model").value.trim(); state.demo = false; renderUsage(); }
      error(err);
    } finally { setBusy(false); }
  }
  async function bootstrap() {
    setBusy(true, "Загружаем настройки…");
    try {
      const data = await api("/api/general/bootstrap"); state.defaults = data.defaults || {}; state.examples = data.examples || []; state.hasKey = Boolean(data.configuration?.has_api_key ?? data.configuration?.ready); state.maxChars = data.limits?.max_text_chars || 30000;
      $("g-source").value = state.defaults.text || ""; $("g-model").value = data.configuration?.model || ""; $("g-prompt").value = state.defaults.prompt_template || "";
      clear($("g-example")); for (const example of state.examples) { const option = node("option", "", example.title); option.value = example.id; $("g-example").append(option); }
      state.bootstrapped = true; updateReady(); updateCount();
    } catch (err) { $("g-connection").lastElementChild.textContent = "Сервер недоступен"; error(err); }
    finally { setBusy(false); }
  }
  $("g-extract").addEventListener("click", () => run("extract")); $("g-demo").addEventListener("click", () => run("demo")); $("g-solve").addEventListener("click", () => run("solve"));
  $("g-close-error").addEventListener("click", resetError); $("g-model").addEventListener("input", updateReady);
  $("g-source").addEventListener("input", () => { updateCount(); markStale(true); }); $("g-json").addEventListener("input", () => { markStale(true); updateSolveAvailability(); });
  $("g-reset-prompt").addEventListener("click", () => { $("g-prompt").value = state.defaults.prompt_template || ""; });
  $("g-accept-definitions").addEventListener("click", () => {
    const definitions = (state.preview?.proposals || []).filter((item) => item.origin === "definition"), allAccepted = definitions.every((item) => state.accepted.has(item.id));
    for (const item of definitions) { if (allAccepted) state.accepted.delete(item.id); else state.accepted.add(item.id); }
    renderProposals(state.preview?.proposals || []); markStale();
  });
  $("g-download").addEventListener("click", () => {
    if (!state.result) return;
    const result = { ...state.result, raw_json: $("g-json").value, accepted_proposal_ids: [...state.accepted], usage: state.usage, model: state.usageModel || state.result.model, edited_since_analysis: state.jsonEdited || !$("g-stale").hidden };
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: "application/json;charset=utf-8" })), link = node("a"); link.href = url; link.download = "формализация-и-следствия.json"; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  bootstrap();
})();
