"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const state = { busy: false, ready: false, hasApiKey: false, bootstrapped: false, maxChars: 30000, result: null, defaults: {}, profile: null, profiles: [], factMap: new Map(), entityMap: new Map() };
  const actionIds = ["analyze-button", "extract-button", "demo-button", "infer-button", "reset-prompt"];
  const tabs = [...document.querySelectorAll("[role=tab]")];

  function node(tag, className, content) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (content !== undefined && content !== null) el.textContent = String(content);
    return el;
  }
  function clear(el) { el.replaceChildren(); }
  function show(el, visible = true) { el.hidden = !visible; }
  function fmt(value) { return Number(value || 0).toLocaleString("ru-RU"); }
  function listEmpty(target, message) { target.append(node("p", "inline-empty", message)); }
  function resetError() { show($("error-banner"), false); }
  function error(message, raw, code) {
    $("error-title").textContent = code === "timeout" ? "Сервер не успел ответить" : "Не удалось выполнить запрос";
    $("error-message").textContent = message;
    $("error-raw").textContent = raw || "";
    show($("error-raw-details"), Boolean(raw));
    show($("error-banner"));
    $("error-banner").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  function setBusy(busy, label) {
    state.busy = busy;
    for (const id of actionIds) $(id).disabled = busy;
    $("analyze-button").disabled = busy || !state.ready;
    $("extract-button").disabled = busy || !state.ready;
    $("source-text").disabled = busy;
    $("knowledge-profile").disabled = busy || !state.profiles.length;
    $("model-name").disabled = busy;
    $("prompt-template").disabled = busy;
    $("json-editor").disabled = busy;
    $("download-report").disabled = busy || !state.result;
    $("download-json").disabled = busy || !$("json-editor").value.trim();
    show($("progress-status"), busy);
    $("progress-label").textContent = label || "Обрабатываем текст…";
    document.querySelector(".results-panel").setAttribute("aria-busy", String(busy));
  }
  function updateAvailability() {
    state.ready = state.hasApiKey && Boolean($("model-name").value.trim());
    $("analyze-button").disabled = state.busy || !state.ready;
    $("extract-button").disabled = state.busy || !state.ready;
    const status = $("connection-status");
    status.classList.toggle("ready", state.ready);
    status.classList.toggle("not-ready", !state.ready);
    status.lastElementChild.textContent = state.ready ? "LLM настроена" : state.hasApiKey ? "Укажите модель" : "Режим без LLM";
    $("configuration-notice").querySelector("p").textContent = state.hasApiKey
      ? "Ключ API настроен. Укажите имя модели в настройках извлечения, чтобы отправить текст в LLM. Готовый пример доступен без платного запроса."
      : "LLM ещё не настроена на сервере. Пока можно запустить готовый пример или вставить JSON во вкладке «JSON» и выполнить вывод.";
    show($("configuration-notice"), !state.ready);
  }
  function clearResult() {
    state.result = null; state.factMap.clear(); state.entityMap.clear();
    show($("empty-state")); show($("fact-results"), false); show($("statistics"), false);
    show($("result-mode"), false); show($("result-context"), false); show($("usage-footer"), false);
    for (const id of ["initial-facts", "derived-facts", "entity-list", "unparsed-list", "usage-footer"]) clear($(id));
    listEmpty($("entity-list"), "Сущности появятся после извлечения.");
    listEmpty($("unparsed-list"), "Пока нет обработанного текста.");
    $("entities-count").textContent = "0"; $("unparsed-count").textContent = "0";
    $("json-editor").value = "";
    clear($("review-notice")); show($("review-notice"), false);
    clear($("graph-checks")); show($("graph-section"), false);
    $("initial-count").textContent = "0"; $("derived-count").textContent = "0";
    setStep(false, false);
  }
  function ruleTitle(id) { return state.profile?.rule_descriptions?.[id]?.title || id; }
  function literalFormula(literal) {
    if (!literal || typeof literal !== "object") return "?";
    const args = (literal.args || []).map((arg) => {
      if (typeof arg !== "object" || arg === null) return String(arg);
      if (Object.hasOwn(arg, "var")) return String(arg.var);
      if (Object.hasOwn(arg, "const")) return String(arg.const);
      return "?";
    });
    return (literal.negated ? "¬" : "") + String(literal.predicate || "?") + "(" + args.join(", ") + ")";
  }
  function renderProfile() {
    const select = $("knowledge-profile"), current = state.profile?.id;
    clear(select);
    for (const profile of state.profiles) {
      const option = node("option", "", profile.title || profile.id); option.value = profile.id;
      option.selected = profile.id === current; select.append(option);
    }
    select.value = current || "";
    $("profile-description").textContent = state.profile?.description || "";
    const target = $("profile-rules"); clear(target);
    let rules = [];
    try { rules = JSON.parse(state.defaults.rules_json || "{}").rules || []; } catch (_) { /* The raw rules remain available in settings. */ }
    $("profile-rule-count").textContent = `Правил: ${fmt(rules.length)}`;
    for (const rule of rules) {
      const description = state.profile?.rule_descriptions?.[rule.id] || {};
      const card = node("article", "profile-rule");
      const heading = node("div", "profile-rule-heading"); heading.append(node("strong", "", description.title || rule.id));
      if (description.kind) heading.append(node("span", "rule-kind", description.kind));
      card.append(heading, node("code", "rule-formula", (rule.body || []).map(literalFormula).join(" ∧ ") + " → " + literalFormula(rule.head)));
      if (description.description) card.append(node("p", "", description.description));
      target.append(card);
    }
    if (!rules.length) listEmpty(target, "В профиле пока нет правил вывода.");
  }
  function setStep(extracted, inferred) {
    $("step-text").className = "pipeline-step " + (extracted ? "is-done" : "is-active");
    $("step-facts").className = "pipeline-step " + (extracted ? "is-done" : "");
    $("step-inference").className = "pipeline-step " + (inferred ? "is-done" : extracted ? "is-active" : "");
  }
  function selectTab(name, focus = false) {
    for (const tab of tabs) {
      const selected = tab.dataset.view === name;
      tab.classList.toggle("is-selected", selected);
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      show($("view-" + tab.dataset.view), selected);
      if (selected && focus) tab.focus();
    }
  }
  function updateCharacters() {
    const count = $("source-text").value.length;
    $("character-count").textContent = `${fmt(count)} / ${fmt(state.maxChars)}`;
    document.querySelector(".editor-meta").classList.toggle("over-limit", count > state.maxChars);
  }
  async function api(path, payload) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 120000);
    try {
      const options = { method: payload === undefined ? "GET" : "POST", signal: controller.signal, headers: { Accept: "application/json" } };
      if (payload !== undefined) { options.headers["Content-Type"] = "application/json"; options.body = JSON.stringify(payload); }
      const response = await fetch(path, options);
      let data;
      try { data = await response.json(); } catch (_) { throw { message: "Сервер вернул ответ, который не удалось прочитать. Проверьте, что приложение FastAPI запущено.", code: "invalid_response" }; }
      if (!response.ok) {
        const detail = data.error || {};
        throw { message: detail.message || (typeof data.detail === "string" ? data.detail : "Ошибка сервера. Проверьте исходные данные и повторите запрос."), raw: detail.raw_output, code: detail.code, usage: data.usage };
      }
      return data;
    } catch (err) {
      if (err.name === "AbortError") throw { code: "timeout", message: "Ожидание превысило 120 секунд. Запрос не отправляется повторно автоматически. При обрыве связи провайдер мог уже обработать его." };
      if (err instanceof TypeError) throw { code: "network", message: "Нет связи с сервером приложения. Проверьте, что FastAPI работает, и повторите запрос." };
      throw err;
    } finally { clearTimeout(timeout); }
  }
  function validateText() {
    const text = $("source-text").value;
    if (!text.trim()) { error("Добавьте исходный текст. Для готового сценария используйте кнопку «Запустить пример без LLM»."); $("source-text").focus(); return null; }
    if (text.length > state.maxChars) { error(`Текст слишком длинный. Допустимо до ${fmt(state.maxChars)} символов.`); return null; }
    return text;
  }
  async function run(kind) {
    if (state.busy) return;
    resetError();
    let payload = { profile: state.profile?.id || "university" };
    if (kind === "analyze" || kind === "extract") {
      const text = validateText(); if (text === null) return;
      payload.text = text;
      const model = $("model-name").value.trim(); if (model) payload.model = model;
      const prompt = $("prompt-template").value;
      if (prompt && prompt !== state.defaults.prompt_template) payload.prompt_template = prompt;
    } else if (kind === "infer") {
      const text = validateText(); if (text === null) return;
      const extractionJson = $("json-editor").value.trim();
      if (!extractionJson) { error("Вставьте JSON с сущностями и фактами во вкладке «JSON»."); selectTab("json", true); return; }
      try { JSON.parse(extractionJson); } catch (_) { error("В редакторе JSON синтаксическая ошибка. Проверьте кавычки, запятые и скобки."); selectTab("json", true); return; }
      payload.text = text; payload.extraction_json = extractionJson;
    }
    const labels = { analyze: "Извлекаем факты через LLM и выполняем вывод…", extract: "LLM извлекает факты из текста…", infer: "Проверяем JSON и выводим новые факты…", demo: "Запускаем пример без обращения к LLM…" };
    if (kind === "analyze" || kind === "extract") clearResult();
    else if (state.result) {
      $("result-context").textContent = "Результат предыдущего запуска. Ожидаем новый результат обработки.";
      show($("result-context"));
    }
    setBusy(true, labels[kind]);
    try {
      const result = await api("/api/" + kind, payload);
      renderResult(result);
      selectTab("facts");
      if (window.matchMedia("(max-width: 820px)").matches) document.querySelector(".results-panel").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      if (state.result) {
        $("result-context").textContent = "Результат предыдущего запуска. Последний запрос завершился ошибкой; эти факты к нему не относятся.";
        show($("result-context"));
      }
      if (err.raw) { $("json-editor").value = String(err.raw); selectTab("json"); }
      if (err.usage) renderUsage(err.usage, $("model-name").value.trim(), false, null);
      error(err.message || "Неизвестная ошибка при обработке запроса.", err.raw, err.code);
    }
    finally { setBusy(false); }
  }
  function factDisplay(fact) {
    if (fact.display) return fact.display;
    return (fact.negated ? "¬" : "") + String(fact.predicate || "?") + "(" + (fact.args || []).map((id) => state.entityMap.get(id)?.label || id).join(", ") + ")";
  }
  function addEvidence(container, evidence) {
    if (!Array.isArray(evidence) || !evidence.length) return;
    const quotes = node("div", "fact-evidence");
    for (const quote of evidence) quotes.append(node("p", "", "«" + String(quote) + "»"));
    container.append(quotes);
  }
  function proofNode(factId, depth, visited) {
    const item = node("li");
    const fact = state.factMap.get(factId);
    if (!fact) { item.append(node("p", "proof-node-meta", String(factId) + ": основание не найдено в отчёте")); return item; }
    item.append(node("div", "proof-node-expression", `${fact.id} · ${factDisplay(fact)}`));
    if (visited.has(factId)) { item.append(node("div", "proof-node-meta", "Это основание уже показано выше.")); return item; }
    if (fact.origin === "extracted") {
      item.append(node("div", "proof-node-meta", "Исходный факт"));
      for (const quote of fact.evidence || []) item.append(node("p", "proof-quote", "«" + String(quote) + "»"));
    } else if (fact.proof) {
      item.append(node("div", "proof-node-meta", `${ruleTitle(fact.proof.rule)} · раунд ${fact.proof.round} · ${fact.proof.rule}`));
      if (depth >= 10) { item.append(node("div", "proof-node-meta", "Продолжение цепочки — в полном JSON-отчёте.")); return item; }
      const nextVisited = new Set(visited); nextVisited.add(factId);
      const children = node("ul", "proof-tree");
      for (const parent of fact.proof.parents || []) children.append(proofNode(parent, depth + 1, nextVisited));
      item.append(children);
    }
    return item;
  }
  function factCard(fact, derived) {
    const card = node(derived ? "details" : "article", "fact-card");
    const summary = node(derived ? "summary" : "div", "fact-summary");
    summary.append(node("span", "fact-id", fact.id || ""), node("span", "fact-expression" + (fact.negated ? " is-negative" : ""), factDisplay(fact)));
    if (fact.negated) summary.append(node("span", "negative-label", "отрицание"));
    if (derived) summary.setAttribute("aria-label", "Показать объяснение: " + factDisplay(fact));
    card.append(summary);
    if (!derived) addEvidence(card, fact.evidence);
    else {
      const body = node("div", "proof-body");
      if (fact.proof) {
        const rule = node("div", "proof-rule", "Применено правило: "); rule.append(node("strong", "", ruleTitle(fact.proof.rule)), document.createTextNode(` · раунд ${fact.proof.round}`), node("code", "proof-rule-id", fact.proof.rule)); body.append(rule);
        const tree = node("ul", "proof-tree");
        for (const parent of fact.proof.parents || []) tree.append(proofNode(parent, 0, new Set([fact.id])));
        body.append(tree);
        const bindings = Object.entries(fact.proof.bindings || {}).map(([key, id]) => `${key} = ${state.entityMap.get(id)?.label || id}`);
        if (bindings.length) body.append(node("p", "proof-node-meta", "Подстановка: " + bindings.join("; ")));
      } else body.append(node("p", "proof-node-meta", "Подробное основание отсутствует в отчёте."));
      card.append(body);
    }
    return card;
  }
  function renderEntities(entities) {
    const target = $("entity-list"); clear(target);
    if (!entities.length) { listEmpty(target, "В тексте не найдено сущностей текущей схемы."); return; }
    for (const entity of entities) {
      const card = node("article", "entity-card"), top = node("div", "entity-top");
      top.append(node("span", "entity-id", entity.id), node("span", "entity-label", entity.label), node("span", "entity-type", entity.type)); card.append(top);
      if (entity.mentions?.length) card.append(node("p", "entity-mentions", "В тексте: " + entity.mentions.join(" · ")));
      target.append(card);
    }
  }
  function renderUnparsed(unparsed) {
    const target = $("unparsed-list"); clear(target);
    if (!unparsed.length) { listEmpty(target, "Нет фрагментов, отмеченных как неподдержанные."); return; }
    for (const fragment of unparsed) {
      const card = node("article", "unparsed-card");
      card.append(node("p", "unparsed-text", "«" + String(fragment.text || "") + "»"), node("p", "unparsed-reason", fragment.reason || "Причина не указана")); target.append(card);
    }
  }
  function renderReview(report) {
    const notice = $("review-notice"); clear(notice);
    const issues = [];
    if (report?.conflicts?.length) issues.push(`Найдены противоречия: ${report.conflicts.length}. В базе есть утверждение и его явное отрицание. Требуется проверить исходные факты.`);
    for (const check of report?.graph_checks || []) {
      if ((check.acyclic_required ?? check.strict_order) && check.strict_cycle_vertices?.length) issues.push(`В отношении ${check.predicate} обнаружен недопустимый цикл. Проверьте исходные связи.`);
      if (check.comparison === "failed") issues.push(`Проверка ${check.predicate}: результат BFS не совпал с логическим выводом. Подробности — в полном отчёте.`);
    }
    if (report?.status === "needs_review" && !issues.length) issues.push("Результат требует проверки. Подробности — в полном JSON-отчёте.");
    if (report?.status === "verification_failed" && !issues.length) issues.push("Контрольная проверка не пройдена. Подробности — в полном JSON-отчёте.");
    if (issues.length) { notice.append(node("strong", "", "Нужно проверить")); for (const issue of issues) notice.append(node("p", "", issue)); }
    show(notice, issues.length > 0);
  }
  function renderGraphs(checks) {
    const target = $("graph-checks"); clear(target); show($("graph-section"), checks.length > 0);
    for (const check of checks) {
      const card = node("div", "graph-check");
      card.append(node("strong", "", check.predicate + (check.comparison === "passed" ? " · BFS и логический вывод совпадают" : " · проверка графа")));
      const acyclicRequired = check.acyclic_required ?? check.strict_order;
      const cycleCount = acyclicRequired ? (check.strict_cycle_vertices || []).length : (check.cycle_vertices || []).length;
      if (acyclicRequired) card.append(node("p", cycleCount ? "graph-warning" : "", cycleCount ? `Недопустимый цикл в отношении ${check.predicate}: ${fmt(cycleCount)} вершин.` : "Недопустимых циклов нет."));
      if (check.example_shortest_explanation?.labels?.length) card.append(node("p", "", "Пример кратчайшей цепочки: " + check.example_shortest_explanation.labels.join(" → ")));
      if (check.predicate === "requires" && check.prerequisite_order_labels?.length && check.strict_order && !cycleCount) card.append(node("p", "", "Возможный порядок изучения: " + check.prerequisite_order_labels.join(" → ")));
      target.append(card);
    }
  }
  function renderUsage(usage, model, demo, report) {
    const footer = $("usage-footer"); clear(footer);
    const pieces = [];
    if (demo) pieces.push("Демо · платных вызовов нет");
    else if (model) pieces.push("Модель: " + model);
    else if (report) pieces.push("Локальный вывод · без вызова LLM");
    if (usage) {
      if (usage.prompt_tokens != null) pieces.push(`Вход: ${fmt(usage.prompt_tokens)} токенов`);
      if (usage.completion_tokens != null) pieces.push(`Выход: ${fmt(usage.completion_tokens)} токенов`);
      if (usage.total_tokens != null) pieces.push(`Всего: ${fmt(usage.total_tokens)}`);
      const reasoningTokens = usage.completion_tokens_details?.reasoning_tokens ?? usage.reasoning_tokens;
      const cachedTokens = usage.prompt_tokens_details?.cached_tokens ?? usage.cached_tokens;
      if (reasoningTokens != null) pieces.push(`Рассуждения: ${fmt(reasoningTokens)}`);
      if (cachedTokens) pieces.push(`Кэш: ${fmt(cachedTokens)}`);
    }
    for (const piece of pieces) footer.append(node("span", "", piece));
    show(footer, pieces.length > 0);
  }
  function renderResult(result) {
    state.result = result;
    if (result.profile) state.profile = result.profile;
    const extraction = result.extraction || {}, report = result.report;
    if (typeof result.source_text === "string") { $("source-text").value = result.source_text; updateCharacters(); }
    const entities = report?.entities || extraction.entities || [];
    const unparsed = report?.unparsed || extraction.unparsed || [];
    state.entityMap = new Map(entities.map((entity) => [entity.id, entity]));
    const facts = report?.facts || (extraction.facts || []).map((fact, index) => ({ ...fact, id: `E${index + 1}`, origin: "extracted" }));
    state.factMap = new Map(facts.map((fact) => [fact.id, fact]));
    const initial = facts.filter((fact) => fact.origin !== "derived"), derived = facts.filter((fact) => fact.origin === "derived");
    $("json-editor").value = typeof result.raw_json === "string" ? result.raw_json : JSON.stringify(extraction, null, 2);
    show($("empty-state"), false); show($("fact-results")); show($("statistics")); show($("result-mode"));
    $("result-mode").textContent = result.demo ? "Демо" : report ? "Вывод выполнен" : "Только извлечение";
    $("result-mode").title = state.profile?.title || "";
    $("result-mode").className = "small-badge" + (result.demo ? "" : " badge-teal");
    const context = $("result-context");
    context.textContent = result.demo ? "Учебный пример: используются готовые факты, а не извлечение из вашего текста. Исходный текст заменён текстом примера." : !report ? "Факты извлечены и проверены. Логический вывод ещё не запускался — его можно выполнить во вкладке «JSON»." : "";
    show(context, Boolean(context.textContent));
    $("stat-initial").textContent = fmt(initial.length); $("stat-new").textContent = report ? fmt(derived.length) : "—";
    $("stat-entities").textContent = fmt(entities.length); $("stat-rounds").textContent = report ? fmt(report.statistics?.productive_rounds) : "—";
    $("initial-count").textContent = fmt(initial.length); $("derived-count").textContent = report ? fmt(derived.length) : "—";
    $("entities-count").textContent = fmt(entities.length); $("unparsed-count").textContent = fmt(unparsed.length);
    const initialTarget = $("initial-facts"), derivedTarget = $("derived-facts"); clear(initialTarget); clear(derivedTarget);
    for (const fact of initial) initialTarget.append(factCard(fact, false));
    for (const fact of derived) derivedTarget.append(factCard(fact, true));
    if (!initial.length) listEmpty(initialTarget, "В тексте не найдено фактов текущего словаря.");
    if (!derived.length) listEmpty(derivedTarget, report ? "По текущим правилам новых фактов нет." : "Запустите вывод во вкладке «JSON» — без повторного обращения к LLM.");
    $("derived-description").textContent = report ? "Следствия из фактов и фиксированных правил. Откройте факт, чтобы увидеть объяснение." : "На этом шаге выполнено только извлечение.";
    renderEntities(entities); renderUnparsed(unparsed); renderReview(report); renderGraphs(report?.graph_checks || []); renderUsage(result.usage, result.model, result.demo, report); setStep(true, Boolean(report));
  }
  function download(data, filename, mime = "application/json") {
    const blob = new Blob([data], { type: mime + ";charset=utf-8" }), url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = filename; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function bootstrap(profileId) {
    if (state.busy) return;
    const firstLoad = !state.bootstrapped;
    const customPrompt = !firstLoad && $("prompt-template").value !== (state.defaults.prompt_template || "");
    setBusy(true, firstLoad ? "Загружаем настройки…" : "Загружаем профиль знаний…");
    resetError();
    try {
      const data = await api("/api/bootstrap" + (profileId ? "?profile=" + encodeURIComponent(profileId) : ""));
      state.defaults = data.defaults || {}; state.hasApiKey = Boolean(data.configuration?.has_api_key ?? data.configuration?.ready);
      state.profile = data.profile || { id: profileId || "university", title: "Университет — расширенный", description: "", rule_descriptions: {} };
      state.profiles = data.profiles || [state.profile];
      state.maxChars = data.limits?.max_text_chars || 30000;
      $("source-text").value = state.defaults.text || "";
      if (firstLoad) $("model-name").value = data.configuration?.model || "";
      if (!customPrompt) $("prompt-template").value = state.defaults.prompt_template || "";
      $("schema-preview").textContent = state.defaults.schema_json || ""; $("rules-preview").textContent = state.defaults.rules_json || "";
      clearResult(); renderProfile(); selectTab("facts"); state.bootstrapped = true;
      updateAvailability(); updateCharacters();
    } catch (err) {
      $("knowledge-profile").value = state.profile?.id || "university";
      if (firstLoad) $("connection-status").lastElementChild.textContent = "Сервер недоступен";
      error(err.message || "Не удалось загрузить настройки приложения.");
    }
    finally { setBusy(false); }
  }

  for (const tab of tabs) {
    tab.addEventListener("click", () => selectTab(tab.dataset.view));
    tab.addEventListener("keydown", (event) => {
      const index = tabs.indexOf(tab); let next;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
      if (event.key === "Home") next = 0; if (event.key === "End") next = tabs.length - 1;
      if (next !== undefined) { event.preventDefault(); selectTab(tabs[next].dataset.view, true); }
    });
  }
  $("source-text").addEventListener("input", updateCharacters);
  $("knowledge-profile").addEventListener("change", () => bootstrap($("knowledge-profile").value));
  $("model-name").addEventListener("input", updateAvailability);
  $("json-editor").addEventListener("input", () => { $("download-json").disabled = state.busy || !$("json-editor").value.trim(); });
  $("analyze-button").addEventListener("click", () => run("analyze")); $("extract-button").addEventListener("click", () => run("extract"));
  $("infer-button").addEventListener("click", () => run("infer")); $("demo-button").addEventListener("click", () => run("demo"));
  $("close-error").addEventListener("click", resetError); $("reset-prompt").addEventListener("click", () => { $("prompt-template").value = state.defaults.prompt_template || ""; });
  $("download-report").addEventListener("click", () => { if (state.result) download(JSON.stringify(state.result, null, 2), "результат-обработки.json"); });
  $("download-json").addEventListener("click", () => download($("json-editor").value, "извлечённые-факты.json"));
  bootstrap();
})();
