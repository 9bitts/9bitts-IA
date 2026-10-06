const state = {
  me: null,
  status: null,
  conversations: [],
  activeId: null,
  model: null,
  streaming: false,
  abort: null,
};

const $ = (id) => document.getElementById(id);

const TOOL_LABEL = {
  search_web: "Busca na web",
  search_documents: "Documentos",
  read_file: "Arquivo",
  remember: "Memória",
};

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: options.body && !(options.body instanceof FormData)
      ? { "Content-Type": "application/json", ...(options.headers || {}) }
      : options.headers,
    ...options,
  });
  if (response.status === 204) return null;
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.detail || "Não foi possível concluir.");
    error.status = response.status;
    throw error;
  }
  return data;
}

function showLogin(message = "") {
  $("login").classList.remove("hidden");
  $("shell").classList.add("hidden");
  $("login-error").textContent = message;
}

function showApp() {
  $("login").classList.add("hidden");
  $("shell").classList.remove("hidden");
  $("logout").classList.toggle("hidden", state.me.deploy_mode !== "product");
}

function renderRuntime() {
  const status = state.status;
  if (!status) return;
  const compute = status.compute || {};
  let where = "CPU";
  if (compute.using_gpu && compute.vulkan) where = "Radeon via Vulkan";
  else if (compute.vulkan) where = "Vulkan disponível";
  else if (compute.using_gpu) where = "GPU";
  else if (compute.igpu_dropped && compute.radeon_860m) where = "Radeon vista, GPU integrada desligada";
  const place = status.inference_location === "remote" ? "servidor GPU" : "este notebook";
  const online = status.ollama_online ? "modelo no ar" : "modelo offline";
  $("runtime-line").textContent = `${online} · ${where} · ${place}`;
}

function renderModels() {
  const box = $("models");
  box.innerHTML = "";
  const models = state.status?.models;
  if (!models) return;
  for (const [id, label] of [[models.fast, "Rápido"], [models.quality, "Melhor"]]) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.title = id;
    const ready = id === models.fast ? models.fast_ready : models.quality_ready;
    button.disabled = !ready;
    button.setAttribute("aria-pressed", String(state.model === id));
    button.addEventListener("click", () => chooseModel(id));
    box.appendChild(button);
  }
}

function renderConversations() {
  const list = $("conv-list");
  list.innerHTML = "";
  for (const conversation of state.conversations) {
    const row = document.createElement("div");
    row.className = "conv" + (conversation.id === state.activeId ? " active" : "");
    const open = document.createElement("button");
    open.type = "button";
    open.className = "open";
    open.textContent = conversation.title || "Nova conversa";
    open.addEventListener("click", () => openConversation(conversation.id));
    const trash = document.createElement("button");
    trash.type = "button";
    trash.className = "trash";
    trash.setAttribute("aria-label", "Apagar conversa");
    trash.textContent = "×";
    trash.addEventListener("click", () => removeConversation(conversation.id));
    row.append(open, trash);
    list.appendChild(row);
  }
}

function renderMessages(messages) {
  const transcript = $("transcript");
  const thread = document.createElement("div");
  thread.className = "thread";
  if (!messages.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = state.status?.ollama_online
      ? "Pergunte como se fosse um assistente seu. O modelo roda na máquina configurada."
      : "O Ollama não está respondendo neste endereço. Abra o Ollama e tente de novo.";
    thread.appendChild(empty);
  }
  for (const message of messages) thread.appendChild(messageNode(message));
  transcript.replaceChildren(thread);
  transcript.scrollTop = transcript.scrollHeight;
}

function messageNode(message) {
  const item = document.createElement("article");
  item.className = "msg " + (message.role === "user" ? "user" : "assistant");
  if (message.tool_trace?.length) {
    const tools = document.createElement("div");
    tools.className = "tools";
    for (const tool of message.tool_trace) {
      const chip = document.createElement("span");
      chip.className = "chip";
      const label = TOOL_LABEL[tool.name] || tool.name;
      chip.textContent = tool.preview ? `${label}: ${tool.preview}` : label;
      tools.appendChild(chip);
    }
    item.appendChild(tools);
  }
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = message.content || "";
  item.appendChild(bubble);
  return item;
}

async function refreshConversations() {
  state.conversations = await api("/api/conversations");
  renderConversations();
}

async function openConversation(id) {
  state.activeId = id;
  const conversation = state.conversations.find((item) => item.id === id);
  $("conv-title").disabled = false;
  $("conv-title").value = conversation?.title || "";
  if (conversation?.model) state.model = conversation.model;
  renderModels();
  renderConversations();
  const messages = await api(`/api/conversations/${id}/messages`);
  renderMessages(messages);
}

function newChat() {
  state.activeId = null;
  $("conv-title").value = "";
  $("conv-title").disabled = true;
  renderConversations();
  renderMessages([]);
  $("input").focus();
}

async function chooseModel(model) {
  state.model = model;
  renderModels();
  if (!state.activeId) return;
  const updated = await api(`/api/conversations/${state.activeId}`, {
    method: "PATCH",
    body: JSON.stringify({ model }),
  });
  const conversation = state.conversations.find((item) => item.id === state.activeId);
  if (conversation) conversation.model = updated.model;
}

async function removeConversation(id) {
  await api(`/api/conversations/${id}`, { method: "DELETE" });
  if (state.activeId === id) newChat();
  await refreshConversations();
}

function growInput() {
  const input = $("input");
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
}

async function sendMessage(text) {
  if (state.streaming) return;
  if (!state.activeId) {
    const created = await api("/api/conversations", {
      method: "POST",
      body: JSON.stringify({ model: state.model }),
    });
    state.activeId = created.id;
    $("conv-title").disabled = false;
    await refreshConversations();
  }
  const transcript = $("transcript");
  let thread = transcript.querySelector(".thread");
  if (!thread) {
    thread = document.createElement("div");
    thread.className = "thread";
    transcript.replaceChildren(thread);
  }
  thread.querySelector(".empty")?.remove();
  thread.appendChild(messageNode({ role: "user", content: text }));
  const live = messageNode({ role: "assistant", content: "", tool_trace: [] });
  const bubble = live.querySelector(".bubble");
  const tools = document.createElement("div");
  tools.className = "tools";
  live.prepend(tools);
  thread.appendChild(live);
  transcript.scrollTop = transcript.scrollHeight;

  state.streaming = true;
  $("send").textContent = "Parar";
  state.abort = new AbortController();
  try {
    const response = await fetch(`/api/conversations/${state.activeId}/messages`, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: text, model: state.model }),
      signal: state.abort.signal,
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || "Não foi possível enviar.");
    }
    await readStream(response, bubble, tools);
  } catch (error) {
    if (error.name !== "AbortError") bubble.textContent = error.message;
  } finally {
    state.streaming = false;
    state.abort = null;
    $("send").textContent = "Enviar";
    await refreshConversations();
    const current = state.conversations.find((item) => item.id === state.activeId);
    if (current) $("conv-title").value = current.title;
  }
}

async function readStream(response, bubble, tools) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let split = buffer.indexOf("\n\n");
    while (split >= 0) {
      applyEvent(buffer.slice(0, split), bubble, tools);
      buffer = buffer.slice(split + 2);
      split = buffer.indexOf("\n\n");
    }
    const transcript = $("transcript");
    transcript.scrollTop = transcript.scrollHeight;
  }
}

function applyEvent(raw, bubble, tools) {
  let event = "message";
  let data = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!data) return;
  const payload = JSON.parse(data);
  if (event === "token") bubble.textContent += payload.text || "";
  if (event === "meta" && payload.title) $("conv-title").value = payload.title;
  if (event === "tool") {
    const label = TOOL_LABEL[payload.name] || payload.name;
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.dataset.name = payload.name + (payload.status || "");
    chip.textContent = payload.status === "done" && payload.preview
      ? `${label}: ${payload.preview}`
      : `${label}…`;
    const previous = tools.querySelector(`[data-name="${payload.name}start"]`);
    if (payload.status === "done" && previous) previous.replaceWith(chip);
    else tools.appendChild(chip);
  }
  if (event === "error") bubble.textContent = payload.message || bubble.textContent;
}

async function openSettings() {
  const settings = await api("/api/settings");
  $("persona").value = settings.persona || "";
  $("folder").value = settings.folder_path || "";
  $("folder-block").classList.toggle("hidden", !settings.folder_access);
  $("upload-block").classList.toggle("hidden", settings.folder_access);
  $("folder-note").textContent = settings.folder_locked_reason || "Só entram arquivos desta pasta.";
  renderKnowledge(settings.knowledge);
  const deploy = settings.deploy;
  $("deploy").innerHTML = `
    <strong>Como isso vira produto</strong><br />
    Modo atual: ${escapeHtml(deploy.mode)}.<br />
    Inferência: ${deploy.inference_location === "remote" ? "servidor remoto" : "esta máquina"}.
    ${deploy.ollama_base_url ? `<br />Ollama: ${escapeHtml(deploy.ollama_base_url)}` : ""}
    <br />Banco: ${escapeHtml(deploy.database)}.
    ${deploy.daily_limit ? `<br />Limite: ${deploy.daily_limit} mensagens por dia.` : ""}
    <br />Para clientes: DEPLOY_MODE=product, DATABASE_URL no Postgres e OLLAMA_BASE_URL no servidor com GPU.
  `;
  $("settings-error").textContent = "";
  $("admin-block").classList.toggle("hidden", !(state.me.is_admin && state.me.deploy_mode === "product"));
  if (state.me.is_admin && state.me.deploy_mode === "product") {
    const users = await api("/api/admin/users");
    $("user-list").innerHTML = users.map((user) => `<li>${escapeHtml(user.name)} · ${escapeHtml(user.email)}</li>`).join("");
  }
  $("settings").showModal();
  if (settings.knowledge.state === "running") pollKnowledge();
}

function renderKnowledge(knowledge) {
  const stateLabel = knowledge.state === "running" ? "Indexando…" : knowledge.state === "error" ? knowledge.error : `${knowledge.chunks} trechos`;
  $("index-line").textContent = stateLabel || "";
  $("sources").innerHTML = (knowledge.sources || []).map((source) => `<li>${escapeHtml(source)}</li>`).join("");
}

async function pollKnowledge() {
  if (!$("settings").open) return;
  const knowledge = await api("/api/knowledge");
  renderKnowledge(knowledge);
  if (knowledge.state === "running") setTimeout(pollKnowledge, 1000);
}

function bind() {
  $("login-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await api("/api/login", {
        method: "POST",
        body: JSON.stringify({
          email: $("login-email").value,
          password: $("login-password").value,
        }),
      });
      await boot();
    } catch (error) {
      $("login-error").textContent = error.message;
    }
  });

  $("new-chat").addEventListener("click", newChat);
  $("logout").addEventListener("click", async () => {
    await api("/api/logout", { method: "POST" });
    showLogin();
  });
  $("open-settings").addEventListener("click", () => openSettings().catch((error) => alert(error.message)));
  $("conv-title").addEventListener("change", async () => {
    if (!state.activeId) return;
    const updated = await api(`/api/conversations/${state.activeId}`, {
      method: "PATCH",
      body: JSON.stringify({ title: $("conv-title").value }),
    });
    const conversation = state.conversations.find((item) => item.id === state.activeId);
    if (conversation) conversation.title = updated.title;
    renderConversations();
  });
  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.streaming) {
      state.abort?.abort();
      return;
    }
    const text = $("input").value.trim();
    if (!text) return;
    $("input").value = "";
    growInput();
    sendMessage(text).catch((error) => alert(error.message));
  });
  $("input").addEventListener("input", growInput);
  $("input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });
  $("save-persona").addEventListener("click", async () => {
    try {
      await api("/api/settings", { method: "PUT", body: JSON.stringify({ persona: $("persona").value }) });
      $("settings-error").textContent = "Persona salva.";
      $("settings-error").classList.add("ok");
    } catch (error) {
      $("settings-error").classList.remove("ok");
      $("settings-error").textContent = error.message;
    }
  });
  $("save-folder").addEventListener("click", async () => {
    try {
      const settings = await api("/api/settings", {
        method: "PUT",
        body: JSON.stringify({ folder_path: $("folder").value }),
      });
      renderKnowledge(settings.knowledge);
      if (settings.knowledge.state === "running") pollKnowledge();
    } catch (error) {
      $("settings-error").textContent = error.message;
    }
  });
  $("reindex").addEventListener("click", async () => {
    try {
      await api("/api/knowledge/reindex", { method: "POST" });
      pollKnowledge();
    } catch (error) {
      $("settings-error").textContent = error.message;
    }
  });
  $("upload").addEventListener("change", async () => {
    const file = $("upload").files[0];
    if (!file) return;
    const body = new FormData();
    body.append("file", file);
    try {
      const result = await fetch("/api/knowledge/documents", { method: "POST", body, credentials: "same-origin" });
      const data = await result.json();
      if (!result.ok) throw new Error(data.detail || "Falha no envio.");
      renderKnowledge({ state: "done", chunks: data.chunks, sources: data.sources, error: null });
    } catch (error) {
      $("settings-error").textContent = error.message;
    }
  });
  $("create-user").addEventListener("click", async () => {
    try {
      await api("/api/admin/users", {
        method: "POST",
        body: JSON.stringify({
          name: $("new-name").value,
          email: $("new-email").value,
          password: $("new-password").value,
        }),
      });
      await openSettings();
    } catch (error) {
      $("settings-error").textContent = error.message;
    }
  });
}

async function boot() {
  try {
    state.me = await api("/api/me");
  } catch (error) {
    if (error.status === 401) {
      showLogin();
      return;
    }
    throw error;
  }
  showApp();
  state.status = await api("/api/status");
  state.model = state.status.models.fast;
  renderRuntime();
  renderModels();
  await refreshConversations();
  newChat();
}

bind();
boot().catch((error) => {
  document.body.textContent = error.message;
});
