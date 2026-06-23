const dropZone      = document.getElementById("drop-zone");
const fileInput     = document.getElementById("file-input");
const uploadStatus  = document.getElementById("upload-status");
const docList       = document.getElementById("doc-list");
const askForm       = document.getElementById("ask-form");
const questionInput = document.getElementById("question-input");
const askBtn        = document.getElementById("ask-btn");
const chatMessages  = document.getElementById("chat-messages");
const docFilter     = document.getElementById("doc-filter");
const sessionList   = document.getElementById("session-list");
const newChatBtn    = document.getElementById("new-chat-btn");

// ── Chat sessions (persisted in the browser via localStorage) ────────────────
// Stored client-side (not on the server) because the deployment filesystem is
// ephemeral; localStorage survives page reloads, restarts, and redeploys.

const STORE_KEY = "docintel.sessions.v1";
let sessions = [];   // [{ id, title, messages: [{role, content, sources?, warning?}], doc }]
let activeId = null;

function loadStore() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORE_KEY));
    if (raw && Array.isArray(raw.sessions) && raw.sessions.length) {
      sessions = raw.sessions;
      activeId = raw.activeId;
    }
  } catch {}
  if (!sessions.length) createSession();
  if (!sessions.find(s => s.id === activeId)) activeId = sessions[0].id;
}

function saveStore() {
  localStorage.setItem(STORE_KEY, JSON.stringify({ sessions, activeId }));
}

function activeSession() {
  return sessions.find(s => s.id === activeId);
}

function createSession() {
  const id = Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  const s = { id, title: "New chat", messages: [], doc: "" };
  sessions.unshift(s);
  activeId = id;
  saveStore();
  return s;
}

function switchSession(id) {
  activeId = id;
  saveStore();
  renderSessionList();
  renderActiveSession();
}

function deleteSession(id) {
  sessions = sessions.filter(s => s.id !== id);
  if (!sessions.length) createSession();
  else if (activeId === id) activeId = sessions[0].id;
  saveStore();
  renderSessionList();
  renderActiveSession();
}

function renameSession(id, title) {
  const s = sessions.find(x => x.id === id);
  if (s) { s.title = (title || "").trim() || s.title; saveStore(); }
  renderSessionList();
}

// ── Session sidebar rendering ────────────────────────────────────────────────

function renderSessionList() {
  sessionList.innerHTML = "";
  sessions.forEach(s => {
    const li = document.createElement("li");
    li.className = "session-item" + (s.id === activeId ? " active" : "");
    li.addEventListener("click", () => switchSession(s.id));

    const title = document.createElement("span");
    title.className = "session-title";
    title.textContent = s.title;

    const actions = document.createElement("span");
    actions.className = "session-actions";

    const editBtn = document.createElement("button");
    editBtn.className = "session-btn";
    editBtn.title = "Rename chat";
    editBtn.textContent = "✎";
    editBtn.addEventListener("click", (e) => { e.stopPropagation(); startRename(li, s); });

    const delBtn = document.createElement("button");
    delBtn.className = "session-btn";
    delBtn.title = "Delete chat";
    delBtn.textContent = "×";
    delBtn.addEventListener("click", (e) => { e.stopPropagation(); deleteSession(s.id); });

    actions.appendChild(editBtn);
    actions.appendChild(delBtn);
    li.appendChild(title);
    li.appendChild(actions);
    sessionList.appendChild(li);
  });
}

function startRename(li, s) {
  const input = document.createElement("input");
  input.className = "session-edit-input";
  input.value = s.title;
  input.addEventListener("click", (e) => e.stopPropagation());
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") input.blur();
    if (e.key === "Escape") renderSessionList();
  });
  input.addEventListener("blur", () => renameSession(s.id, input.value));
  li.innerHTML = "";
  li.appendChild(input);
  input.focus();
  input.select();
}

// ── Active session → chat area ───────────────────────────────────────────────

function renderActiveSession() {
  const s = activeSession();
  chatMessages.innerHTML = "";
  if (!s || !s.messages.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "Upload a document, then ask a question about it.";
    chatMessages.appendChild(empty);
  } else {
    s.messages.forEach(m => {
      if (m.role === "user") {
        appendMessage("user", m.content);
      } else {
        const div = appendMessage("assistant", m.content, m.warning);
        if (m.sources && m.sources.length) attachSources(div, m.sources);
      }
    });
  }
  docFilter.value = (s && s.doc) || "";
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

newChatBtn.addEventListener("click", () => {
  createSession();
  renderSessionList();
  renderActiveSession();
  questionInput.focus();
});

// ── On load ──────────────────────────────────────────────────────────────────

loadStore();
renderSessionList();
renderActiveSession();
loadDocuments();

async function loadDocuments() {
  try {
    const res = await fetch("/documents");
    const data = await res.json();
    data.documents.forEach(name => addDocToList(name));
    // Re-apply the active session's saved document filter now that options exist.
    docFilter.value = (activeSession() && activeSession().doc) || "";
  } catch {}
}

// ── Upload ───────────────────────────────────────────────────────────────────

dropZone.addEventListener("click", (e) => {
  if (e.target.tagName === "LABEL") return;  // label already triggers the input
  fileInput.click();
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
});

async function uploadFile(file) {
  const allowed = [".pdf", ".docx", ".txt", ".md"];
  if (!allowed.some(ext => file.name.toLowerCase().endsWith(ext))) {
    showStatus("Supported formats: PDF, DOCX, TXT, MD", "error");
    return;
  }

  showStatus(`Uploading "${file.name}"…`, "loading");
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) { showStatus(data.error || "Upload failed.", "error"); return; }
    showStatus(`✓ ${data.message} (${data.chunks} chunks)`, "success");
    addDocToList(file.name);
  } catch {
    showStatus("Network error during upload.", "error");
  }

  fileInput.value = "";
}

function showStatus(msg, type) {
  uploadStatus.textContent = msg;
  uploadStatus.className = `status-msg ${type}`;
}

function addDocToList(name) {
  if ([...docList.querySelectorAll("li")].find(li => li.dataset.name === name)) return;

  const li = document.createElement("li");
  li.dataset.name = name;

  const label = document.createElement("span");
  label.className = "doc-name";
  label.textContent = name;

  const btn = document.createElement("button");
  btn.className = "delete-btn";
  btn.title = "Remove document";
  btn.textContent = "×";
  btn.addEventListener("click", () => removeDoc(name, li));

  li.appendChild(label);
  li.appendChild(btn);
  docList.appendChild(li);

  // Add to filter dropdown
  const opt = document.createElement("option");
  opt.value = name;
  opt.textContent = name;
  opt.dataset.name = name;
  docFilter.appendChild(opt);
}

async function removeDoc(name, listItem) {
  try {
    const res = await fetch(`/document/${encodeURIComponent(name)}`, { method: "DELETE" });
    const data = await res.json();
    if (res.ok) {
      listItem.remove();
      showStatus(`Removed "${name}".`, "success");
      const opt = docFilter.querySelector(`option[data-name="${name}"]`);
      if (opt) opt.remove();
    } else {
      showStatus(data.error || "Failed to remove document.", "error");
    }
  } catch {
    showStatus("Network error during removal.", "error");
  }
}

// ── Chat ─────────────────────────────────────────────────────────────────────

// Auto-grow the textarea as the user types (up to a max), and send on Enter
// (Shift+Enter inserts a newline) — like a modern chat input.
const MAX_INPUT_HEIGHT = 160;

function autoResizeInput() {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, MAX_INPUT_HEIGHT) + "px";
}

questionInput.addEventListener("input", autoResizeInput);

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askForm.requestSubmit();
  }
});

askForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  const s = activeSession();
  const history = s.messages.slice(-6).map(m => ({ role: m.role, content: m.content }));

  questionInput.value = "";
  questionInput.style.height = "auto";  // reset the grown textarea back to one line
  askBtn.disabled = true;
  clearEmptyState();

  appendMessage("user", question);
  s.messages.push({ role: "user", content: question });
  if (s.title === "New chat") { s.title = question.slice(0, 40); renderSessionList(); }

  const selectedDoc = docFilter.value || null;
  s.doc = docFilter.value || "";
  saveStore();

  const thinking = appendThinking();

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history, document: selectedDoc }),
    });

    thinking.remove();

    if (!res.ok) {
      const data = await res.json();
      const msg = data.error || "Something went wrong.";
      appendMessage("assistant", msg, true);
      s.messages.push({ role: "assistant", content: msg, warning: true });
      saveStore();
      askBtn.disabled = false;
      return;
    }

    const { answer, sources } = await readStream(res);
    s.messages.push({ role: "assistant", content: answer, sources });
    saveStore();

  } catch (err) {
    thinking.remove();
    const msg = "Network error. Is the server running?";
    appendMessage("assistant", msg, true);
    s.messages.push({ role: "assistant", content: msg, warning: true });
    saveStore();
  }

  askBtn.disabled = false;
  questionInput.focus();
});

async function readStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answerText = "";
  let bubble = null;
  let messageDiv = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();  // hold incomplete line

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6);

      try {
        const data = JSON.parse(payload);

        if (data.token) {
          if (!bubble) {
            messageDiv = document.createElement("div");
            messageDiv.className = "message assistant";
            bubble = document.createElement("div");
            bubble.className = "bubble";
            messageDiv.appendChild(bubble);
            chatMessages.appendChild(messageDiv);
          }
          answerText += data.token;
          bubble.textContent = answerText;
          chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        if (data.done) {
          if (data.sources && data.sources.length > 0 && messageDiv) {
            attachSources(messageDiv, data.sources);
          }
          return { answer: answerText, sources: data.sources || [] };
        }
      } catch {}
    }
  }

  return { answer: answerText, sources: [] };
}

function attachSources(messageDiv, sources) {
  const sourceDiv = document.createElement("div");
  sourceDiv.className = "sources";
  sources.forEach(s => {
    const tag = document.createElement("span");
    tag.className = "source-tag";
    tag.textContent = `${s.source} · p.${s.page_number}`;
    sourceDiv.appendChild(tag);
  });
  messageDiv.appendChild(sourceDiv);
}

function appendMessage(role, text, isWarning = false) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = `bubble${isWarning ? " warning" : ""}`;
  bubble.textContent = text;
  div.appendChild(bubble);
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return div;
}

function appendThinking() {
  const div = document.createElement("div");
  div.className = "thinking";
  div.textContent = "Thinking…";
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return div;
}

function clearEmptyState() {
  const empty = chatMessages.querySelector(".empty-state");
  if (empty) empty.remove();
}
