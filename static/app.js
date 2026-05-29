const dropZone      = document.getElementById("drop-zone");
const fileInput     = document.getElementById("file-input");
const uploadStatus  = document.getElementById("upload-status");
const docList       = document.getElementById("doc-list");
const askForm       = document.getElementById("ask-form");
const questionInput = document.getElementById("question-input");
const askBtn        = document.getElementById("ask-btn");
const chatMessages  = document.getElementById("chat-messages");
const docFilter     = document.getElementById("doc-filter");

let conversationHistory = [];  // [{role, content}, ...]

// ── On load: fetch persisted documents ──────────────────────────────────────

async function loadDocuments() {
  try {
    const res = await fetch("/documents");
    const data = await res.json();
    data.documents.forEach(name => addDocToList(name));
  } catch {}
}
loadDocuments();

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
    clearEmptyState();
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
      // Remove from dropdown
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

askForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  questionInput.value = "";
  askBtn.disabled = true;
  clearEmptyState();

  appendMessage("user", question);
  const thinking = appendThinking();

  const selectedDoc = docFilter.value || null;

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        history: conversationHistory.slice(-6),  // last 3 exchanges
        document: selectedDoc,
      }),
    });

    thinking.remove();

    if (!res.ok) {
      const data = await res.json();
      appendMessage("assistant", data.error || "Something went wrong.", true);
      askBtn.disabled = false;
      return;
    }

    const { answer, sources } = await readStream(res);

    // Add to conversation history (without context block — just Q&A)
    conversationHistory.push({ role: "user", content: question });
    conversationHistory.push({ role: "assistant", content: answer });

  } catch (err) {
    thinking.remove();
    appendMessage("assistant", "Network error. Is the server running?", true);
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
            const sourceDiv = document.createElement("div");
            sourceDiv.className = "sources";
            data.sources.forEach(s => {
              const tag = document.createElement("span");
              tag.className = "source-tag";
              tag.textContent = `${s.source} · p.${s.page_number}`;
              sourceDiv.appendChild(tag);
            });
            messageDiv.appendChild(sourceDiv);
          }
          return { answer: answerText, sources: data.sources || [] };
        }
      } catch {}
    }
  }

  return { answer: answerText, sources: [] };
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
