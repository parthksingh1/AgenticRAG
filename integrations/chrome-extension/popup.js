/**
 * The popup.
 *
 * It messages the service worker rather than calling the API itself. A popup is
 * destroyed the moment it loses focus, so a fetch started here would be
 * cancelled mid-flight — which for a thirty-second answer is most of the time.
 */

const q = document.getElementById("q");
const out = document.getElementById("out");
const sources = document.getElementById("sources");
const askButton = document.getElementById("ask");
const saveButton = document.getElementById("save");
const usePage = document.getElementById("usePage");

askButton.addEventListener("click", run);

q.addEventListener("keydown", (e) => {
  // Cmd/Ctrl+Enter sends. Plain Enter must insert a newline: this is a textarea
  // because questions are sometimes two sentences.
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run();
});

saveButton.addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.url) return;

  setBusy(true, "Saving...");
  const result = await chrome.runtime.sendMessage({ type: "save", url: tab.url });
  setBusy(false);
  render(result.error ? { error: result.error } : { content: "Saved. Indexing has been queued." });
});

async function run() {
  const question = q.value.trim();
  if (!question) return;

  setBusy(true, "Thinking...");
  const context = usePage.checked ? await pageText() : "";
  const answer = await chrome.runtime.sendMessage({ type: "ask", question, context });
  setBusy(false);
  render(answer);
}

/** Read the visible text of the active tab, using activeTab's one-click grant. */
async function pageText() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return "";

  try {
    const [result] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => document.body.innerText.slice(0, 8000),
    });
    return result?.result ?? "";
  } catch {
    // Chrome refuses injection into its own pages and the web store. Asking
    // without page context still works, so this is not an error worth showing.
    return "";
  }
}

function setBusy(busy, message) {
  askButton.disabled = busy;
  saveButton.disabled = busy;
  if (busy) {
    out.textContent = message;
    out.className = "";
    sources.textContent = "";
  }
}

function render(answer) {
  if (answer.error) {
    out.textContent = answer.error;
    out.className = "error";
    return;
  }

  out.className = "";
  out.textContent = answer.content ?? "No answer.";

  const citations = answer.citations ?? [];
  // Stated explicitly when there are none. An uncited answer looks identical to
  // a cited one in a small popup, and nobody checks.
  sources.textContent = citations.length
    ? citations.map((c) => `[${c.index}] ${c.document_title}`).join("\n")
    : "No sources - treat with caution.";
}
