/**
 * Service worker: the context menu and every API call.
 *
 * All network calls live here rather than in the popup, because a popup is
 * destroyed the moment it loses focus. A fetch started in the popup is
 * cancelled mid-flight when the user clicks away — which, for a thirty-second
 * RAG answer, is most of the time.
 */

const DEFAULTS = { apiUrl: "http://localhost:8000", apiKey: "" };

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "agrag-ask-selection",
    title: 'Ask AgenticRAG about "%s"',
    contexts: ["selection"],
  });
  chrome.contextMenus.create({
    id: "agrag-save-page",
    title: "Save this page to AgenticRAG",
    contexts: ["page"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "agrag-ask-selection" && info.selectionText) {
    const answer = await ask(info.selectionText, "");
    await notify(answer.error ? "Failed" : "Answer ready", answer.error ?? answer.content ?? "");
  } else if (info.menuItemId === "agrag-save-page" && tab?.url) {
    const result = await savePage(tab.url);
    await notify(result.error ? "Could not save" : "Saved", result.error ?? "Queued for indexing.");
  }
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  // Returning true keeps the channel open for the async reply. Without it the
  // popup receives `undefined` the moment this listener returns.
  if (message.type === "ask") {
    ask(message.question, message.context ?? "").then(sendResponse);
    return true;
  }
  if (message.type === "save") {
    savePage(message.url).then(sendResponse);
    return true;
  }
  return false;
});

async function settings() {
  return { ...DEFAULTS, ...(await chrome.storage.local.get(Object.keys(DEFAULTS))) };
}

async function ask(question, pageContext) {
  const { apiUrl, apiKey } = await settings();
  if (!apiKey) return { error: "Set your API key in the extension options." };

  // The page's text goes in its own field, not prepended to the question.
  // Embedding a whole article into the retrieval query destroys the query.
  const body = { message: question };
  if (pageContext) body.page_context = pageContext.slice(0, 8000);

  try {
    const response = await fetch(`${apiUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify(body),
    });
    if (!response.ok) return { error: `The API returned ${response.status}.` };
    return await response.json();
  } catch {
    return { error: "Could not reach the API. Is it running?" };
  }
}

async function savePage(url) {
  const { apiUrl, apiKey } = await settings();
  if (!apiKey) return { error: "Set your API key in the extension options." };

  try {
    const response = await fetch(`${apiUrl}/api/documents/url`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({ url }),
    });
    if (!response.ok) return { error: `The API returned ${response.status}.` };
    return await response.json();
  } catch {
    return { error: "Could not reach the API." };
  }
}

async function notify(title, message) {
  // Best-effort: the notifications permission is not requested, because the
  // popup shows the result anyway and an unnecessary permission is one more
  // thing a reviewer has to justify.
  try {
    await chrome.notifications?.create({
      type: "basic",
      iconUrl: "icon128.png",
      title,
      message: String(message).slice(0, 300),
    });
  } catch {
    /* no notifications permission; the popup still renders the result */
  }
}
