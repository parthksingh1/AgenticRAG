# Chrome extension

Two things, from any page:

- **Ask** your corpus a question, optionally with the page's text as context.
- **Save** the page into the corpus.

## Install

```
chrome://extensions → Developer mode → Load unpacked → this directory
```

Then open the options page and set your API URL and key.

## Why `activeTab` rather than `<all_urls>`

`activeTab` grants access to one page, only when you click the extension, and
only until you navigate away. `<all_urls>` grants permanent read access to every
page you ever visit — far more than "ask a question about this page" needs, and
the difference between a tool and a liability.

The host permission is `localhost:8000` by default. Point it at your deployment
by editing the manifest; an extension permitted to talk to any host is one that
can send your browsing anywhere.

## Where the key is stored

`chrome.storage.local` — scoped to the extension and unreadable by pages.
Deliberately not `storage.sync`: an API key is a credential, and syncing puts a
copy on every machine signed into the same Google account, including ones you
have forgotten about.

## Why the network calls live in the service worker

A popup is destroyed the moment it loses focus, taking any in-flight `fetch`
with it. For a thirty-second RAG answer that is most of the time, so the popup
sends a message and the worker does the work.

## Page context is a separate field

The page's text is sent as `page_context`, not prepended to the question.
Embedding a whole article into the retrieval query destroys the query — the
signal from eight words of question disappears into eight thousand characters of
article.
