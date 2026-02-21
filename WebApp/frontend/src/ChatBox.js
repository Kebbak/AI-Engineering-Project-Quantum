import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import {
  FiCopy, FiCheck, FiPaperclip, FiMic, FiImage, FiSend, FiPlus, FiRefreshCw,
} from "react-icons/fi";

/* ---------- Small helper: Code block with copy button ---------- */
function CodeBlock({ language = "text", value = "" }) {
  const [copied, setCopied] = useState(false);

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1100);
    } catch {}
  };

  return (
    <div className="my-2 rounded-xl overflow-hidden border border-slate-700/40 bg-[#0b1021]">
      <div className="flex items-center justify-between px-3 py-2 border-b border-slate-700/50 text-slate-200 text-xs">
        <span className="uppercase tracking-wider">{language}</span>
        <button
          onClick={onCopy}
          className="inline-flex items-center gap-1 px-2 py-1 rounded-md border border-white/15 hover:bg-white/10"
        >
          {copied ? <FiCheck size={14} /> : <FiCopy size={14} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <SyntaxHighlighter
        language={language}
        style={oneDark}
        customStyle={{ margin: 0, background: "transparent" }}
        wrapLongLines
      >
        {value}
      </SyntaxHighlighter>
    </div>
  );
}

/* ---------- One message bubble (user/assistant) ---------- */
function Message({ role, content, citations }) {
    // Filter out single-letter or empty citations
    const filteredCitations = (citations || []).filter(c => typeof c === 'string' && c.trim().length > 2);
    // Debug log to inspect citations
    if (filteredCitations.length > 0) {
      console.log('Message citations:', filteredCitations);
    }
  const components = {
    code({ inline, className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || "");
      const text = String(children).replace(/\n$/, "");
      if (inline) {
        return (
          <code className="px-1.5 py-0.5 rounded-md bg-slate-200/60 dark:bg-slate-700/60">
            {children}
          </code>
        );
      }
      return <CodeBlock language={match ? match[1] : "text"} value={text} />;
    },
  };

  // ...existing code...
  return (
    <div className={`flex mb-3 ${role === "user" ? "justify-end" : "justify-start"}`}>
      <div
        className={[
          "px-3 py-2 rounded-2xl text-[0.95rem] max-w-[80vw] sm:max-w-2xl leading-relaxed shadow",
          role === "user"
            ? "bg-blue-600 text-white rounded-br-none"
            : "bg-white border border-slate-200 rounded-bl-none",
        ].join(" ")}
      >
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
          {content}
        </ReactMarkdown>

        {!!filteredCitations.length && (
          <div className="pt-2 mt-2 border-t text-xs text-slate-500">
            <span className="font-semibold">Citations:</span>
            <ul className="list-disc ml-4 mt-1">
              {filteredCitations.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------- Main ChatBox component (with fixed footer) ---------- */
export default function ChatBox() {
  const [conversations, setConversations] = useState([]);
  const [activeConversation, setActiveConversation] = useState(1);

  // Store messages per conversation id
  const [store, setStore] = useState({ 1: [], 2: [], 3: [], 4: [], 5: [], 6: [] });

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scroller = useRef(null);

  // Smooth autoscroll on new messages
  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [store, activeConversation]);

  // Initial greeting
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("http://localhost:5000/init");
        const data = await res.json();
        setStore(prev => ({
          ...prev,
          [activeConversation]: [
            ...(prev[activeConversation] || []),
            { id: Date.now(), role: "assistant", content: data.greeting || "Hello!" },
          ],
        }));
      } catch {
        setStore(prev => ({
          ...prev,
          [activeConversation]: [
            ...(prev[activeConversation] || []),
            {
              id: Date.now(),
              role: "assistant",
              content:
                "Welcome! I can answer questions about company policies. Ask me about Code of Conduct, Holidays, or Expenses.",
            },
          ],
        }));
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // run once

  const sanitize = (text = "") =>
    text
      .split(/\n?[-=]{10,}\n?/g) // remove big separator lines
      .map(s =>
        s.replace(/Debug:.*|\[warn\].*|Falling back to retrieval-only mode.*/gi, "")
      )
      .join("\n")
      .trim();

  const messages = store[activeConversation] || [];

  const handleSend = async (e) => {
    e.preventDefault();
    const question = input.trim();
    if (!question) return;

    const userMsg = { id: Date.now(), role: "user", content: question };
    setStore(prev => ({
      ...prev,
      [activeConversation]: [...(prev[activeConversation] || []), userMsg],
    }));
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();

      const botMsg = {
        id: Date.now() + 1,
        role: "assistant",
        content: sanitize(data.answer || "I couldn't generate a response."),
        citations: data.citations || [],
      };

      setStore(prev => ({
        ...prev,
        [activeConversation]: [...(prev[activeConversation] || []), botMsg],
      }));
    } catch (err) {
      setStore(prev => ({
        ...prev,
        [activeConversation]: [
          ...(prev[activeConversation] || []),
          { id: Date.now() + 2, role: "assistant", content: "Sorry, something went wrong." },
        ],
      }));
    } finally {
      setLoading(false);
    }
  };

  const newChat = () => {
    const id = Number(String(Date.now()).slice(-9)); // simple unique-ish id
    setConversations(prev => [{ id, title: "New chat", updated: "Just now" }, ...prev]);
    setStore(prev => ({ ...prev, [id]: [] }));
    setActiveConversation(id);
  };

  return (
    <div className="h-screen w-full bg-gray-50 text-slate-900">
      {/* Ensure grid can shrink/scroll */}
      <div className="grid grid-cols-[280px,1fr] h-full min-h-0">
        {/* Sidebar */}
        <aside className="bg-white border-r">
          <div className="flex items-center justify-between px-4 h-14 border-b">
            <span className="font-semibold">Conversations</span>
            <button className="inline-flex items-center gap-2 px-2 py-1 rounded-md border" onClick={newChat}>
              <FiPlus /> New
            </button>
          </div>
          <div className="p-3 border-b">
            <input
              type="search"
              placeholder="Search…"
              className="w-full px-3 py-2 rounded-md border outline-none"
            />
          </div>
          <nav className="overflow-y-auto h-[calc(100%-7rem)] p-2">
            {conversations.map((c) => (
              <button
                key={c.id}
                onClick={() => setActiveConversation(c.id)}
                className={[
                  "w-full text-left px-3 py-2 rounded-lg border",
                  activeConversation === c.id
                    ? "bg-blue-50 border-blue-200 text-blue-700"
                    : "bg-white border-transparent hover:bg-slate-50",
                ].join(" ")}
                title={c.title}
              >
                <div className="font-semibold truncate">{c.title}</div>
                <div className="text-xs text-slate-500">{c.updated}</div>
              </button>
            ))}
          </nav>
        </aside>

        {/* Main area */}
        <main className="grid grid-rows-[56px,1fr] min-h-0">
          {/* Top bar */}
          <div className="h-14 bg-white border-b flex items-center justify-between px-3">
            <div className="font-semibold">Message Chatbot</div>
            <div className="flex items-center gap-2">
              <button className="inline-flex items-center gap-2 px-2 py-1 rounded-md border" onClick={newChat}>
                <FiRefreshCw /> Reset
              </button>
            </div>
          </div>

          {/* Messages (only this scrolls). Add bottom padding to clear the fixed footer */}
          <div
            ref={scroller}
            className="overflow-y-auto min-h-0 p-3 sm:p-4 pb-[96px] bg-gradient-to-b from-blue-50/40 via-transparent to-transparent"
          >
            <div className="max-w-3xl mx-auto">
              {messages.length === 0 && (
                <div className="text-center text-slate-500 py-10">
                  Start a new conversation or ask a question.
                </div>
              )}
              {messages.map((m) => (
                <Message key={m.id} role={m.role} content={m.content} citations={m.citations} />
              ))}
              {loading && (
                <div className="text-slate-500 text-sm animate-pulse px-2">Thinking…</div>
              )}
            </div>
          </div>

          {/* Fixed footer input */}
          <form
            onSubmit={handleSend}
            className="fixed inset-x-0 bottom-0 bg-white border-t z-20 [padding-bottom:env(safe-area-inset-bottom)]"
          >
            <div className="max-w-3xl mx-auto p-3 sm:p-4">
              <div className="flex items-center gap-2">
                <div className="flex-1 flex items-center gap-2 border rounded-2xl px-2 bg-white">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Message Chatbot"
                    className="flex-1 px-2 py-3 rounded-2xl outline-none"
                    disabled={loading}
                  />
                  <button type="button" className="p-2 text-slate-500 hover:text-slate-700" title="Attach">
                    <FiPaperclip />
                  </button>
                  <button type="button" className="p-2 text-slate-500 hover:text-slate-700" title="Image">
                    <FiImage />
                  </button>
                  <button type="button" className="p-2 text-slate-500 hover:text-slate-700" title="Voice">
                    <FiMic />
                  </button>
                </div>
                <button
                  type="submit"
                  disabled={loading || !input.trim()}
                  className="inline-flex items-center gap-2 px-4 py-3 rounded-xl bg-blue-600 text-white disabled:opacity-50"
                >
                  <FiSend /> Send
                </button>
              </div>
              <div className="flex items-center text-xs text-slate-500 mt-2">
                Press <b className="mx-1">Enter</b> to send • <b className="mx-1">Shift+Enter</b> for newline
              </div>
            </div>
          </form>
        </main>
      </div>
    </div>
  );
}