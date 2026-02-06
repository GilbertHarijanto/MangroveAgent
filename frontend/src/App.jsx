import { useEffect, useRef, useState } from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer
} from "recharts";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = "http://localhost:8000";

function SeriesChart({ series, chartType }) {
  const data = series.points.map((point) => ({
    t: point.t,
    v: point.v
  }));

  const tooltipStyle = {
    backgroundColor: "#111827",
    border: "1px solid #2d3748",
    borderRadius: 10,
    color: "#e2e8f0"
  };

  const tooltipLabelStyle = {
    color: "#94a3b8",
    fontWeight: 500
  };

  const tooltipFormatter = (value) => [
    value,
    series.name.toUpperCase()
  ];

  return (
    <div className="chart">
      <h4>
        {series.name} {series.unit ? `(${series.unit})` : ""}
      </h4>
      <ResponsiveContainer width="100%" height={240}>
        {chartType === "bar" ? (
          <BarChart data={data}>
            <XAxis dataKey="t" stroke="#94a3b8" tick={{ fill: "#94a3b8" }} />
            <YAxis stroke="#94a3b8" tick={{ fill: "#94a3b8" }} />
            <Tooltip
              contentStyle={tooltipStyle}
              labelStyle={tooltipLabelStyle}
              formatter={tooltipFormatter}
              cursor={{ fill: "rgba(0,0,0,0)" }}
            />
            <Legend wrapperStyle={{ color: "#94a3b8" }} />
            <Bar dataKey="v" name={series.name} fill="#38bdf8" />
          </BarChart>
        ) : (
          <LineChart data={data}>
            <XAxis dataKey="t" stroke="#94a3b8" tick={{ fill: "#94a3b8" }} />
            <YAxis stroke="#94a3b8" tick={{ fill: "#94a3b8" }} />
            <Tooltip
              contentStyle={tooltipStyle}
              labelStyle={tooltipLabelStyle}
              formatter={tooltipFormatter}
            />
            <Legend wrapperStyle={{ color: "#94a3b8" }} />
            <Line
              type="monotone"
              dataKey="v"
              name={series.name}
              stroke="#38bdf8"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        )}
      </ResponsiveContainer>
    </div>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState("");
  const [sessions, setSessions] = useState([]);
  const [renamingSessionId, setRenamingSessionId] = useState(null);
  const [renameValue, setRenameValue] = useState("");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const chatRef = useRef(null);
  const bottomRef = useRef(null);
  const [menuSessionId, setMenuSessionId] = useState(null);

  const canSend = input.trim() && !isLoading;
  const hasFirstChunkRef = useRef(false);

  async function handleSend(event) {
    event.preventDefault();
    if (!canSend) return;

    const userMessage = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setIsLoading(true);
    hasFirstChunkRef.current = false;

    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", graphData: null }
    ]);

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId || undefined,
          message: userMessage
        })
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || "Request failed");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const block of lines) {
          if (!block.trim()) continue;
          let eventType = "";
          let dataStr = "";
          for (const line of block.split("\n")) {
            if (line.startsWith("event: ")) eventType = line.slice(7).trim();
            if (line.startsWith("data: ")) dataStr = line.slice(6).trim();
          }
          let data = null;
          try {
            data = dataStr ? JSON.parse(dataStr) : null;
          } catch (_) {
            data = dataStr;
          }

          if (eventType === "session" && data?.session_id) {
            setSessionId(data.session_id);
            if (!sessionId) fetchSessions();
          }
          if (eventType === "chunk" && data?.text) {
            if (!hasFirstChunkRef.current) {
              hasFirstChunkRef.current = true;
              setIsLoading(false);
            }
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant")
                next[next.length - 1] = {
                  ...last,
                  content: (last.content || "") + data.text
                };
              return next;
            });
          }
          if (eventType === "graph" && data?.graph_data) {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant")
                next[next.length - 1] = { ...last, graphData: data.graph_data };
              return next;
            });
          }
          if (eventType === "error" && data?.message) {
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant")
                next[next.length - 1] = { ...last, content: `Error: ${data.message}` };
              return next;
            });
            setIsLoading(false);
            return;
          }
          if (eventType === "done") {
            setIsLoading(false);
          }
        }
      }
      setIsLoading(false);
    } catch (error) {
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === "assistant")
          next[next.length - 1] = { ...last, content: `Error: ${error.message}` };
        else next.push({ role: "assistant", content: `Error: ${error.message}` });
        return next;
      });
      setIsLoading(false);
    }
  }

  async function handleDeleteHistory() {
    setMessages([]);
    if (!sessionId) {
      return;
    }
    try {
      await fetch(`${API_BASE}/sessions/${sessionId}`, {
        method: "DELETE"
      });
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${error.message}` }
      ]);
    } finally {
      setSessionId("");
      await fetchSessions();
    }
  }

  async function handleDeleteChat(targetSessionId, event) {
    event.stopPropagation();
    try {
      await fetch(`${API_BASE}/sessions/${targetSessionId}`, {
        method: "DELETE"
      });
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${error.message}` }
      ]);
    } finally {
      if (sessionId === targetSessionId) {
        setSessionId("");
        setMessages([]);
      }
      setMenuSessionId(null);
      await fetchSessions();
    }
  }

  function handleStartRename(targetSessionId, event) {
    event.stopPropagation();
    const current = sessions.find(
      (session) => session.session_id === targetSessionId
    );
    setRenamingSessionId(targetSessionId);
    setRenameValue(current?.title || "Untitled chat");
    setMenuSessionId(null);
  }

  async function handleSubmitRename(targetSessionId) {
    const nextTitle = renameValue.trim();
    if (!nextTitle) {
      setRenamingSessionId(null);
      return;
    }
    try {
      await fetch(`${API_BASE}/sessions/${targetSessionId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: nextTitle })
      });
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${error.message}` }
      ]);
    } finally {
      setRenamingSessionId(null);
      await fetchSessions();
    }
  }

  async function fetchSessions() {
    try {
      const response = await fetch(`${API_BASE}/sessions`);
      const payload = await response.json();
      setSessions(payload.sessions || []);
    } catch (error) {
      setSessions([]);
    }
  }

  async function loadSession(targetSessionId) {
    try {
      const response = await fetch(`${API_BASE}/sessions/${targetSessionId}`);
      const payload = await response.json();
      setSessionId(targetSessionId);
      setMenuSessionId(null);
      const history = (payload.messages || []).map((message) => ({
        role: message.role,
        content: message.content,
        graphData: message.graph_data || null
      }));
      setMessages(history);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${error.message}` }
      ]);
    }
  }

  function handleNewChat() {
    setSessionId("");
    setMessages([]);
    setMenuSessionId(null);
  }

  useEffect(() => {
    fetchSessions();
  }, []);

  useEffect(() => {
    if (!autoScroll) {
      return;
    }
    const timer = setTimeout(() => {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 120);
    return () => clearTimeout(timer);
  }, [messages, isLoading, autoScroll]);

  function handleChatScroll() {
    const container = chatRef.current;
    if (!container) return;
    const threshold = 120;
    const distanceFromBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight;
    setAutoScroll(distanceFromBottom <= threshold);
  }

  return (
    <div className={`page ${isSidebarOpen ? "sidebar-open" : ""}`}>
      <aside className={`sidebar ${isSidebarOpen ? "open" : ""}`}>
        {isSidebarOpen ? (
          <button
            className="sidebar-toggle"
            type="button"
            onClick={() => setIsSidebarOpen(false)}
            aria-label="Close settings"
          >
            <img src="/images/close.png" alt="" />
          </button>
        ) : null}
        <div className="sidebar-header" />
        <button
          type="button"
          className="secondary-button new-chat"
          onClick={handleNewChat}
        >
          <img
            src="/images/new_chat.png"
            alt=""
            className="menu-icon new-chat"
          />
          New chat
        </button>
        <div className="sidebar-section">Your chats</div>
        <div className="sidebar-list">
          {sessions.map((session) => (
            <button
              key={session.session_id}
              type="button"
              className={`chat-item ${
                sessionId === session.session_id ? "active" : ""
              }`}
              data-open={menuSessionId === session.session_id}
              onClick={() => loadSession(session.session_id)}
            >
              {renamingSessionId === session.session_id ? (
                <input
                  className="chat-rename-input"
                  value={renameValue}
                  onChange={(event) => setRenameValue(event.target.value)}
                  onClick={(event) => event.stopPropagation()}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      handleSubmitRename(session.session_id);
                    }
                    if (event.key === "Escape") {
                      setRenamingSessionId(null);
                    }
                  }}
                  onBlur={() => handleSubmitRename(session.session_id)}
                  autoFocus
                />
              ) : (
                <span className="chat-title">
                  {session.title || "Untitled chat"}
                </span>
              )}
              <button
                type="button"
                className="chat-actions"
                aria-label="Chat options"
                onClick={(event) => {
                  event.stopPropagation();
                  setMenuSessionId((prev) =>
                    prev === session.session_id ? null : session.session_id
                  );
                }}
              >
                •••
              </button>
              {menuSessionId === session.session_id ? (
                <div className="chat-menu" onClick={(event) => event.stopPropagation()}>
                  <button
                    type="button"
                    className="chat-menu-item rename"
                    onClick={(event) =>
                      handleStartRename(session.session_id, event)
                    }
                  >
                    <img src="/images/rename.png" alt="" className="menu-icon large" />
                    Rename
                  </button>
                  <button
                    type="button"
                    className="chat-menu-item delete"
                    onClick={(event) =>
                      handleDeleteChat(session.session_id, event)
                    }
                  >
                    <img src="/images/delete.png" alt="" className="menu-icon small" />
                    Delete
                  </button>
                </div>
              ) : null}
            </button>
          ))}
        </div>
      </aside>

      {!isSidebarOpen ? (
        <button
          className="sidebar-toggle global-toggle"
          type="button"
          onClick={() => setIsSidebarOpen(true)}
          aria-label="Open settings"
        >
          <img src="/images/open.png" alt="" />
        </button>
      ) : null}
      <div className="app">
        <header className="hero">
        <img
          className="hero-image"
          src="/images/img.png"
          alt="MangroveAgent"
        />
        <p>Real-time insights on mangrove health 🌱</p>
      </header>

        <section className="chat" ref={chatRef} onScroll={handleChatScroll}>
        {messages.map((message, idx) => (
          <div key={idx} className={`message ${message.role}`}>
              {message.role === "assistant" ? (
              <div className="message-content">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.content}
                </ReactMarkdown>
              </div>
            ) : (
                <div className="user-bubble">{message.content}</div>
            )}
            {message.graphData && message.graphData.series
              ? message.graphData.series
                  .filter(
                    (series) =>
                      series.name === "ndvi" && series.points.length > 0
                  )
                  .map((series) => (
                    <SeriesChart
                      key={series.name}
                      series={series}
                      chartType={message.graphData.chart_type}
                    />
                  ))
              : null}
          </div>
        ))}
        {isLoading && <div className="message assistant">Thinking...</div>}
          <div ref={bottomRef} />
      </section>

        <form className="composer" onSubmit={handleSend}>
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="How can I help?"
        />
        <button type="submit" disabled={!canSend} className="send-button">
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <path
              d="M12 6l5 5M12 6l-5 5M12 6v12"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
        </form>
      </div>
    </div>
  );
}
