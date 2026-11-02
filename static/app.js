const chatWindow = document.getElementById("chatWindow");
const inputBox = document.getElementById("inputBox");
const sendBtn = document.getElementById("sendBtn");
const statusEl = document.getElementById("status");
const quickEl = document.getElementById("quickQuestions");
const modelSelect = document.getElementById("modelSelect");

const QUICK_QUESTIONS = [
  "怎么申请退货退款？",
  "退款多久到账？",
  "手机保修多长时间？",
  "支持哪些支付方式？",
  "发货后能改地址吗？",
];

let sending = false;
let lastExchange = null; // {question, answer, model}

function init() {
  QUICK_QUESTIONS.forEach((q) => {
    const btn = document.createElement("button");
    btn.textContent = q;
    btn.onclick = () => send(q);
    quickEl.appendChild(btn);
  });
  checkHealth();
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.llm_models && data.llm_models.length) {
      modelSelect.innerHTML = "";
      data.llm_models.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        if (m === data.llm_model) opt.selected = true;
        modelSelect.appendChild(opt);
      });
    }
    if (data.kb_chunks === 0) {
      statusEl.textContent = "知识库为空，请先执行 python cli.py ingest";
    } else {
      const features = [];
      if (data.hybrid) features.push("混合检索");
      if (data.rerank) features.push("Rerank");
      if (data.cache) features.push("缓存");
      statusEl.textContent = `在线 · 知识库 ${data.kb_chunks} 条 · ${features.join("+") || "向量检索"}`;
    }
  } catch {
    statusEl.textContent = "服务未启动，请运行 python server.py";
  }
}

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (text) bubble.textContent = text;
  div.appendChild(bubble);
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return { el: div, bubble };
}

function addTyping() {
  const { bubble } = addMessage("bot", "");
  bubble.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
  return bubble;
}

function attachFeedback(msgEl) {
  if (!lastExchange || !lastExchange.answer) return;
  const bar = document.createElement("div");
  bar.className = "fb-bar";
  const label = document.createElement("span");
  label.className = "fb-label";
  label.textContent = "这个回答有帮助吗？";
  const up = document.createElement("button");
  up.className = "fb-btn";
  up.textContent = "👍";
  const down = document.createElement("button");
  down.className = "fb-btn";
  down.textContent = "👎";
  bar.appendChild(label);
  bar.appendChild(up);
  bar.appendChild(down);

  const sendFeedback = async (rating) => {
    up.disabled = down.disabled = true;
    try {
      await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...lastExchange, rating }),
      });
    } catch {}
    label.textContent = rating === "up" ? "感谢你的认可！" : "感谢反馈，我们会继续改进";
  };
  up.onclick = () => sendFeedback("up");
  down.onclick = () => sendFeedback("down");

  msgEl.appendChild(bar);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function setSending(value) {
  sending = value;
  sendBtn.disabled = value;
  inputBox.disabled = value;
}

async function send(presetQuestion) {
  const question = (presetQuestion || inputBox.value).trim();
  if (!question || sending) return;

  inputBox.value = "";
  autoResize();
  addMessage("user", question);
  const { el: botMsg, bubble: botBubble } = addTyping();
  setSending(true);
  const usedModel = modelSelect.value || null;

  let answer = "";
  let firstToken = true;
  let hasToolCall = false;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, model: modelSelect.value || null }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const event = JSON.parse(line.slice(6));
        if (event.type === "token") {
          if (firstToken) {
            botBubble.textContent = "";
            firstToken = false;
          }
          answer += event.content;
          botBubble.textContent = answer;
          chatWindow.scrollTop = chatWindow.scrollHeight;
        } else if (event.type === "tool_call") {
          hasToolCall = true;
          const stepDiv = document.createElement("div");
          stepDiv.className = "tool-step";
          stepDiv.textContent = `🔧 ${event.content.name}：${event.content.summary}`;
          botBubble.appendChild(stepDiv);
          chatWindow.scrollTop = chatWindow.scrollHeight;
        } else if (event.type === "error") {
          botBubble.textContent = answer || event.content;
        }
      }
    }
  } catch {
    botBubble.textContent = answer || "网络异常，请稍后重试。";
  } finally {
    if (firstToken) {
      botBubble.textContent = "抱歉，服务暂时没有返回结果，请稍后重试。";
    }
    if (answer && !firstToken) {
      lastExchange = { question, answer, model: usedModel };
      attachFeedback(botBubble);
    }
    setSending(false);
    inputBox.focus();
  }
}

function autoResize() {
  inputBox.style.height = "auto";
  inputBox.style.height = Math.min(inputBox.scrollHeight, 120) + "px";
}

sendBtn.onclick = () => send();
modelSelect.addEventListener("change", () => {
  addMessage("bot", `已切换到模型：${modelSelect.value}`);
});
inputBox.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
inputBox.addEventListener("input", autoResize);

init();
