import glob
import numpy as np
import pandas as pd
import ta
import torch
import torch.nn as nn
import torch.optim as optim
import random

from collections import deque
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

from flask import Flask, request, jsonify
from flask_cors import CORS

# --- LangChain / Ollama imports (optional, for LLM explanations) ---
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)   # ← This must come right here, before any routes

# =========
# 1) TRADING ENVIRONMENT + AGENT + LSTM
# =========

class TradingEnv:
    def __init__(self, data, window_size=10, initial_balance=10000):
        self.data = data.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = float(initial_balance)
        self.current_step = None
        self.balance = None
        self.shares_held = None
        self.total_profit = None

    def reset(self):
        if len(self.data) <= self.window_size:
            raise ValueError(f"Need more data ({len(self.data)}) than window_size ({self.window_size})")
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0.0
        return self._get_state()

    def _get_state(self):
        window = self.data.iloc[self.current_step - self.window_size : self.current_step]
        state = []
        for i in range(self.window_size):
            state.extend([
                window['open'].iat[i],
                window['high'].iat[i],
                window['low'].iat[i],
                window['close'].iat[i],
                window['volume'].iat[i],
            ])
        state.extend([self.balance, self.shares_held])
        return np.array(state, dtype=np.float32)

    def step(self, action, amount=None):
        """
        action: 0=Hold, 1=Buy, 2=Sell
        amount: how much cash to spend if buying (None → spend all)
        Returns: (next_state, reward, done, info)
        """
        price = float(self.data['close'].iat[self.current_step])
        reward = 0.0

        if action == 1:  # BUY
            spend = self.balance if amount is None else min(self.balance, amount)
            shares = int(spend // price)
            if shares > 0:
                self.shares_held += shares
                self.balance -= shares * price

        elif action == 2 and self.shares_held > 0:  # SELL
            self.balance += self.shares_held * price
            self.total_profit += self.shares_held * price
            reward = self.total_profit
            self.shares_held = 0

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        return self._get_state(), reward, done, {}

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        S, A, R, NS, D = zip(*batch)
        return np.vstack(S), A, R, np.vstack(NS), D

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500):
        self.mem = ReplayBuffer()
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.e_end = epsilon_end
        self.e_decay = epsilon_decay
        self.steps = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = QNetwork(state_dim, action_dim).to(self.device)
        self.target = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.target.load_state_dict(self.policy.state_dict())

    def select_action(self, state):
        eps_thresh = self.e_end + (self.epsilon - self.e_end) * np.exp(-self.steps / self.e_decay)
        self.steps += 1
        if random.random() > eps_thresh:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                return int(self.policy(s).max(1)[1])
        return random.randrange(self.policy.fc3.out_features)

    def learn(self, batch_size=64):
        if len(self.mem) < batch_size:
            return
        S, A, R, NS, D = self.mem.sample(batch_size)

        S = torch.tensor(S, dtype=torch.float32).to(self.device)
        A = torch.tensor(A, dtype=torch.long).unsqueeze(1).to(self.device)
        R = torch.tensor(R, dtype=torch.float32).unsqueeze(1).to(self.device)
        NS = torch.tensor(NS, dtype=torch.float32).to(self.device)
        D = torch.tensor(D, dtype=torch.float32).unsqueeze(1).to(self.device)

        curr_q = self.policy(S).gather(1, A)
        next_q = self.target(NS).max(1)[0].detach().unsqueeze(1)
        expected_q = R + self.gamma * next_q * (1.0 - D)

        loss = nn.MSELoss()(curr_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_lstm(data, seq_len=10, epochs=20, batch_size=32, lr=1e-3):
    vals = data['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(vals)

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len : i])
        y.append(scaled[i])
    X, y = np.stack(X), np.stack(y)

    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # If you want debug output, uncomment:
        # print(f"Epoch {ep}/{epochs} — LSTM Loss: {total_loss/len(loader):.5f}")
    return model, scaler

def get_hint(df, step):
    rsi = df['rsi'].iat[step]
    sma = df['sma20'].iat[step]
    price = df['close'].iat[step]
    if rsi > 70:
        rsi_msg = f"RSI {rsi:.1f} → overbought"
    elif rsi < 30:
        rsi_msg = f"RSI {rsi:.1f} → oversold"
    else:
        rsi_msg = f"RSI {rsi:.1f} → neutral"
    if price > sma:
        trend_msg = f"Price ${price:.2f} > SMA (${sma:.2f}) → uptrend"
    else:
        trend_msg = f"Price ${price:.2f} < SMA (${sma:.2f}) → downtrend"
    return f"{rsi_msg}. {trend_msg}."

# =========
# 2) LANGCHAIN / LLM EXPLANATION (optional)
# =========
callbacks = [StreamingStdOutCallbackHandler()]
llm = OllamaLLM(model="llama2", streaming=False, callbacks=callbacks)
_template = (
    "You are an expert finance tutor writing for absolute beginners.\n"
    "- Today's close: {close}\n"
    "- Next-day forecast: {pred}\n"
    "- User chose to {ua}\n"
    "- Agent recommended {ba}\n"
    "Write 3–4 concise paragraphs, each clarifying one concept. End with a one-sentence takeaway."
)
prompt = PromptTemplate(input_variables=["close", "pred", "ua", "ba"], template=_template)
chain = LLMChain(llm=llm, prompt=prompt)

def explain_with_llm(close, pred, ua, ba):
    return chain.run({"close": close, "pred": pred, "ua": ua, "ba": ba})

# =========
# 3) FLASK APP SETUP
# =========

app = Flask(__name__)
CORS(app)  # allow all origins (for development)

trained_models = {}  # Will hold: { ticker_str: { "df":df, "scaler":scaler, "lstm":lstm_model, "agent":agent } }

sessions = {}  # Will hold: { session_id_str: { "env":TradingEnv, "df":df, "scaler":scaler, "lstm":lstm_model, "agent":agent } }

def train_all_tickers():
    """
    Discover all *.csv in this folder, train an LSTM + DQNAgent for each,
    and store results in `trained_models[ticker]`.
    """
    global trained_models
    files = glob.glob("*.csv")
    raw_tickers = [f.replace(".csv", "") for f in files]
    print("Tickers found:", raw_tickers)

    for raw in raw_tickers:
        ticker = raw.upper()   # ← convert to uppercase here
        print(f"=== TRAINING: {raw} (storing as {ticker}) ===")
        df = pd.read_csv(f"{raw}.csv", parse_dates=["Date"], index_col="Date")

        # Normalize column names & types:
        df.columns = df.columns.str.lower().str.replace("/", "_").str.replace(" ", "_")
        for col in ["close_last", "open", "high", "low"]:
            if col in df.columns:
                df[col] = df[col].replace(r"[\$,]", "", regex=True).astype(float)
        if "close_last" in df.columns:
            df.rename(columns={"close_last": "close"}, inplace=True)
        df["volume"] = df["volume"].replace(r"[,\s]", "", regex=True).fillna(0).astype(int)
        df.sort_index(inplace=True)

        # Compute indicators
        df["sma20"] = ta.trend.sma_indicator(df["close"], 20)
        df["rsi"] = ta.momentum.rsi(df["close"], 14)
        df["macd"] = ta.trend.macd(df["close"])
        df.bfill(inplace=True)

        # Train LSTM
        lstm_model, scaler = train_lstm(df, seq_len=10, epochs=10)

        # Train DQN
        env = TradingEnv(df, window_size=10, initial_balance=10_000.0)
        agent = DQNAgent(state_dim=len(env.reset()), action_dim=3)

        for ep in range(1, 51):
            state = env.reset()
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.mem.push(state, action, reward, next_state, done)
                agent.learn()
                state = next_state
            agent.target.load_state_dict(agent.policy.state_dict())

        print(f"{ticker} training complete. Profit: {env.total_profit:.2f}")

        trained_models[ticker] = {
            "df": df,
            "scaler": scaler,
            "lstm": lstm_model,
            "agent": agent
        }

    print("\n*** All tickers trained! ***")


# ────────────────────────────────────────────────────────────────────────────
# 3.1) GET /tickers → returns ["AAPL","TSLA",…]
# ────────────────────────────────────────────────────────────────────────────
@app.route("/tickers", methods=["GET"])
def get_tickers():
    return jsonify({"tickers": list(trained_models.keys())})


# ────────────────────────────────────────────────────────────────────────────
# 3.2) POST /start_session { "ticker": "AAPL" } → returns { "session_id": "...", "initial_state": [ … ] }
# ────────────────────────────────────────────────────────────────────────────
@app.route("/start_session", methods=["POST"])
def start_session():
    data = request.get_json()
    ticker = data.get("ticker", "").upper()

    if ticker not in trained_models:
        return jsonify({"error": f"Ticker '{ticker}' not found."}), 400

    entry = trained_models[ticker]
    df = entry["df"]
    scaler = entry["scaler"]
    lstm_model = entry["lstm"]
    agent = entry["agent"]

    # Create a fresh environment for this session
    env = TradingEnv(df, window_size=10, initial_balance=10_000.0)
    initial_state = env.reset()

    # Generate a unique session ID
    import uuid
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "env": env,
        "df": df,
        "scaler": scaler,
        "lstm": lstm_model,
        "agent": agent
    }

    return jsonify({
        "session_id": session_id,
        "initial_state": initial_state.tolist()
    })


# ────────────────────────────────────────────────────────────────────────────
# 3.3) GET /state_and_hint?session_id=… → returns { step, balance, shares_held, hint, recent_bars:[…] }
# ────────────────────────────────────────────────────────────────────────────
@app.route("/state_and_hint", methods=["GET"])
def state_and_hint():
    session_id = request.args.get("session_id", "")
    if session_id not in sessions:
        return jsonify({"error": "Invalid session_id."}), 400

    sess = sessions[session_id]
    env = sess["env"]
    df = sess["df"]

    step = env.current_step
    hint_text = get_hint(df, step)

    window_size = env.window_size
    idx_start = step - window_size

    bars = []
    for i in range(idx_start, step):
        bars.append({
            "date": str(df.index[i]),
            "open": float(df["open"].iat[i]),
            "high": float(df["high"].iat[i]),
            "low": float(df["low"].iat[i]),
            "close": float(df["close"].iat[i]),
            "volume": int(df["volume"].iat[i])
        })

    return jsonify({
        "step": step,
        "balance": env.balance,
        "shares_held": env.shares_held,
        "hint": hint_text,
        "recent_bars": bars
    })


# ────────────────────────────────────────────────────────────────────────────
# 3.4) POST /take_action { "session_id": "...", "action":0|1|2, "amount":123.45? }
#         → returns { next_state:[…], reward:…, done:bool, correct:bool, agent_recommendation:0|1|2, predicted_price:…, llm_explanation:"…" }
# ────────────────────────────────────────────────────────────────────────────
@app.route("/take_action", methods=["POST"])
def take_action():
    data = request.get_json()
    session_id = data.get("session_id", "")
    action = data.get("action", None)
    amount = data.get("amount", None)

    if session_id not in sessions:
        return jsonify({"error": "Invalid session_id."}), 400
    if action not in [0, 1, 2]:
        return jsonify({"error": "Action must be 0 (Hold), 1 (Buy), or 2 (Sell)."}), 400

    sess = sessions[session_id]
    env = sess["env"]
    df = sess["df"]
    scaler = sess["scaler"]
    lstm_model = sess["lstm"]
    agent = sess["agent"]

    # 1) Compute agent's recommendation on current state
    current_state = env._get_state()
    recommended = agent.select_action(current_state)
    is_correct = (action == recommended)

    # 2) LSTM prediction of next day's price
    seq = scaler.transform(df["close"].values.reshape(-1, 1))
    seq_tensor = torch.tensor(
        seq[env.current_step - 10 : env.current_step].reshape(1, 10, 1),
        dtype=torch.float32
    )
    with torch.no_grad():
        pred_scaled = lstm_model(seq_tensor).item()
    predicted_price = scaler.inverse_transform([[pred_scaled]])[0][0]

    # 3) Generate LLM explanation (optional; can be slow)
    close_str = f"${df['close'].iat[env.current_step]:.2f}"
    pred_str = f"${predicted_price:.2f}"
    ua_str = ["hold", "buy", "sell"][action]
    ba_str = ["hold", "buy", "sell"][recommended]
    llm_explanation = explain_with_llm(close_str, pred_str, ua_str, ba_str)

    # 4) Step the environment with the user's action
    next_state, reward, done, _ = env.step(action, amount=amount)

    return jsonify({
        "next_state": next_state.tolist(),
        "reward": reward,
        "done": done,
        "correct": is_correct,
        "agent_recommendation": recommended,
        "predicted_price": predicted_price,
        "llm_explanation": llm_explanation
    })


# ────────────────────────────────────────────────────────────────────────────
# 4) MAIN: Train all tickers, then run Flask
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Train (or load) all tickers into memory
    train_all_tickers()

    # 2) Start Flask server
    app.run(host="0.0.0.0", port=5001, debug=True)
