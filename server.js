// // server.js

// const express = require("express");
// const axios = require("axios");
// const path = require("path");

// const app = express();
// const PORT = 3000;

// // So we can parse JSON if needed:
// app.use(express.json());
// app.use(express.urlencoded({ extended: true }));

// // Set EJS as templating engine:
// app.set("view engine", "ejs");
// app.set("views", path.join(__dirname, "views"));

// // =========
// // 1) Show home page with ticker dropdown
// // =========
// app.get("/", async (req, res) => {
//   try {
//     // Fetch list of tickers from Flask API:
//     const resp = await axios.get("http://localhost:5001/tickers");
//     const tickers = resp.data.tickers; // e.g. ["AAPL","TSLA",...]

//     // Render index.ejs, passing tickers:
//     res.render("index", { tickers });
//   } catch (err) {
//     console.error("Error fetching tickers:", err.message);
//     res.send("Unable to fetch tickers. Is Flask running on port 5001?");
//   }
// });

// // =========
// // 2) Handle “start practice” submission
// // =========
// app.post("/start", async (req, res) => {
//   const chosenTicker = req.body.ticker; // from the form
//   try {
//     const resp = await axios.post("http://localhost:5001/start_session", {
//       ticker: chosenTicker,
//     });
//     const session_id = resp.data.session_id;
//     // We’ll redirect to /practice?session_id=...
//     res.redirect(`/practice?session_id=${session_id}`);
//   } catch (err) {
//     console.error("Error starting session:", err.message, err.response?.data);
//     res.send("Failed to start session: " + JSON.stringify(err.response?.data));
//   }
// });

// // =========
// // 3) Render practice page (front end will call /state_and_hint, /take_action via AJAX)
// // =========
// app.get("/practice", (req, res) => {
//   const sessionId = req.query.session_id;
//   if (!sessionId) {
//     return res.send("No session_id provided. Go back to home and select a ticker.");
//   }
//   // Render the EJS template, passing the session ID so front end JS can use it.
//   res.render("practice", { sessionId });
// });

// // =========
// // 4) Static files (Plotly.js, optional CSS, etc.)
// // =========
// app.use("/plotly", express.static(path.join(__dirname, "node_modules", "plotly.js", "dist")));

// // =========
// // Start Express:
// app.listen(PORT, () => {
//   console.log(`Front‐end listening at http://localhost:${PORT}`);
// });
// server.js

const express = require("express");
const axios = require("axios");
const path = require("path");
const bcrypt = require("bcrypt");
const session = require("express-session");
const sqlite3 = require("sqlite3").verbose();

// ─── Open (or create) our SQLite auth database ─────────────────────────────
const db = new sqlite3.Database("./auth.db", (err) => {
  if (err) return console.error("Failed to open auth.db:", err);
  console.log("Connected to auth.db");
});
// Create the `users` table if it doesn't exist:
db.run(
  `CREATE TABLE IF NOT EXISTS users (
     id INTEGER PRIMARY KEY AUTOINCREMENT,
     username TEXT UNIQUE,
     password_hash TEXT
   )`,
  (err) => {
    if (err) console.error("Error creating users table:", err);
  }
);

// ⟵ NEW CODE: Create trade_history table
db.run(
  `CREATE TABLE IF NOT EXISTS trade_history (
     id INTEGER PRIMARY KEY AUTOINCREMENT,
     user_id INTEGER NOT NULL,
     ticker TEXT NOT NULL,
     session_uuid TEXT NOT NULL,
     action INTEGER NOT NULL,
     correct INTEGER NOT NULL,
     reward REAL NOT NULL,
     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
     FOREIGN KEY(user_id) REFERENCES users(id)
   )`,
  (err) => {
    if (err) console.error("Error creating trade_history table:", err);
  }
);


const app = express();
const PORT = 3000;

// ─── Middleware setup ────────────────────────────────────────────────────────
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Session configuration (must come before any route handlers that use req.session)
app.use(
  session({
    secret: "replace-with-a-long-random-string",
    resave: false,
    saveUninitialized: false,
    cookie: { maxAge: 1000 * 60 * 60 * 24 }, // 1 day
  })
);

// Make `currentUser` available in every EJS view
app.use((req, res, next) => {
  res.locals.currentUser = req.session.user || null;
  next();
});

// Set up EJS
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));

// Serve Plotly (or any other static files) from node_modules
app.use(
  "/plotly",
  express.static(path.join(__dirname, "node_modules", "plotly.js", "dist"))
);

// ─── “Require login” middleware ───────────────────────────────────────────────
function requireLogin(req, res, next) {
  if (!req.session.user) {
    return res.redirect("/login");
  }
  next();
}

// ─── Authentication Routes ────────────────────────────────────────────────────

// Show registration form
app.get("/register", (req, res) => {
  res.render("register", { error: null });
});

// Handle registration
app.post("/register", async (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) {
    return res.render("register", { error: "Both fields are required." });
  }

  try {
    // Hash the password
    const password_hash = await bcrypt.hash(password, 10);
    // Insert into users table
    db.run(
      `INSERT INTO users (username, password_hash) VALUES (?, ?)`,
      [username, password_hash],
      function (err) {
        if (err) {
          if (err.code === "SQLITE_CONSTRAINT") {
            // Unique constraint failed → username already exists
            return res.render("register", {
              error: "Username already taken.",
            });
          }
          console.error(err);
          return res.render("register", { error: "Database error." });
        }
        // Auto-log in after successful registration
        req.session.user = { id: this.lastID, username };
        return res.redirect("/");
      }
    );
  } catch (bcryptError) {
    console.error(bcryptError);
    return res.render("register", { error: "Server error." });
  }
});

// Show login form
app.get("/login", (req, res) => {
  res.render("login", { error: null });
});

// Handle login
app.post("/login", (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) {
    return res.render("login", { error: "Both fields are required." });
  }

  db.get(
    `SELECT id, password_hash FROM users WHERE username = ?`,
    [username],
    async (err, row) => {
      if (err) {
        console.error(err);
        return res.render("login", { error: "Database error." });
      }
      if (!row) {
        return res.render("login", { error: "Invalid username or password." });
      }
      // Compare submitted password against stored hash
      const match = await bcrypt.compare(password, row.password_hash);
      if (!match) {
        return res.render("login", { error: "Invalid username or password." });
      }
      // Credentials are valid → set session and redirect to home
      req.session.user = { id: row.id, username };
      return res.redirect("/");
    }
  );
});

// Handle logout
app.get("/logout", (req, res) => {
  req.session.destroy((err) => {
    if (err) console.error("Session destroy error:", err);
    res.redirect("/login");
  });
});
// ─── Change Password Routes ─────────────────────────────────────────────────

// Show “Change Password” form (only for logged‐in users)
app.get("/change-password", requireLogin, (req, res) => {
  res.render("change_password", { error: null, success: null });
});

// Handle form submission
app.post("/change-password", requireLogin, async (req, res) => {
  const userId = req.session.user.id;
  const { currentPassword, newPassword, confirmPassword } = req.body;

  // 1) Check that all fields are provided
  if (!currentPassword || !newPassword || !confirmPassword) {
    return res.render("change_password", {
      error: "All fields are required.",
      success: null,
    });
  }

  // 2) Confirm new passwords match
  if (newPassword !== confirmPassword) {
    return res.render("change_password", {
      error: "New password and confirmation do not match.",
      success: null,
    });
  }

  // 3) Fetch the user’s existing password hash
  db.get(
    `SELECT password_hash FROM users WHERE id = ?`,
    [userId],
    async (err, row) => {
      if (err) {
        console.error("DB error:", err);
        return res.render("change_password", {
          error: "Server error. Please try again later.",
          success: null,
        });
      }
      if (!row) {
        // Shouldn’t happen if session is valid, but just in case
        return res.render("change_password", {
          error: "User not found.",
          success: null,
        });
      }

      // 4) Verify current password
      const match = await bcrypt.compare(currentPassword, row.password_hash);
      if (!match) {
        return res.render("change_password", {
          error: "Current password is incorrect.",
          success: null,
        });
      }

      // 5) Hash the new password and update in DB
      try {
        const newHash = await bcrypt.hash(newPassword, 10);
        db.run(
          `UPDATE users SET password_hash = ? WHERE id = ?`,
          [newHash, userId],
          function (updateErr) {
            if (updateErr) {
              console.error("Failed to update password:", updateErr);
              return res.render("change_password", {
                error: "Could not update password. Please try again.",
                success: null,
              });
            }
            // 6) Render the same form with a success message
            return res.render("change_password", {
              error: null,
              success: "Password successfully updated.",
            });
          }
        );
      } catch (hashErr) {
        console.error("Hashing error:", hashErr);
        return res.render("change_password", {
          error: "Server error. Please try again later.",
          success: null,
        });
      }
    }
  );
});

// ─── Protected Application Routes ─────────────────────────────────────────────

// Home page (must be logged in)
app.get("/", requireLogin, async (req, res) => {
  try {
    const resp = await axios.get("http://localhost:5001/tickers");
    const tickers = resp.data.tickers;
    res.render("index", { tickers });
  } catch (err) {
    console.error("Error fetching tickers:", err.message);
    res.send("Unable to fetch tickers. Is Flask running on port 5001?");
  }
});

// Start practice (must be logged in)
app.post("/start", requireLogin, async (req, res) => {
  const chosenTicker = req.body.ticker;
  if (!chosenTicker) {
    return res.send("No ticker selected.");
  }
  try {
    const resp = await axios.post("http://localhost:5001/start_session", {
      ticker: chosenTicker,
    });
    const session_id = resp.data.session_id;
    // Make sure we include ticker here, too:
    res.redirect(`/practice?session_id=${session_id}&ticker=${chosenTicker}`);
  } catch (err) {
    console.error(err);
    res.send("Failed to start session.");
  }
});


// Practice page (must be logged in)
// In server.js:
app.get("/practice", requireLogin, (req, res) => {
  const sessionId = req.query.session_id;
  const ticker    = req.query.ticker;    // ← you must grab this
  if (!sessionId || !ticker) {
    return res.send(
      "No session_id or ticker provided. Go back to home and select a ticker."
    );
  }
  // Pass both sessionId and ticker into the template:
  res.render("practice", { sessionId, ticker });
});


// ⟵ NEW ROUTE: record_trade
// This will be called from practice.ejs (front end) after each take_action
app.post("/record_trade", requireLogin, (req, res) => {
  const userId = req.session.user.id;
  const { session_uuid, ticker, action, correct, reward } = req.body;
  if (
    typeof session_uuid !== "string" ||
    typeof ticker !== "string" ||
    ![0, 1, 2].includes(action) ||
    ![0, 1].includes(correct) ||
    typeof reward !== "number"
  ) {
    return res.status(400).json({ error: "Invalid payload." });
  }

  const stmt = db.prepare(
    `INSERT INTO trade_history
       (user_id, ticker, session_uuid, action, correct, reward)
     VALUES (?, ?, ?, ?, ?, ?)`
  );
  stmt.run(
    userId,
    ticker,
    session_uuid,
    action,
    correct,
    reward,
    function (err) {
      if (err) {
        console.error("DB insert error:", err);
        return res.status(500).json({ error: "DB error." });
      }
      res.json({ success: true });
    }
  );
});

// ⟵ NEW ROUTE: profile
// Renders a dashboard for the logged-in user, pulling from trade_history
app.get("/profile", requireLogin, (req, res) => {
  const userId = req.session.user.id;

  // 1. Fetch all rows for this user
  db.all(
    `SELECT id, ticker, session_uuid, action, correct, reward, timestamp
       FROM trade_history
       WHERE user_id = ?
       ORDER BY timestamp DESC`,
    [userId],
    (err, rows) => {
      if (err) {
        console.error("DB fetch error:", err);
        return res.send("Error loading profile.");
      }

      // 2. Compute summary metrics
      const totalTrades = rows.length;
      const totalCorrect = rows.filter((r) => r.correct === 1).length;
      const accuracy = totalTrades > 0 ? (totalCorrect / totalTrades) * 100 : 0;
      const totalProfit = rows.reduce((sum, r) => sum + r.reward, 0);

      // 3. Choose simple “tips to improve” based on accuracy
      let tips = [];
      if (accuracy < 50 && totalTrades > 0) {
        tips.push(
          "Your overall accuracy is below 50%. Try to pay closer attention to the RSI hints (e.g., avoid buying when RSI > 70)."
        );
      }
      if (accuracy >= 50 && accuracy < 75) {
        tips.push(
          "Your accuracy is between 50–75%. Keep practicing, and consider reviewing how price vs. SMA trends correlate."
        );
      }
      if (accuracy >= 75) {
        tips.push("Great job! Your accuracy is above 75%. Keep honing your entry/exit timing.");
      }
      if (totalTrades === 0) {
        tips.push("You haven't traded yet. Start a practice session to record your first trades.");
      }

      // 4. Render profile.ejs
      res.render("profile", {
        username: req.session.user.username,
        history: rows,
        summary: {
          totalTrades,
          accuracy: accuracy.toFixed(1),
          totalProfit: totalProfit.toFixed(2),
        },
        tips,
      });
    }
  );
});
// ─── Start the server ─────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`Front‐end listening at http://localhost:${PORT}`);
});
