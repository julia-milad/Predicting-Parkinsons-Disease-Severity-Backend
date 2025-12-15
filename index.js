const express = require("express");
const cors = require("cors");
const fetch = require("node-fetch");

const app = express();

const PORT = process.env.PORT || 5000;

const FLASK_URL = process.env.FLASK_URL;

if (!FLASK_URL) {
  console.error("Error: FLASK_URL env variable not set!");
  process.exit(1);
}

app.use(cors());
app.use(express.json());

app.post("/predict", async (req, res) => {
  try {
    const response = await fetch(`${FLASK_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });

    const data = await response.json();

    if (!response.ok) return res.status(response.status).json(data);

    res.json(data);
  } catch (err) {
    console.error("Gateway error:", err.message);
    res.status(500).json({ error: "Python service unavailable" });
  }
});

app.get("/", (req, res) => res.send("Node Gateway Running"));

app.listen(PORT, () =>
  console.log(`Node gateway running on port ${PORT}`)
);
