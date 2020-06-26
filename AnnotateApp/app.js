const express = require("express");
const cors = require("cors");
const path = require("path");
const app = express();
const http = require("http").createServer(app);
const io = require("socket.io")(http);
const { Client } = require("pg");

function randomChoice(list) {
  var idx = Math.floor(Math.random() * list.length);
  return list[idx];
}

app.use(express.static(path.join(__dirname, "client/build")));
app.use(express.json());

app.get("/db", function(req, res, next) {
  const client = new Client({
    connectionString: process.env.DATABASE_URL,
    ssl: true
  });
  client.connect().catch(e => res.status(500));
  client
    .query("SELECT COUNT(*) FROM vectorImages;")
    .then(dbRes => {
      const n = dbRes.rows[0].count;
      const id = Math.floor(Math.random() * n);
      return client.query("SELECT * FROM vectorImages WHERE id=$1;", [id]);
    })
    .then(dbRes => res.send(dbRes.rows[0]))
    .catch(e => res.status(500))
    .then(() => client.end());
});

app.post("/save", function(req, res, next) {
  const client = new Client({
    connectionString: process.env.DATABASE_URL,
    ssl: true
  });
  client.connect().catch(e => res.status(500));
  const { graph, id } = req.body;
  let graphStr = JSON.stringify(graph);
  client
    .query("INSERT INTO vectorgraphs (id, graph) VALUES ($1, $2::json);", [
      id,
      graphStr
    ])
    .then(dbRes => res.send({}))
    .catch(e => res.status(500))
    .then(() => client.end());
});

app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname + "/client/build/index.html"));
});

const online = [];
const ids = {};
io.on("connection", socket => {
  socket.on("add-user", name => {
    socket.username = name;
    ids[name] = socket.id;
    online.push(name);
    io.emit("update-online", online);
  });

  socket.on("send-data", msg => {
    const { username, graphic, id } = msg;
    io.to(ids[username]).emit("recieve-data", { graphic, id });
  });

  socket.on("request-graph", username => {
    io.to(ids[username]).emit("request-graph", socket.username);
  });
  
  socket.on("recieve-graph", msg => {
    io.to(ids[msg.username]).emit("recieve-graph", msg.graph);
  });

  socket.on("disconnect", () => {
    if (online.includes(socket.username)) {
      online.splice(online.indexOf(socket.username), 1);
      io.emit("update-online", online);
    }
  });
});

const PORT = process.env.PORT || 5000;
http.listen(PORT, () => {
  console.log(`Listening on port ${PORT}`);
});
