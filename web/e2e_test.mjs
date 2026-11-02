// Brauzer demosini haqiqiy brauzerda (headless Edge) oxirigacha sinash, Chrome DevTools Protocol orqali.
// Ishlatish:  node web/e2e_test.mjs <sahifa URL>   (Node 22+; SHOT=fayl.png bo'lsa skrinshot ham oladi)
// Qadamlar: sahifa ochiladi -> model yuklanishi kutiladi -> 2-namuna bosiladi -> natija va konsol xatolari chiqariladi.
import { spawn } from "node:child_process";

const EDGE = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const URL = process.argv[2] || "http://127.0.0.1:8765/";
const PORT = 9333;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const edge = spawn(EDGE, ["--headless=new", `--remote-debugging-port=${PORT}`, "--no-first-run",
  "--user-data-dir=" + process.env.TEMP + "\\edge_e2e", "--autoplay-policy=no-user-gesture-required", "about:blank"],
  { stdio: "ignore" });

let tab;
for (let i = 0; i < 40 && !tab; i++) {
  await sleep(500);
  try { tab = (await (await fetch(`http://127.0.0.1:${PORT}/json`)).json()).find((t) => t.type === "page"); } catch {}
}
if (!tab) { console.log("Edge ishga tushmadi"); edge.kill(); process.exit(1); }

const ws = new WebSocket(tab.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0; const wait = new Map(); const xatolar = [];
ws.onmessage = (m) => {
  const d = JSON.parse(m.data);
  if (d.id && wait.has(d.id)) { wait.get(d.id)(d); wait.delete(d.id); }
  if (d.method === "Runtime.exceptionThrown") xatolar.push("EXCEPTION: " + d.params.exceptionDetails.exception?.description);
  if (d.method === "Runtime.consoleAPICalled" && ["error", "warning"].includes(d.params.type))
    xatolar.push(d.params.type.toUpperCase() + ": " + d.params.args.map((a) => a.value ?? a.description).join(" ").slice(0, 300));
};
const cdp = (method, params = {}) => new Promise((r) => { const i = ++id; wait.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expr) => (await cdp("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;

await cdp("Runtime.enable");
await cdp("Page.enable");
await cdp("Page.navigate", { url: URL });
const t0 = Date.now();
let holat = "";
while (Date.now() - t0 < 25 * 60 * 1000) {        // model yuklanishi uchun 25 daqiqagacha
  await sleep(5000);
  holat = await ev(`document.getElementById("loadText").textContent + " | " + document.getElementById("loadPct").textContent`);
  const err = await ev(`document.getElementById("err").textContent`);
  if (err) { console.log("SAHIFA XATOSI:", err); break; }
  if (holat?.includes("tayyor")) break;
}
console.log(`[${((Date.now() - t0) / 1000).toFixed(0)} s] yuklash holati:`, holat);
console.log("crossOriginIsolated:", await ev("crossOriginIsolated"), "| yadrolar:", await ev("navigator.hardwareConcurrency"),
  "| webgpu:", await ev("!!navigator.gpu"));

if (holat?.includes("tayyor")) {
  await ev(`document.querySelectorAll(".ex .btn")[1].click()`);
  const t1 = Date.now();
  let out = "";
  while (Date.now() - t1 < 5 * 60 * 1000) {
    await sleep(2000);
    out = await ev(`document.getElementById("out").textContent`);
    if (out && !out.includes("Tinglanmoqda")) break;
  }
  console.log(`[${((Date.now() - t1) / 1000).toFixed(0)} s] NATIJA:`, out);
  console.log("META:", await ev(`document.getElementById("meta").textContent`));
  console.log("REF:", (await ev(`document.getElementById("ref").textContent`))?.slice(0, 80));
}
if (process.env.SHOT) {          // natija ko'rinib turgan holatdagi skrinshot
  await cdp("Emulation.setDeviceMetricsOverride", { width: 1280, height: 1450, deviceScaleFactor: 1, mobile: false });
  await ev(`window.scrollTo(0, 0)`);
  await sleep(1500);
  const shot = await cdp("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
  (await import("node:fs")).writeFileSync(process.env.SHOT, Buffer.from(shot.result.data, "base64"));
  console.log("skrinshot:", process.env.SHOT);
}
console.log("Konsol xato/ogohlantirishlar:", xatolar.length ? "\n  " + xatolar.slice(0, 12).join("\n  ") : "yo'q");
ws.close();
edge.kill();
process.exit(0);
