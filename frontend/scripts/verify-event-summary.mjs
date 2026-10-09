import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import ts from "typescript";

const source = await readFile(new URL("../src/eventSummary.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } });
const { eventSummary } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);
const event = (event_type, payload) => ({ event_type, payload });
assert.equal(eventSummary(event("assistant_message", { content: "承认误解，解释修改旧文件的原因。" })), "承认误解，解释修改旧文件的原因。");
assert.equal(eventSummary(event("assistant_message", { content: "摘要", full_content: "完整模型回答" })), "完整模型回答");
assert.match(eventSummary(event("decision", { rationale: "创建独立计算器", path: "new-calculator/index.html" })), /new-calculator\/index.html/);
assert.match(eventSummary(event("claim_review", { claims: [{ source_matched: false }] })), /1 条未匹配证据；未替换模型回答/);
assert.match(eventSummary(event("model_response", { tool_names: ["create"], tool_targets: ["new.html"], protocol: "responses-tools" })), /create.*new.html/);
console.log("Event summary checks passed.");
