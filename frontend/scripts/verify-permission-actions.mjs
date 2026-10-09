import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import ts from "typescript";
import { parse, compileTemplate } from "vue/compiler-sfc";

const source = await readFile(new URL("../src/permissionActions.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } });
const { approvalActions } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);
const command = ["powershell", "-Command", "Write-Output 'a b'; Get-Content x"];
const actions = approvalActions({ action: "batch", actions: [
  { action: "create", path: "calculator.py", content: "print('ok')" },
  { action: "batch", actions: [
    { action: "run_command", command },
    { action: "mcp", mcp_arguments: { path: "x", count: 3 } },
  ] },
] });
assert.equal(actions.length, 3);
assert.deepEqual(actions.map(a => a.number), ["1", "2.1", "2.2"]);
assert.deepEqual(actions[1].command, command);
assert.equal(actions[0].raw.content, "print('ok')");
assert.deepEqual(actions[2].raw.mcp_arguments, { path: "x", count: 3 });
assert.deepEqual(approvalActions(null), []);
const deletion = approvalActions({ action: "delete_path", path: "calculator.html",
  rationale: "删除用户指定的文件" });
assert.equal(deletion[0].action, "delete_path");
assert.deepEqual(deletion[0].command, []);
assert.equal(deletion[0].raw.path, "calculator.html");
assert.equal(deletion[0].rationale, "删除用户指定的文件");
// The details renderer and legacy command renderer must be alternative branches.
const dialog = await readFile(new URL("../vue/src/PermissionDialog.vue", import.meta.url), "utf8");
const { descriptor, errors } = parse(dialog);
assert.deepEqual(errors, []);
assert.ok(descriptor.template, "Permission dialog must have a Vue template");
const compiled = compileTemplate({
  source: descriptor.template.content,
  filename: "PermissionDialog.vue",
  id: "permission-display-regression",
});
assert.deepEqual(compiled.errors, []);
let exclusiveCommandDisplay = false;
function inspect(node) {
  // Vue compiles adjacent v-if/v-else sections into one conditional node.
  if (node.branches?.length === 2
      && node.branches[0].condition?.loc.source === "actions.length"
      && node.branches[0].children[0]?.loc.source.includes('class="permission-batch"')
      && node.branches[1].condition === undefined
      && node.branches[1].children[0]?.loc.source.includes("将执行的完整命令")) {
    exclusiveCommandDisplay = true;
  }
  for (const child of [...(node.children ?? []), ...(node.branches ?? [])]) {
    if (typeof child === "object" && child !== null) inspect(child);
  }
}
inspect(compiled.ast);
assert.ok(exclusiveCommandDisplay, "Action details and legacy command display must be mutually exclusive");
const hostBatch = approvalActions({ action: "batch", actions: [
  { action: "run_command", execution_mode: "sandbox", command: ["python", "check.py"] },
  { action: "start_terminal", execution_mode: "host", command: ["python", "app.py"] },
] });
assert.equal(hostBatch[1].raw.execution_mode, "host");
assert.ok(dialog.includes('!hostExecution &&'), "Host execution must not show a session-grant button");
assert.ok(dialog.includes("允许本次沙箱外执行？"), "Host approval title must disclose the boundary");
assert.ok(dialog.includes("将启动的本地 MCP 服务完整命令"), "MCP host approval must display the server argv");
console.log("Permission action display: passed");
