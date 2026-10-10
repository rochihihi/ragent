// Real Vue build + isolated Python backend. No provider requests or user-profile writes.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { existsSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const root = fileURLToPath(new URL("../../", import.meta.url));
const reserved = net.createServer(); reserved.listen(0, "127.0.0.1"); await once(reserved, "listening");
const port = reserved.address().port; await new Promise(resolve => reserved.close(resolve));
const python = path.join(root, ".venv/Scripts/python.exe");
const server = spawn(python, [path.join(root, "tests/skills_ui_server.py"), "--port", String(port)], { cwd: root, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
let output = "", stderr = "";
server.stdout.on("data", chunk => { output += chunk; }); server.stderr.on("data", chunk => { stderr += chunk; });
const url = `http://127.0.0.1:${port}`;
let browser;
const artifact = path.join(root, "build/skills-ui-check");
await mkdir(artifact, { recursive: true });
try {
  const deadline = Date.now() + 20000;
  while (true) {
    if (server.exitCode !== null) throw new Error(`Fixture exited: ${stderr}`);
    try { if ((await fetch(`${url}/studio`)).ok) break; } catch {}
    if (Date.now() > deadline) throw new Error(`Fixture timeout: ${stderr}`);
    await new Promise(resolve => setTimeout(resolve, 150));
  }
  const folder = JSON.parse(output.split("\n").find(line => line.startsWith("FIXTURE:")).slice(8)).folder;
  const executable = [process.env.SKILLS_BROWSER, "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe", "C:/Program Files/Google/Chrome/Application/chrome.exe"].find(p => p && existsSync(p));
  browser = await chromium.launch({ executablePath: executable, headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 940 } });
  const errors = []; page.on("pageerror", e => errors.push(e.message));
  await page.goto(`${url}/studio`);
  await page.getByRole("button", { name: "技能", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "技能管理" });
  await dialog.getByText("1 个技能无法加载").waitFor();
  await dialog.getByRole("button", { name: "＋ 导入技能" }).click();
  await dialog.locator('input[webkitdirectory]').setInputFiles(folder);
  await dialog.getByRole("button", { name: "检查并预览" }).click();
  await dialog.getByText("3 个文件 ·", { exact: false }).waitFor();
  assert.equal(await dialog.getByRole("button", { name: "确认导入", exact: true }).isEnabled(), false);
  await dialog.getByLabel("我已确认来源可信。导入不会执行脚本或安装依赖。").check();
  await dialog.getByRole("button", { name: "确认导入", exact: true }).click();
  await dialog.getByLabel("本会话使用方式").waitFor();
  assert.equal(await dialog.getByLabel("本会话使用方式").inputValue(), "auto");
  await dialog.getByRole("button", { name: "references/rules.md", exact: true }).click();
  await dialog.locator("pre").filter({ hasText: "金额是每行总金额。" }).waitFor();
  await dialog.getByLabel("本会话使用方式").selectOption("pinned");
  await dialog.getByRole("button", { name: "保存使用方式" }).click();
  await dialog.getByText("本会话的技能使用方式已保存。").waitFor();
  await dialog.getByRole("button", { name: "修复执行权限", exact: true }).click();
  await dialog.getByRole("button", { name: "确认修复并备份", exact: true }).click();
  await dialog.getByText("已按项目权限重新保存", { exact: false }).waitFor();
  assert.equal(await dialog.getByLabel("本会话使用方式").inputValue(), "pinned");
  await dialog.getByRole("button", { name: "编辑说明" }).click();
  const editor = dialog.getByLabel("编辑技能说明");
  await editor.fill((await editor.inputValue()) + "\n必须使用实际结果。\n");
  await dialog.getByRole("button", { name: "保存说明", exact: true }).click();
  await dialog.getByText("说明已更新，配套资源保留。", { exact: false }).waitFor();
  await page.screenshot({ path: path.join(artifact, "desktop.png") });
  const overflow = await dialog.evaluate(node => node.scrollWidth > node.clientWidth + 2);
  assert.equal(overflow, false, "Desktop dialog overflows horizontally");
  await page.setViewportSize({ width: 720, height: 900 });
  await page.screenshot({ path: path.join(artifact, "narrow.png") });
  assert.equal(await dialog.evaluate(node => node.scrollWidth > node.clientWidth + 2), false);
  await page.setViewportSize({ width: 1440, height: 940 });
  await dialog.getByRole("button", { name: "删除技能", exact: true }).click();
  await dialog.getByRole("button", { name: "确认移除并备份" }).click();
  await dialog.getByText("已从项目技能中移除", { exact: false }).waitFor();
  const list = await (await fetch(`${url}/studio-api/sessions/skills-ui/skills`)).json();
  assert.equal(list.items.length, 0);
  await dialog.getByRole("button", { name: "历史版本", exact: true }).click();
  await dialog.getByRole("heading", { name: "历史版本与恢复" }).waitFor();
  assert.ok(await dialog.getByRole("button", { name: "恢复", exact: true }).count() >= 2);
  await dialog.getByRole("button", { name: "恢复", exact: true }).first().click();
  await dialog.getByRole("button", { name: "确认恢复", exact: true }).click();
  await dialog.getByText("历史版本已恢复；已有任务不切换，外部操作未撤销。").waitFor();
  await dialog.locator("summary").filter({ hasText: "发布、灰度与冒烟测试" }).click();
  await dialog.getByLabel("新版流量（%）").fill("100");
  await dialog.getByLabel("指标异常自动回退后续任务").check();
  await dialog.getByLabel("最少执行样本").fill("3");
  await dialog.getByRole("button", { name: "保存发布策略", exact: true }).click();
  await dialog.getByText("已发布；只影响后续新任务。").waitFor();
  await page.screenshot({ path: path.join(artifact, "release.png") });
  assert.deepEqual(errors, []);
  console.log("PASS: folder import, trust/preview, lazy resources, modes, permission repair/backup, edit, delete/history restore, release policy, desktop/narrow layout.");
  console.log(`Screenshots: ${artifact}`);
} finally {
  await browser?.close();
  try { await fetch(`${url}/_test/stop`, { method: "POST" }); } catch {}
  if (server.exitCode === null) await Promise.race([once(server, "exit"), new Promise(resolve => setTimeout(resolve, 3000))]);
  if (server.exitCode === null) server.kill();
}
