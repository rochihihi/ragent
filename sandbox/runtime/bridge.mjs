// Host broker. Never evaluate the requested command here or use a host shell
// to parse it on Windows. Stdout belongs exclusively to the target (MCP).
import { spawn } from "node:child_process";
import { readFile, unlink, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { SandboxManager, installWindowsSandboxAsync, checkWindowsDependenciesAsync,
  resolveSrtWin, getWindowsSandboxUserStatusAsync,
  grantWindowsAcl, revokeWindowsAcl, restoreWindowsAcl } from "@anthropic-ai/sandbox-runtime";

const action = process.argv[2];
let child;
let statusPath;
let shuttingDown = false;
let bootstrap;
const runtimeDirectory = dirname(fileURLToPath(import.meta.url));
// Pinned upstream helper + reviewed object-only parent-FDC/no-op ACL patch.
// Never choose a helper supplied by the workspace or model arguments.
const windowsHelper = join(runtimeDirectory, "ragent-srt-win.exe");
async function bootstrapWindowsRuntime() {
  if (process.platform !== "win32") return;
  const srtWin = resolveSrtWin({ path: windowsHelper });
  const user = await getWindowsSandboxUserStatusAsync({ srtWin });
  if (!user.provisioned || !user.credPresent || !user.sid) {
    throw new Error("Windows sandbox account is not provisioned; install from desktop settings.");
  }
  // initialize() verifies WFP by launching this helper as srt-sandbox BEFORE
  // its usual filesystem grants. PyInstaller's private _MEI directory is not
  // readable by that account. Grant ONLY this host-derived runtime subtree,
  // never _MEI itself / the user's Temp tree, and never grant write access.
  // Upstream maintains holder-PID leases and traversal-only parent ACEs.
  bootstrap = { sandboxUserSid: user.sid, srtWin };
  grantWindowsAcl({ ...bootstrap, read: [runtimeDirectory], write: [] });
}
async function cleanup() {
  let failure;
  if (bootstrap) {
    // Required even if initialize() failed before recording its own grants.
    // Inspect deny cleanup ourselves BEFORE reset(): upstream reset only logs
    // per-path failures, which must not be mistaken for verified success.
    const acceptable = new Set(["revoked", "stillHeld", "restored", "alreadyOriginal"]);
    // Attempt every cleanup step, even if an earlier helper call throws.
    // reset() must still close network proxies and retry its own leases.
    for (const release of [restoreWindowsAcl, revokeWindowsAcl]) {
      try {
        const items = release(bootstrap);
        if (!Array.isArray(items) || items.some(item => !acceptable.has(item.status))) {
          failure ??= new Error("Could not confirm removal of sandbox ACL leases.");
        }
      } catch (error) { failure ??= error; }
    }
    bootstrap = undefined;
  }
  try { await SandboxManager.reset(); } catch (error) { failure ??= error; }
  if (failure) throw failure;
}
async function finish(code) {
  if (shuttingDown) return;
  shuttingDown = true;
  try { await cleanup(); }
    catch (error) { console.error("RAGENT_SANDBOX_ERROR: cleanup:", error.message); code = 125; }
  process.exitCode = code;
}
process.on("SIGTERM", () => child ? child.kill("SIGTERM") : void finish(125));
process.on("SIGINT", () => child ? child.kill("SIGINT") : void finish(130));

try {
  if (action === "install") {
    if (process.platform !== "win32") throw new Error("Windows installation only");
    console.log(JSON.stringify(await installWindowsSandboxAsync({
      srtWin: resolveSrtWin({ path: windowsHelper }),
    })));
  } else if (action === "probe") {
    const check = process.platform === "win32"
      ? await checkWindowsDependenciesAsync({ srtWin: resolveSrtWin({ path: windowsHelper }) })
      : await SandboxManager.checkDependenciesAsync();
    let startupVerified = false;
    if (check.errors.length === 0 && process.platform === "win32") {
      try {
        await bootstrapWindowsRuntime();
        await SandboxManager.initialize({
          windows: { srtWin: { path: windowsHelper } },
          network: { allowedDomains: [], deniedDomains: [], strictAllowlist: true },
          filesystem: { allowRead: [runtimeDirectory], allowWrite: [], denyRead: [], denyWrite: [runtimeDirectory] },
        }, undefined, false);
        if (!SandboxManager.isSandboxingEnabled()) throw new Error("Sandbox not enabled");
        const marker = "RAGENT_SANDBOX_PROBE_OK";
        const { argv, env } = await SandboxManager.wrapWithSandboxArgv(
          `process.stdout.write('${marker}')`, { exe: process.execPath, args: ["-e"] },
          undefined, undefined, runtimeDirectory);
        await new Promise((resolve, reject) => {
          let output = "", errors = "";
          child = spawn(argv[0], argv.slice(1), { shell: false, cwd: runtimeDirectory,
            stdio: ["ignore", "pipe", "pipe"], env, windowsHide: true });
          child.stdout.on("data", data => { output = (output + data.toString()).slice(-4096); });
          child.stderr.on("data", data => { errors = (errors + data.toString()).slice(-4096); });
          child.once("error", reject);
          child.once("exit", code => code === 0 && output === marker ? resolve() :
            reject(new Error(`Sandbox startup self-test failed (exit=${code}): ${errors || output}`)));
        });
        startupVerified = true;
      } catch (error) { check.errors.push(`Sandbox startup self-test: ${error.message}`); }
      try { await cleanup(); }
      catch (error) { check.errors.push(`Sandbox cleanup self-test: ${error.message}`); }
    }
    console.log(JSON.stringify({ ...check, ready: check.errors.length === 0,
      startupVerified, platform: process.platform, backend: "anthropic-srt", version: "0.0.78" }));
  } else if (action === "execute") {
    const requestPath = process.argv[3];
    const request = JSON.parse(await readFile(requestPath, "utf8"));
    statusPath = request.statusPath;
    await unlink(requestPath); // Request is never readable by the sandbox target.
    if (!Array.isArray(request.argv) || !request.argv.length ||
        request.argv.some(arg => typeof arg !== "string" || arg.includes("\0"))) {
      throw new Error("Invalid argv");
    }
    // TLS termination is deliberately disabled; no CA installation or MITM.
    if (process.platform === "win32") request.config.windows = {
      srtWin: { path: windowsHelper },
    };
    await bootstrapWindowsRuntime();
    await SandboxManager.initialize(request.config, undefined, false);
    if (!SandboxManager.isSandboxingEnabled()) throw new Error("Sandbox not enabled");
    const encoded = Buffer.from(JSON.stringify({ argv: request.argv, env: request.env })).toString("base64");
    // This JavaScript is ONLY evaluated by Node INSIDE the OS sandbox. Base64
    // contains no quote or shell metacharacter; arbitrary argv remains data.
    const trampoline = `const{spawn}=require('node:child_process');const r=JSON.parse(Buffer.from('${encoded}','base64').toString());const env={...process.env,...r.env};const c=spawn(r.argv[0],r.argv.slice(1),{shell:false,stdio:'inherit',env});c.on('error',e=>{console.error(e.message);process.exitCode=127});c.on('exit',(n,s)=>{process.exitCode=n??1});process.on('SIGTERM',()=>c.kill('SIGTERM'));process.on('SIGINT',()=>c.kill('SIGINT'));`;
    let argv, env;
    if (process.platform === "win32") {
      ({ argv, env } = await SandboxManager.wrapWithSandboxArgv(
        trampoline, { exe: process.execPath, args: ["-e"] }, undefined, undefined, request.cwd));
    } else {
      const quote = value => "'" + value.replaceAll("'", "'\\''") + "'";
      const wrapped = await SandboxManager.wrapWithSandbox(`${quote(process.execPath)} -e ${quote(trampoline)}`);
      argv = ["/bin/sh", "-c", wrapped];
      env = process.env;
    }
    child = spawn(argv[0], argv.slice(1), { shell: false, cwd: request.cwd,
      stdio: ["pipe", "inherit", "inherit"], env });
    child.stdin.on("error", () => {}); // Target may exit before its parent input closes.
    process.stdin.pipe(child.stdin);
    if (request.interactive) process.stdin.once("end", () => {
      if (child.exitCode !== null) return;
      if (process.platform === "win32") {
        // SDK closes stdin on teardown. Kill the full known helper tree, not
        // just the Node proxy host, before releasing its filesystem grants.
        const stop = spawn(join(process.env.SYSTEMROOT ?? "C:\\Windows", "System32", "taskkill.exe"),
          ["/PID", String(child.pid), "/T", "/F"], { shell: false, stdio: "ignore", windowsHide: true });
        stop.on("error", () => child.kill());
      } else child.kill("SIGTERM");
    });
    child.once("spawn", async () => {
      if (statusPath) await writeFile(statusPath, JSON.stringify({ state: "ready" }), { mode: 0o600 });
    });
    child.once("error", async error => {
      if (statusPath) await writeFile(statusPath, JSON.stringify({ state: "error", message: error.message }), { mode: 0o600 });
      console.error("RAGENT_SANDBOX_ERROR:", error.message); await finish(125);
    });
    child.once("exit", async (code) => { await finish(code ?? 1); });
  } else throw new Error("Unknown bridge action");
} catch (error) {
  if (statusPath) await writeFile(statusPath, JSON.stringify({ state: "error", message: error.message }), { mode: 0o600 });
  console.error("RAGENT_SANDBOX_ERROR:", error.code ?? "unavailable", error.message);
  await finish(125);
}
