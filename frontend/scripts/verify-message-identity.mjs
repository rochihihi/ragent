import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import ts from "typescript";

const source = await readFile(new URL("../src/messageIdentity.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext } });
const { MessageIdentity } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);
let counter = 0;
const ids = new MessageIdentity(() => String(++counter));
const first = ids.acquire("s", "send");
assert.equal(ids.acquire("s", "send"), first, "network failure must retain identity");
assert.notEqual(ids.acquire("other", "send"), first, "session scopes are independent");
ids.acknowledge("s", "wrong");
assert.equal(ids.acquire("s", "send"), first);
ids.acknowledge("s", first);
assert.notEqual(ids.acquire("s", "send"), first, "intentional repeated text is a new operation");
const changed = ids.acquire("s", "edited");
assert.equal(ids.acquire("s", "edited"), changed);
console.log("Message retry identity tests passed");
