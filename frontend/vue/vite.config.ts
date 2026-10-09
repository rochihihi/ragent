import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { fileURLToPath } from "node:url";

export default defineConfig({
  root: fileURLToPath(new URL(".", import.meta.url)),
  plugins: [vue()],
  build: { outDir: "../dist", emptyOutDir: true },
  server: {
    port: 5174, strictPort: true,
    proxy: Object.fromEntries([
      "/studio-api", "/providers", "/provider-models", "/system", "/studio-brand", "/mcp-servers", "/sandbox-api",
    ].map(path => [path, "http://127.0.0.1:2002"])),
  },
});
