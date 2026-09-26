import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/studio-api": "http://127.0.0.1:2002", "/providers": "http://127.0.0.1:2002", "/provider-models": "http://127.0.0.1:2002", "/system": "http://127.0.0.1:2002", "/studio-brand": "http://127.0.0.1:2002" } },
});
