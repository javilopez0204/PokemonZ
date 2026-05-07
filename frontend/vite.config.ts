import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// El backend FastAPI sirve los estáticos del build, así que en producción
// las llamadas relativas a /api/* van al mismo origen. En desarrollo usamos
// el proxy de Vite hacia uvicorn local en :8080.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8080",
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
