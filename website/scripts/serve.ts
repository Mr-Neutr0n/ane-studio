import { resolve, sep } from "node:path";
import { securityHeaders } from "./headers";

const output = resolve(import.meta.dir, "../dist");
const port = Number(process.env.PORT || 4173);
const mime: Record<string, string> = {
  html: "text/html; charset=utf-8",
  css: "text/css; charset=utf-8",
  js: "text/javascript; charset=utf-8",
  json: "application/json; charset=utf-8",
  webmanifest: "application/manifest+json; charset=utf-8",
  svg: "image/svg+xml",
  webp: "image/webp",
  png: "image/png",
  txt: "text/plain; charset=utf-8",
  xml: "application/xml; charset=utf-8",
};

const server = Bun.serve({
  hostname: "127.0.0.1",
  port,
  async fetch(request) {
    const headers = { ...securityHeaders };
    if (!["GET", "HEAD"].includes(request.method)) {
      return new Response("Method not allowed", { status: 405, headers });
    }
    let pathname: string;
    try {
      pathname = decodeURIComponent(new URL(request.url).pathname);
    } catch {
      return new Response("Bad request", { status: 400, headers });
    }
    if (pathname.includes("\0")) return new Response("Bad request", { status: 400, headers });
    const path = resolve(output, `.${pathname === "/" ? "/index.html" : pathname}`);
    if (!path.startsWith(`${output}${sep}`)) {
      return new Response("Not found", { status: 404, headers });
    }
    const file = Bun.file(path);
    if (!(await file.exists())) return new Response("Not found", { status: 404, headers });
    headers["Content-Type"] = mime[path.split(".").at(-1) || ""] || "application/octet-stream";
    headers["Cache-Control"] = "no-store";
    return new Response(request.method === "HEAD" ? null : file, { headers });
  },
});
console.log(`ANE Studio preview: ${server.url}`);
