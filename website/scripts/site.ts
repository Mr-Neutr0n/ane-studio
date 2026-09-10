export function resolveSiteURL(configured: unknown, previewHost?: string): string | null {
  if (configured === null && !previewHost) return null;
  const candidate = configured === null ? `https://${previewHost}/` : configured;
  if (typeof candidate !== "string")
    throw new Error("site.json url must be null or an HTTPS origin.");
  const url = new URL(candidate);
  if (
    url.protocol !== "https:" ||
    url.username ||
    url.password ||
    url.port ||
    url.pathname !== "/" ||
    url.search ||
    url.hash
  ) {
    throw new Error(
      "Site URL must be a plain HTTPS origin without credentials, port, path, or query.",
    );
  }
  if (configured === null && !url.hostname.endsWith(".vercel.app")) {
    throw new Error("Automatic preview URL must be a Vercel deployment hostname.");
  }
  return url.href;
}
