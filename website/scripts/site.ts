/**
 * `productionHost` is Vercel's VERCEL_PROJECT_PRODUCTION_URL. It is a vercel.app
 * hostname until a custom domain is attached, then the shortest custom domain.
 */
export function resolveSiteURL(configured: unknown, productionHost?: string): string | null {
  if (configured === null && !productionHost) return null;
  const candidate = configured === null ? `https://${productionHost}/` : configured;
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
  return url.href;
}
