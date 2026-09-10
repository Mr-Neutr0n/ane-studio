export const securityHeaders: Record<string, string> = {
  "Content-Security-Policy":
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; font-src 'self'; connect-src 'none'; manifest-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; form-action 'none'",
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "Permissions-Policy": "camera=(), microphone=(), geolocation=(), browsing-topics=()",
  "X-Frame-Options": "DENY",
};

// WebKit upgrades even loopback HTTP URLs. Keep this production-only directive off the local preview.
export const productionSecurityHeaders: Record<string, string> = {
  ...securityHeaders,
  "Content-Security-Policy": `${securityHeaders["Content-Security-Policy"]}; upgrade-insecure-requests`,
};
