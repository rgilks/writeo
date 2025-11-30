import type { NextConfig } from "next";

const isDevelopment = process.env.NODE_ENV === "development";

// CSP configuration - stricter in production
const buildCSP = (): string => {
  const scriptSrc = isDevelopment
    ? "'self' 'unsafe-inline' 'unsafe-eval' https://static.cloudflareinsights.com"
    : "'self' 'unsafe-inline' https://static.cloudflareinsights.com";

  const apiBase = process.env.NEXT_PUBLIC_API_BASE || "https://your-api-worker.workers.dev";

  return [
    "default-src 'self'",
    `script-src ${scriptSrc}`,
    "style-src 'self' 'unsafe-inline'", // Next.js requires 'unsafe-inline' for injected styles
    "img-src 'self' data: blob: https://storage.ko-fi.com",
    "font-src 'self'",
    "media-src 'self' data:",
    `connect-src 'self' ${apiBase} https://*.groq.com https://*.cloudflare.com data:`,
    "object-src 'none'",
    "frame-ancestors 'self'",
    "form-action 'self'",
    "base-uri 'self'",
  ].join("; ");
};

// Security headers configuration
const securityHeaders = [
  {
    key: "Content-Security-Policy",
    value: buildCSP(),
  },
  {
    key: "X-Content-Type-Options",
    value: "nosniff",
  },
  {
    key: "X-Frame-Options",
    value: "DENY",
  },
  {
    key: "X-XSS-Protection",
    value: "1; mode=block",
  },
  {
    key: "Referrer-Policy",
    value: "strict-origin-when-cross-origin",
  },
];

const nextConfig: NextConfig = {
  output: "standalone",
  eslint: {
    // Enable ESLint during builds
    ignoreDuringBuilds: false,
    // Only run ESLint on these directories
    dirs: ["app", "components"],
  },
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "storage.ko-fi.com",
        pathname: "/**",
      },
    ],
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: securityHeaders,
      },
      // Ensure service worker and manifest are served with correct headers
      {
        source: "/service-worker.js",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=0, must-revalidate",
          },
          {
            key: "Service-Worker-Allowed",
            value: "/",
          },
        ],
      },
      {
        source: "/manifest.json",
        headers: [
          {
            key: "Content-Type",
            value: "application/manifest+json",
          },
        ],
      },
    ];
  },
};

export default nextConfig;

// Initialize OpenNext for local development with Cloudflare bindings
if (isDevelopment) {
  try {
    const { initOpenNextCloudflareForDev } = require("@opennextjs/cloudflare");
    initOpenNextCloudflareForDev();
  } catch {
    // OpenNext not installed yet, skip initialization
    console.warn("OpenNext not available for dev initialization");
  }
}
