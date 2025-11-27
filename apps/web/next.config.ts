import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // Disable ESLint during build - ESLint config is incomplete (missing @typescript-eslint packages)
  // Linting should be done separately via npm run lint if needed
  eslint: {
    ignoreDuringBuilds: true,
  },
  async headers() {
    const isDevelopment = process.env.NODE_ENV === "development";

    // CSP configuration - stricter in production
    const scriptSrc = isDevelopment
      ? "'self' 'unsafe-inline' 'unsafe-eval' https://static.cloudflareinsights.com" // Development: allow eval for hot reload
      : "'self' 'unsafe-inline' https://static.cloudflareinsights.com"; // Production: no eval, only inline scripts (Next.js requirement)

    // style-src: Next.js requires 'unsafe-inline' for injected styles
    // This is a Next.js limitation, but we restrict it to styles only
    const styleSrc = "'self' 'unsafe-inline'";

    return [
      {
        source: "/:path*",
        headers: [
          {
            key: "Content-Security-Policy",
            value: [
              "default-src 'self'",
              `script-src ${scriptSrc}`,
              `style-src ${styleSrc}`,
              "img-src 'self' data: blob: https://storage.ko-fi.com", // Ko-fi button images
              "font-src 'self'",
              "object-src 'none'",
              "frame-ancestors 'self'",
              "form-action 'self'",
              "base-uri 'self'",
              "media-src 'self' data:",
              "connect-src 'self' " +
                (process.env.NEXT_PUBLIC_API_BASE || "https://your-api-worker.workers.dev") +
                " https://*.groq.com https://*.cloudflare.com data:", // Your API, Groq, Cloudflare
            ].join("; "),
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
        ],
      },
    ];
  },
};

export default nextConfig;

// Initialize OpenNext for local development with Cloudflare bindings
if (process.env.NODE_ENV === "development") {
  try {
    const { initOpenNextCloudflareForDev } = require("@opennextjs/cloudflare");
    initOpenNextCloudflareForDev();
  } catch (error) {
    // OpenNext not installed yet, skip initialization
    console.warn("OpenNext not available for dev initialization");
  }
}
