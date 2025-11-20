"use client";

import Link from "next/link";
import Image from "next/image";

export function Footer() {
  return (
    <footer
      style={{
        marginTop: "auto",
        borderTop: `1px solid var(--border-color)`,
        backgroundColor: "var(--bg-primary)",
        width: "100%",
        padding: "var(--spacing-xl) var(--spacing-md)",
      }}
    >
      <div
        style={{
          maxWidth: "1200px",
          margin: "0 auto",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "var(--spacing-lg)",
          textAlign: "center",
        }}
      >
        {/* Ko-fi Button */}
        <div>
          <a
            href="https://ko-fi.com/N4N31DPNUS"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: "inline-block",
              transition: "opacity 0.2s",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.opacity = "0.75")}
            onMouseLeave={(e) => (e.currentTarget.style.opacity = "1")}
            lang="en"
          >
            <Image
              width={145}
              height={36}
              src="https://storage.ko-fi.com/cdn/kofi2.png?v=6"
              alt="Buy Me a Coffee at ko-fi.com"
              style={{ display: "block" }}
              unoptimized
            />
          </a>
        </div>

        {/* Legal Links */}
        <nav
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "center",
            gap: "var(--spacing-md)",
          }}
          aria-label="Footer navigation"
        >
          <a
            href="https://discord.gg/9rtwCKp2"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              fontSize: "14px",
              fontWeight: 500,
              color: "var(--text-secondary)",
              textDecoration: "none",
              padding: "var(--spacing-xs) var(--spacing-sm)",
              transition: "color 0.2s",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text-primary)")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-secondary)")}
            lang="en"
          >
            Support
          </a>
          <span
            style={{
              color: "var(--border-color)",
              userSelect: "none",
            }}
            aria-hidden="true"
          >
            •
          </span>
          <Link
            href="/terms"
            style={{
              fontSize: "14px",
              fontWeight: 500,
              color: "var(--text-secondary)",
              textDecoration: "none",
              padding: "var(--spacing-xs) var(--spacing-sm)",
              transition: "color 0.2s",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text-primary)")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-secondary)")}
            lang="en"
          >
            Terms of Service
          </Link>
          <span
            style={{
              color: "var(--border-color)",
              userSelect: "none",
            }}
            aria-hidden="true"
          >
            •
          </span>
          <Link
            href="/privacy"
            style={{
              fontSize: "14px",
              fontWeight: 500,
              color: "var(--text-secondary)",
              textDecoration: "none",
              padding: "var(--spacing-xs) var(--spacing-sm)",
              transition: "color 0.2s",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text-primary)")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-secondary)")}
            lang="en"
          >
            Privacy Policy
          </Link>
          <span
            style={{
              color: "var(--border-color)",
              userSelect: "none",
            }}
            aria-hidden="true"
          >
            •
          </span>
          <Link
            href="/accessibility"
            style={{
              fontSize: "14px",
              fontWeight: 500,
              color: "var(--text-secondary)",
              textDecoration: "none",
              padding: "var(--spacing-xs) var(--spacing-sm)",
              transition: "color 0.2s",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text-primary)")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-secondary)")}
            lang="en"
          >
            Accessibility
          </Link>
          <span
            style={{
              color: "var(--border-color)",
              userSelect: "none",
            }}
            aria-hidden="true"
          >
            •
          </span>
          <a
            href="https://github.com/rgilks/writeo"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              fontSize: "14px",
              fontWeight: 500,
              color: "var(--text-secondary)",
              textDecoration: "none",
              padding: "var(--spacing-xs) var(--spacing-sm)",
              transition: "color 0.2s",
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text-primary)")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-secondary)")}
            lang="en"
          >
            <svg
              style={{ width: "16px", height: "16px" }}
              fill="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                fillRule="evenodd"
                d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                clipRule="evenodd"
              />
            </svg>
            GitHub
          </a>
        </nav>
      </div>
    </footer>
  );
}
