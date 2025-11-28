"use client";

import Link from "next/link";

export function SkipLink() {
  return (
    <Link
      href="#main-content"
      className="skip-link"
      style={{
        position: "absolute",
        top: "-40px",
        left: "0",
        background: "var(--primary-color)",
        color: "white",
        padding: "var(--spacing-sm) var(--spacing-md)",
        textDecoration: "none",
        borderRadius: "0 0 var(--border-radius) 0",
        zIndex: 1000,
        fontWeight: 600,
        fontSize: "16px",
      }}
      onFocus={(e) => {
        e.currentTarget.style.top = "0";
      }}
      onBlur={(e) => {
        e.currentTarget.style.top = "-40px";
      }}
    >
      Skip to main content
    </Link>
  );
}
