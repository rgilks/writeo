"use client";

import React, { useState, type ReactNode, type ReactElement } from "react";
import Link from "next/link";
import { GitHubIcon } from "./icons/GitHubIcon";

const DISCORD_URL = "https://discord.gg/YxuFAXWuzw";
const GITHUB_URL = "https://github.com/rgilks/writeo";

const LINK_STYLES = {
  fontSize: "14px",
  fontWeight: 500,
  color: "var(--text-secondary)",
  textDecoration: "none",
  padding: "var(--spacing-xs) var(--spacing-md)",
  transition: "all 0.2s ease",
  borderRadius: "4px",
  position: "relative" as const,
} as const;

const SEPARATOR_STYLES = {
  color: "var(--border-color)",
  userSelect: "none" as const,
  fontSize: "12px",
  padding: "0 var(--spacing-xs)",
} as const;

interface FooterLinkProps {
  href: string;
  children: ReactNode;
  external?: boolean;
  icon?: ReactElement;
}

function FooterLink({ href, children, external = false, icon }: FooterLinkProps) {
  const [isHovered, setIsHovered] = useState(false);
  const linkStyle = {
    ...LINK_STYLES,
    color: isHovered ? "var(--text-primary)" : "var(--text-secondary)",
    backgroundColor: isHovered ? "rgba(0, 0, 0, 0.02)" : "transparent",
    ...(icon && { display: "inline-flex", alignItems: "center", gap: "6px" }),
  };

  const commonProps = {
    style: linkStyle,
    onMouseEnter: () => setIsHovered(true),
    onMouseLeave: () => setIsHovered(false),
  };

  if (external) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" {...commonProps}>
        {icon}
        {children}
      </a>
    );
  }

  return (
    <Link href={href} {...commonProps}>
      <span>{children}</span>
    </Link>
  );
}

function Separator() {
  return (
    <span style={SEPARATOR_STYLES} aria-hidden="true">
      •
    </span>
  );
}

interface FooterLinkData {
  href: string;
  label: string;
  external?: boolean;
  hasIcon?: boolean;
}

const FOOTER_LINKS: FooterLinkData[] = [
  { href: DISCORD_URL, label: "Support", external: true },
  { href: "/terms", label: "Terms of Service" },
  { href: "/privacy", label: "Privacy Policy" },
  { href: "/accessibility", label: "Accessibility" },
  { href: GITHUB_URL, label: "GitHub", external: true, hasIcon: true },
];

export function Footer() {
  return (
    <footer
      lang="en"
      style={{
        marginTop: "auto",
        borderTop: "1px solid var(--border-color)",
        backgroundColor: "var(--bg-primary)",
        width: "100%",
        padding: "var(--spacing-2xl) var(--spacing-md) var(--spacing-xl)",
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
          gap: "var(--spacing-xl)",
          textAlign: "center",
        }}
      >
        <nav
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "center",
            gap: "var(--spacing-xs)",
            padding: "var(--spacing-sm) 0",
          }}
          aria-label="Footer navigation"
        >
          {FOOTER_LINKS.map((link, index) => (
            <React.Fragment key={link.href}>
              <FooterLink
                href={link.href}
                external={link.external}
                icon={link.hasIcon ? <GitHubIcon size={16} /> : undefined}
              >
                {link.label}
              </FooterLink>
              {index < FOOTER_LINKS.length - 1 && <Separator />}
            </React.Fragment>
          ))}
        </nav>
      </div>
    </footer>
  );
}
