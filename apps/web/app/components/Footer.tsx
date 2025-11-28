"use client";

import React, { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { GitHubIcon } from "./icons/GitHubIcon";

const KOFI_URL = "https://ko-fi.com/N4N31DPNUS";
const DISCORD_URL = "https://discord.gg/9rtwCKp2";
const GITHUB_URL = "https://github.com/rgilks/writeo";
const KOFI_IMAGE_URL = "https://storage.ko-fi.com/cdn/kofi2.png?v=6";

const LINK_STYLES = {
  fontSize: "14px",
  fontWeight: 500,
  color: "var(--text-secondary)",
  textDecoration: "none",
  padding: "var(--spacing-xs) var(--spacing-sm)",
  transition: "color 0.2s",
} as const;

const SEPARATOR_STYLES = {
  color: "var(--border-color)",
  userSelect: "none" as const,
} as const;

interface FooterLinkProps {
  href: string;
  children: React.ReactNode;
  external?: boolean;
  icon?: React.ReactNode;
}

function FooterLink({ href, children, external = false, icon }: FooterLinkProps) {
  const [isHovered, setIsHovered] = useState(false);
  const linkStyle = {
    ...LINK_STYLES,
    color: isHovered ? "var(--text-primary)" : "var(--text-secondary)",
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
      {children}
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

interface KoFiButtonProps {
  url: string;
  imageUrl: string;
}

function KoFiButton({ url, imageUrl }: KoFiButtonProps) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      style={{
        display: "inline-block",
        transition: "opacity 0.2s",
        opacity: isHovered ? 0.75 : 1,
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <Image
        width={145}
        height={36}
        src={imageUrl}
        alt="Buy Me a Coffee at ko-fi.com"
        style={{ display: "block" }}
        loading="lazy"
        priority={false}
      />
    </a>
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
        <KoFiButton url={KOFI_URL} imageUrl={KOFI_IMAGE_URL} />

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
