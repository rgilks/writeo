import Link from "next/link";

interface SectionCardProps {
  title: string;
  children: React.ReactNode;
  isFirst?: boolean;
  highlight?: boolean;
}

export function SectionCard({
  title,
  children,
  isFirst = false,
  highlight = false,
}: SectionCardProps) {
  return (
    <div
      className="card"
      style={{
        marginTop: isFirst ? "var(--spacing-xl)" : "var(--spacing-lg)",
        ...(highlight && { backgroundColor: "var(--primary-bg-light)" }),
      }}
    >
      <h2 style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}>
        {title}
      </h2>
      {children}
    </div>
  );
}

export function ContentList({ children }: { children: React.ReactNode }) {
  return (
    <ul
      style={{
        marginLeft: "var(--spacing-lg)",
        marginBottom: "var(--spacing-md)",
        lineHeight: "1.5",
        paddingLeft: "var(--spacing-lg)",
      }}
    >
      {children}
    </ul>
  );
}

interface TextProps {
  children: React.ReactNode;
  note?: boolean;
}

export function Text({ children, note = false }: TextProps) {
  return (
    <p
      style={{
        marginBottom: "var(--spacing-md)",
        lineHeight: "1.5",
        ...(note && { fontSize: "16px", color: "var(--text-secondary)" }),
      }}
    >
      {children}
    </p>
  );
}

export function ExternalLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      style={{ color: "var(--primary-color)", textDecoration: "underline" }}
    >
      {children}
    </a>
  );
}

export function InternalLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <Link href={href} style={{ color: "var(--primary-color)", textDecoration: "underline" }}>
      <span>{children}</span>
    </Link>
  );
}
