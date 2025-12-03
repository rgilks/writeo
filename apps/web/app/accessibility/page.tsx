"use client";

import Link from "next/link";
import { Logo } from "@/app/components/Logo";

// Helper component for section cards
function SectionCard({
  title,
  children,
  isFirst = false,
}: {
  title: string;
  children: React.ReactNode;
  isFirst?: boolean;
}) {
  return (
    <div
      className="card"
      style={{ marginTop: isFirst ? "var(--spacing-xl)" : "var(--spacing-lg)" }}
    >
      <h2 style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}>
        {title}
      </h2>
      {children}
    </div>
  );
}

// Helper component for content lists
function ContentList({ children }: { children: React.ReactNode }) {
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

// Helper component for paragraph text
function Text({ children }: { children: React.ReactNode }) {
  return <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }}>{children}</p>;
}

// External link component with consistent styling
function ExternalLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-blue-600 hover:underline"
    >
      {children}
    </a>
  );
}

const LAST_UPDATED = new Date().toLocaleDateString("en-US", {
  year: "numeric",
  month: "long",
  day: "numeric",
});

export default function AccessibilityPage() {
  return (
    <>
      <header className="header">
        <div className="header-content">
          <div className="logo-group">
            <Logo />
          </div>
          <nav className="header-actions" aria-label="Accessibility actions">
            <Link href="/" className="nav-back-link">
              <span aria-hidden="true">←</span> Back to Home
            </Link>
          </nav>
        </div>
      </header>

      <div className="container">
        <div style={{ maxWidth: "800px", margin: "0 auto" }}>
          <h1 className="page-title">Accessibility Statement</h1>
          <p className="page-subtitle">Last updated: {LAST_UPDATED}</p>

          <SectionCard title="♿ Our Commitment" isFirst>
            <Text>
              Writeo is committed to ensuring digital accessibility for people with disabilities. We
              are continually improving the user experience for everyone and applying the relevant
              accessibility standards to achieve these goals.
            </Text>
          </SectionCard>

          <SectionCard title="🎯 Conformance Status">
            <Text>
              The{" "}
              <ExternalLink href="https://www.w3.org/WAI/WCAG21/quickref/?currentsidebar=%23col_overview&levels=aaa">
                Web Content Accessibility Guidelines (WCAG)
              </ExternalLink>{" "}
              defines requirements for designers and developers to improve accessibility for people
              with disabilities. Writeo aims to conform to WCAG 2.1 Level AA standards.
            </Text>
            <Text>
              <strong>Current Status:</strong> Writeo is partially conformant with WCAG 2.1 Level
              AA. Partially conformant means that some parts of the content do not fully conform to
              the accessibility standard. We are actively working to improve accessibility across
              all features.
            </Text>
          </SectionCard>

          <SectionCard title="✅ Accessibility Features">
            <Text>Writeo includes the following accessibility features:</Text>
            <ContentList>
              <li>
                <strong>Semantic HTML:</strong> Proper use of HTML5 semantic elements for better
                screen reader navigation
              </li>
              <li>
                <strong>ARIA Labels:</strong> Descriptive labels for interactive elements to assist
                screen readers
              </li>
              <li>
                <strong>Keyboard Navigation:</strong> All interactive elements are accessible via
                keyboard
              </li>
              <li>
                <strong>Color Contrast:</strong> Text and background colors meet WCAG contrast
                requirements
              </li>
              <li>
                <strong>Focus Indicators:</strong> Visible focus indicators for keyboard navigation
              </li>
              <li>
                <strong>Alt Text:</strong> Images include descriptive alt text where appropriate
              </li>
              <li>
                <strong>Responsive Design:</strong> Works across different screen sizes and devices
              </li>
            </ContentList>
          </SectionCard>

          <SectionCard title="⚠️ Known Limitations">
            <Text>
              Despite our best efforts to ensure accessibility, there may be some limitations:
            </Text>
            <ContentList>
              <li>
                Some interactive features may require mouse interaction (we're working on full
                keyboard alternatives)
              </li>
              <li>
                Error highlighting and heat map visualizations may need additional screen reader
                descriptions
              </li>
              <li>
                Some third-party services (AI feedback, grammar checking) may have their own
                accessibility limitations
              </li>
            </ContentList>
            <Text>
              We are committed to addressing these limitations and improving accessibility over
              time.
            </Text>
          </SectionCard>

          <SectionCard title="🔧 Browser and Assistive Technology Support">
            <Text>Writeo is designed to work with the following:</Text>
            <ContentList>
              <li>Modern browsers (Chrome, Firefox, Safari, Edge)</li>
              <li>Screen readers (NVDA, JAWS, VoiceOver)</li>
              <li>Keyboard-only navigation</li>
              <li>Mobile devices and tablets</li>
            </ContentList>
          </SectionCard>

          <SectionCard title="📧 Feedback and Contact">
            <Text>
              We welcome your feedback on the accessibility of Writeo. If you encounter
              accessibility barriers or have suggestions for improvement, please contact us:
            </Text>
            <ContentList>
              <li>
                Join our{" "}
                <ExternalLink href="https://discord.gg/YxuFAXWuzw">Discord server</ExternalLink> and
                use the support channel
              </li>
              <li>
                Report accessibility issues or request accommodations through our support channel
              </li>
            </ContentList>
            <Text>We aim to respond to accessibility feedback within 48 hours.</Text>
          </SectionCard>

          <SectionCard title="🔄 Ongoing Improvements">
            <Text>
              We are committed to continuously improving accessibility. Our ongoing efforts include:
            </Text>
            <ContentList>
              <li>Regular accessibility audits and testing</li>
              <li>User feedback integration</li>
              <li>Training on accessibility best practices</li>
              <li>Updates to meet evolving accessibility standards</li>
            </ContentList>
          </SectionCard>

          <SectionCard title="📚 Standards and Guidelines">
            <Text>This accessibility statement is based on:</Text>
            <ContentList>
              <li>
                <ExternalLink href="https://www.w3.org/WAI/WCAG21/quickref/">
                  WCAG 2.1 Level AA
                </ExternalLink>{" "}
                (Web Content Accessibility Guidelines)
              </li>
              <li>
                <ExternalLink href="https://www.ada.gov/">
                  Americans with Disabilities Act (ADA)
                </ExternalLink>{" "}
                requirements
              </li>
              <li>
                <ExternalLink href="https://www.ontario.ca/laws/statute/11a11">
                  Accessibility for Ontarians with Disabilities Act (AODA)
                </ExternalLink>{" "}
                (for Canadian users)
              </li>
              <li>
                <ExternalLink href="https://www.etsi.org/deliver/etsi_en/301500_301599/301549/02.01.02_60/en_301549v020102p.pdf">
                  EN 301 549
                </ExternalLink>{" "}
                (European accessibility standard)
              </li>
            </ContentList>
          </SectionCard>

          <div style={{ marginTop: "var(--spacing-xl)", textAlign: "center" }}>
            <Link href="/" className="btn btn-primary">
              ← Back to Home
            </Link>
          </div>
        </div>
      </div>
    </>
  );
}
