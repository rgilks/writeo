"use client";

import Link from "next/link";

export default function AccessibilityPage() {
  return (
    <>
      <header className="header" lang="en">
        <div className="header-content">
          <div className="logo-group">
            <Link href="/" className="logo" lang="en">
              Writeo
            </Link>
          </div>
          <nav className="header-actions" aria-label="Accessibility actions" lang="en">
            <Link href="/" className="nav-back-link" lang="en">
              <span aria-hidden="true">←</span> Back to Home
            </Link>
          </nav>
        </div>
      </header>

      <div className="container" lang="en">
        <div
          style={{ marginBottom: "var(--spacing-xl)", maxWidth: "800px", margin: "0 auto" }}
          lang="en"
        >
          <h1 className="page-title">Accessibility Statement</h1>
          <p className="page-subtitle">
            Last updated:{" "}
            {new Date().toLocaleDateString("en-US", {
              year: "numeric",
              month: "long",
              day: "numeric",
            })}
          </p>

          <div className="card" style={{ marginTop: "var(--spacing-xl)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              ♿ Our Commitment
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              Writeo is committed to ensuring digital accessibility for people with disabilities. We
              are continually improving the user experience for everyone and applying the relevant
              accessibility standards to achieve these goals.
            </p>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              🎯 Conformance Status
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              The{" "}
              <a
                href="https://www.w3.org/WAI/WCAG21/quickref/?currentsidebar=%23col_overview&levels=aaa"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                Web Content Accessibility Guidelines (WCAG)
              </a>{" "}
              defines requirements for designers and developers to improve accessibility for people
              with disabilities. Writeo aims to conform to WCAG 2.1 Level AA standards.
            </p>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>Current Status:</strong> Writeo is partially conformant with WCAG 2.1 Level
              AA. Partially conformant means that some parts of the content do not fully conform to
              the accessibility standard. We are actively working to improve accessibility across
              all features.
            </p>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              ✅ Accessibility Features
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              Writeo includes the following accessibility features:
            </p>
            <ul
              style={{
                marginLeft: "var(--spacing-lg)",
                marginBottom: "var(--spacing-md)",
                lineHeight: "1.5",
                paddingLeft: "var(--spacing-lg)",
              }}
              lang="en"
            >
              <li lang="en">
                <strong>Semantic HTML:</strong> Proper use of HTML5 semantic elements for better
                screen reader navigation
              </li>
              <li lang="en">
                <strong>ARIA Labels:</strong> Descriptive labels for interactive elements to assist
                screen readers
              </li>
              <li lang="en">
                <strong>Keyboard Navigation:</strong> All interactive elements are accessible via
                keyboard
              </li>
              <li lang="en">
                <strong>Color Contrast:</strong> Text and background colors meet WCAG contrast
                requirements
              </li>
              <li lang="en">
                <strong>Focus Indicators:</strong> Visible focus indicators for keyboard navigation
              </li>
              <li lang="en">
                <strong>Alt Text:</strong> Images include descriptive alt text where appropriate
              </li>
              <li lang="en">
                <strong>Responsive Design:</strong> Works across different screen sizes and devices
              </li>
            </ul>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              ⚠️ Known Limitations
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              Despite our best efforts to ensure accessibility, there may be some limitations:
            </p>
            <ul
              style={{
                marginLeft: "var(--spacing-lg)",
                marginBottom: "var(--spacing-md)",
                lineHeight: "1.5",
                paddingLeft: "var(--spacing-lg)",
              }}
              lang="en"
            >
              <li lang="en">
                Some interactive features may require mouse interaction (we're working on full
                keyboard alternatives)
              </li>
              <li lang="en">
                Error highlighting and heat map visualizations may need additional screen reader
                descriptions
              </li>
              <li lang="en">
                Some third-party services (AI feedback, grammar checking) may have their own
                accessibility limitations
              </li>
            </ul>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              We are committed to addressing these limitations and improving accessibility over
              time.
            </p>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              🔧 Browser and Assistive Technology Support
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              Writeo is designed to work with the following:
            </p>
            <ul
              style={{
                marginLeft: "var(--spacing-lg)",
                marginBottom: "var(--spacing-md)",
                lineHeight: "1.5",
                paddingLeft: "var(--spacing-lg)",
              }}
              lang="en"
            >
              <li lang="en">Modern browsers (Chrome, Firefox, Safari, Edge)</li>
              <li lang="en">Screen readers (NVDA, JAWS, VoiceOver)</li>
              <li lang="en">Keyboard-only navigation</li>
              <li lang="en">Mobile devices and tablets</li>
            </ul>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              📧 Feedback and Contact
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              We welcome your feedback on the accessibility of Writeo. If you encounter
              accessibility barriers or have suggestions for improvement, please contact us:
            </p>
            <ul
              style={{
                marginLeft: "var(--spacing-lg)",
                marginBottom: "var(--spacing-md)",
                lineHeight: "1.5",
                paddingLeft: "var(--spacing-lg)",
              }}
              lang="en"
            >
              <li lang="en">
                Join our{" "}
                <a
                  href="https://discord.gg/9rtwCKp2"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline"
                >
                  Discord server
                </a>{" "}
                and use the support channel
              </li>
              <li lang="en">
                Report accessibility issues or request accommodations through our support channel
              </li>
            </ul>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              We aim to respond to accessibility feedback within 48 hours.
            </p>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              🔄 Ongoing Improvements
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              We are committed to continuously improving accessibility. Our ongoing efforts include:
            </p>
            <ul
              style={{
                marginLeft: "var(--spacing-lg)",
                marginBottom: "var(--spacing-md)",
                lineHeight: "1.5",
                paddingLeft: "var(--spacing-lg)",
              }}
              lang="en"
            >
              <li lang="en">Regular accessibility audits and testing</li>
              <li lang="en">User feedback integration</li>
              <li lang="en">Training on accessibility best practices</li>
              <li lang="en">Updates to meet evolving accessibility standards</li>
            </ul>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              📚 Standards and Guidelines
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              This accessibility statement is based on:
            </p>
            <ul
              style={{
                marginLeft: "var(--spacing-lg)",
                marginBottom: "var(--spacing-md)",
                lineHeight: "1.5",
                paddingLeft: "var(--spacing-lg)",
              }}
              lang="en"
            >
              <li lang="en">
                <a
                  href="https://www.w3.org/WAI/WCAG21/quickref/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline"
                >
                  WCAG 2.1 Level AA
                </a>{" "}
                (Web Content Accessibility Guidelines)
              </li>
              <li lang="en">
                <a
                  href="https://www.ada.gov/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline"
                >
                  Americans with Disabilities Act (ADA)
                </a>{" "}
                requirements
              </li>
              <li lang="en">
                <a
                  href="https://www.ontario.ca/laws/statute/11a11"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline"
                >
                  Accessibility for Ontarians with Disabilities Act (AODA)
                </a>{" "}
                (for Canadian users)
              </li>
              <li lang="en">
                <a
                  href="https://www.etsi.org/deliver/etsi_en/301500_301599/301549/02.01.02_60/en_301549v020102p.pdf"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline"
                >
                  EN 301 549
                </a>{" "}
                (European accessibility standard)
              </li>
            </ul>
          </div>

          <div style={{ marginTop: "var(--spacing-xl)", textAlign: "center" }} lang="en">
            <Link href="/" className="btn btn-primary" lang="en">
              ← Back to Home
            </Link>
          </div>
        </div>
      </div>
    </>
  );
}
