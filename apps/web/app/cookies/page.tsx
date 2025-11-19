"use client";

import Link from "next/link";

export default function CookiesPage() {
  return (
    <>
      <header className="header" lang="en">
        <div className="header-content">
          <div className="logo-group">
            <Link href="/" className="logo" lang="en">
              Writeo
            </Link>
          </div>
          <nav className="header-actions" aria-label="Cookies actions" lang="en">
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
          <h1 className="page-title">Cookie Policy</h1>
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
              🍪 Our Cookie Policy
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>Writeo does not use HTTP cookies.</strong> We do not set, read, or store any
              cookies on your device.
            </p>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              This means you do not need to accept or manage cookie preferences when using Writeo.
              There is no cookie consent banner because there are no cookies to consent to.
            </p>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              💾 Browser Storage (Not Cookies)
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              While we don't use cookies, Writeo does use browser storage APIs to provide
              functionality:
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
                <strong>localStorage:</strong> Used to store your essay results and progress data
                locally on your device. This data never leaves your device and is not sent to our
                servers unless you explicitly opt in to server storage.
              </li>
              <li lang="en">
                <strong>sessionStorage:</strong> Used temporarily during your session to maintain
                application state. This data is cleared when you close your browser tab.
              </li>
            </ul>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              These storage mechanisms are different from cookies and do not require consent under
              GDPR, CCPA, or other privacy regulations. They are essential for the service to
              function and allow you to:
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
              <li lang="en">Save your essay results locally</li>
              <li lang="en">Track your progress across sessions</li>
              <li lang="en">Maintain your draft history</li>
              <li lang="en">Remember your preferences</li>
            </ul>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              🔒 Your Control
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              You have full control over browser storage:
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
                You can clear localStorage and sessionStorage at any time through your browser
                settings
              </li>
              <li lang="en">
                Clearing browser data will remove all locally stored results and progress
              </li>
              <li lang="en">
                You can use browser privacy/incognito modes if you don't want any data stored
              </li>
            </ul>
            <p
              style={{
                marginBottom: "var(--spacing-md)",
                lineHeight: "1.5",
                fontSize: "16px",
                color: "var(--text-secondary)",
              }}
              lang="en"
            >
              <strong>Note:</strong> If you clear browser storage, you will lose access to locally
              stored results. If you've opted in to server storage, your results will still be
              accessible via the submission ID for 90 days.
            </p>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              📧 Questions?
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              If you have questions about our cookie policy or browser storage practices, please
              contact us through our{" "}
              <a
                href="https://discord.gg/9rtwCKp2"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                Discord server
              </a>
              . For more information about how we handle your data, please see our{" "}
              <Link href="/privacy" className="text-blue-600 hover:underline">
                Privacy Policy
              </Link>
              .
            </p>
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
