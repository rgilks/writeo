"use client";

import Link from "next/link";

export default function PrivacyPage() {
  return (
    <>
      <header className="header" lang="en">
        <div className="header-content">
          <div className="logo-group">
            <Link href="/" className="logo" lang="en">
              Writeo
            </Link>
          </div>
          <nav className="header-actions" aria-label="Privacy actions" lang="en">
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
          <h1 className="page-title">Privacy & Data Ethics</h1>
          <p className="page-subtitle">How we handle your writing and protect your privacy</p>

          <div className="card" style={{ marginTop: "var(--spacing-xl)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              🔒 Your Privacy Matters
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>Writeo is a study tool.</strong> Your writing isn't shared with other
              learners. Your essays and assessment results are private and only accessible to you.
            </p>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              📝 How We Store Your Writing
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>By default, we don't store your data on our servers.</strong> Your privacy is
              our priority.
            </p>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              When you submit an essay for feedback:
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
                <strong>Default (no server storage):</strong> Your results are stored only in your
                browser's localStorage. This means your data stays on your device and is never sent
                to our servers for storage.
              </li>
              <li lang="en">
                <strong>Optional server storage:</strong> You can choose to enable server storage
                when submitting. If you opt in, your essay and assessment results are stored
                securely on our servers for 90 days, allowing you to access your results from any
                device.
              </li>
              <li lang="en">Your writing is never publicly accessible</li>
              <li lang="en">We don't store your data unless you explicitly opt in</li>
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
              <strong>Note:</strong> Even when server storage is enabled, your data is processed
              only to provide immediate feedback and is automatically deleted after 90 days.
            </p>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              🤖 AI & Third-Party Services
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>We do not use your writing to train or improve our AI models</strong> without
              your explicit consent.
            </p>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              Your essays are used only to:
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
              <li lang="en">Provide you with immediate feedback</li>
              <li lang="en">Track your progress across drafts</li>
              <li lang="en">Show you your writing history</li>
            </ul>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>Third-Party AI Services:</strong> Your text is sent to third-party AI services
              for analysis:
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
                <strong>Groq API</strong> (Llama 3.3 70B model) - Used for generating AI feedback
                and teacher-style comments
              </li>
              <li lang="en">
                <strong>Cloudflare Workers AI</strong> - Used for relevance checking via embeddings
                (to verify your answer addresses the question)
              </li>
              <li lang="en">
                <strong>Modal</strong> - Hosts essay scoring models and LanguageTool for grammar
                checking
              </li>
            </ul>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              These services process your text to provide feedback, but they do not retain your data
              for training purposes. Your text is not shared with other users or publicly
              accessible.
            </p>
            <p
              style={{
                marginBottom: "var(--spacing-md)",
                lineHeight: "1.5",
                fontSize: "16px",
                color: "var(--text-secondary)",
              }}
              lang="en"
            >
              If we ever want to use anonymized writing samples for research or model improvement,
              we will ask for your explicit opt-in consent first.
            </p>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              🔐 Data Access & Sharing
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              Your data is private and secure:
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
              <li lang="en">Only you can access your essays and assessment results</li>
              <li lang="en">Your writing is not shared with other users</li>
              <li lang="en">Your data is not used for advertising or marketing</li>
              <li lang="en">We do not sell or rent your personal information</li>
            </ul>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              🗑️ Data Retention
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>By default, we don't store your data on our servers.</strong> Your results are
              stored only in your browser and can be cleared at any time by clearing your browser
              data.
            </p>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              If you opt in to server storage:
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
                <strong>Assessment results:</strong> Stored for 90 days, then automatically deleted
              </li>
              <li lang="en">
                <strong>Your essays and submissions:</strong> Stored for 90 days, then automatically
                deleted
              </li>
              <li lang="en">
                <strong>Questions and answers:</strong> Stored for 90 days as part of your
                submission history
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
              <strong>Browser storage:</strong> Results stored in your browser (localStorage) remain
              until you clear your browser data or manually delete them. You have full control over
              this data.
            </p>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              📧 Contact & Support
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              If you have questions about your privacy or want to exercise your rights (access,
              deletion, etc.), please contact us. We're committed to being transparent and helpful.
            </p>
            <p
              style={{
                marginBottom: "var(--spacing-md)",
                lineHeight: "1.5",
                fontSize: "16px",
                color: "var(--text-secondary)",
              }}
              lang="en"
            >
              <strong>Note:</strong> Writeo is designed as a learning tool. We prioritize your
              privacy and the security of your writing data.
            </p>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              ✅ Your Rights
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              You have the right to:
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
              <li lang="en">Access your stored writing and assessment data</li>
              <li lang="en">Request deletion of your data</li>
              <li lang="en">Opt out of any research or model improvement programs</li>
              <li lang="en">Export your writing and progress data</li>
            </ul>
          </div>

          <div
            className="card"
            style={{ marginTop: "var(--spacing-lg)", backgroundColor: "rgba(102, 126, 234, 0.1)" }}
            lang="en"
          >
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              🛡️ Our Commitment
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              Writeo is designed with privacy and ethics in mind. We follow the principles of:
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
                <strong>Transparency:</strong> We're clear about how we use your data
              </li>
              <li lang="en">
                <strong>Fairness:</strong> Automated decisions are advisory, not final
              </li>
              <li lang="en">
                <strong>Conservatism:</strong> We prioritize precision over coverage in feedback
              </li>
              <li lang="en">
                <strong>User control:</strong> You stay in control of your data and can request
                deletion at any time
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
