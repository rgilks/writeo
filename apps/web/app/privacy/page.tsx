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
              👶 Children's Privacy & Age Guidelines
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>Writeo is designed to be safe for users of all ages.</strong> Because Writeo
              uses an opt-in server storage model where no data is stored on servers by default,
              Writeo doesn't collect personal information from children. This means COPPA
              (Children's Online Privacy Protection Act) requirements don't apply to Writeo's
              default usage.
            </p>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>For children under 13:</strong> Writeo can be used safely by children under
              parental guidance. By default, all data stays on the child's device and is never sent
              to our servers. We recommend parents:
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
              <li lang="en">Review the privacy policy with their child</li>
              <li lang="en">Understand that server storage is opt-in only</li>
              <li lang="en">Monitor their child's use of the service</li>
              <li lang="en">Help their child understand how to use the tool safely</li>
            </ul>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>For users 13-16 in the European Union:</strong> If you opt in to server
              storage, parental consent may be required depending on your jurisdiction. By default
              (no server storage), no consent is needed.
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
              <strong>Note:</strong> Writeo is an educational tool designed to help learners improve
              their writing. We encourage safe, responsible use by learners of all ages, with
              appropriate parental guidance for younger users.
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
              <strong>By default, Writeo doesn't store your data on servers.</strong> Your privacy
              is a priority.
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
              🍪 Cookies & Browser Storage
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>Writeo does not use HTTP cookies.</strong> We do not set, read, or store any
              cookies on your device. This means you do not need to accept or manage cookie
              preferences when using Writeo. There is no cookie consent banner because there are no
              cookies to consent to.
            </p>
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
              function and allow you to save your essay results locally, track your progress across
              sessions, maintain your draft history, and remember your preferences.
            </p>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>Your Control:</strong> You have full control over browser storage. You can
              clear localStorage and sessionStorage at any time through your browser settings.
              Clearing browser data will remove all locally stored results and progress. You can use
              browser privacy/incognito modes if you don't want any data stored.
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
              🤖 AI & Third-Party Services
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>Writeo does not use your writing to train or improve AI models</strong>{" "}
              without your explicit consent.
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
              <strong>Third-Party AI Services & Data Processors:</strong> Your text is sent to
              third-party services for analysis. These services act as data processors and are bound
              by data processing agreements:
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
                and teacher-style comments. Groq processes your text but does not retain it for
                training purposes.
              </li>
              <li lang="en">
                <strong>Cloudflare Workers AI</strong> - Used for relevance checking via embeddings
                (to verify your answer addresses the question). Cloudflare processes data in
                accordance with their privacy policy and terms.
              </li>
              <li lang="en">
                <strong>Modal</strong> - Hosts essay scoring models and LanguageTool for grammar
                checking. Modal processes your text for scoring but does not retain it.
              </li>
            </ul>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>Data Controller vs. Data Processor:</strong> Writeo acts as the data
              controller, and these third-party services act as data processors. All processors are
              contractually bound to:
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
              <li lang="en">Process data only for the purpose of providing feedback</li>
              <li lang="en">Not retain your data for training or other purposes</li>
              <li lang="en">Maintain appropriate security measures</li>
              <li lang="en">Comply with applicable data protection laws</li>
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
              🚫 Do Not Sell My Personal Information (CCPA)
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>Writeo does not sell your personal information.</strong> We do not sell, rent,
              or trade your personal data to third parties for monetary or other valuable
              consideration.
            </p>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              Under the California Consumer Privacy Act (CCPA), California residents have the right
              to opt out of the sale of personal information. Since Writeo does not sell personal
              information, no opt-out is necessary. However, if you have concerns or questions,
              please contact the project maintainer via{" "}
              <a
                href="https://discord.gg/9rtwCKp2"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                Discord server
              </a>
              .
            </p>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              ⚖️ Non-Discrimination (CCPA)
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>We will not discriminate against you</strong> for exercising your privacy
              rights under the California Consumer Privacy Act (CCPA) or any other applicable
              privacy laws.
            </p>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              This means:
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
                We will not deny you goods or services for exercising your privacy rights
              </li>
              <li lang="en">
                We will not charge you different prices or rates for exercising your privacy rights
              </li>
              <li lang="en">
                We will not provide you a different level or quality of services for exercising your
                privacy rights
              </li>
              <li lang="en">
                We will not suggest that you may receive different treatment for exercising your
                privacy rights
              </li>
            </ul>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              Your service quality and access will remain the same regardless of whether you choose
              to exercise your privacy rights.
            </p>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              🚨 Data Breach Notification
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              In the unlikely event of a data breach that affects your personal information, we are
              committed to:
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
                <strong>Immediate assessment:</strong> Investigating and containing the breach as
                quickly as possible
              </li>
              <li lang="en">
                <strong>Notification:</strong> Notifying affected users within 72 hours (as required
                by GDPR) or as soon as reasonably possible
              </li>
              <li lang="en">
                <strong>Transparency:</strong> Providing clear information about what happened, what
                data was affected, and what steps we're taking
              </li>
              <li lang="en">
                <strong>Remediation:</strong> Taking steps to prevent future breaches and mitigate
                any harm
              </li>
            </ul>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>Note:</strong> Because Writeo uses an opt-in server storage model, most users
              don't have data stored on servers. This significantly reduces the risk of data
              breaches. However, if you've opted in to server storage and a breach occurs, you will
              be notified through the contact information you've provided or via the{" "}
              <a
                href="https://discord.gg/9rtwCKp2"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                Discord server
              </a>
              .
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
              If you believe your data may have been compromised, please contact the project
              maintainer immediately via the Discord server.
            </p>
          </div>

          <div className="card" style={{ marginTop: "var(--spacing-lg)" }} lang="en">
            <h2
              style={{ fontSize: "24px", marginBottom: "var(--spacing-md)", fontWeight: 600 }}
              lang="en"
            >
              🗑️ Data Retention
            </h2>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>By default, Writeo doesn't store your data on servers.</strong> Your results
              are stored only in your browser and can be cleared at any time by clearing your
              browser data.
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
              deletion, etc.), please contact the project maintainer. I'm committed to being
              transparent and helpful.
            </p>
            <p style={{ marginBottom: "var(--spacing-md)", lineHeight: "1.5" }} lang="en">
              <strong>Support:</strong> Join the{" "}
              <a
                href="https://discord.gg/9rtwCKp2"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                Discord server
              </a>{" "}
              for support and questions.
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
              <strong>Privacy Requests:</strong> For formal privacy requests (data access, deletion,
              etc.), please use the support channel in the Discord server or contact the project
              maintainer directly through Discord.
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
