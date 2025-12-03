"use client";

import Link from "next/link";
import { Logo } from "@/app/components/Logo";
import { SectionCard, ContentList, Text, ExternalLink } from "@/app/components/ContentComponents";

export default function PrivacyPage() {
  return (
    <>
      <header className="header" lang="en">
        <div className="header-content">
          <div className="logo-group">
            <Logo />
          </div>
          <nav className="header-actions" aria-label="Privacy actions">
            <Link href="/" className="nav-back-link">
              <span aria-hidden="true">←</span> Back to Home
            </Link>
          </nav>
        </div>
      </header>

      <div className="container" lang="en">
        <div style={{ maxWidth: "800px", margin: "0 auto" }}>
          <h1 className="page-title">Privacy & Data Ethics</h1>
          <p className="page-subtitle">How we handle your writing and protect your privacy</p>

          <SectionCard title="🔒 Your Privacy Matters" isFirst>
            <Text>
              <strong>Writeo is a study tool.</strong> Your writing isn't shared with other
              learners. Your essays and assessment results are private and only accessible to you.
            </Text>
          </SectionCard>

          <SectionCard title="👶 Children's Privacy & Age Guidelines">
            <Text>
              <strong>Writeo is designed to be safe for users of all ages.</strong> Because Writeo
              uses an opt-in server storage model where no data is stored on servers by default,
              Writeo doesn't collect personal information from children. This means COPPA
              (Children's Online Privacy Protection Act) requirements don't apply to Writeo's
              default usage.
            </Text>
            <Text>
              <strong>For children under 13:</strong> Writeo can be used safely by children under
              parental guidance. By default, all data stays on the child's device and is never sent
              to our servers. We recommend parents:
            </Text>
            <ContentList>
              <li>Review the privacy policy with their child</li>
              <li>Understand that server storage is opt-in only</li>
              <li>Monitor their child's use of the service</li>
              <li>Help their child understand how to use the tool safely</li>
            </ContentList>
            <Text>
              <strong>For users 13-16 in the European Union:</strong> If you opt in to server
              storage, parental consent may be required depending on your jurisdiction. By default
              (no server storage), no consent is needed.
            </Text>
            <Text note>
              <strong>Note:</strong> Writeo is an educational tool designed to help learners improve
              their writing. We encourage safe, responsible use by learners of all ages, with
              appropriate parental guidance for younger users.
            </Text>
          </SectionCard>

          <SectionCard title="📝 How We Store Your Writing">
            <Text>
              <strong>By default, Writeo doesn't store your data on servers.</strong> Your privacy
              is a priority.
            </Text>
            <Text>When you submit an essay for feedback:</Text>
            <ContentList>
              <li>
                <strong>Default (no server storage):</strong> Your results are stored only in your
                browser's localStorage. This means your data stays on your device and is never sent
                to our servers for storage.
              </li>
              <li>
                <strong>Optional server storage:</strong> You can choose to enable server storage
                when submitting. If you opt in, your essay and assessment results are stored
                securely on our servers for 90 days, allowing you to access your results from any
                device.
              </li>
              <li>Your writing is never publicly accessible</li>
              <li>We don't store your data unless you explicitly opt in</li>
            </ContentList>
            <Text note>
              <strong>Note:</strong> Even when server storage is enabled, your data is processed
              only to provide immediate feedback and is automatically deleted after 90 days.
            </Text>
          </SectionCard>

          <SectionCard title="🍪 Cookies & Browser Storage">
            <Text>
              <strong>Writeo does not use HTTP cookies.</strong> We do not set, read, or store any
              cookies on your device. This means you do not need to accept or manage cookie
              preferences when using Writeo. There is no cookie consent banner because there are no
              cookies to consent to.
            </Text>
            <Text>
              While we don't use cookies, Writeo does use browser storage APIs to provide
              functionality:
            </Text>
            <ContentList>
              <li>
                <strong>localStorage:</strong> Used to store your essay results and progress data
                locally on your device. This data never leaves your device and is not sent to our
                servers unless you explicitly opt in to server storage.
              </li>
              <li>
                <strong>sessionStorage:</strong> Used temporarily during your session to maintain
                application state. This data is cleared when you close your browser tab.
              </li>
            </ContentList>
            <Text>
              These storage mechanisms are different from cookies and do not require consent under
              GDPR, CCPA, or other privacy regulations. They are essential for the service to
              function and allow you to save your essay results locally, track your progress across
              sessions, maintain your draft history, and remember your preferences.
            </Text>
            <Text>
              <strong>Your Control:</strong> You have full control over browser storage. You can
              clear localStorage and sessionStorage at any time through your browser settings.
              Clearing browser data will remove all locally stored results and progress. You can use
              browser privacy/incognito modes if you don't want any data stored.
            </Text>
            <Text note>
              <strong>Note:</strong> If you clear browser storage, you will lose access to locally
              stored results. If you've opted in to server storage, your results will still be
              accessible via the submission ID for 90 days.
            </Text>
          </SectionCard>

          <SectionCard title="🤖 AI & Third-Party Services">
            <Text>
              <strong>Writeo does not use your writing to train or improve AI models</strong>{" "}
              without your explicit consent.
            </Text>
            <Text>Your essays are used only to:</Text>
            <ContentList>
              <li>Provide you with immediate feedback</li>
              <li>Track your progress across drafts</li>
              <li>Show you your writing history</li>
            </ContentList>
            <Text>
              <strong>Third-Party AI Services & Data Processors:</strong> Your text is sent to
              third-party services for analysis. These services act as data processors and are bound
              by data processing agreements:
            </Text>
            <ContentList>
              <li>
                <strong>OpenAI API</strong> (GPT-4o-mini model) - Used for generating AI feedback
                and teacher-style comments. OpenAI processes your text but does not retain it for
                training purposes. See OpenAI's{" "}
                <ExternalLink href="https://openai.com/policies/privacy-policy">
                  privacy policy
                </ExternalLink>{" "}
                for details.
              </li>
              <li>
                <strong>Groq API</strong> (Llama 3.3 70B model) - Used for generating AI feedback
                and teacher-style comments when Groq is selected as the provider. Groq processes
                your text but does not retain it for training purposes. See Groq's{" "}
                <ExternalLink href="https://groq.com/legal/privacy">privacy policy</ExternalLink>{" "}
                for details.
              </li>
              <li>
                <strong>Cloudflare Workers AI</strong> - Used for relevance checking via embeddings
                (to verify your answer addresses the question). Cloudflare processes data in
                accordance with their privacy policy and terms.
              </li>
              <li>
                <strong>Modal</strong> - Hosts essay scoring models and LanguageTool for grammar
                checking. Modal processes your text for scoring but does not retain it.
              </li>
            </ContentList>
            <Text>
              <strong>Data Controller vs. Data Processor:</strong> Writeo acts as the data
              controller, and these third-party services act as data processors. All processors are
              contractually bound to:
            </Text>
            <ContentList>
              <li>Process data only for the purpose of providing feedback</li>
              <li>Not retain your data for training or other purposes</li>
              <li>Maintain appropriate security measures</li>
              <li>Comply with applicable data protection laws</li>
            </ContentList>
            <Text>
              These services process your text to provide feedback, but they do not retain your data
              for training purposes. Your text is not shared with other users or publicly
              accessible.
            </Text>
            <Text note>
              If we ever want to use anonymized writing samples for research or model improvement,
              we will ask for your explicit opt-in consent first.
            </Text>
          </SectionCard>

          <SectionCard title="🔐 Data Access & Sharing">
            <Text>Your data is private and secure:</Text>
            <ContentList>
              <li>Only you can access your essays and assessment results</li>
              <li>Your writing is not shared with other users</li>
              <li>Your data is not used for advertising or marketing</li>
              <li>We do not sell or rent your personal information</li>
            </ContentList>
          </SectionCard>

          <SectionCard title="🚫 Do Not Sell My Personal Information (CCPA)">
            <Text>
              <strong>Writeo does not sell your personal information.</strong> We do not sell, rent,
              or trade your personal data to third parties for monetary or other valuable
              consideration.
            </Text>
            <Text>
              Under the California Consumer Privacy Act (CCPA), California residents have the right
              to opt out of the sale of personal information. Since Writeo does not sell personal
              information, no opt-out is necessary. However, if you have concerns or questions,
              please contact the project maintainer via{" "}
              <ExternalLink href="https://discord.gg/YxuFAXWuzw">Discord server</ExternalLink>.
            </Text>
          </SectionCard>

          <SectionCard title="⚖️ Non-Discrimination (CCPA)">
            <Text>
              <strong>We will not discriminate against you</strong> for exercising your privacy
              rights under the California Consumer Privacy Act (CCPA) or any other applicable
              privacy laws.
            </Text>
            <Text>This means:</Text>
            <ContentList>
              <li>We will not deny you goods or services for exercising your privacy rights</li>
              <li>
                We will not charge you different prices or rates for exercising your privacy rights
              </li>
              <li>
                We will not provide you a different level or quality of services for exercising your
                privacy rights
              </li>
              <li>
                We will not suggest that you may receive different treatment for exercising your
                privacy rights
              </li>
            </ContentList>
            <Text>
              Your service quality and access will remain the same regardless of whether you choose
              to exercise your privacy rights.
            </Text>
          </SectionCard>

          <SectionCard title="🚨 Data Breach Notification">
            <Text>
              In the unlikely event of a data breach that affects your personal information, we are
              committed to:
            </Text>
            <ContentList>
              <li>
                <strong>Immediate assessment:</strong> Investigating and containing the breach as
                quickly as possible
              </li>
              <li>
                <strong>Notification:</strong> Notifying affected users within 72 hours (as required
                by GDPR) or as soon as reasonably possible
              </li>
              <li>
                <strong>Transparency:</strong> Providing clear information about what happened, what
                data was affected, and what steps we're taking
              </li>
              <li>
                <strong>Remediation:</strong> Taking steps to prevent future breaches and mitigate
                any harm
              </li>
            </ContentList>
            <Text>
              <strong>Note:</strong> Because Writeo uses an opt-in server storage model, most users
              don't have data stored on servers. This significantly reduces the risk of data
              breaches. However, if you've opted in to server storage and a breach occurs, you will
              be notified through the contact information you've provided or via the{" "}
              <ExternalLink href="https://discord.gg/YxuFAXWuzw">Discord server</ExternalLink>.
            </Text>
            <Text note>
              If you believe your data may have been compromised, please contact the project
              maintainer immediately via the Discord server.
            </Text>
          </SectionCard>

          <SectionCard title="🗑️ Data Retention">
            <Text>
              <strong>By default, Writeo doesn't store your data on servers.</strong> Your results
              are stored only in your browser and can be cleared at any time by clearing your
              browser data.
            </Text>
            <Text>If you opt in to server storage:</Text>
            <ContentList>
              <li>
                <strong>Assessment results:</strong> Stored for 90 days, then automatically deleted
              </li>
              <li>
                <strong>Your essays and submissions:</strong> Stored for 90 days, then automatically
                deleted
              </li>
              <li>
                <strong>Questions and answers:</strong> Stored for 90 days as part of your
                submission history
              </li>
            </ContentList>
            <Text note>
              <strong>Browser storage:</strong> Results stored in your browser (localStorage) remain
              until you clear your browser data or manually delete them. You have full control over
              this data.
            </Text>
          </SectionCard>

          <SectionCard title="📧 Contact & Support">
            <Text>
              If you have questions about your privacy or want to exercise your rights (access,
              deletion, etc.), please contact the project maintainer. I'm committed to being
              transparent and helpful.
            </Text>
            <Text>
              <strong>Support:</strong> Join the{" "}
              <ExternalLink href="https://discord.gg/YxuFAXWuzw">Discord server</ExternalLink> for
              support and questions.
            </Text>
            <Text note>
              <strong>Privacy Requests:</strong> For formal privacy requests (data access, deletion,
              etc.), please use the support channel in the Discord server or contact the project
              maintainer directly through Discord.
            </Text>
            <Text note>
              <strong>Note:</strong> Writeo is designed as a learning tool. We prioritize your
              privacy and the security of your writing data.
            </Text>
          </SectionCard>

          <SectionCard title="✅ Your Rights">
            <Text>You have the right to:</Text>
            <ContentList>
              <li>Access your stored writing and assessment data</li>
              <li>Request deletion of your data</li>
              <li>Opt out of any research or model improvement programs</li>
              <li>Export your writing and progress data</li>
            </ContentList>
          </SectionCard>

          <SectionCard title="🛡️ Our Commitment" highlight>
            <Text>
              Writeo is designed with privacy and ethics in mind. We follow the principles of:
            </Text>
            <ContentList>
              <li>
                <strong>Transparency:</strong> We're clear about how we use your data
              </li>
              <li>
                <strong>Fairness:</strong> Automated decisions are advisory, not final
              </li>
              <li>
                <strong>Conservatism:</strong> We prioritize precision over coverage in feedback
              </li>
              <li>
                <strong>User control:</strong> You stay in control of your data and can request
                deletion at any time
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
