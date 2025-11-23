# Legal Compliance Guide

**Important:** Writeo uses an **opt-in server storage model** - by default, no data is stored on servers. Results are stored only in the user's browser (localStorage). This significantly reduces legal compliance requirements, as most users don't have data stored on servers.

---

## Implemented Compliance Features

### Terms of Service
- Comprehensive Terms of Service page (`apps/web/app/terms/page.tsx`)
- Age guidelines (all ages welcome, parental guidance recommended for under 13)
- API rate limiting terms disclosed
- User responsibilities, IP rights, donations policy
- Linked in footer and privacy policy

### Privacy Policy
- Donations/payment processing section
- Third-party service disclosures with DPA information
- Contact information (Discord support: https://discord.gg/9rtwCKp2)
- CCPA "Do Not Sell" section (no data sold)
- CCPA non-discrimination policy
- Data breach notification procedures
- Children's privacy section
- Linked to Terms of Service

### Cookie Policy
- Comprehensive Cookie Policy page (`apps/web/app/cookies/page.tsx`)
- States no HTTP cookies are used
- Explains localStorage and sessionStorage usage
- Clarifies browser storage doesn't require consent
- Linked from footer

### Accessibility Statement
- Comprehensive accessibility statement page (`apps/web/app/accessibility/page.tsx`)
- WCAG 2.1 Level AA conformance status (partially conformant, actively improving)
- Lists accessibility features
- Documents known limitations
- Linked from footer

### License
- Apache 2.0 license implemented
- LICENSE file created
- All package.json files updated
- README updated with license notice

### Age Restrictions & COPPA
- No age restrictions (all ages welcome)
- COPPA doesn't apply due to opt-in storage model (no data collection by default)
- Age guidelines added (parental guidance recommended for under 13)
- EU users 13-16 may need parental consent only if opting into server storage

---

## Pending Items

### Data Deletion API
**Priority:** MEDIUM (Lower priority due to opt-in storage model)

Required for users who opt into server storage:
- `DELETE /text/submissions/{id}` endpoint
- Delete from R2 (essays, submissions)
- Delete from KV (results)
- User-facing deletion request form
- Verification process

### Data Export API
**Priority:** MEDIUM (Lower priority due to opt-in storage model)

Required for users who opt into server storage:
- `GET /text/submissions/{id}/export` endpoint
- Export all user data (JSON format)
- User-facing export request page

### Third-Party Service Terms Verification
**Priority:** HIGH

- Review OpenAI Terms and Privacy Policy
- Review Groq Terms and Privacy Policy
- Review Cloudflare Terms and Privacy Policy
- Review LanguageTool Terms
- Verify HuggingFace model licenses
- Create `THIRD_PARTY_LICENSES.md` documenting all licenses

### Security Documentation
**Priority:** MEDIUM

- Document security measures (encryption, access controls)
- Create breach response plan
- Add security section to privacy policy
- Create `SECURITY.md`

---

## Compliance Status

### GDPR Compliance
- ✅ Privacy policy with required sections
- ✅ Legal basis for processing stated (opt-in consent for server storage)
- ✅ Data subject rights (access via localStorage by default, deletion via browser, export via localStorage)
- ✅ Data breach notification procedures documented
- ✅ Cookie consent (NOT REQUIRED - no cookies used, only localStorage/sessionStorage)
- ✅ Age restrictions (NOT REQUIRED - no data collection by default)

### CCPA Compliance (if serving California users)
- ✅ Privacy policy with disclosures
- ✅ "Do Not Sell" section (no data sold)
- ✅ Non-discrimination policy
- ⚠️ Data deletion API (pending - only needed for opt-in users)
- ⚠️ Data export API (pending - only needed for opt-in users)

### COPPA Compliance
- ✅ NOT APPLICABLE - No data collection by default (opt-in storage model)
- ✅ Age guidelines provided (all ages welcome)

### General Legal
- ✅ Terms of Service
- ✅ Privacy Policy
- ✅ Cookie Policy
- ✅ Accessibility Statement
- ✅ Contact information (Discord support)
- ⚠️ Third-party licenses verification (pending)
- ⚠️ Model licenses verification (pending)

---

## Resources

### Legal Resources
- GDPR: https://gdpr.eu/
- CCPA: https://oag.ca.gov/privacy/ccpa
- COPPA: https://www.ftc.gov/tips-advice/business-center/privacy-and-security/children's-privacy
- ePrivacy Directive: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32002L0058

### Third-Party Terms
- OpenAI: https://openai.com/policies/terms-of-use
- Groq: https://groq.com/legal/terms
- Cloudflare: https://www.cloudflare.com/terms/
- LanguageTool: https://languagetool.org/legal/
- HuggingFace: https://huggingface.co/terms

### Model Licenses
- KevSun/Engessay_grading_ML: https://huggingface.co/KevSun/Engessay_grading_ML
  - Citation: Sun, K., & Wang, R. (2024). Automatic Essay Multi-dimensional Scoring with Fine-tuning and Multiple Regression. _ArXiv_. https://arxiv.org/abs/2406.01198
- Michau96/distilbert-base-uncased-essay_scoring: https://huggingface.co/Michau96/distilbert-base-uncased-essay_scoring
