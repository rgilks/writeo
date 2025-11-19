# Legal Compliance Guide

**Last Updated:** January 2025  
**Status:** ⚠️ **IN PROGRESS** - Some items completed, others pending  
**Last Review:** January 2025

---

## Executive Summary

This document tracks legal compliance requirements for Writeo, including GDPR, CCPA, COPPA, and general legal requirements. Some items have been completed (Terms of Service, Privacy Policy updates), while others remain pending.

**Important:** Writeo uses an **opt-in server storage model** - by default, no data is stored on servers. Results are stored only in the user's browser (localStorage). This significantly reduces legal compliance requirements, as most users don't have data stored on servers.

---

## ✅ Completed Items

### 1. Terms of Service ✅

- ✅ Created comprehensive Terms of Service page (`apps/web/app/terms/page.tsx`)
- ✅ Age guidelines added (all ages welcome, parental guidance recommended for under 13)
- ✅ API rate limiting terms disclosed
- ✅ Includes user responsibilities, IP rights, donations policy
- ✅ Linked in footer and privacy policy

### 2. Privacy Policy Enhancements ✅

- ✅ Added donations/payment processing section
- ✅ Added third-party service disclosures
- ⚠️ **ISSUE FOUND:** Contact information section exists but no actual contact details provided (email, address, etc.)
- ✅ Linked to Terms of Service

### 3. License ✅

- ✅ Apache 2.0 license implemented
- ✅ LICENSE file created
- ✅ All package.json files updated
- ✅ README updated with license notice

### 4. Donations/Ko-fi Integration ✅

- ✅ Ko-fi button added to footer (all pages)
- ✅ Privacy policy updated with payment processing disclosure
- ✅ Terms of Service includes donation terms

---

## ⚠️ Pending Items

### 1. Age Restrictions & COPPA Compliance ✅ **RESOLVED**

**Priority:** NOT REQUIRED  
**Status:** ✅ COMPLETED - Age restrictions removed, all ages welcome

**Analysis:**

- ✅ Writeo uses opt-in server storage - NO data collected by default
- ✅ COPPA only applies when collecting personal information from children under 13
- ✅ Since no data is collected by default, COPPA doesn't apply
- ✅ Age guidelines added (all ages welcome, parental guidance recommended)
- ✅ No age verification component needed (no data collection = no COPPA requirement)

**Resolution:**

- ✅ Updated Terms of Service to welcome all ages with parental guidance for under 13
- ✅ Updated Privacy Policy with children's privacy section
- ✅ Clarified that COPPA doesn't apply due to opt-in storage model
- ✅ Noted that EU users 13-16 may need parental consent only if opting into server storage

**Files Updated:**

- ✅ `apps/web/app/terms/page.tsx` - Age guidelines added (no restrictions)
- ✅ `apps/web/app/privacy/page.tsx` - Children's privacy section added

---

### 2. Cookie Consent Banner

**Priority:** NOT REQUIRED  
**Status:** ✅ VERIFIED - NOT NEEDED

**Analysis:**

- ✅ Application does NOT use HTTP cookies
- ✅ Only uses `localStorage` and `sessionStorage` (browser storage APIs)
- ✅ These do NOT require cookie consent under GDPR/CCPA
- ✅ Privacy policy already mentions localStorage/sessionStorage usage

**Conclusion**: Cookie consent banner is NOT required. localStorage/sessionStorage usage is already disclosed in privacy policy.

---

### 3. Data Deletion API

**Priority:** MEDIUM (Lower priority due to opt-in storage model)  
**Status:** ⚠️ PENDING

**Note:** With opt-in server storage, most users don't store data on servers. This reduces the urgency, but deletion API is still needed for users who opt in.

**Required:**

- [ ] Create `DELETE /text/submissions/{id}` endpoint
- [ ] Delete from R2 (essays, submissions) - only for opt-in submissions
- [ ] Delete from KV (results) - only for opt-in submissions
- [ ] Create user-facing deletion request form
- [ ] Implement verification process

**Files to Create:**

- `apps/api-worker/src/routes/deletion.ts`
- `apps/web/app/account/delete/page.tsx`

---

### 4. Data Export API

**Priority:** MEDIUM (Lower priority due to opt-in storage model)  
**Status:** ⚠️ PENDING

**Note:** With opt-in server storage, most users don't store data on servers. Browser localStorage provides immediate access. Export API is still useful for opt-in users.

**Required:**

- [ ] Create `GET /text/submissions/{id}/export` endpoint
- [ ] Export all user data (JSON format) - only for opt-in submissions
- [ ] Create user-facing export request page

**Files to Create:**

- `apps/api-worker/src/routes/export.ts`
- `apps/web/app/account/export/page.tsx`

---

### 5. Verify Third-Party Service Terms

**Priority:** HIGH  
**Status:** ⚠️ PENDING

**Required:**

- [ ] Review Groq Terms: https://groq.com/legal/terms
- [ ] Review Groq Privacy: https://groq.com/legal/privacy
- [ ] Review Cloudflare Terms: https://www.cloudflare.com/terms/
- [ ] Review Cloudflare Privacy: https://www.cloudflare.com/privacy/
- [ ] Verify HuggingFace model licenses
- [ ] Create `THIRD_PARTY_LICENSES.md` documenting all licenses

---

### 6. Security Documentation

**Priority:** MEDIUM  
**Status:** ⚠️ PENDING

**Required:**

- [ ] Document security measures (encryption, access controls)
- [ ] Create breach response plan
- [ ] Add security section to privacy policy
- [ ] Create `SECURITY.md`

---

### 7. Missing Contact Information ✅ **RESOLVED**

**Priority:** HIGH  
**Status:** ✅ COMPLETED

**Issue:** Privacy policy and Terms of Service reference "contact us" but provide no actual contact information (email, mailing address, phone, etc.). This violates GDPR Article 13(1)(a) and CCPA requirements.

**Resolution:**

- ✅ Discord support link added to privacy policy (https://discord.gg/9rtwCKp2)
- ✅ Discord support link added to terms of service (https://discord.gg/9rtwCKp2)
- ✅ Discord support link added to footer (https://discord.gg/9rtwCKp2)

**Note:** Discord is acceptable for support and satisfies legal requirements. For formal privacy/data requests, users can contact through Discord. If needed in the future, consider also providing:

- An email address (optional enhancement for formal requests)
- A contact form (optional alternative)

**Files Updated:**

- ✅ `apps/web/app/privacy/page.tsx` - Discord link added
- ✅ `apps/web/app/terms/page.tsx` - Discord link added
- ✅ `apps/web/app/components/Footer.tsx` - Discord link added

---

### 8. Missing Age Restriction Statement ✅ **RESOLVED**

**Priority:** HIGH  
**Status:** ✅ COMPLETED

**Issue:** Terms of Service mention "age restrictions" but don't actually state what the minimum age is. This creates legal ambiguity.

**Resolution:**

- ✅ Added explicit age restriction section to Terms of Service (13+ minimum, 16+ for EU or parental consent)
- ✅ Added age restriction statement to Privacy Policy
- ✅ Noted opt-in storage model reduces compliance requirements
- ✅ Included jurisdiction-specific requirements (EU: 16+, US: 13+)

**Files Updated:**

- ✅ `apps/web/app/terms/page.tsx` - Age restriction section added
- ✅ `apps/web/app/privacy/page.tsx` - Age restriction statement added

---

### 9. API Rate Limiting Terms Not Disclosed ✅ **RESOLVED**

**Priority:** MEDIUM  
**Status:** ✅ COMPLETED

**Issue:** API implements rate limiting (10 submissions/min, 30 general requests/min, 60 results requests/min) but these limits are not disclosed in Terms of Service. Users may be surprised by rate limit errors.

**Resolution:**

- ✅ Added comprehensive API usage and rate limiting section to Terms of Service
- ✅ Documented all rate limits (submissions: 10/min, general: 30/min, results: 60/min)
- ✅ Explained 429 error behavior and reset timing
- ✅ Noted that limits are per IP address
- ✅ Provided contact method for higher limits

**Files Updated:**

- ✅ `apps/web/app/terms/page.tsx` - API usage and rate limiting section added
- ⚠️ `docs/openapi.yaml` - Still pending (can be added if API docs are needed)

---

### 10. Missing Cookie Policy ✅ **RESOLVED**

**Priority:** LOW  
**Status:** ✅ COMPLETED

**Issue:** While cookies are not used, a Cookie Policy explaining this is a best practice and may be required in some jurisdictions. Privacy policy mentions localStorage/sessionStorage but doesn't have a dedicated cookie policy.

**Resolution:**

- ✅ Created comprehensive Cookie Policy page
- ✅ Clearly stated that no HTTP cookies are used
- ✅ Explained localStorage and sessionStorage usage
- ✅ Clarified that browser storage doesn't require consent
- ✅ Explained user control over browser storage
- ✅ Linked from footer

**Files Created:**

- ✅ `apps/web/app/cookies/page.tsx` - Full cookie policy

**Files Updated:**

- ✅ `apps/web/app/components/Footer.tsx` - Added cookies link

---

### 11. Missing Accessibility Statement ✅ **RESOLVED**

**Priority:** MEDIUM  
**Status:** ✅ COMPLETED

**Issue:** No accessibility statement or WCAG compliance information. Required for public-facing educational services in many jurisdictions (ADA in US, AODA in Canada, EN 301 549 in EU).

**Resolution:**

- ✅ Created comprehensive accessibility statement page
- ✅ Documented WCAG 2.1 Level AA conformance status (partially conformant, actively improving)
- ✅ Listed accessibility features (semantic HTML, ARIA labels, keyboard navigation, etc.)
- ✅ Documented known limitations and ongoing improvements
- ✅ Provided contact method for accessibility concerns (Discord support)
- ✅ Linked from footer
- ✅ Referenced relevant standards (WCAG, ADA, AODA, EN 301 549)

**Files Created:**

- ✅ `apps/web/app/accessibility/page.tsx` - Full accessibility statement

**Files Updated:**

- ✅ `apps/web/app/components/Footer.tsx` - Added accessibility link

---

### 12. Missing CCPA "Do Not Sell" Option ✅ **RESOLVED**

**Priority:** MEDIUM (if serving California users)  
**Status:** ✅ COMPLETED

**Issue:** CCPA requires a "Do Not Sell My Personal Information" option. While Writeo doesn't sell data, the opt-out mechanism should be provided if serving California residents.

**Resolution:**

- ✅ Added "Do Not Sell My Personal Information" section to privacy policy
- ✅ Clearly stated that Writeo does not sell personal information
- ✅ Explained that no opt-out is necessary since no data is sold
- ✅ Provided contact method for questions

**Files Updated:**

- ✅ `apps/web/app/privacy/page.tsx` - CCPA "Do Not Sell" section added

---

### 13. Missing Non-Discrimination Policy (CCPA) ✅ **RESOLVED**

**Priority:** MEDIUM (if serving California users)  
**Status:** ✅ COMPLETED

**Issue:** CCPA requires a non-discrimination policy stating that users won't be discriminated against for exercising their privacy rights.

**Resolution:**

- ✅ Added comprehensive non-discrimination policy to privacy policy
- ✅ Stated that service quality won't be reduced for exercising privacy rights
- ✅ Listed all non-discrimination commitments (no denial of service, no price differences, etc.)
- ✅ Clarified that service quality remains the same regardless of privacy rights exercise

**Files Updated:**

- ✅ `apps/web/app/privacy/page.tsx` - Non-discrimination section added

---

### 14. Data Processing Agreement (DPA) Information Missing ✅ **RESOLVED**

**Priority:** MEDIUM  
**Status:** ✅ COMPLETED

**Issue:** Privacy policy mentions third-party services (Groq, Cloudflare, Modal, LanguageTool) but doesn't clarify data processor relationships or provide DPA information for enterprise users.

**Resolution:**

- ✅ Expanded third-party services section with DPA information
- ✅ Clarified data controller vs. data processor relationships (Writeo = controller, third parties = processors)
- ✅ Listed all data processors (Groq, Cloudflare, Modal)
- ✅ Stated that processors are bound by data processing agreements
- ✅ Listed processor obligations (process only for feedback, no retention, security, compliance)

**Files Updated:**

- ✅ `apps/web/app/privacy/page.tsx` - Expanded third-party services section with DPA information

---

### 15. Missing Data Breach Notification Procedures ✅ **RESOLVED**

**Priority:** HIGH (for opt-in server storage users)  
**Status:** ✅ COMPLETED

**Issue:** GDPR and CCPA require data breach notification procedures, but these are not documented. While opt-in storage reduces risk, procedures are still needed.

**Resolution:**

- ✅ Added comprehensive data breach notification section to privacy policy
- ✅ Documented breach notification procedures (assessment, notification, transparency, remediation)
- ✅ Stated notification timeline (72 hours for GDPR)
- ✅ Noted that opt-in storage model reduces breach risk
- ✅ Provided contact method for breach concerns

**Files Updated:**

- ✅ `apps/web/app/privacy/page.tsx` - Data breach notification section added
- ⚠️ `docs/BREACH_RESPONSE.md` - Internal breach response plan still pending (can be created separately for internal use)

---

## Compliance Checklist

### GDPR Compliance

- [x] Privacy policy with required sections
- [x] Legal basis for processing stated (opt-in consent for server storage)
- [x] Data subject rights implemented (access via localStorage by default, deletion via browser, export via localStorage)
- [ ] Data breach notification procedures (only needed for opt-in server storage)
- [x] Cookie consent mechanism (NOT REQUIRED - no cookies used, only localStorage/sessionStorage)
- [x] Age restrictions (NOT REQUIRED - no data collection by default means COPPA doesn't apply, all ages welcome)

### CCPA Compliance (if serving California users)

- [x] Privacy policy with disclosures
- [ ] "Do Not Sell" option (if applicable)
- [ ] Data deletion rights (API pending)
- [ ] Data access rights (export API pending)
- [ ] Non-discrimination policy

### COPPA Compliance (if serving users under 13)

- [ ] Verifiable parental consent
- [ ] Limited data collection
- [ ] Parental access to data
- [ ] Parental deletion rights
- [ ] No behavioral advertising

### General Legal

- [x] Terms of Service (✅ API rate limiting disclosed)
- [x] Privacy Policy (✅ contact information, CCPA sections, DPA info, breach notification added)
- [x] Cookie Policy (✅ created)
- [x] Accessibility Statement (✅ created)
- [x] Third-party licenses documented (partial)
- [ ] Model licenses verified
- [ ] Contact information provided (⚠️ CRITICAL - currently missing)
- [ ] API rate limiting terms disclosed (⚠️ currently not disclosed)
- [ ] Data breach notification procedures documented (⚠️ required for GDPR/CCPA)

---

## Resources

### Legal Resources

- GDPR: https://gdpr.eu/
- CCPA: https://oag.ca.gov/privacy/ccpa
- COPPA: https://www.ftc.gov/tips-advice/business-center/privacy-and-security/children's-privacy
- ePrivacy Directive: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32002L0058

### Third-Party Terms to Review

- Groq: https://groq.com/legal/terms
- Cloudflare: https://www.cloudflare.com/terms/
- LanguageTool: https://languagetool.org/legal/
- HuggingFace: https://huggingface.co/terms

### Model Licenses to Verify

- KevSun/Engessay_grading_ML: https://huggingface.co/KevSun/Engessay_grading_ML
  - **Citation**: Sun, K., & Wang, R. (2024). Automatic Essay Multi-dimensional Scoring with Fine-tuning and Multiple Regression. _ArXiv_. https://arxiv.org/abs/2406.01198
- Michau96/distilbert-base-uncased-essay_scoring: https://huggingface.co/Michau96/distilbert-base-uncased-essay_scoring

---

## ⚠️ Critical Issues Found in January 2025 Review

### Must Fix Before Public Launch:

1. **Missing Contact Information** - Privacy policy and Terms reference "contact us" but provide no actual contact details (email, address). Violates GDPR Article 13(1)(a) and CCPA.

2. ~~**Missing Age Restriction Statement**~~ ✅ **RESOLVED** - Age guidelines added (all ages welcome, parental guidance recommended).

3. ~~**Age Verification Implementation**~~ ✅ **NOT REQUIRED** - With opt-in storage model (no data collection by default), COPPA doesn't apply. No age gate needed.

### Should Fix Soon:

4. ~~**API Rate Limiting Not Disclosed**~~ ✅ **RESOLVED** - Rate limits documented in Terms of Service.

5. ~~**Missing Accessibility Statement**~~ ✅ **RESOLVED** - Accessibility statement created and linked from footer.

6. ~~**Missing Cookie Policy**~~ ✅ **RESOLVED** - Cookie policy created explaining no cookies are used.

7. ~~**CCPA "Do Not Sell" Option**~~ ✅ **RESOLVED** - Added to privacy policy.

8. ~~**Non-Discrimination Policy**~~ ✅ **RESOLVED** - Added to privacy policy.

9. ~~**DPA Information**~~ ✅ **RESOLVED** - Expanded third-party services section with DPA details.

10. ~~**Data Breach Notification Procedures**~~ ✅ **RESOLVED** - Added to privacy policy.

11. **Missing Data Breach Notification Procedures** - Required for GDPR/CCPA compliance (even with opt-in storage model).

### Nice to Have:

7. **Cookie Policy** - While not required (no cookies used), a dedicated policy is best practice.

8. **CCPA "Do Not Sell" Option** - Required if serving California users (even if not selling data).

9. **Non-Discrimination Policy** - Required for CCPA compliance.

10. **DPA Information** - Clarify data processor relationships for enterprise users.

---

## Notes

- This is a living document - update as items are completed
- Regular legal reviews recommended (annually or when adding new features)
- Consult with qualified attorney for jurisdiction-specific requirements
- Some requirements vary by jurisdiction (EU vs US vs others)
- **Last comprehensive review:** January 2025
