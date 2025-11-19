# Legal Compliance Guide

**Last Updated:** 2024  
**Status:** ⚠️ **IN PROGRESS** - Some items completed, others pending

---

## Executive Summary

This document tracks legal compliance requirements for Writeo, including GDPR, CCPA, COPPA, and general legal requirements. Some items have been completed (Terms of Service, Privacy Policy updates), while others remain pending.

**Important:** Writeo uses an **opt-in server storage model** - by default, no data is stored on servers. Results are stored only in the user's browser (localStorage). This significantly reduces legal compliance requirements, as most users don't have data stored on servers.

---

## ✅ Completed Items

### 1. Terms of Service ✅

- ✅ Created comprehensive Terms of Service page (`apps/web/app/terms/page.tsx`)
- ✅ Includes age restrictions, user responsibilities, IP rights, donations policy
- ✅ Linked in footer and privacy policy

### 2. Privacy Policy Enhancements ✅

- ✅ Added donations/payment processing section
- ✅ Added third-party service disclosures
- ✅ Added contact information section
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

### 1. Age Restrictions & COPPA Compliance

**Priority:** CRITICAL  
**Status:** ⚠️ PENDING

**Required:**

- [ ] Add age verification component
- [ ] Require age confirmation (13+ or 16+)
- [ ] Block users under minimum age
- [ ] Update privacy policy with age restrictions
- [ ] Add parental consent flow if allowing 13-16 (EU)

**Files to Create:**

- `apps/web/app/components/AgeGate.tsx`

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

## Compliance Checklist

### GDPR Compliance

- [x] Privacy policy with required sections
- [x] Legal basis for processing stated (opt-in consent for server storage)
- [x] Data subject rights implemented (access via localStorage by default, deletion via browser, export via localStorage)
- [ ] Data breach notification procedures (only needed for opt-in server storage)
- [x] Cookie consent mechanism (NOT REQUIRED - no cookies used, only localStorage/sessionStorage)
- [ ] Age restrictions (16+ or parental consent) - **LOWER PRIORITY** - No server storage by default means COPPA/GDPR requirements are significantly reduced

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

- [x] Terms of Service
- [x] Privacy Policy
- [ ] Cookie Policy
- [ ] Accessibility Statement
- [x] Third-party licenses documented (partial)
- [ ] Model licenses verified

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

## Notes

- This is a living document - update as items are completed
- Regular legal reviews recommended (annually or when adding new features)
- Consult with qualified attorney for jurisdiction-specific requirements
- Some requirements vary by jurisdiction (EU vs US vs others)
