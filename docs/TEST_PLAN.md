# Writeo Test Plan

**Last Updated**: 2025-01-18  
**Purpose**: Lean test plan focusing on automated tests and essential manual browser verification  
**Test Coverage**: ~85 tests (25 API + 60 E2E) covering critical functionality, visual design, and responsive layout

## Quick Start

```bash
# Run all automated tests
npm test                    # API tests (Vitest)
npm run test:e2e            # E2E tests (Playwright)
npm run test:all            # Both API + E2E

# Watch mode
npm run test:watch          # API tests in watch mode
npm run test:e2e:ui         # E2E tests with UI mode
```

## Test Coverage Overview

### ✅ Automated Tests

#### API Tests (Vitest) - `tests/api.test.ts`

**Coverage**: Core API functionality, error handling, validation, synchronous processing

| Test Suite                 | Coverage                                                       |
| -------------------------- | -------------------------------------------------------------- |
| `smoke`                    | Full E2E workflow (TC-E2E-001, TC-API-001-036)                 |
| `ai-feedback`              | AI feedback integration (TC-AI-001-010)                        |
| `lt`                       | Grammar error detection (TC-GRAM-001-010)                      |
| `lt - confidence scores`   | Confidence tiers and scores (TC-ERRDET-003, TC-MEDCONF-005)    |
| `lt - context-aware tense` | Context-aware tense detection (TC-ERRDET-001-002)              |
| `llm - assessment`         | LLM assessment integration (TC-ERRDET-011-013)                 |
| `teacher-feedback`         | Teacher feedback persistence (TC-FE-040-043, TC-LEARN-010-018) |
| `streaming`                | AI feedback streaming endpoint                                 |
| `timing`                   | Performance timing (TC-PERF-001-002)                           |
| `validation`               | Input validation (TC-API-002-003, TC-FORM-014-017)             |
| `error handling`           | Error handling and retry logic (TC-ERR-002-004)                |
| `synchronous`              | Synchronous processing verification                            |
| `relevance`                | Answer relevance check                                         |
| `cost-controls`            | Essay truncation for long essays                               |

**Total**: ~20 test cases covering critical API functionality

#### E2E Tests (Playwright) - `tests/e2e/*.spec.ts`

**Coverage**: User-facing flows, UI interactions, visual design, and responsive layout

| Test File                      | Coverage                                                                                       |
| ------------------------------ | ---------------------------------------------------------------------------------------------- |
| `homepage.spec.ts`             | Homepage loading, task cards, navigation (TC-FE-001-007)                                       |
| `writing.spec.ts`              | Form submission, validation, word count (TC-FE-009-020, TC-FORM-014-017)                       |
| `results.spec.ts`              | Results display, scores, errors, teacher feedback (TC-FE-024-044)                              |
| `interactive-learning.spec.ts` | Editing, resubmission, teacher feedback (TC-LEARN-001-018)                                     |
| `draft-tracking.spec.ts`       | Draft tracking and navigation (TC-DRAFT-001-020)                                               |
| `error-handling.spec.ts`       | Friendly error messages (TC-ERR-011-017)                                                       |
| `visual.spec.ts`               | Button sizes, error colors, tooltips, text contrast (TC-STYLE-011-012, TC-GRAM-001, TC-FE-036) |
| `responsive.spec.ts`           | Mobile/tablet/desktop layouts, touch targets (TC-STYLE-013-017)                                |

**Total**: ~60 test cases covering critical user flows, visual design, and responsive layout

### ⚠️ Manual Browser Tests Required

The following tests require manual browser verification due to visual/subjective nature:

#### 1. Visual UI Verification (TC-STYLE-001-010)

**Priority**: Medium  
**Frequency**: Before releases, after major UI changes

**Automated**: Button sizes (44px+), text contrast basics, responsive layouts

**Manual Checklist**:

- [ ] Color palette matches style guide (medium blue #3b82f6, gentle green)
- [ ] Typography uses system fonts (San Francisco, Roboto, Segoe UI)
- [ ] Spacing follows 8pt grid (8, 16, 24, 32px)
- [ ] Animations are subtle (Framer Motion)
- [ ] Visual hierarchy is clear (headings, body text distinct)

**Tools**: Browser DevTools, color picker

#### 2. Error Highlighting Visual Verification (TC-GRAM-002-010, TC-PREC-001-014)

**Priority**: High  
**Frequency**: After error detection changes

**Automated**: Error colors (red/orange/amber), tooltip presence, basic color verification

**Manual Checklist**:

- [ ] High-confidence errors highlighted with red underline (40% opacity)
- [ ] Medium-confidence errors highlighted with orange underline
- [ ] Experimental errors highlighted with amber underline
- [ ] Heat map shows general problem areas (not exact errors)
- [ ] Heat map opacity is subtle (40% errors, 30% context)
- [ ] Context area spans ~50 characters around error
- [ ] Error tooltips display correctly on hover (content verified)
- [ ] Confidence indicators show "(Medium Confidence)" or "(Experimental)" text

**Test Essay**:

```
I goes to park yesterday. The dog was happy and we plays together. He are very nice. I has a good time.
```

#### 3. Cross-Browser Visual Consistency

**Priority**: Medium  
**Frequency**: Before major releases

**Browsers**: Chrome, Firefox, Safari

**Checklist**:

- [ ] Layout renders correctly in all browsers
- [ ] Colors display consistently
- [ ] Animations work smoothly
- [ ] Touch interactions work (mobile Safari)
- [ ] No browser-specific console errors

#### 4. Responsive Design Verification

**Priority**: High  
**Frequency**: After layout changes

**Automated**: Mobile/tablet/desktop layouts, touch target sizes, responsive results page

**Manual Checklist**:

- [ ] Proper margins and padding on all sizes (visual verification)
- [ ] Text is readable (16px minimum)
- [ ] Navigation works on small screens
- [ ] Progress dashboard adapts to screen size

#### 5. Complex UI Interactions

**Priority**: Low  
**Frequency**: As needed

**Checklist**:

- [ ] Draft history navigation hover effects (scale, color change)
- [ ] Achievement sparkle effects display correctly
- [ ] CEFR progress bar animations
- [ ] Loading state transitions are smooth
- [ ] Error state styling is friendly (not red/scary)

## Test Execution Strategy

### Pre-Commit

- Run `npm test` (API tests) - should pass
- Run `npm run test:e2e` (E2E tests) - should pass

### Pre-Release

1. **Automated Tests**: All must pass

   ```bash
   npm run test:all
   ```

2. **Manual Visual Verification**:
   - Visual UI verification (TC-STYLE-001-020)
   - Error highlighting visual check (TC-GRAM-001-010)
   - Cross-browser consistency check
   - Responsive design verification

3. **Smoke Test**: Full user flow in browser
   - Navigate to homepage
   - Submit essay with errors
   - Verify results display
   - Edit and resubmit
   - Check draft history

### CI/CD

- Run automated tests only (`npm run test:ci`)
- Manual tests performed before release

## Test Data

### Standard Test Essays

**Short Essay (<50 words)**:

```
Last weekend I went to the park. I played with my dog. We had fun together. It was a nice day.
```

**Essay with Errors**:

```
I goes to park yesterday. The dog was happy and we plays together. He are very nice. I has a good time.
```

**Corrected Essay**:

```
I went to the park yesterday. The dog was happy and we played together. He was very nice. I had a good time.
```

**Valid Essay (250-500 words)**:
Use essay generator in test helpers or write ~300 words about a weekend experience.

## Known Limitations

### Not Automated (Require Manual Testing)

- Color palette verification (specific hex values)
- Typography (font-family verification)
- Spacing (8pt grid verification)
- Error highlighting opacity (40% errors, 30% context)
- Heat map visual appearance
- Cross-browser visual consistency
- Complex animation verification
- Visual hierarchy (subjective assessment)

### Partially Automated

- Error tooltips (automated hover interaction, but tooltip content/styling requires manual check)
- Teacher feedback formatting (automated content check, but paragraph formatting requires visual check)
- Responsive design (automated layout checks, but visual spacing/padding requires manual verification)

## Test Maintenance

### When to Update Tests

- **Add tests**: When adding new features or endpoints
- **Update tests**: When API contracts change or UI flows change
- **Remove tests**: When features are deprecated

### Test Quality Guidelines

- Tests should be independent (no shared state)
- Tests should use unique IDs (UUIDs) to avoid conflicts
- Tests should have clear descriptions matching test case IDs
- Tests should fail fast with clear error messages

## Troubleshooting

### API Tests Failing

- Check `API_KEY` and `API_BASE` environment variables
- Verify API worker is running and accessible
- Check network connectivity
- Review API worker logs for errors

### E2E Tests Failing

- Check `PLAYWRIGHT_BASE_URL` environment variable
- Verify frontend is running and accessible
- Check browser console for errors
- Review Playwright trace files (`playwright-report/`)

### Flaky Tests

- Increase timeouts for slow operations
- Add explicit waits for async operations
- Check for race conditions
- Review test isolation (unique IDs, no shared state)

## References

- **Test Suite**: See [tests/README.md](../tests/README.md) for test suite documentation
- **Test Scripts**: See [scripts/README.md](../scripts/README.md) for test script usage
- **API Documentation**: See [docs/openapi.yaml](./openapi.yaml) for API specification
