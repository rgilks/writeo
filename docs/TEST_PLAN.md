# Test Plan

**Last Updated**: January 2025  
**Test Coverage**: ~85 tests (25 API + 60 E2E) covering critical functionality

## Quick Start

```bash
npm test              # API tests (Vitest)
npm run test:e2e      # E2E tests (Playwright)
npm run test:all      # Both API + E2E
```

See [TESTING.md](TESTING.md) for detailed test documentation.

## Test Coverage

### API Tests (Vitest) - `tests/api.test.ts`

**Coverage**: Core API functionality, error handling, validation, synchronous processing

- Full E2E workflow
- AI feedback integration
- Grammar error detection
- Confidence tiers and scores
- Context-aware tense detection
- LLM assessment integration
- Teacher feedback persistence
- Streaming endpoints
- Performance timing
- Input validation
- Error handling and retry logic
- Relevance checking
- Cost controls (essay truncation)

**Total**: ~25 test cases covering critical API functionality

### E2E Tests (Playwright) - `tests/e2e/*.spec.ts`

**Coverage**: User-facing flows, UI interactions, visual design, responsive layout

- Homepage and navigation
- Form submission and validation
- Results display and feedback
- Interactive learning flow
- Draft tracking and navigation
- Error handling
- Visual design verification
- Responsive layouts (mobile/tablet/desktop)

**Total**: ~60 test cases covering critical user flows

## Manual Testing

Some visual and subjective aspects require manual browser verification:

- Color palette and typography
- Spacing and layout (8pt grid)
- Error highlighting opacity
- Heat map visual appearance
- Cross-browser consistency
- Animation verification

## Test Maintenance

- **Add tests**: When adding new features or endpoints
- **Update tests**: When API contracts change or UI flows change
- **Remove tests**: When features are deprecated

## References

- [TESTING.md](TESTING.md) - Test suite documentation
- [SCRIPTS.md](SCRIPTS.md) - Scripts documentation
