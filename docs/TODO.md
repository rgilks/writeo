# TODO

## Testing (Future Improvements)

**Philosophy**: Keep tests lean and focused. Avoid over-testing simple utilities. Prioritize critical paths.

### High Priority (Only if needed)

1. **Text Processing Utilities** (if logic becomes more complex)
   - `tests/api-worker/utils.text-processing.test.ts`
   - Only add if the position validation/confidence scoring logic grows in complexity
   - Current integration tests may be sufficient

2. **Service Layer Tests** (only if integration tests become slow/unreliable)
   - `tests/api-worker/services/*.test.ts`
   - Only add if we need faster feedback loops or if mocking becomes necessary
   - Current integration tests cover these well

### Medium Priority (Consider when adding new features)

3. **Component Tests** (not recommended)
   - ❌ **No React Testing Library** - E2E tests already cover user interactions
   - If component logic becomes complex, consider extracting to utilities and testing those instead
   - Keep components simple and rely on E2E tests for UI behavior

4. **Route Handler Unit Tests** (only if route logic becomes complex)
   - Current integration tests cover routes well
   - Only add if routes develop complex business logic that needs isolated testing

### Low Priority / Future

5. **Test Utilities** (if patterns emerge)
   - Centralized mock factories
   - Better test data builders
   - Only add if we find ourselves repeating setup code

---

**Note**: Current test suite is in good shape:

- ✅ 453 unit tests passing
- ✅ 31 test files covering critical paths
- ✅ Good coverage of security, validation, middleware, storage
- ✅ Fast execution (<1s for unit tests)
- ✅ Data-driven tests using `it.each` pattern

**Guidelines for adding tests**:

- ✅ Add tests for new security-critical code
- ✅ Add tests for complex business logic
- ✅ Add tests for utilities used across multiple modules
- ❌ Don't add tests for simple getters/setters
- ❌ Don't add tests for thin wrappers around well-tested functions
- ❌ Don't duplicate coverage already provided by integration tests

---

## Hooks & State Management (Future Considerations)

**Status**: ✅ All critical and medium-priority hook issues have been resolved.

### Low Priority / Future

1. **Consider Zustand stores for shared state** (only if needed)
   - TeacherFeedback state - Only if feedback mode/explanation needs to persist across page reloads
   - Results page loading state - Only if loading state needs to be shown in header/navigation
   - AI Feedback streaming state - Only if streaming state needs to be shared or persisted
   - **Note**: Current local state is appropriate for component-specific state

2. **Monitor hook performance** (if issues arise)
   - Watch for unnecessary re-renders from Zustand selectors
   - Consider using shallow equality for complex selectors if needed
   - Current implementation is efficient with proper selector usage

---

## Styling & CSS (Future Considerations)

**Status**: ✅ All styling improvements completed. CSS is well-organized and maintainable.

### Low Priority / Future

1. **CSS Modules** (only if class name conflicts become an issue)
   - Current global CSS approach is working well
   - Consider only if component-specific styles need better encapsulation

2. **Design System Documentation** (if team grows)
   - Document all CSS variables in a style guide
   - Create usage examples for common patterns
   - Only needed if multiple developers are working on styling

3. **Performance Optimization** (if CSS file size becomes an issue)
   - Current 1,430 lines is reasonable
   - Consider CSS splitting only if bundle size becomes a concern
