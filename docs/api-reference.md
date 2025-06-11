# API Reference

## Overview

The LanguageTool integration provides server-side actions for grammar and spell checking. All APIs use Zod schemas for type safety and validation.

## Type Definitions

### Core Types

#### `LanguageToolMatch`

Represents a single grammar or spelling issue found in text.

```typescript
interface LanguageToolMatch {
  message: string; // Detailed error message
  shortMessage: string; // Brief error description
  offset: number; // Character position in text
  length: number; // Length of the error span
  rule: {
    id: string; // Rule identifier
    description: string; // Rule description
    category: {
      id: string; // Category ID
      name: string; // Category display name
    };
  };
  replacements: Array<{
    value: string; // Suggested replacement
  }>;
  context: {
    text: string; // Surrounding text context
    offset: number; // Context offset
    length: number; // Context length
  };
}
```

#### `LanguageToolResponse`

Complete response from the LanguageTool service.

```typescript
interface LanguageToolResponse {
  software: {
    name: string; // "LanguageTool"
    version: string; // Version number
    buildDate: string; // Build date
  };
  warnings?: Array<{
    incompleteResults: boolean;
  }>;
  language: {
    name: string; // Detected language name
    code: string; // Language code
    detectedLanguage: {
      name: string; // Auto-detected language
      code: string; // Auto-detected code
      confidence: number; // Detection confidence (0-1)
    };
  };
  matches: LanguageToolMatch[];
}
```

#### `LanguageToolCheckRequest`

Request payload for text checking.

```typescript
interface LanguageToolCheckRequest {
  text: string; // Text to check (required, min length 1)
  language: string; // Language code or "auto" (default: "auto")
  enabledOnly: boolean; // Only enabled rules (default: false)
  level: 'default' | 'picky'; // Checking level (default: "default")
}
```

## Server Actions

### `checkText(request: LanguageToolCheckRequest)`

Analyzes text for grammar and spelling issues.

#### Parameters

- `request`: `LanguageToolCheckRequest` - The text analysis request

#### Returns

```typescript
Promise<{ success: true; data: LanguageToolResponse } | { success: false; error: string }>;
```

#### Example Usage

```typescript
import { checkText } from '@/lib/actions';

const result = await checkText({
  text: 'This are a test sentence with errors.',
  language: 'en-US',
  level: 'default',
  enabledOnly: false,
});

if (result.success) {
  console.log(`Found ${result.data.matches.length} issues`);
  result.data.matches.forEach(match => {
    console.log(`${match.shortMessage}: ${match.message}`);
  });
} else {
  console.error(`Error: ${result.error}`);
}
```

#### Error Handling

Common error scenarios:

- Network connectivity issues
- LanguageTool service unavailable
- Invalid request format
- Text too long or empty

### `getAvailableLanguages()`

Retrieves the list of supported languages from LanguageTool.

#### Parameters

None

#### Returns

```typescript
Promise<
  | { success: true; data: Array<{ name: string; code: string; longCode: string }> }
  | { success: false; error: string }
>;
```

#### Example Usage

```typescript
import { getAvailableLanguages } from '@/lib/actions';

const result = await getAvailableLanguages();

if (result.success) {
  result.data.forEach(lang => {
    console.log(`${lang.name} (${lang.code})`);
  });
}
```

### `checkLanguageToolHealth()`

Performs a health check on the LanguageTool service.

#### Parameters

None

#### Returns

```typescript
Promise<{
  success: boolean;
  message: string;
}>;
```

#### Example Usage

```typescript
import { checkLanguageToolHealth } from '@/lib/actions';

const health = await checkLanguageToolHealth();
console.log(`Service status: ${health.success ? 'Healthy' : 'Unhealthy'}`);
console.log(`Message: ${health.message}`);
```

## Zod Schemas

### Validation Schemas

All request and response data is validated using Zod schemas for type safety.

#### `LanguageToolCheckRequestSchema`

```typescript
const LanguageToolCheckRequestSchema = z.object({
  text: z.string().min(1, 'Text is required'),
  language: z.string().default('auto'),
  enabledOnly: z.boolean().default(false),
  level: z.enum(['default', 'picky']).default('default'),
});
```

#### `LanguageToolMatchSchema`

```typescript
const LanguageToolMatchSchema = z.object({
  message: z.string(),
  shortMessage: z.string(),
  offset: z.number(),
  length: z.number(),
  rule: z.object({
    id: z.string(),
    description: z.string(),
    category: z.object({
      id: z.string(),
      name: z.string(),
    }),
  }),
  replacements: z.array(
    z.object({
      value: z.string(),
    })
  ),
  context: z.object({
    text: z.string(),
    offset: z.number(),
    length: z.number(),
  }),
});
```

#### `LanguageToolResponseSchema`

```typescript
const LanguageToolResponseSchema = z.object({
  software: z.object({
    name: z.string(),
    version: z.string(),
    buildDate: z.string(),
  }),
  warnings: z
    .array(
      z.object({
        incompleteResults: z.boolean(),
      })
    )
    .optional(),
  language: z.object({
    name: z.string(),
    code: z.string(),
    detectedLanguage: z.object({
      name: z.string(),
      code: z.string(),
      confidence: z.number(),
    }),
  }),
  matches: z.array(LanguageToolMatchSchema),
});
```

## Language Codes

### Supported Languages

LanguageTool supports numerous languages. Common codes include:

- `auto` - Automatic detection
- `en-US` - English (United States)
- `en-GB` - English (United Kingdom)
- `de-DE` - German (Germany)
- `es` - Spanish
- `fr` - French
- `it` - Italian
- `pt-BR` - Portuguese (Brazil)
- `nl` - Dutch
- `pl` - Polish
- `ru` - Russian
- `ca-ES` - Catalan
- `be-BY` - Belarusian
- `zh-CN` - Chinese

Use `getAvailableLanguages()` to get the complete, up-to-date list.

## Error Codes and Messages

### Common Errors

#### Network Errors

- `LanguageTool API error: 500 Internal Server Error`
- `LanguageTool API error: 503 Service Unavailable`
- `Network request failed`

#### Validation Errors

- `Text is required` - Empty text field
- `Invalid language code` - Unsupported language
- `Text too long` - Exceeds service limits

#### Configuration Errors

- `LanguageTool endpoint not configured`
- `LANGUAGETOOL_ENDPOINT environment variable is not set`

## Rate Limits and Constraints

### Service Limits

- **Text Length**: Maximum 50,000 characters per request
- **Request Rate**: No explicit limits in current configuration
- **Concurrent Requests**: Limited by ECS service capacity

### Best Practices

1. **Debounce Requests**: Avoid sending requests on every keystroke
2. **Batch Processing**: Combine multiple checks when possible
3. **Error Handling**: Implement graceful degradation
4. **Caching**: Cache results for unchanged text

## Client Integration

### React Hook Example

```typescript
import { useState, useCallback } from 'react';
import { checkText } from '@/lib/actions';
import type { LanguageToolMatch } from '@/lib/types';

export function useLanguageTool() {
  const [matches, setMatches] = useState<LanguageToolMatch[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const check = useCallback(async (text: string, language = 'auto') => {
    if (!text.trim()) return;

    setLoading(true);
    setError(null);

    const result = await checkText({
      text: text.trim(),
      language,
      enabledOnly: false,
      level: 'default',
    });

    if (result.success) {
      setMatches(result.data.matches);
    } else {
      setError(result.error);
      setMatches([]);
    }

    setLoading(false);
  }, []);

  return { matches, loading, error, check };
}
```

### Usage in Component

```typescript
export default function TextEditor() {
  const [text, setText] = useState('');
  const { matches, loading, error, check } = useLanguageTool();

  const handleCheck = () => check(text);

  return (
    <div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text to check..."
      />
      <button onClick={handleCheck} disabled={loading}>
        {loading ? 'Checking...' : 'Check Grammar'}
      </button>
      {error && <div className="error">{error}</div>}
      {matches.map((match, i) => (
        <div key={i} className="issue">
          <strong>{match.shortMessage}</strong>: {match.message}
        </div>
      ))}
    </div>
  );
}
```

## Testing

### Unit Tests

```typescript
import { checkText } from '@/lib/actions';

describe('LanguageTool API', () => {
  it('should detect grammar errors', async () => {
    const result = await checkText({
      text: 'This are wrong.',
      language: 'en-US',
      level: 'default',
      enabledOnly: false,
    });

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.matches.length).toBeGreaterThan(0);
      expect(result.data.matches[0].shortMessage).toContain('grammar');
    }
  });

  it('should handle empty text', async () => {
    const result = await checkText({
      text: '',
      language: 'auto',
      level: 'default',
      enabledOnly: false,
    });

    expect(result.success).toBe(false);
    expect(result.error).toContain('required');
  });
});
```

### Integration Tests

```typescript
import { test, expect } from '@playwright/test';

test('grammar checking workflow', async ({ page }) => {
  await page.goto('/');

  await page.fill('[placeholder*="text to check"]', 'This are a test.');
  await page.click('button:has-text("Check Text")');

  await expect(page.locator('.issue')).toBeVisible();
  await expect(page.locator('.issue')).toContainText('grammar');
});
```
