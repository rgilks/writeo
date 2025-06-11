import { z } from 'zod';

export const LanguageToolMatchSchema = z.object({
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

export const LanguageToolResponseSchema = z.object({
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

export const LanguageToolCheckRequestSchema = z.object({
  text: z.string().min(1, 'Text is required'),
  language: z.string().default('auto'),
  enabledOnly: z.boolean().default(false),
  level: z.enum(['default', 'picky']).default('default'),
});

export type LanguageToolMatch = z.infer<typeof LanguageToolMatchSchema>;
export type LanguageToolResponse = z.infer<typeof LanguageToolResponseSchema>;
export type LanguageToolCheckRequest = z.infer<typeof LanguageToolCheckRequestSchema>;
