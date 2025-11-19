/**
 * Grammar Rules Utility
 * Provides pedagogical explanations for common grammar errors
 */

export interface GrammarRule {
  errorType: string;
  why: string; // Why this error happens
  rule: string; // Formal grammar rule
  examples: string[]; // 2-3 correct usage examples
}

// Grammar rule mappings by error type
const grammarRules: Record<string, GrammarRule> = {
  "Subject-verb agreement": {
    errorType: "Subject-verb agreement",
    why: "In English, the verb must match the subject in number. Singular subjects need singular verbs, and plural subjects need plural verbs.",
    rule: "The verb must agree with its subject in number (singular/plural). Singular subjects take singular verbs (e.g., 'he goes'), and plural subjects take plural verbs (e.g., 'they go').",
    examples: [
      "Correct: The student goes to school. (singular subject + singular verb)",
      "Correct: The students go to school. (plural subject + plural verb)",
      "Correct: She is happy. They are happy. (verb matches subject number)",
    ],
  },
  "Verb tense": {
    errorType: "Verb tense",
    why: "Verb tenses show when an action happens. When writing about the past, all verbs in that narrative should be in past tense for consistency.",
    rule: "Use consistent verb tenses within a narrative. If you're writing about past events, use past tense verbs throughout. Present tense is for current actions, and future tense is for actions that will happen.",
    examples: [
      "Correct: Yesterday I went to the store and bought some milk. (both verbs in past tense)",
      "Correct: I am studying now, and I will finish later. (present and future tenses)",
      "Correct: Last week I visited my friend and we had dinner together. (consistent past tense)",
    ],
  },
  "Article use": {
    errorType: "Article use",
    why: "Articles (a, an, the) help specify whether we're talking about something specific or general. 'A/an' is for general or first mention, 'the' is for specific or already mentioned things.",
    rule: "Use 'a' before consonant sounds, 'an' before vowel sounds, and 'the' for specific or previously mentioned nouns. Use 'a/an' for general or first-time mentions, 'the' for specific or known items.",
    examples: [
      "Correct: I saw a cat. The cat was black. (first mention uses 'a', second uses 'the')",
      "Correct: An apple a day keeps the doctor away. ('an' before vowel sound, 'the' for specific doctor)",
      "Correct: I need a pen. Can you give me the pen? (general vs. specific)",
    ],
  },
  "Preposition use": {
    errorType: "Preposition use",
    why: "Prepositions show relationships between words (time, place, direction). Different prepositions have different meanings, and some are used in fixed expressions.",
    rule: "Prepositions indicate relationships: time (at, on, in), place (at, on, in), direction (to, from, into). Many preposition uses are fixed expressions that must be memorized (e.g., 'good at', 'depend on').",
    examples: [
      "Correct: I arrived at the station. (not 'to' the station)",
      "Correct: She is good at mathematics. (fixed expression: 'good at')",
      "Correct: The book is on the table. (not 'in' the table)",
    ],
  },
  Spelling: {
    errorType: "Spelling",
    why: "English spelling can be tricky because many words don't follow simple rules. Some words have silent letters or unusual spellings that must be memorized.",
    rule: "English spelling follows patterns but has many exceptions. Common patterns include: 'i before e except after c' (receive), doubling consonants before -ing (running), and silent letters (knight, write).",
    examples: [
      "Correct: receive (not 'recieve')",
      "Correct: running (double 'n' before -ing)",
      "Correct: write, wrote, written (irregular spelling patterns)",
    ],
  },
  Grammar: {
    errorType: "Grammar",
    why: "Grammar rules help make your writing clear and understandable. Following grammar rules helps readers understand your meaning.",
    rule: "Grammar rules provide structure and clarity to language. Following standard grammar conventions helps ensure your writing is clear and professional.",
    examples: [
      "Correct: I like reading books. (proper sentence structure)",
      "Correct: She speaks English fluently. (adverb placement)",
      "Correct: The dog that I saw was friendly. (relative clause)",
    ],
  },
  Typo: {
    errorType: "Typo",
    why: "Typos are accidental mistakes in typing or writing. They often happen when typing quickly or not proofreading carefully.",
    rule: "Typos are unintentional errors in spelling, punctuation, or word choice. Proofreading helps catch and correct typos before submitting your work.",
    examples: [
      "Correct: I went to the store. (not 'I wnet to the store')",
      "Correct: She is my friend. (not 'She is my freind')",
      "Correct: They are happy. (not 'They are hapy')",
    ],
  },
  Style: {
    errorType: "Style",
    why: "Writing style affects how clear and professional your writing sounds. Some phrases are wordy or unclear and can be improved.",
    rule: "Good writing style is clear, concise, and appropriate for the context. Avoid wordiness, redundancy, and overly informal language in academic writing.",
    examples: [
      "Better: I think that... → I think... (remove unnecessary 'that')",
      "Better: very good → excellent (more concise)",
      "Better: in order to → to (more concise)",
    ],
  },
  Punctuation: {
    errorType: "Punctuation",
    why: "Punctuation marks help readers understand your meaning by showing pauses, connections, and emphasis. Missing or incorrect punctuation can confuse readers.",
    rule: "Punctuation marks clarify meaning: periods (.) end sentences, commas (,) separate ideas, apostrophes (') show possession or contractions, and question marks (?) indicate questions.",
    examples: [
      "Correct: I like apples, oranges, and bananas. (commas in lists)",
      "Correct: It's a beautiful day. (apostrophe for contraction)",
      "Correct: What time is it? (question mark for questions)",
    ],
  },
  "Word order": {
    errorType: "Word order",
    why: "English has a specific word order (Subject-Verb-Object). Changing this order can make sentences confusing or grammatically incorrect.",
    rule: "English follows Subject-Verb-Object (SVO) order. Adjectives come before nouns, and adverbs usually come after verbs. Questions invert the subject and verb.",
    examples: [
      "Correct: I like pizza. (Subject-Verb-Object order)",
      "Correct: She speaks English well. (adverb after verb)",
      "Correct: Do you like pizza? (question: verb before subject)",
    ],
  },
  "Word choice": {
    errorType: "Word choice",
    why: "Some words sound similar but have different meanings. Using the wrong word can change your meaning or make your writing unclear.",
    rule: "Choose words that accurately express your meaning. Pay attention to commonly confused words (their/there, your/you're, its/it's) and use a dictionary when unsure.",
    examples: [
      "Correct: Their house is big. (possession) vs. There is a house. (location)",
      "Correct: You're happy. (you are) vs. Your book. (possession)",
      "Correct: It's raining. (it is) vs. The dog wagged its tail. (possession)",
    ],
  },
  "Sentence structure": {
    errorType: "Sentence structure",
    why: "Sentences need a subject and a verb to be complete. Incomplete sentences or run-on sentences can confuse readers.",
    rule: "A complete sentence needs a subject (who/what) and a verb (action). Avoid sentence fragments (incomplete thoughts) and run-on sentences (too many ideas without proper punctuation).",
    examples: [
      "Correct: I went to the store. (complete sentence with subject and verb)",
      "Correct: She likes reading, and he likes writing. (two ideas connected properly)",
      "Correct: Because I was tired, I went to bed early. (complex sentence with proper structure)",
    ],
  },
};

/**
 * Get grammar rule information for an error type
 */
export function getGrammarRule(errorType?: string): GrammarRule | null {
  if (!errorType) return null;

  // Try exact match first
  if (grammarRules[errorType]) {
    return grammarRules[errorType];
  }

  // Try case-insensitive match
  const lowerErrorType = errorType.toLowerCase();
  for (const [key, rule] of Object.entries(grammarRules)) {
    if (key.toLowerCase() === lowerErrorType) {
      return rule;
    }
  }

  // Try partial match for common patterns
  if (lowerErrorType.includes("subject") && lowerErrorType.includes("verb")) {
    return grammarRules["Subject-verb agreement"];
  }
  if (lowerErrorType.includes("tense") || lowerErrorType.includes("verb")) {
    return grammarRules["Verb tense"];
  }
  if (lowerErrorType.includes("article")) {
    return grammarRules["Article use"];
  }
  if (lowerErrorType.includes("preposition")) {
    return grammarRules["Preposition use"];
  }

  // Default fallback
  return grammarRules["Grammar"] || null;
}

/**
 * Get all available grammar rule types
 */
export function getAvailableGrammarRuleTypes(): string[] {
  return Object.keys(grammarRules);
}
