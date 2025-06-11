import { useState, useCallback } from 'react';
import { checkText } from './actions';
import { LanguageToolMatch } from './types';
import { produce } from 'immer';

export const useLanguageTool = () => {
  const [matches, setMatches] = useState<LanguageToolMatch[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const check = useCallback(async (text: string) => {
    if (!text.trim()) {
      setMatches([]);
      return;
    }

    setLoading(true);
    setError(null);

    const result = await checkText({
      text: text.trim(),
      language: 'en-US',
      motherTongue: 'en-US',
      level: 'picky',
      enabledOnly: false,
    });

    if (result.success) {
      setMatches(result.data.matches);
    } else {
      setError(result.error);
      setMatches([]);
    }

    setLoading(false);
  }, []);

  const applySuggestion = useCallback(
    (text: string, match: LanguageToolMatch, replacement: string) => {
      const newText =
        text.substring(0, match.offset) + replacement + text.substring(match.offset + match.length);

      const offsetDifference = replacement.length - match.length;

      const newMatches = produce(matches, draft => {
        const currentMatchIndex = draft.findIndex(
          m => m.offset === match.offset && m.message === match.message
        );
        if (currentMatchIndex > -1) {
          draft.splice(currentMatchIndex, 1);
        }

        return draft
          .map(m => {
            if (m.offset > match.offset) {
              return {
                ...m,
                offset: m.offset + offsetDifference,
              };
            }
            return m;
          })
          .filter(Boolean) as LanguageToolMatch[];
      });

      setMatches(newMatches);
      return newText;
    },
    [matches]
  );

  return { matches, loading, error, check, applySuggestion };
};
