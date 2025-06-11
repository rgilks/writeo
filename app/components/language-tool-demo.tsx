'use client';

import { useState } from 'react';
import { checkText, checkLanguageToolHealth } from '../lib/actions';
import { LanguageToolMatch } from '../lib/types';

export default function LanguageToolDemo() {
  const [text, setText] = useState('');
  const [language, setLanguage] = useState('auto');
  const [level, setLevel] = useState<'default' | 'picky'>('default');
  const [matches, setMatches] = useState<LanguageToolMatch[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [healthStatus, setHealthStatus] = useState<{ success: boolean; message: string } | null>(
    null
  );

  const handleCheckText = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    setMatches([]);

    const result = await checkText({
      text: text.trim(),
      language,
      enabledOnly: true,
      level,
      enabledCategories: 'GRAMMAR,PUNCTUATION,STYLE,TYPOGRAPHY,SPELLING',
    });

    if (result.success) {
      setMatches(result.data.matches);
    } else {
      setError(result.error);
      setMatches([]);
    }

    setLoading(false);
  };

  const handleHealthCheck = async () => {
    const result = await checkLanguageToolHealth();
    setHealthStatus(result);
  };

  return (
    <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="bg-slate-800 rounded-2xl shadow-2xl overflow-hidden border border-slate-700">
        <div className="p-6 sm:p-8">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl sm:text-3xl font-bold text-white">LanguageTool Demo</h2>
            <button
              onClick={handleHealthCheck}
              className="px-4 py-2 text-sm font-medium text-white bg-indigo-600 rounded-lg shadow-md hover:bg-indigo-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 focus:ring-offset-slate-800 transition-all duration-300 hover:shadow-indigo-500/50"
            >
              Check Service Status
            </button>
          </div>

          {healthStatus && (
            <div
              className={`mb-6 p-4 rounded-lg text-sm ${
                healthStatus.success
                  ? 'bg-green-900/30 text-green-300 border border-green-500/30'
                  : 'bg-red-900/30 text-red-300 border border-red-500/30'
              }`}
            >
              {healthStatus.message}
            </div>
          )}

          <div className="space-y-6">
            <div>
              <label
                htmlFor="text-to-check"
                className="block text-sm font-medium text-gray-300 mb-2"
              >
                Text to check
              </label>
              <textarea
                id="text-to-check"
                value={text}
                onChange={e => setText(e.target.value)}
                className="w-full p-4 bg-slate-700 border border-slate-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-white transition-shadow shadow-sm placeholder-gray-400"
                rows={6}
                placeholder="Start typing or paste your text here..."
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              <div>
                <label htmlFor="language" className="block text-sm font-medium text-gray-300 mb-2">
                  Language
                </label>
                <select
                  id="language"
                  value={language}
                  onChange={e => setLanguage(e.target.value)}
                  className="w-full p-3 bg-slate-700 border border-slate-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-white shadow-sm"
                >
                  <option value="auto">Auto-detect</option>
                  <option value="en-US">English (US)</option>
                  <option value="en-GB">English (UK)</option>
                  <option value="es">Spanish</option>
                  <option value="fr">French</option>
                  <option value="de">German</option>
                </select>
              </div>
              <div>
                <label htmlFor="level" className="block text-sm font-medium text-gray-300 mb-2">
                  Level
                </label>
                <select
                  id="level"
                  value={level}
                  onChange={e => setLevel(e.target.value as 'default' | 'picky')}
                  className="w-full p-3 bg-slate-700 border border-slate-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-white shadow-sm"
                >
                  <option value="default">Default</option>
                  <option value="picky">Picky</option>
                </select>
              </div>
            </div>

            <div className="text-center pt-2">
              <button
                onClick={handleCheckText}
                disabled={loading || !text.trim()}
                className="w-full sm:w-auto px-8 py-3 text-base font-medium text-white bg-green-600 rounded-full shadow-lg hover:bg-green-500 disabled:bg-gray-600 disabled:cursor-not-allowed transform hover:scale-105 transition-all duration-300 ease-in-out hover:shadow-green-500/50"
              >
                {loading ? (
                  <span className="flex items-center justify-center">
                    <svg
                      className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Checking...
                  </span>
                ) : (
                  'Check Text'
                )}
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className="p-6 sm:p-8 border-t border-slate-700">
            <div className="p-4 bg-red-900/30 border border-red-500/30 text-red-300 rounded-lg">
              <strong className="font-semibold">Error:</strong> {error}
            </div>
          </div>
        )}

        {matches.length > 0 && (
          <div className="p-6 sm:p-8 border-t border-slate-700">
            <h3 className="text-xl font-bold text-white mb-4">Issues Found ({matches.length})</h3>
            <div className="space-y-4">
              {matches.map((match, index) => (
                <div
                  key={index}
                  className="p-4 bg-yellow-900/20 border border-yellow-500/30 rounded-lg shadow-sm"
                >
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="font-semibold text-yellow-200">{match.shortMessage}</h4>
                    <span className="text-xs font-medium text-yellow-200 bg-yellow-900/40 px-2 py-1 rounded-full">
                      {match.rule.category.name}
                    </span>
                  </div>
                  <p className="text-yellow-300 mb-3">{match.message}</p>
                  <div className="text-sm text-gray-400 bg-slate-700 rounded p-3">
                    <p className="mb-2">
                      <strong>Context:</strong> &quot;
                      <span className="font-mono bg-slate-600 px-1 rounded">
                        {match.context.text}
                      </span>
                      &quot;
                    </p>
                    {match.replacements.length > 0 && (
                      <p>
                        <strong>Suggestions:</strong>{' '}
                        <span className="font-medium text-green-400">
                          {match.replacements.map(r => r.value).join(', ')}
                        </span>
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {!loading && matches.length === 0 && text.trim() && !error && (
          <div className="p-6 sm:p-8 border-t border-slate-700">
            <div className="p-4 bg-green-900/30 border border-green-500/30 text-green-300 rounded-lg">
              No issues found! Your text looks good.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
