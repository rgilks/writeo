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

    const result = await checkText({
      text: text.trim(),
      language,
      enabledOnly: false,
      level,
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
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-bold mb-4">LanguageTool Demo</h2>

        <div className="mb-4">
          <button
            onClick={handleHealthCheck}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Check Service Status
          </button>
          {healthStatus && (
            <div
              className={`mt-2 p-2 rounded ${healthStatus.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}
            >
              {healthStatus.message}
            </div>
          )}
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Text to check:</label>
            <textarea
              value={text}
              onChange={e => setText(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={4}
              placeholder="Enter text to check for grammar and spelling mistakes..."
            />
          </div>

          <div className="flex gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Language:</label>
              <select
                value={language}
                onChange={e => setLanguage(e.target.value)}
                className="p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
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
              <label className="block text-sm font-medium mb-2">Level:</label>
              <select
                value={level}
                onChange={e => setLevel(e.target.value as 'default' | 'picky')}
                className="p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
              >
                <option value="default">Default</option>
                <option value="picky">Picky</option>
              </select>
            </div>
          </div>

          <button
            onClick={handleCheckText}
            disabled={loading || !text.trim()}
            className="px-6 py-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
          >
            {loading ? 'Checking...' : 'Check Text'}
          </button>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded">
            <strong>Error:</strong> {error}
          </div>
        )}

        {matches.length > 0 && (
          <div className="mt-6">
            <h3 className="text-lg font-semibold mb-3">Issues Found ({matches.length})</h3>
            <div className="space-y-3">
              {matches.map((match, index) => (
                <div key={index} className="p-4 bg-yellow-50 border border-yellow-200 rounded-md">
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="font-medium text-yellow-800">{match.shortMessage}</h4>
                    <span className="text-sm text-yellow-600 bg-yellow-100 px-2 py-1 rounded">
                      {match.rule.category.name}
                    </span>
                  </div>
                  <p className="text-yellow-700 mb-2">{match.message}</p>
                  <div className="text-sm text-yellow-600">
                    <p>
                      <strong>Context:</strong> &quot;{match.context.text}&quot;
                    </p>
                    {match.replacements.length > 0 && (
                      <p>
                        <strong>Suggestions:</strong>{' '}
                        {match.replacements.map(r => r.value).join(', ')}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {!loading && matches.length === 0 && text.trim() && !error && (
          <div className="mt-4 p-4 bg-green-100 border border-green-400 text-green-700 rounded">
            No issues found! Your text looks good.
          </div>
        )}
      </div>
    </div>
  );
}
