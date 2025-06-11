'use client';

import { checkLanguageToolHealth } from '../lib/actions';
import { useState } from 'react';
import { TextChecker } from './text-checker';

export default function LanguageToolDemo() {
  const [healthStatus, setHealthStatus] = useState<{ success: boolean; message: string } | null>(
    null
  );

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
          <TextChecker />
        </div>
      </div>
    </div>
  );
}
