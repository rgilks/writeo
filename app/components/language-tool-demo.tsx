'use client';

import { checkLanguageToolHealth } from '../lib/actions';
import { useState, useEffect } from 'react';
import { TextChecker } from './text-checker';

const StatusIndicator = ({ success }: { success: boolean }) => (
  <div className="flex items-center space-x-2">
    <span className={`h-3 w-3 rounded-full ${success ? 'bg-green-500' : 'bg-red-500'}`}></span>
    <span className="text-sm text-slate-400">
      {success ? 'Service Operational' : 'Service Down'}
    </span>
  </div>
);

export default function LanguageToolDemo() {
  const [healthStatus, setHealthStatus] = useState<{ success: boolean; message: string } | null>(
    null
  );

  useEffect(() => {
    const performHealthCheck = async () => {
      const result = await checkLanguageToolHealth();
      setHealthStatus(result);
    };

    const timer = setTimeout(() => {
      performHealthCheck();
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="bg-slate-800 rounded-2xl shadow-2xl overflow-hidden border border-slate-700">
        <div className="p-6 sm:p-8">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl sm:text-3xl font-bold text-white">LanguageTool Demo</h2>
            {healthStatus && <StatusIndicator success={healthStatus.success} />}
          </div>

          <TextChecker />
        </div>
      </div>
    </div>
  );
}
