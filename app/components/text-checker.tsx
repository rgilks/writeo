'use client';

import React, { useState, useMemo, useRef, useEffect } from 'react';
import { useLanguageTool } from '../lib/use-language-tool';
import { LanguageToolMatch } from '../lib/types';
import { produce } from 'immer';

type TextChunk = {
  text: string;
  isError: boolean;
  match?: LanguageToolMatch;
};

const getHighlightColor = (category: string) => {
  switch (category) {
    case 'GRAMMAR':
      return 'bg-red-500/30';
    case 'TYPOS':
    case 'CONFUSED_WORDS':
    case 'NONSTANDARD_PHRASES':
      return 'bg-yellow-500/40';
    case 'STYLE':
      return 'bg-blue-500/30';
    default:
      return 'bg-purple-500/30';
  }
};

const SuggestionPopover = ({
  match,
  onApply,
  onClose,
  style,
}: {
  match: LanguageToolMatch;
  onApply: (replacement: string) => void;
  onClose: () => void;
  style: React.CSSProperties;
}) => {
  return (
    <div
      style={style}
      className="absolute z-10 bg-slate-700 rounded-lg shadow-2xl p-4 w-72 border border-slate-600 animate-fade-in"
    >
      <p className="text-sm text-gray-200 mb-3 font-medium">{match.message}</p>
      <div className="flex flex-wrap gap-2">
        {match.replacements.map((replacement, index) => (
          <button
            key={index}
            onClick={() => onApply(replacement.value)}
            className="bg-indigo-600 text-white px-3 py-1 text-sm rounded-md hover:bg-indigo-500 transition-colors shadow-lg"
          >
            {replacement.value}
          </button>
        ))}
      </div>
      <button
        onClick={onClose}
        className="absolute top-2 right-2 text-gray-400 hover:text-white transition-colors"
      >
        &times;
      </button>
    </div>
  );
};

export const TextChecker = () => {
  const [text, setText] = useState('');
  const { matches, loading, error, check, applySuggestion } = useLanguageTool();
  const [activeMatch, setActiveMatch] = useState<LanguageToolMatch | null>(null);
  const [popoverPosition, setPopoverPosition] = useState<{ top: number; left: number } | null>(
    null
  );

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const highlightsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      check(text);
    }, 500);
    return () => clearTimeout(timer);
  }, [text, check]);

  const getTextChunks = useMemo((): TextChunk[] => {
    const sortedMatches = produce(matches, draft => {
      draft.sort((a, b) => a.offset - b.offset);
    });

    const chunks: TextChunk[] = [];
    let lastIndex = 0;

    sortedMatches.forEach(match => {
      if (match.offset > lastIndex) {
        chunks.push({ text: text.substring(lastIndex, match.offset), isError: false });
      }
      chunks.push({
        text: text.substring(match.offset, match.offset + match.length),
        isError: true,
        match,
      });
      lastIndex = match.offset + match.length;
    });

    if (lastIndex < text.length) {
      chunks.push({ text: text.substring(lastIndex), isError: false });
    }

    return chunks;
  }, [text, matches]);

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value);
    setActiveMatch(null);
    setPopoverPosition(null);
  };

  const handleApplySuggestion = (replacement: string) => {
    if (activeMatch) {
      const newText = applySuggestion(text, activeMatch, replacement);
      setText(newText);
      setActiveMatch(null);
      setPopoverPosition(null);
    }
  };

  const handleHighlightClick = (match: LanguageToolMatch, e: React.MouseEvent<HTMLSpanElement>) => {
    e.stopPropagation();
    setActiveMatch(match);

    if (highlightsRef.current) {
      const target = e.currentTarget;
      setPopoverPosition({
        top: target.offsetTop - highlightsRef.current.scrollTop + target.offsetHeight + 5,
        left: target.offsetLeft - highlightsRef.current.scrollLeft,
      });
    }
  };

  const handleScroll = () => {
    if (highlightsRef.current && textareaRef.current) {
      highlightsRef.current.scrollTop = textareaRef.current.scrollTop;
      highlightsRef.current.scrollLeft = textareaRef.current.scrollLeft;
    }
  };

  const handleClosePopover = () => {
    setActiveMatch(null);
    setPopoverPosition(null);
  };

  return (
    <div className="space-y-6" onClick={handleClosePopover}>
      <div className="relative w-full">
        <div
          ref={highlightsRef}
          className="absolute inset-0 p-4 whitespace-pre-wrap text-transparent pointer-events-none overflow-hidden font-mono text-lg leading-relaxed"
        >
          {getTextChunks.map((chunk, index) => (
            <span
              key={index}
              className={
                chunk.isError
                  ? `${getHighlightColor(
                      chunk.match!.rule.category.id
                    )} rounded-sm pointer-events-auto cursor-pointer`
                  : ''
              }
              onClick={chunk.isError ? e => handleHighlightClick(chunk.match!, e) : undefined}
            >
              {chunk.text}
            </span>
          ))}
        </div>
        <textarea
          ref={textareaRef}
          value={text}
          onScroll={handleScroll}
          onChange={handleTextChange}
          spellCheck="false"
          className="w-full p-4 bg-slate-700/50 border border-slate-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-white transition-shadow shadow-sm placeholder-gray-400 caret-white font-mono text-lg leading-relaxed resize-none"
          rows={15}
          placeholder="Start typing or paste your text here..."
        />
        {activeMatch && popoverPosition && (
          <SuggestionPopover
            match={activeMatch}
            onApply={handleApplySuggestion}
            onClose={handleClosePopover}
            style={{ top: `${popoverPosition.top}px`, left: `${popoverPosition.left}px` }}
          />
        )}
      </div>
      {error && (
        <div className="p-4 bg-red-900/30 border border-red-500/30 text-red-300 rounded-lg">
          <strong className="font-semibold">Error:</strong> {error}
        </div>
      )}
      <div className="text-sm text-gray-400 flex items-center justify-between">
        <span>
          Found <span className="font-bold text-white">{matches.length}</span> issues.
        </span>
        {loading && <span className="text-sm text-gray-400">Checking...</span>}
      </div>
    </div>
  );
};
