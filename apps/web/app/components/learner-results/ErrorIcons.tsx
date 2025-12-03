import React from "react";

export function getErrorIcon(type: string) {
  const normalizedType = type.toLowerCase();

  if (normalizedType.includes("grammar") || normalizedType.includes("syntax")) {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-blue-500"
      >
        <path d="M12 19l7-7 3 3-7 7-3-3z" />
        <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z" />
        <path d="M2 2l7.586 7.586" />
        <circle cx="11" cy="11" r="2" />
      </svg>
    );
  }

  if (normalizedType.includes("spelling") || normalizedType.includes("typo")) {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-red-500"
      >
        <path d="M4 7V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v3" />
        <path d="M9 5h4a2 2 0 0 1 2 2v2h-3" />
        <path d="M12 9V7a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v3" />
        <path d="M22 13h-6M16 13v8M22 13v8" />
        <path d="M5 17h3" />
        <path d="M16 13h-3" />
        <path d="M13 13h-3" />
      </svg>
    );
  }

  if (normalizedType.includes("punctuation")) {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-yellow-500"
      >
        <path d="M10 5a2 2 0 1 1 4 0a7 7 0 0 1 4 6v3a4 4 0 0 1-4 4v0a4 4 0 0 1-4-4v-3a6 6 0 0 1 4-6" />
        <path d="M8 21h8" />
        <circle cx="12" cy="17" r="1" />
      </svg>
    );
  }

  if (
    normalizedType.includes("article") ||
    normalizedType.includes("preposition") ||
    normalizedType.includes("word")
  ) {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-green-500"
      >
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
      </svg>
    );
  }

  // Default icon
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="text-gray-500"
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
  );
}
