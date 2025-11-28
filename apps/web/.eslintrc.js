module.exports = {
  root: true,
  parser: "@typescript-eslint/parser",
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "module",
    ecmaFeatures: {
      jsx: true,
    },
  },
  plugins: ["@typescript-eslint"],
  extends: [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    // Temporarily disabled due to circular reference issue - will fix in follow-up
    // "next/core-web-vitals",
  ],
  rules: {
    "@typescript-eslint/no-explicit-any": "warn",
    "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
    "react-hooks/exhaustive-deps": "warn",
    "react/no-unescaped-entities": "off",
    "no-console": "off",
  },
  ignorePatterns: [".next", "node_modules", "dist", "build", "*.config.ts", "*.config.js"],
};
