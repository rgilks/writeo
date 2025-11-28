import { FlatCompat } from "@eslint/eslintrc";
import js from "@eslint/js";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";
import react from "eslint-plugin-react";
import jsxA11y from "eslint-plugin-jsx-a11y";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: js.configs.recommended,
});

export default [
  js.configs.recommended,
  ...tseslint.configs.recommended,
  // Note: next/core-web-vitals has compatibility issues with flat config
  // Using manual Next.js rules instead
  ...compat.extends("prettier"),
  {
    files: ["**/*.{ts,tsx}"],
    plugins: {
      "react-hooks": reactHooks,
      react: react,
      "jsx-a11y": jsxA11y,
    },
    rules: {
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/no-unused-vars": [
        "warn",
        { argsIgnorePattern: "^_" },
      ],
      "react-hooks/exhaustive-deps": "warn",
      "react/no-unescaped-entities": "off",
      "no-console": "off",
    },
  },
  {
    ignores: [
      ".next/**",
      ".open-next/**",
      "node_modules/**",
      "dist/**",
      "build/**",
      "*.config.ts",
      "*.config.js",
      "*.config.mjs",
      "next-env.d.ts", // Next.js generated file
    ],
  },
];

