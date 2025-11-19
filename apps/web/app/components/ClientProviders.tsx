"use client";

import { ModeProvider } from "@/app/lib/mode-context";

export function ClientProviders({ children }: { children: React.ReactNode }) {
  return <ModeProvider>{children}</ModeProvider>;
}
