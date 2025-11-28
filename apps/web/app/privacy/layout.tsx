import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Privacy & Data Ethics",
  description:
    "Learn how Writeo handles your writing and protects your privacy. We respect your data and give you control over how it's stored.",
  robots: {
    index: true,
    follow: true,
  },
  openGraph: {
    title: "Privacy & Data Ethics - Writeo",
    description: "Learn how Writeo handles your writing and protects your privacy.",
    type: "website",
  },
};

export default function PrivacyLayout({ children }: { children: React.ReactNode }) {
  return children;
}
