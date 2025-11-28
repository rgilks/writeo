import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Terms of Service",
  description:
    "Read Writeo's terms of service. Understand the rules and guidelines for using our essay scoring platform.",
  robots: {
    index: true,
    follow: true,
  },
  openGraph: {
    title: "Terms of Service - Writeo",
    description: "Read Writeo's terms of service and usage guidelines.",
    type: "website",
  },
};

export default function TermsLayout({ children }: { children: React.ReactNode }) {
  return children;
}
