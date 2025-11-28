import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Accessibility Statement",
  description:
    "Writeo's commitment to digital accessibility. Learn about our accessibility features and how we're working to make our platform accessible to everyone.",
  robots: {
    index: true,
    follow: true,
  },
  openGraph: {
    title: "Accessibility Statement - Writeo",
    description: "Writeo's commitment to digital accessibility and WCAG compliance.",
    type: "website",
  },
};

export default function AccessibilityLayout({ children }: { children: React.ReactNode }) {
  return children;
}
