import type { Metadata } from "next";
import "./globals.css";
import { Footer } from "./components/Footer";
import { SkipLink } from "./components/SkipLink";

export const metadata: Metadata = {
  title: {
    default: "Writeo - Essay Scoring",
    template: "%s | Writeo",
  },
  description:
    "Practice your writing skills and get detailed AI-powered feedback on your essays. Improve your writing with instant scoring and personalized feedback.",
  keywords: [
    "essay scoring",
    "writing practice",
    "essay feedback",
    "writing assessment",
    "IELTS writing",
    "academic writing",
  ],
  authors: [{ name: "Writeo" }],
  creator: "Writeo",
  publisher: "Writeo",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL || "https://writeo.app"),
  alternates: {
    canonical: "/",
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "/",
    siteName: "Writeo",
    title: "Writeo - Essay Scoring",
    description:
      "Practice your writing skills and get detailed AI-powered feedback on your essays.",
  },
  twitter: {
    card: "summary_large_image",
    title: "Writeo - Essay Scoring",
    description:
      "Practice your writing skills and get detailed AI-powered feedback on your essays.",
  },
  icons: {
    icon: "/icon.svg",
    shortcut: "/icon.svg",
    apple: "/icon.svg",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <SkipLink />
        <main id="main-content">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
