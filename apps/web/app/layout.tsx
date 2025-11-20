import type { Metadata } from "next";
import "./globals.css";
import { Footer } from "./components/Footer";

export const metadata: Metadata = {
  title: "Writeo - Essay Scoring",
  description: "Modern essay scoring system for essay assessment",
  icons: {
    icon: "/icon.svg",
    shortcut: "/icon.svg",
    apple: "/icon.svg",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // Wrap in try/catch to prevent layout errors from causing Server Components render errors
  try {
    return (
      <html lang="en">
        <body className="flex min-h-screen flex-col">
          <main className="flex-1">{children}</main>
          <Footer />
        </body>
      </html>
    );
  } catch (error) {
    // This should never happen, but if it does, return a minimal layout
    console.error("Layout error:", error);
    return (
      <html lang="en">
        <body>
          <div style={{ padding: "20px", textAlign: "center" }}>
            <h1>An error occurred</h1>
            <p>Please refresh the page or contact support.</p>
          </div>
        </body>
      </html>
    );
  }
}
