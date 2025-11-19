import type { Metadata } from "next";
import "./globals.css";
import { ClientProviders } from "./components/ClientProviders";
import { Footer } from "./components/Footer";

export const metadata: Metadata = {
  title: "Writeo - Essay Scoring",
  description: "Modern essay scoring system for essay assessment",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  // Wrap in try/catch to prevent layout errors from causing Server Components render errors
  try {
    return (
      <html lang="en">
        <body className="flex min-h-screen flex-col">
          <ClientProviders>
            <main className="flex-1">{children}</main>
            <Footer />
          </ClientProviders>
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
