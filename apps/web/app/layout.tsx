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
  return (
    <html lang="en">
      <body>
        <main>{children}</main>
        <Footer />
      </body>
    </html>
  );
}
