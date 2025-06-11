import type { Metadata } from 'next';
import { Inter, Exo_2 } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });
const exo2 = Exo_2({ subsets: ['latin'], variable: '--font-exo2' });

export const metadata: Metadata = {
  title: 'Writeo',
  description: 'AI-powered writing assistant with LanguageTool integration',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} ${exo2.variable}`}>{children}</body>
    </html>
  );
}
