import type { Metadata } from "next";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ id: string }>;
}): Promise<Metadata> {
  const { id } = await params;

  const title = "Writing Feedback - Writeo";
  const description =
    "Review your essay feedback, scores, and detailed analysis. See where you can improve and track your progress.";

  return {
    title,
    description,
    openGraph: {
      title,
      description,
      type: "website",
      siteName: "Writeo",
    },
    twitter: {
      card: "summary",
      title,
      description,
    },
    robots: {
      index: false, // Don't index individual result pages
      follow: false,
    },
  };
}

export default function ResultsLayout({ children }: { children: React.ReactNode }) {
  return children;
}
