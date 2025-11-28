import type { Metadata } from "next";
import { TASK_DATA } from "@/app/lib/constants/tasks";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ id: string }>;
}): Promise<Metadata> {
  const { id } = await params;
  const isCustom = id === "custom";
  const task = isCustom ? null : TASK_DATA[id];

  const title = isCustom
    ? "Custom Question - Writeo"
    : task
      ? `${task.title} - Writeo`
      : "Writing Practice - Writeo";

  const description = isCustom
    ? "Write your own question or practice free writing without a specific prompt. Get detailed feedback on your essays."
    : task
      ? `${task.description} Practice your writing and get detailed feedback.`
      : "Practice your writing skills and get detailed feedback on your essays.";

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
  };
}

export default function WriteLayout({ children }: { children: React.ReactNode }) {
  return children;
}
