import Link from "next/link";
import { motion } from "framer-motion";
import { cardVariants } from "@/app/lib/constants/animations";

interface TaskCardProps {
  title: string;
  description: string;
  icon: string;
  href: string;
}

export function TaskCard({ title, description, icon, href }: TaskCardProps) {
  // Extract task ID from href for test ID
  const taskId = href.replace("/write/", "");
  return (
    <motion.div variants={cardVariants}>
      <Link
        href={href}
        style={{ textDecoration: "none", display: "block" }}
        className="task-card-link"
        data-testid={`task-card-link-${taskId}`}
      >
        <div className="task-card" data-testid="task-card">
          <div className="task-header">
            <div className="task-title">{title}</div>
            <div className="task-icon">{icon}</div>
          </div>
          <div className="task-description">{description}</div>
          <div className="btn btn-primary" style={{ width: "100%" }}>
            Start Writing →
          </div>
        </div>
      </Link>
    </motion.div>
  );
}
