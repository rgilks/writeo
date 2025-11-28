"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ProgressDashboard } from "@/app/components/ProgressDashboard";
import { TaskCard } from "@/app/components/TaskCard";
import { TASKS } from "@/app/lib/constants/tasks";
import { fadeInUp, fadeInUpDelayed, staggerContainer } from "@/app/lib/constants/animations";

export default function HomePage() {
  return (
    <>
      <header className="header">
        <div className="header-content">
          <div className="logo-group">
            <Link href="/" className="logo">
              Writeo
            </Link>
          </div>
          <nav className="header-actions" aria-label="Primary navigation"></nav>
        </div>
      </header>

      <div className="container">
        <motion.div style={{ marginBottom: "var(--spacing-3xl)" }} {...fadeInUp}>
          <motion.h1 className="hero-title" {...fadeInUpDelayed(0.1)}>
            Practice Writing
          </motion.h1>
          <motion.p className="hero-subtitle" {...fadeInUpDelayed(0.2)}>
            Choose a task to practice your writing. Get detailed feedback on your essays and improve
            with each draft.
          </motion.p>
        </motion.div>

        <div style={{ marginBottom: "var(--spacing-xl)" }}>
          <ProgressDashboard />
        </div>

        <motion.div
          className="grid"
          layout
          transition={{ duration: 0.5, ease: "easeInOut" }}
          initial="hidden"
          animate="visible"
          variants={staggerContainer}
        >
          <TaskCard
            title="Custom Question"
            description="Write your own question or practice free writing without a specific prompt."
            icon="✍️"
            href="/write/custom"
          />
          {TASKS.map((task) => (
            <TaskCard
              key={task.id}
              title={task.title}
              description={task.description}
              icon={task.icon}
              href={`/write/${task.id}`}
            />
          ))}
        </motion.div>
      </div>
    </>
  );
}
