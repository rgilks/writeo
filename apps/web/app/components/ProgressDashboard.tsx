"use client";

import { useState, useEffect, useMemo } from "react";
import { motion, useSpring, useTransform, AnimatePresence } from "framer-motion";
import { useDraftStore } from "@/app/lib/stores/draft-store";
import { AchievementList } from "./AchievementBadge";
import Link from "next/link";

/**
 * AnimatedCounter - Animates a number from 0 to target value
 */
function AnimatedCounter({ value, duration = 1.5 }: { value: number; duration?: number }) {
  const spring = useSpring(0, { stiffness: 50, damping: 30 });
  const display = useTransform(spring, (current) => {
    // Handle decimal values (for average improvement)
    if (value % 1 !== 0) {
      return current.toFixed(1);
    }
    return Math.round(current).toString();
  });

  useEffect(() => {
    spring.set(value);
  }, [spring, value]);

  return <motion.span>{display}</motion.span>;
}

/**
 * Sparkle - Animated sparkle effect for achievements
 */
function Sparkle({
  delay = 0,
  top,
  left,
  right,
}: {
  delay?: number;
  top?: string;
  left?: string;
  right?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0 }}
      animate={{
        opacity: [0, 1, 0],
        scale: [0, 1, 0],
        rotate: [0, 180, 360],
      }}
      transition={{
        duration: 1.5,
        delay,
        repeat: Infinity,
        repeatDelay: 2,
        ease: "easeInOut",
      }}
      style={{
        position: "absolute",
        fontSize: "16px",
        pointerEvents: "none",
        top: top || "10%",
        left: left || undefined,
        right: right || undefined,
        zIndex: 1,
      }}
    >
      ✨
    </motion.div>
  );
}

/**
 * StatCard - Animated stat card with hover effects
 */
function StatCard({
  value,
  label,
  icon,
  color,
  delay = 0,
  gradient,
}: {
  value: number;
  label: string;
  icon?: string;
  color: string;
  delay?: number;
  gradient?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{
        duration: 0.5,
        delay,
        type: "spring",
        stiffness: 200,
        damping: 20,
      }}
      whileHover={{
        scale: 1.05,
        y: -4,
        transition: { duration: 0.2 },
      }}
      style={{
        padding: "var(--spacing-lg)",
        background: gradient ? `linear-gradient(135deg, ${gradient})` : "var(--bg-secondary)",
        borderRadius: "var(--border-radius-lg)",
        textAlign: "center",
        position: "relative",
        overflow: "hidden",
        boxShadow: "var(--shadow-md)",
        border: `2px solid ${color}20`,
        cursor: "pointer",
      }}
      lang="en"
    >
      {/* Decorative gradient overlay */}
      <motion.div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `linear-gradient(135deg, ${color}08 0%, transparent 100%)`,
          pointerEvents: "none",
        }}
      />

      <motion.div
        style={{
          fontSize: "40px",
          fontWeight: 800,
          color: color,
          marginBottom: "var(--spacing-xs)",
          lineHeight: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: "var(--spacing-xs)",
        }}
        lang="en"
        suppressHydrationWarning
      >
        {icon && (
          <motion.span
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{
              delay: delay + 0.2,
              type: "spring",
              stiffness: 200,
              damping: 15,
            }}
            style={{
              display: "inline-block",
            }}
          >
            {icon}
          </motion.span>
        )}
        <AnimatedCounter value={value} />
      </motion.div>

      <div
        style={{
          fontSize: "14px",
          color: "var(--text-secondary)",
          fontWeight: 600,
          letterSpacing: "0.3px",
        }}
        lang="en"
      >
        {label}
      </div>
    </motion.div>
  );
}

/**
 * ProgressDashboard - Shows overall learner progress with celebratory animations
 */
export function ProgressDashboard() {
  // Fix hydration mismatch: defer all localStorage-dependent rendering until after client mount
  const [mounted, setMounted] = useState(false);
  // Track store hydration state - data may not be available until store is hydrated
  const [isHydrated, setIsHydrated] = useState(() => useDraftStore.persist.hasHydrated());

  useEffect(() => {
    setMounted(true);
  }, []);

  // Listen for hydration completion
  useEffect(() => {
    if (useDraftStore.persist.hasHydrated()) {
      setIsHydrated(true);
      return;
    }

    const unsubscribe = useDraftStore.persist.onFinishHydration(() => {
      setIsHydrated(true);
    });

    return () => {
      unsubscribe();
    };
  }, []);

  // Use selectors to subscribe only to needed state slices
  // This prevents re-renders when unrelated state changes
  // Separate selectors are more efficient - Zustand handles them optimally
  const drafts = useDraftStore((state) => state.drafts);
  const progress = useDraftStore((state) => state.progress);
  const streak = useDraftStore((state) => state.streak);
  const achievements = useDraftStore((state) => state.achievements);

  // Use computed selectors from store for better performance
  // These are stable functions, so we can call them directly
  const totalDrafts = useDraftStore((state) => state.getTotalDrafts());
  const totalWritings = useDraftStore((state) => state.getTotalWritings());
  const averageImprovement = useDraftStore((state) => state.getAverageImprovement());

  // Calculate practice streak (simplified - based on drafts in last 7 days)
  // Use memoization to avoid recalculating on every render
  const practiceStreak = useMemo(() => {
    if (!mounted) return 0;
    const allDrafts = Object.values(drafts).flat();
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
    const recentDrafts = allDrafts.filter((d) => new Date(d.timestamp) >= sevenDaysAgo);
    return recentDrafts.length > 0 ? Math.min(7, recentDrafts.length) : 0;
  }, [mounted, drafts]);

  return (
    <motion.div
      className="card"
      lang="en"
      data-testid="progress-dashboard"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <motion.h2
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.1, duration: 0.5 }}
        style={{
          fontSize: "24px",
          fontWeight: 700,
          marginBottom: "var(--spacing-lg)",
          color: "var(--text-primary)",
          display: "flex",
          alignItems: "center",
          gap: "var(--spacing-sm)",
        }}
        lang="en"
      >
        <motion.span
          animate={{ rotate: [0, 10, -10, 0] }}
          transition={{
            duration: 2,
            repeat: Infinity,
            repeatDelay: 3,
            ease: "easeInOut",
          }}
        >
          📊
        </motion.span>
        Your Progress
      </motion.h2>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: "var(--spacing-lg)",
          marginBottom: "var(--spacing-xl)",
        }}
      >
        {mounted && isHydrated && (
          <>
            <StatCard
              value={totalWritings}
              label="Writings Completed"
              color="var(--primary-color)"
              delay={0.1}
              gradient="rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.05)"
            />

            <StatCard
              value={totalDrafts}
              label="Total Drafts"
              color="var(--primary-color)"
              delay={0.2}
              gradient="rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.05)"
            />

            {averageImprovement > 0 && (
              <StatCard
                value={averageImprovement}
                label="Avg. Improvement"
                icon="📈"
                color="var(--secondary-accent)"
                delay={0.3}
                gradient="rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05)"
              />
            )}

            {streak.currentStreak > 0 && (
              <StatCard
                value={streak.currentStreak}
                label="Day Streak"
                icon="🔥"
                color="var(--warm-accent)"
                delay={0.4}
                gradient="rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.05)"
              />
            )}
          </>
        )}
      </div>

      {/* Achievements Section - Only render after client-side mount and hydration to avoid hydration mismatch */}
      <AnimatePresence>
        {mounted && isHydrated && achievements.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.5 }}
            style={{
              marginTop: "var(--spacing-xl)",
              paddingTop: "var(--spacing-xl)",
              borderTop: "2px solid var(--border-color)",
              position: "relative",
            }}
            lang="en"
          >
            <motion.h3
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.6, type: "spring", stiffness: 200 }}
              style={{
                fontSize: "22px",
                fontWeight: 700,
                marginBottom: "var(--spacing-lg)",
                color: "var(--text-primary)",
                display: "flex",
                alignItems: "center",
                gap: "var(--spacing-sm)",
              }}
              lang="en"
            >
              <motion.span
                animate={{
                  rotate: [0, -15, 15, 0],
                  scale: [1, 1.2, 1],
                }}
                transition={{
                  duration: 1.5,
                  repeat: Infinity,
                  repeatDelay: 2,
                  ease: "easeInOut",
                }}
              >
                🏆
              </motion.span>
              Your Achievements
              <motion.span
                style={{ fontSize: "16px", fontWeight: 400, color: "var(--text-secondary)" }}
              >
                ({achievements.length})
              </motion.span>
            </motion.h3>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7, duration: 0.5 }}
            >
              <AchievementList achievements={achievements} maxDisplay={6} />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {mounted && isHydrated && totalWritings === 0 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.5 }}
            style={{
              padding: "var(--spacing-xl)",
              background:
                "linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(59, 130, 246, 0.02) 100%)",
              border: "2px dashed rgba(59, 130, 246, 0.3)",
              borderRadius: "var(--border-radius-lg)",
              textAlign: "center",
              marginTop: "var(--spacing-lg)",
            }}
            lang="en"
          >
            <motion.div
              animate={{
                rotate: [0, 10, -10, 0],
                scale: [1, 1.1, 1],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                repeatDelay: 2,
                ease: "easeInOut",
              }}
              style={{
                fontSize: "64px",
                marginBottom: "var(--spacing-md)",
              }}
              lang="en"
            >
              📊
            </motion.div>
            <p
              style={{
                fontSize: "18px",
                color: "var(--text-primary)",
                marginBottom: "var(--spacing-sm)",
                fontWeight: 700,
              }}
              lang="en"
            >
              Your Progress Dashboard
            </p>
            <p
              style={{
                fontSize: "16px",
                color: "var(--text-secondary)",
                marginBottom: "var(--spacing-lg)",
                lineHeight: "1.6",
              }}
              lang="en"
            >
              You haven't written anything yet. Start practicing to see your progress, streaks, and
              achievements here!
            </p>
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Link href="/write/1" className="btn btn-primary" lang="en">
                Start Writing →
              </Link>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
