"use client";

import { useState, useEffect, useMemo } from "react";
import { motion, useSpring, useTransform, AnimatePresence } from "framer-motion";
import { useDraftStore } from "@/app/lib/stores/draft-store";
import { useStoreHydration } from "@/app/hooks/useStoreHydration";
import { AchievementList } from "./AchievementBadge";
import Link from "next/link";

const SPRING_CONFIG = { stiffness: 50, damping: 30 };
const STAT_CARD_ANIMATION = {
  duration: 0.5,
  type: "spring" as const,
  stiffness: 200,
  damping: 20,
};
const ICON_ANIMATION_DELAY_OFFSET = 0.2;
const STAT_CARD_DELAYS = {
  WRITINGS: 0.1,
  DRAFTS: 0.2,
  IMPROVEMENT: 0.3,
  STREAK: 0.4,
} as const;
const PRIMARY_GRADIENT = "rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.05)";
const SUCCESS_GRADIENT = "rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05)";
const WARM_GRADIENT = "rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.05)";
const MAX_DISPLAY_ACHIEVEMENTS = 6;

interface AnimatedCounterProps {
  value: number;
}

function AnimatedCounter({ value }: AnimatedCounterProps) {
  const spring = useSpring(0, SPRING_CONFIG);
  const display = useTransform(spring, (current) => {
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

interface StatCardProps {
  value: number;
  label: string;
  icon?: string;
  color: string;
  delay?: number;
  gradient?: string;
}

const ABSOLUTE_FILL = {
  position: "absolute" as const,
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  pointerEvents: "none" as const,
};

function StatCardSkeleton() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.15 }}
      style={{
        padding: "var(--spacing-lg)",
        background: "var(--bg-secondary)",
        borderRadius: "var(--border-radius-lg)",
        textAlign: "center",
        position: "relative",
        overflow: "hidden",
        boxShadow: "var(--shadow-md)",
        border: "2px solid var(--border-color)",
      }}
    >
      <div
        style={{
          height: "40px",
          marginBottom: "var(--spacing-xs)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            width: "60px",
            height: "40px",
            backgroundColor: "var(--bg-tertiary)",
            borderRadius: "var(--border-radius)",
            animation: "pulse 1.5s ease-in-out infinite",
          }}
          aria-hidden="true"
        />
      </div>
      <div
        style={{
          height: "14px",
          width: "80px",
          margin: "0 auto",
          backgroundColor: "var(--bg-tertiary)",
          borderRadius: "var(--border-radius)",
          animation: "pulse 1.5s ease-in-out infinite",
        }}
        aria-hidden="true"
      />
    </motion.div>
  );
}

function StatCard({ value, label, icon, color, delay = 0, gradient }: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{
        ...STAT_CARD_ANIMATION,
        delay,
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
          ...ABSOLUTE_FILL,
          background: `linear-gradient(135deg, ${color}08 0%, transparent 100%)`,
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
        suppressHydrationWarning
      >
        {icon && (
          <motion.span
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{
              delay: delay + ICON_ANIMATION_DELAY_OFFSET,
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
      >
        {label}
      </div>
    </motion.div>
  );
}

function useClientHydration() {
  const [mounted, setMounted] = useState(false);
  const isHydrated = useStoreHydration(useDraftStore);

  useEffect(() => {
    setMounted(true);
  }, []);

  return { mounted, isHydrated, isReady: mounted && isHydrated };
}

interface StatCardData {
  value: number;
  label: string;
  icon?: string;
  color: string;
  delay: number;
  gradient: string;
  condition?: boolean;
}

export function ProgressDashboard() {
  const { isReady } = useClientHydration();

  const streak = useDraftStore((state) => state.streak);
  const achievements = useDraftStore((state) => state.achievements);
  const totalDrafts = useDraftStore((state) => state.getTotalDrafts());
  const totalWritings = useDraftStore((state) => state.getTotalWritings());
  const averageImprovement = useDraftStore((state) => state.getAverageImprovement());

  const statCards = useMemo<StatCardData[]>(
    () => [
      {
        value: totalWritings,
        label: "Writings Completed",
        color: "var(--primary-color)",
        delay: STAT_CARD_DELAYS.WRITINGS,
        gradient: PRIMARY_GRADIENT,
      },
      {
        value: totalDrafts,
        label: "Total Drafts",
        color: "var(--primary-color)",
        delay: STAT_CARD_DELAYS.DRAFTS,
        gradient: PRIMARY_GRADIENT,
      },
      {
        value: averageImprovement,
        label: "Avg. Improvement",
        icon: "📈",
        color: "var(--secondary-accent)",
        delay: STAT_CARD_DELAYS.IMPROVEMENT,
        gradient: SUCCESS_GRADIENT,
        condition: averageImprovement > 0,
      },
      {
        value: streak.currentStreak,
        label: "Day Streak",
        icon: "🔥",
        color: "var(--warm-accent)",
        delay: STAT_CARD_DELAYS.STREAK,
        gradient: WARM_GRADIENT,
        condition: streak.currentStreak > 0,
      },
    ],
    [totalWritings, totalDrafts, averageImprovement, streak.currentStreak],
  );

  const showAchievements = isReady && achievements.length > 0;

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
        <AnimatePresence mode="wait">
          {!isReady ? (
            <motion.div
              key="skeletons"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              style={{ display: "contents" }}
            >
              {[1, 2, 3, 4].map((i) => (
                <StatCardSkeleton key={`skeleton-${i}`} />
              ))}
            </motion.div>
          ) : (
            <motion.div
              key="cards"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.15 }}
              style={{ display: "contents" }}
            >
              {statCards
                .filter((card) => card.condition !== false)
                .map((card) => (
                  <StatCard
                    key={card.label}
                    value={card.value}
                    label={card.label}
                    icon={card.icon}
                    color={card.color}
                    delay={Math.max(0, card.delay - 0.1)}
                    gradient={card.gradient}
                  />
                ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Achievements Section - Always render container to prevent layout shift */}
      <motion.div
        layout
        animate={{
          opacity: showAchievements ? 1 : 0,
        }}
        transition={{
          opacity: { duration: 0.5, delay: 0.5 },
          layout: { duration: 0.5, ease: "easeInOut" },
        }}
        style={{
          marginTop: showAchievements ? "var(--spacing-xl)" : 0,
          paddingTop: showAchievements ? "var(--spacing-xl)" : 0,
          borderTop: showAchievements ? "2px solid var(--border-color)" : "none",
          position: "relative",
          minHeight: 0,
        }}
      >
        {showAchievements && (
          <>
            <motion.h3
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
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
              <AchievementList achievements={achievements} maxDisplay={MAX_DISPLAY_ACHIEVEMENTS} />
            </motion.div>
          </>
        )}
      </motion.div>

      <AnimatePresence>
        {isReady && totalWritings === 0 && (
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
            >
              You haven't written anything yet. Start practicing to see your progress, streaks, and
              achievements here!
            </p>
            <Link href="/write/1" className="btn btn-primary" style={{ display: "inline-block" }}>
              <motion.span
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                style={{ display: "block" }}
              >
                Start Writing →
              </motion.span>
            </Link>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
