"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import type { Achievement } from "@/app/lib/stores/draft-store";

/**
 * Sparkle - Enhanced sparkle effect that randomly appears across the entire card area
 */
function Sparkle() {
  const sparkleVariants = ["✨", "⭐"];
  // Randomly select a sparkle variant, position, size, and timing (stable per instance)
  const [sparkleData] = useState(() => {
    const top = Math.random() * 90 + 5; // 5% to 95%
    const left = Math.random() * 90 + 5; // 5% to 95%
    const size = Math.random() * 6 + 10; // 10px to 16px
    const initialDelay = Math.random() * 2; // 0 to 2 seconds
    const duration = Math.random() * 1.5 + 2; // 2 to 3.5 seconds
    const repeatDelay = Math.random() * 3 + 1; // 1 to 4 seconds
    const sparkle = sparkleVariants[Math.floor(Math.random() * sparkleVariants.length)];
    const xOffset = (Math.random() - 0.5) * 12; // -6px to 6px horizontal drift
    const yOffset = (Math.random() - 0.5) * 12; // -6px to 6px vertical drift

    return {
      sparkle,
      top: `${top}%`,
      left: `${left}%`,
      size,
      initialDelay,
      duration,
      repeatDelay,
      xOffset,
      yOffset,
    };
  });

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0 }}
      animate={{
        opacity: [0, 0.6, 0.5, 0.4, 0],
        scale: [0, 1.2, 1.1, 1, 0.7],
        rotate: [0, 90, 180, 270, 360],
        x: [0, sparkleData.xOffset, -sparkleData.xOffset, 0],
        y: [0, sparkleData.yOffset, -sparkleData.yOffset, 0],
      }}
      transition={{
        duration: sparkleData.duration,
        delay: sparkleData.initialDelay,
        repeat: Infinity,
        repeatDelay: sparkleData.repeatDelay,
        ease: "easeInOut",
      }}
      style={{
        position: "absolute",
        fontSize: `${sparkleData.size}px`,
        pointerEvents: "none",
        top: sparkleData.top,
        left: sparkleData.left,
        transform: "translate(-50%, -50%)",
        zIndex: 10,
        filter: "drop-shadow(0 0 2px rgba(255, 215, 0, 0.4))",
      }}
    >
      {sparkleData.sparkle}
    </motion.div>
  );
}

interface AchievementBadgeProps {
  achievement: Achievement;
  size?: "small" | "medium" | "large";
  showDescription?: boolean;
  animated?: boolean;
}

/**
 * AchievementBadge - Displays an achievement badge with icon and name
 */
export function AchievementBadge({
  achievement,
  size = "medium",
  showDescription = false,
  animated = false,
}: AchievementBadgeProps) {
  const sizeStyles = {
    small: { icon: "24px", fontSize: "12px", padding: "var(--spacing-xs) var(--spacing-sm)" },
    medium: { icon: "32px", fontSize: "14px", padding: "var(--spacing-sm) var(--spacing-md)" },
    large: { icon: "48px", fontSize: "16px", padding: "var(--spacing-md) var(--spacing-lg)" },
  };

  const style = sizeStyles[size];

  // Fixed dimensions for consistent sizing
  const dimensions = {
    small: { width: "90px", height: "90px" },
    medium: { width: "140px", height: "140px" },
    large: { width: "180px", height: "180px" },
  };

  const dims = dimensions[size];

  const badgeContent = (
    <motion.div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: "var(--spacing-xs)",
        padding: style.padding,
        background:
          "linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(102, 126, 234, 0.05) 100%)",
        border: "2px solid rgba(102, 126, 234, 0.4)",
        borderRadius: "var(--border-radius-lg)",
        width: dims.width,
        height: dims.height,
        boxShadow: "var(--shadow-sm)",
        position: "relative",
        overflow: "hidden",
        cursor: "pointer",
      }}
      lang="en"
      whileHover={{
        scale: 1.08,
        y: -6,
        transition: {
          duration: 0.2,
          type: "spring",
          stiffness: 400,
          damping: 25,
        },
      }}
      whileTap={{ scale: 0.95 }}
    >
      {/* Animated gradient overlay on hover */}
      <motion.div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background:
            "linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(147, 51, 234, 0.2) 50%, rgba(236, 72, 153, 0.2) 100%)",
          opacity: 0,
          pointerEvents: "none",
        }}
        whileHover={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      />

      {/* Shine effect overlay */}
      <motion.div
        style={{
          position: "absolute",
          top: 0,
          left: "-100%",
          width: "100%",
          height: "100%",
          background: "linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.5), transparent)",
          pointerEvents: "none",
        }}
        animate={{
          left: ["-100%", "200%"],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          repeatDelay: 3,
          ease: "easeInOut",
        }}
        whileHover={{
          left: ["-100%", "200%"],
          transition: {
            duration: 0.8,
            repeat: Infinity,
            repeatDelay: 0.5,
          },
        }}
      />

      {/* Glow effect on hover */}
      <motion.div
        style={{
          position: "absolute",
          inset: -4,
          borderRadius: "var(--border-radius-lg)",
          background:
            "linear-gradient(135deg, rgba(102, 126, 234, 0.6), rgba(147, 51, 234, 0.6), rgba(236, 72, 153, 0.6))",
          opacity: 0,
          filter: "blur(8px)",
          pointerEvents: "none",
          zIndex: -1,
        }}
        whileHover={{
          opacity: [0, 0.8, 0.6],
          scale: [1, 1.1, 1],
        }}
        transition={{
          duration: 1,
          repeat: Infinity,
          repeatType: "reverse",
        }}
      />

      {/* Sparkles randomly appearing across the entire card */}
      {Array.from({ length: 5 }).map((_, i) => (
        <Sparkle key={i} />
      ))}

      {/* Content wrapper with relative positioning */}
      <div
        style={{
          position: "relative",
          zIndex: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "var(--spacing-xs)",
          width: "100%",
          height: "100%",
        }}
      >
        <motion.div
          style={{
            fontSize: style.icon,
            lineHeight: 1,
          }}
          lang="en"
          whileHover={{
            scale: 1.15,
          }}
          transition={{
            duration: 0.2,
            type: "spring",
            stiffness: 400,
          }}
        >
          {achievement.icon}
        </motion.div>
        <div
          style={{
            fontSize: style.fontSize,
            fontWeight: 600,
            textAlign: "center",
            color: "var(--text-primary)",
            lineHeight: 1.2,
          }}
          lang="en"
          suppressHydrationWarning
        >
          {achievement.name}
        </div>
        {showDescription && (
          <div
            style={{
              fontSize: "11px",
              color: "var(--text-secondary)",
              textAlign: "center",
              lineHeight: "1.3",
              padding: "0 var(--spacing-xs)",
            }}
            lang="en"
            suppressHydrationWarning
          >
            {achievement.description}
          </div>
        )}
      </div>
    </motion.div>
  );

  if (animated) {
    return (
      <motion.div
        initial={{ scale: 0, rotate: -180 }}
        animate={{ scale: 1, rotate: 0 }}
        transition={{
          type: "spring",
          stiffness: 200,
          damping: 15,
          duration: 0.5,
        }}
        lang="en"
      >
        {badgeContent}
      </motion.div>
    );
  }

  return <div lang="en">{badgeContent}</div>;
}

interface AchievementListProps {
  achievements: Achievement[];
  maxDisplay?: number;
}

/**
 * AchievementList - Displays a list of achievement badges with staggered animations
 */
export function AchievementList({ achievements, maxDisplay }: AchievementListProps) {
  const displayAchievements = maxDisplay ? achievements.slice(0, maxDisplay) : achievements;

  if (achievements.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        style={{
          padding: "var(--spacing-lg)",
          textAlign: "center",
          color: "var(--text-secondary)",
        }}
        lang="en"
      >
        <p lang="en">No achievements yet. Keep practicing to unlock badges!</p>
      </motion.div>
    );
  }

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
        gap: "var(--spacing-lg)",
        justifyContent: "center",
        position: "relative",
      }}
      lang="en"
    >
      {displayAchievements.map((achievement, index) => (
        <motion.div
          key={achievement.id}
          initial={{ opacity: 0, scale: 0.5, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{
            delay: index * 0.1,
            type: "spring",
            stiffness: 200,
            damping: 20,
          }}
          style={{ position: "relative" }}
        >
          <AchievementBadge
            achievement={achievement}
            size="medium"
            showDescription={true}
            animated={true}
          />
        </motion.div>
      ))}
    </div>
  );
}
