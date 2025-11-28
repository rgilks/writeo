import type { Variants, HTMLMotionProps } from "framer-motion";

export const fadeInUp: Pick<HTMLMotionProps<"div">, "initial" | "animate" | "transition"> = {
  initial: { opacity: 0, y: -10 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, ease: "easeOut" },
};

export const fadeInUpDelayed = (
  delay: number,
): Pick<HTMLMotionProps<"div">, "initial" | "animate" | "transition"> => ({
  initial: { opacity: 0, y: -10 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6, delay, ease: "easeOut" },
});

export const cardVariants: Variants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
      ease: "easeOut",
    },
  },
};

export const staggerContainer = {
  visible: {
    transition: {
      staggerChildren: 0.05,
      delayChildren: 1.0,
    },
  },
};
