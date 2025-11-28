"use client";

import { useEffect, useRef } from "react";

interface LiveRegionProps {
  message: string | null;
  priority?: "polite" | "assertive";
  id?: string;
}

/**
 * LiveRegion component for screen reader announcements
 * Use for dynamic content updates that users need to be aware of
 */
export function LiveRegion({ message, priority = "polite", id = "live-region" }: LiveRegionProps) {
  const regionRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (message && regionRef.current) {
      // Clear and set message to trigger announcement
      regionRef.current.textContent = "";
      // Use setTimeout to ensure the clear is processed first
      setTimeout(() => {
        if (regionRef.current) {
          regionRef.current.textContent = message;
        }
      }, 100);
    }
  }, [message]);

  return (
    <div
      id={id}
      ref={regionRef}
      role="status"
      aria-live={priority}
      aria-atomic="true"
      style={{
        position: "absolute",
        left: "-10000px",
        width: "1px",
        height: "1px",
        overflow: "hidden",
      }}
    />
  );
}
