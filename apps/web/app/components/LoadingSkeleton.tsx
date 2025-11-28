"use client";

interface SkeletonProps {
  width?: string;
  height?: string;
  className?: string;
  rounded?: boolean;
}

function Skeleton({
  width = "100%",
  height = "1rem",
  className = "",
  rounded = true,
  style = {},
}: SkeletonProps & { style?: React.CSSProperties }) {
  return (
    <div
      className={className}
      style={{
        width,
        height,
        backgroundColor: "var(--bg-tertiary)",
        borderRadius: rounded ? "var(--border-radius)" : "0",
        animation: "pulse 1.5s ease-in-out infinite",
        ...style,
      }}
      aria-hidden="true"
    />
  );
}

export function LoadingSkeleton() {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "var(--spacing-md)",
        padding: "var(--spacing-lg)",
      }}
    >
      <Skeleton height="2rem" width="60%" />
      <Skeleton height="1rem" />
      <Skeleton height="1rem" width="80%" />
    </div>
  );
}

export function CardSkeleton() {
  return (
    <div className="card" style={{ padding: "var(--spacing-xl)" }}>
      <Skeleton height="1.5rem" width="40%" style={{ marginBottom: "var(--spacing-md)" }} />
      <Skeleton height="1rem" style={{ marginBottom: "var(--spacing-sm)" }} />
      <Skeleton height="1rem" width="90%" style={{ marginBottom: "var(--spacing-sm)" }} />
      <Skeleton height="1rem" width="75%" />
    </div>
  );
}

export function ResultsLoadingSkeleton() {
  return (
    <div className="container" style={{ minHeight: "calc(100vh - 200px)" }}>
      <div style={{ marginBottom: "var(--spacing-xl)" }}>
        <Skeleton height="2rem" width="50%" style={{ marginBottom: "var(--spacing-sm)" }} />
        <Skeleton height="1.25rem" width="70%" />
      </div>

      <div
        className="card"
        style={{ marginBottom: "var(--spacing-lg)", padding: "var(--spacing-xl)" }}
      >
        <div
          style={{
            display: "flex",
            gap: "var(--spacing-lg)",
            alignItems: "center",
            marginBottom: "var(--spacing-lg)",
          }}
        >
          <Skeleton height="4rem" width="4rem" rounded />
          <div style={{ flex: 1 }}>
            <Skeleton height="1.5rem" width="60%" style={{ marginBottom: "var(--spacing-sm)" }} />
            <Skeleton height="1rem" width="80%" style={{}} />
          </div>
        </div>
        <Skeleton height="1rem" style={{ marginBottom: "var(--spacing-sm)" }} />
        <Skeleton height="1rem" width="90%" style={{}} />
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: "var(--spacing-md)",
          marginTop: "var(--spacing-lg)",
        }}
      >
        {[1, 2, 3, 4].map((i) => (
          <CardSkeleton key={i} />
        ))}
      </div>
    </div>
  );
}

export function WritePageSkeleton() {
  return (
    <div className="container">
      <div style={{ marginBottom: "var(--spacing-xl)" }}>
        <Skeleton height="2rem" width="40%" style={{ marginBottom: "var(--spacing-sm)" }} />
        <Skeleton height="1.25rem" width="60%" />
      </div>

      <div className="writing-container">
        <div className="card question-card" style={{ padding: "var(--spacing-xl)" }}>
          <Skeleton height="1.5rem" width="30%" style={{ marginBottom: "var(--spacing-md)" }} />
          <Skeleton height="4rem" />
        </div>

        <div className="card answer-card" style={{ padding: "var(--spacing-xl)" }}>
          <Skeleton height="1.5rem" width="25%" style={{ marginBottom: "var(--spacing-md)" }} />
          <Skeleton height="20rem" />
          <div
            style={{
              display: "flex",
              gap: "var(--spacing-md)",
              marginTop: "var(--spacing-lg)",
              justifyContent: "flex-end",
            }}
          >
            <Skeleton height="2.75rem" width="8rem" />
            <Skeleton height="2.75rem" width="6rem" />
          </div>
        </div>
      </div>
    </div>
  );
}
