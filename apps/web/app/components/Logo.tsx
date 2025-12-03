import Link from "next/link";

export function Logo() {
  return (
    <Link href="/" className="logo">
      <img
        src="/icon-192.png"
        alt="Writeo Logo"
        width={32}
        height={32}
        style={{
          display: "block",
          flexShrink: 0,
          width: "32px",
          height: "32px",
        }}
      />
      Writeo
    </Link>
  );
}
