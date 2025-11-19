import Link from "next/link";
import Image from "next/image";

export function Footer() {
  return (
    <footer className="mt-auto border-t border-gray-200 bg-white w-full">
      <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
        <div
          className="flex flex-col items-center justify-center gap-5 text-center"
          style={{
            width: "100%",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          {/* Ko-fi Button */}
          <div style={{ display: "flex", justifyContent: "center", width: "100%" }}>
            <a
              href="https://ko-fi.com/N4N31DPNUS"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block transition-opacity hover:opacity-75 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded"
            >
              <Image
                width={145}
                height={36}
                src="https://storage.ko-fi.com/cdn/kofi2.png?v=6"
                alt="Buy Me a Coffee at ko-fi.com"
                className="block"
                unoptimized
              />
            </a>
          </div>

          {/* Legal Links */}
          <nav
            className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2"
            aria-label="Footer navigation"
            style={{
              display: "flex",
              flexWrap: "wrap",
              alignItems: "center",
              justifyContent: "center",
              width: "100%",
            }}
          >
            <Link
              href="/terms"
              className="text-base font-medium text-gray-600 hover:text-gray-900 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-1"
              style={{ lineHeight: "1.6" }}
            >
              Terms of Service
            </Link>
            <span
              className="text-gray-300 text-lg leading-none select-none"
              aria-hidden="true"
              style={{ margin: "0 4px" }}
            >
              •
            </span>
            <Link
              href="/privacy"
              className="text-base font-medium text-gray-600 hover:text-gray-900 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-1"
              style={{ lineHeight: "1.6" }}
            >
              Privacy Policy
            </Link>
          </nav>
        </div>
      </div>
    </footer>
  );
}
