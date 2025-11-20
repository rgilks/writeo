import Link from "next/link";
import Image from "next/image";

export function Footer() {
  return (
    <footer className="mt-auto border-t border-gray-200 bg-white w-full">
      <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="flex flex-col items-center justify-center gap-8 text-center">
          {/* Ko-fi Button */}
          <div>
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
            className="flex flex-wrap items-center justify-center gap-x-4 gap-y-3"
            aria-label="Footer navigation"
          >
            <a
              href="https://discord.gg/9rtwCKp2"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-2 py-1"
            >
              Support
            </a>
            <span className="text-gray-300 select-none" aria-hidden="true">
              •
            </span>
            <Link
              href="/terms"
              className="text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-2 py-1"
            >
              Terms of Service
            </Link>
            <span className="text-gray-300 select-none" aria-hidden="true">
              •
            </span>
            <Link
              href="/privacy"
              className="text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-2 py-1"
            >
              Privacy Policy
            </Link>
            <span className="text-gray-300 select-none" aria-hidden="true">
              •
            </span>
            <Link
              href="/accessibility"
              className="text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-2 py-1"
            >
              Accessibility
            </Link>
            <span className="text-gray-300 select-none" aria-hidden="true">
              •
            </span>
            <a
              href="https://github.com/rgilks/writeo"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-2 py-1 inline-flex items-center gap-1.5"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path
                  fillRule="evenodd"
                  d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                  clipRule="evenodd"
                />
              </svg>
              GitHub
            </a>
          </nav>
        </div>
      </div>
    </footer>
  );
}
