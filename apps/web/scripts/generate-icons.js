#!/usr/bin/env node

/**
 * Script to generate PWA icons from SVG
 *
 * This script requires sharp to be installed:
 * npm install --save-dev sharp
 *
 * Usage: npm run generate-icons
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Check if sharp is available
let sharp;
try {
  sharp = (await import("sharp")).default;
} catch (error) {
  console.error("Error: sharp is not installed.");
  console.error("Please install it with: npm install --save-dev sharp");
  process.exit(1);
}

const publicDir = path.join(__dirname, "..", "public");
const iconSvgPath = path.join(__dirname, "..", "app", "icon.svg");

async function generateIcons() {
  try {
    // Ensure public directory exists
    if (!fs.existsSync(publicDir)) {
      fs.mkdirSync(publicDir, { recursive: true });
    }

    // Read the SVG file
    const svgBuffer = fs.readFileSync(iconSvgPath);

    // Generate 192x192 icon
    await sharp(svgBuffer).resize(192, 192).png().toFile(path.join(publicDir, "icon-192.png"));
    console.log("✓ Generated icon-192.png");

    // Generate 512x512 icon
    await sharp(svgBuffer).resize(512, 512).png().toFile(path.join(publicDir, "icon-512.png"));
    console.log("✓ Generated icon-512.png");

    console.log("\n✅ All icons generated successfully!");
  } catch (error) {
    console.error("Error generating icons:", error);
    process.exit(1);
  }
}

generateIcons();
