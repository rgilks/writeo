# PWA Setup Guide

This document describes the Progressive Web App (PWA) implementation for Writeo.

## What's Included

✅ **Web App Manifest** (`/public/manifest.json`)

- App metadata and configuration
- Icon definitions
- App shortcuts for quick access

✅ **Service Worker** (`/public/service-worker.js`)

- Offline support and caching
- Static asset precaching
- Runtime caching for better performance

✅ **PWA Registration Component** (`/app/components/PWARegistration.tsx`)

- Automatic service worker registration
- Install prompt handling
- Update detection

## Optional: Generate PNG Icons

The manifest references PNG icons (`icon-192.png` and `icon-512.png`) for better compatibility across platforms. While the SVG icon works, PNG icons are recommended for:

- Better Android support
- iOS home screen icons
- App shortcuts

To generate the PNG icons:

```bash
cd apps/web
npm install --save-dev sharp
npm run generate-icons
```

This will create:

- `public/icon-192.png` (192x192 pixels)
- `public/icon-512.png` (512x512 pixels)

## Testing the PWA

### Local Testing

1. **Start the development server:**

   ```bash
   npm run dev
   ```

2. **Open Chrome DevTools:**
   - Go to **Application** tab
   - Check **Service Workers** section - you should see the service worker registered
   - Check **Manifest** section - verify all metadata is correct
   - Check **Cache Storage** - verify assets are being cached

3. **Test Install Prompt:**
   - The install prompt will appear automatically when PWA criteria are met
   - Or use Chrome's "Install" button in the address bar

4. **Test Offline Mode:**
   - Open DevTools → Network tab
   - Enable "Offline" mode
   - Refresh the page - it should still work from cache

### Production Testing

After deployment, test on:

- **Chrome/Edge**: Full PWA support
- **Safari (iOS)**: Add to Home Screen (limited PWA features)
- **Firefox**: Basic PWA support
- **Mobile devices**: Install and test offline functionality

## PWA Features

### ✅ Implemented

- **Offline Support**: Cached pages work offline
- **Installable**: Users can install the app
- **Fast Loading**: Service worker caches static assets
- **App-like Experience**: Standalone display mode
- **Auto-updates**: Service worker updates automatically

### 🔄 Service Worker Strategy

- **Static Assets**: Cache-first (fast loading)
- **Pages**: Network-first with cache fallback (always fresh content)
- **API Requests**: Network-only (no caching for dynamic data)

## Troubleshooting

### Service Worker Not Registering

1. Check browser console for errors
2. Verify `/service-worker.js` is accessible
3. Ensure you're on HTTPS (or localhost for development)
4. Check CSP headers allow service workers

### Install Prompt Not Showing

The install prompt requires:

- ✅ Valid manifest.json
- ✅ Registered service worker
- ✅ HTTPS (or localhost)
- ✅ User engagement (visited site multiple times)
- ✅ Meets browser's installability criteria

### Icons Not Displaying

- Verify icon files exist in `/public` directory
- Check manifest.json icon paths are correct
- Clear browser cache and service worker cache
- Regenerate icons if needed: `npm run generate-icons`

## Deployment Notes

The PWA works automatically with your existing Cloudflare Workers deployment. No additional configuration needed - the service worker and manifest are served from the `public` directory.

## Additional Resources

- [MDN: Progressive Web Apps](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps)
- [Web.dev: PWA Checklist](https://web.dev/pwa-checklist/)
- [Service Worker API](https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API)
