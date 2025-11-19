# Writeo Web App

Next.js frontend for Writeo essay scoring system, deployed to Cloudflare Workers via OpenNext.

## Features

- **Task Interface**: Browse and select writing prompts
- **Essay Editor**: Clean writing interface with question and answer panels
- **Results Display**: View scores, CEFR levels, and detailed dimension scores
- **Real-time Polling**: Automatically checks for results after submission
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### Prerequisites

- Node.js 20+
- npm or yarn

### Setup

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

The app will be available at `http://localhost:3000`

### Environment Variables

Create `.env.local`:

```
NEXT_PUBLIC_API_BASE=http://localhost:8787
```

For production, set via Cloudflare:

```bash
wrangler secret put NEXT_PUBLIC_API_BASE
```

Or update `wrangler.toml`:

```toml
[vars]
NEXT_PUBLIC_API_BASE = "https://your-api-worker.workers.dev"
```

## Building for Production

```bash
# Build with OpenNext
npm run build

# Preview production build locally
npm run preview

# Deploy to Cloudflare
npm run deploy
```

## Project Structure

```
app/
├── layout.tsx          # Root layout with header
├── page.tsx            # Home page (task list)
├── write/
│   └── [id]/
│       └── page.tsx     # Writing interface
├── results/
│   └── [id]/
│       └── page.tsx     # Results display
├── lib/
│   ├── actions.ts      # Server Actions for API calls
│   └── api-config.ts   # API configuration helpers
└── globals.css         # Global styles
```

## Pages

### Home Page (`/`)

Displays a grid of available writing tasks/prompts. Users can click to start writing.

**Available Tasks:**

- Education: Practical vs Theoretical (Agree/Disagree)
- Technology: Social Media Impact (Discussion)
- Environment: Individual vs Government (Opinion)
- Work: Remote Working (Advantages/Disadvantages)
- Health: Fast Food Problem (Problem/Solution)
- Society: Ageing Population (Two-part Question)
- Culture: Global vs Local (Agree/Disagree)
- Crime: Punishment vs Rehabilitation (Discussion)

### Writing Page (`/write/[id]`)

Two-panel layout:

- **Left**: Question/prompt display
- **Right**: Essay editor with submit button

Users write their essay and click "Submit for Scoring" to send it to the API.

### Results Page (`/results/[id]`)

Shows assessment results:

- **Overall Score**: Band score (0-9)
- **CEFR Level**: Language proficiency level (A2-C2)
- **Detailed Scores**: Table with dimension scores (TA, CC, Vocab, Grammar, Overall)

The page automatically polls the API until results are ready.

## API Integration

The app uses the Writeo API Worker endpoints:

1. **Create Submission**: `PUT /text/submissions/{id}` (with inline answer and optional question)
   - Answers must always be sent inline with the submission
   - Questions can be sent inline (`question-text`) or referenced by ID (`question-id` only)
   - Optional: Create question separately via `PUT /text/questions/{id}` if you want to reference it
2. **Get Results**: `GET /text/submissions/{id}` (results are returned immediately in PUT response, GET is for retrieval)

See `app/lib/actions.ts` for the implementation (uses Next.js Server Actions).

## Styling

The app uses CSS custom properties for theming:

- `--primary-color`: Main brand color (#2563eb)
- `--text-primary`: Main text color
- `--text-secondary`: Secondary text color
- `--bg-primary`: Background color
- `--bg-secondary`: Secondary background

All styles are in `app/globals.css` using a modern, clean design.

## Customization

### Adding New Tasks

Edit `app/page.tsx` and `app/write/[id]/page.tsx` to add more task prompts:

```typescript
const tasks = [
  {
    id: "5",
    title: "Your New Task",
    description: "Description here",
    prompt: "Your writing prompt here...",
  },
  // ... more tasks
];
```

### Changing API Base URL

Set `API_BASE_URL` environment variable in Cloudflare Workers (via `wrangler secret put API_BASE_URL`) or update `app/lib/api-config.ts`.

### Styling Changes

Modify `app/globals.css` to customize colors, fonts, spacing, etc.

## Troubleshooting

### Build Fails: "opennext: command not found"

Install OpenNext:

```bash
npm install -D @opennextjs/cloudflare
```

### Wrangler Can't Find Build Output

Ensure `wrangler.toml` has:

```toml
pages_build_output_dir = ".opennext"
```

### API Calls Fail with CORS

Ensure your API worker has CORS headers configured. Check `apps/api-worker/src/index.ts`.

### Results Page Stuck on "Processing"

- Check API worker logs: `wrangler tail`
- Verify API worker is processing submissions correctly
- Check Modal service is deployed and accessible
- Verify `MODAL_GRADE_URL` secret is set correctly

## Deployment

The app is configured for Cloudflare Workers deployment using OpenNext:

1. Build: `npm run build` (creates `.opennext/` directory)
2. Deploy: `wrangler deploy` (deploys from `.opennext/`)

Update `wrangler.toml` with your domain configuration before deploying.

## References

- [OpenNext Documentation](https://opennext.js.org/cloudflare)
- [Next.js App Router](https://nextjs.org/docs/app)
- [Cloudflare Workers](https://developers.cloudflare.com/workers/)
