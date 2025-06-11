# Writeo

AI-powered writing assistant with LanguageTool integration built on AWS Amplify and Next.js.

## Features

- **Grammar and Spell Checking**: Powered by LanguageTool running on AWS ECS
- **Multi-language Support**: Auto-detection and support for multiple languages
- **Real-time Processing**: Server-side actions for fast text analysis
- **Scalable Infrastructure**: ECS Fargate service with Application Load Balancer
- **Type Safety**: Full TypeScript support with Zod schemas
- **Modern UI**: Clean, responsive interface built with Tailwind CSS

## Architecture

- **Frontend**: Next.js 15 with React 19 and Tailwind CSS
- **Backend**: AWS Amplify Gen 2 with custom CDK constructs
- **Grammar Service**: LanguageTool running on AWS ECS Fargate
- **Infrastructure**: AWS VPC, Application Load Balancer, CloudWatch Logs
- **State Management**: Zustand for client-side state
- **Validation**: Zod schemas for type-safe API contracts

## Prerequisites

- Node.js 18 or later
- AWS CLI configured with appropriate permissions
- AWS Amplify CLI v6 or later

## Getting Started

### 1. Clone and Install

```bash
git clone <repository-url>
cd writeo
npm install
```

### 2. Deploy the Backend

```bash
npx amplify sandbox
```

This will deploy:

- AWS Amplify authentication and data services
- ECS cluster with LanguageTool service
- LanguageTool service running in your default VPC (cost-optimized)
- Application Load Balancer for internal access
- CloudWatch logs for monitoring

### 3. Start Development Server

```bash
npm run dev
```

The app will be available at http://localhost:3000

## Environment Variables

The application automatically loads configuration from Amplify outputs. For local development or custom deployments, you can set:

- `LANGUAGETOOL_ENDPOINT`: URL of the LanguageTool service

## Scripts

- `npm run dev` - Start development server with Turbopack
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run verify` - Format code, lint, and type check
- `npm run check` - Run verification and end-to-end tests
- `npm run test:e2e` - Run Playwright tests
- `npm run nuke` - Clean install and restart development

## API Reference

### Server Actions

#### `checkText(request: LanguageToolCheckRequest)`

Analyzes text for grammar and spelling issues.

```typescript
const result = await checkText({
  text: 'Your text to check',
  language: 'auto', // or specific language code
  level: 'default', // or "picky"
  enabledOnly: false,
});
```

#### `getAvailableLanguages()`

Retrieves list of supported languages.

#### `checkLanguageToolHealth()`

Checks if the LanguageTool service is healthy.

## Infrastructure Details

### ECS Service Configuration

- **Container**: `meyay/languagetool:latest`
- **Resources**: 1 vCPU, 2GB RAM
- **Health Check**: `/v2/languages` endpoint
- **Scaling**: Single instance (can be configured for auto-scaling)
- **Network**: Default VPC with public subnets (cost-optimized, no NAT gateway required)

### Security

- Private Application Load Balancer (internal access only)
- ECS tasks run in public subnets with restrictive security groups
- Security groups allow only VPC traffic to LanguageTool port
- CloudWatch logs for monitoring and debugging

## Development

### Adding New Features

1. Update Zod schemas in `app/lib/types.ts`
2. Add server actions in `app/lib/actions.ts`
3. Update client components as needed
4. Add tests in `tests/` directory

### Testing

Run the full test suite:

```bash
npm run check
```

This runs:

- Prettier formatting
- ESLint linting
- TypeScript type checking
- Playwright end-to-end tests

## Deployment

### Production Deployment

```bash
npx amplify deploy --branch main
```

### Environment-Specific Deployments

```bash
npx amplify deploy --branch staging
```

## Monitoring

- **CloudWatch Logs**: ECS task logs at `/ecs/languagetool`
- **Health Checks**: Application Load Balancer health checks
- **Metrics**: ECS service metrics in CloudWatch

## Troubleshooting

### LanguageTool Service Issues

1. Check ECS service status in AWS Console
2. Review CloudWatch logs for error messages
3. Verify security group rules allow traffic
4. Test health check endpoint manually

### Local Development Issues

1. Ensure Amplify backend is deployed
2. Check `amplify_outputs.json` is present
3. Verify environment variables are set correctly
4. Run `npm run verify` to check for linting issues

## Cost Optimization

- Uses default VPC to avoid VPC creation costs
- No NAT gateways required (saves ~$45/month)
- Single ECS task reduces costs for development
- CloudWatch log retention set to 1 week
- Resources can be scaled up for production use

## Contributing

1. Follow the established patterns for server actions and Zod schemas
2. Update documentation when adding new features
3. Ensure all tests pass before committing
4. Use `npm run verify` to check code quality
