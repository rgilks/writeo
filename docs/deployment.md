# Deployment Guide

## Overview

This guide covers deploying the Writeo application with LanguageTool integration using AWS Amplify Gen 2 and custom CDK constructs.

## Prerequisites

### Required Tools

- **Node.js**: Version 18 or later
- **AWS CLI**: Version 2.x configured with appropriate credentials
- **Amplify CLI**: Version 6 or later (`npm install -g @aws-amplify/cli`)
- **Git**: For version control and deployments

### AWS Permissions

Your AWS credentials need the following permissions:

- CloudFormation (full access)
- ECS (full access)
- EC2 (VPC management)
- Elastic Load Balancing
- IAM (role management)
- CloudWatch Logs
- Amplify (full access)

## Deployment Process

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd writeo

# Install dependencies
npm install

# Install Amplify backend dependencies
cd amplify
npm install
cd ..
```

### 2. Configure AWS Credentials

```bash
# Configure AWS CLI (if not already done)
aws configure

# Verify credentials
aws sts get-caller-identity
```

### 3. Development Deployment (Sandbox)

```bash
# Start Amplify sandbox environment
npx amplify sandbox

# This creates:
# - Temporary CloudFormation stack
# - ECS cluster with LanguageTool service
# - VPC with networking components
# - Application Load Balancer
```

The sandbox deployment typically takes 10-15 minutes to complete.

### 4. Production Deployment

```bash
# Deploy to production
npx amplify deploy --branch main

# For staging environment
npx amplify deploy --branch staging
```

## Infrastructure Components

### Deployed Resources

The deployment creates the following AWS resources:

#### Core Amplify Resources

- **CloudFormation Stack**: Main application stack
- **Cognito User Pool**: Authentication
- **AppSync API**: GraphQL data layer
- **Lambda Functions**: Server-side logic

#### Custom ECS Resources

- **Default VPC**: Uses existing VPC to reduce costs
- **ECS Cluster**: Fargate-based container orchestration
- **Task Definition**: LanguageTool container specification
- **ECS Service**: Managed service with health checks in public subnets
- **Application Load Balancer**: Internal load balancer
- **CloudWatch Log Group**: Centralized logging
- **Security Groups**: Restrictive network access control

#### Networking

- **Default VPC**: No additional VPC costs
- **Public Subnets**: Direct internet access (no NAT gateway needed)
- **Internet Gateway**: Existing gateway for container registry access
- **Security Groups**: VPC-only access to LanguageTool service

### Resource Specifications

#### ECS Task Configuration

```yaml
CPU: 1024 (1 vCPU)
Memory: 2048 MB (2 GB)
Network Mode: awsvpc
Launch Type: FARGATE
```

#### Container Configuration

```yaml
Image: meyay/languagetool:latest
Port: 8081
Environment Variables:
  LISTEN_PORT: '8081'
  Java_Xms: '1g'
  Java_Xmx: '1g'
```

## Configuration Management

### Environment Variables

The application automatically loads configuration from:

1. **Amplify Outputs** (`amplify_outputs.json`)
2. **Environment Variables** (fallback)

#### Key Configuration Values

- `LANGUAGETOOL_ENDPOINT`: Auto-generated ALB DNS name
- `LANGUAGETOOL_VPC_ID`: VPC identifier for debugging

### Custom Configuration

For advanced configurations, modify `amplify/languagetool/resource.ts`:

```typescript
// Adjust ECS task resources
const taskDefinition = new ecs.FargateTaskDefinition(scope, `${id}Task`, {
  memoryLimitMiB: 4096, // Increase memory
  cpu: 2048,            // Increase CPU
});

// Modify container environment
environment: {
  LISTEN_PORT: '8081',
  Java_Xms: '2g',       // Increase heap size
  Java_Xmx: '2g',
  download_ngrams_for_langs: 'en,es,fr', // Language models
},
```

## Monitoring and Logging

### CloudWatch Integration

#### Log Groups

- `/ecs/languagetool`: Container application logs
- `/aws/lambda/*`: Amplify function logs
- `/aws/amplify/*`: Build and deployment logs

#### Viewing Logs

```bash
# View LanguageTool service logs
aws logs tail /ecs/languagetool --follow

# View recent errors
aws logs filter-log-events \
  --log-group-name /ecs/languagetool \
  --filter-pattern "ERROR"
```

### Health Monitoring

#### ECS Service Health

- **Target Group Health**: ALB health checks
- **Service Status**: ECS console monitoring
- **Task Health**: Individual container status

#### Application Health Check

```bash
# Test service endpoint (replace with actual ALB DNS)
curl http://internal-alb-xxx.region.elb.amazonaws.com/v2/languages
```

## Scaling Configuration

### Auto Scaling

Configure auto-scaling based on resource utilization:

```typescript
// Add to languagetool/resource.ts
const scaling = fargateService.service.autoScaleTaskCount({
  minCapacity: 1,
  maxCapacity: 10,
});

scaling.scaleOnCpuUtilization('CpuScaling', {
  targetUtilizationPercent: 70,
  scaleInCooldown: cdk.Duration.minutes(5),
  scaleOutCooldown: cdk.Duration.minutes(2),
});
```

### Manual Scaling

```bash
# Scale service via AWS CLI
aws ecs update-service \
  --cluster languagetool-cluster \
  --service languagetool-service \
  --desired-count 3
```

## Environment Management

### Multiple Environments

#### Branch-based Deployment

```bash
# Development
git checkout develop
npx amplify deploy --branch develop

# Staging
git checkout staging
npx amplify deploy --branch staging

# Production
git checkout main
npx amplify deploy --branch main
```

#### Environment Variables per Branch

```typescript
// Different configurations per environment
const isProd = process.env.NODE_ENV === 'production';
const taskMemory = isProd ? 4096 : 2048;
const taskCpu = isProd ? 2048 : 1024;
const desiredCount = isProd ? 2 : 1;
```

### Configuration Overrides

Create environment-specific configurations:

```typescript
// amplify/env/production.ts
export const productionConfig = {
  languageTool: {
    memoryLimitMiB: 4096,
    cpu: 2048,
    desiredCount: 2,
    enableAutoScaling: true,
  },
};

// amplify/env/staging.ts
export const stagingConfig = {
  languageTool: {
    memoryLimitMiB: 2048,
    cpu: 1024,
    desiredCount: 1,
    enableAutoScaling: false,
  },
};
```

## Security Considerations

### Network Security

#### Security Groups

- **ECS Tasks**: Only allow inbound traffic from ALB
- **ALB**: Only allow traffic from within VPC
- **NAT Gateway**: Outbound internet access for container registry

#### VPC Configuration

- **Private Subnets**: ECS tasks isolated from internet
- **Public Subnets**: ALB and NAT Gateway only
- **Network ACLs**: Additional layer of security

### IAM Roles and Policies

#### Task Execution Role

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

## Troubleshooting

### Common Issues

#### 1. ECS Service Won't Start

**Symptoms**: Tasks fail to start or constantly restart

**Solutions**:

```bash
# Check task logs
aws logs tail /ecs/languagetool --follow

# Verify task definition
aws ecs describe-task-definition --task-definition languagetool

# Check service events
aws ecs describe-services --cluster languagetool-cluster --services languagetool-service
```

#### 2. Health Check Failures

**Symptoms**: ALB marks targets as unhealthy

**Solutions**:

```bash
# Test container health endpoint
aws ecs execute-command --cluster languagetool-cluster --task <task-arn> --command "wget -q -O- http://localhost:8081/v2/languages"

# Adjust health check settings
# Increase health check timeout or interval
```

#### 3. Out of Memory Errors

**Symptoms**: Tasks killed with exit code 137

**Solutions**:

```typescript
// Increase task memory
const taskDefinition = new ecs.FargateTaskDefinition(scope, `${id}Task`, {
  memoryLimitMiB: 4096, // Increase from 2048
  cpu: 2048, // May need to increase CPU too
});
```

#### 4. Network Connectivity Issues

**Symptoms**: Cannot reach LanguageTool service from application

**Solutions**:

```bash
# Check security group rules
aws ec2 describe-security-groups --filters "Name=group-name,Values=*languagetool*"

# Verify ALB target health
aws elbv2 describe-target-health --target-group-arn <target-group-arn>

# Test connectivity from within VPC
# Use AWS Systems Manager Session Manager or temporary EC2 instance
```

### Debug Commands

#### View Deployment Status

```bash
# Check CloudFormation stack status
aws cloudformation describe-stacks --stack-name amplify-writeo-main

# Monitor stack events
aws cloudformation describe-stack-events --stack-name amplify-writeo-main
```

#### Service Debugging

```bash
# Get service status
aws ecs describe-services --cluster languagetool-cluster --services languagetool-service

# List running tasks
aws ecs list-tasks --cluster languagetool-cluster --service languagetool-service

# Get task details
aws ecs describe-tasks --cluster languagetool-cluster --tasks <task-arn>
```

#### Load Balancer Debugging

```bash
# Check target group health
aws elbv2 describe-target-health --target-group-arn <arn>

# View load balancer attributes
aws elbv2 describe-load-balancers --load-balancer-arns <arn>
```

## Performance Optimization

### Resource Tuning

#### JVM Optimization

```typescript
environment: {
  LISTEN_PORT: '8081',
  Java_Xms: '2g',
  Java_Xmx: '2g',
  JAVA_OPTS: '-XX:+UseG1GC -XX:MaxGCPauseMillis=200 -XX:+UseStringDeduplication',
},
```

#### Container Optimization

```typescript
// Add resource limits
const container = taskDefinition.addContainer('languagetool', {
  // ... other config
  memoryReservationMiB: 1536, // Soft limit
  cpu: 1024,

  // Health check optimization
  healthCheck: {
    command: [
      'CMD-SHELL',
      'wget --spider --timeout=5 http://localhost:8081/v2/languages || exit 1',
    ],
    interval: cdk.Duration.seconds(15),
    timeout: cdk.Duration.seconds(5),
    retries: 2,
    startPeriod: cdk.Duration.seconds(30),
  },
});
```

## Cost Optimization

### Development Environment

```typescript
// Minimal resources for development
const devConfig = {
  memoryLimitMiB: 1024,
  cpu: 512,
  desiredCount: 1,
  natGateways: 1, // Single NAT Gateway
};
```

### Production Environment

```typescript
// Optimized for production
const prodConfig = {
  memoryLimitMiB: 4096,
  cpu: 2048,
  desiredCount: 2,
  natGateways: 2, // Multi-AZ for high availability
  enableAutoScaling: true,
};
```

### Scheduled Scaling

```typescript
// Scale down during low usage periods
const scheduledScaling = new applicationautoscaling.Schedule(scope, 'NightScale', {
  schedule: applicationautoscaling.Schedule.cron({
    hour: '23',
    minute: '0',
  }),
  timezone: 'UTC',
});

scaling.scaleOnSchedule('NightScaleDown', {
  schedule: scheduledScaling,
  minCapacity: 0,
  maxCapacity: 0,
});
```

## Rollback Procedures

### Application Rollback

```bash
# Rollback to previous Amplify deployment
npx amplify rollback

# Rollback specific service version
aws ecs update-service \
  --cluster languagetool-cluster \
  --service languagetool-service \
  --task-definition languagetool:PREVIOUS_REVISION
```

### Infrastructure Rollback

```bash
# Rollback CloudFormation stack
aws cloudformation cancel-update-stack --stack-name amplify-writeo-main

# Or rollback to previous version
aws cloudformation continue-update-rollback --stack-name amplify-writeo-main
```

## Cleanup

### Development Environment

```bash
# Delete sandbox environment
npx amplify sandbox delete

# Or delete specific branch deployment
npx amplify delete --branch develop
```

### Production Environment

```bash
# Full cleanup (WARNING: This deletes all resources)
npx amplify delete --branch main
```

### Partial Cleanup

```bash
# Delete only custom resources
aws cloudformation delete-stack --stack-name LanguageToolStack
```
