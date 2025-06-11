# Cost Optimization Guide

## Overview

The LanguageTool service integration has been optimized for cost-effectiveness while maintaining security and functionality.

## Key Cost Savings

### Default VPC Usage

**Savings: ~$45-90/month**

Instead of creating a new VPC with NAT gateways, the service uses your AWS account's default VPC:

- ✅ **No VPC Creation Costs**: Uses existing default VPC
- ✅ **No NAT Gateway Costs**: Saves ~$45/month per availability zone
- ✅ **No Additional Routing Costs**: Uses existing internet gateway
- ✅ **Simplified Management**: Fewer resources to monitor and maintain

### Architecture Comparison

#### ❌ Expensive Approach (Original)

```
Custom VPC + Private Subnets + NAT Gateway
Monthly Cost: ~$50-100 (NAT Gateway + data transfer)
```

#### ✅ Cost-Optimized Approach (Current)

```
Default VPC + Public Subnets + Security Groups
Monthly Cost: ~$5-15 (ECS tasks only)
```

## Security Considerations

### Still Secure

Even though ECS tasks run in public subnets, the service remains secure:

- **Internal Load Balancer**: ALB is not internet-facing
- **Security Groups**: Only allow VPC traffic to port 8081
- **No Public Access**: LanguageTool service not directly accessible from internet
- **Amplify Integration**: Service only accessible through your application

### Security Group Configuration

```typescript
// Only allows traffic from within the VPC
securityGroup.addIngressRule(
  ec2.Peer.ipv4(vpc.vpcCidrBlock), // Only VPC traffic
  ec2.Port.tcp(8081), // Only port 8081
  'Allow ALB to reach LanguageTool service'
);
```

## Monthly Cost Breakdown

### Development Environment

```
ECS Fargate Tasks:     $5-10/month
CloudWatch Logs:       $1-2/month
Application Load Balancer: $16/month
Total:                 $22-28/month
```

### Production Environment (with auto-scaling)

```
ECS Fargate Tasks:     $20-50/month
CloudWatch Logs:       $2-5/month
Application Load Balancer: $16/month
Total:                 $38-71/month
```

### What You're NOT Paying For

```
NAT Gateway:           $45/month (saved)
Additional VPC:        $0 (but complexity saved)
Data Transfer (NAT):   $10-20/month (saved)
Total Savings:         $55-65/month
```

## Scaling Considerations

### Development

- **Single Task**: 1 ECS task running
- **Minimal Resources**: 1 vCPU, 2GB RAM
- **Basic Monitoring**: Standard CloudWatch metrics

### Production

- **Auto-Scaling**: 2-10 tasks based on demand
- **Enhanced Resources**: 2 vCPU, 4GB RAM per task
- **Advanced Monitoring**: Custom metrics and alerts

## Further Cost Optimizations

### Spot Instances

For non-critical workloads, consider Fargate Spot:

```typescript
const taskDefinition = new ecs.FargateTaskDefinition(scope, 'Task', {
  memoryLimitMiB: 2048,
  cpu: 1024,
  // Add Spot capacity provider for additional savings
});
```

### Reserved Capacity

For consistent workloads, consider:

- **Compute Savings Plans**: Up to 50% savings
- **Reserved Instances**: For predictable usage patterns

### Scheduled Scaling

Scale down during low usage:

```typescript
// Scale to 0 during night hours
scaling.scaleOnSchedule('NightScaleDown', {
  schedule: applicationautoscaling.Schedule.cron({
    hour: '23',
    minute: '0',
  }),
  minCapacity: 0,
  maxCapacity: 0,
});
```

## Monitoring Costs

### AWS Cost Explorer

Track spending by service:

- Filter by ECS service
- Monitor trending costs
- Set up budget alerts

### Tagging Strategy

```typescript
cdk.Tags.of(scope).add('Project', 'Writeo');
cdk.Tags.of(scope).add('Environment', 'production');
cdk.Tags.of(scope).add('Service', 'LanguageTool');
```

## Alternative Approaches

### If You Need Private Subnets

If your organization requires private subnets, consider:

1. **VPC Endpoints**: Access ECR without NAT gateway
2. **Single NAT Gateway**: Shared across availability zones
3. **Egress-Only Internet Gateway**: IPv6 only (if supported)

### Serverless Alternative

For very low usage, consider:

- **AWS Lambda**: Pay per request
- **Container Images**: LanguageTool in Lambda container
- **API Gateway**: For HTTP interface

## Best Practices

### Development

1. Use single availability zone
2. Minimal task resources
3. Short log retention (1 week)
4. Scale to zero when not needed

### Production

1. Multi-AZ for high availability
2. Right-sized task resources
3. Appropriate log retention (30 days)
4. Auto-scaling based on actual demand

### Monitoring

1. Set up cost alerts
2. Regular cost reviews
3. Monitor unused resources
4. Optimize based on usage patterns

## Implementation Notes

The current implementation uses:

- Default VPC from `ec2.Vpc.fromLookup()`
- Public subnets with `assignPublicIp: true`
- Internal ALB with `publicLoadBalancer: false`
- Restrictive security groups for network isolation

This approach provides the optimal balance of cost, security, and functionality for most use cases.
