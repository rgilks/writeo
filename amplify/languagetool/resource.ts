import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ecsPatterns from 'aws-cdk-lib/aws-ecs-patterns';
import * as logs from 'aws-cdk-lib/aws-logs';
import { Construct } from 'constructs';

export function createLanguageToolService(
  scope: Construct,
  id: string
): {
  endpoint: string;
  vpcId: string;
} {
  const vpc = new ec2.Vpc(scope, `${id}Vpc`, {
    maxAzs: 2,
    natGateways: 0,
    subnetConfiguration: [
      {
        cidrMask: 24,
        name: 'Public',
        subnetType: ec2.SubnetType.PUBLIC,
      },
    ],
  });

  const cluster = new ecs.Cluster(scope, `${id}Cluster`, {
    vpc,
    clusterName: 'languagetool-cluster',
  });

  const securityGroup = new ec2.SecurityGroup(scope, `${id}SecurityGroup`, {
    vpc,
    description: 'Security group for LanguageTool ECS service',
    allowAllOutbound: true,
  });

  securityGroup.addIngressRule(
    ec2.Peer.ipv4(vpc.vpcCidrBlock),
    ec2.Port.tcp(8081),
    'Allow ALB to reach LanguageTool service'
  );

  const logGroup = new logs.LogGroup(scope, `${id}LogGroup`, {
    logGroupName: '/ecs/languagetool',
    removalPolicy: cdk.RemovalPolicy.DESTROY,
    retention: logs.RetentionDays.ONE_WEEK,
  });

  const taskDefinition = new ecs.FargateTaskDefinition(scope, `${id}Task`, {
    memoryLimitMiB: 2048,
    cpu: 1024,
  });

  taskDefinition.addContainer('languagetool', {
    image: ecs.ContainerImage.fromRegistry('meyay/languagetool:latest'),
    containerName: 'languagetool',
    portMappings: [{ containerPort: 8081 }],
    environment: {
      LISTEN_PORT: '8081',
      Java_Xms: '1g',
      Java_Xmx: '1g',
    },
    logging: ecs.LogDrivers.awsLogs({
      streamPrefix: 'languagetool',
      logGroup,
    }),
    healthCheck: {
      command: [
        'CMD-SHELL',
        'wget --no-verbose --tries=1 --spider http://localhost:8081/v2/languages || exit 1',
      ],
      interval: cdk.Duration.seconds(30),
      timeout: cdk.Duration.seconds(5),
      retries: 3,
      startPeriod: cdk.Duration.seconds(60),
    },
  });

  const fargateService = new ecsPatterns.ApplicationLoadBalancedFargateService(
    scope,
    `${id}FargateService`,
    {
      cluster,
      taskDefinition,
      publicLoadBalancer: false,
      assignPublicIp: true,
      desiredCount: 1,
      serviceName: 'languagetool-service',
      listenerPort: 80,
      protocol: cdk.aws_elasticloadbalancingv2.ApplicationProtocol.HTTP,
      taskSubnets: {
        subnetType: ec2.SubnetType.PUBLIC,
      },
      platformVersion: ecs.FargatePlatformVersion.LATEST,
    }
  );

  fargateService.service.connections.addSecurityGroup(securityGroup);

  fargateService.targetGroup.configureHealthCheck({
    path: '/v2/languages',
    port: '8081',
    protocol: cdk.aws_elasticloadbalancingv2.Protocol.HTTP,
    healthyHttpCodes: '200',
  });

  const endpoint = `http://${fargateService.loadBalancer.loadBalancerDnsName}`;

  new cdk.CfnOutput(scope, `${id}Endpoint`, {
    value: endpoint,
    description: 'LanguageTool service endpoint',
  });

  return {
    endpoint,
    vpcId: vpc.vpcId,
  };
}
