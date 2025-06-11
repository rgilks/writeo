import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as servicediscovery from 'aws-cdk-lib/aws-servicediscovery';
import * as cdk from 'aws-cdk-lib';

export function createLanguageToolService(backend: { createStack: (name: string) => cdk.Stack }) {
  const languageToolStack = backend.createStack('LanguageToolStack');

  const vpc = new ec2.Vpc(languageToolStack, 'LanguageToolVPC', {
    maxAzs: 2,
    subnetConfiguration: [
      {
        cidrMask: 24,
        name: 'PublicSubnet',
        subnetType: ec2.SubnetType.PUBLIC,
      },
    ],
    enableDnsHostnames: true,
    enableDnsSupport: true,
  });

  const cluster = new ecs.Cluster(languageToolStack, 'LanguageToolCluster', {
    vpc,
    clusterName: 'languagetool-cluster',
  });

  const namespace = new servicediscovery.PrivateDnsNamespace(
    languageToolStack,
    'ServiceNamespace',
    {
      name: 'languagetool.local',
      vpc,
    }
  );

  const logGroup = new logs.LogGroup(languageToolStack, 'LanguageToolLogGroup', {
    logGroupName: '/ecs/languagetool',
    retention: logs.RetentionDays.ONE_WEEK,
  });

  const taskDefinition = new ecs.FargateTaskDefinition(languageToolStack, 'LanguageToolTaskDef', {
    memoryLimitMiB: 2048,
    cpu: 1024,
  });

  taskDefinition.addContainer('LanguageToolContainer', {
    image: ecs.ContainerImage.fromRegistry('meyay/languagetool:latest'),
    memoryLimitMiB: 2048,
    cpu: 1024,
    portMappings: [
      {
        containerPort: 8081,
        protocol: ecs.Protocol.TCP,
      },
    ],
    environment: {
      JAVA_TOOL_OPTIONS: '-Xms1g -Xmx1800m',
    },
    logging: ecs.LogDrivers.awsLogs({
      streamPrefix: 'languagetool',
      logGroup,
    }),
    healthCheck: {
      command: ['CMD-SHELL', 'curl -f http://localhost:8081/v2/check?text=test || exit 1'],
      interval: cdk.Duration.seconds(30),
      timeout: cdk.Duration.seconds(5),
      retries: 3,
      startPeriod: cdk.Duration.seconds(60),
    },
  });

  const ecsSecurityGroup = new ec2.SecurityGroup(languageToolStack, 'ECSSecurityGroup', {
    vpc,
    description: 'Security group for LanguageTool ECS service',
    allowAllOutbound: true,
  });

  ecsSecurityGroup.addIngressRule(
    ec2.Peer.ipv4(vpc.vpcCidrBlock),
    ec2.Port.tcp(8081),
    'Allow LanguageTool access from within VPC'
  );

  const service = new ecs.FargateService(languageToolStack, 'LanguageToolService', {
    cluster,
    taskDefinition,
    desiredCount: 1,
    assignPublicIp: true,
    securityGroups: [ecsSecurityGroup],
    serviceName: 'languagetool-service',
    cloudMapOptions: {
      name: 'languagetool',
      cloudMapNamespace: namespace,
      dnsRecordType: servicediscovery.DnsRecordType.A,
    },
  });

  new cdk.CfnOutput(languageToolStack, 'LanguageToolServiceUrl', {
    value: 'http://languagetool.languagetool.local:8081',
    description: 'LanguageTool Service Discovery URL',
    exportName: 'LanguageToolServiceUrl',
  });

  new cdk.CfnOutput(languageToolStack, 'LanguageToolVPCId', {
    value: vpc.vpcId,
    description: 'VPC ID for LanguageTool service',
    exportName: 'LanguageToolVPCId',
  });

  return {
    vpc,
    cluster,
    service,
    serviceUrl: 'http://languagetool.languagetool.local:8081',
  };
}
