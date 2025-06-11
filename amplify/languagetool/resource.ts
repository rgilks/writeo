import { Construct } from 'constructs';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as servicediscovery from 'aws-cdk-lib/aws-servicediscovery';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as cdk from 'aws-cdk-lib';

export class LanguageToolService extends Construct {
  public readonly publicUrl: string;
  public readonly vpc: ec2.Vpc;

  constructor(scope: Construct, id: string) {
    super(scope, id);

    this.vpc = new ec2.Vpc(this, 'LanguageToolVPC', {
      maxAzs: 2,
      subnetConfiguration: [
        {
          cidrMask: 24,
          name: 'PublicSubnet',
          subnetType: ec2.SubnetType.PUBLIC,
        },
        {
          cidrMask: 24,
          name: 'PrivateSubnet',
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
        },
      ],
      enableDnsHostnames: true,
      enableDnsSupport: true,
    });

    const cluster = new ecs.Cluster(this, 'LanguageToolCluster', {
      vpc: this.vpc,
      clusterName: 'languagetool-cluster',
    });

    const namespace = new servicediscovery.PrivateDnsNamespace(this, 'ServiceNamespace', {
      name: 'languagetool.local',
      vpc: this.vpc,
    });

    const logGroup = new logs.LogGroup(this, 'LanguageToolLogGroup', {
      logGroupName: '/ecs/languagetool',
      retention: logs.RetentionDays.ONE_WEEK,
    });

    const taskDefinition = new ecs.FargateTaskDefinition(this, 'LanguageToolTaskDef', {
      memoryLimitMiB: 2048,
      cpu: 1024,
    });

    const container = taskDefinition.addContainer('LanguageToolContainer', {
      image: ecs.ContainerImage.fromRegistry('meyay/languagetool:latest'),
      memoryLimitMiB: 2048,
      cpu: 1024,
      environment: {
        JAVA_TOOL_OPTIONS: '-Xms1g -Xmx1800m',
      },
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: 'languagetool',
        logGroup,
      }),
    });

    container.addPortMappings({
      containerPort: 8081,
      protocol: ecs.Protocol.TCP,
    });

    const ecsSecurityGroup = new ec2.SecurityGroup(this, 'ECSSecurityGroup', {
      vpc: this.vpc,
      description: 'Security group for LanguageTool ECS service',
      allowAllOutbound: true,
    });

    const albSecurityGroup = new ec2.SecurityGroup(this, 'ALBSecurityGroup', {
      vpc: this.vpc,
      description: 'Security group for LanguageTool ALB',
      allowAllOutbound: true,
    });

    albSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(80),
      'Allow HTTP access from anywhere'
    );

    ecsSecurityGroup.addIngressRule(
      albSecurityGroup,
      ec2.Port.tcp(8081),
      'Allow ALB access to LanguageTool'
    );

    const alb = new elbv2.ApplicationLoadBalancer(this, 'LanguageToolALB', {
      vpc: this.vpc,
      internetFacing: true,
      securityGroup: albSecurityGroup,
    });

    const listener = alb.addListener('LanguageToolListener', {
      port: 80,
      protocol: elbv2.ApplicationProtocol.HTTP,
    });

    const service = new ecs.FargateService(this, 'LanguageToolService', {
      cluster,
      taskDefinition,
      desiredCount: 1,
      assignPublicIp: false,
      securityGroups: [ecsSecurityGroup],
      serviceName: 'languagetool-service-1',
      cloudMapOptions: {
        name: 'languagetool',
        cloudMapNamespace: namespace,
        dnsRecordType: servicediscovery.DnsRecordType.A,
      },
      vpcSubnets: {
        subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
      },
    });

    listener.addTargets('LanguageToolTargets', {
      port: 8081,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targets: [
        service.loadBalancerTarget({
          containerName: 'LanguageToolContainer',
          containerPort: 8081,
        }),
      ],
      healthCheck: {
        path: '/v2/languages',
        interval: cdk.Duration.seconds(60),
        timeout: cdk.Duration.seconds(30),
        healthyThresholdCount: 2,
        unhealthyThresholdCount: 5,
      },
    });

    this.publicUrl = `http://${alb.loadBalancerDnsName}`;

    new cdk.CfnOutput(this, 'LanguageToolServiceUrl', {
      value: 'http://languagetool.languagetool.local:8081',
      description: 'LanguageTool Service Discovery URL (internal)',
    });

    new cdk.CfnOutput(this, 'LanguageToolPublicUrl', {
      value: this.publicUrl,
      description: 'LanguageTool Public URL via ALB',
    });

    new cdk.CfnOutput(this, 'LanguageToolVPCId', {
      value: this.vpc.vpcId,
      description: 'VPC ID for LanguageTool service',
    });
  }
}
