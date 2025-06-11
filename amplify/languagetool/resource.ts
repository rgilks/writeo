import { Construct } from 'constructs';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as logs from 'aws-cdk-lib/aws-logs';
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
      ],
      natGateways: 0,
    });

    const cluster = new ecs.Cluster(this, 'LanguageToolCluster', {
      vpc: this.vpc,
      clusterName: `languagetool-cluster-${cdk.Aws.STACK_NAME}`,
    });

    const logGroup = new logs.LogGroup(this, 'LanguageToolLogGroup', {
      logGroupName: `/ecs/languagetool-${cdk.Aws.STACK_NAME}`,
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
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

    ecsSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(8081),
      'Allow NLB access to LanguageTool'
    );

    const nlb = new elbv2.NetworkLoadBalancer(this, 'LanguageToolNLB', {
      vpc: this.vpc,
      internetFacing: true,
    });

    const listener = nlb.addListener('LanguageToolListener', {
      port: 80,
      protocol: elbv2.Protocol.TCP,
    });

    const service = new ecs.FargateService(this, 'LanguageToolService', {
      cluster,
      taskDefinition,
      desiredCount: 1,
      assignPublicIp: true,
      securityGroups: [ecsSecurityGroup],
      serviceName: `languagetool-service-${cdk.Aws.STACK_NAME}`,
      vpcSubnets: {
        subnetType: ec2.SubnetType.PUBLIC,
      },
    });

    const targetGroup = listener.addTargets('LanguageToolTargets', {
      port: 8081,
      protocol: elbv2.Protocol.TCP,
      targets: [
        service.loadBalancerTarget({
          containerName: 'LanguageToolContainer',
          containerPort: 8081,
        }),
      ],
    });

    targetGroup.configureHealthCheck({
      protocol: elbv2.Protocol.TCP,
      interval: cdk.Duration.seconds(30),
      timeout: cdk.Duration.seconds(10),
      healthyThresholdCount: 2,
      unhealthyThresholdCount: 2,
    });

    this.publicUrl = `http://${nlb.loadBalancerDnsName}`;

    new cdk.CfnOutput(this, 'LanguageToolPublicUrl', {
      value: this.publicUrl,
      description: 'LanguageTool Public URL via NLB',
    });

    new cdk.CfnOutput(this, 'LanguageToolVPCId', {
      value: this.vpc.vpcId,
      description: 'VPC ID for LanguageTool service',
    });
  }
}
