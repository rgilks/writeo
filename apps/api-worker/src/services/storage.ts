import type {
  CreateQuestionRequest,
  CreateAnswerRequest,
  CreateSubmissionRequest,
  AssessmentResults,
} from "@writeo/shared";

const JSON_CONTENT_TYPE = "application/json";
const R2_PATHS = {
  questions: "questions",
  answers: "answers",
  submissions: "submissions",
} as const;

const KV_PREFIX = "submission:";

export class StorageService {
  constructor(
    private r2: R2Bucket,
    private kv: KVNamespace,
  ) {}

  private async getR2Object<T>(path: string): Promise<T | null> {
    const obj = await this.r2.get(path);
    if (!obj) return null;
    return obj.json<T>();
  }

  private async putR2Object(path: string, data: unknown): Promise<void> {
    await this.r2.put(path, JSON.stringify(data), {
      httpMetadata: { contentType: JSON_CONTENT_TYPE },
    });
  }

  private buildR2Path(prefix: string, id: string): string {
    return `${prefix}/${id}.json`;
  }

  async getQuestion(id: string): Promise<CreateQuestionRequest | null> {
    return this.getR2Object<CreateQuestionRequest>(this.buildR2Path(R2_PATHS.questions, id));
  }

  async putQuestion(id: string, data: CreateQuestionRequest): Promise<void> {
    await this.putR2Object(this.buildR2Path(R2_PATHS.questions, id), data);
  }

  async getAnswer(id: string): Promise<CreateAnswerRequest | null> {
    return this.getR2Object<CreateAnswerRequest>(this.buildR2Path(R2_PATHS.answers, id));
  }

  async putAnswer(id: string, data: CreateAnswerRequest): Promise<void> {
    await this.putR2Object(this.buildR2Path(R2_PATHS.answers, id), data);
  }

  async getSubmission(id: string): Promise<CreateSubmissionRequest | null> {
    return this.getR2Object<CreateSubmissionRequest>(this.buildR2Path(R2_PATHS.submissions, id));
  }

  async putSubmission(id: string, data: CreateSubmissionRequest): Promise<void> {
    await this.putR2Object(this.buildR2Path(R2_PATHS.submissions, id), data);
  }

  async getResults(submissionId: string): Promise<AssessmentResults | null> {
    const json = await this.kv.get(`${KV_PREFIX}${submissionId}`);
    if (!json) return null;
    return JSON.parse(json) as AssessmentResults;
  }

  async putResults(submissionId: string, results: AssessmentResults, ttl?: number): Promise<void> {
    await this.kv.put(
      `${KV_PREFIX}${submissionId}`,
      JSON.stringify(results),
      ttl ? { expirationTtl: ttl } : undefined,
    );
  }
}
