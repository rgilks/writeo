import type {
  CreateQuestionRequest,
  CreateAnswerRequest,
  CreateSubmissionRequest,
  AssessmentResults,
} from "@writeo/shared";

export class StorageService {
  constructor(
    private r2: R2Bucket,
    private kv: KVNamespace,
  ) {}

  async getQuestion(id: string): Promise<CreateQuestionRequest | null> {
    const obj = await this.r2.get(`questions/${id}.json`);
    if (!obj) return null;
    return obj.json<CreateQuestionRequest>();
  }

  async putQuestion(id: string, data: CreateQuestionRequest): Promise<void> {
    await this.r2.put(`questions/${id}.json`, JSON.stringify(data), {
      httpMetadata: { contentType: "application/json" },
    });
  }

  async getAnswer(id: string): Promise<CreateAnswerRequest | null> {
    const obj = await this.r2.get(`answers/${id}.json`);
    if (!obj) return null;
    return obj.json<CreateAnswerRequest>();
  }

  async putAnswer(id: string, data: CreateAnswerRequest): Promise<void> {
    await this.r2.put(`answers/${id}.json`, JSON.stringify(data), {
      httpMetadata: { contentType: "application/json" },
    });
  }

  async getSubmission(id: string): Promise<CreateSubmissionRequest | null> {
    const obj = await this.r2.get(`submissions/${id}.json`);
    if (!obj) return null;
    return obj.json<CreateSubmissionRequest>();
  }

  async putSubmission(id: string, data: CreateSubmissionRequest): Promise<void> {
    await this.r2.put(`submissions/${id}.json`, JSON.stringify(data), {
      httpMetadata: { contentType: "application/json" },
    });
  }

  async getResults(submissionId: string): Promise<AssessmentResults | null> {
    const json = await this.kv.get(`submission:${submissionId}`);
    if (!json) return null;
    return JSON.parse(json) as AssessmentResults;
  }

  async putResults(submissionId: string, results: AssessmentResults, ttl?: number): Promise<void> {
    await this.kv.put(
      `submission:${submissionId}`,
      JSON.stringify(results),
      ttl ? { expirationTtl: ttl } : undefined,
    );
  }
}
