export interface RelevanceCheck {
  addressesQuestion: boolean;
  score: number;
  threshold: number;
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error("Vectors must have the same length");
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  if (denominator === 0) {
    return 0;
  }

  return dotProduct / denominator;
}

function extractEmbedding(embedding: any): number[] {
  let vec: number[] = [];

  if (embedding && typeof embedding === "object") {
    if ("data" in embedding && Array.isArray(embedding.data)) {
      const data = embedding.data;
      if (data.length > 0 && Array.isArray(data[0])) {
        vec = (data as unknown as number[][]).flat();
      } else {
        vec = data as unknown as number[];
      }
    } else if ("shape" in embedding && "data" in embedding) {
      const data = embedding.data;
      if (data instanceof Float32Array) {
        vec = Array.from(data);
      } else if (Array.isArray(data)) {
        if (data.length > 0 && Array.isArray(data[0])) {
          vec = (data as unknown as number[][]).flat();
        } else {
          vec = Array.from(data as unknown as number[]);
        }
      }
    }
  }

  return vec;
}

export async function checkAnswerRelevance(
  ai: Ai,
  questionText: string,
  answerText: string,
  threshold: number = 0.5
): Promise<RelevanceCheck> {
  const embeddingModel = "@cf/baai/bge-base-en-v1.5";

  const [questionEmbedding, answerEmbedding] = await Promise.all([
    ai.run(embeddingModel, { text: [questionText] }),
    ai.run(embeddingModel, { text: [answerText] }),
  ]);

  const questionVec = extractEmbedding(questionEmbedding);
  const answerVec = extractEmbedding(answerEmbedding);

  if (questionVec.length === 0 || answerVec.length === 0) {
    throw new Error("Failed to extract embeddings from AI response");
  }

  const similarity = cosineSimilarity(questionVec, answerVec);
  const addressesQuestion = similarity >= threshold;

  return {
    addressesQuestion,
    score: similarity,
    threshold,
  };
}
