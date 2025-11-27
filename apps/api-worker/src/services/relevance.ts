export interface RelevanceCheck {
  addressesQuestion: boolean;
  score: number;
  threshold: number;
}

/** Default similarity threshold for determining if an answer addresses the question */
const DEFAULT_THRESHOLD = 0.5;
/** Cloudflare Workers AI embedding model for semantic similarity */
const EMBEDDING_MODEL = "@cf/baai/bge-base-en-v1.5";

interface EmbeddingWithData {
  data: unknown;
  shape?: unknown;
}

/**
 * Calculates cosine similarity between two vectors.
 * Returns a value between -1 and 1, where 1 indicates identical vectors.
 */
function cosineSimilarity(vectorA: number[], vectorB: number[]): number {
  if (vectorA.length !== vectorB.length) {
    throw new Error("Vectors must have the same length");
  }

  let dotProduct = 0;
  let sumSquaresA = 0;
  let sumSquaresB = 0;

  for (let i = 0; i < vectorA.length; i++) {
    const aVal = vectorA[i] ?? 0;
    const bVal = vectorB[i] ?? 0;
    dotProduct += aVal * bVal;
    sumSquaresA += aVal * aVal;
    sumSquaresB += bVal * bVal;
  }

  const normA = Math.sqrt(sumSquaresA);
  const normB = Math.sqrt(sumSquaresB);
  const denominator = normA * normB;

  if (denominator === 0) {
    return 0;
  }

  return dotProduct / denominator;
}

function flattenNestedArray(data: unknown): number[] {
  if (Array.isArray(data)) {
    if (data.length > 0 && Array.isArray(data[0])) {
      return (data as number[][]).flat();
    }
    return data as number[];
  }
  return [];
}

function extractEmbeddingFromData(data: unknown): number[] {
  if (data instanceof Float32Array) {
    return Array.from(data);
  }
  if (Array.isArray(data)) {
    return flattenNestedArray(data);
  }
  return [];
}

function extractEmbedding(embedding: unknown): number[] {
  if (!embedding || typeof embedding !== "object") {
    return [];
  }

  const embeddingObj = embedding as EmbeddingWithData;

  if ("data" in embeddingObj) {
    return extractEmbeddingFromData(embeddingObj.data);
  }

  return [];
}

export async function checkAnswerRelevance(
  ai: Ai,
  questionText: string,
  answerText: string,
  threshold: number = DEFAULT_THRESHOLD,
): Promise<RelevanceCheck> {
  const [questionEmbedding, answerEmbedding] = await Promise.all([
    ai.run(EMBEDDING_MODEL, { text: [questionText] }),
    ai.run(EMBEDDING_MODEL, { text: [answerText] }),
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
