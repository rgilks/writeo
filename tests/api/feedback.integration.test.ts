import { test, expect, describe } from "vitest";
import {
  apiRequest,
  generateIds,
  createSubmission,
  getAssessorResults,
  API_BASE,
  API_KEY,
} from "../helpers";
import { ASSESSOR_IDS } from "../constants";

describe("API Feedback Tests", () => {
  test.concurrent("teacher-feedback - persistence and modes", async () => {
    // Create submission - results returned immediately
    // Note: storeResults: true is required for GET endpoint to work
    const { status, json, submissionId, answerId } = await createSubmission(
      "Last weekend I go to the park with my friend. We was playing football.",
      "Describe your weekend. What did you do?",
      true, // Required for GET endpoint to retrieve results
    );
    expect(status).toBe(201);
    expect(json.status).toBe("success");

    // Request clues feedback
    const cluesResponse = await apiRequest(
      "POST",
      `/v1/text/submissions/${submissionId}/teacher-feedback`,
      {
        answerId,
        mode: "clues",
        answerText: "Last weekend I go to the park with my friend. We was playing football.",
      },
    );
    expect(cluesResponse.status).toBe(200);
    expect(cluesResponse.json.message).toBeDefined();
    const cluesMessage1 = cluesResponse.json.message;

    // Request clues again - should return the same stored feedback
    const cluesResponse2 = await apiRequest(
      "POST",
      `/v1/text/submissions/${submissionId}/teacher-feedback`,
      {
        answerId,
        mode: "clues",
        answerText: "Last weekend I go to the park with my friend. We was playing football.",
      },
    );
    expect(cluesResponse2.status).toBe(200);
    expect(cluesResponse2.json.message).toBe(cluesMessage1); // Should be identical (stored)

    // Request explanation feedback
    const explanationResponse = await apiRequest(
      "POST",
      `/v1/text/submissions/${submissionId}/teacher-feedback`,
      {
        answerId,
        mode: "explanation",
        answerText: "Last weekend I go to the park with my friend. We was playing football.",
      },
    );
    expect(explanationResponse.status).toBe(200);
    expect(explanationResponse.json.message).toBeDefined();
    const explanationMessage1 = explanationResponse.json.message;

    // Request explanation again - should return the same stored feedback
    const explanationResponse2 = await apiRequest(
      "POST",
      `/v1/text/submissions/${submissionId}/teacher-feedback`,
      {
        answerId,
        mode: "explanation",
        answerText: "Last weekend I go to the park with my friend. We was playing football.",
      },
    );
    expect(explanationResponse2.status).toBe(200);
    expect(explanationResponse2.json.message).toBe(explanationMessage1); // Should be identical (stored)

    // Verify stored feedback is in results (requires storeResults: true)
    const results = await apiRequest("GET", `/v1/text/submissions/${submissionId}`);
    expect(results.status).toBe(200);
    const teacherAssessor = getAssessorResults(results.json.results.parts[0]).find(
      (a: any) => a.id === ASSESSOR_IDS.TEACHER,
    );
    expect(teacherAssessor).toBeDefined();
    expect(teacherAssessor.meta).toBeDefined();
    expect(teacherAssessor.meta.cluesMessage).toBe(cluesMessage1);
    expect(teacherAssessor.meta.explanationMessage).toBe(explanationMessage1);
  });

  test.concurrent("streaming - AI feedback stream", async () => {
    const { questionId, answerId, submissionId } = generateIds();
    const questionText = "Describe your weekend. What did you do?";
    const answerText = "Last weekend I go to the park with my friend.";

    // Create submission - results returned immediately
    const { status, json } = await apiRequest("POST", `/v1/text/submissions`, {
      submissionId,
      submission: [
        {
          part: 1,
          answers: [
            {
              id: answerId,
              questionId: questionId,
              questionText: questionText,
              text: answerText,
            },
          ],
        },
      ],

      storeResults: false, // No server storage - using assessment data from response
    });
    expect(status).toBe(201);
    expect(json.status).toBe("success");

    // Extract assessment data from response
    const firstPart = json.results.parts[0];
    const firstAnswer = firstPart.answers[0];
    const assessorResults = firstAnswer.assessorResults || [];
    const essayAssessor = assessorResults.find((a: any) => a.id === "AES-ESSAY");
    const ltAssessor = assessorResults.find((a: any) => a.id === "GEC-LT");

    const requestBody: any = {
      answerId,
      answerText,
      questionText,
    };

    if (essayAssessor || ltAssessor) {
      requestBody.assessmentData = {
        essayScores: essayAssessor
          ? {
              overall: essayAssessor.overall,
              dimensions: essayAssessor.dimensions,
            }
          : undefined,
        ltErrors: ltAssessor?.errors || undefined,
      };
    }

    let chunks = 0;
    const response = await fetch(
      `${API_BASE}/v1/text/submissions/${submissionId}/ai-feedback/stream`,
      {
        method: "POST",
        headers: {
          Authorization: `Token ${API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      },
    );

    if (!response.body) {
      throw new Error("No response body");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let lastProcessedIndex = 0;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Decode and accumulate text
        buffer += decoder.decode(value, { stream: true });

        // Process new data only (from lastProcessedIndex onwards)
        const newData = buffer.slice(lastProcessedIndex);

        // Count all occurrences of "type":"chunk" in the new data
        const chunkMatches = newData.match(/"type":"chunk"/g);
        if (chunkMatches) {
          chunks += chunkMatches.length;
        }

        // Update last processed index
        lastProcessedIndex = buffer.length;

        // Check for done event
        if (buffer.includes('"type":"done"')) {
          break;
        }
      }
    } finally {
      reader.releaseLock();
    }

    expect(chunks).toBeGreaterThan(0);
  });

  test.concurrent("teacher-feedback - includes all assessment sources", async () => {
    const { questionId, answerId, submissionId } = generateIds();
    const questionText = "What are the benefits of exercise?";
    const answerText =
      "Exercise has many benefit. It improve physical health and mental well-being.";

    // Create submission with errors - use inline format
    const { status, json } = await apiRequest("POST", `/v1/text/submissions`, {
      submissionId,
      submission: [
        {
          part: 1,
          answers: [
            {
              id: answerId,
              questionId: questionId,
              questionText: questionText,
              text: answerText,
            },
          ],
        },
      ],

      storeResults: false, // No server storage - using assessment data from response
    });

    // If submission fails, log the error for debugging
    if (status !== 201) {
      console.error("Submission failed:", json);
    }

    expect(status).toBe(201);
    expect(json.status).toBe("success");

    // Extract assessment data from response
    const firstPart = json.results.parts[0];
    const firstAnswer = firstPart.answers[0];
    const assessorResults = firstAnswer.assessorResults || [];
    const essayAssessor = assessorResults.find((a: any) => a.id === "AES-ESSAY");
    const ltAssessor = assessorResults.find((a: any) => a.id === "GEC-LT");
    const llmAssessor = assessorResults.find((a: any) => a.id === "GEC-LLM");
    const relevanceAssessor = assessorResults.find((a: any) => a.id === "RELEVANCE-CHECK");

    const requestBody: any = {
      answerId,
      mode: "clues",
      answerText,
      questionText,
    };

    if (essayAssessor || ltAssessor || llmAssessor || relevanceAssessor) {
      requestBody.assessmentData = {
        essayScores: essayAssessor
          ? {
              overall: essayAssessor.overall,
              dimensions: essayAssessor.dimensions,
            }
          : undefined,
        ltErrors: ltAssessor?.errors || undefined,
        llmErrors: llmAssessor?.errors || undefined,
        relevanceCheck: relevanceAssessor?.meta
          ? {
              addressesQuestion: relevanceAssessor.meta.addressesQuestion ?? false,
              score: relevanceAssessor.meta.similarityScore ?? 0,
              threshold: relevanceAssessor.meta.threshold ?? 0.5,
            }
          : undefined,
      };
    }

    // Request teacher feedback - should include context from all sources
    // Note: API only accepts "clues" or "explanation" mode, not "initial"
    const teacherResponse = await apiRequest(
      "POST",
      `/v1/text/submissions/${submissionId}/teacher-feedback`,
      requestBody,
    );

    // Debug: log error if not 200
    if (teacherResponse.status !== 200) {
      console.error("Teacher feedback error:", {
        status: teacherResponse.status,
        json: teacherResponse.json,
        requestBody: JSON.stringify(requestBody).substring(0, 200),
      });
    }

    expect(teacherResponse.status).toBe(200);
    expect(teacherResponse.json.message).toBeDefined();

    // Verify feedback message is present and meaningful
    expect(typeof teacherResponse.json.message).toBe("string");
    expect(teacherResponse.json.message.length).toBeGreaterThan(0);

    // For "clues" mode, the response includes message and clues property
    // focusArea is only returned for "initial" mode (which is not supported via API)
    expect(teacherResponse.json.clues).toBeDefined();
    expect(typeof teacherResponse.json.clues).toBe("string");
    expect(teacherResponse.json.clues.length).toBeGreaterThan(0);

    // Verify the clues message matches the main message
    expect(teacherResponse.json.clues).toBe(teacherResponse.json.message);
  });
});
