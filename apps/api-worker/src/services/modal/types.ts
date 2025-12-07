import type { ModalRequest } from "@writeo/shared";

export interface ModalService {
  gradeEssay(request: ModalRequest): Promise<Response>;
  checkGrammar(text: string, language: string, answerId: string): Promise<Response>;
  scoreCorpus(text: string): Promise<Response>; // Corpus CEFR scoring (dev mode)
  scoreFeedback(text: string): Promise<Response>; // AES-FEEDBACK scoring (dev mode)
  scoreDeberta(text: string): Promise<Response>; // AES-DEBERTA scoring (dimensional)
  correctGrammar(text: string): Promise<Response>; // GEC-SEQ2SEQ correction
  correctGrammarGector(text: string): Promise<Response>; // GEC-GECTOR fast correction
}
