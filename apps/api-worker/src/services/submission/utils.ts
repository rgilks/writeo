import type { ModalRequest } from "@writeo/shared";

export type ModalAnswer = ModalRequest["parts"][number]["answers"][number];
type ModalPart = ModalRequest["parts"][number];

export function* iterateAnswers(modalParts: ModalRequest["parts"]): Generator<ModalAnswer> {
  for (const part of modalParts) {
    for (const answer of part.answers) {
      yield answer;
    }
  }
}

export function buildAnswerLookup(modalParts: ModalRequest["parts"]): Map<string, ModalAnswer> {
  const answersById = new Map<string, ModalAnswer>();
  for (const answer of iterateAnswers(modalParts)) {
    answersById.set(answer.id, answer);
  }
  return answersById;
}

export function buildPartLookup(
  modalParts: ModalRequest["parts"],
): Map<ModalPart["part"], ModalPart> {
  const partsById = new Map<ModalPart["part"], ModalPart>();
  for (const part of modalParts) {
    partsById.set(part.part, part);
  }
  return partsById;
}
