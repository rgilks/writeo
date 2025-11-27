import { z, type ZodError } from "zod";

export function uuidStringSchema(fieldName: string) {
  return z.string().uuid(`Invalid ${fieldName} format`);
}

export function formatZodMessage(error: ZodError, fallbackMessage: string): string {
  return error.issues[0]?.message ?? fallbackMessage;
}
