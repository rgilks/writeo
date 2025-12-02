import type { AppConfig } from "../config";
import type { ModalService } from "./types";
import { ModalClient } from "./client";
import { MockModalClient } from "./mock";

export function createModalService(config: AppConfig): ModalService {
  if (config.features.mockServices) {
    return new MockModalClient();
  }
  return new ModalClient(config);
}
