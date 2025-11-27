import type { Context } from "hono";
import type { Env } from "../types/env";
import { buildConfig } from "../services/config";
import { StorageService } from "../services/storage";

export function getServices(c: Context<{ Bindings: Env; Variables: { requestId?: string } }>) {
  const config = buildConfig(c.env);
  return {
    config,
    storage: new StorageService(config.storage.r2Bucket, config.storage.kvNamespace),
  };
}
