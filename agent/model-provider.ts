import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import type { BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import { ChatOpenAI } from "@langchain/openai";
import { ChatOpenRouter } from "@langchain/openrouter";

export type LlmProvider = "openrouter" | "foundry";
type Environment = Readonly<Record<string, string | undefined>>;

export type AgentModelConfig =
  | { provider: "openrouter"; model: string; apiKey: string }
  | { provider: "foundry"; model: string; apiKey: string; baseURL: string };

export const DEFAULT_AGENT_MODEL = "deepseek/deepseek-v4-flash";

function requiredValue(env: Environment, name: string): string {
  const value = env[name]?.trim();
  if (!value || value.toLowerCase().startsWith("your_") || /[<>]/.test(value)) {
    throw new Error(`Set ${name} in .env before running the agent.`);
  }
  return value;
}

export function normalizeFoundryEndpoint(endpoint: string): string {
  const message =
    "FOUNDRY_ENDPOINT must be an HTTPS resource URL or /openai/v1/ endpoint, without credentials, query parameters or fragments.";
  if (/[\\?#@\s\u0000-\u001f\u007f]/.test(endpoint)) {
    throw new Error(message);
  }
  const pathStart = endpoint.indexOf("/", endpoint.indexOf("://") + 3);
  const rawPath = pathStart < 0 ? "" : endpoint.slice(pathStart);
  let url: URL;
  try {
    url = new URL(endpoint);
  } catch {
    throw new Error(message);
  }
  if (
    url.protocol !== "https:" ||
    url.username ||
    url.password ||
    url.search ||
    url.hash ||
    !["", "/", "/openai/v1", "/openai/v1/"].includes(rawPath) ||
    !["/", "/openai/v1", "/openai/v1/"].includes(url.pathname)
  ) {
    throw new Error(message);
  }
  return `${url.origin}/openai/v1/`;
}

// Read lazily: CLI entrypoints load .env after their imports have been evaluated.
export function getAgentModelConfig(
  env: Environment = process.env,
): AgentModelConfig {
  const provider = env.LLM_PROVIDER?.trim().toLowerCase() || "openrouter";
  if (provider === "openrouter") {
    return {
      provider,
      model: env.OPENROUTER_AGENT_MODEL?.trim() || DEFAULT_AGENT_MODEL,
      apiKey: requiredValue(env, "OPENROUTER_API_KEY"),
    };
  }
  if (provider === "foundry") {
    return {
      provider,
      model: requiredValue(env, "FOUNDRY_AGENT_DEPLOYMENT"),
      apiKey: requiredValue(env, "FOUNDRY_API_KEY"),
      baseURL: normalizeFoundryEndpoint(requiredValue(env, "FOUNDRY_ENDPOINT")),
    };
  }
  throw new Error("LLM_PROVIDER must be openrouter or foundry.");
}

/** Preserve OpenRouter's raw usage/cost on non-streaming LangChain calls. */
class CostTrackingChatOpenRouter extends ChatOpenRouter {
  override async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    const result = await super._generate(messages, options, runManager);
    const usage = result.llmOutput?.tokenUsage;
    if (usage && typeof usage === "object") {
      for (const generation of result.generations) {
        generation.message.response_metadata = {
          ...generation.message.response_metadata,
          usage,
        };
      }
    }
    return result;
  }
}

export function createAgentModel(config = getAgentModelConfig()) {
  const model =
    config.provider === "foundry"
      ? new ChatOpenAI({
          model: config.model,
          apiKey: config.apiKey,
          configuration: { baseURL: config.baseURL },
          useResponsesApi: false,
          streamUsage: true,
        })
      : new CostTrackingChatOpenRouter({
          model: config.model,
          apiKey: config.apiKey,
        });
  return { model, modelName: config.model, provider: config.provider };
}
