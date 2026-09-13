import path from "node:path";
import { MemorySaver } from "@langchain/langgraph-checkpoint";
import { createDeepAgent, LocalShellBackend } from "deepagents";
import { createGeneratePptxTool } from "./generate-pptx-tool";
import { createAgentModel } from "./model-provider";
import { SYSTEM_PROMPT } from "./system-prompt";
import {
  createWeaveAgentTraceMiddleware,
  weaveTraceContextSchema,
} from "./weave-agent-tracing";

export async function createSlideAgent(workspaceDir: string) {
  const { model, modelName, provider } = createAgentModel();

  const backend = await LocalShellBackend.create({
    rootDir: path.resolve(workspaceDir),
    virtualMode: true,
    inheritEnv: true,
  });

  const agent = createDeepAgent({
    model,
    name: "slide-generator",
    systemPrompt: SYSTEM_PROMPT,
    tools: [createGeneratePptxTool(backend)],
    middleware: [createWeaveAgentTraceMiddleware()],
    contextSchema: weaveTraceContextSchema,
    skills: ["./.agent/skills/"],
    memory: ["./AGENTS.md"],
    backend,
    checkpointer: new MemorySaver(),
  });

  return { agent, backend, modelName, provider };
}
