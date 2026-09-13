import assert from "node:assert/strict";
import { test } from "node:test";
import { z } from "zod";
import {
  createAgentModel,
  DEFAULT_AGENT_MODEL,
  getAgentModelConfig,
  normalizeFoundryEndpoint,
} from "../model-provider";
import { toModelUsageAttributes } from "../weave-agent-tracing";

const foundryEnvironment = {
  LLM_PROVIDER: "foundry",
  FOUNDRY_ENDPOINT: "https://example.services.ai.azure.com",
  FOUNDRY_API_KEY: "foundry-test-key",
  FOUNDRY_AGENT_DEPLOYMENT: "slide-agent-deployment",
};

test("OpenRouter remains the default and ignores inactive Foundry settings", () => {
  assert.deepEqual(
    getAgentModelConfig({
      OPENROUTER_API_KEY: "openrouter-test-key",
      FOUNDRY_ENDPOINT: "invalid",
    }),
    {
      provider: "openrouter",
      model: DEFAULT_AGENT_MODEL,
      apiKey: "openrouter-test-key",
    },
  );
});

test("OpenRouter allows an explicit agent model", () => {
  const config = getAgentModelConfig({
    LLM_PROVIDER: " OpenRouter ",
    OPENROUTER_API_KEY: "openrouter-test-key",
    OPENROUTER_AGENT_MODEL: " other/model ",
  });
  assert.equal(config.model, "other/model");
});

test("Foundry needs neither an OpenRouter key nor a judge deployment for the UI", () => {
  assert.deepEqual(getAgentModelConfig(foundryEnvironment), {
    provider: "foundry",
    model: "slide-agent-deployment",
    apiKey: "foundry-test-key",
    baseURL: "https://example.services.ai.azure.com/openai/v1/",
  });
});

test("configuration is read after imports, not captured at module load", (t) => {
  for (const [key, value] of Object.entries(foundryEnvironment)) {
    const previous = process.env[key];
    t.after(() => {
      if (previous === undefined) delete process.env[key];
      else process.env[key] = previous;
    });
    process.env[key] = value;
  }
  assert.equal(getAgentModelConfig().provider, "foundry");
});

test("unknown providers do not silently fall back to OpenRouter", () => {
  assert.throws(
    () => getAgentModelConfig({ LLM_PROVIDER: "unknown" }),
    /LLM_PROVIDER must be openrouter or foundry/,
  );
});

for (const key of [
  "FOUNDRY_ENDPOINT",
  "FOUNDRY_API_KEY",
  "FOUNDRY_AGENT_DEPLOYMENT",
]) {
  for (const value of ["", "  ", "your_placeholder_here", "<placeholder>"]) {
    test(`Foundry rejects an unset/placeholder ${key}: ${JSON.stringify(value)}`, () => {
      assert.throws(
        () => getAgentModelConfig({ ...foundryEnvironment, [key]: value }),
        new RegExp(`Set ${key}`),
      );
    });
  }
}

test("OpenRouter rejects its placeholder key", () => {
  assert.throws(
    () => getAgentModelConfig({ OPENROUTER_API_KEY: "your_openrouter_api_key_here" }),
    /Set OPENROUTER_API_KEY/,
  );
});

for (const suffix of ["", "/", "/openai/v1", "/openai/v1/"]) {
  test(`Foundry normalizes endpoint suffix ${JSON.stringify(suffix)}`, () => {
    assert.equal(
      normalizeFoundryEndpoint(`https://example.openai.azure.com${suffix}`),
      "https://example.openai.azure.com/openai/v1/",
    );
  });
}

for (const endpoint of [
  "not-a-url",
  "http://example.openai.azure.com",
  "https://user:secret@example.openai.azure.com",
  "https://example.openai.azure.com?api-key=secret",
  "https://example.openai.azure.com?",
  "https://example.openai.azure.com#secret",
  "https://example.openai.azure.com#",
  "https://@example.openai.azure.com",
  "https://example.openai.azure.com\\openai\\v1",
  "https://example.\nopenai.azure.com",
  "https://example.\u0000openai.azure.com",
  "https://example.\u007fopenai.azure.com",
  "https://example.openai.azure.com/other/../",
  "https://example.openai.azure.com:99999",
  "https://example.services.ai.azure.com/api/projects/demo",
  "https://example.openai.azure.com/openai/deployments/demo",
]) {
  test(`Foundry rejects an invalid endpoint without exposing it: ${JSON.stringify(endpoint)}`, () => {
    assert.throws(
      () => normalizeFoundryEndpoint(endpoint),
      (error: unknown) =>
        error instanceof Error &&
        error.message.includes("FOUNDRY_ENDPOINT must be") &&
        !error.message.includes("secret") &&
        !error.message.includes(endpoint),
    );
  });
}

for (const provider of ["openrouter", "foundry"] as const) {
  test(`${provider} does not serialize its API key into the model definition`, () => {
    const config = getAgentModelConfig(
      provider === "foundry"
        ? foundryEnvironment
        : { OPENROUTER_API_KEY: "openrouter-test-key" },
    );
    const { model } = createAgentModel(config);
    assert.equal(JSON.stringify(model.toJSON()).includes(config.apiKey), false);
  });

  for (const streaming of [false, true]) {
    test(`${provider} routes ${streaming ? "streaming" : "non-streaming"} tool-capable calls without network access`, async (t) => {
      const requests: Request[] = [];
      const config = getAgentModelConfig(
        provider === "foundry"
          ? foundryEnvironment
          : { OPENROUTER_API_KEY: "openrouter-test-key" },
      );
      const usage = {
        prompt_tokens: 2,
        completion_tokens: 3,
        total_tokens: 5,
        cost: 0.001,
      };
      t.mock.method(globalThis, "fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
        requests.push(new Request(input, init));
        if (streaming) {
          const chunks = [
            {
              id: "chatcmpl-test",
              object: "chat.completion.chunk",
              created: 1,
              model: config.model,
              choices: [{ index: 0, delta: { role: "assistant", content: "ok" }, finish_reason: null }],
            },
            {
              id: "chatcmpl-test",
              object: "chat.completion.chunk",
              created: 1,
              model: config.model,
              choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
              usage,
            },
          ];
          return new Response(
            chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("") +
              "data: [DONE]\n\n",
            { headers: { "Content-Type": "text/event-stream" } },
          );
        }
        return new Response(
          JSON.stringify({
            id: "chatcmpl-test",
            object: "chat.completion",
            created: 1,
            model: config.model,
            choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
            usage,
          }),
          { headers: { "Content-Type": "application/json" } },
        );
      });

      const { model, modelName, provider: selectedProvider } = createAgentModel(config);
      const boundModel = model.bindTools([
        {
          name: "generate_pptx",
          description: "Generate a presentation from a JSON file.",
          schema: z.object({ file_path: z.string() }),
        },
      ]);
      if (streaming) {
        let text = "";
        for await (const chunk of await boundModel.stream("test")) {
          text += chunk.content;
        }
        assert.equal(text, "ok");
      } else {
        const response = await boundModel.invoke("test");
        assert.equal(response.content, "ok");
        assert.equal(response.usage_metadata?.total_tokens, 5);
        const attributes = toModelUsageAttributes(response, provider);
        assert.equal(attributes["gen_ai.usage.total_tokens"], 5);
        assert.equal(
          attributes["openrouter.usage.cost"],
          provider === "openrouter" ? 0.001 : undefined,
        );
      }
      assert.equal(selectedProvider, provider);
      assert.equal(modelName, config.model);
      assert.equal(requests.length, 1);
      const request = requests[0];
      assert.equal(
        request.url,
        provider === "foundry"
          ? "https://example.services.ai.azure.com/openai/v1/chat/completions"
          : "https://openrouter.ai/api/v1/chat/completions",
      );
      assert.equal(request.headers.get("Authorization"), `Bearer ${config.apiKey}`);
      const body = await request.json();
      assert.equal(body.model, config.model);
      assert.equal(body.tools[0].function.name, "generate_pptx");
      assert.equal(Boolean(body.stream), streaming);
    });
  }
}
