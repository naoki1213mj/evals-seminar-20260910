import assert from "node:assert/strict";
import { test } from "node:test";
import { AIMessage } from "@langchain/core/messages";
import { toModelUsageAttributes } from "../weave-agent-tracing";

test("Foundry token usage is recorded without OpenRouter pricing attributes", () => {
  const response = new AIMessage({
    content: "ok",
    usage_metadata: { input_tokens: 3, output_tokens: 4, total_tokens: 7 },
    response_metadata: {
      usage: { cost: 10, cost_details: { upstream_inference_cost: 5 } },
    },
  });
  assert.deepEqual(toModelUsageAttributes(response, "foundry"), {
    "gen_ai.usage.total_tokens": 7,
  });
});

test("token totals do not require raw provider-specific usage", () => {
  const response = new AIMessage({
    content: "ok",
    usage_metadata: { input_tokens: 3, output_tokens: 4, total_tokens: 7 },
  });
  for (const provider of ["foundry", "openrouter"] as const) {
    assert.deepEqual(toModelUsageAttributes(response, provider), {
      "gen_ai.usage.total_tokens": 7,
    });
  }
});

test("OpenRouter still retains its actual cost and upstream cost", () => {
  const response = new AIMessage({
    content: "ok",
    response_metadata: {
      usage: {
        total_tokens: 7,
        cost: 0.01,
        cost_details: { upstream_inference_cost: 0.008 },
      },
    },
  });
  assert.deepEqual(toModelUsageAttributes(response, "openrouter"), {
    "gen_ai.usage.total_tokens": 7,
    "openrouter.usage.cost": 0.01,
    "openrouter.usage.upstream_inference_cost": 0.008,
  });
});

test("missing usage and nonfinite charges are not fabricated as zero", () => {
  assert.deepEqual(
    toModelUsageAttributes(new AIMessage({ content: "ok" }), "foundry"),
    {},
  );
  assert.deepEqual(
    toModelUsageAttributes(
      new AIMessage({
        content: "ok",
        response_metadata: { usage: { total_tokens: NaN, cost: Infinity } },
      }),
      "openrouter",
    ),
    {},
  );
});
