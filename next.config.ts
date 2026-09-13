import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Docker listens on 0.0.0.0, while the host browser uses the loopback address.
  allowedDevOrigins: ["127.0.0.1"],
  // workspaces/配下のAGENTS.mdは教材の一部のため、Nextによる自動生成と紛れないよう無効化
  agentRules: false,
  serverExternalPackages: [
    "deepagents",
    "@langchain/core",
    "@langchain/langgraph",
    "@langchain/langgraph-checkpoint",
    "@langchain/openrouter",
    "langchain-copilotkit",
    "weave",
  ],
};

export default nextConfig;
