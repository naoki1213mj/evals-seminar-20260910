import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { VARIANTS } from "../../app/variants";

test("every selectable variant has a memory and slide-generation skill", () => {
  for (const variant of VARIANTS) {
    const root = path.join(process.cwd(), "workspaces", variant);
    assert.ok(fs.existsSync(path.join(root, "AGENTS.md")), variant);
    const skill = fs.readFileSync(
      path.join(root, ".agent", "skills", "pptx-generator", "SKILL.md"),
      "utf8",
    );
    assert.match(skill, /generate_pptx/);
    assert.match(skill, /計6枚/);
  }
});

test("the additional experiment preserves improvement-2 and adds an evidence gate", () => {
  const readSkill = (variant: string) =>
    fs.readFileSync(
      path.join(process.cwd(), "workspaces", variant, ".agent", "skills", "pptx-generator", "SKILL.md"),
      "utf8",
    ).replaceAll("\r\n", "\n").trim();
  const original = readSkill("improvement-2");
  const experiment = readSkill("improvement-3");
  assert.ok(experiment.startsWith(original));
  assert.match(experiment.slice(original.length), /\.evidence\.json/);
  assert.match(experiment.slice(original.length), /タイトル・サブタイトル/);
});
