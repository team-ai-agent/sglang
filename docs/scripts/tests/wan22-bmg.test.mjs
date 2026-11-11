import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../../src/snippets/configs/Wan/wan22.jsx", import.meta.url), "utf8");
const { config } = await import("data:text/javascript," + encodeURIComponent(source));
const selection = (overrides = {}) => ({ ...config.commandBuilder.defaultSelection, ...overrides });

test("existing non-XPU Wan commands and requests remain identical", () => {
  const cases = [];
  for (const hw of config.supportedHardware.filter((id) => id !== "arc_b"))
    for (const variant of ["t2v-14b", "i2v-14b", "ti2v-5b"])
      for (const gpus_per_node of [1, 2, 4, 8])
        for (const lora of ["none", "t2v-14b", "i2v-14b"]) {
          const s = selection({ hw, variant, gpus_per_node, lora });
          cases.push({ s, result: config.commandBuilder.resolveDeployment(s), curl: config.curl(s) });
        }
  assert.equal(cases.length, 324);
  assert.equal(
    createHash("sha256").update(JSON.stringify(cases)).digest("hex"),
    "de95a427376a2112eedf2ff4ddf43acba2c025e92f02eba3b5782618cc0cd967",
  );
});

test("BMG emits exactly the checked resident TP4 recipe and official video payload", () => {
  const recipe = config.commandBuilder.resource.verifiedRecipes.find((entry) => entry.hw === "arc_b");
  assert.equal(recipe.variant, "ti2v-5b");
  assert.equal(recipe.gpus_per_node, 4);
  const result = config.commandBuilder.resolveDeployment(recipe);
  assert.deepEqual(result.builder.errors, []);
  assert.deepEqual(result.flags, [
    "--model-path Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "--num-gpus 4", "--tp-size 4", "--attention-backend fa",
    "--component-residency all=resident", "--port {{PORT}}",
  ]);
  assert.deepEqual(result.env, [
    "UR_L0_USE_COPY_ENGINE=0", "UR_L0_USE_IMMEDIATE_COMMANDLISTS=0",
    "UR_L0_USE_DRIVER_COUNTER_BASED_EVENTS=0", "UR_L0_DISABLE_EVENTS_CACHING=1",
    "SGLANG_DIFFUSION_PLATFORM_OVERRIDE=xpu",
  ]);
  assert.deepEqual(result.builder.topology, { tp_size: 4, ulysses_degree: 1, ring_degree: 1 });
  assert.deepEqual(JSON.parse(config.curl(recipe).match(/-d '([\s\S]*)'/)[1]), {
    model: "Wan-AI/Wan2.2-TI2V-5B-Diffusers", prompt: "A quiet street at dusk",
  });
  assert.deepEqual(config.runModes, ["python"]);
});

test("BMG rejects unvalidated checkpoints, LoRA, GPU counts and topology", () => {
  const recipe = config.commandBuilder.resource.verifiedRecipes.find((entry) => entry.hw === "arc_b");
  for (const overrides of [
    { variant: "t2v-14b" }, { variant: "i2v-14b" }, { lora: "t2v-14b" },
    { gpus_per_node: 2 }, { topology_mode: "manual", tp_size: 2 },
  ]) {
    const result = config.commandBuilder.resolveDeployment({ ...recipe, ...overrides });
    assert.ok(result.builder.errors.length);
    assert.deepEqual(result.builder.verification, { serve: "error", request: "error" });
    assert.ok(!result.flags.some((flag) => /lora|ulysses|ring|cfg-parallel|compile/.test(flag)));
    assert.equal(result.flags[0], "--model-path Wan-AI/Wan2.2-TI2V-5B-Diffusers");
  }
  const variants = config.overlayDims.find((dim) => dim.id === "variant").options;
  assert.deepEqual(variants.filter((option) => !option.disabled?.(recipe)).map((option) => option.id), ["ti2v-5b"]);
  const loras = config.overlayDims.find((dim) => dim.id === "lora").options;
  assert.deepEqual(loras.filter((option) => !option.showWhen || option.showWhen(recipe)).map((option) => option.id), ["none"]);
  assert.deepEqual(config.commandBuilder.resource.limits.gpus_per_node.allowedValues(recipe), [4]);
});
