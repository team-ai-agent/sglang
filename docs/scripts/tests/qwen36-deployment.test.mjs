import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { test } from 'node:test';

const source = readFileSync(new URL('../../src/snippets/autoregressive/qwen36-deployment.jsx', import.meta.url), 'utf8');
const setupEnd = source.indexOf('  const [values, setValues]');
const commandStart = source.indexOf('  const generateCommand = () => {');
const commandEnd = source.indexOf('  const containerStyle', commandStart);
assert.ok(setupEnd > 0 && commandStart > setupEnd && commandEnd > commandStart);
const render = new Function(
  'selection',
  `${source.slice(0, setupEnd).replace(/^export /, '')}
  const values = { ...getInitialState(), ...selection };
  ${source.slice(commandStart, commandEnd)}
  return {
    options, values, generateCommand, getInitialState,
    getUpdatedValues: typeof getUpdatedValues === 'undefined' ? null : getUpdatedValues,
  };
};
return Qwen36Deployment();`,
);
const snapshot = new URL('qwen36-non-xpu-commands.snapshot.json', import.meta.url);
const initial = render({}).values;
const itemsOf = (option, selection) =>
  option.getDynamicItems ? option.getDynamicItems(selection) : option.items;

test('non-XPU commands remain byte-identical across every legal selector combination', () => {
  const { options } = render({});
  const result = {};
  for (const hardware of options.hardware.items.filter((item) => item.id !== 'arc_b')) {
    const commands = [];
    const names = Object.keys(options).filter((name) => name !== 'hardware');
    const visit = (index, selection) => {
      if (index === names.length) {
        commands.push([selection, render(selection).generateCommand()]);
        return;
      }
      const name = names[index];
      const option = options[name];
      if (option.condition && !option.condition(selection)) {
        visit(index + 1, selection);
        return;
      }
      for (const item of itemsOf(option, selection).filter((item) => !item.disabled)) {
        visit(index + 1, { ...selection, [name]: item.id });
      }
    };
    visit(0, { ...initial, hardware: hardware.id });
    result[hardware.id] = {
      count: commands.length,
      sha256: createHash('sha256').update(JSON.stringify(commands)).digest('hex'),
    };
  }
  if (process.env.UPDATE_SNAPSHOTS) {
    writeFileSync(snapshot, JSON.stringify(result, null, 2) + '\n');
  } else {
    assert.deepEqual(result, JSON.parse(readFileSync(snapshot, 'utf8')));
  }
});

test('BMG emits only the validated BF16 TP4 GDN recipe', () => {
  const state = render({}).getUpdatedValues(initial, 'hardware', 'arc_b');
  const { options, generateCommand } = render(state);
  assert.equal(state.modelSize, '35b-a3b');
  assert.equal(state.quantization, 'bf16');
  assert.equal(state.speculative, 'disabled');
  assert.equal(generateCommand(), 'sglang serve --model-path Qwen/Qwen3.6-35B-A3B \\\n  --device xpu --attention-backend intel_xpu \\\n  --tp 4 --linear-attn-backend intel_xpu --page-size 1 \\\n  --context-length 16384 --mem-fraction-static 0.8 --max-running-requests 4');
  for (const name of ['reasoning', 'toolcall', 'mambaCache']) {
    assert.equal(options[name].condition(state), false);
  }
  assert.deepEqual(itemsOf(options.quantization, state).map((item) => item.id), ['bf16']);
});

test('BMG resets incompatible state and rejects disabled selections', () => {
  const { getUpdatedValues } = render({});
  const previous = {
    ...initial, hardware: 'b300', modelSize: '27b', quantization: 'nvfp4',
    speculative: 'enabled', reasoning: 'enabled', toolcall: 'enabled', mambaCache: 'v2',
  };
  const bmg = getUpdatedValues(previous, 'hardware', 'arc_b');
  for (const [name, value] of [
    ['modelSize', '27b'], ['quantization', 'fp8'], ['quantization', 'nvfp4'],
    ['speculative', 'enabled'], ['mambaCache', 'v2'],
  ]) {
    assert.deepEqual(getUpdatedValues(bmg, name, value), bmg);
  }
  const h100 = getUpdatedValues(bmg, 'hardware', 'h100');
  assert.match(render(h100).generateCommand(), /--model-path Qwen\/Qwen3.6-35B-A3B/);
  assert.doesNotMatch(render(h100).generateCommand(), /--device xpu|intel_xpu|--page-size 1/);
});
