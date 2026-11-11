import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { test } from 'node:test';

const source = readFileSync(new URL('../../src/snippets/autoregressive/qwen3-deployment.jsx', import.meta.url), 'utf8');
const setupEnd = source.indexOf('  const [values, setValues]');
const handleStart = source.indexOf('  const handleRadioChange');
const commandStart = source.indexOf('  const generateCommand');
const commandEnd = source.indexOf('  const containerStyle', commandStart);
assert.ok(setupEnd > 0 && handleStart > setupEnd && commandStart > handleStart && commandEnd > commandStart);
const render = new Function(
  'selection',
  `${source.slice(0, setupEnd).replace(/^export /, '')}
  const values = { ...getInitialState(), ...selection };
  let updated;
  const setValues = (updater) => { updated = updater(values); };
  ${source.slice(handleStart, commandStart)}
  ${source.slice(commandStart, commandEnd)}
  return {
    baseOptions, values, generateCommand, getDisplayOptions,
    update: (name, value) => { handleRadioChange(name, value); return updated; },
  };
};
return Qwen3Deployment();`,
);
const snapshot = new URL('qwen3-existing-commands.snapshot.json', import.meta.url);

test('existing commands remain identical for non-XPU and original BMG BF16 selections', () => {
  const { baseOptions, values } = render({});
  const names = Object.keys(baseOptions).filter((name) => name !== 'hardware');
  const result = {};
  for (const hardware of baseOptions.hardware.items) {
    const commands = [];
    const visit = (index, selection) => {
      if (index === names.length) {
        commands.push([selection, render(selection).generateCommand()]);
        return;
      }
      const name = names[index];
      const option = render(selection).getDisplayOptions(selection)[name];
      if (!option) {
        visit(index + 1, selection);
        return;
      }
      for (const item of option.items.filter((item) =>
        !item.disabled &&
        !(hardware.id === 'arc_b' && name === 'quantization' && item.id !== 'bf16') &&
        !(hardware.id === 'arc_b' && name === 'modelsize' && item.id === '14b'))) {
        visit(index + 1, { ...selection, [name]: item.id });
      }
    };
    visit(0, { ...values, hardware: hardware.id });
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

test('BMG 14B BF16 preserves its bounded TP4 recipe and rejects unsupported selections', () => {
  const state = render({
    hardware: 'h200', modelsize: '14b', category: 'base',
    quantization: 'bf16', reasoningParser: 'enabled', toolcall: 'enabled',
  }).update('hardware', 'arc_b');
  assert.equal(state.modelsize, '14b');
  assert.equal(state.reasoningParser, 'disabled');
  assert.equal(state.toolcall, 'disabled');
  const command = render(state).generateCommand();
  for (const flag of [
    '--model Qwen/Qwen3-14B', '--device xpu', '--attention-backend intel_xpu',
    '--tp 4', '--page-size 64', '--context-length 16384',
    '--mem-fraction-static 0.8', '--max-running-requests 4',
  ]) assert.ok(command.includes(flag), flag);
  for (const [name, value] of [
    ['quantization', 'fp8'], ['category', 'thinking'],
    ['reasoningParser', 'enabled'], ['toolcall', 'enabled'],
  ]) {
    assert.deepEqual(render(state).update(name, value), state);
    assert.match(render({ ...state, [name]: value }).generateCommand(), /Select the validated BMG 14B BF16/);
  }
});

test('BMG FP8 selects only the validated bounded 32B TP4 recipe', () => {
  const previous = { hardware: 'arc_b', modelsize: '32b', category: 'base', reasoningParser: 'enabled', toolcall: 'enabled' };
  const state = render(previous).update('quantization', 'fp8');
  assert.equal(state.reasoningParser, 'disabled');
  assert.equal(state.toolcall, 'disabled');
  assert.equal(render(state).generateCommand(), 'UR_L0_USE_COPY_ENGINE=0 sglang serve --model Qwen/Qwen3-32B-FP8 \\\n  --device xpu --attention-backend intel_xpu --tp 4 --page-size 64 \\\n  --context-length 16384 --mem-fraction-static 0.8 --max-running-requests 4');
  for (const [name, value] of [['reasoningParser', 'enabled'], ['toolcall', 'enabled'], ['category', 'thinking']]) {
    assert.deepEqual(render(state).update(name, value), state);
  }
  const moe = render(state).update('modelsize', '30b');
  assert.equal(moe.quantization, 'bf16');
  assert.deepEqual(render(moe).update('quantization', 'fp8'), moe);
  assert.match(render({ ...state, modelsize: '30b' }).generateCommand(), /Select the validated BMG 32B FP8/);
});
