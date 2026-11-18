// Run with: node --test tests/test_viewer_denoise.cjs
// Exercise the actual viewer filter without requiring WebGL or a camera.
const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../web_viewer/index.html'), 'utf8');
const filterCode = html.slice(html.indexOf('function denoiseHeight('),
                              html.indexOf('function updateMesh('));
const constants = ['TEMPORAL_ALPHA', 'NOISE_FLOOR_MM', 'USM_RADIUS', 'USM_AMOUNT']
  .map(name => html.match(new RegExp(`const ${name} = [^;]+;`))[0]).join('\n');

function viewer() {
  const context = vm.createContext({ state: { verts_per_row: 16, ema_height: null } });
  vm.runInContext(constants + '\n' + filterCode, context);
  return {
    state: context.state,
    frame(depth) { return context.denoiseHeight(new Float32Array(256).fill(depth)); },
  };
}

test('new shallow presses accumulate from a flat frame', () => {
  for (const depth of [0.06, 0.08, 0.10]) {
    const v = viewer();
    v.frame(0);
    for (let i = 0; i < 20; i++) v.frame(depth);
    assert.ok(Math.abs(v.frame(depth)[128] - depth) < 1e-5);
  }
});

test('display floor does not erase sub-threshold temporal history', () => {
  const v = viewer();
  assert.equal(v.frame(0.04)[128], 0);
  assert.ok(Math.abs(v.state.ema_height[128] - 0.04) < 1e-6);
  assert.ok(v.frame(0.07)[128] > 0.05);
});

test('background remains hidden and released presses disappear', () => {
  const v = viewer();
  v.frame(0);
  for (let i = 0; i < 20; i++) assert.equal(v.frame(0.02)[128], 0);
  for (let i = 0; i < 20; i++) v.frame(0.1);
  for (let i = 0; i < 20; i++) v.frame(0);
  assert.equal(v.frame(0)[128], 0);
  assert.ok(v.state.ema_height[128] < 1e-5);
});

test('reconnection reset discards the previous press', () => {
  const v = viewer();
  v.frame(0.1);
  v.state.ema_height = null;
  assert.equal(v.frame(0)[128], 0);
  assert.equal(v.state.ema_height[128], 0);
});
