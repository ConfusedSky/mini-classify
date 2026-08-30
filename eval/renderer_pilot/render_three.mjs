// Arms: model-browser's three.js thumbnail chain via the dev hook, three
// variants per model: shipped (GTAO + red/blue rims), noao, white rims.
// Output renders/three_<variant>/<key>_v<i>.png (RGBA, transparent bg).
import { createRequire } from 'node:module';
import { mkdirSync, existsSync, writeFileSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const req = createRequire(process.env.HOME + '/.npm/_npx/a5b920f00216d246/node_modules/');
const { chromium } = req('playwright-core');
const S = JSON.parse(readFileSync(join(HERE, 'sample.json'), 'utf8'));
if (process.argv[2]) S.sample = S.sample.filter((m) => m.key.includes(process.argv[2]));
const VARIANTS = { ao: { ao: true }, noao: { ao: false }, white: { ao: true, neutralRims: true } };
// STL up (file axes) -> OrbitAxis: models.ts bakes rotateX(-pi/2), (x,y,z) -> (x,z,-y).
const AXIS = { '0,0,1': 'y', '0,0,-1': '-y', '0,1,0': '-z', '0,-1,0': 'z', '1,0,0': 'x', '-1,0,0': '-x' };
const DIST_R = 2.6;
for (const v of Object.keys(VARIANTS)) mkdirSync(join(HERE, 'renders', `three_${v}`), { recursive: true });

const browser = await chromium.launch({
  executablePath: process.env.HOME + '/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome',
  headless: true,
  args: ['--ignore-gpu-blocklist', '--use-gl=angle', '--use-angle=gl'],
});
const page = await browser.newPage({ viewport: { width: 800, height: 600 } });
page.on('pageerror', (e) => console.log('pageerror', e.message));
await page.goto('http://localhost:5173/', { waitUntil: 'domcontentloaded' });
await page.waitForFunction(() => typeof window.__pilot !== 'undefined', null, { timeout: 60000 });
console.log('hook ready; GL =', await page.evaluate(() => {
  const gl = document.createElement('canvas').getContext('webgl2');
  const d = gl.getExtension('WEBGL_debug_renderer_info');
  return gl.getParameter(d.UNMASKED_RENDERER_WEBGL);
}));

const states = S.angles.map(([az, el]) => ({ az, el, distR: DIST_R, target: [0, 0, 0] }));
const t0 = Date.now();
let n = 0;
for (const m of S.sample) {
  n++;
  const done = Object.keys(VARIANTS).every((v) =>
    S.angles.every((_, i) => existsSync(join(HERE, 'renders', `three_${v}`, `${m.key}_v${i}.png`))));
  if (done) continue;
  const axis = AXIS[m.up.map((c) => Math.round(c)).join(',')];
  try {
    const out = await page.evaluate(
      ({ path, axis, states, variants }) => window.__pilot.render(path, axis, states, variants),
      { path: '/' + m.rel, axis, states, variants: VARIANTS },
    );
    for (const [v, pngs] of Object.entries(out)) {
      pngs.forEach((b64, i) =>
        writeFileSync(join(HERE, 'renders', `three_${v}`, `${m.key}_v${i}.png`), Buffer.from(b64, 'base64')));
    }
  } catch (e) {
    console.log(`FAIL ${m.key}: ${String(e.message).slice(0, 200)}`);
  }
  if (n % 20 === 0) console.log(`[${n}/${S.sample.length}] ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
console.log(`done ${((Date.now() - t0) / 1000).toFixed(0)}s`);
await browser.close();
