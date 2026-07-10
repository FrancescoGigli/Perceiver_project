import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const htmlPath = path.join(root, "perceiver_lezione.html");
const html = fs.readFileSync(htmlPath, "utf8");

const requiredVisuals = [
  "visual-byte-array",
  "visual-latent-array",
  "visual-attention-pipeline",
  "visual-output-head",
  "visual-permutation-result",
  "visual-backward-flow",
];

for (const id of requiredVisuals) {
  assert.match(html, new RegExp(`id="${id}"`), `Missing figure visual ${id}`);
}

assert.match(
  html,
  /interactive_trainer\/img\/fig1_architecture\.png/,
  "Architecture figure should use the sharper trainer asset",
);

assert.match(
  html,
  /interactive_trainer\/img\/fig3_attention_maps\.png/,
  "Results chapter should include attention maps from the trainer assets",
);

const figureCount = (html.match(/<figure/g) || []).length;
assert.ok(figureCount >= 14, `Expected at least 14 figures, found ${figureCount}`);

const imgSrcs = [...html.matchAll(/<img[^>]+src="([^"]+)"/g)].map((match) => match[1]);
assert.ok(imgSrcs.length >= 11, "Expected at least 11 raster image assets");

for (const src of imgSrcs) {
  if (/^https?:\/\//.test(src)) continue;
  const assetPath = path.resolve(root, src.replace(/^\.\//, ""));
  assert.ok(fs.existsSync(assetPath), `Missing image asset: ${src}`);
}

console.log(`perceiver_lezione checks passed: ${figureCount} figures, ${imgSrcs.length} image assets`);
