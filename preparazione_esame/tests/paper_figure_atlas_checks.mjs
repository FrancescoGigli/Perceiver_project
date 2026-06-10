import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const appDir = path.join(root, "perceiver_condivisibile", "perceiver_interattivo");
const indexPath = path.join(appDir, "index.html");
const stylePath = path.join(appDir, "css", "interactive-labs.css");

assert.ok(fs.existsSync(indexPath), "Missing shareable perceiver_interattivo/index.html");
assert.ok(fs.existsSync(stylePath), "Missing shareable interactive-labs.css");

const index = fs.readFileSync(indexPath, "utf8");
const style = fs.readFileSync(stylePath, "utf8");

assert.match(index, /paper-figure-atlas/, "Bibliografia should include the paper figure atlas");
assert.match(style, /\.paper-figure-atlas\b/, "Atlas styles should live in interactive-labs.css");
assert.match(style, /\.paper-figure-card\b/, "Figure card styles should be defined");

const figureCards = index.match(/class="paper-figure-card"/g) || [];
assert.ok(figureCards.length >= 55, `Expected at least 55 representative figure cards, found ${figureCards.length}`);
assert.ok(figureCards.length <= 95, `Expected a curated gallery, found ${figureCards.length} cards`);

const imageSources = [...index.matchAll(/<img[^>]+src="(paper_figures\/[^"]+)"/g)].map((match) => match[1]);
assert.equal(imageSources.length, figureCards.length, "Each atlas card should contain one local figure image");
assert.equal(new Set(imageSources).size, imageSources.length, "Atlas image sources should be unique");

for (const src of imageSources) {
  const file = path.join(appDir, src.replaceAll("/", path.sep));
  assert.ok(fs.existsSync(file), `Missing atlas image asset: ${src}`);
}

const lazyImages = index.match(/<img[^>]+src="paper_figures\/[^"]+"[^>]+loading="lazy"/g) || [];
assert.equal(lazyImages.length, imageSources.length, "Atlas images should be lazy loaded");
assert.doesNotMatch(index, /src="C:\\Users\\/i, "HTML should not reference Desktop absolute paths");
assert.match(index, /Atlante visivo dai paper/, "Atlas heading should be present and visible");
assert.match(index, /Didascalie estratte/, "Atlas should explain where descriptions come from");

console.log("paper figure atlas checks passed");
