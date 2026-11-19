const assert = require('node:assert/strict');
const { readFileSync, statSync } = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

test('i link condivisibili del ripasso portano a figure locali e passi degli appunti', () => {
  const studio = path.resolve(__dirname, '../../sito/studio');
  const course = JSON.parse(readFileSync(path.join(studio, 'media/course-project.json'), 'utf8'));
  const window = {};
  vm.runInNewContext(readFileSync(path.join(studio, 'ripasso-tappe.js'), 'utf8'), { window });
  const stages = window.perceiverRipasso;
  const slugs = new Set();
  for (const stage of stages) {
    assert.match(stage.slug, /^[a-z0-9]+(?:-[a-z0-9]+)*$/);
    assert.ok(!slugs.has(stage.slug) && stage.slug !== 'inizia', stage.slug);
    slugs.add(stage.slug);
    const chapter = course.chapters.find(item => item.id === stage.chapter);
    assert.ok(chapter?.steps.some(step => step.id === stage.stepId), stage.stepId);
    assert.ok(statSync(path.resolve(studio, stage.figure)).isFile(), stage.figure);
    assert.ok(stage.alt && stage.caption && stage.explanation && stage.question && stage.answer, stage.slug);
    assert.ok(stage.flow.length >= 2 && stage.flow.every(label => label.trim()), stage.slug);
  }
  assert.ok(slugs.has('cross-attention'), 'collegamento alla lezione animata');
});
