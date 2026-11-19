const assert = require('node:assert/strict');
const { readFileSync, statSync } = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

function loadStudy(studio) {
  const window = {};
  for (const name of ['ripasso-basi.js', 'ripasso-tappe.js', 'ripasso-scene.js', 'ripasso-codice.js']) {
    vm.runInNewContext(readFileSync(path.join(studio, name), 'utf8'), { window });
  }
  return { stages: [...window.perceiverBasi, ...window.perceiverRipasso],
           projectFrames: window.perceiverSceneProgetto, lessons: window.perceiverRipassoCodice };
}

test('i link condivisibili del ripasso portano a figure locali e passi degli appunti', () => {
  const studio = path.resolve(__dirname, '../../sito/studio');
  const course = JSON.parse(readFileSync(path.join(studio, 'media/course-project.json'), 'utf8'));
  const { stages } = loadStudy(studio);
  const slugs = new Set();
  for (const stage of stages) {
    assert.match(stage.slug, /^[a-z0-9]+(?:-[a-z0-9]+)*$/);
    assert.ok(!slugs.has(stage.slug) && stage.slug !== 'inizia', stage.slug);
    slugs.add(stage.slug);
    const chapter = course.chapters.find(item => item.id === stage.chapter);
    assert.ok(chapter?.steps.some(step => step.id === stage.stepId), stage.stepId);
    assert.ok(statSync(path.resolve(studio, stage.figure)).isFile(), stage.figure);
    assert.ok(stage.intro && stage.alt && stage.caption && stage.explanation && stage.question && stage.answer, stage.slug);
    assert.ok(stage.flow.length >= 2 && stage.flow.every(label => label.trim()), stage.slug);
  }
  assert.ok(slugs.has('cross-attention'), 'collegamento alla lezione animata');
});

test('ogni tappa ha una sequenza di fotogrammi con nodi visibili e frecce collegate', () => {
  const studio = path.resolve(__dirname, '../../sito/studio');
  const { stages, projectFrames } = loadStudy(studio);
  assert.equal(stages[0].slug, 'token');
  assert.equal(stages.at(-1).slug, 'risultati');
  for (const stage of stages) {
    const frames = stage.frames || projectFrames[stage.slug];
    assert.ok(frames?.length > 1, stage.slug);
    let previousDiagram;
    for (const frame of frames) {
      assert.ok(frame.title && frame.caption && frame.nodes.length, stage.slug);
      const ids = new Set(), cells = new Set();
      for (const node of frame.nodes) {
        assert.ok(node.id && node.label && !ids.has(node.id), frame.title);
        ids.add(node.id);
        assert.ok(Number.isInteger(node.x) && node.x >= 0 && node.x <= 2, node.id);
        assert.ok(Number.isInteger(node.y) && node.y >= 0 && node.y <= 3, node.id);
        const cell = [node.x, node.y].join(',');
        assert.ok(!cells.has(cell), frame.title);
        cells.add(cell);
        if (node.bar !== undefined) assert.ok(node.bar >= 0 && node.bar <= 1, node.id);
      }
      for (const [from, to] of frame.edges) assert.ok(ids.has(from) && ids.has(to), frame.title);
      const diagram = JSON.stringify([frame.nodes, frame.edges]);
      assert.notEqual(diagram, previousDiagram, frame.title);
      previousDiagram = diagram;
    }
  }
});

test('ogni passaggio del ripasso riporta esattamente le righe citate del progetto', () => {
  const root = path.resolve(__dirname, '../..');
  const studio = path.join(root, 'sito/studio');
  const { stages, lessons } = loadStudy(studio);
  assert.deepEqual(Object.keys(lessons).sort(), stages.map(s => s.slug).sort());
  for (const [slug, lesson] of Object.entries(lessons)) {
    assert.ok(lesson.orale.trim() && lesson.steps.length >= 2, slug);
    for (const step of lesson.steps) {
      assert.ok(step.label && step.input && step.output && step.explain, slug);
      assert.ok(step.path.startsWith('progetto/') && step.path.endsWith('.py'), step.path);
      const lines = readFileSync(path.join(root, step.path), 'utf8').split(/\r?\n/);
      assert.ok(step.start >= 1 && step.end >= step.start && step.end <= lines.length, step.label);
      assert.equal(step.code, lines.slice(step.start - 1, step.end).join('\n'), step.path + ':' + step.start);
    }
  }
});
