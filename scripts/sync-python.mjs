import fs from 'node:fs';
import path from 'node:path';

const rootDir = process.cwd();
const srcPyDir = path.join(rootDir, 'dsp_engine');
const destPyDir = path.join(rootDir, 'public', 'dsp_engine');

function collectPyFiles(dir, base = '') {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  let files = [];
  for (const entry of entries) {
    if (entry.name === '__pycache__' || entry.name.startsWith('.')) continue;
    const full = path.join(dir, entry.name);
    const rel = base ? `${base}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      files = files.concat(collectPyFiles(full, rel));
    } else if (entry.name.endsWith('.py')) {
      files.push(rel);
    }
  }
  return files;
}

fs.rmSync(destPyDir, { recursive: true, force: true });
fs.mkdirSync(destPyDir, { recursive: true });

const pyFiles = collectPyFiles(srcPyDir);
for (const relPath of pyFiles) {
  const srcFile = path.join(srcPyDir, relPath);
  const destFile = path.join(destPyDir, relPath);
  fs.mkdirSync(path.dirname(destFile), { recursive: true });
  fs.copyFileSync(srcFile, destFile);
}

fs.writeFileSync(
  path.join(destPyDir, 'manifest.json'),
  JSON.stringify({ files: pyFiles, generatedAt: new Date().toISOString() }, null, 2),
  'utf-8'
);

console.log(`[sync-python] Sincronizados ${pyFiles.length} archivos Python hacia public/dsp_engine/`);
