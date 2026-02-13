import fs from 'node:fs'
import path from 'node:path'
import process from 'node:process'

const rootDir = path.resolve(process.cwd())
const srcDir = path.join(rootDir, 'src')
const entryFile = path.join(srcDir, 'main.tsx')

const importPattern = /\b(?:import|export)\b[\s\S]*?\bfrom\s*['"]([^'"]+)['"]/g
const dynamicImportPattern = /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g

const resolveCandidates = (basePath) => [
  basePath,
  `${basePath}.ts`,
  `${basePath}.tsx`,
  `${basePath}.js`,
  `${basePath}.jsx`,
  `${basePath}.css`,
  path.join(basePath, 'index.ts'),
  path.join(basePath, 'index.tsx'),
  path.join(basePath, 'index.js'),
  path.join(basePath, 'index.jsx'),
]

function resolveImport(fromFile, specifier) {
  if (!specifier.startsWith('.')) return null
  const basePath = path.resolve(path.dirname(fromFile), specifier)
  for (const candidate of resolveCandidates(basePath)) {
    if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
      return candidate
    }
  }
  return null
}

function readImportSpecifiers(filePath) {
  const text = fs.readFileSync(filePath, 'utf8')
  const imports = []

  for (const pattern of [importPattern, dynamicImportPattern]) {
    pattern.lastIndex = 0
    let match = pattern.exec(text)
    while (match) {
      imports.push(match[1])
      match = pattern.exec(text)
    }
  }

  return imports
}

if (!fs.existsSync(entryFile)) {
  console.error(`Mounted API contract check failed: missing entry file ${entryFile}`)
  process.exit(1)
}

const queue = [entryFile]
const visited = new Set()
const violations = []

while (queue.length > 0) {
  const current = queue.pop()
  if (!current || visited.has(current)) continue
  visited.add(current)

  const importSpecs = readImportSpecifiers(current)
  for (const specifier of importSpecs) {
    if (specifier.includes('legacyApi')) {
      violations.push({
        file: path.relative(rootDir, current),
        specifier,
      })
    }

    const resolved = resolveImport(current, specifier)
    if (!resolved) continue
    if (path.basename(resolved).startsWith('legacyApi.')) {
      violations.push({
        file: path.relative(rootDir, current),
        specifier,
      })
    }
    queue.push(resolved)
  }
}

if (violations.length > 0) {
  console.error('Mounted API contract check failed:')
  for (const violation of violations) {
    console.error(`- ${violation.file} imports "${violation.specifier}"`)
  }
  process.exit(1)
}

console.log('Mounted API contract check passed.')
