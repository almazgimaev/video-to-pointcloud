#!/usr/bin/env node
// Copies the published content from docs/ into the site, so it is never duplicated by hand.
//
//   docs/assets/*             -> public/assets/        (served as static files)
//   docs/report/charts/*.svg  -> src/assets/charts/    (inlined into pages at build time)
//   docs/report/report.md     -> src/content/report.md (links rewritten for the site)
//
// Link rewriting in the report:
//   ../assets/X        -> <base>assets/X
//   charts/X.svg       -> <base>charts/X.svg (a marker the report page replaces with the inline SVG)
//   other repo files   -> the file on GitHub
//   #anchors, http(s)  -> unchanged

import { cpSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, posix } from 'node:path';
import { fileURLToPath } from 'node:url';
import { repoFileUrl, siteBase } from './site-config.mjs';

const SITE = join(dirname(fileURLToPath(import.meta.url)), '..');
const REPO = join(SITE, '..');
const REPORT_DIR_IN_REPO = 'docs/report';

const base = siteBase();

function syncDir(from, to, filter = () => true) {
  rmSync(to, { recursive: true, force: true });
  mkdirSync(to, { recursive: true });
  const names = readdirSync(from).filter(filter);
  for (const name of names) cpSync(join(from, name), join(to, name));
  return names.length;
}

function rewriteUrl(url) {
  if (/^([a-z][a-z0-9+.-]*:|#|\/\/)/i.test(url)) return url; // absolute URL or in-page anchor
  const [path, hash = ''] = url.split(/(?=#)/);
  const fromRoot = posix.normalize(posix.join(REPORT_DIR_IN_REPO, path));
  if (fromRoot.startsWith('docs/assets/')) return `${base}assets/${fromRoot.slice('docs/assets/'.length)}${hash}`;
  if (fromRoot.startsWith('docs/report/charts/')) return `${base}charts/${fromRoot.slice('docs/report/charts/'.length)}${hash}`;
  if (fromRoot.startsWith('..')) throw new Error(`report link leaves the repository: ${url}`);
  return repoFileUrl(fromRoot) + hash;
}

function rewriteReport(markdown) {
  // Inline links and images: [text](url) / ![alt](url "title"). Code spans are left alone
  // because link syntax inside backticks does not occur in the report.
  return markdown.replace(/(!?\[[^\]]*\]\()([^)\s]+)((?:\s+"[^"]*")?\))/g, (_, open, url, close) => open + rewriteUrl(url) + close);
}

const assets = syncDir(join(REPO, 'docs/assets'), join(SITE, 'public/assets'));
const charts = syncDir(join(REPO, 'docs/report/charts'), join(SITE, 'src/assets/charts'), (n) => n.endsWith('.svg'));

const report = readFileSync(join(REPO, REPORT_DIR_IN_REPO, 'report.md'), 'utf8');
mkdirSync(join(SITE, 'src/content'), { recursive: true });
writeFileSync(
  join(SITE, 'src/content/report.md'),
  `<!-- Generated from docs/report/report.md by site/scripts/sync-content.mjs. Do not edit. -->\n\n${rewriteReport(report)}`,
);

console.log(`sync-content: ${assets} assets, ${charts} charts, report.md (base ${base})`);
