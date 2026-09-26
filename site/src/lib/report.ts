// Renders the synced report (src/content/report.md) to HTML at build time.
//
// On top of plain Markdown:
// - headings get GitHub-style ids, so the report's own "Contents" links keep working;
// - an image alone in a paragraph becomes a <figure>; an italic paragraph right after it
//   becomes its caption, otherwise the alt text is the caption;
// - chart images are replaced with the inline SVG;
// - empty alt text is filled from assets.json, and images get width/height and lazy loading;
// - tables are wrapped so they scroll horizontally on narrow screens instead of the page.
import { Marked, Renderer, type Token, type Tokens } from 'marked';
import GithubSlugger from 'github-slugger';
import source from '../content/report.md?raw';
import { chartSvg } from './charts';
import { asset } from './assets';

export interface Heading {
  depth: number;
  text: string;
  slug: string;
}

const BASE = import.meta.env.BASE_URL;

function escapeAttr(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
}

function assetFile(href: string): string | null {
  return href.startsWith(`${BASE}assets/`) ? href.slice(`${BASE}assets/`.length) : null;
}

function chartName(href: string): string | null {
  const m = href.startsWith(`${BASE}charts/`) ? href.slice(`${BASE}charts/`.length).match(/^([\w-]+)\.svg$/) : null;
  return m ? m[1] : null;
}

function imgTag(href: string, alt: string): string {
  const file = assetFile(href);
  const info = file ? asset(file) : null;
  const text = alt || (info ? info.what.charAt(0).toUpperCase() + info.what.slice(1) : '');
  const size = info?.width && info?.height ? ` width="${info.width}" height="${info.height}"` : '';
  return `<img src="${escapeAttr(href)}" alt="${escapeAttr(text)}"${size} loading="lazy" decoding="async">`;
}

function plainText(tokens: Token[] | undefined): string {
  return (tokens ?? []).map((t) => ('tokens' in t && t.tokens ? plainText(t.tokens) : 'text' in t ? String(t.text) : '')).join('');
}

export function renderReport(): { html: string; headings: Heading[]; title: string } {
  const slugger = new GithubSlugger();
  const headings: Heading[] = [];
  let title = '';

  const renderer = new Renderer();
  renderer.heading = function ({ tokens, depth }) {
    const text = plainText(tokens);
    const slug = slugger.slug(text);
    if (depth === 1 && !title) title = text;
    else headings.push({ depth, text, slug });
    const inner = this.parser.parseInline(tokens);
    if (depth === 1) return `<h1 id="${slug}">${inner}</h1>\n`;
    return `<h${depth} id="${slug}"><a class="anchor" href="#${slug}" aria-hidden="true" tabindex="-1">#</a>${inner}</h${depth}>\n`;
  };
  renderer.image = function ({ href, text }) {
    const chart = chartName(href);
    if (chart) return `<span class="chart" role="group" aria-label="${escapeAttr(text)}">${chartSvg(chart)}</span>`;
    return imgTag(href, text);
  };
  renderer.table = function (token) {
    return `<div class="table-wrap">${Renderer.prototype.table.call(this, token)}</div>\n`;
  };
  renderer.link = function (token) {
    const html = Renderer.prototype.link.call(this, token);
    return /^https?:\/\//.test(token.href) ? html.replace('<a ', '<a rel="noopener" ') : html;
  };

  const marked = new Marked({ gfm: true, renderer });
  const tokens = marked.lexer(source.replace(/^<!--[\s\S]*?-->\s*/, ''));

  const soleImage = (t: Token | undefined): Tokens.Image | null => {
    if (t?.type !== 'paragraph') return null;
    const inner = (t as Tokens.Paragraph).tokens.filter((x) => !(x.type === 'text' && !x.raw.trim()));
    return inner.length === 1 && inner[0].type === 'image' ? (inner[0] as Tokens.Image) : null;
  };
  const soleEmphasis = (t: Token | undefined): Tokens.Em | null => {
    if (t?.type !== 'paragraph') return null;
    const inner = (t as Tokens.Paragraph).tokens;
    return inner.length === 1 && inner[0].type === 'em' ? (inner[0] as Tokens.Em) : null;
  };

  const parts: string[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const image = soleImage(tokens[i]);
    if (!image) {
      parts.push(marked.parser([tokens[i]]));
      continue;
    }
    let j = i + 1;
    while (tokens[j]?.type === 'space') j++;
    const em = soleEmphasis(tokens[j]);
    const isChart = chartName(image.href) !== null;
    const caption = em ? marked.parseInline(em.tokens.map((t) => t.raw).join('')) : isChart ? '' : image.text;
    const media = marked.parseInline(image.raw) as string;
    parts.push(`<figure class="${isChart ? 'figure-chart' : 'figure-image'}">${media}${caption ? `<figcaption>${caption}</figcaption>` : ''}</figure>\n`);
    if (em) i = j;
  }

  return { html: parts.join(''), headings, title };
}
