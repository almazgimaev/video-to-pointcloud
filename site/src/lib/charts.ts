// Charts synced from docs/report/charts/, as raw SVG markup for inlining.
// Inlining lets the page theme override the chart classes (see global.css). An inline <style>
// applies to the whole document, so its selectors are scoped to svg.viz here: otherwise chart
// classes such as .grid would also style page elements that use the Tailwind class "grid".
const raw = import.meta.glob('../assets/charts/*.svg', { query: '?raw', import: 'default', eager: true }) as Record<string, string>;

function scopeSelectors(css: string): string {
  return css.replace(/(^|[{}])(\s*)([^{}@]+?)(\s*\{)/g, (_, before, space, selectors: string, open) => {
    const scoped = selectors
      .split(',')
      .map((s) => s.trim())
      .map((s) => (s.startsWith('svg.viz') ? s : `svg.viz ${s}`))
      .join(', ');
    return `${before}${space}${scoped}${open}`;
  });
}

function scopeStyles(svg: string): string {
  return svg.replace(/<style>([\s\S]*?)<\/style>/g, (_, css: string) => `<style>${scopeSelectors(css)}</style>`);
}

const charts = new Map(
  Object.entries(raw).map(([path, svg]) => [path.split('/').pop()!.replace(/\.svg$/, ''), scopeStyles(svg.replace(/<\?xml[^>]*>\s*/, ''))]),
);

export function chartSvg(name: string): string {
  const svg = charts.get(name);
  if (!svg) throw new Error(`unknown chart: ${name}`);
  return svg;
}
