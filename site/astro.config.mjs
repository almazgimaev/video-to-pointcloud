// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import { siteBase } from './scripts/site-config.mjs';

// SITE_BASE sets the URL prefix, e.g. SITE_BASE=/video-to-pointcloud/ for GitHub Pages.
export default defineConfig({
  output: 'static',
  base: siteBase(),
  trailingSlash: 'ignore',
  // Keep whitespace between inline text and elements (the v7 'jsx' default drops it).
  compressHTML: true,
  build: { format: 'directory', assets: '_assets' },
  vite: {
    plugins: [tailwindcss()],
    // three.js is one lazily loaded chunk of about 560 kB (140 kB gzipped); it is expected.
    build: { chunkSizeWarningLimit: 700 },
  },
});
