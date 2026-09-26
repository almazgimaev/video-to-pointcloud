// Metadata of the published assets (synced from docs/assets/assets.json).
import manifest from '../../public/assets/assets.json';

export interface AssetInfo {
  file: string;
  what: string;
  source_run: string;
  width?: number;
  height?: number;
}

const byFile = new Map<string, AssetInfo>((manifest.assets as AssetInfo[]).map((a) => [a.file, a]));

export function asset(file: string): AssetInfo {
  const info = byFile.get(file);
  if (!info) throw new Error(`unknown asset: ${file}`);
  return info;
}

/** URL of a published asset, respecting the configured base. */
export function assetUrl(file: string): string {
  return `${import.meta.env.BASE_URL}assets/${file}`;
}

/** Sentence-case version of the "what" text, for alt text. */
export function altOf(file: string): string {
  const what = asset(file).what;
  return what.charAt(0).toUpperCase() + what.slice(1);
}
