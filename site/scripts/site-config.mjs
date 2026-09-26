// Shared settings for the Astro config and the content sync script.

/** The public repository; links to repository files point here. */
export const REPO_URL = 'https://github.com/almazgimaev/video-to-pointcloud';
export const REPO_BRANCH = 'main';

/** Link to a repository file on GitHub, given its path from the repository root. */
export function repoFileUrl(repoPath) {
  return `${REPO_URL}/blob/${REPO_BRANCH}/${repoPath.replace(/^\/+/, '')}`;
}

/**
 * The URL prefix the site is served under, from SITE_BASE (default "/").
 * Always starts and ends with a slash, e.g. "/" or "/video-to-pointcloud/".
 */
export function siteBase(env = process.env) {
  const raw = (env.SITE_BASE ?? '/').trim();
  const trimmed = raw.replace(/^\/+|\/+$/g, '');
  return trimmed ? `/${trimmed}/` : '/';
}
