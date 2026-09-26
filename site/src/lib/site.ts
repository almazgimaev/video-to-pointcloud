// Site-wide constants.
import { REPO_URL } from '../../scripts/site-config.mjs';

export { REPO_URL };
export const VGGT_URL = 'https://github.com/facebookresearch/vggt';
export const VGGT_WEIGHTS_URL = 'https://huggingface.co/facebook/VGGT-1B';
export const CC_BY_NC_URL = 'https://creativecommons.org/licenses/by-nc/4.0/';

export const SITE_TITLE = 'Video-to-3D';
export const SITE_PITCH =
  'A phone video of a static object in; a coloured 3D point cloud with camera poses out, stood upright, with honest diagnostics.';

/** Internal URL respecting the configured base. */
export function href(path = ''): string {
  return `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`;
}
