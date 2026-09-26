// The interactive point cloud view. Loaded with a dynamic import() only when the viewer opens,
// so three.js and the PLY are never downloaded with the landing page.
import {
  BufferGeometry,
  PerspectiveCamera,
  Points,
  PointsMaterial,
  Scene,
  SRGBColorSpace,
  Vector3,
  WebGLRenderer,
} from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export interface ViewerHandle {
  dispose(): void;
}

export interface ViewerOptions {
  container: HTMLElement;
  url: string;
  onProgress?: (fraction: number | null) => void;
  reducedMotion?: boolean;
}

let cachedGeometry: Promise<BufferGeometry> | null = null;

function loadGeometry(url: string, onProgress?: (f: number | null) => void): Promise<BufferGeometry> {
  cachedGeometry ??= new PLYLoader()
    .loadAsync(url, (e) => onProgress?.(e.lengthComputable && e.total ? e.loaded / e.total : null))
    .then((geometry) => {
      // PLYLoader already converts the sRGB colour bytes to the linear working colour space.
      geometry.computeBoundingSphere();
      return geometry;
    })
    .catch((err) => {
      cachedGeometry = null; // allow a retry
      throw err;
    });
  return cachedGeometry;
}

export async function mountViewer({ container, url, onProgress, reducedMotion = false }: ViewerOptions): Promise<ViewerHandle> {
  const geometry = await loadGeometry(url, onProgress);
  const sphere = geometry.boundingSphere!;
  const center = sphere.center.clone();
  const radius = sphere.radius || 1;

  const renderer = new WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = SRGBColorSpace;
  renderer.setClearColor(0x0b0b0a, 1);
  renderer.domElement.style.display = 'block';
  renderer.domElement.style.width = '100%';
  renderer.domElement.style.height = '100%';
  renderer.domElement.style.touchAction = 'none';
  renderer.domElement.setAttribute('aria-label', 'Interactive point cloud: drag to orbit, scroll or pinch to zoom');
  renderer.domElement.setAttribute('role', 'img');
  container.appendChild(renderer.domElement);

  const scene = new Scene();
  const material = new PointsMaterial({ size: radius * 0.009, sizeAttenuation: true, vertexColors: true });
  scene.add(new Points(geometry, material));

  // The cloud is aligned to +Y up; frame its bounding sphere from slightly above the front.
  const camera = new PerspectiveCamera(40, 1, radius / 100, radius * 100);
  camera.up.set(0, 1, 0);
  const fitDistance = radius / Math.sin((camera.fov * Math.PI) / 360);
  const direction = new Vector3(0, 0.38, 1).normalize();
  camera.position.copy(center).addScaledVector(direction, fitDistance * 0.95);
  camera.lookAt(center);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(center);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = radius * 0.3;
  controls.maxDistance = radius * 6;
  controls.autoRotate = !reducedMotion;
  controls.autoRotateSpeed = 0.8;
  const stopAutoRotate = () => {
    controls.autoRotate = false;
  };
  controls.addEventListener('start', stopAutoRotate);
  controls.update();

  const resize = () => {
    const w = container.clientWidth || 1;
    const h = container.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  const observer = new ResizeObserver(resize);
  observer.observe(container);
  resize();

  let frame = 0;
  const tick = () => {
    frame = requestAnimationFrame(tick);
    controls.update();
    renderer.render(scene, camera);
  };
  tick();

  return {
    dispose() {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.removeEventListener('start', stopAutoRotate);
      controls.dispose();
      material.dispose();
      // The geometry is kept in the module cache so reopening does not download the PLY again.
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
    },
  };
}
