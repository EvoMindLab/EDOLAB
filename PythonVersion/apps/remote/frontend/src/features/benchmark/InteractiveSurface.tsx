import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { readThemeCanvasHex } from "../../lib/theme";
import type { BenchmarkSurfaceResult } from "./BenchmarkPage";
import { tr } from "../../lib/i18n";

export interface InteractiveSurfaceHandle {
  resetView: () => void;
}

export interface InteractiveSurfaceProps {
  result: BenchmarkSurfaceResult;
  className?: string;
  active?: boolean;
}

const INITIAL_CAMERA_POSITION = new THREE.Vector3(3.7, 3.15, 2.7);
const INITIAL_TARGET = new THREE.Vector3(0, 0, 0);
const SURFACE_COLOR_STOPS = [
  new THREE.Color(0x173f5f),
  new THREE.Color(0x20639b),
  new THREE.Color(0x3caea3),
  new THREE.Color(0xf6d55c),
  new THREE.Color(0xed553b),
] as const;

/**
 * A real WebGL surface preview. OrbitControls provides mouse/touch navigation:
 * left drag rotates, the wheel zooms, and right drag (or modifier + left drag)
 * pans the camera.
 */
export const InteractiveSurface = forwardRef<InteractiveSurfaceHandle, InteractiveSurfaceProps>(
  function InteractiveSurface({ result, className, active = true }, ref) {
    const viewportRef = useRef<HTMLDivElement | null>(null);
    const resetViewRef = useRef<() => void>(() => undefined);
    const pauseAnimationRef = useRef<() => void>(() => undefined);
    const resumeAnimationRef = useRef<() => void>(() => undefined);
    const activeRef = useRef(active);
    const [renderError, setRenderError] = useState<string | null>(null);

    activeRef.current = active;

    useImperativeHandle(ref, () => ({
      resetView: () => resetViewRef.current(),
    }), []);

    useEffect(() => {
      const viewport = viewportRef.current;
      if (!viewport) return;

      let animationFrame = 0;
      let resizeObserver: ResizeObserver | null = null;
      let renderer: THREE.WebGLRenderer | null = null;
      let controls: OrbitControls | null = null;
      let scene: THREE.Scene | null = null;
      let resize: (() => void) | null = null;
      let suppressContextMenu: ((event: MouseEvent) => void) | null = null;
      let markCameraRevision: (() => void) | null = null;
      let windowResizeAttached = false;
      let contextMenuAttached = false;
      let controlsChangeAttached = false;
      let disposed = false;
      let running = false;

      setRenderError(null);
      viewport.replaceChildren();

      const pauseAnimation = () => {
        running = false;
        window.cancelAnimationFrame(animationFrame);
      };
      const cleanup = () => {
        if (disposed) return;
        disposed = true;
        pauseAnimation();
        if (windowResizeAttached && resize) {
          const resizeListener = resize;
          safelyCleanup(() => window.removeEventListener("resize", resizeListener));
        }
        safelyCleanup(() => resizeObserver?.disconnect());
        if (controlsChangeAttached && controls && markCameraRevision) {
          const currentControls = controls;
          const changeListener = markCameraRevision;
          safelyCleanup(() => currentControls.removeEventListener("change", changeListener));
        }
        safelyCleanup(() => controls?.dispose());
        if (contextMenuAttached && renderer && suppressContextMenu) {
          const currentRenderer = renderer;
          const contextMenuListener = suppressContextMenu;
          safelyCleanup(() => currentRenderer.domElement.removeEventListener("contextmenu", contextMenuListener));
        }
        if (scene) safelyCleanup(() => disposeObjectTree(scene as THREE.Scene));
        safelyCleanup(() => renderer?.renderLists.dispose());
        safelyCleanup(() => renderer?.dispose());
        safelyCleanup(() => renderer?.forceContextLoss());
        safelyCleanup(() => viewport.replaceChildren());
        scene = null;
        controls = null;
        renderer = null;
        resizeObserver = null;
        resetViewRef.current = () => undefined;
        pauseAnimationRef.current = () => undefined;
        resumeAnimationRef.current = () => undefined;
      };

      try {
        scene = new THREE.Scene();
        scene.background = new THREE.Color(readThemeCanvasHex());

        const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
        camera.up.set(0, 1, 0);

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.domElement.dataset.testid = "benchmark-surface-canvas";
        renderer.domElement.setAttribute("aria-label", tr("Interactive 3D surface canvas of {0}", result.problem_name));
        renderer.domElement.setAttribute("role", "img");
        renderer.domElement.tabIndex = 0;
        renderer.domElement.style.display = "block";
        renderer.domElement.style.width = "100%";
        renderer.domElement.style.height = "100%";
        renderer.domElement.style.touchAction = "none";
        renderer.domElement.dataset.cameraRevision = "0";
        viewport.dataset.cameraRevision = "0";
        viewport.appendChild(renderer.domElement);

        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.075;
        controls.enablePan = true;
        controls.enableRotate = true;
        controls.enableZoom = true;
        controls.minDistance = 1.8;
        controls.maxDistance = 14;
        controls.minPolarAngle = 0.08;
        controls.maxPolarAngle = Math.PI - 0.08;
        controls.target.copy(INITIAL_TARGET);

        const resetView = () => {
          camera.position.copy(INITIAL_CAMERA_POSITION);
          camera.zoom = 1;
          camera.updateProjectionMatrix();
          controls?.target.copy(INITIAL_TARGET);
          controls?.reset();
          controls?.update();
        };
        camera.position.copy(INITIAL_CAMERA_POSITION);
        controls.saveState();
        resetViewRef.current = resetView;
        resetView();

        const surface = createSurface(result);
        addOwnedObject(scene, surface);

        const grid = new THREE.GridHelper(3.45, 10, 0x8f887c, 0xd5cfc3);
        addOwnedObject(scene, grid);
        grid.position.y = -0.86;
        const gridMaterials = Array.isArray(grid.material) ? grid.material : [grid.material];
        for (const material of gridMaterials) {
          material.transparent = true;
          material.opacity = 0.48;
        }
        addOwnedObject(scene, createLabeledAxes());

        scene.add(new THREE.HemisphereLight(0xffffff, 0x8e887e, 2.15));
        const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
        keyLight.position.set(3.5, 4.5, 4);
        scene.add(keyLight);

        resize = () => {
          if (!renderer || disposed) return;
          const rect = viewport.getBoundingClientRect();
          const width = Math.max(1, Math.round(rect.width));
          const height = Math.max(1, Math.round(rect.height));
          renderer.setSize(width, height, false);
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
        };
        resizeObserver = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(resize);
        resizeObserver?.observe(viewport);
        window.addEventListener("resize", resize);
        windowResizeAttached = true;
        resize();

        suppressContextMenu = (event: MouseEvent) => event.preventDefault();
        renderer.domElement.addEventListener("contextmenu", suppressContextMenu);
        contextMenuAttached = true;

        let cameraRevision = 0;
        markCameraRevision = () => {
          cameraRevision += 1;
          const revision = String(cameraRevision);
          viewport.dataset.cameraRevision = revision;
          if (renderer) renderer.domElement.dataset.cameraRevision = revision;
        };
        controls.addEventListener("change", markCameraRevision);
        controlsChangeAttached = true;
        markCameraRevision();

        const animate = () => {
          const currentRenderer = renderer;
          const currentControls = controls;
          const currentScene = scene;
          if (disposed || !activeRef.current || !currentRenderer || !currentControls || !currentScene) {
            running = false;
            return;
          }
          animationFrame = window.requestAnimationFrame(animate);
          try {
            currentControls.update();
            currentRenderer.render(currentScene, camera);
          } catch (error) {
            const message = error instanceof Error ? error.message : tr("The browser cannot continue rendering WebGL.");
            cleanup();
            setRenderError(message);
          }
        };
        const resumeAnimation = () => {
          if (disposed || running || !activeRef.current) return;
          resize?.();
          running = true;
          animate();
        };
        pauseAnimationRef.current = pauseAnimation;
        resumeAnimationRef.current = resumeAnimation;
        resumeAnimation();

        return cleanup;
      } catch (error) {
        const message = error instanceof Error ? error.message : tr("The browser cannot initialize WebGL.");
        cleanup();
        setRenderError(message);
      }
    }, [result]);

    useEffect(() => {
      if (active) resumeAnimationRef.current();
      else pauseAnimationRef.current();
    }, [active]);

    return (
      <figure
        className={["benchmark-svg-figure", className].filter(Boolean).join(" ")}
        data-testid="benchmark-surface-3d"
        style={{ minHeight: 420 }}
      >
        <div
          style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}
        >
          <span style={{ color: "var(--text-secondary)", fontSize: "0.75rem" }}>
            fitness {formatFitness(result.z_min)} – {formatFitness(result.z_max)}
          </span>
          <button
            type="button"
            className="button button--secondary"
            data-testid="benchmark-surface-reset"
            aria-label={tr("Reset the 3D surface view")}
            onClick={() => resetViewRef.current()}
          >{tr("Reset view")}</button>
        </div>
        <div
          ref={viewportRef}
          data-testid="benchmark-surface-viewport"
          aria-label={tr("Interactive 3D surface of {0}; drag with the left button to rotate, wheel to zoom, right button or a modifier key to pan", result.problem_name)}
          style={{
            width: "100%",
            height: "clamp(420px, 58vh, 650px)",
            minHeight: 420,
            overflow: "hidden",
            border: "1px solid var(--border)",
            borderRadius: 12,
            background: "var(--bg-canvas)",
            cursor: "grab",
          }}
        />
        {renderError ? (
          <figcaption role="alert">{tr("The 3D surface could not be initialized: {0}", renderError)}</figcaption>
        ) : (
          <figcaption>{tr("Drag with the left button to rotate · wheel zooms · right button or Shift / Ctrl / ⌘ + left button drag pans")}</figcaption>
        )}
      </figure>
    );
  },
);

InteractiveSurface.displayName = "InteractiveSurface";

function createSurface(result: BenchmarkSurfaceResult): THREE.Group {
  const group = new THREE.Group();
  const rows = result.y.length;
  const columns = result.x.length;
  if (rows === 0 || columns === 0) return group;

  const xBounds = finiteBounds(result.x, [-1, 1]);
  const yBounds = finiteBounds(result.y, [-1, 1]);
  const finiteZ = result.z.flatMap((row) => row.filter(isFiniteNumber));
  const zBounds = finiteBounds(finiteZ, [result.z_min, result.z_max]);
  const zSpan = Math.max(zBounds[1] - zBounds[0], Number.EPSILON);
  const positions: number[] = [];
  const colors: number[] = [];
  const valid: boolean[] = [];

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const rawZ = result.z[row]?.[column];
      const hasValue = isFiniteNumber(rawZ);
      const z = hasValue ? rawZ : zBounds[0];
      const normalizedZ = clamp01((z - zBounds[0]) / zSpan);
      positions.push(
        normalize(result.x[column], xBounds[0], xBounds[1], 3.2),
        normalizedZ * 1.65 - 0.825,
        normalize(result.y[row], yBounds[0], yBounds[1], 3.2),
      );
      const color = surfaceColor(normalizedZ);
      colors.push(color.r, color.g, color.b);
      valid.push(hasValue);
    }
  }

  const indices: number[] = [];
  for (let row = 0; row < rows - 1; row += 1) {
    for (let column = 0; column < columns - 1; column += 1) {
      const a = row * columns + column;
      const b = a + 1;
      const c = a + columns;
      const d = c + 1;
      if (valid[a] && valid[b] && valid[c]) indices.push(a, c, b);
      if (valid[b] && valid[c] && valid[d]) indices.push(b, c, d);
    }
  }

  let geometry: THREE.BufferGeometry | null = null;
  let material: THREE.MeshStandardMaterial | null = null;
  try {
    geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    material = new THREE.MeshStandardMaterial({
      side: THREE.DoubleSide,
      vertexColors: true,
      roughness: 0.66,
      metalness: 0.02,
    });
    group.add(new THREE.Mesh(geometry, material));
    return group;
  } catch (error) {
    safelyCleanup(() => geometry?.dispose());
    safelyCleanup(() => material?.dispose());
    throw error;
  }
}

function createLabeledAxes(): THREE.Group {
  const group = new THREE.Group();
  const origin = new THREE.Vector3(-1.72, -0.86, -1.72);
  const xLength = 3.48;
  const zLength = 3.48;
  const fitnessLength = 1.9;

  try {
    addOwnedObject(group, new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), origin, xLength, 0xe53e4f, 0.16, 0.1));
    addOwnedObject(group, new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), origin, zLength, 0x2777d3, 0.16, 0.1));
    addOwnedObject(group, new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), origin, fitnessLength, 0x219653, 0.16, 0.1));

    const x1Label = createAxisLabel("x1", "#b91c31", "#fff5f6");
    x1Label.position.set(origin.x + xLength + 0.14, origin.y, origin.z);
    addOwnedObject(group, x1Label);

    const x2Label = createAxisLabel("x2", "#175da9", "#f1f7ff");
    x2Label.position.set(origin.x, origin.y, origin.z + zLength + 0.14);
    addOwnedObject(group, x2Label);

    const fitnessLabel = createAxisLabel("f", "#14733d", "#f0fff6");
    fitnessLabel.position.set(origin.x, origin.y + fitnessLength + 0.14, origin.z);
    addOwnedObject(group, fitnessLabel);

    return group;
  } catch (error) {
    disposeObjectTree(group);
    throw error;
  }
}

function addOwnedObject<T extends THREE.Object3D>(parent: THREE.Object3D, object: T): T {
  try {
    parent.add(object);
    return object;
  } catch (error) {
    safelyCleanup(() => object.removeFromParent());
    disposeObjectTree(object);
    throw error;
  }
}

function createAxisLabel(text: string, color: string, background: string): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  if (!context) throw new Error(tr("The browser cannot create the canvas for the axis labels."));

  context.clearRect(0, 0, canvas.width, canvas.height);
  roundedRectPath(context, 12, 12, 232, 104, 24);
  context.fillStyle = background;
  context.fill();
  context.lineWidth = 8;
  context.strokeStyle = color;
  context.stroke();
  context.fillStyle = color;
  context.font = "700 70px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, 128, 66);

  let texture: THREE.CanvasTexture | null = null;
  let material: THREE.SpriteMaterial | null = null;
  try {
    texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
    material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      sizeAttenuation: false,
      toneMapped: false,
    });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(0.1, 0.05, 1);
    sprite.renderOrder = 100;
    return sprite;
  } catch (error) {
    safelyCleanup(() => texture?.dispose());
    safelyCleanup(() => material?.dispose());
    throw error;
  }
}

function roundedRectPath(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
): void {
  const right = x + width;
  const bottom = y + height;
  context.beginPath();
  context.moveTo(x + radius, y);
  context.lineTo(right - radius, y);
  context.quadraticCurveTo(right, y, right, y + radius);
  context.lineTo(right, bottom - radius);
  context.quadraticCurveTo(right, bottom, right - radius, bottom);
  context.lineTo(x + radius, bottom);
  context.quadraticCurveTo(x, bottom, x, bottom - radius);
  context.lineTo(x, y + radius);
  context.quadraticCurveTo(x, y, x + radius, y);
  context.closePath();
}

function disposeObjectTree(root: THREE.Object3D): void {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  const textures = new Set<THREE.Texture>();

  root.traverse((object) => {
    const renderable = object as THREE.Object3D & {
      geometry?: THREE.BufferGeometry;
      material?: THREE.Material | THREE.Material[];
    };
    if (renderable.geometry) geometries.add(renderable.geometry);
    const material = renderable.material;
    if (Array.isArray(material)) material.forEach((item) => materials.add(item));
    else if (material) materials.add(material);
  });

  for (const material of materials) {
    const values = Object.values(material as unknown as Record<string, unknown>);
    for (const value of values) {
      if (value instanceof THREE.Texture) textures.add(value);
    }
    if (material instanceof THREE.ShaderMaterial) {
      for (const uniform of Object.values(material.uniforms)) {
        collectTextures(uniform.value, textures);
      }
    }
  }
  geometries.forEach((geometry) => safelyCleanup(() => geometry.dispose()));
  textures.forEach((texture) => safelyCleanup(() => texture.dispose()));
  materials.forEach((material) => safelyCleanup(() => material.dispose()));
  safelyCleanup(() => root.clear());
}

function collectTextures(value: unknown, textures: Set<THREE.Texture>): void {
  if (value instanceof THREE.Texture) {
    textures.add(value);
  } else if (Array.isArray(value)) {
    value.forEach((item) => collectTextures(item, textures));
  }
}

function safelyCleanup(action: () => void): void {
  try {
    action();
  } catch {
    // Continue releasing the remaining resources even if one disposer fails.
  }
}

function finiteBounds(values: number[], fallback: [number, number]): [number, number] {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return fallback;
  const lower = Math.min(...finite);
  const upper = Math.max(...finite);
  return lower === upper ? [lower - 0.5, upper + 0.5] : [lower, upper];
}

function normalize(value: number, lower: number, upper: number, scale: number): number {
  if (!Number.isFinite(value)) return 0;
  return ((value - lower) / Math.max(upper - lower, Number.EPSILON) - 0.5) * scale;
}

function surfaceColor(value: number): THREE.Color {
  const position = clamp01(value) * (SURFACE_COLOR_STOPS.length - 1);
  const index = Math.min(SURFACE_COLOR_STOPS.length - 2, Math.floor(position));
  return SURFACE_COLOR_STOPS[index].clone().lerp(SURFACE_COLOR_STOPS[index + 1], position - index);
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function isFiniteNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function formatFitness(value: number): string {
  return Number.isFinite(value) ? value.toPrecision(4) : "—";
}
