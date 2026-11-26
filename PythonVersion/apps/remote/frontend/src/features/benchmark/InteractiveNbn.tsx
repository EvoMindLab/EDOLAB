import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { readThemeCanvasHex } from "../../lib/theme";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { BenchmarkNbnResult } from "./BenchmarkPage";
import { tr, LOCALE } from "../../lib/i18n";

const MAX_RENDERED_NODES = 4_000;
const MAX_RENDERED_EDGES = 6_000;

export interface InteractiveNbnProps {
  result: BenchmarkNbnResult;
  active?: boolean;
}

interface ViewportHandle {
  reset: () => void;
  setActive: (active: boolean) => void;
  dispose: () => void;
}

/**
 * A real WebGL NBN viewport. The component deliberately keeps rendering limits
 * separate from the result model, so sampling never changes the reported
 * research data or its original node/edge counts.
 */
export function InteractiveNbn({ result, active = true }: InteractiveNbnProps) {
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const handleRef = useRef<ViewportHandle | null>(null);
  const activeRef = useRef(active);
  const [renderError, setRenderError] = useState<string | null>(null);
  activeRef.current = active;

  useEffect(() => {
    const container = viewportRef.current;
    if (!container) return;
    setRenderError(null);
    try {
      handleRef.current = createViewport(container, result);
      handleRef.current.setActive(activeRef.current);
    } catch (error) {
      handleRef.current = null;
      setRenderError(viewportErrorMessage(error));
    }
    return () => {
      handleRef.current?.dispose();
      handleRef.current = null;
    };
  }, [result]);

  useEffect(() => {
    handleRef.current?.setActive(active);
  }, [active]);

  const sampled = result.nodes.length > MAX_RENDERED_NODES || result.edges.length > MAX_RENDERED_EDGES;

  return (
    <figure
      data-testid="interactive-nbn"
      style={{
        display: "grid",
        gridTemplateRows: "auto minmax(440px, 1fr)",
        gap: 10,
        width: "100%",
        minHeight: 520,
        height: "min(68vh, 680px)",
        margin: 0,
      }}
    >
      <figcaption
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: 8,
          color: "var(--text-muted)",
          fontSize: ".66rem",
        }}
      >
        <span>{tr("{0} nodes · {1} edges{2}", result.nodes.length.toLocaleString(LOCALE), result.edges.length.toLocaleString(LOCALE), sampled
            ? tr(" · to stay responsive, at most {0} nodes / {1} edges are drawn", MAX_RENDERED_NODES.toLocaleString(LOCALE), MAX_RENDERED_EDGES.toLocaleString(LOCALE))
            : "")}</span>
        <span>{tr("Left button rotates · wheel zooms · right button or Shift / Ctrl / ⌘ + left button pans")}</span>
      </figcaption>

      <div
        ref={viewportRef}
        data-testid="interactive-nbn-viewport"
        role="application"
        tabIndex={0}
        aria-label={tr("Interactive 3D NBN view of {0}; left button rotates, wheel zooms, right button or a modifier key with the left button pans", result.problem_name)}
        style={{
          position: "relative",
          width: "100%",
          height: "100%",
          minHeight: 440,
          overflow: "hidden",
          border: "1px solid var(--border)",
          borderRadius: 12,
          background: "var(--bg-canvas)",
          touchAction: "none",
          cursor: "grab",
        }}
      >
        {renderError ? (
          <div
            role="alert"
            data-testid="interactive-nbn-error"
            style={{
              position: "absolute",
              inset: 0,
              display: "grid",
              placeItems: "center",
              padding: 28,
              color: "var(--danger)",
              fontSize: ".72rem",
              lineHeight: 1.65,
              textAlign: "center",
            }}
          >
            {renderError}
          </div>
        ) : (
          <button
            type="button"
            data-testid="interactive-nbn-reset"
            aria-label={tr("Reset the 3D NBN view")}
            onClick={() => handleRef.current?.reset()}
            style={{
              position: "absolute",
              zIndex: 2,
              top: 10,
              right: 10,
              minHeight: 32,
              padding: "5px 10px",
              border: "1px solid var(--border)",
              borderRadius: 6,
              background: "rgba(255, 255, 255, .92)",
              color: "var(--text-secondary)",
              font: "600 0.75rem/1 var(--sans)",
              cursor: "pointer",
              backdropFilter: "blur(4px)",
            }}
          >{tr("Reset view")}</button>
        )}
      </div>
    </figure>
  );
}

function createViewport(container: HTMLDivElement, result: BenchmarkNbnResult): ViewportHandle {
  let scene: THREE.Scene | null = null;
  let camera: THREE.PerspectiveCamera | null = null;
  let renderer: THREE.WebGLRenderer | null = null;
  let controls: OrbitControls | null = null;
  let resizeObserver: ResizeObserver | null = null;
  let intersectionObserver: IntersectionObserver | null = null;
  let recordCameraChange: (() => void) | null = null;
  let setDragCursor: (() => void) | null = null;
  let resetDragCursor: (() => void) | null = null;
  let active = true;
  let visible = true;
  let disposed = false;
  let animationFrame = 0;

  const stopAnimation = () => {
    if (!animationFrame) return;
    window.cancelAnimationFrame(animationFrame);
    animationFrame = 0;
  };
  const resize = () => {
    if (!renderer || !camera) return;
    const bounds = container.getBoundingClientRect();
    if (bounds.width < 2 || bounds.height < 2) return;
    renderer.setSize(Math.round(bounds.width), Math.round(bounds.height), false);
    camera.aspect = bounds.width / bounds.height;
    camera.updateProjectionMatrix();
  };
  const animate = () => {
    animationFrame = 0;
    if (disposed || !active || !visible || !controls || !renderer || !scene || !camera) return;
    controls.update();
    renderer.render(scene, camera);
    animationFrame = window.requestAnimationFrame(animate);
  };
  const startAnimation = () => {
    if (disposed || !active || !visible || animationFrame || !renderer || !scene || !camera) return;
    resize();
    renderer.render(scene, camera);
    animationFrame = window.requestAnimationFrame(animate);
  };
  const cleanup = () => {
    if (disposed) return;
    disposed = true;
    stopAnimation();
    safely(() => resizeObserver?.disconnect());
    safely(() => intersectionObserver?.disconnect());
    if (renderer && setDragCursor) {
      safely(() => renderer?.domElement.removeEventListener("pointerdown", setDragCursor as () => void));
    }
    if (resetDragCursor) {
      safely(() => window.removeEventListener("pointerup", resetDragCursor as () => void));
      safely(() => window.removeEventListener("pointercancel", resetDragCursor as () => void));
    }
    if (controls && recordCameraChange) {
      safely(() => controls?.removeEventListener("change", recordCameraChange as () => void));
    }
    safely(() => controls?.dispose());
    if (scene) safely(() => disposeScene(scene as THREE.Scene));
    safely(() => renderer?.renderLists.dispose());
    safely(() => renderer?.dispose());
    safely(() => renderer?.forceContextLoss());
    safely(() => renderer?.domElement.remove());
    delete container.dataset.cameraRevision;
    container.style.cursor = "grab";
  };

  try {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(readThemeCanvasHex());

    camera = new THREE.PerspectiveCamera(42, 1, 0.05, 100);
    camera.position.set(3.6, 3.0, 5.2);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.dataset.testid = "interactive-nbn-canvas";
    renderer.domElement.setAttribute("aria-hidden", "true");
    renderer.domElement.style.display = "block";
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    container.prepend(renderer.domElement);

    const graph = buildRenderableGraph(result);
    const group = new THREE.Group();
    group.name = "nbn-graph";
    scene.add(group);
    populateGraph(group, graph);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.075;
    controls.enablePan = true;
    controls.screenSpacePanning = true;
    controls.minDistance = 1.45;
    controls.maxDistance = 18;
    controls.minPolarAngle = 0.05;
    controls.maxPolarAngle = Math.PI - 0.05;
    let cameraRevision = 0;
    container.dataset.cameraRevision = "0";
    renderer.domElement.dataset.cameraRevision = "0";
    recordCameraChange = () => {
      cameraRevision += 1;
      const revision = String(cameraRevision);
      container.dataset.cameraRevision = revision;
      if (renderer) renderer.domElement.dataset.cameraRevision = revision;
    };
    controls.addEventListener("change", recordCameraChange);
    controls.update();
    controls.saveState();

    resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);
    resize();

    intersectionObserver = new IntersectionObserver(
      (entries) => {
        visible = entries[0]?.isIntersecting ?? true;
        if (visible) startAnimation();
        else stopAnimation();
      },
      { rootMargin: "100px" },
    );
    intersectionObserver.observe(container);

    setDragCursor = () => { container.style.cursor = "grabbing"; };
    resetDragCursor = () => { container.style.cursor = "grab"; };
    renderer.domElement.addEventListener("pointerdown", setDragCursor);
    window.addEventListener("pointerup", resetDragCursor);
    window.addEventListener("pointercancel", resetDragCursor);

    startAnimation();

    return {
      reset: () => {
        if (!controls || !renderer || !scene || !camera) return;
        controls.reset();
        controls.update();
        renderer.render(scene, camera);
      },
      setActive: (nextActive) => {
        active = nextActive;
        if (controls) controls.enabled = nextActive;
        if (active) startAnimation();
        else stopAnimation();
      },
      dispose: cleanup,
    };
  } catch (error) {
    cleanup();
    throw error;
  }
}

type NbnNode = BenchmarkNbnResult["nodes"][number];

interface RenderableGraph {
  nodes: NbnNode[];
  edges: BenchmarkNbnResult["edges"];
  layoutRadius: number;
}

function buildRenderableGraph(result: BenchmarkNbnResult): RenderableGraph {
  const nodes = selectNodes(result.nodes, MAX_RENDERED_NODES);
  const nodeIds = new Set(nodes.map((node) => node.id));
  const eligibleEdges = result.edges.filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target));
  return {
    nodes,
    edges: evenlySample(eligibleEdges, MAX_RENDERED_EDGES),
    layoutRadius: radialLayoutRadius(result.nodes),
  };
}

function selectNodes(nodes: NbnNode[], limit: number): NbnNode[] {
  if (nodes.length <= limit) return nodes;
  const roots = nodes.filter((node) => node.is_root);
  if (roots.length >= limit) return evenlySample(roots, limit);
  const rootIds = new Set(roots.map((node) => node.id));
  const nonRoots = nodes.filter((node) => !rootIds.has(node.id));
  return [...roots, ...evenlySample(nonRoots, limit - roots.length)];
}

function evenlySample<T>(values: T[], limit: number): T[] {
  if (limit <= 0) return [];
  if (values.length <= limit) return values;
  const step = values.length / limit;
  return Array.from({ length: limit }, (_, index) => values[Math.floor(index * step)]);
}

function populateGraph(group: THREE.Group, graph: RenderableGraph) {
  const positions = new Map<number, THREE.Vector3>();
  const radialScale = 1.65 / Math.max(graph.layoutRadius, Number.EPSILON);
  let edgeGeometry: THREE.BufferGeometry | null = null;
  let edgeMaterial: THREE.LineBasicMaterial | null = null;

  try {
    for (const node of graph.nodes) {
      const x = Number.isFinite(node.x) ? node.x : 0;
      const y = Number.isFinite(node.y) ? node.y : 0;
      const height = Number.isFinite(node.z) ? node.z : node.normalized_fitness;
      positions.set(
        node.id,
        new THREE.Vector3(
          x * radialScale,
          (clamp01(height) - 0.5) * 1.55,
          y * radialScale,
        ),
      );
    }

    const edgePositions: number[] = [];
    for (const edge of graph.edges) {
      const source = positions.get(edge.source);
      const target = positions.get(edge.target);
      if (!source || !target) continue;
      edgePositions.push(source.x, source.y, source.z, target.x, target.y, target.z);
    }
    edgeGeometry = new THREE.BufferGeometry();
    edgeGeometry.setAttribute("position", new THREE.Float32BufferAttribute(edgePositions, 3));
    edgeMaterial = new THREE.LineBasicMaterial({
      color: 0x617985,
      transparent: true,
      opacity: 0.42,
      depthWrite: false,
    });
    group.add(new THREE.LineSegments(edgeGeometry, edgeMaterial));

    addPointCloud(group, graph.nodes.filter((node) => !node.is_root), positions, 0.06);
    addPointCloud(group, graph.nodes.filter((node) => node.is_root), positions, 0.13);
  } catch (error) {
    safely(() => edgeGeometry?.dispose());
    safely(() => edgeMaterial?.dispose());
    disposeObjectTree(group);
    group.clear();
    throw error;
  }
}

function addPointCloud(
  group: THREE.Group,
  nodes: NbnNode[],
  positions: Map<number, THREE.Vector3>,
  size: number,
) {
  if (!nodes.length) return;
  let geometry: THREE.BufferGeometry | null = null;
  let material: THREE.PointsMaterial | null = null;
  const pointPositions: number[] = [];
  const pointColors: number[] = [];
  for (const node of nodes) {
    const position = positions.get(node.id);
    if (!position) continue;
    pointPositions.push(position.x, position.y, position.z);
    const color = fitnessColor(node.normalized_fitness);
    pointColors.push(color.r, color.g, color.b);
  }
  try {
    geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(pointPositions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(pointColors, 3));
    material = new THREE.PointsMaterial({
      size,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.98,
    });
    const points = new THREE.Points(geometry, material);
    points.frustumCulled = false;
    group.add(points);
  } catch (error) {
    safely(() => geometry?.dispose());
    safely(() => material?.dispose());
    throw error;
  }
}

function fitnessColor(value: number): THREE.Color {
  const normalized = clamp01(value);
  // Keep the original Web UI's blue -> cyan/green -> yellow -> red hue
  // progression, while making high-fitness red nodes visually deeper.
  return new THREE.Color().setHSL(
    0.62 - 0.62 * normalized,
    0.82,
    0.58 - 0.14 * normalized,
  );
}

function radialLayoutRadius(nodes: NbnNode[]): number {
  let radius = 0;
  for (const node of nodes) {
    const x = Number.isFinite(node.x) ? node.x : 0;
    const y = Number.isFinite(node.y) ? node.y : 0;
    radius = Math.max(radius, Math.hypot(x, y));
  }
  return radius || 1;
}

function clamp01(value: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
}

function disposeScene(scene: THREE.Scene) {
  disposeObjectTree(scene);
  scene.clear();
}

function disposeObjectTree(root: THREE.Object3D) {
  root.traverse((object) => {
    const renderable = object as THREE.Object3D & {
      geometry?: THREE.BufferGeometry;
      material?: THREE.Material | THREE.Material[];
    };
    safely(() => renderable.geometry?.dispose());
    const material = renderable.material;
    if (Array.isArray(material)) {
      material.forEach((item) => safely(() => item.dispose()));
    } else {
      safely(() => material?.dispose());
    }
  });
}

function safely(action: () => void) {
  try {
    action();
  } catch {
    // Cleanup must remain idempotent and must not mask the initialization error.
  }
}

function viewportErrorMessage(error: unknown): string {
  const detail = error instanceof Error && error.message ? tr(": {0}", error.message) : "";
  return tr("The 3D NBN view could not be initialized{0}. Check that WebGL is enabled in the browser, then generate the interactive NBN again.", detail);
}

export default InteractiveNbn;
