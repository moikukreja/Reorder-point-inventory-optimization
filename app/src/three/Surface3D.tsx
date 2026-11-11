/**
 * Optional 3D (WebGL) view of the optimiser's GP cost surface over (s, Q). It is lazy-loaded,
 * so three.js only downloads when the user opens it. It is useful here because the 3D shape
 * shows the flat "valley" of near-optimal policies that a 2D heatmap flattens.
 * Height and colour = predicted cost; dimmed = predicted infeasible; spheres = simulations run;
 * gold = optimum.
 */
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { useMemo } from 'react';
import * as THREE from 'three';
import type { OptimiseIteration, SurfaceGrid } from '../engine/types';
import { SEQ_BLUE } from '../charts/tokens';
import { fmtInt, fmtMoney } from '../charts/EChart';

const W = 4, D = 4, Hmax = 1.6;

function useGeometry(sf: SurfaceGrid) {
  return useMemo(() => {
    const nx = sf.s.length, nz = sf.q.length;
    const feas: number[] = [];
    sf.mean.forEach((r, j) => r.forEach((v, i) => sf.pFeasible[j][i] >= 0.5 && feas.push(v)));
    const lo = Math.min(...(feas.length ? feas : sf.mean.flat()));
    const hi = Math.min(Math.max(...sf.mean.flat()), lo * 3);
    const norm = (v: number) => Math.min(1, Math.max(0, (v - lo) / (hi - lo || 1)));
    const geo = new THREE.PlaneGeometry(W, D, nx - 1, nz - 1);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.attributes.position as THREE.BufferAttribute;
    const colors = new Float32Array(pos.count * 3);
    const ramp = SEQ_BLUE.map((h) => new THREE.Color(h));
    for (let k = 0; k < pos.count; k++) {
      const i = k % nx, j = Math.floor(k / nx);
      const jj = nz - 1 - j; // plane rows run from +z to −z
      const v = norm(sf.mean[jj][i]);
      pos.setY(k, v * Hmax);
      const c = ramp[Math.min(ramp.length - 1, Math.floor(v * (ramp.length - 1)))].clone();
      if (sf.pFeasible[jj][i] < 0.5) c.multiplyScalar(0.28);
      colors.set([c.r, c.g, c.b], k * 3);
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    const place = (s: number, q: number, cost: number) => new THREE.Vector3(((s - sf.s[0]) / (sf.s.at(-1)! - sf.s[0]) - 0.5) * W, norm(cost) * Hmax + 0.04, (0.5 - (q - sf.q[0]) / (sf.q.at(-1)! - sf.q[0])) * D);
    return { geo, place };
  }, [sf]);
}

export default function Surface3D({ surface, history, best }: { surface: SurfaceGrid; history: OptimiseIteration[]; best: OptimiseIteration | null }) {
  const { geo, place } = useGeometry(surface);
  return (
    <div className="relative h-[380px] overflow-hidden rounded-xl border border-white/[0.06] bg-[#0b111c]" role="img" aria-label="3D surface of predicted cost over reorder point and order quantity">
      <Canvas camera={{ position: [4.2, 3.4, 4.6], fov: 42 }} dpr={[1, 2]}>
        <ambientLight intensity={0.55} />
        <directionalLight position={[3, 6, 2]} intensity={1.1} />
        <mesh geometry={geo}>
          <meshStandardMaterial vertexColors side={THREE.DoubleSide} roughness={0.75} metalness={0.05} />
        </mesh>
        <gridHelper args={[W, 8, '#2a3546', '#1a2333']} position={[0, -0.01, 0]} />
        {history.map((h) => (
          <mesh key={h.iter} position={place(h.s, h.Q, h.cost)}>
            <sphereGeometry args={[0.045, 12, 12]} />
            <meshStandardMaterial color={h.feasible ? '#e8edf5' : '#5b6577'} />
          </mesh>
        ))}
        {best && (
          <mesh position={place(best.s, best.Q, best.cost).add(new THREE.Vector3(0, 0.08, 0))}>
            <octahedronGeometry args={[0.11]} />
            <meshStandardMaterial color="#fab219" emissive="#fab219" emissiveIntensity={0.4} />
          </mesh>
        )}
        <OrbitControls enablePan={false} minDistance={3} maxDistance={11} />
      </Canvas>
      <div className="pointer-events-none absolute left-3 top-3 space-y-0.5 text-[11.5px] text-ink-3">
        <div>x → reorder point s · z → order quantity Q · height = predicted cost</div>
        {best && <div className="text-ink-2">◆ optimum s* {fmtInt(best.s)}, Q* {fmtInt(best.Q)}, {fmtMoney(best.cost)}/day</div>}
        <div>Drag to rotate · scroll to zoom</div>
      </div>
    </div>
  );
}
