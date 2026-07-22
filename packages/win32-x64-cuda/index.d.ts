// Type definitions for soulx-singer-dit-win32-x64-cuda 0.1.0
//
// This subpackage is a thin loader around the prebuilt `cuda.node` binary
// and re-exports the native surface. The authoritative, full type
// definitions live in the main `soulx-singer-dit` package
// (../../index.d.ts); the declarations below are a self-contained minimal
// mirror of the native export surface so that direct imports of this
// subpackage still type-check.
//
// Native export surface (from src/binding.cc):
//   - class Model { constructor(path, options?); forward(...); reverseDiffusion(...); release(); static loadModel(...); }
//   - function getVersion(): string

/// <reference types="node" />

/**
 * Options accepted by `new Model(...)`. The `backend` field is accepted for
 * parity with the main package but is informational only at this layer — this
 * subpackage always loads the CUDA binary.
 */
export interface LoadOptions {
  backend?: 'cpu' | 'vulkan' | 'cuda';
}

/** Arguments to `model.forward(...)`. See the main package types for details. */
export interface ForwardOptions {
  x: Float32Array;
  cond: Float32Array;
  t: number;
  T: number;
}

/** Arguments to `model.reverseDiffusion(...)`. See the main package types for details. */
export interface ReverseDiffusionOptions {
  promptMel: Float32Array;
  cond: Float32Array;
  z: Float32Array;
  promptLen: number;
  targetLen: number;
  nSteps: number;
  seed?: number;
}

/**
 * A loaded SoulX-Singer-DiT model backed by the CUDA binary.
 * See the main `soulx-singer-dit` package for full documentation.
 */
export class Model {
  constructor(path: string, options?: LoadOptions);
  forward(opts: ForwardOptions): Float32Array;
  reverseDiffusion(opts: ReverseDiffusionOptions): Float32Array;
  release(): void;
  static loadModel(path: string, options?: LoadOptions): Model;
}

/** Returns the binding version string (e.g. `"0.1.0"`). */
export function getVersion(): string;
