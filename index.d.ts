// Type definitions for soulx-singer-dit 0.1.0
// Project: https://github.com/Henley04/soulx-singer-dit-node
//
// These definitions describe the public surface of the JS shim (index.js).
// The actual implementation lives in platform-specific prebuilt subpackages
// (e.g. `soulx-singer-dit-win32-x64-cpu`); this file is the single source of
// truth for consumers of the main package.

/**
 * Supported inference backends.
 *
 * - `'cpu'`    — pure CPU execution via ggml (always available on supported
 *                platforms).
 * - `'vulkan'` — GPU acceleration via the Vulkan backend.
 * - `'cuda'`   — GPU acceleration via the CUDA backend (NVIDIA only).
 */
export type Backend = 'cpu' | 'vulkan' | 'cuda';

/**
 * Options accepted by `new Model(...)` and `loadModel(...)`.
 *
 * `backend` selects which prebuilt subpackage to load. When omitted it
 * defaults to `'cpu'`. The chosen backend must be installed for the current
 * platform (see `listBackends()`).
 */
export interface LoadOptions {
  backend?: Backend;
}

/**
 * Arguments to `model.forward(...)`.
 *
 * The native binding runs a single forward pass of the DiT and returns the
 * predicted velocity field. Tensor layout is row-major with the leading
 * dimension being `MEL_DIM` (i.e. arrays are flattened as
 * `[MEL_DIM, T]` / `[HIDDEN, T]`).
 *
 *  - `x`    — input mel state, length `MEL_DIM * T`  (128 * T).
 *  - `cond` — conditioning hidden states, length `HIDDEN * T` (1024 * T).
 *  - `t`    — continuous diffusion timestep in [0, 1].
 *  - `T`    — number of frames in this chunk; must be positive.
 */
export interface ForwardOptions {
  x: Float32Array;
  cond: Float32Array;
  t: number;
  T: number;
}

/**
 * Arguments to `model.reverseDiffusion(...)`.
 *
 * Performs reverse diffusion (sampling) to generate a target mel chunk that
 * continues from a prompt mel chunk, conditioned on hidden states.
 *
 *  - `promptMel`  — prompt mel spectrogram, length `MEL_DIM * promptLen`.
 *  - `cond`       — conditioning hidden states, length
 *                   `HIDDEN * (promptLen + targetLen)`.
 *  - `z`          — initial noise, length `MEL_DIM * targetLen`.
 *  - `promptLen`  — number of prompt frames; must be positive.
 *  - `targetLen`  — number of frames to generate; must be positive.
 *  - `nSteps`     — number of reverse-diffusion steps; must be positive.
 *  - `seed`       — optional RNG seed (defaults to 12345 for parity with the
 *                   reference C++ implementation).
 */
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
 * A loaded SoulX-Singer-DiT model.
 *
 * Construct with `new Model(path, options?)` or via the async factory
 * `loadModel(path, options?)`. Once loaded, call `forward` for a single
 * diffusion step or `reverseDiffusion` for full sampling. Call `release()`
 * to free native resources eagerly; otherwise the underlying GGUF context
 * is freed when the instance is garbage-collected.
 */
export class Model {
  /**
   * Load a GGUF model from `path`.
   *
   * @param path     Absolute or relative path to the `.gguf` model file.
   * @param options  Optional backend selection (defaults to `'cpu'`).
   */
  constructor(path: string, options?: LoadOptions);

  /**
   * Run a single forward pass.
   *
   * @returns A `Float32Array` of length `MEL_DIM * T` (128 * T) holding the
   *          predicted velocity field.
   */
  forward(opts: ForwardOptions): Float32Array;

  /**
   * Run reverse diffusion to generate a target mel chunk.
   *
   * @returns A `Float32Array` of length `MEL_DIM * targetLen`
   *          (128 * targetLen) holding the generated mel spectrogram.
   */
  reverseDiffusion(opts: ReverseDiffusionOptions): Float32Array;

  /**
   * Free native resources eagerly. After this call the instance is no longer
   * usable. Calling `release()` more than once is a no-op.
   */
  release(): void;

  /**
   * Static convenience factory mirroring the native binding.
   * Prefer the top-level `loadModel()` for promise-based usage.
   */
  static loadModel(path: string, options?: LoadOptions): Model;
}

/**
 * Convenience factory: asynchronously load a `Model`.
 *
 * Equivalent to `new Model(path, options)` but returns a Promise so callers
 * can `await loadModel(...)` and handle errors with `try/catch` around an
 * `await` expression rather than around a synchronous constructor.
 *
 * @param path     Path to the `.gguf` model file.
 * @param options  Optional backend selection (defaults to `'cpu'`).
 */
export function loadModel(path: string, options?: LoadOptions): Promise<Model>;

/**
 * Returns the binding version string (e.g. `"0.1.0"`).
 *
 * Works even when no backend subpackage is installed, so it can be used for
 * diagnostics without forcing a native load.
 */
export function getVersion(): string;

/**
 * Returns the list of backends whose prebuilt subpackage is installed and
 * resolvable on the current platform. The `.node` files are not loaded by
 * this call; it only checks resolvability. Returns `[]` on platforms with no
 * prebuilt support at all.
 */
export function listBackends(): Backend[];

/**
 * Mel dimensionality of the model. Input/output mel arrays are laid out with
 * this as their leading dimension. Mirrors `MEL_DIM` on the C++ side.
 */
export const MEL_DIM: 128;

/**
 * Hidden-state dimensionality of the conditioning stream. Conditioning arrays
 * are laid out with this as their leading dimension. Mirrors `HIDDEN` on the
 * C++ side.
 */
export const HIDDEN: 1024;
