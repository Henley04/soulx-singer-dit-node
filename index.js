'use strict';

// index.js — JS shim for the soulx-singer-dit native binding.
//
// This module is the public entry point of the `soulx-singer-dit` package.
// It does not contain the native .node binary itself; instead it selects and
// loads the appropriate prebuilt subpackage for the current platform/arch
// (e.g. `soulx-singer-dit-win32-x64-cpu`) and re-exports its `Model` class
// and `getVersion` function.
//
// The native binding is responsible for all inference; this file only handles
// backend selection, friendly error messages, and a couple of convenience
// helpers (`loadModel`, `listBackends`).

// Architecture constants mirrored from the C++ side (see src/infer.h).
// Kept in sync with the native binding so JS callers can size their tensors
// without hard-coding magic numbers.
const MEL_DIM = 128;
const HIDDEN = 1024;

// Allowed backend identifiers. Anything outside this set is rejected with a
// clear TypeError before we attempt to require a (potentially attacker-named)
// subpackage.
const VALID_BACKENDS = ['cpu', 'vulkan', 'cuda'];

// Cache of already-loaded native modules keyed by backend name. Subsequent
// loads of the same backend return the cached export object so we do not pay
// the dlopen cost twice for the same .node file.
const nativeCache = new Map();

// Per-platform mapping of backend -> subpackage name.
//
// Each (platform, arch) ships a CPU backend subpackage whose .node was built
// with GGML_NATIVE=OFF so it runtime-dispatches to the best ISA (AMX/AVX512/
// AVX2 on x86, NEON/i8mm/SVE on ARM) via ggml_cpu_has_*() — the SAME binary
// runs on any CPU of that arch family. Windows x64 additionally ships Vulkan
// and CUDA subpackages. Keys are `${platform}-${arch}` strings; a null entry
// means the platform is recognized but has no prebuilt backend shipped.
const PLATFORM_BACKENDS = {
  'win32-x64': {
    cpu: 'soulx-singer-dit-win32-x64-cpu',
    vulkan: 'soulx-singer-dit-win32-x64-vulkan',
    cuda: 'soulx-singer-dit-win32-x64-cuda',
  },
  'linux-x64': {
    cpu: 'soulx-singer-dit-linux-x64-cpu',
  },
  'linux-arm64': {
    cpu: 'soulx-singer-dit-linux-arm64-cpu',
  },
  'darwin-arm64': {
    cpu: 'soulx-singer-dit-darwin-arm64-cpu',
  },
  // No prebuilts shipped for the following; listed explicitly so the error
  // message can mention them rather than emit a generic "unsupported".
  'linux-arm': null,
};

/**
 * Returns the platform-key string used to look up available backends.
 * Exposed for tests; not part of the public API.
 */
function platformKey() {
  return process.platform + '-' + process.arch;
}

/**
 * Validates that `backend` is one of the supported backend identifiers.
 * Throws a TypeError with a helpful message otherwise.
 */
function assertValidBackend(backend) {
  if (backend !== undefined && backend !== null) {
    if (typeof backend !== 'string' || !VALID_BACKENDS.includes(backend)) {
      throw new TypeError(
        "options.backend must be one of 'cpu', 'vulkan', or 'cuda' " +
          '(got ' + JSON.stringify(backend) + ')'
      );
    }
  }
}

/**
 * Resolves the requested backend to a concrete one, defaulting to 'cpu' when
 * not specified. The caller is responsible for ensuring the backend is
 * available on the current platform (see `listBackends`).
 */
function resolveBackend(options) {
  const opts = options || {};
  const backend = opts.backend === undefined ? 'cpu' : opts.backend;
  assertValidBackend(backend);
  return backend;
}

/**
 * Loads (and caches) the native module for the given backend on the current
 * platform. Returns the raw export object `{ Model, getVersion }`.
 *
 * Throws a friendly Error if the platform has no prebuilt backend shipped at
 * all, or if the requested subpackage is not installed (e.g. the user only
 * installed the CPU variant but asked for 'cuda').
 */
function loadNative(backend) {
  assertValidBackend(backend);

  // Fast path: already loaded this backend in this process.
  const cached = nativeCache.get(backend);
  if (cached) return cached;

  const key = platformKey();
  const backends = PLATFORM_BACKENDS[key];

  if (!backends) {
    throw new Error(
      'soulx-singer-dit has no prebuilt binary for platform ' +
        JSON.stringify(key) + '.\n' +
        'Supported prebuilt platforms: win32-x64, linux-x64, linux-arm64, ' +
        'darwin-arm64.\n' +
        'To build from source, see the repository README.'
    );
  }

  const subpackageName = backends[backend];
  if (!subpackageName) {
    // Should not happen given the validation above, but guard anyway.
    throw new Error(
      "Backend '" + backend + "' is not available on platform " +
        JSON.stringify(key)
    );
  }

  let nativeModule;
  try {
    // The subpackage's index.js in turn requires the prebuilt .node file.
    // Use the platform-specific subpackage name resolved above (e.g.
    // 'soulx-singer-dit-linux-x64-cpu') so the correct binary is loaded on
    // every supported platform.
    nativeModule = require(subpackageName);
  } catch (err) {
    throw new Error(
      'soulx-singer-dit: failed to load backend ' +
        JSON.stringify(backend) + ' (' + subpackageName + ') on ' +
        JSON.stringify(key) + '.\n' +
        'The subpackage may not be installed, or the prebuilt .node is ' +
        'missing or corrupt.\n' +
        'Install it with:  npm install ' + subpackageName + '\n' +
        'Original error: ' + (err && err.message ? err.message : String(err))
    );
  }

  if (!nativeModule || typeof nativeModule.Model !== 'function') {
    throw new Error(
      'soulx-singer-dit: backend ' + JSON.stringify(backend) +
        ' loaded but did not export a Model class. The .node file may be ' +
        'corrupt or built for a different ABI.'
    );
  }

  nativeCache.set(backend, nativeModule);
  return nativeModule;
}

/**
 * Returns the list of backends that have an installed prebuilt subpackage on
 * the current platform. This is determined by resolving each candidate
 * subpackage's module path; missing ones are silently skipped. No .node
 * file is dlopen'd by this call.
 *
 * @returns {Backend[]} e.g. ['cpu'] or ['cpu', 'vulkan', 'cuda']
 */
function listBackends() {
  const key = platformKey();
  const backends = PLATFORM_BACKENDS[key];
  if (!backends) return [];

  const available = [];
  for (const backend of VALID_BACKENDS) {
    const subpackageName = backends[backend];
    if (!subpackageName) continue;
    // Use the cache so we do not re-dlopen already-loaded backends, and so
    // we do not disturb a backend that was already successfully loaded.
    if (nativeCache.has(backend)) {
      available.push(backend);
      continue;
    }
    try {
      // require.resolve only checks that the module is resolvable from the
      // current module graph; it does not execute or dlopen the .node file.
      require.resolve(subpackageName);
      available.push(backend);
    } catch (_e) {
      // Subpackage not installed; skip silently.
    }
  }
  return available;
}

/**
 * Convenience factory: load a Model asynchronously.
 *
 * Mirrors the static `Model.loadModel` on the native side but returns a
 * Promise so callers can `await loadModel(...)` for ergonomic error handling
 * without an explicit `try/catch` around `new Model(...)`. The actual native
 * load is synchronous; the Promise wrapper is for ergonomics and to keep the
 * surface stable if loading becomes async in the future.
 *
 * @param {string} path  Path to the GGUF model file.
 * @param {LoadOptions} [options]  Optional backend selection.
 * @returns {Promise<Model>}
 */
function loadModel(path, options) {
  // Defer to a microtask so the returned promise rejects rather than throws
  // synchronously when the backend is invalid or unavailable.
  return Promise.resolve().then(() => {
    const backend = resolveBackend(options);
    const nativeModule = loadNative(backend);
    // Return the native Model instance directly. The native class already
    // has forward / reverseDiffusion / release on its prototype.
    return new nativeModule.Model(path, options);
  });
}

// --- Public surface -------------------------------------------------------
//
// `Model` is exposed through a proxy class so that simply requiring this shim
// does not force a dlopen of the default backend. Callers that only want
// `getVersion()` or `listBackends()` (e.g. a diagnostic CLI) can do so without
// needing a prebuilt binary to be present.
//
// `new Model(path, options?)` resolves the backend from `options.backend`
// (default 'cpu'), loads the corresponding native subpackage, and returns the
// native Model instance. Returning a non-`this` object from a constructor is
// a long-standing JS pattern; the returned instance carries the native
// `forward` / `reverseDiffusion` / `release` methods on its own prototype.
// Callers should rely on the documented method surface rather than
// `instanceof Model`.

let _ModelProxy = null;

function getModelClass() {
  if (_ModelProxy) return _ModelProxy;

  _ModelProxy = class Model {
    constructor(path, options) {
      const backend = resolveBackend(options);
      const nativeModule = loadNative(backend);
      // Delegate to the native Model constructor and return the real native
      // instance (which carries forward/reverseDiffusion/release).
      return new nativeModule.Model(path, options);
    }

    // Static convenience method: Model.loadModel(path, options?) -> Model.
    // Provided for parity with the native binding; prefer the top-level
    // `loadModel` for promise-based usage.
    static loadModel(path, options) {
      const backend = resolveBackend(options);
      const nativeModule = loadNative(backend);
      return nativeModule.Model.loadModel(path, options);
    }
  };

  return _ModelProxy;
}

/**
 * Returns the binding version string. Purely informational and works even
 * when no backend subpackage is installed (the shim version is kept in
 * lockstep with the native binding's reported version).
 *
 * @returns {string} e.g. "0.1.0"
 */
function getVersion() {
  return '0.1.0';
}

module.exports = {
  // Lazy getter so requiring the shim is side-effect free (no dlopen until
  // the caller actually constructs a Model or calls loadModel).
  get Model() {
    return getModelClass();
  },
  loadModel,
  getVersion,
  listBackends,
  // Constants — also re-exported from index.d.ts.
  MEL_DIM,
  HIDDEN,
  // Exposed for advanced users / tests that want the raw native module for a
  // specific backend without going through the Model proxy.
  // Not part of the documented public API.
  _loadNative: loadNative,
};
