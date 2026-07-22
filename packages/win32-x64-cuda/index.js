'use strict';

// index.js — loader for the prebuilt CUDA backend .node binary.
//
// This subpackage is selected by the main `soulx-singer-dit` shim when the
// user requests the `'cuda'` backend on win32-x64. It simply loads the
// adjacent `cuda.node` native addon and re-exports its surface (`Model` and
// `getVersion`). The C++ binding is identical to the CPU build; only the
// compiled ggml backend differs (linked against the CUDA runtime / cuBLAS).

let native;
try {
  // The prebuilt binary sits next to this file in the published package.
  native = require('./cuda.node');
} catch (err) {
  // The .node file is missing or failed to dlopen. This usually means the
  // package was installed incompletely, the binary was built for a different
  // Node ABI / architecture, or the CUDA runtime is not available on the
  // system. Surface a clear, actionable error rather than the raw dlopen msg.
  throw new Error(
    'soulx-singer-dit-win32-x64-cuda: failed to load the prebuilt CUDA ' +
      'binary (./cuda.node).\n' +
      'This usually means the package was installed incompletely, the binary ' +
      'does not match your Node.js ABI / OS / architecture, or the CUDA ' +
      'runtime (cudart / cuBLAS) is not available on this system.\n' +
      'Try reinstalling:  npm install soulx-singer-dit-win32-x64-cuda\n' +
      'Original error: ' + (err && err.message ? err.message : String(err))
  );
}

// Sanity check: the native module must export the Model class. If it does
// not, the .node was likely replaced by a stale or unrelated build.
if (!native || typeof native.Model !== 'function') {
  throw new Error(
    'soulx-singer-dit-win32-x64-cuda: the prebuilt CUDA binary loaded but ' +
      'did not export a Model class. The .node file may be corrupt or ' +
      'built for a different binding version. Try reinstalling the package.'
  );
}

module.exports = native;
