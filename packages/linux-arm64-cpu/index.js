'use strict';

// index.js — loader for the prebuilt CPU backend .node binary (Linux arm64).
//
// This subpackage is selected by the main `soulx-singer-dit` shim when the
// user requests (or defaults to) the `'cpu'` backend on linux-arm64. It
// simply loads the adjacent `cpu.node` native addon and re-exports its
// surface (`Model` and `getVersion`).
//
// The .node was built with GGML_NATIVE=OFF, so it runtime-dispatches to the
// best ARM ISA (NEON/i8mm/SVE) via ggml_cpu_has_*() — the same binary runs on
// any aarch64 CPU (Graviton, Apple Silicon under Linux, Snapdragon).

let native;
try {
  native = require('./cpu.node');
} catch (err) {
  throw new Error(
    'soulx-singer-dit-linux-arm64-cpu: failed to load the prebuilt CPU ' +
      'binary (./cpu.node).\n' +
      'This usually means the package was installed incompletely or the ' +
      'binary does not match your Node.js ABI / OS / architecture.\n' +
      'Try reinstalling:  npm install soulx-singer-dit-linux-arm64-cpu\n' +
      'Original error: ' + (err && err.message ? err.message : String(err))
  );
}

if (!native || typeof native.Model !== 'function') {
  throw new Error(
    'soulx-singer-dit-linux-arm64-cpu: the prebuilt CPU binary loaded but ' +
      'did not export a Model class. The .node file may be corrupt or ' +
      'built for a different binding version. Try reinstalling the package.'
  );
}

module.exports = native;
