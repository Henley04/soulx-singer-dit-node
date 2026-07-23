'use strict';

// index.js — loader for the prebuilt CPU backend .node binary (macOS x64).
//
// This subpackage is selected by the main `soulx-singer-dit` shim when the
// user requests (or defaults to) the `'cpu'` backend on darwin-x64 (Intel
// Mac). It simply loads the adjacent `cpu.node` native addon and re-exports
// its surface (`Model` and `getVersion`).
//
// Built with GGML_NATIVE=OFF + Accelerate; ggml-cpu runtime-dispatches among
// AVX/AVX2/AVX512 kernels so the same binary runs on any Intel Mac.

let native;
try {
  native = require('./cpu.node');
} catch (err) {
  throw new Error(
    'soulx-singer-dit-darwin-x64-cpu: failed to load the prebuilt CPU ' +
      'binary (./cpu.node).\n' +
      'This usually means the package was installed incompletely or the ' +
      'binary does not match your Node.js ABI / OS / architecture.\n' +
      'Try reinstalling:  npm install soulx-singer-dit-darwin-x64-cpu\n' +
      'Original error: ' + (err && err.message ? err.message : String(err))
  );
}

if (!native || typeof native.Model !== 'function') {
  throw new Error(
    'soulx-singer-dit-darwin-x64-cpu: the prebuilt CPU binary loaded but ' +
      'did not export a Model class. The .node file may be corrupt or ' +
      'built for a different binding version. Try reinstalling the package.'
  );
}

module.exports = native;
