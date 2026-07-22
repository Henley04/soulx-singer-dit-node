'use strict';

// index.js — loader for the prebuilt CPU backend .node binary.
//
// This subpackage is selected by the main `soulx-singer-dit` shim when the
// user requests (or defaults to) the `'cpu'` backend on win32-x64. It simply
// loads the adjacent `cpu.node` native addon and re-exports its surface
// (`Model` and `getVersion`).

let native;
try {
  // The prebuilt binary sits next to this file in the published package.
  native = require('./cpu.node');
} catch (err) {
  // The .node file is missing or failed to dlopen. This usually means the
  // package was installed incompletely or the binary was built for a
  // different Node ABI / architecture. Surface a clear, actionable error
  // rather than the raw dlopen message.
  throw new Error(
    'soulx-singer-dit-win32-x64-cpu: failed to load the prebuilt CPU binary ' +
      '(./cpu.node).\n' +
      'This usually means the package was installed incompletely or the ' +
      'binary does not match your Node.js ABI / OS / architecture.\n' +
      'Try reinstalling:  npm install soulx-singer-dit-win32-x64-cpu\n' +
      'Original error: ' + (err && err.message ? err.message : String(err))
  );
}

// Sanity check: the native module must export the Model class. If it does
// not, the .node was likely replaced by a stale or unrelated build.
if (!native || typeof native.Model !== 'function') {
  throw new Error(
    'soulx-singer-dit-win32-x64-cpu: the prebuilt CPU binary loaded but ' +
      'did not export a Model class. The .node file may be corrupt or ' +
      'built for a different binding version. Try reinstalling the package.'
  );
}

module.exports = native;
