'use strict';

// scripts/postinstall.js
//
// Invoked by `npm install` via the package.json `install` script. Its only job
// is to be informative: it prints which prebuilt subpackages are available
// for the current platform and, if none are installed, prints a hint about
// how to install one.
//
// CRITICAL: this script MUST always exit 0. A missing prebuilt subpackage is
// an expected state (the subpackages are optionalDependencies and may not be
// installed yet, e.g. when the user only wants a specific backend). Blocking
// `npm install` here would break every downstream consumer, so any error is
// swallowed and the process exits successfully.

const packageName = 'soulx-singer-dit';

// Backend -> npm subpackage name, for the only platform that currently ships
// prebuilts. Other platforms get a different hint.
const WIN32_X64_PACKAGES = {
  cpu: 'soulx-singer-dit-win32-x64-cpu',
  vulkan: 'soulx-singer-dit-win32-x64-vulkan',
  cuda: 'soulx-singer-dit-win32-x64-cuda',
};

function log(msg) {
  // Use console.error so output is visible even when npm suppresses stdout.
  process.stderr.write(msg + '\n');
}

function canResolve(name) {
  try {
    // require.resolve throws if the module is not resolvable. We do not want
    // to actually dlopen the .node, and require.resolve alone does not.
    require.resolve(name);
    return true;
  } catch (_e) {
    return false;
  }
}

function main() {
  const platform = process.platform;
  const arch = process.arch;
  const key = platform + '-' + arch;

  log('');
  log('[' + packageName + '] postinstall check');
  log('  platform: ' + key);

  if (key !== 'win32-x64') {
    log('  No prebuilt subpackages are shipped for ' + key + '.');
    log('  Supported prebuilt platform: win32-x64.');
    log('  To build from source, see the repository README.');
    log('');
    return;
  }

  log('  Available prebuilt subpackages for win32-x64:');
  const installed = [];
  const missing = [];
  for (const backend of Object.keys(WIN32_X64_PACKAGES)) {
    const sub = WIN32_X64_PACKAGES[backend];
    if (canResolve(sub)) {
      installed.push(backend + ' (' + sub + ')');
    } else {
      missing.push(backend + ' (' + sub + ')');
    }
  }

  if (installed.length > 0) {
    log('    installed: ' + installed.join(', '));
  } else {
    log('    none installed');
  }
  if (missing.length > 0) {
    log('    not installed: ' + missing.join(', '));
  }

  if (installed.length === 0) {
    log('');
    log('  No prebuilt backend is currently installed.');
    log('  Install one to enable inference, e.g.:');
    log('    npm install ' + WIN32_X64_PACKAGES.cpu);
    log('    npm install ' + WIN32_X64_PACKAGES.vulkan);
    log('    npm install ' + WIN32_X64_PACKAGES.cuda);
  }

  log('');
}

try {
  main();
} catch (err) {
  // Never block install: log and continue.
  log('[' + packageName + '] postinstall: ' +
    (err && err.message ? err.message : String(err)));
}

// Always exit 0 — see file header.
process.exit(0);
