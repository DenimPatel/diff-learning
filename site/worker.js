// Runs lesson code with Pyodide (CPython compiled to WebAssembly), off the main thread.
// Protocol:  main -> {type: 'run', id, code}
//            worker -> {type: 'status', text} | {type: 'ready'} | {type: 'out', id, stream, text} | {type: 'done', id, error?, seconds}
// Stopping a run = terminating this worker (Python can't be interrupted without cross-origin isolation).

const PYODIDE_URL = 'https://cdn.jsdelivr.net/pyodide/v0.29.5/full/';
importScripts(PYODIDE_URL + 'pyodide.js');

let currentId = null;
const post = (msg) => self.postMessage(msg);

const ready = (async () => {
  post({ type: 'status', text: 'Loading Python…' });
  const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
  pyodide.setStdout({ batched: (text) => post({ type: 'out', id: currentId, stream: 'stdout', text }) });
  pyodide.setStderr({ batched: (text) => post({ type: 'out', id: currentId, stream: 'stderr', text }) });
  // the dataset: lesson code reads input.txt, exactly like when you run it locally
  post({ type: 'status', text: 'Loading dataset…' });
  const names = await (await fetch('data/names.txt')).text();
  pyodide.FS.writeFile('input.txt', names);
  post({ type: 'ready' });
  return pyodide;
})();

self.onmessage = async (event) => {
  const { type, id, code } = event.data;
  if (type !== 'run') return;
  let pyodide;
  try {
    pyodide = await ready;
  } catch (err) {
    post({ type: 'done', id, error: 'Could not load Python: ' + err.message });
    return;
  }
  currentId = id;
  const t0 = performance.now();
  try {
    if (/^\s*(import|from)\s+numpy/m.test(code)) {
      post({ type: 'status', text: 'Loading numpy…' });
      await pyodide.loadPackage('numpy');
    }
    post({ type: 'status', text: 'Running…' });
    const globals = pyodide.globals.get('dict')(); // fresh namespace per run
    globals.set('__name__', '__main__');
    await pyodide.runPythonAsync(code, { globals, filename: 'code.py' });
    globals.destroy();
    post({ type: 'done', id, seconds: (performance.now() - t0) / 1000 });
  } catch (err) {
    post({ type: 'done', id, error: String(err.message || err), seconds: (performance.now() - t0) / 1000 });
  }
};
