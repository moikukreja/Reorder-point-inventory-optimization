/// <reference lib="webworker" />
/**
 * Must be imported before TensorFlow.js in the worker. TF.js yields between training batches with
 * requestAnimationFrame, and in a worker that callback is paused while the page is hidden, which
 * would stall model.fit() in background tabs. Yielding through setTimeout keeps training running.
 */
(self as unknown as { requestAnimationFrame: (cb: (t: number) => void) => number }).requestAnimationFrame = (cb) => setTimeout(() => cb(performance.now()), 0) as unknown as number;
export {};
