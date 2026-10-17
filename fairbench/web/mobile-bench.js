/* Browser-only FP32 matrix benchmark. No host benchmark APIs or external libraries. */
(function (root) {
  'use strict';
  const VERSION = 'fairbench.browser/2';
  const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

  function inputs(n, seed = 1337) {
    let state = seed >>> 0;
    const values = () => {
      state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
      return ((state >>> 0) / 4294967296 - 0.5);
    };
    return { a: Float32Array.from({ length: n * n }, values), b: Float32Array.from({ length: n * n }, values) };
  }

  function matmul(a, b, n, out = new Float32Array(n * n)) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) sum = Math.fround(sum + Math.fround(a[i * n + k] * b[k * n + j]));
        out[i * n + j] = sum;
      }
    }
    return out;
  }

  function validate(actual, expected, stride = 1) {
    if (actual.length !== expected.length * stride) throw new Error('出力サイズが一致しません');
    let error = 0, norm = 0, maxError = 0;
    for (let i = 0; i < expected.length; i++) {
      const value = actual[i * stride];
      if (!Number.isFinite(value) || !Number.isFinite(expected[i])) throw new Error('計算結果が有限数ではありません');
      const delta = value - expected[i];
      error += delta * delta; norm += expected[i] * expected[i];
      maxError = Math.max(maxError, Math.abs(delta));
    }
    const relative = Math.sqrt(error / Math.max(norm, 1e-30));
    if (relative > 0.001 && maxError > 0.00001) throw new Error('CPUとの正解照合に失敗しました');
    return { valid: true, relative_l2_error: relative, max_abs_error: maxError, tolerance: 0.001 };
  }

  function sample(iterations, wallMs, n) {
    return { iterations, wall_ms: wallMs, mean_ms: wallMs / iterations, gflops: 2 * n ** 3 * iterations / (wallMs / 1000) / 1e9 };
  }

  function summarise(samples) {
    return { gflops: median(samples.map((s) => s.gflops)), mean_ms: median(samples.map((s) => s.mean_ms)), samples };
  }

  function memoryCopy() {
    const source = new Uint8Array(8 * 1024 * 1024).fill(123), destination = new Uint8Array(source.length);
    destination.set(source);
    const rates = [];
    for (let i = 0; i < 5; i++) {
      const started = performance.now();
      destination.set(source);
      const elapsed = performance.now() - started;
      if (elapsed > 0) rates.push(2 * source.length / (elapsed / 1000) / 1e9);
    }
    if (!rates.length || destination[destination.length - 1] !== 123) return { available: false };
    return { available: true, gb_per_s: median(rates), buffer_bytes: source.length, samples_gb_per_s: rates,
      method: 'TypedArray.set, read + write, includes cache effects; not NPU or DRAM peak bandwidth' };
  }

  function cpu(n, notify, durationMs = 1000) {
    const { a, b } = inputs(n), out = new Float32Array(n * n), samples = [];
    // A known 2x2 product checks the kernel independently of its timed inputs.
    validate(matmul(new Float32Array([1, 2, 3, 4]), new Float32Array([5, 6, 7, 8]), 2), new Float32Array([19, 22, 43, 50]));
    matmul(a, b, n, out); matmul(a, b, n, out);
    for (let repeat = 0; repeat < 3; repeat++) {
      let iterations = 0;
      const started = performance.now();
      do { matmul(a, b, n, out); iterations++; } while (performance.now() - started < durationMs);
      samples.push(sample(iterations, performance.now() - started, n));
      notify(repeat + 1);
    }
    return { result: { backend: 'javascript-fp32', threads: 1, ...summarise(samples), correctness: { valid: true, method: 'known 2x2 kernel check' } }, reference: out };
  }

  function gpuInputs(n) {
    if (![256, 512, 1024].includes(n)) throw new Error('GPU行列サイズが無効です');
    const { a } = inputs(n), b = new Float32Array(n * n);
    const scale = Math.fround(1 / Math.sqrt(n));
    // Dense, symmetric orthogonal weights keep dependent GEMMs bounded. Every
    // pass consumes the preceding output, so overwriting draws cannot be elided.
    for (let row = 0; row < n; row++) for (let col = 0; col < n; col++) {
      let bits = row & col, parity = 0;
      while (bits) { parity ^= bits & 1; bits >>>= 1; }
      b[row * n + col] = parity ? -scale : scale;
    }
    return { a, b };
  }

  function gpuReference(n, passes) {
    const { a } = inputs(n), out = Float64Array.from(a);
    const scale = Math.fround(1 / Math.sqrt(n));
    if (passes % 2) {
      // Independent O(N^2 log N) CPU Walsh-Hadamard transform validates ALL
      // GPU output values, without an expensive N^3 JavaScript reference.
      for (let row = 0; row < n; row++) for (let step = 1; step < n; step *= 2) {
        for (let block = 0; block < n; block += step * 2) for (let j = 0; j < step; j++) {
          const x = row * n + block + j, y = x + step, left = out[x], right = out[y];
          out[x] = left + right; out[y] = left - right;
        }
      }
      for (let i = 0; i < out.length; i++) out[i] *= scale;
    }
    const factor = Math.pow(n * scale * scale, Math.floor(passes / 2));
    for (let i = 0; i < out.length; i++) out[i] *= factor;
    return out;
  }

  const cancelled = (signal) => { if (signal?.aborted) throw new DOMException('中止しました', 'AbortError'); };

  async function waitForGpu(gl, signal) {
    const fence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (!fence) throw new Error('GPU完了待ちを作成できません');
    const started = performance.now();
    gl.flush();
    try {
      for (;;) {
        cancelled(signal);
        if (gl.isContextLost()) throw new Error('GPUとの接続が切れました');
        const state = gl.clientWaitSync(fence, 0, 0);
        if (state === gl.ALREADY_SIGNALED || state === gl.CONDITION_SATISFIED) return;
        if (state !== gl.TIMEOUT_EXPIRED) throw new Error('GPU完了待ちに失敗しました');
        if (performance.now() - started > 30000) throw new Error('GPU処理が30秒を超えました。負荷を下げて再実行してください。');
        // WebGL fences can only become signalled after returning to the event loop.
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    } finally { gl.deleteSync(fence); }
  }

  function createGpu(n) {
    const canvas = document.createElement('canvas');
    canvas.width = n / 4; canvas.height = n;
    const gl = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false });
    if (!gl) throw new Error('このブラウザではWebGL2を利用できません');
    const textures = [], shaders = [], framebuffers = [];
    let program;
    const dispose = () => {
      textures.forEach((texture) => gl.deleteTexture(texture));
      shaders.forEach((shader) => gl.deleteShader(shader));
      framebuffers.forEach((framebuffer) => gl.deleteFramebuffer(framebuffer));
      if (program) gl.deleteProgram(program);
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    };
    try {
      if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('GPUの浮動小数点出力に未対応です');
      if (gl.getParameter(gl.MAX_TEXTURE_SIZE) < n) throw new Error('この行列サイズに対応していません');
      const shader = (type, source) => {
        const handle = gl.createShader(type); shaders.push(handle);
        gl.shaderSource(handle, source); gl.compileShader(handle);
        if (!gl.getShaderParameter(handle, gl.COMPILE_STATUS)) throw new Error('GPUプログラムの準備に失敗しました: ' + gl.getShaderInfoLog(handle));
        return handle;
      };
      program = gl.createProgram();
      gl.attachShader(program, shader(gl.VERTEX_SHADER, `#version 300 es
        void main() { vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`));
      gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `#version 300 es
        precision highp float; precision highp int;
        uniform highp sampler2D leftMat; uniform highp sampler2D rightMat;
        out vec4 result;
        void main() {
          ivec2 pos = ivec2(gl_FragCoord.xy); int col = pos.x * 4; vec4 sum = vec4(0.0);
          for (int k = 0; k < ${n / 4}; k++) {
            vec4 a = texelFetch(leftMat, ivec2(k, pos.y), 0);
            sum += vec4(dot(a, texelFetch(rightMat, ivec2(k, col), 0)),
                        dot(a, texelFetch(rightMat, ivec2(k, col + 1), 0)),
                        dot(a, texelFetch(rightMat, ivec2(k, col + 2), 0)),
                        dot(a, texelFetch(rightMat, ivec2(k, col + 3), 0)));
          }
          result = sum;
        }`));
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error('GPUプログラムをリンクできません');
      gl.useProgram(program); gl.disable(gl.DITHER);
      const texture = (unit, data) => {
        const handle = gl.createTexture(); textures.push(handle);
        gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, handle);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, n / 4, n, 0, gl.RGBA, gl.FLOAT, data);
        return handle;
      };
      const { a, b } = gpuInputs(n);
      const buffers = [texture(0, a), texture(2, null)];
      texture(1, b); // Hadamard is symmetric, so its transpose has the same data.
      gl.uniform1i(gl.getUniformLocation(program, 'leftMat'), 0);
      gl.uniform1i(gl.getUniformLocation(program, 'rightMat'), 1);
      for (const buffer of buffers) {
        const framebuffer = gl.createFramebuffer(); framebuffers.push(framebuffer);
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, buffer, 0);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('GPU出力バッファを作成できません');
      }
      const extension = gl.getExtension('WEBGL_debug_renderer_info');
      const renderer = extension ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
      const out = new Float32Array(n * n);
      let source = 0, passes = 0, readbacks = 0;
      return { dispose, get passes() { return passes; }, get readbacks() { return readbacks; },
        info: { renderer, software_renderer: /swiftshader|llvmpipe|software/i.test(renderer), backend: 'webgl2-packed-fp32',
          workload: 'dense-orthogonal-gemm', size: n, flops_per_pass: 2 * n ** 3 },
        async batch(count, signal) {
          cancelled(signal);
          gl.viewport(0, 0, n / 4, n);
          for (let i = 0; i < count; i++) {
            gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, buffers[source]);
            gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers[1 - source]);
            gl.drawArrays(gl.TRIANGLES, 0, 3); source = 1 - source; passes++;
          }
          await waitForGpu(gl, signal);
          if (gl.getError() !== gl.NO_ERROR) throw new Error('GPU演算に失敗しました');
        },
        read() {
          readbacks++;
          gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers[source]);
          gl.readPixels(0, 0, n / 4, n, gl.RGBA, gl.FLOAT, out);
          if (gl.getError() !== gl.NO_ERROR) throw new Error('GPUの読戻しに失敗しました');
          return out;
        } };
    } catch (error) { dispose(); throw error; }
  }

  async function gpuWindow(gpu, n, batchSize, durationMs, signal) {
    let completed = 0;
    const started = performance.now();
    do { await gpu.batch(batchSize, signal); completed += batchSize; }
    while (performance.now() - started < durationMs);
    return { ...sample(completed, performance.now() - started, n), batch_size: batchSize };
  }

  async function measureGpu(size, signal, notify) {
    const candidates = size === 'auto' ? [256, 512, 1024] : [Number(size)];
    if (candidates.some((n) => ![256, 512, 1024].includes(n))) throw new Error('GPU負荷の指定が無効です');
    const trials = [];
    let best = null;
    for (const n of candidates) {
      let gpu;
      try {
        cancelled(signal); notify(`GPUの負荷を確認しています（${n} × ${n}）`);
        gpu = createGpu(n);
        await gpu.batch(1, signal);
        validate(gpu.read(), gpuReference(n, gpu.passes));
        const started = performance.now();
        await gpu.batch(4, signal);
        const msPerPass = (performance.now() - started) / 4;
        const batchSize = Math.max(1, Math.min(32, Math.floor(48 / Math.max(msPerPass, 0.1))));
        const trial = await gpuWindow(gpu, n, batchSize, 400, signal);
        validate(gpu.read(), gpuReference(n, gpu.passes));
        trials.push({ size: n, ok: true, ...trial });
        if (!best || trial.gflops > best.gflops) best = { size: n, batchSize, gflops: trial.gflops };
      } catch (error) {
        if (error.name === 'AbortError') throw error;
        trials.push({ size: n, ok: false, error: error.message });
      } finally { gpu?.dispose(); }
    }
    if (!best) throw new Error(trials.map((trial) => `${trial.size}: ${trial.error}`).join(' / '));
    const gpu = createGpu(best.size);
    try {
      await gpu.batch(5, signal);
      validate(gpu.read(), gpuReference(best.size, gpu.passes));
      const samples = [];
      for (let repeat = 0; repeat < 3; repeat++) {
        notify(`GPUを測定しています（${best.size}角・${repeat + 1}/3回）`, 3 + repeat);
        samples.push(await gpuWindow(gpu, best.size, best.batchSize, 1000, signal));
      }
      const correctness = validate(gpu.read(), gpuReference(best.size, gpu.passes));
      return { ok: true, ...gpu.info, ...summarise(samples), correctness, trials, batch_size: best.batchSize,
        completed_passes_including_warmup: gpu.passes, warmup_passes: 5, readbacks: gpu.readbacks, requested_size: size,
        measurement: 'gpu-resident-throughput-v2',
        timing: 'completed dependent GEMMs / wall time including submission, fence waits and scheduling; readback excluded',
        selection: 'best of tested sizes by 400ms trial; reported score is median of three separate 1s windows' };
    } finally { gpu.dispose(); }
  }

  const api = { VERSION, inputs, matmul, validate, sample, summarise, memoryCopy, cpu, gpuInputs, gpuReference, waitForGpu, gpuWindow, createGpu, measureGpu };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root) root.FairbenchBrowser = api;
  if (typeof document === 'undefined' && typeof self !== 'undefined') {
    self.onmessage = ({ data }) => {
      try {
        if (![128, 256].includes(data.size)) throw new Error('行列サイズが無効です');
        const memory = memoryCopy();
        const measured = cpu(data.size, (repeat) => self.postMessage({ type: 'progress', repeat }));
        self.postMessage({ type: 'done', ...measured, memory }, [measured.reference.buffer]);
      } catch (error) { self.postMessage({ type: 'error', error: error.message }); }
    };
  }
})(typeof self === 'undefined' ? null : self);
