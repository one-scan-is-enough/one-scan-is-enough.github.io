// Hero animation: scan -> Gaussian reconstruction -> Scan2Instruction -> PRPT-KS.
//
// The scene is pre-projected for one fixed bird's-eye camera (tools/build_hero.py):
// a full-quality image plus the strongest 2D Gaussians in back-to-front order.
// The scan replays the real capture: the camera path and photo poses of the
// training views, and scan points that appear once the LiDAR first saw them.
// The Gaussians grow at the reveal wavefront and hand over to the image behind it.
// Overlays work in the reference pixel frame of that image ("R"); one 2D affine
// map takes R to device pixels, shared by WebGL and the 2D canvas.

const BASE = new URL(".", import.meta.url);

const TIMELINE = { scan: 0, reconstruct: 4.7, instruct: 7.7, train: 17.6, outro: 30.2, end: 32.4 };
// The whole capture replays between these times, linearly in capture time.
const SCAN = { start: 0.6, end: 4.5 };
const CHAPTERS = [
  { start: TIMELINE.scan, title: "One handheld scan" },
  { start: TIMELINE.reconstruct, title: "Frames become a 3D Gaussian scene" },
  { start: TIMELINE.instruct, title: "Scan2Instruction compiles episodes" },
  { start: TIMELINE.train, title: "PRPT-KS post-trains RealNav" },
];
// Caption for the overview figure that closes the animation.
const OVERVIEW = { title: "From one scan to real-world navigation" };
const TYPE = {
  vln: { label: "Route following", color: "#f08c00" },
  object: { label: "Object Goal", color: "#1c7ed6" },
  need: { label: "Human Demand", color: "#d6336c" },
};
const ACCENT = "#2563eb";
const GOOD = "#0ca678";
const BAD = "#e03131";
const CAMERA_HALF_FOV = 0.62; // Half the horizontal field of view of the capture, in radians.

// PRPT-KS rollout settings.
const PRPT = {
  lanes: 8,
  decision: 0.32, // Animation seconds per decision.
  spacing: 0.125, // Meters between reference poses and between predicted waypoints.
  executed: [1, 3, 5, 7], // Every second of the eight predicted waypoints executes.
  budget: 300, // Executed waypoints per run.
  stopDistance: 0.04,
  minWaypoints: 10,
  successRadius: 1.0,
  tolerance: 0.05,
  gamma: 0.99,
  obstaclePenalty: 0.15,
  lookahead: 10,
  sigma: [1.5, 0.5],
  curriculum: 1.0,
};

const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const lerp = (a, b, t) => a + (b - a) * t;
const ramp = (a, b, x) => clamp((x - a) / (b - a));
const smooth = (a, b, x) => {
  const t = ramp(a, b, x);
  return t * t * (3 - 2 * t);
};
const easeOut = (t) => 1 - Math.pow(1 - clamp(t), 3);
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

function random(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand) {
  return Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
}

function loadImage(name) {
  const image = new Image();
  image.decoding = "async";
  image.src = new URL(name, BASE).href;
  return image.decode().then(() => image);
}

/** Fetch gzip columnar planes and interleave them into records of the given byte slots. */
async function loadRecords(name, slots, stride) {
  const response = await fetch(new URL(name, BASE));
  let bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    bytes = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  const count = bytes.length / slots.length;
  const out = new Uint8Array(count * stride);
  slots.forEach((slot, plane) => {
    const src = bytes.subarray(plane * count, (plane + 1) * count);
    for (let i = 0; i < count; i++) out[i * stride + slot] = src[i];
  });
  return { count, data: out };
}

// Splats: x u16, y u16, sigma1, sigma2, angle, pad, rgba. Scan points: x u16, y u16, rgb, first-seen time.
const loadSplats = (name) => loadRecords(name, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11], 12);
const loadPoints = (name) => loadRecords(name, [0, 1, 2, 3, 4, 5, 6, 7], 8);

/* ------------------------------------------------------------------ WebGL */

const FRONT_GLSL = `
uniform vec2 uCenterR;
uniform float uRadius;
uniform float uBand;
uniform float uTime;
float frontAt(vec2 p) {
  vec2 d = p - uCenterR;
  float a = atan(d.y, d.x);
  return uRadius * (1.0 + 0.035 * sin(a * 5.0 + uTime * 0.7) + 0.02 * sin(a * 11.0 - uTime * 1.3));
}
// 0 ahead of the wavefront, rising to 1 one band behind it.
float revealAt(vec2 p, float jitter) {
  return clamp((frontAt(p) - length(p - uCenterR) - jitter * uBand * 0.35) / uBand, 0.0, 1.0);
}`;

const VIEW_GLSL = `
uniform vec4 uAffine;
uniform vec2 uOffset;
uniform vec2 uCanvas;
vec2 toDevice(vec2 p) { return vec2(uAffine.x * p.x + uAffine.z * p.y, uAffine.y * p.x + uAffine.w * p.y) + uOffset; }
vec4 toClip(vec2 q) { return vec4(q.x / uCanvas.x * 2.0 - 1.0, 1.0 - q.y / uCanvas.y * 2.0, 0.0, 1.0); }`;

const SPLAT_VS = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aQuad;
layout(location = 1) in vec2 aCenter;
layout(location = 2) in vec3 aShape;
layout(location = 3) in vec4 aColor;
uniform vec2 uRef;
uniform vec2 uSigmaRange;
uniform float uScale;
uniform float uLowPass;
${VIEW_GLSL}
${FRONT_GLSL}
out vec4 vColor;
out vec2 vQuad;
float hash(float n) { return fract(sin(n * 12.9898) * 43758.5453); }
void main() {
  vec2 p = aCenter * uRef;
  float t = revealAt(p, hash(float(gl_InstanceID)));
  if (t <= 0.0 || t >= 1.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  float g = clamp(t / 0.5, 0.0, 1.0);
  float grow = 1.0 - pow(1.0 - g, 3.0) + 0.18 * sin(g * 3.14159);
  float s1 = exp2(mix(uSigmaRange.x, uSigmaRange.y, aShape.x)) * uScale * grow;
  float s2 = exp2(mix(uSigmaRange.x, uSigmaRange.y, aShape.y)) * uScale * grow;
  s1 = min(sqrt(s1 * s1 + uLowPass), 600.0);
  s2 = min(sqrt(s2 * s2 + uLowPass), 600.0);
  float th = aShape.z * (255.0 / 256.0) * 3.14159265;
  vec2 axis = vec2(uAffine.x * cos(th) + uAffine.z * sin(th), uAffine.y * cos(th) + uAffine.w * sin(th)) / uScale;
  vec2 q = toDevice(p) + aQuad.x * 3.0 * s1 * axis + aQuad.y * 3.0 * s2 * vec2(-axis.y, axis.x);
  gl_Position = toClip(q);
  float glow = 1.0 - smoothstep(0.0, 0.45, t);
  vec3 rgb = mix(aColor.rgb, vec3(0.15, 0.42, 0.95), glow * 0.7);
  vColor = vec4(rgb, aColor.a * smoothstep(0.0, 0.1, t) * (1.0 - smoothstep(0.72, 1.0, t)));
  vQuad = aQuad * 3.0;
}`;

const SPLAT_FS = `#version 300 es
precision mediump float;
in vec4 vColor;
in vec2 vQuad;
out vec4 outColor;
void main() {
  float m = dot(vQuad, vQuad);
  if (m > 9.0) discard;
  float a = min(vColor.a * exp(-0.5 * m), 0.99);
  if (a < 1.0 / 255.0) discard;
  outColor = vec4(vColor.rgb * a, a);
}`;

// Scan points pop in blue when the LiDAR first sees them and leave once the wavefront passes.
const POINT_VS = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aQuad;
layout(location = 1) in vec2 aCenter;
layout(location = 2) in vec4 aColor;
uniform vec2 uRef;
uniform vec3 uScan;
uniform float uPointRadius;
uniform float uPointAlpha;
${VIEW_GLSL}
${FRONT_GLSL}
out vec4 vColor;
out vec2 vQuad;
void main() {
  vec2 p = aCenter * uRef;
  float age = uScan.x - (uScan.y + aColor.a * uScan.z);
  float fade = uPointAlpha * (1.0 - revealAt(p, 0.6));
  if (age < 0.0 || fade <= 0.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  float pop = smoothstep(0.0, 0.22, age);
  float flash = exp(-age * 2.6);
  float r = uPointRadius * (0.55 + 0.45 * pop + 0.9 * flash);
  gl_Position = toClip(toDevice(p) + aQuad * r);
  vColor = vec4(mix(aColor.rgb, vec3(0.15, 0.42, 0.95), flash * 0.8), fade * pop);
  vQuad = aQuad;
}`;

const POINT_FS = `#version 300 es
precision mediump float;
in vec4 vColor;
in vec2 vQuad;
out vec4 outColor;
void main() {
  float a = vColor.a * (1.0 - smoothstep(0.55, 1.0, length(vQuad)));
  if (a < 1.0 / 255.0) discard;
  outColor = vec4(vColor.rgb * a, a);
}`;

const IMAGE_VS = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aQuad;
uniform vec2 uRef;
${VIEW_GLSL}
out vec2 vUv;
void main() {
  vUv = aQuad * 0.5 + 0.5;
  gl_Position = toClip(toDevice(vUv * uRef));
}`;

const IMAGE_FS = `#version 300 es
precision highp float;
uniform sampler2D uImage;
uniform vec2 uRef;
uniform float uOpacity;
uniform float uSaturation;
uniform float uRing;
${FRONT_GLSL}
in vec2 vUv;
out vec4 outColor;
void main() {
  vec2 p = vUv * uRef;
  vec4 c = texture(uImage, vUv);
  float t = revealAt(p, 0.0);
  float lum = dot(c.rgb, vec3(0.299, 0.587, 0.114));
  vec3 rgb = mix(vec3(lum), c.rgb, uSaturation);
  float shown = smoothstep(0.5, 1.0, t) * uOpacity;
  float edge = (frontAt(p) - length(p - uCenterR)) / (uBand * 0.16);
  float ring = exp(-edge * edge) * c.a * uRing * 0.45;
  vec4 base = vec4(rgb, c.a) * shown;
  outColor = vec4(0.15, 0.42, 0.95, 1.0) * ring + base * (1.0 - ring);
}`;

function compile(gl, vs, fs) {
  const program = gl.createProgram();
  for (const [type, source] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    gl.attachShader(program, shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  const uniforms = {};
  for (let i = 0; i < gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS); i++) {
    const name = gl.getActiveUniform(program, i).name;
    uniforms[name] = gl.getUniformLocation(program, name);
  }
  return { program, uniforms };
}

class SceneRenderer {
  constructor(canvas, manifest) {
    const gl = canvas.getContext("webgl2", { alpha: true, premultipliedAlpha: true, antialias: false });
    if (!gl) throw new Error("WebGL2 unavailable");
    this.gl = gl;
    this.manifest = manifest;
    this.splatProgram = compile(gl, SPLAT_VS, SPLAT_FS);
    this.pointProgram = compile(gl, POINT_VS, POINT_FS);
    this.imageProgram = compile(gl, IMAGE_VS, IMAGE_FS);
    this.quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    this.imageVao = this.instances(null, 0, []);
    this.splatCount = 0;
    this.pointCount = 0;
    this.texture = null;
  }

  /** A vertex array of the unit quad plus per-instance attributes [location, size, type, offset]. */
  instances(data, stride, attributes) {
    const gl = this.gl;
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    if (data) {
      gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      for (const [location, size, type, offset] of attributes) {
        gl.enableVertexAttribArray(location);
        gl.vertexAttribPointer(location, size, type, true, stride, offset);
        gl.vertexAttribDivisor(location, 1);
      }
    }
    gl.bindVertexArray(null);
    return vao;
  }

  setImage(image) {
    const gl = this.gl;
    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  setSplats({ count, data }) {
    const gl = this.gl;
    this.splatVao = this.instances(data, 12, [
      [1, 2, gl.UNSIGNED_SHORT, 0],
      [2, 3, gl.UNSIGNED_BYTE, 4],
      [3, 4, gl.UNSIGNED_BYTE, 8],
    ]);
    this.splatCount = count;
  }

  setPoints({ count, data }) {
    const gl = this.gl;
    this.pointVao = this.instances(data, 8, [
      [1, 2, gl.UNSIGNED_SHORT, 0],
      [2, 4, gl.UNSIGNED_BYTE, 4],
    ]);
    this.pointCount = count;
  }

  get ready() {
    return this.texture !== null && this.splatCount > 0;
  }

  render(view, width, height, state) {
    const gl = this.gl;
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    const m = this.manifest;
    const common = (u) => {
      gl.uniform2f(u.uRef, m.width, m.height);
      gl.uniform4f(u.uAffine, view.a, view.b, view.c, view.d);
      gl.uniform2f(u.uOffset, view.e, view.f);
      gl.uniform2f(u.uCanvas, width, height);
      gl.uniform2f(u.uCenterR, state.center[0], state.center[1]);
      gl.uniform1f(u.uRadius, state.radius);
      gl.uniform1f(u.uBand, state.band);
      gl.uniform1f(u.uTime, state.time);
    };

    if (this.pointCount > 0 && state.points > 0) {
      const point = this.pointProgram;
      gl.useProgram(point.program);
      common(point.uniforms);
      gl.uniform3f(point.uniforms.uScan, state.t, SCAN.start, SCAN.end - SCAN.start);
      gl.uniform1f(point.uniforms.uPointRadius, state.pointRadius);
      gl.uniform1f(point.uniforms.uPointAlpha, state.points);
      gl.bindVertexArray(this.pointVao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.pointCount);
    }

    if (this.ready && state.radius > 0) {
      const image = this.imageProgram;
      gl.useProgram(image.program);
      common(image.uniforms);
      gl.uniform1f(image.uniforms.uOpacity, state.opacity);
      gl.uniform1f(image.uniforms.uSaturation, state.saturation);
      gl.uniform1f(image.uniforms.uRing, state.ring);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.uniform1i(image.uniforms.uImage, 0);
      gl.bindVertexArray(this.imageVao);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      if (state.splats) {
        const splat = this.splatProgram;
        gl.useProgram(splat.program);
        common(splat.uniforms);
        gl.uniform2f(splat.uniforms.uSigmaRange, m.log2Sigma[0], m.log2Sigma[1]);
        gl.uniform1f(splat.uniforms.uScale, view.s);
        gl.uniform1f(splat.uniforms.uLowPass, m.lowPass);
        gl.bindVertexArray(this.splatVao);
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.splatCount);
      }
    }
    gl.bindVertexArray(null);
  }
}

/* ------------------------------------------------------------- Scene data */

function makeProjector(manifest) {
  const { eye, rotation: r, focal } = manifest.camera;
  const cx = manifest.width / 2;
  const cy = manifest.height / 2;
  return (x, y, z) => {
    const dx = x - eye[0], dy = y - eye[1], dz = z - eye[2];
    const px = r[0] * dx + r[1] * dy + r[2] * dz;
    const py = r[3] * dx + r[4] * dy + r[5] * dz;
    const pz = r[6] * dx + r[7] * dy + r[8] * dz;
    return [(focal * px) / pz + cx, (focal * py) / pz + cy];
  };
}

/**
 * Track for the phone from the scan path (u, v, angle triples): (u, v, du, dv) quadruples holding the
 * position smoothed with sigmaPosition samples and the velocity per sample smoothed with sigmaVelocity.
 */
/** Gaussian blur of a sequence with sigma in samples, clamping at both ends. */
function blur(values, sigma) {
  const n = values.length;
  const half = Math.ceil(3 * sigma);
  const kernel = Array.from({ length: 2 * half + 1 }, (_, i) => Math.exp(-0.5 * ((i - half) / sigma) ** 2));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0, weight = 0;
    for (let j = -half; j <= half; j++) {
      const w = kernel[j + half];
      sum += w * values[Math.min(n - 1, Math.max(0, i + j))];
      weight += w;
    }
    out[i] = sum / weight;
  }
  return out;
}

/** Scan path (u, v, angle triples) with the hand shake blurred out: positions and unwrapped angles. */
function steadyPath(path, sigma) {
  const n = path.length / 3;
  const angle = Float32Array.from({ length: n }, (_, i) => path[i * 3 + 2]);
  for (let i = 1; i < n; i++) {
    const d = angle[i] - angle[i - 1];
    angle[i] = angle[i - 1] + d - Math.round(d / (2 * Math.PI)) * 2 * Math.PI;
  }
  const axes = [0, 1].map((k) => blur(Float32Array.from({ length: n }, (_, i) => path[i * 3 + k]), sigma));
  const turn = blur(angle, sigma);
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) out.set([axes[0][i], axes[1][i], turn[i]], i * 3);
  return out;
}

function phoneTrack(path, sigmaPosition, sigmaVelocity) {
  const n = path.length / 3;
  const axis = (k) => Float32Array.from({ length: n }, (_, i) => path[i * 3 + k]);
  const u = blur(axis(0), sigmaPosition), v = blur(axis(1), sigmaPosition);
  const slow = [blur(axis(0), sigmaVelocity), blur(axis(1), sigmaVelocity)];
  const out = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1);
    out.set([u[i], v[i], (slow[0][b] - slow[0][a]) / (b - a), (slow[1][b] - slow[1][a]) / (b - a)], i * 4);
  }
  return out;
}

/** Clamp to [lo, hi] with exponential easing over the last soft units, so motion slows instead of stopping dead. */
function softClamp(x, lo, hi, soft) {
  soft = Math.min(soft, (hi - lo) / 2);
  if (x < lo + soft) return lo + soft * Math.exp((x - lo - soft) / soft);
  if (x > hi - soft) return hi - soft * Math.exp((hi - soft - x) / soft);
  return x;
}

/** Points along a polyline every step meters, keeping both ends. */
function resample(points, step) {
  const out = [points[0]];
  let carry = 0;
  for (let i = 1; i < points.length; i++) {
    const [ax, az] = points[i - 1], [bx, bz] = points[i];
    const length = Math.hypot(bx - ax, bz - az);
    let s = step - carry;
    while (s <= length) {
      out.push([ax + ((bx - ax) * s) / length, az + ((bz - az) * s) / length]);
      s += step;
    }
    carry = length - (s - step);
  }
  const last = points[points.length - 1];
  if (distance(out[out.length - 1], last) > 1e-6) out.push(last);
  return out;
}

class Polyline {
  constructor(world, project) {
    this.world = world;
    this.screen = world.map(([x, z]) => project(x, 0.02, z));
    this.lengths = [0];
    for (let i = 1; i < world.length; i++) this.lengths.push(this.lengths[i - 1] + distance(world[i], world[i - 1]));
    this.length = this.lengths[this.lengths.length - 1];
  }

  /** Trace the screen polyline up to a fraction of its length; return the head point. */
  trace(ctx, fraction) {
    const target = fraction * this.length;
    ctx.moveTo(this.screen[0][0], this.screen[0][1]);
    for (let i = 1; i < this.screen.length; i++) {
      if (this.lengths[i] >= target) {
        const t = (target - this.lengths[i - 1]) / (this.lengths[i] - this.lengths[i - 1] || 1);
        const x = lerp(this.screen[i - 1][0], this.screen[i][0], t);
        const y = lerp(this.screen[i - 1][1], this.screen[i][1], t);
        ctx.lineTo(x, y);
        return [x, y];
      }
      ctx.lineTo(this.screen[i][0], this.screen[i][1]);
    }
    return this.screen[this.screen.length - 1];
  }
}

function rgba(hex, alpha) {
  const v = parseInt(hex.slice(1), 16);
  return `rgba(${v >> 16},${(v >> 8) & 255},${v & 255},${alpha})`;
}

function glowSprite(color, size = 64) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, color);
  gradient.addColorStop(0.35, color.replace(/[\d.]+\)$/, "0.35)"));
  gradient.addColorStop(1, color.replace(/[\d.]+\)$/, "0)"));
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);
  return canvas;
}

/** Tint the warped free-space map: a soft fill, a dot grid and strong outlines. */
function mapOverlay(image) {
  const w = image.width, h = image.height;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(image, 0, 0);
  const source = ctx.getImageData(0, 0, w, h);
  const free = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) free[i] = source.data[i * 4] > 127 ? 1 : 0;
  const out = ctx.createImageData(w, h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      if (!free[i]) continue;
      const edge = !free[i - 1] || !free[i + 1] || !free[i - w] || !free[i + w];
      const dot = x % 10 === 0 && y % 10 === 0;
      const alpha = edge ? 230 : dot ? 130 : 46;
      out.data[i * 4] = 37;
      out.data[i * 4 + 1] = 99;
      out.data[i * 4 + 2] = 235;
      out.data[i * 4 + 3] = alpha;
    }
  }
  ctx.putImageData(out, 0, 0);
  return { canvas, free, w, h };
}

/* --------------------------------------------------------- PRPT-KS rollouts */

/** Progress potential: completion along the reference, with a deviation dead zone. */
function potential(completion, deviation) {
  return 0.9 * (0.1 + 0.9 * clamp(completion)) * Math.exp(-(deviation < 0.8 ? 0 : deviation));
}

function dominates(a, b) {
  const mine = [a.delta, a.collision, a.stop];
  const theirs = [b.delta, b.collision, b.stop];
  return mine.every((v, i) => v >= theirs[i]) && mine.some((v, i) => v > theirs[i]);
}

/**
 * Rollouts of the training episodes with the decision rules of PRPT-KS.
 *
 * Every decision predicts eight waypoints and executes every second one. The
 * admissibility test rejects an action that leaves the planning map or loses
 * more progress potential than the tolerance; the planner then follows the
 * reference instead and the pair becomes a preference pair weighted by Pareto
 * dominance. Within the success radius the planner finishes the episode.
 * Policy errors shrink as training proceeds.
 */
class Rollouts {
  constructor(episodes, isFree, start, end) {
    const lengths = episodes.map((episode) => episode.line.length);
    const low = Math.min(...lengths), high = Math.max(...lengths);
    this.episodes = episodes.map((episode, i) => ({
      episode,
      index: i,
      ref: resample(episode.line.world, PRPT.spacing),
      norm: (lengths[i] - low) / (high - low || 1),
    }));
    this.isFree = isFree;
    this.start = start;
    this.end = end;
    this.reset();
  }

  reset() {
    this.rand = random(11);
    this.t = -Infinity;
    this.lanes = Array.from({ length: PRPT.lanes }, (_, k) => ({ id: k, next: this.start + k * 0.13, agent: null, last: null }));
    this.events = [];
    this.focus = null;
  }

  progress(t) {
    return clamp((t - this.start) / (this.end - this.start));
  }

  /** Cosine decay of the likelihood scale over training. */
  sigma(q) {
    const [start, end] = PRPT.sigma;
    return end + 0.5 * (start - end) * (1 + Math.cos(Math.PI * q));
  }

  advance(t) {
    if (t < this.t) this.reset();
    this.t = t;
    for (const lane of this.lanes) {
      while (lane.next <= t) this.step(lane, lane.next);
    }
    this.events = this.events.filter((event) => t - event.time < 1.2);
  }

  step(lane, time) {
    const agent = lane.agent;
    if (!agent || agent.done) {
      lane.last = agent;
      lane.agent = this.spawn(time, lane.id);
      lane.next = time + 0.28;
      return;
    }
    this.decide(agent, time);
    lane.next = time + PRPT.decision + (agent.done ? 0.35 : 0);
  }

  spawn(time, lane) {
    // Length curriculum: weights exp(strength * (2 progress - 1) * normalized length).
    const q = this.progress(time);
    const weights = this.episodes.map((e) => Math.exp(PRPT.curriculum * (2 * q - 1) * e.norm));
    let r = this.rand() * weights.reduce((a, b) => a + b, 0);
    let i = 0;
    while (i < weights.length - 1 && r > weights[i]) r -= weights[i++];
    const e = this.episodes[i];
    const [x, z] = e.ref[0];
    return { e, lane, born: time, x, z, bias: 0, waypoints: 0, takeover: false, trail: [[x, z, 0]], decisions: [], done: false, endTime: Infinity, success: false };
  }

  nearest(p, ref) {
    let best = 0, bestDistance = Infinity;
    for (let i = 0; i < ref.length; i++) {
      const d = distance(p, ref[i]);
      if (d < bestDistance) {
        best = i;
        bestDistance = d;
      }
    }
    return { index: best, completion: ref.length > 1 ? best / (ref.length - 1) : 1, deviation: bestDistance };
  }

  /** Policy action: the reference continuation, turned, bent and stretched by errors that shrink with training. */
  policyAction(agent, n, q) {
    const r = this.rand;
    const ref = agent.e.ref, last = ref.length - 1;
    const p = [agent.x, agent.z];
    const skill = Math.pow(q, 0.8);
    agent.bias = agent.bias * 0.55 + gaussian(r) * lerp(0.36, 0.03, skill);
    const wrongWay = r() < 0.1 * (1 - skill) ** 2 ? (r() < 0.5 ? -1 : 1) * lerp(1.1, 2.2, r()) : 0;
    const turn = agent.bias + gaussian(r) * lerp(0.26, 0.03, skill) + wrongWay;
    const bend = gaussian(r) * lerp(0.45, 0.04, skill);
    const stretch = Math.max(0.3, 1 + gaussian(r) * lerp(0.22, 0.03, skill));
    if (n / last > 0.45 && r() < 0.07 * (1 - skill) ** 1.5) return Array.from({ length: 8 }, () => [...p]);
    const offset = [p[0] - ref[n][0], p[1] - ref[n][1]];
    const points = [];
    for (let k = 1; k <= 8; k++) {
      const target = ref[Math.min(n + k, last)];
      const pull = (1 - k / 8) ** 2;
      const ix = target[0] + offset[0] * pull - p[0], iz = target[1] + offset[1] * pull - p[1];
      const angle = turn + (bend * k) / 8;
      const c = Math.cos(angle), s = Math.sin(angle);
      points.push([p[0] + stretch * (c * ix - s * iz), p[1] + stretch * (s * ix + c * iz)]);
    }
    return points;
  }

  /** Planner substitute: waypoints that rejoin the reference lookahead poses ahead of the nearest one. */
  planAlong(p, ref, n) {
    const last = ref.length - 1;
    const target = ref[Math.min(n + PRPT.lookahead, last)];
    return Array.from({ length: 8 }, (_, i) => {
      const f = (i + 1) / 8;
      const along = ref[Math.min(n + Math.round(PRPT.lookahead * f), last)];
      return [lerp(lerp(p[0], target[0], f), along[0], f), lerp(lerp(p[1], target[1], f), along[1], f)];
    });
  }

  /** Planner to the goal within the success radius: straight waypoints that come to rest on it. */
  planToGoal(p, goal) {
    const d = distance(p, goal);
    return Array.from({ length: 8 }, (_, i) => {
      const f = d > 0 ? Math.min(1, ((i + 1) * PRPT.spacing) / d) : 1;
      return [lerp(p[0], goal[0], f), lerp(p[1], goal[1], f)];
    });
  }

  /** The Pareto utility of an action from p: executed path, progress, safety and stop axes. */
  utility(agent, points, p, before) {
    const ref = agent.e.ref;
    const path = [p];
    let stopped = false;
    for (let i = 0; i < PRPT.executed.length; i++) {
      const waypoint = points[PRPT.executed[i]];
      const previous = path[path.length - 1];
      const still = Math.abs(waypoint[0] - previous[0]) < PRPT.stopDistance && Math.abs(waypoint[1] - previous[1]) < PRPT.stopDistance;
      if (still && agent.waypoints + i >= PRPT.minWaypoints) {
        stopped = true;
        break;
      }
      path.push(waypoint);
    }
    const end = path[path.length - 1];
    const after = this.nearest(end, ref);
    const valueAfter = potential(after.completion, after.deviation);
    const collision = path.slice(1).some(([x, z]) => !this.isFree(x, z)) ? -PRPT.obstaclePenalty : 0;
    const reached = stopped && distance(end, ref[ref.length - 1]) < PRPT.successRadius;
    return { path, stopped, valueAfter, delta: PRPT.gamma * valueAfter - before, collision, stop: reached ? 1 : 0 };
  }

  decide(agent, time) {
    const q = this.progress(time);
    const ref = agent.e.ref, goal = ref[ref.length - 1];
    const p = [agent.x, agent.z];
    const near = this.nearest(p, ref);
    const before = potential(near.completion, near.deviation);
    agent.takeover = agent.takeover || distance(p, goal) < PRPT.successRadius;
    const policy = this.policyAction(agent, near.index, q);
    const policyU = this.utility(agent, policy, p, before);
    let kind = "policy", reason = null, plan = null, planU = null, weight = 0;
    if (agent.takeover) {
      kind = "takeover";
      plan = this.planToGoal(p, goal);
    } else {
      // Safety and progress are tested separately, so neither offsets the other.
      if (policyU.collision < 0) reason = "collision";
      else if (policyU.delta < -PRPT.tolerance) reason = "progress";
      if (reason) {
        kind = "reject";
        plan = this.planAlong(p, ref, near.index);
      }
    }
    if (plan) {
      planU = this.utility(agent, plan, p, before);
      weight = dominates(planU, policyU) ? (planU.stop > 0 ? 1.1 : 1.0) : 0.9;
    }
    const used = planU || policyU;
    const decision = { time, kind, reason, policy, policyU, plan, planU, weight, sigma: this.sigma(q), trailIndex: agent.trail.length, path: used.path };
    for (let i = 1; i < used.path.length; i++) agent.trail.push([...used.path[i], plan ? 1 : 0]);
    agent.waypoints += used.path.length - 1;
    [agent.x, agent.z] = used.path[used.path.length - 1];
    if (kind === "reject") {
      const bad = policyU.path.slice(1).find(([x, z]) => !this.isFree(x, z)) || policy[7];
      this.events.push({ kind: "pair", time: time + 0.08, x: bad[0], z: bad[1] });
    }
    if (used.stopped || agent.waypoints >= PRPT.budget) {
      agent.done = true;
      agent.success = used.stop > 0;
      agent.endTime = time + PRPT.decision;
      this.events.push({ kind: agent.success ? "success" : "fail", time: agent.endTime, x: agent.x, z: agent.z, type: agent.e.episode.type });
    }
    agent.decisions.push(decision);
    if (agent.decisions.length > 2) agent.decisions.shift();
    if (agent.lane === 0) this.focus = { agent, decision };
  }

  /** Position of an agent at time t, inside its current decision. */
  head(agent, t) {
    const d = agent.decisions[agent.decisions.length - 1];
    if (!d) return [agent.trail[0][0], agent.trail[0][1], 0];
    const u = clamp((t - d.time) / PRPT.decision);
    const eased = u - (Math.sin(2 * Math.PI * u) / (2 * Math.PI)) * 0.6;
    const segments = d.path.length - 1;
    if (segments === 0) return [d.path[0][0], d.path[0][1], 1];
    const x = eased * segments;
    const i = Math.min(Math.floor(x), segments - 1);
    const f = x - i;
    return [lerp(d.path[i][0], d.path[i + 1][0], f), lerp(d.path[i][1], d.path[i + 1][1], f), u];
  }
}

/* --------------------------------------------------------------- The hero */

class Hero {
  constructor(root, manifest, assets) {
    this.root = root;
    this.manifest = manifest;
    this.assets = assets;
    this.stage = root.querySelector(".hero-stage");
    this.glCanvas = root.querySelector(".hero-gl");
    this.fxCanvas = root.querySelector(".hero-fx");
    this.ctx = this.fxCanvas.getContext("2d");
    this.phone = root.querySelector(".hero-phone");
    this.feedLayers = [...root.querySelectorAll(".hero-feed")];
    this.caption = root.querySelector(".hero-caption");
    this.teaser = root.querySelector(".hero-teaser");
    this.teaserImage = this.teaser.querySelector("img");
    this.time = 0;
    this.playing = true;
    this.visible = true;
    this.lastFrame = performance.now();

    const project = makeProjector(manifest);
    this.project = project;
    this.episodes = manifest.episodes.map((episode) => {
      const world = [];
      for (let i = 0; i < episode.path.length; i += 2) world.push([episode.path[i], episode.path[i + 1]]);
      return { ...episode, line: new Polyline(world, project) };
    });
    // Reference pixels per meter on the floor at the middle of the scene.
    const middle = this.episodes[0].line.world[0];
    this.pxPerMeter = distance(project(middle[0], 0, middle[1]), project(middle[0] + 1, 0, middle[1]));

    // Scan: the camera path and the photos, timed by the capture.
    const scan = manifest.scan;
    const rawPath = Float32Array.from(scan.path, (v, i) => (i % 3 === 2 ? v / 100 : v));
    this.scanSamples = rawPath.length / 3;
    // Smoothing in seconds of animation, so the phone glides along the route instead of every step,
    // and the camera keeps a steady view however fast the capture replays.
    const perSecond = this.scanSamples / (SCAN.end - SCAN.start);
    this.scanPath = steadyPath(rawPath, 0.05 * perSecond);
    this.phonePath = phoneTrack(this.scanPath, 0.4 * perSecond, 0.6 * perSecond);
    this.phonePerSecond = perSecond;
    const span = SCAN.end - SCAN.start;
    const [tw, th, columns] = manifest.photoTile;
    this.photoRows = Math.ceil(manifest.photos.length / columns);
    this.photos = manifest.photos.map(([view, u, v, angle], i) => ({
      c: [u, v],
      angle,
      sx: (i % columns) * tw,
      sy: Math.floor(i / columns) * th,
      emit: SCAN.start + (view / (scan.views - 1)) * span,
      index: i,
    }));

    const box = manifest.contentBox;
    this.center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
    this.maxRadius = Math.hypot(box[2] - box[0], box[3] - box[1]) * 0.56;
    this.sprites = {
      good: glowSprite(rgba(GOOD, 0.55)),
      scan: glowSprite(rgba(ACCENT, 0.6)),
      ...Object.fromEntries(Object.entries(TYPE).map(([k, v]) => [k, glowSprite(rgba(v.color, 0.5))])),
    };
    try {
      this.renderer = new SceneRenderer(this.glCanvas, manifest);
      if (assets.points) this.renderer.setPoints(assets.points);
    } catch (error) {
      console.warn("Hero animation falls back to a 2D canvas:", error);
      this.renderer = null;
    }
    this.buildChapters();
    this.buildPanels();
    this.resize();
    new ResizeObserver(() => this.resize()).observe(this.stage);
    new IntersectionObserver(([entry]) => (this.visible = entry.isIntersecting), { threshold: 0.05 }).observe(this.stage);
    document.addEventListener("visibilitychange", () => (this.lastFrame = performance.now()));
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
      this.playing = false;
      this.time = TIMELINE.instruct + 8.5;
    }
    this.updatePlayButton();
    requestAnimationFrame((now) => this.frame(now));
  }

  attachScene(image, splats, map) {
    if (this.renderer) {
      this.renderer.setImage(image);
      this.renderer.setSplats(splats);
    } else {
      this.fallbackImage = image;
    }
    this.map = mapOverlay(map);
    this.sceneReady = true;
    const scale = this.manifest.mapScale;
    this.isFree = (x, z) => {
      const [u, v] = this.project(x, 0, z);
      const i = Math.floor(v * scale) * this.map.w + Math.floor(u * scale);
      return this.map.free[i] === 1;
    };
    this.rollouts = new Rollouts(this.episodes, this.isFree, TIMELINE.train + 0.9, TIMELINE.outro);
    this.teaserImage.src = this.teaserImage.dataset.src;
  }

  /* ------------------------------------------------------------ Layout */

  resize() {
    const rect = this.stage.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.cssWidth = rect.width;
    this.cssHeight = rect.height;
    this.width = Math.max(1, Math.round(rect.width * dpr));
    this.height = Math.max(1, Math.round(rect.height * dpr));
    this.dpr = dpr;
    for (const canvas of [this.glCanvas, this.fxCanvas]) {
      canvas.width = this.width;
      canvas.height = this.height;
    }
    this.portrait = rect.height > rect.width * 1.05;
    this.phoneSize = [this.phone.offsetWidth, this.phone.offsetHeight];
    this.drawn = null;
  }

  view(zoom, focus) {
    const box = this.manifest.contentBox;
    const bw = box[2] - box[0], bh = box[3] - box[1];
    // In portrait the scene turns a quarter.
    const w = this.width, h = this.height;
    const fit = this.portrait ? Math.min(w / bh, h / bw) : Math.min(w / bw, h / bh);
    const s = fit * zoom * 0.97;
    const [a, b, c, d] = this.portrait ? [0, s, -s, 0] : [s, 0, 0, s];
    const cx = w / 2, cy = h / 2;
    return { a, b, c, d, e: cx - (a * focus[0] + c * focus[1]), f: cy - (b * focus[0] + d * focus[1]), s };
  }

  toDevice(view, x, y) {
    return [view.a * x + view.c * y + view.e, view.b * x + view.d * y + view.f];
  }

  toReference(view, x, y) {
    const det = view.a * view.d - view.b * view.c;
    const dx = x - view.e, dy = y - view.f;
    return [(view.d * dx - view.c * dy) / det, (-view.b * dx + view.a * dy) / det];
  }

  /* -------------------------------------------------------------- DOM */

  buildChapters() {
    const nav = this.root.querySelector(".hero-chapters");
    this.chapterButtons = CHAPTERS.map((chapter, i) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "hero-chapter";
      button.innerHTML = `<span>${["Scan", "Reconstruct", "Scan2Instruction", "PRPT-KS"][i]}</span><i><b></b></i>`;
      button.addEventListener("click", () => {
        this.seek(chapter.start + (i === 0 ? 0 : 0.01));
        this.playing = true;
        this.updatePlayButton();
      });
      nav.appendChild(button);
      return button;
    });
    this.playButton = this.root.querySelector(".hero-play");
    this.playButton.addEventListener("click", () => {
      if (this.time >= TIMELINE.end) {
        this.seek(0);
        this.playing = true;
      } else {
        this.playing = !this.playing;
      }
      this.lastFrame = performance.now();
      this.updatePlayButton();
    });
    this.root.querySelector(".hero-replay").addEventListener("click", () => {
      this.seek(0);
      this.playing = true;
      this.updatePlayButton();
    });
    this.chapterIndex = -1;
  }

  updatePlayButton() {
    const label = this.playing ? "Pause animation" : this.time >= TIMELINE.end ? "Replay animation" : "Play animation";
    this.playButton.setAttribute("aria-label", label);
    this.playButton.classList.toggle("paused", !this.playing);
  }

  buildPanels() {
    const find = (selector) => this.root.querySelector(selector);
    this.instructionPanel = find(".hp-instruction");
    this.trainingPanel = find(".hp-training");
    this.legendCounts = Object.fromEntries(Object.keys(TYPE).map((key) => [key, find(`[data-count="${key}"]`)]));
    this.goalImage = find(".hp-goal");
    this.ui = {
      time: find("[data-time]"),
      views: find("[data-views]"),
      flash: find(".hero-flash"),
      progress: find(".hero-progress b"),
      episodes: find("[data-episodes]"),
      trainProgress: find("[data-progress]"),
      laneTag: find("[data-lane-tag]"),
      laneText: find("[data-lane-text]"),
      verdict: find("[data-verdict]"),
      bars: Object.fromEntries(["prog", "safe", "stop"].map((k) => [k, find(`[data-bar="${k}"]`)])),
      values: Object.fromEntries(["prog", "safe", "stop"].map((k) => [k, find(`[data-value="${k}"]`)])),
    };
    const columns = this.manifest.photoTile[2];
    for (const layer of this.feedLayers) layer.style.backgroundSize = `${columns * 100}% ${this.photoRows * 100}%`;
  }

  seek(time) {
    this.time = time;
    this.drawn = null;
    this.lastFrame = performance.now();
  }

  setChapter(index) {
    if (index === this.chapterIndex) return;
    this.chapterIndex = index;
    const chapter = CHAPTERS[index] || OVERVIEW;
    this.caption.classList.remove("show");
    clearTimeout(this.captionTimer);
    this.captionTimer = setTimeout(() => {
      this.caption.querySelector(".hc-title").textContent = chapter.title;
      this.caption.classList.add("show");
    }, 260);
  }

  /* ------------------------------------------------------------ Frame */

  frame(now) {
    requestAnimationFrame((next) => this.frame(next));
    const elapsed = Math.min((now - this.lastFrame) / 1000, 0.1);
    this.lastFrame = now;
    if (!this.visible || document.hidden) return;
    if (this.playing) {
      this.time += elapsed;
      // Hold the scan until the scene has arrived.
      if (!this.sceneReady && this.time > TIMELINE.reconstruct - 0.2) this.time = TIMELINE.reconstruct - 0.2;
      // Played through: rest on the overview figure.
      if (this.time >= TIMELINE.end) {
        this.time = TIMELINE.end;
        this.playing = false;
        this.updatePlayButton();
      }
    }
    if (!this.playing && this.drawn === this.time) return;
    this.drawn = this.time;
    this.draw(this.time, now / 1000);
  }

  draw(t, clock) {
    const chapter = t >= TIMELINE.train ? 3 : t >= TIMELINE.instruct ? 2 : t >= TIMELINE.reconstruct ? 1 : 0;
    this.setChapter(t >= TIMELINE.end - 0.6 ? CHAPTERS.length : chapter);
    this.chapterButtons.forEach((button, i) => {
      const start = CHAPTERS[i].start;
      const end = i < 3 ? CHAPTERS[i + 1].start : TIMELINE.end;
      button.classList.toggle("active", i === chapter);
      button.querySelector("b").style.transform = `scaleX(${ramp(start, end, t)})`;
    });

    // Camera: a slow drift while scanning, centered for planning, closer for training.
    const drift = Math.sin(t * 0.21) * 0.012;
    const zoom =
      1.0 + drift +
      0.04 * (1 - smooth(TIMELINE.reconstruct, TIMELINE.reconstruct + 3.4, t)) +
      0.07 * smooth(TIMELINE.train, TIMELINE.train + 3, t) * (1 - smooth(TIMELINE.outro, TIMELINE.end - 0.6, t));
    const view = this.view(zoom, [this.center[0] + Math.sin(t * 0.13) * 14, this.center[1] + Math.cos(t * 0.11) * 9]);

    const fade = 1 - smooth(TIMELINE.end - 1.6, TIMELINE.end - 0.4, t);
    const reveal = easeInOut(ramp(TIMELINE.reconstruct + 0.05, TIMELINE.reconstruct + 3.6, t));
    // The scene steps back while the geometric layer and the episodes are on screen, and a little in training.
    const planning = smooth(TIMELINE.instruct, TIMELINE.instruct + 1, t) * (1 - smooth(TIMELINE.train - 0.5, TIMELINE.train + 1.5, t));
    const training = smooth(TIMELINE.train, TIMELINE.train + 1.5, t) * (1 - smooth(TIMELINE.outro + 0.5, TIMELINE.end - 0.6, t));
    const state = {
      t,
      center: this.center,
      radius: this.sceneReady ? reveal * (this.maxRadius + 520) : 0,
      band: 420,
      time: clock,
      opacity: fade * lerp(1, 0.4, planning) * lerp(1, 0.72, training),
      saturation: lerp(1, 0.3, planning) * lerp(1, 0.6, training),
      ring: 1 - smooth(TIMELINE.reconstruct + 2.5, TIMELINE.reconstruct + 3.3, t),
      splats: t < TIMELINE.reconstruct + 3.6,
      points: smooth(SCAN.start - 0.2, SCAN.start + 0.4, t) * (t < TIMELINE.reconstruct + 3.6 ? 1 : 0),
      pointRadius: 0.42 * 10 * view.s,
    };
    if (this.renderer) this.renderer.render(view, this.width, this.height, state);

    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.setTransform(view.a, view.b, view.c, view.d, view.e, view.f);
    if (!this.renderer && this.fallbackImage && state.radius > 0) this.drawFallback(ctx, state, fade);
    this.drawGrid(ctx, view, t);
    if (this.sceneReady) {
      this.drawMap(ctx, t);
      this.drawPaths(ctx, view, t);
    }
    this.drawScan(ctx, view, t, clock);
    this.drawPhone(ctx, view, t);
    this.drawPhotos(ctx, view, t, state);
    if (this.rollouts) this.drawRollouts(ctx, view, t);
    this.updatePanels(t);
    // The overview figure takes over once everything has faded.
    const teaser = smooth(TIMELINE.end - 1.0, TIMELINE.end, t);
    this.teaser.style.opacity = String(teaser);
    this.teaser.style.visibility = teaser > 0 ? "visible" : "hidden";
    this.teaser.style.setProperty("--zoom", String(lerp(1.04, 1, easeOut(teaser))));
    this.root.classList.toggle("finished", t >= TIMELINE.end);
  }

  drawFallback(ctx, state, fade) {
    ctx.save();
    ctx.globalAlpha = fade;
    ctx.beginPath();
    ctx.arc(this.center[0], this.center[1], Math.max(0, state.radius - state.band * 0.5), 0, Math.PI * 2);
    ctx.clip();
    ctx.drawImage(this.fallbackImage, 0, 0, this.manifest.width, this.manifest.height);
    ctx.restore();
  }

  /** A faint survey grid under the scan. */
  drawGrid(ctx, view, t) {
    const alpha = 0.42 * smooth(0, 1.2, t) * (1 - smooth(TIMELINE.reconstruct + 1.0, TIMELINE.reconstruct + 3.0, t));
    if (alpha <= 0) return;
    const box = this.manifest.contentBox;
    const step = 64;
    ctx.save();
    ctx.fillStyle = rgba(ACCENT, 0.45);
    const r = (1.2 * this.dpr) / view.s;
    for (let x = box[0]; x <= box[2]; x += step) {
      for (let y = box[1]; y <= box[3]; y += step) {
        const d = Math.hypot(x - this.center[0], y - this.center[1]) / this.maxRadius;
        ctx.globalAlpha = alpha * (1 - d * 0.6);
        ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
      }
    }
    ctx.restore();
  }

  drawMap(ctx, t) {
    const alpha =
      smooth(TIMELINE.instruct + 0.2, TIMELINE.instruct + 1.6, t) * (1 - 0.72 * smooth(TIMELINE.train, TIMELINE.train + 1.5, t)) *
      (1 - smooth(TIMELINE.outro, TIMELINE.end - 0.5, t));
    if (alpha <= 0) return;
    ctx.save();
    ctx.globalAlpha = alpha * 0.9;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(this.map.canvas, 0, 0, this.manifest.width, this.manifest.height);
    ctx.restore();
  }

  /* ---------------------------------------------------------- Scan */

  /** Fraction of the capture replayed at time t. */
  scanFraction(t) {
    return ramp(SCAN.start, SCAN.end, t);
  }

  /** Camera position and view angle on the scan path at a capture fraction. */
  scanPose(f) {
    const path = this.scanPath;
    const x = f * (this.scanSamples - 1);
    const i = Math.min(Math.floor(x), this.scanSamples - 2);
    const k = x - i;
    let da = path[i * 3 + 5] - path[i * 3 + 2];
    da -= Math.round(da / (2 * Math.PI)) * 2 * Math.PI;
    return [lerp(path[i * 3], path[i * 3 + 3], k), lerp(path[i * 3 + 1], path[i * 3 + 4], k), path[i * 3 + 2] + da * k];
  }

  /** The camera walks the real capture path: a fading trail, the view frustum and LiDAR pulses. */
  drawScan(ctx, view, t, clock) {
    const alpha = smooth(SCAN.start - 0.4, SCAN.start + 0.3, t) * (1 - smooth(TIMELINE.reconstruct + 0.2, TIMELINE.reconstruct + 1.8, t));
    if (alpha <= 0) return;
    const f = this.scanFraction(t);
    const path = this.scanPath;
    const head = f * (this.scanSamples - 1);
    const last = Math.floor(head);
    const [hx, hy, ha] = this.scanPose(f);
    const px = this.dpr / view.s;
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    // The whole walk so far, then a brighter comet over the last stretch.
    ctx.strokeStyle = ACCENT;
    ctx.globalAlpha = alpha * 0.34;
    ctx.lineWidth = 1.2 * px;
    ctx.beginPath();
    ctx.moveTo(path[0], path[1]);
    for (let i = 1; i <= last; i++) ctx.lineTo(path[i * 3], path[i * 3 + 1]);
    ctx.lineTo(hx, hy);
    ctx.stroke();
    const comet = 260;
    const chunks = 13;
    for (let c = 0; c < chunks; c++) {
      const from = Math.max(0, Math.floor(head - comet + (c * comet) / chunks));
      const to = Math.min(last, Math.floor(head - comet + ((c + 1) * comet) / chunks));
      if (to <= from) continue;
      const w = (c + 1) / chunks;
      ctx.globalAlpha = alpha * w * 0.9;
      ctx.lineWidth = (1.2 + 2.4 * w) * px;
      ctx.beginPath();
      ctx.moveTo(path[from * 3], path[from * 3 + 1]);
      for (let i = from + 1; i <= to; i++) ctx.lineTo(path[i * 3], path[i * 3 + 1]);
      if (c === chunks - 1) ctx.lineTo(hx, hy);
      ctx.stroke();
    }
    // The camera: frustum, LiDAR pulses and a bright head while the capture runs.
    const live = alpha * (1 - smooth(SCAN.end, SCAN.end + 0.6, t));
    if (live > 0) {
      const reach = 1.25 * this.pxPerMeter;
      const gradient = ctx.createRadialGradient(hx, hy, 0, hx, hy, reach);
      gradient.addColorStop(0, rgba(ACCENT, 0.34));
      gradient.addColorStop(1, rgba(ACCENT, 0));
      ctx.globalAlpha = live;
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.moveTo(hx, hy);
      ctx.arc(hx, hy, reach, ha - CAMERA_HALF_FOV, ha + CAMERA_HALF_FOV);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = rgba(ACCENT, 0.55);
      ctx.lineWidth = 1 * px;
      ctx.beginPath();
      ctx.moveTo(hx + Math.cos(ha - CAMERA_HALF_FOV) * reach * 0.8, hy + Math.sin(ha - CAMERA_HALF_FOV) * reach * 0.8);
      ctx.lineTo(hx, hy);
      ctx.lineTo(hx + Math.cos(ha + CAMERA_HALF_FOV) * reach * 0.8, hy + Math.sin(ha + CAMERA_HALF_FOV) * reach * 0.8);
      ctx.stroke();
      for (let k = 0; k < 2; k++) {
        const pulse = (clock * 1.1 + k * 0.5) % 1;
        ctx.globalAlpha = live * (1 - pulse) * 0.5;
        ctx.lineWidth = 1.2 * px;
        ctx.beginPath();
        ctx.arc(hx, hy, (0.15 + pulse * 1.4) * this.pxPerMeter, 0, Math.PI * 2);
        ctx.stroke();
      }
      const size = 40 * px;
      ctx.globalAlpha = live;
      ctx.drawImage(this.sprites.scan, hx - size / 2, hy - size / 2, size, size);
      ctx.fillStyle = "#ffffff";
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = 2 * px;
      ctx.beginPath();
      ctx.arc(hx, hy, 4.5 * px, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }

  /** Captured frames fly from the phone to the poses they were taken from, then dissolve into the scene. */
  drawPhotos(ctx, view, t, state) {
    if (t > TIMELINE.reconstruct + 4) return;
    const atlas = this.assets.photos;
    const [tw, th] = this.manifest.photoTile;
    const px = this.dpr / view.s;
    const origin = this.phoneCenter ? this.toReference(view, ...this.phoneCenter) : this.center;
    const cardW = 96, cardH = 72;
    const reach = 26;
    ctx.save();
    for (const photo of this.photos) {
      const age = t - photo.emit;
      if (age <= 0) continue;
      const fly = easeOut(age / 0.75);
      // Distance of the pose from the wavefront: dissolve once the scene grows past it.
      const d = Math.hypot(photo.c[0] - this.center[0], photo.c[1] - this.center[1]);
      const dissolve = smooth(0, 1, clamp((state.radius - d - 120) / 260));
      const alpha = smooth(0, 0.12, age) * (1 - dissolve);
      if (alpha <= 0.002) continue;
      const dir = [Math.cos(photo.angle), Math.sin(photo.angle)];
      const target = [photo.c[0] + dir[0] * (reach + cardH / 2), photo.c[1] + dir[1] * (reach + cardH / 2)];
      const bend = [(origin[0] + target[0]) / 2, Math.min(origin[1], target[1]) - 140];
      const u = fly;
      const x = (1 - u) * (1 - u) * origin[0] + 2 * (1 - u) * u * bend[0] + u * u * target[0];
      const y = (1 - u) * (1 - u) * origin[1] + 2 * (1 - u) * u * bend[1] + u * u * target[1];
      const scale = lerp(0.45, 1, fly) * (1 - 0.5 * dissolve);
      const upright = view.b !== 0 ? -Math.PI / 2 : 0;
      const angle = lerp(upright, photo.angle + Math.PI / 2, fly);

      // Frustum wedge from the camera position to the near edge of the card.
      if (fly > 0.55) {
        const wedge = smooth(0.55, 1, fly) * alpha;
        const half = (cardW / 2) * scale;
        const nx = -dir[1], ny = dir[0];
        const near = [x - dir[0] * cardH * 0.5 * scale, y - dir[1] * cardH * 0.5 * scale];
        ctx.globalAlpha = wedge * 0.85;
        ctx.strokeStyle = ACCENT;
        ctx.lineWidth = 1.1 * px;
        ctx.beginPath();
        ctx.moveTo(near[0] + nx * half, near[1] + ny * half);
        ctx.lineTo(photo.c[0], photo.c[1]);
        ctx.lineTo(near[0] - nx * half, near[1] - ny * half);
        ctx.stroke();
        ctx.fillStyle = ACCENT;
        ctx.beginPath();
        ctx.arc(photo.c[0], photo.c[1], 2.6 * px, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(angle);
      ctx.scale(scale, scale);
      const border = 3;
      ctx.globalAlpha = alpha * 0.16;
      ctx.fillStyle = "#1e2d50";
      ctx.fillRect(-cardW / 2 - border - 1, -cardH / 2 - border + 3, cardW + 2 * border + 2, cardH + 2 * border + 2);
      ctx.globalAlpha = alpha;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(-cardW / 2 - border, -cardH / 2 - border, cardW + 2 * border, cardH + 2 * border);
      ctx.drawImage(atlas, photo.sx, photo.sy, tw, th, -cardW / 2, -cardH / 2, cardW, cardH);
      const shine = 1 - smooth(0, 0.35, age);
      if (dissolve > 0 || shine > 0) {
        ctx.globalAlpha = alpha * Math.max(dissolve, shine * 0.6);
        ctx.fillStyle = dissolve > 0 ? ACCENT : "#ffffff";
        ctx.fillRect(-cardW / 2, -cardH / 2, cardW, cardH);
      }
      ctx.restore();
    }
    ctx.restore();
  }

  /* -------------------------------------------------- Scan2Instruction */

  drawPaths(ctx, view, t) {
    const local = t - (TIMELINE.instruct + 0.5);
    const persist = 1 - smooth(TIMELINE.train, TIMELINE.train + 1.2, t);
    if (local <= 0 || persist <= 0) return;
    const showcase = this.manifest.showcase;
    const slot = 1.3;
    const showStart = 1.4;
    const active = Math.floor((local - showStart) / slot);
    const highlight = local > showStart && active < showcase.length ? showcase[active] : -1;
    const px = this.dpr / view.s;
    const count = this.episodes.length;
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    let drawn = 0;
    this.episodes.forEach((episode, i) => {
      const start = (i / count) * 3.4;
      const progress = easeOut(ramp(start, start + 1.0, local));
      if (progress <= 0) return;
      if (progress >= 1) drawn++;
      const color = TYPE[episode.type].color;
      const settled = 0.3 + 0.7 * (1 - smooth(start + 1.0, start + 1.8, local));
      const dim = highlight >= 0 && highlight !== i ? 0.45 : 1;
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.14 * settled * persist * dim;
      ctx.lineWidth = 6 * px;
      ctx.beginPath();
      const head = episode.line.trace(ctx, progress);
      ctx.stroke();
      ctx.globalAlpha = 0.8 * settled * persist * dim;
      ctx.lineWidth = 1.3 * px;
      ctx.stroke();
      if (progress < 1) {
        const size = 22 * px;
        ctx.globalAlpha = persist;
        ctx.drawImage(this.sprites[episode.type], head[0] - size / 2, head[1] - size / 2, size, size);
      }
    });
    this.episodesDrawn = drawn;
    if (highlight >= 0) {
      const episode = this.episodes[highlight];
      const color = TYPE[episode.type].color;
      const age = local - showStart - active * slot;
      ctx.globalAlpha = smooth(0, 0.25, age) * (1 - smooth(slot - 0.2, slot, age)) * persist;
      ctx.strokeStyle = color;
      ctx.lineWidth = 12 * px;
      ctx.globalAlpha *= 0.35;
      ctx.beginPath();
      episode.line.trace(ctx, 1);
      ctx.stroke();
      ctx.globalAlpha /= 0.35;
      ctx.lineWidth = 3.5 * px;
      ctx.stroke();
      ctx.lineWidth = 1.4 * px;
      ctx.setLineDash([10 * px, 9 * px]);
      ctx.lineDashOffset = -age * 60 * px;
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();
      ctx.setLineDash([]);
      const end = episode.line.screen[episode.line.screen.length - 1];
      const begin = episode.line.screen[0];
      const ring = (age * 1.4) % 1;
      ctx.lineWidth = 2 * px;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.arc(end[0], end[1], (10 + ring * 34) * px, 0, Math.PI * 2);
      ctx.globalAlpha *= 1 - ring;
      ctx.stroke();
      ctx.globalAlpha = persist;
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(begin[0], begin[1], 4.5 * px, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 2 * px;
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(end[0], end[1], 6 * px, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
    this.highlight = { index: active, age: local - showStart - active * slot, on: highlight >= 0 };
  }

  /* ----------------------------------------------------------- PRPT-KS */

  drawRollouts(ctx, view, t) {
    const sim = this.rollouts;
    const alpha = smooth(sim.start - 0.5, sim.start + 0.5, t) * (1 - smooth(TIMELINE.outro + 0.5, TIMELINE.end - 0.8, t));
    if (alpha <= 0) return;
    sim.advance(Math.min(t, TIMELINE.outro + 1.2));
    const px = this.dpr / view.s;
    const P = (x, z) => this.project(x, 0.05, z);
    const halo = sim.sigma(sim.progress(t));
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const lane of sim.lanes) {
      for (const agent of [lane.last, lane.agent]) {
        if (!agent) continue;
        const life = smooth(agent.born, agent.born + 0.25, t) * (agent.done ? 1 - ramp(agent.endTime, agent.endTime + 0.5, t) : 1);
        if (life <= 0) continue;
        const a = alpha * life;
        const color = TYPE[agent.e.episode.type].color;
        const focus = agent.lane === 0 ? 1 : 0.8;
        const line = agent.e.episode.line;
        // Reference path and the success radius around its goal.
        ctx.globalAlpha = a * 0.32;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2 * px;
        ctx.setLineDash([4 * px, 6 * px]);
        ctx.beginPath();
        line.trace(ctx, 1);
        ctx.stroke();
        const goal = line.screen[line.screen.length - 1];
        ctx.globalAlpha = a * (agent.takeover ? 0.75 : 0.4);
        ctx.strokeStyle = agent.takeover ? GOOD : color;
        ctx.beginPath();
        ctx.arc(goal[0], goal[1], PRPT.successRadius * this.pxPerMeter, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        // Executed trail: policy steps in the episode color, planner steps in green.
        const [hx, hz] = this.rollouts.head(agent, t);
        const d = agent.decisions[agent.decisions.length - 1];
        const upto = d ? d.trailIndex : agent.trail.length;
        const points = agent.trail.slice(0, upto).map(([x, z, planner]) => [...P(x, z), planner]);
        points.push([...P(hx, hz), d && d.plan ? 1 : 0]);
        const trace = () => {
          ctx.beginPath();
          points.forEach(([u, v], i) => (i ? ctx.lineTo(u, v) : ctx.moveTo(u, v)));
        };
        ctx.strokeStyle = "#ffffff";
        ctx.globalAlpha = a * 0.85 * focus;
        ctx.lineWidth = 5 * px;
        trace();
        ctx.stroke();
        for (let i = 1; i < points.length; i++) {
          ctx.strokeStyle = points[i][2] ? GOOD : color;
          ctx.globalAlpha = a * 0.95 * focus;
          ctx.lineWidth = 2.4 * px;
          ctx.beginPath();
          ctx.moveTo(points[i - 1][0], points[i - 1][1]);
          ctx.lineTo(points[i][0], points[i][1]);
          ctx.stroke();
        }

        // The current decision: eight predicted waypoints, executed ones with their likelihood scale.
        if (d && !agent.done) this.drawDecision(ctx, d, t, a, color, px, halo, P);
        const head = P(hx, hz);
        const size = 30 * px;
        const planner = d && d.plan && t - d.time > 0.06;
        ctx.globalAlpha = a;
        ctx.drawImage(planner ? this.sprites.good : this.sprites[agent.e.episode.type], head[0] - size / 2, head[1] - size / 2, size, size);
        ctx.fillStyle = "#ffffff";
        ctx.strokeStyle = planner ? GOOD : color;
        ctx.lineWidth = 1.8 * px;
        ctx.beginPath();
        ctx.arc(head[0], head[1], 4 * px, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }
    // Events: preference pairs rising from rejected actions, and episode ends.
    for (const event of sim.events) {
      const age = t - event.time;
      if (age < 0 || age > 0.9) continue;
      const k = age / 0.9;
      const at = P(event.x, event.z);
      ctx.globalAlpha = alpha * (1 - k);
      if (event.kind === "pair") {
        const y = at[1] - easeOut(k) * 26 * px;
        ctx.strokeStyle = "rgba(24,34,58,0.35)";
        ctx.lineWidth = 1 * px;
        ctx.beginPath();
        ctx.moveTo(at[0] - 5 * px, y);
        ctx.lineTo(at[0] + 5 * px, y);
        ctx.stroke();
        ctx.fillStyle = BAD;
        ctx.beginPath();
        ctx.arc(at[0] - 5 * px, y, 3 * px, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = GOOD;
        ctx.beginPath();
        ctx.arc(at[0] + 5 * px, y, 3 * px, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.strokeStyle = event.kind === "success" ? GOOD : BAD;
        ctx.lineWidth = 2.2 * px;
        for (const delay of [0, 0.18]) {
          const r = easeOut(clamp((age - delay) / 0.7));
          if (r <= 0) continue;
          ctx.globalAlpha = alpha * (1 - r);
          ctx.beginPath();
          ctx.arc(at[0], at[1], (6 + 44 * r) * px, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
    }
    ctx.restore();
  }

  drawDecision(ctx, d, t, alpha, color, px, sigma, P) {
    const tau = t - d.time;
    const D = PRPT.decision;
    const rejected = d.kind === "reject";
    const shown = smooth(0, 0.05, tau) * (1 - smooth(rejected ? 0.12 : D * 0.55, rejected ? 0.3 : D * 0.9, tau));
    const fan = (points, tint, strength) => {
      ctx.strokeStyle = tint;
      ctx.globalAlpha = alpha * strength * 0.5;
      ctx.lineWidth = 1 * px;
      ctx.beginPath();
      points.forEach(([x, z], k) => {
        const [u, v] = P(x, z);
        k ? ctx.lineTo(u, v) : ctx.moveTo(u, v);
      });
      ctx.stroke();
      points.forEach(([x, z], k) => {
        const [u, v] = P(x, z);
        const executed = k % 2 === 1;
        if (executed) {
          ctx.globalAlpha = alpha * strength * 0.16;
          ctx.fillStyle = tint;
          ctx.beginPath();
          ctx.arc(u, v, sigma * 0.09 * this.pxPerMeter, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.globalAlpha = alpha * strength;
        ctx.fillStyle = executed ? tint : "#ffffff";
        ctx.strokeStyle = tint;
        ctx.lineWidth = 1.2 * px;
        ctx.beginPath();
        ctx.arc(u, v, (executed ? 2.8 : 1.9) * px, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
    };
    if (shown > 0) fan(d.policy, rejected && tau > 0.06 ? BAD : color, shown);
    if (rejected && tau > 0.06 && tau < 0.32) {
      const bad = d.policyU.path.slice(1).find(([x, z]) => !this.isFree(x, z)) || d.policy[7];
      const [u, v] = P(bad[0], bad[1]);
      const r = 5 * px;
      ctx.globalAlpha = alpha * (1 - smooth(0.2, 0.32, tau));
      ctx.strokeStyle = BAD;
      ctx.lineWidth = 2 * px;
      ctx.beginPath();
      ctx.moveTo(u - r, v - r);
      ctx.lineTo(u + r, v + r);
      ctx.moveTo(u + r, v - r);
      ctx.lineTo(u - r, v + r);
      ctx.stroke();
    }
    if (d.plan) {
      const planned = smooth(0.06, 0.12, tau) * (1 - smooth(D * 0.6, D * 0.95, tau));
      if (planned > 0) fan(d.plan, GOOD, planned);
    }
  }

  /* ------------------------------------------------------------- Phone */

  /** Phone track at scan fraction f: smoothed position and velocity per sample, in the reference frame. */
  phonePose(f) {
    const path = this.phonePath;
    const x = clamp(f) * (this.scanSamples - 1);
    const i = Math.min(Math.floor(x), this.scanSamples - 2);
    const k = x - i;
    return [0, 1, 2, 3].map((j) => lerp(path[i * 4 + j], path[(i + 1) * 4 + j], k));
  }

  /** The phone travels the capture route trailing the camera, tethered to it, and hands over at the center. */
  drawPhone(ctx, view, t) {
    const enter = easeOut(ramp(0.05, 0.9, t));
    const leave = smooth(SCAN.end, TIMELINE.reconstruct + 0.45, t);
    const visible = t < TIMELINE.reconstruct + 0.5 && enter > 0;
    this.phone.style.visibility = visible ? "visible" : "hidden";
    if (!visible) {
      this.phoneCenter = null;
      return;
    }
    const f = this.scanFraction(t);
    const [u, v, du, dv] = this.phonePose(f);
    const head = this.toDevice(view, u, v).map((c) => c / this.dpr);
    // Screen velocity in CSS pixels per second; the phone trails it and floats above the route.
    const walking = smooth(SCAN.start - 0.3, SCAN.start + 0.4, t) * (1 - smooth(SCAN.end - 0.5, SCAN.end, t));
    const rate = (this.phonePerSecond * walking) / this.dpr;
    const vx = (view.a * du + view.c * dv) * rate, vy = (view.b * du + view.d * dv) * rate;
    const cruise = 520;
    let ox = (-0.8 * vx) / Math.max(cruise, Math.hypot(vx, vy)), oy = (-0.8 * vy) / Math.max(cruise, Math.hypot(vx, vy)) - 0.6;
    const norm = Math.hypot(ox, oy) || 1;
    ox /= norm;
    oy /= norm;
    const [pw, ph] = this.phoneSize;
    const scale = lerp(0.9, 1, enter) * (1 - leave * 0.65);
    const reach = (Math.hypot(ox * pw, oy * ph) / 2) * scale + 26;
    const mx = (pw * scale) / 2 + 10, my = (ph * scale) / 2 + 10;
    let x = softClamp(head[0] + ox * reach, mx, this.cssWidth - mx, 60);
    let y = softClamp(head[1] + oy * reach, my, this.cssHeight - my, 40) + (1 - enter) * 90;
    const center = this.toDevice(view, ...this.center).map((c) => c / this.dpr);
    x = lerp(x, center[0], easeInOut(leave));
    y = lerp(y, center[1], easeInOut(leave));
    // Turn and pitch with the walk, as a hand-held phone would.
    const sway = clamp(vx / 80, -14, 14);
    const tilt = 10 + clamp(-vy / 120, -6, 6) + Math.sin(t * 1.7) * 1.2;
    const roll = clamp(vx / 180, -5, 5) + Math.sin(t * 1.1) * 1.2;
    this.phone.style.transform =
      `translate(-50%, -50%) translate(${x}px, ${y}px) ` +
      `perspective(900px) rotateX(${tilt}deg) rotateY(${sway}deg) rotateZ(${roll}deg) scale(${scale})`;
    this.phone.style.opacity = String(enter * (1 - leave));
    this.phoneCenter = [x * this.dpr, y * this.dpr];

    // Tether from the camera to the phone; the phone hides its far end.
    const tether = enter * (1 - smooth(SCAN.end - 0.3, SCAN.end + 0.2, t));
    if (tether > 0) {
      const [hx, hy] = this.scanPose(f);
      const [px, py] = this.toReference(view, x * this.dpr, y * this.dpr);
      ctx.save();
      ctx.globalAlpha = tether * 0.55;
      ctx.strokeStyle = ACCENT;
      ctx.lineWidth = (1.2 * this.dpr) / view.s;
      ctx.setLineDash([(3 * this.dpr) / view.s, (4 * this.dpr) / view.s]);
      ctx.beginPath();
      ctx.moveTo(hx, hy);
      ctx.lineTo(px, py);
      ctx.stroke();
      ctx.restore();
    }

    // Screen: the photo just captured, cross-fading from the previous one.
    const [, , columns] = this.manifest.photoTile;
    let current = -1;
    for (const photo of this.photos) if (t >= photo.emit) current = photo.index;
    const position = (i) => {
      const col = i % columns, row = Math.floor(i / columns);
      return `${(col / (columns - 1)) * 100}% ${(row / (this.photoRows - 1)) * 100}%`;
    };
    const shownIndex = Math.max(0, current);
    const previous = Math.max(0, current - 1);
    const sinceCapture = current >= 0 ? t - this.photos[current].emit : 1;
    this.feedLayers[0].style.backgroundPosition = position(previous);
    this.feedLayers[1].style.backgroundPosition = position(shownIndex);
    this.feedLayers[1].style.opacity = String(current <= 0 ? 1 : smooth(0, 0.12, sinceCapture));
    const zoom = 1 + 0.05 * clamp(sinceCapture / 0.4);
    for (const layer of this.feedLayers) layer.style.transform = `scale(${zoom})`;
    this.ui.flash.style.opacity = String(current >= 0 ? Math.max(0, 1 - sinceCapture / 0.16) * 0.55 : 0);
    const seconds = Math.floor(f * this.manifest.scan.duration);
    this.ui.time.textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
    this.ui.views.textContent = Math.round(f * this.manifest.scan.views).toLocaleString("en-US");
    this.ui.progress.style.transform = `scaleX(${f})`;
  }

  /* -------------------------------------------------------------- HUD */

  updatePanels(t) {
    const instructing = t >= TIMELINE.instruct + 0.5 && t < TIMELINE.train - 0.3;
    const training = t >= TIMELINE.train + 0.6 && t < TIMELINE.end - 1.2;
    this.root.classList.toggle("show-instruction", instructing);
    this.root.classList.toggle("show-training", training);
    if (instructing) {
      const drawn = this.episodesDrawn || 0;
      const counts = { vln: 0, object: 0, need: 0 };
      const total = this.episodes.length;
      this.episodes.forEach((episode, i) => {
        if (i < drawn) counts[episode.type]++;
      });
      for (const key of Object.keys(TYPE)) this.legendCounts[key].textContent = String(counts[key]);
      this.ui.episodes.textContent = String(Math.min(drawn, total));
      const h = this.highlight;
      const card = this.instructionPanel.querySelector(".hp-card");
      if (h && h.on) {
        const episode = this.episodes[this.manifest.showcase[h.index]];
        if (card.dataset.index !== String(h.index)) {
          card.dataset.index = String(h.index);
          const tag = card.querySelector(".hp-tag");
          tag.textContent = TYPE[episode.type].label;
          tag.style.setProperty("--tag", TYPE[episode.type].color);
          const [gw, gh] = this.manifest.goalTile;
          this.goalImage.style.backgroundPosition = `0 ${(h.index * 100) / (this.manifest.showcase.length - 1)}%`;
          this.goalImage.style.aspectRatio = `${gw} / ${gh}`;
        }
        const chars = Math.floor(clamp(h.age / 0.7) * episode.text.length);
        card.querySelector(".hp-text").textContent = episode.text.slice(0, chars);
        card.classList.add("on");
      } else {
        card.classList.remove("on");
      }
    }
    if (training && this.rollouts) {
      const sim = this.rollouts;
      const q = sim.progress(Math.min(t, TIMELINE.outro));
      const ui = this.ui;
      ui.trainProgress.style.transform = `scaleX(${q})`;
      const focus = sim.focus;
      if (focus) {
        const { agent, decision: d } = focus;
        const episode = agent.e.episode;
        if (ui.laneText.dataset.id !== episode.id) {
          ui.laneText.dataset.id = episode.id;
          ui.laneText.textContent = episode.text;
          ui.laneTag.textContent = TYPE[episode.type].label;
          ui.laneTag.style.setProperty("--tag", TYPE[episode.type].color);
        }
        const u = d.policyU;
        const sign = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
        ui.values.prog.textContent = sign(u.delta);
        ui.values.safe.textContent = sign(u.collision);
        ui.values.stop.textContent = u.stop.toFixed(0);
        ui.bars.prog.style.transform = `scaleX(${clamp(0.5 + u.delta * 4)})`;
        ui.bars.safe.style.transform = `scaleX(${u.collision < 0 ? 0.08 : 1})`;
        ui.bars.stop.style.transform = `scaleX(${u.stop > 0 ? 1 : 0.04})`;
        this.trainingPanel.classList.toggle("bad-prog", u.delta < -PRPT.tolerance);
        this.trainingPanel.classList.toggle("bad-safe", u.collision < 0);
        const verdicts = {
          policy: ["Admissible: the policy acts", "ok"],
          reject: [d.reason === "collision" ? "Leaves the map: the planner substitutes" : "Loses progress: the planner substitutes", "bad"],
          takeover: ["Within 1 m: the planner stops at the goal", "goal"],
        };
        const [text, tone] = verdicts[d.kind];
        if (ui.verdict.textContent !== text) ui.verdict.textContent = text;
        ui.verdict.dataset.tone = tone;
      }
    }
  }
}

/* ---------------------------------------------------------------- Startup */

async function start() {
  const root = document.getElementById("hero");
  if (!root) return;
  try {
    const manifest = await (await fetch(new URL("hero.json", BASE))).json();
    const small = Math.min(screen.width, screen.height) < 820;
    const scene = Promise.all([
      loadImage(small ? "scene-sm.webp" : "scene.webp"),
      loadSplats("splats.bin"),
      loadImage("map.png"),
    ]);
    const [photos, points] = await Promise.all([loadImage("photos.webp"), loadPoints("scan.bin")]);
    loadImage("goals.webp");
    const hero = new Hero(root, manifest, { photos, points });
    root.classList.add("ready");
    const [image, splats, map] = await scene;
    hero.attachScene(image, splats, map);
  } catch (error) {
    console.error(error);
    root.classList.add("failed");
  }
}

start();
