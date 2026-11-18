// simcore: native kernels for Simulatics (C ABI, loaded from Python via ctypes).
//
//   sc_mlp_forward_batch  one different MLP per row, threaded with OpenMP
//   sc_inflate            obstacle inflation on an occupancy grid
//   sc_astar              8-connected A* with octile heuristic + soft costs
//   sc_smooth_path        line-of-sight shortcutting
//   sc_integrate_scan     log-odds occupancy mapping from lidar hits (Bresenham)
//
// Grids are row-major: cell (x, y) is grid[y * w + x].
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <queue>
#include <vector>

#if defined(_WIN32)
#define SC_API extern "C" __declspec(dllexport)
#else
#define SC_API extern "C" __attribute__((visibility("default")))
#endif

SC_API int sc_version() { return 1; }

// params: [B x n_params] float32, obs: [B x sizes[0]], out: [B x sizes[n_sizes-1]]
// Layout per network, per layer: W (out x in, row-major) then bias (out). tanh everywhere.
SC_API void sc_mlp_forward_batch(const float* params, int batch, int n_params, const int* sizes, int n_sizes,
                                 const float* obs, float* out) {
  int max_w = 0;
  for (int i = 0; i < n_sizes; ++i) max_w = std::max(max_w, sizes[i]);
  const int obs_dim = sizes[0], act_dim = sizes[n_sizes - 1];
#pragma omp parallel
  {
    std::vector<float> a(max_w), b(max_w);
#pragma omp for schedule(static)
    for (int e = 0; e < batch; ++e) {
      const float* p = params + (size_t)e * n_params;
      std::memcpy(a.data(), obs + (size_t)e * obs_dim, sizeof(float) * obs_dim);
      float* x = a.data();
      float* y = b.data();
      for (int l = 0; l + 1 < n_sizes; ++l) {
        const int in = sizes[l], on = sizes[l + 1];
        const float* w = p;
        const float* bias = p + (size_t)in * on;
        for (int o = 0; o < on; ++o) {
          const float* row = w + (size_t)o * in;
          float acc = bias[o];
          for (int i = 0; i < in; ++i) acc += row[i] * x[i];
          y[o] = std::tanh(acc);
        }
        p = bias + on;
        std::swap(x, y);
      }
      std::memcpy(out + (size_t)e * act_dim, x, sizeof(float) * act_dim);
    }
  }
}

// out[c] = 255 if occupied within `radius` cells, else a soft cost 0..200 that
// decays with distance up to 2*radius (keeps paths away from walls).
SC_API void sc_inflate(const uint8_t* grid, int w, int h, float radius, uint8_t* out) {
  const float far = 1e9f;
  std::vector<float> dist((size_t)w * h, far);
  // two-pass chamfer distance transform
  const float d1 = 1.0f, d2 = 1.41421356f;
  for (int i = 0; i < w * h; ++i) if (grid[i]) dist[i] = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float& d = dist[(size_t)y * w + x];
      if (x > 0) d = std::min(d, dist[(size_t)y * w + x - 1] + d1);
      if (y > 0) {
        d = std::min(d, dist[(size_t)(y - 1) * w + x] + d1);
        if (x > 0) d = std::min(d, dist[(size_t)(y - 1) * w + x - 1] + d2);
        if (x < w - 1) d = std::min(d, dist[(size_t)(y - 1) * w + x + 1] + d2);
      }
    }
  for (int y = h - 1; y >= 0; --y)
    for (int x = w - 1; x >= 0; --x) {
      float& d = dist[(size_t)y * w + x];
      if (x < w - 1) d = std::min(d, dist[(size_t)y * w + x + 1] + d1);
      if (y < h - 1) {
        d = std::min(d, dist[(size_t)(y + 1) * w + x] + d1);
        if (x > 0) d = std::min(d, dist[(size_t)(y + 1) * w + x - 1] + d2);
        if (x < w - 1) d = std::min(d, dist[(size_t)(y + 1) * w + x + 1] + d2);
      }
    }
  for (int i = 0; i < w * h; ++i) {
    const float d = dist[i];
    if (d <= radius) out[i] = 255;
    else if (d < 2 * radius) out[i] = (uint8_t)(200.0f * (2 * radius - d) / radius);
    else out[i] = 0;
  }
}

// cost: 255 = blocked, otherwise extra traversal cost (cost/50 per cell).
// Writes up to max_len (x, y) pairs from start to goal. Returns the number of
// points, 0 if no path, -1 if the buffer is too small.
SC_API int sc_astar(const uint8_t* cost, int w, int h, int sx, int sy, int gx, int gy, int* out_xy, int max_len) {
  auto inside = [&](int x, int y) { return x >= 0 && y >= 0 && x < w && y < h; };
  if (!inside(sx, sy) || !inside(gx, gy)) return 0;
  if (cost[(size_t)gy * w + gx] == 255) return 0;
  const int n = w * h;
  std::vector<float> g(n, 1e30f);
  std::vector<int> parent(n, -1);
  std::vector<uint8_t> closed(n, 0);
  typedef std::pair<float, int> QE;
  std::priority_queue<QE, std::vector<QE>, std::greater<QE>> open;
  auto heur = [&](int x, int y) {
    const float dx = (float)std::abs(x - gx), dy = (float)std::abs(y - gy);
    return (dx + dy) + (1.41421356f - 2.0f) * std::min(dx, dy);
  };
  const int start = sy * w + sx, goal = gy * w + gx;
  g[start] = 0;
  open.push(QE(heur(sx, sy), start));
  static const int DX[8] = {1, -1, 0, 0, 1, 1, -1, -1};
  static const int DY[8] = {0, 0, 1, -1, 1, -1, 1, -1};
  bool found = false;
  while (!open.empty()) {
    const int cur = open.top().second;
    open.pop();
    if (closed[cur]) continue;
    closed[cur] = 1;
    if (cur == goal) { found = true; break; }
    const int cx = cur % w, cy = cur / w;
    for (int k = 0; k < 8; ++k) {
      const int nx = cx + DX[k], ny = cy + DY[k];
      if (!inside(nx, ny)) continue;
      const int ni = ny * w + nx;
      if (closed[ni] || cost[ni] == 255) continue;
      if (k >= 4 && (cost[cy * w + nx] == 255 || cost[ny * w + cx] == 255)) continue;  // no corner cutting
      const float step = (k < 4 ? 1.0f : 1.41421356f) * (1.0f + cost[ni] / 50.0f);
      const float ng = g[cur] + step;
      if (ng < g[ni]) {
        g[ni] = ng;
        parent[ni] = cur;
        open.push(QE(ng + heur(nx, ny), ni));
      }
    }
  }
  if (!found) return 0;
  std::vector<int> rev;
  for (int c = goal; c != -1; c = parent[c]) rev.push_back(c);
  if ((int)rev.size() > max_len) return -1;
  int k = 0;
  for (auto it = rev.rbegin(); it != rev.rend(); ++it, ++k) {
    out_xy[2 * k] = *it % w;
    out_xy[2 * k + 1] = *it / w;
  }
  return k;
}

static bool line_free(const uint8_t* cost, int w, int x0, int y0, int x1, int y1, uint8_t limit) {
  int dx = std::abs(x1 - x0), dy = -std::abs(y1 - y0);
  int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx + dy;
  for (;;) {
    if (cost[(size_t)y0 * w + x0] > limit) return false;
    if (x0 == x1 && y0 == y1) return true;
    const int e2 = 2 * err;
    if (e2 >= dy) { err += dy; x0 += sx; }
    if (e2 <= dx) { err += dx; y0 += sy; }
  }
}

// Greedy shortcutting: keep a point only when the straight line to the next
// candidate crosses a cell with cost > limit. In-place; returns new length.
SC_API int sc_smooth_path(const uint8_t* cost, int w, int h, int* xy, int n, int limit) {
  (void)h;
  if (n <= 2) return n;
  std::vector<int> keep;
  keep.push_back(0);
  int anchor = 0;
  while (anchor < n - 1) {
    int best = anchor + 1;
    for (int j = n - 1; j > anchor + 1; --j) {
      if (line_free(cost, w, xy[2 * anchor], xy[2 * anchor + 1], xy[2 * j], xy[2 * j + 1], (uint8_t)limit)) {
        best = j;
        break;
      }
    }
    keep.push_back(best);
    anchor = best;
  }
  for (size_t i = 0; i < keep.size(); ++i) {
    xy[2 * i] = xy[2 * keep[i]];
    xy[2 * i + 1] = xy[2 * keep[i] + 1];
  }
  return (int)keep.size();
}

// logodds: [h x w] float32. For every ray from cell (ox, oy) to end cell
// (ex[i], ey[i]): cells along the ray get l_free added, and the end cell gets
// l_occ added when hit[i] != 0. Values are clamped to [-l_clamp, l_clamp].
SC_API void sc_integrate_scan(float* logodds, int w, int h, int ox, int oy, const int* ex, const int* ey,
                              const uint8_t* hit, int n, float l_free, float l_occ, float l_clamp) {
  auto inside = [&](int x, int y) { return x >= 0 && y >= 0 && x < w && y < h; };
  auto bump = [&](int x, int y, float v) {
    float& c = logodds[(size_t)y * w + x];
    c = std::max(-l_clamp, std::min(l_clamp, c + v));
  };
  for (int i = 0; i < n; ++i) {
    int x0 = ox, y0 = oy;
    const int x1 = ex[i], y1 = ey[i];
    int dx = std::abs(x1 - x0), dy = -std::abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx + dy;
    for (;;) {
      if (!inside(x0, y0)) break;
      if (x0 == x1 && y0 == y1) {
        if (hit[i]) bump(x0, y0, l_occ);
        break;
      }
      bump(x0, y0, l_free);
      const int e2 = 2 * err;
      if (e2 >= dy) { err += dy; x0 += sx; }
      if (e2 <= dx) { err += dx; y0 += sy; }
    }
  }
}
