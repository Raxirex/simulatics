// slam: native kernels for 2D lidar SLAM (simulatics/plan/slam.py). C ABI, loaded with ctypes.
//
//   sl_likelihood_field  exact Euclidean distance transform of the occupied cells
//                        (Felzenszwalb & Huttenlocher, "Distance Transforms of Sampled Functions",
//                        Theory of Computing 8, 2012) turned into a quantised likelihood field
//                        (Thrun, Burgard & Fox, Probabilistic Robotics, 2005, sec. 6.4) via a lookup table
//   sl_window_max        max of the field over the B x B window anchored at every cell: the coarse level
//                        of the multi-resolution search
//   sl_score_grid        brute-force correlative score of every candidate pose (tests, diagnostics)
//   sl_search            exact two-level branch-and-bound correlative scan matching (Olson, "Real-time
//                        correlative scan matching", ICRA 2009; bounds as in Hess et al., "Real-time loop
//                        closure in 2D LIDAR SLAM", ICRA 2016)
//
// Written for Simulatics (MIT licence); no code is taken from other SLAM packages.
//
// Conventions. Grids are row-major: cell (x, y) is grid[y * w + x]. Field values are integers
// (0..Q, Q chosen in Python) so that sums are exact and the numpy fallback reproduces every score and
// every branch-and-bound decision bit for bit. Scan points arrive as integer cells, one set per
// candidate rotation (rotated and moved to the predicted pose in Python, in double precision):
//     cells[(k * n + i) * 2 + 0] = x cell of point i under rotation k, [.. + 1] = y cell.
// A candidate pose is (rotation k, translation dx, dy cells), |dx|, |dy| <= win. Its score is
//     S(k, dx, dy) = sum_i F[cy + dy][cx + dx]  -  n * (rot_pen[k] + lam * (dx^2 + dy^2))
// (cells outside the grid count 0), i.e. n * Q times "mean likelihood minus a motion prior".
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#if defined(_WIN32)
#define SL_API extern "C" __declspec(dllexport)
#else
#define SL_API extern "C" __attribute__((visibility("default")))
#endif

typedef long long i64;

SL_API int sl_version() { return 1; }

static inline int clamp_threads(int t) { return t < 1 ? 1 : (t > 64 ? 64 : t); }

// ---------------------------------------------------------------------------------------------
// distance transform
// ---------------------------------------------------------------------------------------------
static const double kFar = 1e12;  // "no occupied cell" (finite, so the parabola algebra never sees inf)

// Lower envelope of parabolas (Felzenszwalb & Huttenlocher 2012, algorithm DT(f)):
// d[q] = min_p (q - p)^2 + f[p]. v, z: scratch of n and n + 1 entries.
static void dt_1d(const double* f, int n, double* d, int* v, double* z) {
  int k = 0;
  v[0] = 0;
  z[0] = -1e300;
  z[1] = 1e300;
  for (int q = 1; q < n; ++q) {
    const double fq = f[q] + (double)q * q;
    double s = (fq - (f[v[k]] + (double)v[k] * v[k])) / (2.0 * (q - v[k]));
    while (s <= z[k]) {  // terminates at k = 0: z[0] = -1e300 and s is finite (f <= kFar)
      --k;
      s = (fq - (f[v[k]] + (double)v[k] * v[k])) / (2.0 * (q - v[k]));
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = 1e300;
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < q) ++k;
    const double dq = (double)(q - v[k]);
    d[q] = dq * dq + f[v[k]];
  }
}

// field[c] = lut[d2] where d2 = squared Euclidean distance (cells) from c to the nearest cell with
// logodds > thresh, and 0 when d2 >= lut_n.
SL_API void sl_likelihood_field(const float* logodds, int w, int h, float thresh, const int* lut, int lut_n,
                                int* field, int threads) {
  const int nt = clamp_threads(threads);
  std::vector<double> col((size_t)w * h);
  // pass 1, columns: squared vertical distance to the nearest occupied cell (two linear sweeps)
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int x = 0; x < w; ++x) {
    double last = -1.0;  // y of the last occupied cell seen, -1 = none
    for (int y = 0; y < h; ++y) {
      if (logodds[(size_t)y * w + x] > thresh) last = y;
      col[(size_t)y * w + x] = last < 0 ? kFar : (y - last) * (y - last);
    }
    last = -1.0;
    for (int y = h - 1; y >= 0; --y) {
      if (logodds[(size_t)y * w + x] > thresh) last = y;
      if (last >= 0) {
        const double d = (last - y) * (last - y);
        double& c = col[(size_t)y * w + x];
        if (d < c) c = d;
      }
    }
  }
  // pass 2, rows: parabola envelope over the column distances
#pragma omp parallel num_threads(nt)
  {
    std::vector<double> f(w), d(w), z(w + 1);
    std::vector<int> v(w);
#pragma omp for schedule(static)
    for (int y = 0; y < h; ++y) {
      const double* row = &col[(size_t)y * w];
      int* out = field + (size_t)y * w;
      bool any = false;
      for (int x = 0; x < w; ++x) {
        f[x] = row[x];
        any = any || row[x] < kFar;
      }
      if (!any) {
        std::fill(out, out + w, 0);
        continue;
      }
      dt_1d(f.data(), w, d.data(), v.data(), z.data());
      for (int x = 0; x < w; ++x) {
        const double d2 = d[x];
        out[x] = d2 < (double)lut_n ? lut[(int)(d2 + 0.5)] : 0;
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// coarse level: out has (h + B - 1) x (w + B - 1) cells, out[(y + B - 1) * W + x + B - 1] =
// max of field over [x, x + B) x [y, y + B) for x, y in [-(B - 1), w), cells outside the grid = 0.
// ---------------------------------------------------------------------------------------------
SL_API void sl_window_max(const int* field, int w, int h, int B, int* out, int threads) {
  const int nt = clamp_threads(threads);
  const int W = w + B - 1, H = h + B - 1;
  std::vector<int> hm((size_t)h * W);
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int y = 0; y < h; ++y) {
    const int* row = field + (size_t)y * w;
    for (int X = 0; X < W; ++X) {
      const int x0 = std::max(0, X - (B - 1)), x1 = std::min(w - 1, X);
      int m = 0;
      for (int x = x0; x <= x1; ++x) m = std::max(m, row[x]);
      hm[(size_t)y * W + X] = m;
    }
  }
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int Y = 0; Y < H; ++Y) {
    const int y0 = std::max(0, Y - (B - 1)), y1 = std::min(h - 1, Y);
    int* o = out + (size_t)Y * W;
    for (int X = 0; X < W; ++X) {
      int m = 0;
      for (int y = y0; y <= y1; ++y) m = std::max(m, hm[(size_t)y * W + X]);
      o[X] = m;
    }
  }
}

// ---------------------------------------------------------------------------------------------
// scoring
// ---------------------------------------------------------------------------------------------
static inline i64 sum_at(const int* F, int w, int h, const int* c, int n, int dx, int dy) {
  i64 s = 0;
  for (int i = 0; i < n; ++i) {
    const int x = c[2 * i] + dx, y = c[2 * i + 1] + dy;
    if ((unsigned)x < (unsigned)w && (unsigned)y < (unsigned)h) s += F[(size_t)y * w + x];
  }
  return s;
}

static inline i64 min_sq(int lo, int hi) {  // min of t^2 over integers t in [lo, hi]
  if (lo <= 0 && hi >= 0) return 0;
  const i64 a = lo > 0 ? lo : -hi;
  return a * a;
}

// out[(k * S + (dy + win)) * S + dx + win] = S(k, dx, dy), S = 2 win + 1.
SL_API void sl_score_grid(const int* field, int w, int h, const int* cells, int n_rot, int n, int win,
                          const i64* rot_pen, i64 lam, i64* out, int threads) {
  const int nt = clamp_threads(threads), S = 2 * win + 1;
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int k = 0; k < n_rot; ++k) {
    const int* c = cells + (size_t)k * n * 2;
    for (int dy = -win; dy <= win; ++dy)
      for (int dx = -win; dx <= win; ++dx)
        out[((size_t)k * S + dy + win) * S + dx + win] =
            sum_at(field, w, h, c, n, dx, dy) - (i64)n * (rot_pen[k] + lam * ((i64)dx * dx + (i64)dy * dy));
  }
}

// Exact maximiser of S over all candidates. Coarse "voxels" are (k, X, Y) with X, Y = -win, -win + B, ...;
// each covers dx in [X, min(X + B - 1, win)], same for dy. Its upper bound uses the window-max grid and
// the smallest motion penalty inside it. Voxels are visited in order of decreasing bound (ties: lower
// index first); a voxel is expanded (all its candidates scored, dy-major) only while its bound beats the
// best score so far. Returns the number of expanded voxels; best = (k, dx, dy), best_score = S there.
SL_API int sl_search(const int* field, const int* fmax, int w, int h, const int* cells, int n_rot, int n,
                     int win, int B, const i64* rot_pen, i64 lam, int* best, i64* best_score, int threads) {
  const int nt = clamp_threads(threads);
  const int nv = (2 * win) / B + 1;  // voxel starts per axis
  const int W = w + B - 1, H = h + B - 1;
  const int nvox = n_rot * nv * nv;
  std::vector<i64> ub(nvox);
#pragma omp parallel for num_threads(nt) schedule(static)
  for (int k = 0; k < n_rot; ++k) {
    const int* c = cells + (size_t)k * n * 2;
    for (int iy = 0; iy < nv; ++iy) {
      const int Y = -win + iy * B, Y1 = std::min(Y + B - 1, win);
      for (int ix = 0; ix < nv; ++ix) {
        const int X = -win + ix * B, X1 = std::min(X + B - 1, win);
        // window-max grid is shifted by B - 1 cells
        const i64 s = sum_at(fmax, W, H, c, n, X + B - 1, Y + B - 1);
        ub[((size_t)k * nv + iy) * nv + ix] = s - (i64)n * (rot_pen[k] + lam * (min_sq(X, X1) + min_sq(Y, Y1)));
      }
    }
  }
  std::vector<int> order(nvox);
  for (int i = 0; i < nvox; ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return ub[a] > ub[b]; });
  i64 top = INT64_MIN;
  int bk = -1, bx = 0, by = 0, expanded = 0;
  for (int oi = 0; oi < nvox; ++oi) {
    const int v = order[oi];
    if (bk >= 0 && ub[v] <= top) break;
    ++expanded;
    const int k = v / (nv * nv), iy = (v / nv) % nv, ix = v % nv;
    const int Y = -win + iy * B, Y1 = std::min(Y + B - 1, win);
    const int X = -win + ix * B, X1 = std::min(X + B - 1, win);
    const int* c = cells + (size_t)k * n * 2;
    for (int dy = Y; dy <= Y1; ++dy)
      for (int dx = X; dx <= X1; ++dx) {
        const i64 s = sum_at(field, w, h, c, n, dx, dy) - (i64)n * (rot_pen[k] + lam * ((i64)dx * dx + (i64)dy * dy));
        if (bk < 0 || s > top) {
          top = s;
          bk = k;
          bx = dx;
          by = dy;
        }
      }
  }
  best[0] = bk;
  best[1] = bx;
  best[2] = by;
  *best_score = top;
  return expanded;
}
