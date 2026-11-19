// mpc: dense dual active-set QP solver for the walking MPC (C ABI, loaded via ctypes).
//
//     minimise 1/2 x'Hx + g'x   subject to   C x <= b        (H symmetric positive definite)
//
// Method: Goldfarb & Idnani, "A numerically stable dual method for solving strictly convex quadratic
// programs", Mathematical Programming 27 (1983), written from the paper in its dual-space form:
// every iterate is x = x0 - V_A u_A with x0 = -H^-1 g, V = H^-1 C^T and the active multipliers u_A >= 0,
// so the method only needs W = C H^-1 C^T. Adding a violated constraint p raises its multiplier t while
// the active constraints stay tight (u_A -= t r, r = W_AA^-1 W_Ap); the step is the smaller of the full
// step (p becomes tight) and the partial step (an active multiplier reaches zero and is dropped).
//
// This file mirrors simulatics/control/mpc_qp.py step for step (same tolerances and tie-breaking), so
// the two agree to rounding. Original code for Simulatics (MIT licence); no third-party QP code used.
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#if defined(_WIN32)
#define MPC_API extern "C" __declspec(dllexport)
#else
#define MPC_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

const double kTol = 1e-9;    // constraint violation tolerance
const double kEpsR = 1e-12;  // r_j > kEpsR blocks the step
const double kEpsZ = 1e-11;  // zn <= kEpsZ * W_pp: p depends linearly on the active set

enum Status { kOk = 0, kMaxIter = 1, kInfeasible = 2, kNotConvex = 3 };

// In-place Cholesky (lower triangle) of an n x n row-major matrix. False if not positive definite.
bool cholesky(double* a, int n) {
  for (int j = 0; j < n; ++j) {
    double d = a[j * n + j];
    for (int k = 0; k < j; ++k) d -= a[j * n + k] * a[j * n + k];
    if (!(d > 0.0)) return false;
    d = std::sqrt(d);
    a[j * n + j] = d;
    for (int i = j + 1; i < n; ++i) {
      double s = a[i * n + j];
      for (int k = 0; k < j; ++k) s -= a[i * n + k] * a[j * n + k];
      a[i * n + j] = s / d;
    }
  }
  return true;
}

void forward_solve(const double* L, int n, double* v) {  // L v' = v
  for (int i = 0; i < n; ++i) {
    double s = v[i];
    for (int k = 0; k < i; ++k) s -= L[i * n + k] * v[k];
    v[i] = s / L[i * n + i];
  }
}

void backward_solve(const double* L, int n, double* v) {  // L^T v' = v
  for (int i = n - 1; i >= 0; --i) {
    double s = v[i];
    for (int k = i + 1; k < n; ++k) s -= L[k * n + i] * v[k];
    v[i] = s / L[i * n + i];
  }
}

}  // namespace

MPC_API int mpc_version() { return 1; }

// Returns the status (0 ok, 1 max iterations, 2 infeasible, 3 H not positive definite). Outputs: x (n),
// active (indices, order of activation), mult (their multipliers), n_active, iterations. On a non-ok status
// x is the last iterate (optimal for the constraints activated so far).
MPC_API int mpc_qp_solve(int n, int m, const double* H, const double* g, const double* C, const double* b,
                         const int* warm, int n_warm, int max_iter, double* x, int* active, double* mult,
                         int* n_active, int* iterations) {
  *n_active = 0;
  *iterations = 0;
  std::vector<double> L(H, H + static_cast<size_t>(n) * n);
  if (!cholesky(L.data(), n)) {
    for (int i = 0; i < n; ++i) x[i] = 0.0;
    return kNotConvex;
  }
  std::vector<double> x0(g, g + n);
  forward_solve(L.data(), n, x0.data());
  backward_solve(L.data(), n, x0.data());
  for (int i = 0; i < n; ++i) x0[i] = -x0[i];
  for (int i = 0; i < n; ++i) x[i] = x0[i];
  if (m == 0) return kOk;

  // Y = L^-1 C^T (row i = y_i), V = H^-1 C^T (row i), W = Y^T Y
  std::vector<double> Y(C, C + static_cast<size_t>(m) * n), V(static_cast<size_t>(m) * n);
  for (int i = 0; i < m; ++i) {
    forward_solve(L.data(), n, &Y[static_cast<size_t>(i) * n]);
    std::copy(&Y[static_cast<size_t>(i) * n], &Y[static_cast<size_t>(i) * n] + n, &V[static_cast<size_t>(i) * n]);
    backward_solve(L.data(), n, &V[static_cast<size_t>(i) * n]);
  }
  std::vector<double> W(static_cast<size_t>(m) * m);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j <= i; ++j) {
      double s = 0.0;
      const double* yi = &Y[static_cast<size_t>(i) * n];
      const double* yj = &Y[static_cast<size_t>(j) * n];
      for (int k = 0; k < n; ++k) s += yi[k] * yj[k];
      W[static_cast<size_t>(i) * m + j] = W[static_cast<size_t>(j) * m + i] = s;
    }

  std::vector<int> A;
  std::vector<double> u, r, Lw, s(m);
  std::vector<char> is_active(m, 0);
  A.reserve(m);
  u.reserve(m);
  int qpos = 0, it = 0;

  auto slack = [&](int i) {
    double v = b[i];
    const double* ci = C + static_cast<size_t>(i) * n;
    for (int k = 0; k < n; ++k) v -= ci[k] * x[k];
    return v;
  };
  auto set_x = [&](int p, double up) {  // x = x0 - V_A u_A - V_p up
    for (int k = 0; k < n; ++k) x[k] = x0[k];
    for (size_t j = 0; j < A.size(); ++j) {
      const double* vj = &V[static_cast<size_t>(A[j]) * n];
      for (int k = 0; k < n; ++k) x[k] -= vj[k] * u[j];
    }
    if (p >= 0) {
      const double* vp = &V[static_cast<size_t>(p) * n];
      for (int k = 0; k < n; ++k) x[k] -= vp[k] * up;
    }
  };
  auto finish = [&](int status) {
    *n_active = static_cast<int>(A.size());
    for (size_t j = 0; j < A.size(); ++j) {
      active[j] = A[j];
      mult[j] = u[j];
    }
    *iterations = it;
    return status;
  };

  while (true) {
    for (int i = 0; i < m; ++i) s[i] = slack(i);
    int p = -1;
    while (qpos < n_warm) {
      int c = warm[qpos++];
      if (c >= 0 && c < m && !is_active[c] && s[c] < -kTol) {
        p = c;
        break;
      }
    }
    if (p < 0) {
      double best = std::numeric_limits<double>::infinity();
      for (int i = 0; i < m; ++i)
        if (!is_active[i] && s[i] < best) {
          best = s[i];
          p = i;
        }
      if (p < 0 || best >= -kTol) return finish(kOk);
    }
    double up = 0.0;
    while (true) {
      ++it;
      if (it > max_iter) {
        --it;
        return finish(kMaxIter);
      }
      const int q = static_cast<int>(A.size());
      double zn = W[static_cast<size_t>(p) * m + p];
      r.assign(q, 0.0);
      if (q > 0) {
        Lw.assign(static_cast<size_t>(q) * q, 0.0);
        for (int i = 0; i < q; ++i)
          for (int j = 0; j < q; ++j) Lw[static_cast<size_t>(i) * q + j] = W[static_cast<size_t>(A[i]) * m + A[j]];
        if (!cholesky(Lw.data(), q)) return finish(kNotConvex);
        for (int i = 0; i < q; ++i) r[i] = W[static_cast<size_t>(A[i]) * m + p];
        forward_solve(Lw.data(), q, r.data());
        backward_solve(Lw.data(), q, r.data());
        for (int i = 0; i < q; ++i) zn -= W[static_cast<size_t>(p) * m + A[i]] * r[i];
      }
      double t1 = std::numeric_limits<double>::infinity();
      int k = -1;
      for (int j = 0; j < q; ++j)
        if (r[j] > kEpsR) {
          const double ratio = u[j] / r[j];
          if (ratio < t1) {
            t1 = ratio;
            k = j;
          }
        }
      const double wpp = std::max(W[static_cast<size_t>(p) * m + p], 1e-300);
      const double sp = slack(p);
      const double t2 = zn > kEpsZ * wpp ? -sp / zn : std::numeric_limits<double>::infinity();
      if (!std::isfinite(t1) && !std::isfinite(t2)) return finish(kInfeasible);
      if (t2 <= t1) {  // full step: p joins the active set
        for (int j = 0; j < q; ++j) u[j] -= t2 * r[j];
        u.push_back(up + t2);
        A.push_back(p);
        is_active[p] = 1;
        set_x(-1, 0.0);
        break;
      }
      // partial step: active constraint k leaves, then try p again
      for (int j = 0; j < q; ++j) u[j] = std::max(u[j] - t1 * r[j], 0.0);
      up += t1;
      is_active[A[k]] = 0;
      A.erase(A.begin() + k);
      u.erase(u.begin() + k);
      set_x(p, up);
    }
  }
}
