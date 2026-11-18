// topopt: matrix-free kernels for 3D SIMP topology optimisation (C ABI, loaded via ctypes).
//
// Discretisation (same as top3d, Liu & Tovar, SMO 50, 2014; and top3D125, Ferrari & Sigmund, SMO 62, 2020):
// a regular grid of nx*ny*nz cubic hex8 (trilinear brick) elements, 3 displacement DOFs per node.
// Every element shares one 24x24 stiffness matrix `ke` (unit Young's modulus, side h, passed in from
// Python), scaled per element by E[e] (SIMP: E = Emin + x^p (E0 - Emin)). The global matrix is never
// assembled: y = K u is computed element by element ("matrix-free", as in TopOpt_in_PETSc, Aage et
// al., SMO 51, 2015). Elements are processed in 8 colours by index parity (i%2, j%2, k%2); elements of
// one colour share no node, so their scatter-adds run in parallel without atomics.
//
//   tp_matvec          y = K(E) u, identity rows on fixed DOFs
//   tp_diag            diagonal of K(E) (Jacobi preconditioner)
//   tp_pcg             Jacobi-preconditioned conjugate gradient for K u = f (warm start from u)
//   tp_element_energy  ce[e] = u_e^T ke u_e (compliance sensitivities)
//   tp_filter          out[e] = sum_i max(0, rmin - |c_e - c_i|) in[i] (density filter, Bruns & Tortorelli 2001)
//   tp_mgpcg           multigrid-preconditioned CG for several load cases (Galerkin coarse grids)
//
// Indexing (row-major, z fastest): element (i, j, k) -> (i*ny + j)*nz + k,
// node (i, j, k) -> (i*(ny+1) + j)*(nz+1) + k, DOF = 3*node + component.
// Local node a = 0..7 of an element sits at offset ((a>>2)&1, (a>>1)&1, a&1); local DOF = 3a + c.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#define SC_API extern "C" __declspec(dllexport)
#else
#define SC_API extern "C" __attribute__((visibility("default")))
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct Grid {
  int nx, ny, nz;
  long long sx, sy;  // node strides along i and j (k stride is 1)
  long long off[8];  // DOF offset of each local node relative to the element's first node
  Grid(int a, int b, int c) : nx(a), ny(b), nz(c) {
    sy = (long long)(c + 1);
    sx = (long long)(b + 1) * sy;
    for (int n = 0; n < 8; ++n) off[n] = 3 * (((n >> 2) & 1) * sx + ((n >> 1) & 1) * sy + (n & 1));
  }
  long long nel() const { return (long long)nx * ny * nz; }
  long long ndof() const { return 3LL * (nx + 1) * (ny + 1) * (nz + 1); }
  long long base_dof(int i, int j, int k) const { return 3 * (i * sx + j * sy + k); }
};

inline int clamp_threads(int nt) { return nt < 1 ? 1 : nt; }

// Calls f(i, j, k, e) for every element of one parity colour, in parallel.
template <class F>
void for_colour(const Grid& g, int colour, int nt, F f) {
  const int ci = (colour >> 2) & 1, cj = (colour >> 1) & 1, ck = colour & 1;
  const int mx = (g.nx - ci + 1) / 2, my = (g.ny - cj + 1) / 2, mz = (g.nz - ck + 1) / 2;
  const long long cnt = (long long)mx * my * mz;
  if (cnt <= 0) return;
#pragma omp parallel for num_threads(nt) schedule(static)
  for (long long t = 0; t < cnt; ++t) {
    const int i = ci + 2 * (int)(t / ((long long)my * mz));
    const long long rem = t % ((long long)my * mz);
    const int j = cj + 2 * (int)(rem / mz);
    const int k = ck + 2 * (int)(rem % mz);
    f(i, j, k, ((long long)i * g.ny + j) * g.nz + k);
  }
}

void matvec(const Grid& g, const double* ke, const double* E, const uint8_t* fixed, const double* u, double* y,
            int nt) {
  const long long ndof = g.ndof();
#pragma omp parallel for num_threads(nt) schedule(static)
  for (long long q = 0; q < ndof; ++q) y[q] = 0.0;
  for (int colour = 0; colour < 8; ++colour) {
    for_colour(g, colour, nt, [&](int i, int j, int k, long long e) {
      const double Ee = E[e];
      if (Ee == 0.0) return;
      const long long b = g.base_dof(i, j, k);
      double ue[24];
      for (int a = 0; a < 8; ++a)
        for (int c = 0; c < 3; ++c) {
          const long long d = b + g.off[a] + c;
          ue[3 * a + c] = fixed[d] ? 0.0 : u[d];
        }
      for (int r = 0; r < 24; ++r) {
        const double* row = ke + r * 24;
        double s = 0.0;
        for (int c = 0; c < 24; ++c) s += row[c] * ue[c];
        y[b + g.off[r / 3] + r % 3] += Ee * s;
      }
    });
  }
#pragma omp parallel for num_threads(nt) schedule(static)
  for (long long q = 0; q < ndof; ++q)
    if (fixed[q]) y[q] = u[q];
}

void diagonal(const Grid& g, const double* ke, const double* E, const uint8_t* fixed, double* d, int nt) {
  const long long ndof = g.ndof();
#pragma omp parallel for num_threads(nt) schedule(static)
  for (long long q = 0; q < ndof; ++q) d[q] = 0.0;
  for (int colour = 0; colour < 8; ++colour) {
    for_colour(g, colour, nt, [&](int i, int j, int k, long long e) {
      const long long b = g.base_dof(i, j, k);
      for (int r = 0; r < 24; ++r) d[b + g.off[r / 3] + r % 3] += E[e] * ke[r * 25];
    });
  }
#pragma omp parallel for num_threads(nt) schedule(static)
  for (long long q = 0; q < ndof; ++q)
    if (fixed[q] || d[q] <= 0.0) d[q] = 1.0;
}

double dot(const double* a, const double* b, long long n, int nt) {
  double s = 0.0;
#pragma omp parallel for num_threads(nt) reduction(+ : s) schedule(static)
  for (long long q = 0; q < n; ++q) s += a[q] * b[q];
  return s;
}

}  // namespace

SC_API int tp_version() { return 1; }

SC_API int tp_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

SC_API void tp_matvec(int nx, int ny, int nz, const double* ke, const double* E, const uint8_t* fixed,
                      const double* u, double* y, int threads) {
  matvec(Grid(nx, ny, nz), ke, E, fixed, u, y, clamp_threads(threads));
}

SC_API void tp_diag(int nx, int ny, int nz, const double* ke, const double* E, const uint8_t* fixed, double* d,
                    int threads) {
  diagonal(Grid(nx, ny, nz), ke, E, fixed, d, clamp_threads(threads));
}

// Solves K(E) u = f on the free DOFs (fixed DOFs are held at zero). `u` is the initial guess on entry
// and the solution on exit. Stops when ||f - K u|| <= tol * ||f||. Returns the number of iterations,
// negated when maxit was reached first; *relres receives the final relative residual.
SC_API int tp_pcg(int nx, int ny, int nz, const double* ke, const double* E, const uint8_t* fixed, const double* f,
                  double* u, double tol, int maxit, int threads, double* relres) {
  const Grid g(nx, ny, nz);
  const int nt = clamp_threads(threads);
  const long long n = g.ndof();
  std::vector<double> r(n), z(n), p(n), q(n), dinv(n);
  diagonal(g, ke, E, fixed, dinv.data(), nt);
#pragma omp parallel for num_threads(nt) schedule(static)
  for (long long i = 0; i < n; ++i) {
    dinv[i] = 1.0 / dinv[i];
    if (fixed[i]) u[i] = 0.0;
  }
  matvec(g, ke, E, fixed, u, q.data(), nt);
  double fnorm2 = 0.0;
#pragma omp parallel for num_threads(nt) reduction(+ : fnorm2) schedule(static)
  for (long long i = 0; i < n; ++i) {
    const double fi = fixed[i] ? 0.0 : f[i];
    r[i] = fi - q[i];
    fnorm2 += fi * fi;
  }
  const double fnorm = std::sqrt(fnorm2);
  if (fnorm == 0.0) {
    std::fill(u, u + n, 0.0);
    if (relres) *relres = 0.0;
    return 0;
  }
  double rnorm = std::sqrt(dot(r.data(), r.data(), n, nt));
  if (rnorm <= tol * fnorm) {
    if (relres) *relres = rnorm / fnorm;
    return 0;
  }
#pragma omp parallel for num_threads(nt) schedule(static)
  for (long long i = 0; i < n; ++i) {
    z[i] = r[i] * dinv[i];
    p[i] = z[i];
  }
  double rz = dot(r.data(), z.data(), n, nt);
  int it = 0;
  bool converged = false;
  while (it < maxit) {
    ++it;
    matvec(g, ke, E, fixed, p.data(), q.data(), nt);
    const double pq = dot(p.data(), q.data(), n, nt);
    if (!(pq > 0.0)) break;  // breakdown (should not happen for an SPD system)
    const double alpha = rz / pq;
    double rr = 0.0;
#pragma omp parallel for num_threads(nt) reduction(+ : rr) schedule(static)
    for (long long i = 0; i < n; ++i) {
      u[i] += alpha * p[i];
      r[i] -= alpha * q[i];
      rr += r[i] * r[i];
    }
    rnorm = std::sqrt(rr);
    if (rnorm <= tol * fnorm) {
      converged = true;
      break;
    }
    double rz_new = 0.0;
#pragma omp parallel for num_threads(nt) reduction(+ : rz_new) schedule(static)
    for (long long i = 0; i < n; ++i) {
      z[i] = r[i] * dinv[i];
      rz_new += r[i] * z[i];
    }
    const double beta = rz_new / rz;
    rz = rz_new;
#pragma omp parallel for num_threads(nt) schedule(static)
    for (long long i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
  }
  if (relres) *relres = rnorm / fnorm;
  return converged ? it : -it;
}

SC_API void tp_element_energy(int nx, int ny, int nz, const double* ke, const double* u, double* ce, int threads) {
  const Grid g(nx, ny, nz);
  const int nt = clamp_threads(threads);
  const long long nel = g.nel();
#pragma omp parallel for num_threads(nt) schedule(static)
  for (long long e = 0; e < nel; ++e) {
    const int i = (int)(e / ((long long)ny * nz));
    const long long rem = e % ((long long)ny * nz);
    const int j = (int)(rem / nz), k = (int)(rem % nz);
    const long long b = g.base_dof(i, j, k);
    double ue[24];
    for (int a = 0; a < 8; ++a)
      for (int c = 0; c < 3; ++c) ue[3 * a + c] = u[b + g.off[a] + c];
    double s = 0.0;
    for (int r = 0; r < 24; ++r) {
      const double* row = ke + r * 24;
      double t = 0.0;
      for (int c = 0; c < 24; ++c) t += row[c] * ue[c];
      s += ue[r] * t;
    }
    ce[e] = s;
  }
}

SC_API void tp_filter(int nx, int ny, int nz, double rmin, const double* in, double* out, int threads) {
  const int nt = clamp_threads(threads);
  const int R = std::max(0, (int)std::ceil(rmin) - 1);
  const long long nel = (long long)nx * ny * nz;
#pragma omp parallel for num_threads(nt) schedule(static)
  for (long long e = 0; e < nel; ++e) {
    const int i = (int)(e / ((long long)ny * nz));
    const long long rem = e % ((long long)ny * nz);
    const int j = (int)(rem / nz), k = (int)(rem % nz);
    double s = 0.0;
    for (int di = -R; di <= R; ++di) {
      const int ii = i + di;
      if (ii < 0 || ii >= nx) continue;
      for (int dj = -R; dj <= R; ++dj) {
        const int jj = j + dj;
        if (jj < 0 || jj >= ny) continue;
        for (int dk = -R; dk <= R; ++dk) {
          const int kk = k + dk;
          if (kk < 0 || kk >= nz) continue;
          const double w = rmin - std::sqrt((double)(di * di + dj * dj + dk * dk));
          if (w > 0.0) s += w * in[((long long)ii * ny + jj) * nz + kk];
        }
      }
    }
    out[e] = s;
  }
}

// ---------------------------------------------------------------------------------------------
// Multigrid-preconditioned CG (MGCG; Amir, Aage and Lazarov, SMO 49, 2014).
//
// Levels are obtained by halving the element grid while all three dimensions are even. Coarse
// operators are Galerkin products K_c = P^T K P with trilinear prolongation P, formed element by
// element: a coarse element's 24x24 matrix is sum_children P_c^T K_child P_c (exact, because the
// trilinear interpolation of a child's nodes only involves its coarse element's 8 nodes). Fixed fine
// DOFs are removed from P (their rows/columns are masked); a coarse DOF is fixed when the fine DOF at
// the same node is. Smoother: damped Jacobi, `sweeps` pre and post (symmetric V-cycle, so the
// preconditioner is SPD). Coarsest level: banded Cholesky of the assembled matrix (node ordering
// with the long x axis slowest gives a bandwidth of 3 (ny+1)(nz+1) + O(1)).
// ---------------------------------------------------------------------------------------------
namespace {

struct Level {
  Grid g;
  std::vector<double> Ke;        // explicit element matrices (levels >= 1), nel * 576
  std::vector<uint8_t> fixed;
  std::vector<double> dinv;
  std::vector<double> b, x, r, t;
  double omega = 0.6;            // damped-Jacobi weight of this level
  explicit Level(int nx, int ny, int nz) : g(nx, ny, nz) {}
  long long ndof() const { return g.ndof(); }
};

struct MG {
  std::vector<Level> L;
  const double* ke = nullptr;    // fine element matrix (E = 1)
  const double* E = nullptr;     // fine element moduli
  int nt = 1, sweeps = 2;
  double omega = 0.6;
  // coarsest banded Cholesky
  long long n_c = 0;
  int band = 0;
  std::vector<double> chol;      // row i, offset d = i - j in [0, band]: chol[i * (band + 1) + d]
};

void matvec_explicit(const Grid& g, const double* Ke, const uint8_t* fixed, const double* u, double* y, int nt) {
  const long long ndof = g.ndof();
#pragma omp parallel for num_threads(nt) schedule(static)
  for (long long q = 0; q < ndof; ++q) y[q] = 0.0;
  for (int colour = 0; colour < 8; ++colour) {
    for_colour(g, colour, nt, [&](int i, int j, int k, long long e) {
      const double* K = Ke + e * 576;
      const long long b = g.base_dof(i, j, k);
      double ue[24];
      for (int a = 0; a < 8; ++a)
        for (int c = 0; c < 3; ++c) {
          const long long d = b + g.off[a] + c;
          ue[3 * a + c] = fixed[d] ? 0.0 : u[d];
        }
      for (int r = 0; r < 24; ++r) {
        const double* row = K + r * 24;
        double s = 0.0;
        for (int c = 0; c < 24; ++c) s += row[c] * ue[c];
        y[b + g.off[r / 3] + r % 3] += s;
      }
    });
  }
#pragma omp parallel for num_threads(nt) schedule(static)
  for (long long q = 0; q < ndof; ++q)
    if (fixed[q]) y[q] = u[q];
}

void level_apply(const MG& mg, int l, const double* x, double* y) {
  const Level& lv = mg.L[l];
  if (l == 0)
    matvec(lv.g, mg.ke, mg.E, lv.fixed.data(), x, y, mg.nt);
  else
    matvec_explicit(lv.g, lv.Ke.data(), lv.fixed.data(), x, y, mg.nt);
}

void level_diag(MG& mg, int l) {
  Level& lv = mg.L[l];
  const long long n = lv.ndof();
  lv.dinv.assign(n, 0.0);
  if (l == 0) {
    diagonal(lv.g, mg.ke, mg.E, lv.fixed.data(), lv.dinv.data(), mg.nt);
  } else {
    double* d = lv.dinv.data();
    for (int colour = 0; colour < 8; ++colour)
      for_colour(lv.g, colour, mg.nt, [&](int i, int j, int k, long long e) {
        const long long b = lv.g.base_dof(i, j, k);
        for (int r = 0; r < 24; ++r) d[b + lv.g.off[r / 3] + r % 3] += lv.Ke[e * 576 + r * 25];
      });
    for (long long q = 0; q < n; ++q)
      if (lv.fixed[q] || d[q] <= 0.0) d[q] = 1.0;
  }
  for (long long q = 0; q < n; ++q) lv.dinv[q] = 1.0 / lv.dinv[q];
}

// P_c (24 x 24): coarse element DOFs -> DOFs of child c, trilinear weights (block diagonal per component)
void prolongation_blocks(double P[8][576]) {
  for (int c = 0; c < 8; ++c) {
    std::fill(P[c], P[c] + 576, 0.0);
    const int ci = (c >> 2) & 1, cj = (c >> 1) & 1, ck = c & 1;
    for (int a = 0; a < 8; ++a) {
      const double t[3] = {0.5 * (ci + ((a >> 2) & 1)), 0.5 * (cj + ((a >> 1) & 1)), 0.5 * (ck + (a & 1))};
      for (int A = 0; A < 8; ++A) {
        const int s[3] = {(A >> 2) & 1, (A >> 1) & 1, A & 1};
        double w = 1.0;
        for (int d = 0; d < 3; ++d) w *= s[d] ? t[d] : 1.0 - t[d];
        if (w == 0.0) continue;
        for (int comp = 0; comp < 3; ++comp) P[c][(3 * a + comp) * 24 + 3 * A + comp] = w;
      }
    }
  }
}

void build_coarse(MG& mg, int l) {
  const Level& f = mg.L[l];
  Level& c = mg.L[l + 1];
  std::vector<double> Pbuf(8 * 576), Gbuf(8 * 576);
  double(*P)[576] = reinterpret_cast<double(*)[576]>(Pbuf.data());
  double(*G)[576] = reinterpret_cast<double(*)[576]>(Gbuf.data());  // P_c^T ke P_c per unit E (level 0)
  prolongation_blocks(P);
  if (l == 0) {
    for (int ch = 0; ch < 8; ++ch) {
      double T[576];
      for (int r = 0; r < 24; ++r)
        for (int q = 0; q < 24; ++q) {
          double s = 0.0;
          for (int m = 0; m < 24; ++m) s += mg.ke[r * 24 + m] * P[ch][m * 24 + q];
          T[r * 24 + q] = s;
        }
      for (int r = 0; r < 24; ++r)
        for (int q = 0; q < 24; ++q) {
          double s = 0.0;
          for (int m = 0; m < 24; ++m) s += P[ch][m * 24 + r] * T[m * 24 + q];
          G[ch][r * 24 + q] = s;
        }
    }
  }
  const long long nel_c = c.g.nel();
  c.Ke.assign(nel_c * 576, 0.0);
  const int cny = c.g.ny, cnz = c.g.nz, fny = f.g.ny, fnz = f.g.nz;
#pragma omp parallel for num_threads(mg.nt) schedule(static)
  for (long long ec = 0; ec < nel_c; ++ec) {
    const int I = (int)(ec / ((long long)cny * cnz));
    const long long rem = ec % ((long long)cny * cnz);
    const int J = (int)(rem / cnz), K = (int)(rem % cnz);
    double* acc = c.Ke.data() + ec * 576;
    for (int ch = 0; ch < 8; ++ch) {
      const int i = 2 * I + ((ch >> 2) & 1), j = 2 * J + ((ch >> 1) & 1), k = 2 * K + (ch & 1);
      const long long ef = ((long long)i * fny + j) * fnz + k;
      const long long b = f.g.base_dof(i, j, k);
      bool any_fixed = false;
      uint8_t m[24];
      for (int a = 0; a < 8; ++a)
        for (int comp = 0; comp < 3; ++comp) {
          m[3 * a + comp] = f.fixed[b + f.g.off[a] + comp] ? 0 : 1;
          any_fixed |= !m[3 * a + comp];
        }
      if (l == 0 && !any_fixed) {
        const double Ee = mg.E[ef];
        for (int q = 0; q < 576; ++q) acc[q] += Ee * G[ch][q];
        continue;
      }
      double Kc[576], T[576];
      for (int r = 0; r < 24; ++r)
        for (int q = 0; q < 24; ++q)
          Kc[r * 24 + q] = (m[r] && m[q]) ? (l == 0 ? mg.E[ef] * mg.ke[r * 24 + q] : f.Ke[ef * 576 + r * 24 + q]) : 0.0;
      for (int r = 0; r < 24; ++r)
        for (int q = 0; q < 24; ++q) {
          double s = 0.0;
          for (int mm = 0; mm < 24; ++mm) s += Kc[r * 24 + mm] * P[ch][mm * 24 + q];
          T[r * 24 + q] = s;
        }
      for (int r = 0; r < 24; ++r)
        for (int q = 0; q < 24; ++q) {
          double s = 0.0;
          for (int mm = 0; mm < 24; ++mm) s += P[ch][mm * 24 + r] * T[mm * 24 + q];
          acc[r * 24 + q] += s;
        }
    }
  }
  // coarse DOF fixed when the fine DOF at the same node is fixed
  const long long nnc = (long long)(c.g.nx + 1) * (c.g.ny + 1) * (c.g.nz + 1);
  c.fixed.assign(3 * nnc, 0);
  for (int I = 0; I <= c.g.nx; ++I)
    for (int J = 0; J <= c.g.ny; ++J)
      for (int K = 0; K <= c.g.nz; ++K) {
        const long long bc = c.g.base_dof(I, J, K), bf = f.g.base_dof(2 * I, 2 * J, 2 * K);
        for (int comp = 0; comp < 3; ++comp) c.fixed[bc + comp] = f.fixed[bf + comp];
      }
}

// r_c = P^T r_f (fixed fine DOFs excluded, fixed coarse DOFs zeroed)
void restrict_to(const MG& mg, int l, const double* rf, double* rc) {
  const Level& f = mg.L[l];
  const Level& c = mg.L[l + 1];
  const int fnx = f.g.nx, fny = f.g.ny, fnz = f.g.nz;
  const long long nnc = (long long)(c.g.nx + 1) * (c.g.ny + 1) * (c.g.nz + 1);
  const int cy = c.g.ny + 1, cz = c.g.nz + 1;
#pragma omp parallel for num_threads(mg.nt) schedule(static)
  for (long long N = 0; N < nnc; ++N) {
    const int I = (int)(N / ((long long)cy * cz));
    const long long rem = N % ((long long)cy * cz);
    const int J = (int)(rem / cz), K = (int)(rem % cz);
    double s[3] = {0, 0, 0};
    for (int di = -1; di <= 1; ++di) {
      const int i = 2 * I + di;
      if (i < 0 || i > fnx) continue;
      const double wi = di ? 0.5 : 1.0;
      for (int dj = -1; dj <= 1; ++dj) {
        const int j = 2 * J + dj;
        if (j < 0 || j > fny) continue;
        const double wj = dj ? 0.5 : 1.0;
        for (int dk = -1; dk <= 1; ++dk) {
          const int k = 2 * K + dk;
          if (k < 0 || k > fnz) continue;
          const double w = wi * wj * (dk ? 0.5 : 1.0);
          const long long bf = f.g.base_dof(i, j, k);
          for (int comp = 0; comp < 3; ++comp)
            if (!f.fixed[bf + comp]) s[comp] += w * rf[bf + comp];
        }
      }
    }
    for (int comp = 0; comp < 3; ++comp) rc[3 * N + comp] = c.fixed[3 * N + comp] ? 0.0 : s[comp];
  }
}

// x_f += P e_c (fixed fine DOFs untouched)
void prolong_add(const MG& mg, int l, const double* ec, double* xf) {
  const Level& f = mg.L[l];
  const Level& c = mg.L[l + 1];
  const int fy = f.g.ny + 1, fz = f.g.nz + 1;
  const long long nnf = (long long)(f.g.nx + 1) * fy * fz;
#pragma omp parallel for num_threads(mg.nt) schedule(static)
  for (long long n = 0; n < nnf; ++n) {
    const int i = (int)(n / ((long long)fy * fz));
    const long long rem = n % ((long long)fy * fz);
    const int j = (int)(rem / fz), k = (int)(rem % fz);
    const int ni = (i & 1) ? 2 : 1, nj = (j & 1) ? 2 : 1, nk = (k & 1) ? 2 : 1;
    const double wi = (i & 1) ? 0.5 : 1.0, wj = (j & 1) ? 0.5 : 1.0, wk = (k & 1) ? 0.5 : 1.0;
    double s[3] = {0, 0, 0};
    for (int a = 0; a < ni; ++a)
      for (int b = 0; b < nj; ++b)
        for (int d = 0; d < nk; ++d) {
          const long long bc = c.g.base_dof(i / 2 + a, j / 2 + b, k / 2 + d);
          for (int comp = 0; comp < 3; ++comp) s[comp] += wi * wj * wk * ec[bc + comp];
        }
    for (int comp = 0; comp < 3; ++comp)
      if (!f.fixed[3 * n + comp]) xf[3 * n + comp] += s[comp];
  }
}

bool coarse_factor(MG& mg) {
  const Level& c = mg.L.back();
  const long long n = c.ndof();
  const int band = (int)(3 * ((long long)(c.g.ny + 1) * (c.g.nz + 1) + (c.g.nz + 1) + 1) + 2);
  mg.n_c = n;
  mg.band = band;
  const int W = band + 1;
  mg.chol.assign(n * W, 0.0);
  double* A = mg.chol.data();
  // assemble the lower band
  for (int i = 0; i < c.g.nx; ++i)
    for (int j = 0; j < c.g.ny; ++j)
      for (int k = 0; k < c.g.nz; ++k) {
        const long long e = ((long long)i * c.g.ny + j) * c.g.nz + k;
        const long long b = c.g.base_dof(i, j, k);
        long long d[24];
        for (int a = 0; a < 8; ++a)
          for (int comp = 0; comp < 3; ++comp) d[3 * a + comp] = b + c.g.off[a] + comp;
        for (int r = 0; r < 24; ++r) {
          if (c.fixed[d[r]]) continue;
          for (int q = 0; q < 24; ++q) {
            if (c.fixed[d[q]] || d[q] > d[r]) continue;
            A[d[r] * W + (d[r] - d[q])] += c.Ke[e * 576 + r * 24 + q];
          }
        }
      }
  for (long long q = 0; q < n; ++q)
    if (c.fixed[q]) A[q * W] = 1.0;
  // in-place banded Cholesky, L stored in the same layout
  double maxdiag = 0.0;
  for (long long q = 0; q < n; ++q) maxdiag = std::max(maxdiag, A[q * W]);
  for (long long j = 0; j < n; ++j) {
    const long long k0 = std::max(0LL, j - band);
    double s = A[j * W];
    for (long long k = k0; k < j; ++k) s -= A[j * W + (j - k)] * A[j * W + (j - k)];
    if (!(s > 1e-300)) s = 1e-14 * std::max(maxdiag, 1e-300);  // guard (should not happen: SPD)
    const double ljj = std::sqrt(s);
    A[j * W] = ljj;
    const long long iend = std::min(n - 1, j + band);
#pragma omp parallel for num_threads(mg.nt) schedule(static) if (iend - j > 256)
    for (long long i = j + 1; i <= iend; ++i) {
      const long long kk0 = std::max(k0, i - band);
      double t = A[i * W + (i - j)];
      for (long long k = kk0; k < j; ++k) t -= A[i * W + (i - k)] * A[j * W + (j - k)];
      A[i * W + (i - j)] = t / ljj;
    }
  }
  return true;
}

// Damped-Jacobi weight from the largest eigenvalue of D^-1 A (power iteration): omega_l = 2 c / lambda_max,
// c = mg.omega (0.6 -> 0.6 for lambda_max = 2, as for a Laplacian; smaller where Galerkin levels are stiffer).
void level_omega(MG& mg, int l) {
  Level& lv = mg.L[l];
  const long long n = lv.ndof();
  std::vector<double> v(n), Av(n);
  unsigned long long seed = 0x9E3779B97F4A7C15ULL;
  for (long long q = 0; q < n; ++q) {
    seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
    v[q] = lv.fixed[q] ? 0.0 : (double)(seed % 2001) / 1000.0 - 1.0;
  }
  double lam = 2.0;
  for (int it = 0; it < 15; ++it) {
    const double nv = std::sqrt(dot(v.data(), v.data(), n, mg.nt));
    if (nv == 0.0) break;
    for (long long q = 0; q < n; ++q) v[q] /= nv;
    level_apply(mg, l, v.data(), Av.data());
    for (long long q = 0; q < n; ++q) Av[q] = lv.fixed[q] ? 0.0 : lv.dinv[q] * Av[q];
    lam = std::sqrt(dot(Av.data(), Av.data(), n, mg.nt));
    v.swap(Av);
  }
  lv.omega = 2.0 * mg.omega / std::max(1.0, 1.05 * lam);
}

void coarse_solve(const MG& mg, const double* b, double* x) {
  const long long n = mg.n_c;
  const int band = mg.band, W = band + 1;
  const double* A = mg.chol.data();
  for (long long i = 0; i < n; ++i) {  // L y = b
    double s = b[i];
    for (long long k = std::max(0LL, i - band); k < i; ++k) s -= A[i * W + (i - k)] * x[k];
    x[i] = s / A[i * W];
  }
  for (long long i = n - 1; i >= 0; --i) {  // L^T x = y
    double s = x[i];
    for (long long k = i + 1; k <= std::min(n - 1, i + band); ++k) s -= A[k * W + (k - i)] * x[k];
    x[i] = s / A[i * W];
  }
}

// x = V-cycle(level l) applied to b (x is overwritten)
void vcycle(MG& mg, int l, const double* b, double* x) {
  Level& lv = mg.L[l];
  const long long n = lv.ndof();
  if (l == (int)mg.L.size() - 1) {
    coarse_solve(mg, b, x);
    for (long long q = 0; q < n; ++q)
      if (lv.fixed[q]) x[q] = 0.0;
    return;
  }
  double* r = lv.r.data();
  const double w = lv.omega;
  std::fill(x, x + n, 0.0);
  auto smooth = [&]() {
    level_apply(mg, l, x, r);
#pragma omp parallel for num_threads(mg.nt) schedule(static)
    for (long long q = 0; q < n; ++q)
      if (!lv.fixed[q]) x[q] += w * lv.dinv[q] * (b[q] - r[q]);
  };
  for (int s = 0; s < mg.sweeps; ++s) smooth();
  level_apply(mg, l, x, r);
#pragma omp parallel for num_threads(mg.nt) schedule(static)
  for (long long q = 0; q < n; ++q) r[q] = lv.fixed[q] ? 0.0 : b[q] - r[q];
  (void)w;
  Level& c = mg.L[l + 1];
  restrict_to(mg, l, r, c.b.data());
  vcycle(mg, l + 1, c.b.data(), c.x.data());
  prolong_add(mg, l, c.x.data(), x);
  for (int s = 0; s < mg.sweeps; ++s) smooth();
}

}  // namespace

// Solves K(E) U_k = F_k for nrhs right-hand sides (rows of F / U, warm start from U) with
// multigrid-preconditioned CG. Returns the number of levels used, or 0 when the grid cannot be
// coarsened (an odd dimension at the finest level) or the coarsest level is too large for the
// banded factorisation; the caller then falls back to tp_pcg. iters[k] < 0 = not converged.
SC_API int tp_mgpcg(int nx, int ny, int nz, const double* ke, const double* E, const uint8_t* fixed, int nrhs,
                    const double* F, double* U, double tol, int maxit, int threads, int max_levels, int sweeps,
                    double omega, int* iters, double* relres) {
  MG mg;
  mg.ke = ke;
  mg.E = E;
  mg.nt = clamp_threads(threads);
  mg.sweeps = std::max(1, sweeps);
  mg.omega = omega;
  mg.L.emplace_back(nx, ny, nz);
  {
    Level& f = mg.L[0];
    f.fixed.assign(fixed, fixed + f.ndof());
  }
  while ((int)mg.L.size() < std::max(2, max_levels)) {
    const Grid& g = mg.L.back().g;
    // stop before a level gets thinner than 2 elements: such grids cannot represent bending
    if (g.nx % 2 || g.ny % 2 || g.nz % 2 || g.nx < 4 || g.ny < 4 || g.nz < 4) break;
    mg.L.emplace_back(g.nx / 2, g.ny / 2, g.nz / 2);
  }
  if (mg.L.size() < 2) return 0;
  {
    const Grid& g = mg.L.back().g;
    const double n = (double)g.ndof();
    const double band = 3.0 * ((g.ny + 1.0) * (g.nz + 1.0) + g.nz + 2.0) + 2.0;
    if (n * band * band > 4e9) return 0;
  }
  for (size_t l = 0; l + 1 < mg.L.size(); ++l) build_coarse(mg, (int)l);
  for (size_t l = 0; l < mg.L.size(); ++l) {
    Level& lv = mg.L[l];
    const long long n = lv.ndof();
    lv.b.assign(n, 0.0);
    lv.x.assign(n, 0.0);
    lv.r.assign(n, 0.0);
    if (l + 1 < mg.L.size()) {
      level_diag(mg, (int)l);
      level_omega(mg, (int)l);
    }
  }
  coarse_factor(mg);

  const Level& f0 = mg.L[0];
  const long long n = f0.ndof();
  const int nt = mg.nt;
  std::vector<double> r(n), z(n), p(n), q(n);
  for (int rhs = 0; rhs < nrhs; ++rhs) {
    const double* f = F + (long long)rhs * n;
    double* u = U + (long long)rhs * n;
    for (long long i = 0; i < n; ++i)
      if (f0.fixed[i]) u[i] = 0.0;
    matvec(f0.g, ke, E, f0.fixed.data(), u, q.data(), nt);
    double fnorm2 = 0.0;
#pragma omp parallel for num_threads(nt) reduction(+ : fnorm2) schedule(static)
    for (long long i = 0; i < n; ++i) {
      const double fi = f0.fixed[i] ? 0.0 : f[i];
      r[i] = fi - q[i];
      fnorm2 += fi * fi;
    }
    const double fnorm = std::sqrt(fnorm2);
    if (fnorm == 0.0) {
      std::fill(u, u + n, 0.0);
      iters[rhs] = 0;
      relres[rhs] = 0.0;
      continue;
    }
    double rnorm = std::sqrt(dot(r.data(), r.data(), n, nt));
    int it = 0;
    bool converged = rnorm <= tol * fnorm;
    double rz = 0.0;
    while (!converged && it < maxit) {
      vcycle(mg, 0, r.data(), z.data());
      const double rz_new = dot(r.data(), z.data(), n, nt);
      if (it == 0) {
#pragma omp parallel for num_threads(nt) schedule(static)
        for (long long i = 0; i < n; ++i) p[i] = z[i];
      } else {
        const double beta = rz_new / rz;
#pragma omp parallel for num_threads(nt) schedule(static)
        for (long long i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
      }
      rz = rz_new;
      ++it;
      matvec(f0.g, ke, E, f0.fixed.data(), p.data(), q.data(), nt);
      const double pq = dot(p.data(), q.data(), n, nt);
      if (!(pq > 0.0)) break;
      const double alpha = rz / pq;
      double rr = 0.0;
#pragma omp parallel for num_threads(nt) reduction(+ : rr) schedule(static)
      for (long long i = 0; i < n; ++i) {
        u[i] += alpha * p[i];
        r[i] -= alpha * q[i];
        rr += r[i] * r[i];
      }
      rnorm = std::sqrt(rr);
      converged = rnorm <= tol * fnorm;
    }
    iters[rhs] = converged ? it : -it;
    relres[rhs] = rnorm / fnorm;
  }
  return (int)mg.L.size();
}
