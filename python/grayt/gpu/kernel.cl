/* GPU port of the Fortran ray-tracing core (src/raytracer.f90 and
 * friends), one work-item per pixel. Runs on any OpenCL device: Apple
 * Silicon GPUs (via Apple's OpenCL-on-Metal, fp32 only -- Apple GPUs
 * have no double-precision hardware) and NVIDIA/AMD/Intel GPUs (fp64).
 *
 * The program is compiled per (metric, precision) with:
 *   -DGRAYT_MID=1|2   Kerr | q-metric (compile-time metric dispatch)
 *   -DGRAYT_FP64      double precision; without it the host must also
 *                     pass -cl-single-precision-constant so the
 *                     unsuffixed literals below stay single precision.
 *
 * Integrator: Dormand-Prince 5(4) with FSAL and the free 4th-order
 * dense-output interpolant for event location, exactly mirroring the
 * CPU path (only rkdp45 is available on the GPU backend).
 * DO NOT build with -cl-fast-relaxed-math: the step controller's NaN
 * check and the q-metric clamps rely on IEEE semantics.
 */

#ifdef GRAYT_FP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double real;
#else
typedef float real;
#endif

#define NDIM 6

#define HMIN 1e-12
#define HMAX_NEAR 25.0
#define HMAX_FAR 100.0
#define HMAX_SCALE 0.1
#define SAFETY 0.9
#define MAX_REJECTS 60
#ifdef GRAYT_FP64
#define BISECTIONS 48
#else
#define BISECTIONS 26   /* 2^-26 < fp32 eps: further halving is a no-op */
#endif
#define QM_FLOOR 1e-14
#define HALF_PI 1.5707963267948966

#define EVENT_EQUATOR 1
#define EVENT_SPHERE  2

/* ------------------------------------------------------------ metrics
 * Component ordering (covariant and contravariant), as in the Fortran:
 *   [0] tt, [1] t-phi, [2] rr, [3] theta-theta, [4] phi-phi          */

#if GRAYT_MID == 2

/* q-metric (Zipoy-Voorhees), src/q_metric.f90; par = quadrupole q.
 * Fractional powers via shared logs (2 log + 2 exp instead of 4 pow). */
static void metric_cov(real q, real r, real theta, real *gd)
{
    real s2 = sin(theta); s2 *= s2;
    real f = fmax((real)(1.0 - 2.0/r), (real)QM_FLOOR);
    real h = fmax((real)(1.0 + s2/(r*r - 2.0*r)), (real)QM_FLOOR);
    real alpha = q*(2.0 + q);
    real fq = exp(q*log(f));
    real f1q = f*fq;
    real ha = exp(alpha*log(h));
    gd[0] = -f1q;
    gd[1] = 0.0;
    gd[2] = 1.0/(f1q*ha);
    gd[3] = r*r/(fq*ha);
    gd[4] = r*r*s2/fq;
}

static void metric_contra(real q, real r, real theta,
                          real *gu, real *dgr, real *dgt)
{
    real s = sin(theta), c = cos(theta);
    real s2 = s*s;
    real u = r*r - 2.0*r;
    real f = fmax((real)(1.0 - 2.0/r), (real)QM_FLOOR);
    real h = fmax((real)(1.0 + s2/u), (real)QM_FLOOR);
    real alpha = q*(2.0 + q);

    /* logarithmic derivatives f'/f, h_r/h, h_th/h */
    real flr = (2.0/(r*r))/f;
    real hlr = (-s2*(2.0*r - 2.0)/(u*u))/h;
    real hlt = (2.0*s*c/u)/h;

    real fq = exp(q*log(f));
    real f1q = f*fq;
    real ha = exp(alpha*log(h));

    gu[0] = -1.0/f1q;
    dgr[0] = gu[0]*(-(1.0 + q)*flr);
    dgt[0] = 0.0;

    gu[1] = 0.0; dgr[1] = 0.0; dgt[1] = 0.0;

    gu[2] = f1q*ha;
    dgr[2] = gu[2]*((1.0 + q)*flr + alpha*hlr);
    dgt[2] = gu[2]*(alpha*hlt);

    gu[3] = fq*ha/(r*r);
    dgr[3] = gu[3]*(q*flr + alpha*hlr - 2.0/r);
    dgt[3] = gu[3]*(alpha*hlt);

    gu[4] = fq/(r*r*s2);
    dgr[4] = gu[4]*(q*flr - 2.0/r);
    dgt[4] = gu[4]*(-2.0*c/s);
}

#else /* GRAYT_MID == 1: Kerr in Boyer-Lindquist, src/kerr_metric.f90 */

static void metric_cov(real a, real r, real theta, real *gd)
{
    real s2 = sin(theta); s2 *= s2;
    real ct = cos(theta);
    real sigma = r*r + a*a*ct*ct;
    real delta = r*r - 2.0*r + a*a;
    gd[0] = -(1.0 - 2.0*r/sigma);
    gd[1] = -2.0*a*r*s2/sigma;
    gd[2] = sigma/delta;
    gd[3] = sigma;
    gd[4] = s2*(r*r + a*a + 2.0*a*a*r*s2/sigma);
}

static void metric_contra(real a, real r, real theta,
                          real *gu, real *dgr, real *dgt)
{
    real s = sin(theta), c = cos(theta);
    real s2 = s*s;
    real sin2t = 2.0*s*c;

    real sigma = r*r + a*a*c*c;
    real delta = r*r - 2.0*r + a*a;
    real af = (r*r + a*a)*(r*r + a*a) - a*a*delta*s2;

    real dsig_r = 2.0*r;
    real dsig_th = -a*a*sin2t;
    real ddel_r = 2.0*r - 2.0;
    real daf_r = 4.0*r*(r*r + a*a) - a*a*s2*ddel_r;
    real daf_th = -a*a*delta*sin2t;

    real d = sigma*delta;
    real dd_r = dsig_r*delta + sigma*ddel_r;
    real dd_th = dsig_th*delta;
    real inv_d2 = 1.0/(d*d);
    real inv_s2g = 1.0/(sigma*sigma);

    gu[0] = -af/d;
    dgr[0] = -(daf_r*d - af*dd_r)*inv_d2;
    dgt[0] = -(daf_th*d - af*dd_th)*inv_d2;

    gu[1] = -2.0*a*r/d;
    dgr[1] = -2.0*a*(d - r*dd_r)*inv_d2;
    dgt[1] = 2.0*a*r*dd_th*inv_d2;

    gu[2] = delta/sigma;
    dgr[2] = (ddel_r*sigma - delta*dsig_r)*inv_s2g;
    dgt[2] = -delta*dsig_th*inv_s2g;

    gu[3] = 1.0/sigma;
    dgr[3] = -dsig_r*inv_s2g;
    dgt[3] = -dsig_th*inv_s2g;

    real n = delta - a*a*s2;
    real w = d*s2;
    real dn_r = ddel_r;
    real dn_th = -a*a*sin2t;
    real dw_r = dd_r*s2;
    real dw_th = dd_th*s2 + d*sin2t;
    real inv_w2 = 1.0/(w*w);
    gu[4] = n/w;
    dgr[4] = (dn_r*w - n*dw_r)*inv_w2;
    dgt[4] = (dn_th*w - n*dw_th)*inv_w2;
}

#endif /* GRAYT_MID */

/* ----------------------------------------------- equations of motion */

static void geodesic_rhs(real par, real pt, real pphi,
                         const real *y, real *dy)
{
    real gu[5], dgr[5], dgt[5];
    metric_contra(par, y[1], y[2], gu, dgr, dgt);

    dy[0] = gu[0]*pt + gu[1]*pphi;
    dy[1] = gu[2]*y[4];
    dy[2] = gu[3]*y[5];
    dy[3] = gu[1]*pt + gu[4]*pphi;
    dy[4] = -0.5*(pt*pt*dgr[0] + 2.0*pt*pphi*dgr[1] +
                  y[4]*y[4]*dgr[2] + y[5]*y[5]*dgr[3] + pphi*pphi*dgr[4]);
    dy[5] = -0.5*(pt*pt*dgt[0] + 2.0*pt*pphi*dgt[1] +
                  y[4]*y[4]*dgt[2] + y[5]*y[5]*dgt[3] + pphi*pphi*dgt[4]);
}

static real hamiltonian_constraint(real par, real pt, real pphi,
                                   const real *y)
{
    real gu[5], dgr[5], dgt[5];
    metric_contra(par, y[1], y[2], gu, dgr, dgt);
    return 0.5*(gu[0]*pt*pt + 2.0*gu[1]*pt*pphi +
                gu[2]*y[4]*y[4] + gu[3]*y[5]*y[5] + gu[4]*pphi*pphi);
}

/* --------------------------------- Dormand-Prince 5(4), FSAL + dense */

static void rkdp45_step_fsal(real par, real pt, real pphi,
                             const real *y, real h, real ks[7][NDIM],
                             int k1_valid, real *ynew, real *errv)
{
    real ytmp[NDIM];
    int i;

    if (!k1_valid) geodesic_rhs(par, pt, pphi, y, ks[0]);

    for (i = 0; i < NDIM; i++)
        ytmp[i] = y[i] + h*(0.2*ks[0][i]);
    geodesic_rhs(par, pt, pphi, ytmp, ks[1]);

    for (i = 0; i < NDIM; i++)
        ytmp[i] = y[i] + h*((3.0/40.0)*ks[0][i] + (9.0/40.0)*ks[1][i]);
    geodesic_rhs(par, pt, pphi, ytmp, ks[2]);

    for (i = 0; i < NDIM; i++)
        ytmp[i] = y[i] + h*((44.0/45.0)*ks[0][i] - (56.0/15.0)*ks[1][i] +
                            (32.0/9.0)*ks[2][i]);
    geodesic_rhs(par, pt, pphi, ytmp, ks[3]);

    for (i = 0; i < NDIM; i++)
        ytmp[i] = y[i] + h*((19372.0/6561.0)*ks[0][i] -
                            (25360.0/2187.0)*ks[1][i] +
                            (64448.0/6561.0)*ks[2][i] -
                            (212.0/729.0)*ks[3][i]);
    geodesic_rhs(par, pt, pphi, ytmp, ks[4]);

    for (i = 0; i < NDIM; i++)
        ytmp[i] = y[i] + h*((9017.0/3168.0)*ks[0][i] -
                            (355.0/33.0)*ks[1][i] +
                            (46732.0/5247.0)*ks[2][i] +
                            (49.0/176.0)*ks[3][i] -
                            (5103.0/18656.0)*ks[4][i]);
    geodesic_rhs(par, pt, pphi, ytmp, ks[5]);

    for (i = 0; i < NDIM; i++)
        ynew[i] = y[i] + h*((35.0/384.0)*ks[0][i] +
                            (500.0/1113.0)*ks[2][i] +
                            (125.0/192.0)*ks[3][i] -
                            (2187.0/6784.0)*ks[4][i] +
                            (11.0/84.0)*ks[5][i]);
    geodesic_rhs(par, pt, pphi, ynew, ks[6]);

    for (i = 0; i < NDIM; i++)
        errv[i] = h*((35.0/384.0 - 5179.0/57600.0)*ks[0][i] +
                     (500.0/1113.0 - 7571.0/16695.0)*ks[2][i] +
                     (125.0/192.0 - 393.0/640.0)*ks[3][i] +
                     (-2187.0/6784.0 + 92097.0/339200.0)*ks[4][i] +
                     (11.0/84.0 - 187.0/2100.0)*ks[5][i] -
                     (1.0/40.0)*ks[6][i]);
}

/* Continuous extension (Hairer's DOPRI5 dense output, 4th order). */
static void dp45_dense_prep(const real *y0, const real *y1, real h,
                            real ks[7][NDIM], real rcont[5][NDIM])
{
    const real d1 = -12715105075.0/11282082432.0;
    const real d3 = 87487479700.0/32700410799.0;
    const real d4 = -10690763975.0/1880347072.0;
    const real d5 = 701980252875.0/199316789632.0;
    const real d6 = -1453857185.0/822651844.0;
    const real d7 = 69997945.0/29380423.0;
    int i;
    for (i = 0; i < NDIM; i++) {
        rcont[0][i] = y0[i];
        rcont[1][i] = y1[i] - y0[i];
        rcont[2][i] = h*ks[0][i] - rcont[1][i];
        rcont[3][i] = rcont[1][i] - h*ks[6][i] - rcont[2][i];
        rcont[4][i] = h*(d1*ks[0][i] + d3*ks[2][i] + d4*ks[3][i] +
                         d5*ks[4][i] + d6*ks[5][i] + d7*ks[6][i]);
    }
}

static void dp45_dense_eval(real rcont[5][NDIM], real theta, real *yout)
{
    real t1 = 1.0 - theta;
    int i;
    for (i = 0; i < NDIM; i++)
        yout[i] = rcont[0][i] + theta*(rcont[1][i] + t1*(rcont[2][i] +
                  theta*(rcont[3][i] + t1*rcont[4][i])));
}

/* --------------------------------------------------- shared machinery */

static int advance_adaptive(real par, real pt, real pphi,
                            real rtol, real atol, const real *y,
                            real *h, real *y1, real *hused,
                            real ks[7][NDIM], int *k1_valid)
{
    real ev[NDIM];
    int k, i;

    *hused = *h;
    for (k = 0; k < MAX_REJECTS; k++) {
        rkdp45_step_fsal(par, pt, pphi, y, *h, ks, *k1_valid, y1, ev);
        *k1_valid = 1;   /* ks[0] = f(y) holds for retries at this y */

        real err2 = 0.0;
        for (i = 0; i < NDIM; i++) {
            real sc = atol + rtol*fmax(fabs(y[i]), fabs(y1[i]));
            real e = ev[i]/sc;
            err2 += e*e;
        }
        real err = fmax(sqrt(err2/(real)NDIM), (real)1e-30);

        if (err > 1.0) {
            *h *= fmax((real)0.2, (real)(SAFETY*pow(err, (real)-0.25)));
            if (fabs(*h) < HMIN) return 0;
            continue;
        }

        *hused = *h;
        *h *= fmin((real)5.0, fmax((real)0.2,
                   (real)(SAFETY*pow(err, (real)-0.2))));
        /* In the weak-curvature far field, let the error controller take
         * longer steps.  Keep the original 25 M cap near the compact object
         * and a conservative absolute ceiling for very distant observers. */
        real hmax = fmin((real)HMAX_FAR,
                         fmax((real)HMAX_NEAR, (real)HMAX_SCALE*y1[1]));
        if (fabs(*h) > hmax) *h = copysign(hmax, *h);
        if (fabs(*h) < HMIN) *h = copysign((real)HMIN, *h);
        return (y1[1] == y1[1]);   /* NaN check */
    }
    return 0;
}

static real event_value(int eid, real epar, const real *y)
{
    if (eid == EVENT_EQUATOR) return cos(y[2]);
    return y[1] - epar;   /* EVENT_SPHERE */
}

static void refine_event(const real *y, real hused, int eid, real epar,
                         const real *yfull, real ks[7][NDIM], real *yc)
{
    real rcont[5][NDIM], ys[NDIM];
    real s0, slo = 0.0, shi = 1.0, s;
    int k, i;

    dp45_dense_prep(y, yfull, hused, ks, rcont);
    s0 = event_value(eid, epar, y);
    for (i = 0; i < NDIM; i++) yc[i] = yfull[i];
    for (k = 0; k < BISECTIONS; k++) {
        s = 0.5*(slo + shi);
        dp45_dense_eval(rcont, s, ys);
        if (s0*event_value(eid, epar, ys) <= 0.0) {
            shi = s;
            for (i = 0; i < NDIM; i++) yc[i] = ys[i];
        } else {
            slo = s;
        }
    }
}

/* ------------------------------------------------------------ camera */

static void camera_init(real par, real r0, real theta0, real phi0,
                        real x, real y, real *y0, real *pt, real *pphi)
{
    real gd[5];
    metric_cov(par, r0, theta0, gd);
    real at = sqrt(gd[4]/(gd[1]*gd[1] - gd[0]*gd[4]));
    *pt = 1.0/at - x*gd[1]/(r0*sqrt(gd[4]));
    *pphi = -x*sqrt(gd[4])/r0;
    real pth = y*sqrt(gd[3])/r0;
    real arg = 1.0 - (x/r0)*(x/r0) - (y/r0)*(y/r0);
    real pr = sqrt(gd[2]*fmax(arg, (real)0.0));

    y0[0] = 0.0;
    y0[1] = r0;
    y0[2] = theta0;
    y0[3] = phi0;
    y0[4] = pr;
    y0[5] = pth;
}

/* --------------------------------------------------------- thin disk */

/* Page-Thorne flux: linear interpolation in the host-computed table. */
static real pt_flux(__global const real *tabf, int tabn,
                    real tab_r1, real tab_r2, real tab_dr, real r)
{
    if (tabn < 2) return 0.0;
    if (r <= tab_r1) return 0.0;
    if (r >= tab_r2) return tabf[tabn - 1];
    real s = (r - tab_r1)/tab_dr;
    int i = (int)s;
    if (i >= tabn - 1) i = tabn - 2;
    s -= (real)i;
    return (1.0 - s)*tabf[i] + s*tabf[i + 1];
}

/* g = nu_obs/nu_em for disk matter with constant specific angular
 * momentum l0 (Kerr only, matching src/disk_model.f90). */
static real redshift_factor(real a, real r, real pt, real pphi, real l0)
{
    real gd[5];
    metric_cov(a, r, (real)HALF_PI, gd);
    real om = -(gd[1] + gd[0]*l0)/(gd[4] + gd[1]*l0);
    real uu = -gd[0] - 2.0*om*gd[1] - om*om*gd[4];
    return sqrt(fmax(uu, (real)0.0))/(pt + om*pphi);
}

/* ------------------------------------------------------------ driver */

static int trace_ray(real par, real pt, real pphi, const real *y0,
                     real rtol, real atol, real r_esc, real rcap,
                     int disk_on, real rin, real rout, int maxstep,
                     int errmon, real *yout, real *herr_max)
{
    real y[NDIM], y1[NDIM], yc[NDIM], ks[7][NDIM];
    real h = -1.0, hused;
    int k1v = 0;
    int n, i, status = 3;

    for (i = 0; i < NDIM; i++) { y[i] = y0[i]; yout[i] = y0[i]; }
    *herr_max = 0.0;

    for (n = 0; n < maxstep; n++) {
        if (!advance_adaptive(par, pt, pphi, rtol, atol, y, &h, y1,
                              &hused, ks, &k1v))
            return 3;
        if (errmon)
            *herr_max = fmax(*herr_max,
                             fabs(hamiltonian_constraint(par, pt, pphi, y1)));

        if (disk_on && cos(y[2])*cos(y1[2]) < 0.0) {
            refine_event(y, hused, EVENT_EQUATOR, (real)0.0, y1, ks, yc);
            if (yc[1] >= rin && yc[1] <= rout) {
                for (i = 0; i < NDIM; i++) yout[i] = yc[i];
                return 2;
            }
        }

        if (y1[1] <= rcap || y1[1] <= 0.0) {
            for (i = 0; i < NDIM; i++) yout[i] = y1[i];
            return 1;
        }
        if (y1[1] >= r_esc) {
            refine_event(y, hused, EVENT_SPHERE, r_esc, y1, ks, yout);
            return 0;
        }

        for (i = 0; i < NDIM; i++) y[i] = y1[i];
        for (i = 0; i < NDIM; i++) ks[0][i] = ks[6][i];   /* FSAL shift */
    }
    return status;
}

/* ------------------------------------------------------------ kernel */

/* One work-item per pixel of a row slab [row0, row0+nrows). Output
 * arrays are (nx, ny) C-order on the host: index i*ny + j, matching
 * the f2py layout so the Python post-processing is shared. */
__kernel void render_slab(
    const real par, const real r0, const real theta0, const real phi0,
    const real xmin, const real xmax, const real ymin, const real ymax,
    const int nx, const int ny, const int row0, const int nrows,
    const real rtol, const real atol,
    const int disk_on, const real rin, const real rout, const real l0,
    const int maxstep, const int errmon,
    const real rcap, const real r_esc,
    __global const real *tabf, const int tabn,
    const real tab_r1, const real tab_r2, const real tab_dr,
    __global real *intens, __global real *gmap, __global real *rhit,
    __global int *status, __global real *herrm,
    __global real *thf, __global real *phf)
{
    int gid = get_global_id(0);
    if (gid >= nx*nrows) return;
    int i = gid % nx;          /* x pixel index */
    int j = row0 + gid/nx;     /* y pixel index */
    int idx = i*ny + j;

    real x = xmin + (xmax - xmin)*((real)i + 0.5)/(real)nx;
    real yy = ymin + (ymax - ymin)*((real)j + 0.5)/(real)ny;

    real y0[NDIM], yout[NDIM], pt, pphi, he;
    camera_init(par, r0, theta0, phi0, x, yy, y0, &pt, &pphi);
    int st = trace_ray(par, pt, pphi, y0, rtol, atol, r_esc, rcap,
                       disk_on, rin, rout, maxstep, errmon, yout, &he);

    status[idx] = st;
    herrm[idx] = he;
    intens[idx] = 0.0;
    gmap[idx] = 0.0;
    rhit[idx] = 0.0;
    thf[idx] = 0.0;
    phf[idx] = 0.0;
    if (st == 2) {
        real g = redshift_factor(par, yout[1], pt, pphi, l0);
        gmap[idx] = g;
        rhit[idx] = yout[1];
        intens[idx] = g*g*g*pt_flux(tabf, tabn, tab_r1, tab_r2,
                                    tab_dr, yout[1]);
    } else if (st == 0) {
        thf[idx] = yout[2];
        phf[idx] = yout[3];
    }
}
