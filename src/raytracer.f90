MODULE RAYTRACER
   ! Backward ray tracing: null geodesics launched from the observer's
   ! image plane and integrated toward the compact object (Sec. 2.2-2.3
   ! of arXiv:2202.00086).
   !
   ! Ray status codes:
   !   0 = escaped to r > r_escape (background / celestial sphere)
   !   1 = captured by the event horizon
   !   2 = hit the accretion disk
   !   3 = integration failed or step budget exhausted
   USE KIND_PARAMS, ONLY: WP, PI, HALF_PI
   USE KERR_METRIC, ONLY: METRIC_COV, HORIZON_RADIUS, ISCO_RADIUS
   USE GEODESIC_EOM, ONLY: GEODESIC_RHS, HAMILTONIAN_CONSTRAINT
   USE RK_INTEGRATORS, ONLY: RK_EMBEDDED_STEP
   USE DISK_MODEL, ONLY: INIT_FLUX_TABLE, PT_FLUX, REDSHIFT_FACTOR
   IMPLICIT NONE
   PRIVATE

   PUBLIC :: CAMERA_INIT, TRACE_RAY, RENDER_IMAGE, TRACE_GEODESIC
   PUBLIC :: GET_HORIZON, GET_ISCO

   REAL(WP), PARAMETER :: HMIN = 1.0E-12_WP
   REAL(WP), PARAMETER :: HMAX = 25.0_WP
   REAL(WP), PARAMETER :: CAPTURE_BUFFER = 1.0E-2_WP
   REAL(WP), PARAMETER :: SAFETY = 0.9_WP

CONTAINS

   SUBROUTINE GET_HORIZON(A, RH)
      REAL(WP), INTENT(IN)  :: A
      REAL(WP), INTENT(OUT) :: RH
      RH = HORIZON_RADIUS(A)
   END SUBROUTINE GET_HORIZON

   SUBROUTINE GET_ISCO(A, RISCO)
      REAL(WP), INTENT(IN)  :: A
      REAL(WP), INTENT(OUT) :: RISCO
      RISCO = ISCO_RADIUS(A)
   END SUBROUTINE GET_ISCO

   PURE SUBROUTINE CAMERA_INIT(A, R0, THETA0, PHI0, X, Y, Y0, PT, PPHI)
      ! Initial conditions on the image plane, eq. (11): pixel (X, Y) in
      ! units of M, observer at (t=0, R0, THETA0, PHI0). The photon energy
      ! measured by the observer is normalized to P^t = 1.
      REAL(WP), INTENT(IN)  :: A, R0, THETA0, PHI0, X, Y
      REAL(WP), INTENT(OUT) :: Y0(6), PT, PPHI
      REAL(WP) :: GD(5), AT, PTH, PR, ARG

      ! NOTE: eq. (7) of the paper prints P^t = A_t[p_t + (g_tp/g_pp)L],
      ! but its own base-change matrix (and the ZAMO tetrad) carries a
      ! minus sign there; with the printed sign the initial conditions
      ! are not null for a /= 0. We use p_t = 1/A_t + (g_tp/g_pp)*L.
      CALL METRIC_COV(A, R0, THETA0, GD)
      AT = SQRT(GD(5)/(GD(2)*GD(2) - GD(1)*GD(5)))
      PT = 1.0_WP/AT - X*GD(2)/(R0*SQRT(GD(5)))
      PPHI = -X*SQRT(GD(5))/R0
      PTH = Y*SQRT(GD(4))/R0
      ARG = 1.0_WP - (X/R0)**2 - (Y/R0)**2
      PR = SQRT(GD(3)*MAX(ARG, 0.0_WP))

      Y0(1) = 0.0_WP
      Y0(2) = R0
      Y0(3) = THETA0
      Y0(4) = PHI0
      Y0(5) = PR
      Y0(6) = PTH
   END SUBROUTINE CAMERA_INIT

   SUBROUTINE TRACE_RAY(A, PT, PPHI, Y0, METHOD, RTOL, ATOL, R_ESC, &
                        DISK_ON, RIN, ROUT, MAXSTEP, STATUS, YOUT, &
                        HERR_MAX, NSTEPS)
      ! Integrate one ray backwards (negative affine-parameter steps) with
      ! adaptive step-size control until capture, escape, or disk hit.
      REAL(WP), INTENT(IN)  :: A, PT, PPHI, Y0(6), RTOL, ATOL, R_ESC
      REAL(WP), INTENT(IN)  :: RIN, ROUT
      INTEGER, INTENT(IN)   :: METHOD, DISK_ON, MAXSTEP
      INTEGER, INTENT(OUT)  :: STATUS, NSTEPS
      REAL(WP), INTENT(OUT) :: YOUT(6), HERR_MAX
      REAL(WP) :: Y(6), Y1(6), YC(6), EV(6), SC(6)
      REAL(WP) :: H, HUSED, ERR, RCAP, C0, C1
      INTEGER :: N
      LOGICAL :: CROSSED

      RCAP = HORIZON_RADIUS(A) + CAPTURE_BUFFER
      Y = Y0
      YOUT = Y0
      H = -1.0_WP
      HERR_MAX = 0.0_WP
      STATUS = 3
      NSTEPS = 0

      DO N = 1, MAXSTEP
         CALL RK_EMBEDDED_STEP(METHOD, A, PT, PPHI, Y, H, Y1, EV)
         SC = ATOL + RTOL*MAX(ABS(Y), ABS(Y1))
         ERR = SQRT(SUM((EV/SC)**2)/6.0_WP)
         ERR = MAX(ERR, 1.0E-30_WP)

         IF (ERR > 1.0_WP) THEN
            ! Rejected step: shrink and retry
            H = H*MAX(0.2_WP, SAFETY*ERR**(-0.25_WP))
            IF (ABS(H) < HMIN) RETURN
            CYCLE
         END IF

         NSTEPS = NSTEPS + 1
         HUSED = H
         H = H*MIN(5.0_WP, MAX(0.2_WP, SAFETY*ERR**(-0.2_WP)))
         IF (ABS(H) > HMAX) H = SIGN(HMAX, H)
         IF (ABS(H) < HMIN) H = SIGN(HMIN, H)

         IF (Y1(2) /= Y1(2)) RETURN  ! NaN guard

         HERR_MAX = MAX(HERR_MAX, ABS(HAMILTONIAN_CONSTRAINT(A, PT, PPHI, Y1)))

         ! Equatorial-plane crossing (disk intersection test)
         IF (DISK_ON == 1) THEN
            C0 = COS(Y(3))
            C1 = COS(Y1(3))
            IF (C0*C1 < 0.0_WP) THEN
               CALL REFINE_CROSSING(METHOD, A, PT, PPHI, Y, HUSED, Y1, YC, CROSSED)
               IF (CROSSED .AND. YC(2) >= RIN .AND. YC(2) <= ROUT) THEN
                  STATUS = 2
                  YOUT = YC
                  RETURN
               END IF
            END IF
         END IF

         IF (Y1(2) <= RCAP .OR. Y1(2) <= 0.0_WP) THEN
            STATUS = 1
            YOUT = Y1
            RETURN
         END IF
         IF (Y1(2) >= R_ESC) THEN
            ! Land exactly on the celestial sphere r = R_ESC, otherwise
            ! the recorded escape direction carries step-size jitter.
            STATUS = 0
            CALL REFINE_RADIUS(METHOD, A, PT, PPHI, Y, HUSED, R_ESC, YOUT)
            RETURN
         END IF

         Y = Y1
      END DO
   END SUBROUTINE TRACE_RAY

   SUBROUTINE REFINE_RADIUS(METHOD, A, PT, PPHI, Y, HUSED, RTARGET, YC)
      ! Bisection on the fraction of the accepted step to locate the
      ! crossing of the sphere r = RTARGET (r is monotonic across the
      ! step by construction: Y is inside, the full step lands outside).
      INTEGER, INTENT(IN)   :: METHOD
      REAL(WP), INTENT(IN)  :: A, PT, PPHI, Y(6), HUSED, RTARGET
      REAL(WP), INTENT(OUT) :: YC(6)
      REAL(WP) :: SLO, SHI, S, YS(6), EV(6)
      INTEGER :: K

      SLO = 0.0_WP
      SHI = 1.0_WP
      CALL RK_EMBEDDED_STEP(METHOD, A, PT, PPHI, Y, HUSED, YC, EV)
      DO K = 1, 48
         S = 0.5_WP*(SLO + SHI)
         CALL RK_EMBEDDED_STEP(METHOD, A, PT, PPHI, Y, S*HUSED, YS, EV)
         IF (YS(2) >= RTARGET) THEN
            SHI = S
            YC = YS
         ELSE
            SLO = S
         END IF
      END DO
   END SUBROUTINE REFINE_RADIUS

   SUBROUTINE REFINE_CROSSING(METHOD, A, PT, PPHI, Y, HUSED, YFULL, YC, CROSSED)
      ! Bisection on the fraction of the accepted step to locate the
      ! equatorial crossing (cos(theta) = 0) to machine precision.
      INTEGER, INTENT(IN)   :: METHOD
      REAL(WP), INTENT(IN)  :: A, PT, PPHI, Y(6), HUSED, YFULL(6)
      REAL(WP), INTENT(OUT) :: YC(6)
      LOGICAL, INTENT(OUT)  :: CROSSED
      REAL(WP) :: SLO, SHI, S, C0, YS(6), EV(6)
      INTEGER :: K

      C0 = COS(Y(3))
      SLO = 0.0_WP
      SHI = 1.0_WP
      YC = YFULL
      CROSSED = .TRUE.

      DO K = 1, 48
         S = 0.5_WP*(SLO + SHI)
         CALL RK_EMBEDDED_STEP(METHOD, A, PT, PPHI, Y, S*HUSED, YS, EV)
         IF (C0*COS(YS(3)) <= 0.0_WP) THEN
            SHI = S
            YC = YS
         ELSE
            SLO = S
         END IF
      END DO
   END SUBROUTINE REFINE_CROSSING

   SUBROUTINE RENDER_IMAGE(A, R0, THETA0, PHI0, XMIN, XMAX, YMIN, YMAX, &
                           NX, NY, METHOD, RTOL, ATOL, DISK_ON, RIN_IN, &
                           ROUT, L0, MAXSTEP, INTENS, GMAP, RHIT, &
                           STATUS, HERRM, THF, PHF)
      ! Render the full image plane. Pixel (I, J) maps to
      !   x = XMIN + (XMAX-XMIN)*(I-1/2)/NX,  y = YMIN + (YMAX-YMIN)*(J-1/2)/NY.
      ! Outputs:
      !   INTENS : observed intensity g^3 * F_PageThorne (disk hits, else 0)
      !   GMAP   : redshift factor g (disk hits, else 0)
      !   RHIT   : Boyer-Lindquist radius of the disk hit (else 0)
      !   STATUS : ray status code per pixel
      !   HERRM  : max |H| along each ray (Hamiltonian constraint error)
      !   THF/PHF: final theta, phi for escaped rays (celestial sphere)
      REAL(WP), INTENT(IN) :: A, R0, THETA0, PHI0, XMIN, XMAX, YMIN, YMAX
      REAL(WP), INTENT(IN) :: RTOL, ATOL, RIN_IN, ROUT, L0
      INTEGER, INTENT(IN)  :: NX, NY, METHOD, DISK_ON, MAXSTEP
      REAL(WP), INTENT(OUT) :: INTENS(NX, NY), GMAP(NX, NY), RHIT(NX, NY)
      REAL(WP), INTENT(OUT) :: HERRM(NX, NY), THF(NX, NY), PHF(NX, NY)
      INTEGER, INTENT(OUT)  :: STATUS(NX, NY)
      REAL(WP) :: RIN, R_ESC, X, YY, Y0(6), YOUT(6), PT, PPHI, HE, G
      INTEGER :: I, J, ST, NS

      RIN = RIN_IN
      IF (RIN <= 0.0_WP) RIN = ISCO_RADIUS(A)
      IF (DISK_ON == 1) CALL INIT_FLUX_TABLE(A, ROUT, 4000)
      R_ESC = 1.1_WP*R0

      INTENS = 0.0_WP
      GMAP = 0.0_WP
      RHIT = 0.0_WP
      THF = 0.0_WP
      PHF = 0.0_WP

      !$OMP PARALLEL DO COLLAPSE(2) SCHEDULE(DYNAMIC, 4) DEFAULT(NONE) &
      !$OMP SHARED(A, R0, THETA0, PHI0, XMIN, XMAX, YMIN, YMAX, NX, NY) &
      !$OMP SHARED(METHOD, RTOL, ATOL, DISK_ON, RIN, ROUT, L0, MAXSTEP, R_ESC) &
      !$OMP SHARED(INTENS, GMAP, RHIT, STATUS, HERRM, THF, PHF) &
      !$OMP PRIVATE(I, J, X, YY, Y0, YOUT, PT, PPHI, HE, G, ST, NS)
      DO J = 1, NY
         DO I = 1, NX
            X = XMIN + (XMAX - XMIN)*(REAL(I, WP) - 0.5_WP)/REAL(NX, WP)
            YY = YMIN + (YMAX - YMIN)*(REAL(J, WP) - 0.5_WP)/REAL(NY, WP)
            CALL CAMERA_INIT(A, R0, THETA0, PHI0, X, YY, Y0, PT, PPHI)
            CALL TRACE_RAY(A, PT, PPHI, Y0, METHOD, RTOL, ATOL, R_ESC, &
                           DISK_ON, RIN, ROUT, MAXSTEP, ST, YOUT, HE, NS)
            STATUS(I, J) = ST
            HERRM(I, J) = HE
            IF (ST == 2) THEN
               G = REDSHIFT_FACTOR(A, YOUT(2), PT, PPHI, L0)
               GMAP(I, J) = G
               RHIT(I, J) = YOUT(2)
               INTENS(I, J) = G**3*PT_FLUX(YOUT(2))
            ELSE IF (ST == 0) THEN
               THF(I, J) = YOUT(3)
               PHF(I, J) = YOUT(4)
            END IF
         END DO
      END DO
      !$OMP END PARALLEL DO
   END SUBROUTINE RENDER_IMAGE

   SUBROUTINE TRACE_GEODESIC(A, Y0, PT, PPHI, METHOD, RTOL, ATOL, H0, &
                             LAM_MAX, NMAX, TRAJ, HERR, NOUT)
      ! Integrate a single geodesic from raw initial conditions, recording
      ! the trajectory and the Hamiltonian constraint error at every
      ! accepted step (for reproducing Figs. 3-5 of the paper).
      ! TRAJ(K, :) = (lambda, t, r, theta, phi, p_r, p_theta).
      REAL(WP), INTENT(IN)  :: A, Y0(6), PT, PPHI, RTOL, ATOL, H0, LAM_MAX
      INTEGER, INTENT(IN)   :: METHOD, NMAX
      REAL(WP), INTENT(OUT) :: TRAJ(NMAX, 7), HERR(NMAX)
      INTEGER, INTENT(OUT)  :: NOUT
      REAL(WP) :: Y(6), Y1(6), EV(6), SC(6), H, ERR, LAM, RCAP
      INTEGER :: N

      RCAP = HORIZON_RADIUS(A) + CAPTURE_BUFFER
      Y = Y0
      H = H0
      LAM = 0.0_WP
      NOUT = 0
      TRAJ = 0.0_WP
      HERR = 0.0_WP

      DO N = 1, 100*NMAX
         IF (NOUT >= NMAX) EXIT
         IF (ABS(LAM) >= ABS(LAM_MAX)) EXIT
         CALL RK_EMBEDDED_STEP(METHOD, A, PT, PPHI, Y, H, Y1, EV)
         SC = ATOL + RTOL*MAX(ABS(Y), ABS(Y1))
         ERR = SQRT(SUM((EV/SC)**2)/6.0_WP)
         ERR = MAX(ERR, 1.0E-30_WP)

         IF (ERR > 1.0_WP) THEN
            H = H*MAX(0.2_WP, SAFETY*ERR**(-0.25_WP))
            IF (ABS(H) < HMIN) EXIT
            CYCLE
         END IF

         LAM = LAM + H
         Y = Y1
         NOUT = NOUT + 1
         TRAJ(NOUT, 1) = LAM
         TRAJ(NOUT, 2:7) = Y
         HERR(NOUT) = ABS(HAMILTONIAN_CONSTRAINT(A, PT, PPHI, Y))

         H = H*MIN(5.0_WP, MAX(0.2_WP, SAFETY*ERR**(-0.2_WP)))
         IF (ABS(H) > HMAX) H = SIGN(HMAX, H)
         IF (ABS(H) < HMIN) H = SIGN(HMIN, H)

         IF (Y(2) <= RCAP .OR. Y(2) /= Y(2)) EXIT
      END DO
   END SUBROUTINE TRACE_GEODESIC

END MODULE RAYTRACER
