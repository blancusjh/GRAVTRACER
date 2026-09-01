MODULE RAYTRACER
   ! Ray/geodesic drivers built on two shared pieces of machinery:
   !
   !   ADVANCE_ADAPTIVE -- one error-controlled embedded-RK step
   !                      (attempt, shrink on rejection, suggest next h)
   !   REFINE_EVENT     -- bisection of an accepted step onto the zero of
   !                      a scalar event function g(Y)
   !
   ! The drivers (TRACE_RAY, TRACE_TO_PLANE, TRACE_GEODESIC) are thin
   ! loops over these plus their own stop conditions. The spacetime is
   ! selected by (MID, PAR) -- see module SPACETIME.
   !
   ! Ray status codes:
   !   0 = escaped past r_escape (background / celestial sphere)
   !   1 = captured at the inner boundary (horizon / singular surface)
   !   2 = hit the target surface (disk or plane)
   !   3 = integration failed or step budget exhausted
   USE KIND_PARAMS, ONLY: WP, PI, HALF_PI
   USE KERR_METRIC, ONLY: HORIZON_RADIUS, ISCO_RADIUS
   USE SPACETIME, ONLY: METRIC_COV, INNER_BOUNDARY, MID_KERR
   USE GEODESIC_EOM, ONLY: HAMILTONIAN_CONSTRAINT
   USE RK_INTEGRATORS, ONLY: RK_EMBEDDED_STEP
   USE DISK_MODEL, ONLY: INIT_FLUX_TABLE, PT_FLUX, REDSHIFT_FACTOR
   IMPLICIT NONE
   PRIVATE

   PUBLIC :: CAMERA_INIT, TRACE_RAY, RENDER_IMAGE, TRACE_GEODESIC
   PUBLIC :: TRACE_TO_PLANE, TRACE_BUNDLE_TO_PLANE
   PUBLIC :: GET_HORIZON, GET_ISCO

   REAL(WP), PARAMETER :: HMIN = 1.0E-12_WP
   REAL(WP), PARAMETER :: HMAX = 25.0_WP
   REAL(WP), PARAMETER :: CAPTURE_BUFFER = 1.0E-2_WP
   REAL(WP), PARAMETER :: SAFETY = 0.9_WP
   INTEGER, PARAMETER :: MAX_REJECTS = 60
   INTEGER, PARAMETER :: BISECTIONS = 48

   ! Scalar event functions g(Y) for REFINE_EVENT
   INTEGER, PARAMETER :: EVENT_EQUATOR = 1   ! g = cos(theta)
   INTEGER, PARAMETER :: EVENT_SPHERE = 2    ! g = r - EPAR(1)
   INTEGER, PARAMETER :: EVENT_PLANE = 3     ! g = n.x - d, EPAR = (n, d)

CONTAINS

   SUBROUTINE GET_HORIZON(A, RH)
      ! Kerr outer event horizon (Kerr-only convenience wrapper).
      REAL(WP), INTENT(IN)  :: A
      REAL(WP), INTENT(OUT) :: RH
      RH = HORIZON_RADIUS(A)
   END SUBROUTINE GET_HORIZON

   SUBROUTINE GET_ISCO(A, RISCO)
      ! Kerr prograde ISCO (Kerr-only convenience wrapper).
      REAL(WP), INTENT(IN)  :: A
      REAL(WP), INTENT(OUT) :: RISCO
      RISCO = ISCO_RADIUS(A)
   END SUBROUTINE GET_ISCO

   ! ------------------------------------------------- shared machinery

   SUBROUTINE ADVANCE_ADAPTIVE(METHOD, MID, PAR, PT, PPHI, RTOL, ATOL, &
                               Y, H, Y1, HUSED, OK)
      ! Take one accepted step: attempt with the current H, shrink on
      ! rejection, and on acceptance store the used step in HUSED and
      ! leave the suggestion for the next step in H (clamped to
      ! [HMIN, HMAX]). OK = .FALSE. on step underflow or NaN state.
      INTEGER, INTENT(IN)     :: METHOD, MID
      REAL(WP), INTENT(IN)    :: PAR(4), PT, PPHI, RTOL, ATOL, Y(6)
      REAL(WP), INTENT(INOUT) :: H
      REAL(WP), INTENT(OUT)   :: Y1(6), HUSED
      LOGICAL, INTENT(OUT)    :: OK
      REAL(WP) :: EV(6), SC(6), ERR
      INTEGER :: K

      OK = .FALSE.
      HUSED = H
      DO K = 1, MAX_REJECTS
         CALL RK_EMBEDDED_STEP(METHOD, MID, PAR, PT, PPHI, Y, H, Y1, EV)
         SC = ATOL + RTOL*MAX(ABS(Y), ABS(Y1))
         ERR = SQRT(SUM((EV/SC)**2)/6.0_WP)
         ERR = MAX(ERR, 1.0E-30_WP)

         IF (ERR > 1.0_WP) THEN
            H = H*MAX(0.2_WP, SAFETY*ERR**(-0.25_WP))
            IF (ABS(H) < HMIN) RETURN
            CYCLE
         END IF

         HUSED = H
         H = H*MIN(5.0_WP, MAX(0.2_WP, SAFETY*ERR**(-0.2_WP)))
         IF (ABS(H) > HMAX) H = SIGN(HMAX, H)
         IF (ABS(H) < HMIN) H = SIGN(HMIN, H)
         OK = (Y1(2) == Y1(2))
         RETURN
      END DO
   END SUBROUTINE ADVANCE_ADAPTIVE

   PURE FUNCTION EVENT_VALUE(EID, EPAR, Y) RESULT(S)
      ! Scalar event function g(Y); an event fires where g changes sign.
      INTEGER, INTENT(IN)  :: EID
      REAL(WP), INTENT(IN) :: EPAR(4), Y(6)
      REAL(WP) :: S, XC(3)
      SELECT CASE (EID)
      CASE (EVENT_EQUATOR)
         S = COS(Y(3))
      CASE (EVENT_SPHERE)
         S = Y(2) - EPAR(1)
      CASE DEFAULT  ! EVENT_PLANE
         CALL BL_TO_CART(Y(2), Y(3), Y(4), XC)
         S = EPAR(1)*XC(1) + EPAR(2)*XC(2) + EPAR(3)*XC(3) - EPAR(4)
      END SELECT
   END FUNCTION EVENT_VALUE

   SUBROUTINE REFINE_EVENT(METHOD, MID, PAR, PT, PPHI, Y, HUSED, &
                           EID, EPAR, YFULL, YC)
      ! Bisect the fraction of an accepted step (from Y, size HUSED,
      ! landing at YFULL) onto the sign change of event EID.
      INTEGER, INTENT(IN)   :: METHOD, MID, EID
      REAL(WP), INTENT(IN)  :: PAR(4), PT, PPHI, Y(6), HUSED, EPAR(4)
      REAL(WP), INTENT(IN)  :: YFULL(6)
      REAL(WP), INTENT(OUT) :: YC(6)
      REAL(WP) :: S0, SLO, SHI, S, YS(6), EV(6)
      INTEGER :: K

      S0 = EVENT_VALUE(EID, EPAR, Y)
      SLO = 0.0_WP
      SHI = 1.0_WP
      YC = YFULL
      DO K = 1, BISECTIONS
         S = 0.5_WP*(SLO + SHI)
         CALL RK_EMBEDDED_STEP(METHOD, MID, PAR, PT, PPHI, Y, S*HUSED, YS, EV)
         IF (S0*EVENT_VALUE(EID, EPAR, YS) <= 0.0_WP) THEN
            SHI = S
            YC = YS
         ELSE
            SLO = S
         END IF
      END DO
   END SUBROUTINE REFINE_EVENT

   PURE SUBROUTINE BL_TO_CART(R, TH, PH, XC)
      ! Pseudo-Cartesian embedding for flat-space surfaces (sources,
      ! screens); exact only asymptotically.
      REAL(WP), INTENT(IN)  :: R, TH, PH
      REAL(WP), INTENT(OUT) :: XC(3)
      XC(1) = R*SIN(TH)*COS(PH)
      XC(2) = R*SIN(TH)*SIN(PH)
      XC(3) = R*COS(TH)
   END SUBROUTINE BL_TO_CART

   ! --------------------------------------------------------- camera

   PURE SUBROUTINE CAMERA_INIT(MID, PAR, R0, THETA0, PHI0, X, Y, Y0, PT, PPHI)
      ! Initial conditions on the image plane, eq. (11): pixel (X, Y) in
      ! units of M, observer at (t=0, R0, THETA0, PHI0). The photon
      ! energy measured by the observer is normalized to P^t = 1.
      ! NOTE: eq. (7) of the paper prints P^t = A_t[p_t + (g_tp/g_pp)L],
      ! but its own base-change matrix (and the ZAMO tetrad) carries a
      ! minus sign there; with the printed sign the initial conditions
      ! are not null for g_tp /= 0. We use p_t = 1/A_t + (g_tp/g_pp)*L.
      INTEGER, INTENT(IN)   :: MID
      REAL(WP), INTENT(IN)  :: PAR(4), R0, THETA0, PHI0, X, Y
      REAL(WP), INTENT(OUT) :: Y0(6), PT, PPHI
      REAL(WP) :: GD(5), AT, PTH, PR, ARG

      CALL METRIC_COV(MID, PAR, R0, THETA0, GD)
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

   ! --------------------------------------------------------- drivers

   SUBROUTINE TRACE_RAY(MID, PAR, PT, PPHI, Y0, METHOD, RTOL, ATOL, &
                        R_ESC, DISK_ON, RIN, ROUT, MAXSTEP, STATUS, &
                        YOUT, HERR_MAX, NSTEPS)
      ! Backward ray integration (negative steps) until equatorial disk
      ! hit, capture, or escape (landed exactly on the sphere r = R_ESC).
      INTEGER, INTENT(IN)   :: MID, METHOD, DISK_ON, MAXSTEP
      REAL(WP), INTENT(IN)  :: PAR(4), PT, PPHI, Y0(6), RTOL, ATOL
      REAL(WP), INTENT(IN)  :: R_ESC, RIN, ROUT
      INTEGER, INTENT(OUT)  :: STATUS, NSTEPS
      REAL(WP), INTENT(OUT) :: YOUT(6), HERR_MAX
      REAL(WP) :: Y(6), Y1(6), YC(6), H, HUSED, RCAP, EPAR(4)
      INTEGER :: N
      LOGICAL :: OK

      RCAP = INNER_BOUNDARY(MID, PAR) + CAPTURE_BUFFER
      Y = Y0
      YOUT = Y0
      H = -1.0_WP
      HERR_MAX = 0.0_WP
      STATUS = 3
      NSTEPS = 0

      DO N = 1, MAXSTEP
         CALL ADVANCE_ADAPTIVE(METHOD, MID, PAR, PT, PPHI, RTOL, ATOL, &
                               Y, H, Y1, HUSED, OK)
         IF (.NOT. OK) RETURN
         NSTEPS = NSTEPS + 1
         HERR_MAX = MAX(HERR_MAX, &
                        ABS(HAMILTONIAN_CONSTRAINT(MID, PAR, PT, PPHI, Y1)))

         IF (DISK_ON == 1 .AND. COS(Y(3))*COS(Y1(3)) < 0.0_WP) THEN
            EPAR = 0.0_WP
            CALL REFINE_EVENT(METHOD, MID, PAR, PT, PPHI, Y, HUSED, &
                              EVENT_EQUATOR, EPAR, Y1, YC)
            IF (YC(2) >= RIN .AND. YC(2) <= ROUT) THEN
               STATUS = 2
               YOUT = YC
               RETURN
            END IF
         END IF

         IF (Y1(2) <= RCAP .OR. Y1(2) <= 0.0_WP) THEN
            STATUS = 1
            YOUT = Y1
            RETURN
         END IF
         IF (Y1(2) >= R_ESC) THEN
            STATUS = 0
            EPAR = (/R_ESC, 0.0_WP, 0.0_WP, 0.0_WP/)
            CALL REFINE_EVENT(METHOD, MID, PAR, PT, PPHI, Y, HUSED, &
                              EVENT_SPHERE, EPAR, Y1, YOUT)
            RETURN
         END IF

         Y = Y1
      END DO
   END SUBROUTINE TRACE_RAY

   SUBROUTINE TRACE_TO_PLANE(MID, PAR, PT, PPHI, Y0, METHOD, RTOL, ATOL, &
                             H0, NHAT, DPLANE, RMAX, MAXSTEP, STATUS, &
                             YOUT, NSTEPS)
      ! Integrate one ray (sign of H0 sets the direction) until it
      ! crosses the Cartesian plane n.x = DPLANE (status 2), is captured
      ! (1), escapes past RMAX (0), or fails (3).
      INTEGER, INTENT(IN)   :: MID, METHOD, MAXSTEP
      REAL(WP), INTENT(IN)  :: PAR(4), PT, PPHI, Y0(6), RTOL, ATOL, H0
      REAL(WP), INTENT(IN)  :: NHAT(3), DPLANE, RMAX
      INTEGER, INTENT(OUT)  :: STATUS, NSTEPS
      REAL(WP), INTENT(OUT) :: YOUT(6)
      REAL(WP) :: Y(6), Y1(6), H, HUSED, RCAP, EPAR(4), S0, S1
      INTEGER :: N
      LOGICAL :: OK

      RCAP = INNER_BOUNDARY(MID, PAR) + CAPTURE_BUFFER
      EPAR(1:3) = NHAT
      EPAR(4) = DPLANE
      Y = Y0
      YOUT = Y0
      H = H0
      STATUS = 3
      NSTEPS = 0
      S0 = EVENT_VALUE(EVENT_PLANE, EPAR, Y)

      DO N = 1, MAXSTEP
         CALL ADVANCE_ADAPTIVE(METHOD, MID, PAR, PT, PPHI, RTOL, ATOL, &
                               Y, H, Y1, HUSED, OK)
         IF (.NOT. OK) RETURN
         NSTEPS = NSTEPS + 1

         S1 = EVENT_VALUE(EVENT_PLANE, EPAR, Y1)
         IF (S0*S1 < 0.0_WP) THEN
            STATUS = 2
            CALL REFINE_EVENT(METHOD, MID, PAR, PT, PPHI, Y, HUSED, &
                              EVENT_PLANE, EPAR, Y1, YOUT)
            RETURN
         END IF

         IF (Y1(2) <= RCAP .OR. Y1(2) <= 0.0_WP) THEN
            STATUS = 1
            YOUT = Y1
            RETURN
         END IF
         IF (Y1(2) >= RMAX) THEN
            STATUS = 0
            YOUT = Y1
            RETURN
         END IF

         Y = Y1
         S0 = S1
      END DO
   END SUBROUTINE TRACE_TO_PLANE

   SUBROUTINE TRACE_BUNDLE_TO_PLANE(MID, PAR, PTS, PPHIS, Y0S, NRAY, &
                                    METHOD, RTOL, ATOL, H0, NHAT, DPLANE, &
                                    RMAX, MAXSTEP, STATUS, YOUTS)
      ! OpenMP fan-out of TRACE_TO_PLANE over a bundle of rays.
      INTEGER, INTENT(IN)   :: MID, NRAY, METHOD, MAXSTEP
      REAL(WP), INTENT(IN)  :: PAR(4), PTS(NRAY), PPHIS(NRAY), Y0S(NRAY, 6)
      REAL(WP), INTENT(IN)  :: RTOL, ATOL, H0, NHAT(3), DPLANE, RMAX
      INTEGER, INTENT(OUT)  :: STATUS(NRAY)
      REAL(WP), INTENT(OUT) :: YOUTS(NRAY, 6)
      REAL(WP) :: YOUT(6)
      INTEGER :: I, ST, NS

      !$OMP PARALLEL DO SCHEDULE(DYNAMIC, 16) DEFAULT(NONE) &
      !$OMP SHARED(MID, PAR, PTS, PPHIS, Y0S, NRAY, METHOD, RTOL, ATOL) &
      !$OMP SHARED(H0, NHAT, DPLANE, RMAX, MAXSTEP, STATUS, YOUTS) &
      !$OMP PRIVATE(I, ST, NS, YOUT)
      DO I = 1, NRAY
         CALL TRACE_TO_PLANE(MID, PAR, PTS(I), PPHIS(I), Y0S(I, :), &
                             METHOD, RTOL, ATOL, H0, NHAT, DPLANE, RMAX, &
                             MAXSTEP, ST, YOUT, NS)
         STATUS(I) = ST
         YOUTS(I, :) = YOUT
      END DO
      !$OMP END PARALLEL DO
   END SUBROUTINE TRACE_BUNDLE_TO_PLANE

   SUBROUTINE TRACE_GEODESIC(MID, PAR, Y0, PT, PPHI, METHOD, RTOL, ATOL, &
                             H0, LAM_MAX, NMAX, TRAJ, HERR, NOUT)
      ! Integrate a single geodesic (photon or massive particle) from
      ! raw initial conditions, recording the trajectory and |H| at
      ! every accepted step. TRAJ(K, :) = (lambda, t, r, theta, phi,
      ! p_r, p_theta).
      INTEGER, INTENT(IN)   :: MID, METHOD, NMAX
      REAL(WP), INTENT(IN)  :: PAR(4), Y0(6), PT, PPHI, RTOL, ATOL
      REAL(WP), INTENT(IN)  :: H0, LAM_MAX
      REAL(WP), INTENT(OUT) :: TRAJ(NMAX, 7), HERR(NMAX)
      INTEGER, INTENT(OUT)  :: NOUT
      REAL(WP) :: Y(6), Y1(6), H, HUSED, LAM, RCAP
      LOGICAL :: OK

      RCAP = INNER_BOUNDARY(MID, PAR) + CAPTURE_BUFFER
      Y = Y0
      H = H0
      LAM = 0.0_WP
      NOUT = 0
      TRAJ = 0.0_WP
      HERR = 0.0_WP

      DO
         IF (NOUT >= NMAX) EXIT
         IF (ABS(LAM) >= ABS(LAM_MAX)) EXIT
         CALL ADVANCE_ADAPTIVE(METHOD, MID, PAR, PT, PPHI, RTOL, ATOL, &
                               Y, H, Y1, HUSED, OK)
         IF (.NOT. OK) EXIT

         LAM = LAM + HUSED
         Y = Y1
         NOUT = NOUT + 1
         TRAJ(NOUT, 1) = LAM
         TRAJ(NOUT, 2:7) = Y
         HERR(NOUT) = ABS(HAMILTONIAN_CONSTRAINT(MID, PAR, PT, PPHI, Y))

         IF (Y(2) <= RCAP) EXIT
      END DO
   END SUBROUTINE TRACE_GEODESIC

   ! --------------------------------------------------------- render

   SUBROUTINE RENDER_IMAGE(MID, PAR, R0, THETA0, PHI0, XMIN, XMAX, YMIN, &
                           YMAX, NX, NY, METHOD, RTOL, ATOL, DISK_ON, &
                           RIN_IN, ROUT, L0, MAXSTEP, INTENS, GMAP, RHIT, &
                           STATUS, HERRM, THF, PHF)
      ! Render the full image plane. Pixel (I, J) maps to
      !   x = XMIN + (XMAX-XMIN)*(I-1/2)/NX,  y = YMIN + (YMAX-YMIN)*(J-1/2)/NY.
      ! The thin-disk model (Page-Thorne flux, redshift) is Kerr-only:
      ! with DISK_ON = 1 the caller must pass MID = MID_KERR (enforced
      ! by the Python layer; PAR(1) is then the spin).
      ! Outputs:
      !   INTENS : observed intensity g^3 * F_PageThorne (disk hits, else 0)
      !   GMAP   : redshift factor g (disk hits, else 0)
      !   RHIT   : radius of the disk hit (else 0)
      !   STATUS : ray status code per pixel
      !   HERRM  : max |H| along each ray (constraint error)
      !   THF/PHF: final theta, phi for escaped rays (celestial sphere)
      INTEGER, INTENT(IN)  :: MID, NX, NY, METHOD, DISK_ON, MAXSTEP
      REAL(WP), INTENT(IN) :: PAR(4), R0, THETA0, PHI0
      REAL(WP), INTENT(IN) :: XMIN, XMAX, YMIN, YMAX
      REAL(WP), INTENT(IN) :: RTOL, ATOL, RIN_IN, ROUT, L0
      REAL(WP), INTENT(OUT) :: INTENS(NX, NY), GMAP(NX, NY), RHIT(NX, NY)
      REAL(WP), INTENT(OUT) :: HERRM(NX, NY), THF(NX, NY), PHF(NX, NY)
      INTEGER, INTENT(OUT)  :: STATUS(NX, NY)
      REAL(WP) :: RIN, R_ESC, X, YY, Y0(6), YOUT(6), PT, PPHI, HE, G
      INTEGER :: I, J, ST, NS

      RIN = RIN_IN
      IF (RIN <= 0.0_WP) RIN = ISCO_RADIUS(PAR(1))
      IF (DISK_ON == 1) CALL INIT_FLUX_TABLE(PAR(1), ROUT, 4000)
      R_ESC = 1.1_WP*R0

      INTENS = 0.0_WP
      GMAP = 0.0_WP
      RHIT = 0.0_WP
      THF = 0.0_WP
      PHF = 0.0_WP

      !$OMP PARALLEL DO COLLAPSE(2) SCHEDULE(DYNAMIC, 4) DEFAULT(NONE) &
      !$OMP SHARED(MID, PAR, R0, THETA0, PHI0, XMIN, XMAX, YMIN, YMAX) &
      !$OMP SHARED(NX, NY, METHOD, RTOL, ATOL, DISK_ON, RIN, ROUT, L0) &
      !$OMP SHARED(MAXSTEP, R_ESC, INTENS, GMAP, RHIT, STATUS, HERRM, THF, PHF) &
      !$OMP PRIVATE(I, J, X, YY, Y0, YOUT, PT, PPHI, HE, G, ST, NS)
      DO J = 1, NY
         DO I = 1, NX
            X = XMIN + (XMAX - XMIN)*(REAL(I, WP) - 0.5_WP)/REAL(NX, WP)
            YY = YMIN + (YMAX - YMIN)*(REAL(J, WP) - 0.5_WP)/REAL(NY, WP)
            CALL CAMERA_INIT(MID, PAR, R0, THETA0, PHI0, X, YY, Y0, PT, PPHI)
            CALL TRACE_RAY(MID, PAR, PT, PPHI, Y0, METHOD, RTOL, ATOL, &
                           R_ESC, DISK_ON, RIN, ROUT, MAXSTEP, ST, YOUT, &
                           HE, NS)
            STATUS(I, J) = ST
            HERRM(I, J) = HE
            IF (ST == 2) THEN
               G = REDSHIFT_FACTOR(PAR(1), YOUT(2), PT, PPHI, L0)
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

END MODULE RAYTRACER
