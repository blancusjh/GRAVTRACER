MODULE DISK_MODEL
   ! Thin equatorial accretion disk (Section 5 of arXiv:2202.00086).
   !
   ! Kinematics: circular orbits with constant specific angular momentum
   ! l0 = -u_phi/u_t, giving the angular velocity of eq. (18).
   ! NOTE: eq. (18) as printed reads -(g_tp + g_pp*l0)/(g_pp + g_tp*l0),
   ! which yields Omega = -l0 in the Schwarzschild limit (superluminal).
   ! The correct expression for l0 = -u_phi/u_t, implemented here, is
   !    Omega = -(g_tp + g_tt*l0)/(g_pp + g_tp*l0).
   !
   ! Emission: time-averaged flux of Page & Thorne (1974) [ref. 30 of the
   ! paper], computed from the general quadrature
   !    F(r) = -Mdot/(4 pi r) * Omega'/(E - Omega*L)^2
   !           * INT_{r_isco}^{r} (E - Omega*L) L' dr
   ! with Keplerian E(r), L(r), Omega(r), evaluated once on a fine radial
   ! grid (cumulative trapezoid) and interpolated during rendering.
   ! Mdot/(4 pi) is set to 1; images are normalized downstream.
   USE KIND_PARAMS, ONLY: WP, PI, HALF_PI
   USE KERR_METRIC, ONLY: METRIC_COV, ISCO_RADIUS
   IMPLICIT NONE
   PRIVATE

   PUBLIC :: INIT_FLUX_TABLE, PT_FLUX, DISK_OMEGA, REDSHIFT_FACTOR, FLUX_PROFILE
   PUBLIC :: GET_DISK_OMEGA, GET_REDSHIFT

   INTEGER :: TAB_N = 0
   REAL(WP) :: TAB_R1 = 0.0_WP, TAB_R2 = 0.0_WP, TAB_DR = 1.0_WP
   REAL(WP), ALLOCATABLE :: TAB_F(:)

CONTAINS

   SUBROUTINE INIT_FLUX_TABLE(A, ROUT, N)
      ! Build the Page-Thorne flux table on [r_isco, ROUT] with N points.
      ! Must be called (serially) before PT_FLUX is used.
      REAL(WP), INTENT(IN) :: A, ROUT
      INTEGER, INTENT(IN)  :: N
      REAL(WP), ALLOCATABLE :: RG(:), EK(:), LK(:), OMK(:), DL(:), DOM(:), CI(:)
      REAL(WP) :: RISCO, X, DEN
      INTEGER :: I

      RISCO = ISCO_RADIUS(A)
      TAB_N = N
      TAB_R1 = RISCO
      TAB_R2 = ROUT
      TAB_DR = (TAB_R2 - TAB_R1)/REAL(N - 1, WP)

      ALLOCATE (RG(N), EK(N), LK(N), OMK(N), DL(N), DOM(N), CI(N))
      IF (ALLOCATED(TAB_F)) DEALLOCATE (TAB_F)
      ALLOCATE (TAB_F(N))

      DO I = 1, N
         RG(I) = TAB_R1 + TAB_DR*REAL(I - 1, WP)
         X = SQRT(RG(I))
         DEN = X*SQRT(X*X*X*X*X*X - 3.0_WP*X*X*X*X + 2.0_WP*A*X*X*X)
         ! Keplerian (prograde) circular geodesics in Kerr, M = 1:
         OMK(I) = 1.0_WP/(X*X*X + A)
         EK(I) = (X*X*X*X - 2.0_WP*X*X + A*X)/DEN
         LK(I) = (X*X*X*X*X - 2.0_WP*A*X*X + A*A*X)/DEN
      END DO

      ! Centered differences for L'(r) and Omega'(r)
      DO I = 2, N - 1
         DL(I) = (LK(I + 1) - LK(I - 1))/(2.0_WP*TAB_DR)
         DOM(I) = (OMK(I + 1) - OMK(I - 1))/(2.0_WP*TAB_DR)
      END DO
      DL(1) = (LK(2) - LK(1))/TAB_DR
      DL(N) = (LK(N) - LK(N - 1))/TAB_DR
      DOM(1) = (OMK(2) - OMK(1))/TAB_DR
      DOM(N) = (OMK(N) - OMK(N - 1))/TAB_DR

      ! Cumulative trapezoid of (E - Omega*L) L' from r_isco
      CI(1) = 0.0_WP
      DO I = 2, N
         CI(I) = CI(I - 1) + 0.5_WP*TAB_DR* &
                 ((EK(I) - OMK(I)*LK(I))*DL(I) + &
                  (EK(I - 1) - OMK(I - 1)*LK(I - 1))*DL(I - 1))
      END DO

      DO I = 1, N
         TAB_F(I) = -DOM(I)/(RG(I)*(EK(I) - OMK(I)*LK(I))**2)*CI(I)
      END DO
      TAB_F(1) = 0.0_WP

      DEALLOCATE (RG, EK, LK, OMK, DL, DOM, CI)
   END SUBROUTINE INIT_FLUX_TABLE

   PURE FUNCTION PT_FLUX(R) RESULT(F)
      ! Linear interpolation in the precomputed flux table.
      REAL(WP), INTENT(IN) :: R
      REAL(WP) :: F, S
      INTEGER :: I

      F = 0.0_WP
      IF (TAB_N < 2) RETURN
      IF (R <= TAB_R1 .OR. R >= TAB_R2) THEN
         IF (R >= TAB_R2) F = TAB_F(TAB_N)
         RETURN
      END IF
      S = (R - TAB_R1)/TAB_DR
      I = INT(S) + 1
      IF (I >= TAB_N) I = TAB_N - 1
      S = S - REAL(I - 1, WP)
      F = (1.0_WP - S)*TAB_F(I) + S*TAB_F(I + 1)
   END FUNCTION PT_FLUX

   PURE FUNCTION DISK_OMEGA(A, R, L0) RESULT(OM)
      ! Angular velocity of disk matter with constant specific angular
      ! momentum l0 (eq. 18, with the numerator typo corrected).
      REAL(WP), INTENT(IN) :: A, R, L0
      REAL(WP) :: OM, GD(5)
      CALL METRIC_COV(A, R, HALF_PI, GD)
      OM = -(GD(2) + GD(1)*L0)/(GD(5) + GD(2)*L0)
   END FUNCTION DISK_OMEGA

   PURE FUNCTION REDSHIFT_FACTOR(A, R, PT, PPHI, L0) RESULT(G)
      ! g = nu_obs/nu_em for a photon with conserved (p_t, p_phi) hitting
      ! disk matter at radius r (eqs. 20-22; the observed energy is
      ! normalized to 1 by the camera setup, P^t = 1).
      REAL(WP), INTENT(IN) :: A, R, PT, PPHI, L0
      REAL(WP) :: G, GD(5), OM, UU
      CALL METRIC_COV(A, R, HALF_PI, GD)
      OM = -(GD(2) + GD(1)*L0)/(GD(5) + GD(2)*L0)
      UU = -GD(1) - 2.0_WP*OM*GD(2) - OM*OM*GD(5)
      G = SQRT(MAX(UU, 0.0_WP))/(PT + OM*PPHI)
   END FUNCTION REDSHIFT_FACTOR

   SUBROUTINE FLUX_PROFILE(A, ROUT, N, RS, FS)
      ! Convenience wrapper returning the radial flux profile (for tests
      ! and plots); leaves the module table initialized.
      REAL(WP), INTENT(IN) :: A, ROUT
      INTEGER, INTENT(IN)  :: N
      REAL(WP), INTENT(OUT) :: RS(N), FS(N)
      INTEGER :: I
      CALL INIT_FLUX_TABLE(A, ROUT, N)
      DO I = 1, N
         RS(I) = TAB_R1 + TAB_DR*REAL(I - 1, WP)
         FS(I) = TAB_F(I)
      END DO
   END SUBROUTINE FLUX_PROFILE

   SUBROUTINE GET_DISK_OMEGA(A, R, L0, OM)
      ! f2py-friendly wrapper around DISK_OMEGA.
      REAL(WP), INTENT(IN)  :: A, R, L0
      REAL(WP), INTENT(OUT) :: OM
      OM = DISK_OMEGA(A, R, L0)
   END SUBROUTINE GET_DISK_OMEGA

   SUBROUTINE GET_REDSHIFT(A, R, PT, PPHI, L0, G)
      ! f2py-friendly wrapper around REDSHIFT_FACTOR.
      REAL(WP), INTENT(IN)  :: A, R, PT, PPHI, L0
      REAL(WP), INTENT(OUT) :: G
      G = REDSHIFT_FACTOR(A, R, PT, PPHI, L0)
   END SUBROUTINE GET_REDSHIFT

END MODULE DISK_MODEL
