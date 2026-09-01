MODULE GEODESIC_EOM
   ! Hamiltonian equations of motion for geodesics in a stationary,
   ! axisymmetric spacetime: H = (1/2) g^{mu nu} p_mu p_nu.
   ! State vector Y = (t, r, theta, phi, p_r, p_theta);
   ! p_t and p_phi are conserved and passed as parameters. The
   ! spacetime is selected by (MID, PAR) -- see module SPACETIME.
   ! The equations hold for photons (H = 0) and massive particles
   ! (H = -m^2/2) alike.
   USE KIND_PARAMS, ONLY: WP
   USE SPACETIME, ONLY: METRIC_CONTRA
   IMPLICIT NONE
   PRIVATE

   PUBLIC :: GEODESIC_RHS, HAMILTONIAN_CONSTRAINT

CONTAINS

   PURE SUBROUTINE GEODESIC_RHS(MID, PAR, PT, PPHI, Y, DY)
      ! Right-hand side of Hamilton's equations (eqs. 2-3 of arXiv:2202.00086).
      INTEGER, INTENT(IN)   :: MID
      REAL(WP), INTENT(IN)  :: PAR(4), PT, PPHI, Y(6)
      REAL(WP), INTENT(OUT) :: DY(6)
      REAL(WP) :: GU(5), DGR(5), DGT(5)

      CALL METRIC_CONTRA(MID, PAR, Y(2), Y(3), GU, DGR, DGT)

      DY(1) = GU(1)*PT + GU(2)*PPHI
      DY(2) = GU(3)*Y(5)
      DY(3) = GU(4)*Y(6)
      DY(4) = GU(2)*PT + GU(5)*PPHI
      DY(5) = -0.5_WP*(PT*PT*DGR(1) + 2.0_WP*PT*PPHI*DGR(2) + &
              Y(5)*Y(5)*DGR(3) + Y(6)*Y(6)*DGR(4) + PPHI*PPHI*DGR(5))
      DY(6) = -0.5_WP*(PT*PT*DGT(1) + 2.0_WP*PT*PPHI*DGT(2) + &
              Y(5)*Y(5)*DGT(3) + Y(6)*Y(6)*DGT(4) + PPHI*PPHI*DGT(5))
   END SUBROUTINE GEODESIC_RHS

   PURE FUNCTION HAMILTONIAN_CONSTRAINT(MID, PAR, PT, PPHI, Y) RESULT(H)
      ! H stays zero along a null geodesic (eq. 3); its numerical value
      ! is the error monitor of eq. (12). For time-like geodesics it
      ! stays at -1/2 (unit mass) and the *drift* is the monitor.
      INTEGER, INTENT(IN)  :: MID
      REAL(WP), INTENT(IN) :: PAR(4), PT, PPHI, Y(6)
      REAL(WP) :: H
      REAL(WP) :: GU(5), DGR(5), DGT(5)

      CALL METRIC_CONTRA(MID, PAR, Y(2), Y(3), GU, DGR, DGT)
      H = 0.5_WP*(GU(1)*PT*PT + 2.0_WP*GU(2)*PT*PPHI + &
          GU(3)*Y(5)*Y(5) + GU(4)*Y(6)*Y(6) + GU(5)*PPHI*PPHI)
   END FUNCTION HAMILTONIAN_CONSTRAINT

END MODULE GEODESIC_EOM
