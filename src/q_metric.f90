MODULE Q_METRIC
   ! q-metric (Zipoy-Voorhees / Quevedo) in spherical-like coordinates
   ! {t, r, theta, phi} -- the simplest static, axially symmetric vacuum
   ! solution with a quadrupole (Appendix A of arXiv:2202.00086; the
   ! appendix rendering has a typographic sign slip in the spatial
   ! block; we use the standard form, which reduces to Schwarzschild
   ! for q = 0):
   !
   !   ds^2 = -f^(1+q) dt^2
   !          + f^(-q) [ h^(-q(2+q)) ( f^(-1) dr^2 + r^2 dtheta^2 )
   !                     + r^2 sin^2(theta) dphi^2 ]
   !   f = 1 - 2m/r,   h = 1 + m^2 sin^2(theta)/(r^2 - 2mr),   m = 1.
   !
   ! r = 2m is a curvature singularity for q /= 0 (naked for a range of
   ! q); rays are captured slightly outside it. f and h are clamped to
   ! a tiny positive floor so fractional powers never produce NaNs when
   ! an integrator stage overshoots the boundary.
   ! Component ordering matches KERR_METRIC:
   !    (1) tt, (2) t-phi (= 0 here), (3) rr, (4) theta-theta, (5) phi-phi
   USE KIND_PARAMS, ONLY: WP
   IMPLICIT NONE
   PRIVATE

   PUBLIC :: QMETRIC_COV, QMETRIC_CONTRA

   REAL(WP), PARAMETER :: FLOOR = 1.0E-14_WP

CONTAINS

   PURE SUBROUTINE QMETRIC_COV(Q, R, THETA, GD)
      REAL(WP), INTENT(IN)  :: Q, R, THETA
      REAL(WP), INTENT(OUT) :: GD(5)
      REAL(WP) :: F, H, S2, ALPHA

      S2 = SIN(THETA)**2
      F = MAX(1.0_WP - 2.0_WP/R, FLOOR)
      H = MAX(1.0_WP + S2/(R*R - 2.0_WP*R), FLOOR)
      ALPHA = Q*(2.0_WP + Q)

      GD(1) = -F**(1.0_WP + Q)
      GD(2) = 0.0_WP
      GD(3) = F**(-(1.0_WP + Q))*H**(-ALPHA)
      GD(4) = R*R*F**(-Q)*H**(-ALPHA)
      GD(5) = R*R*S2*F**(-Q)
   END SUBROUTINE QMETRIC_COV

   PURE SUBROUTINE QMETRIC_CONTRA(Q, R, THETA, GU, DGU_DR, DGU_DTH)
      ! Contravariant components and derivatives via logarithmic
      ! derivatives: for X = f^a h^b r^c s^d,
      !   dX/dr = X (a f'/f + b h_r/h + c/r),
      !   dX/dth = X (b h_th/h + d cos/sin).
      REAL(WP), INTENT(IN)  :: Q, R, THETA
      REAL(WP), INTENT(OUT) :: GU(5), DGU_DR(5), DGU_DTH(5)
      REAL(WP) :: F, H, S, C, S2, U, ALPHA
      REAL(WP) :: FLR, HLR, HLT

      S = SIN(THETA)
      C = COS(THETA)
      S2 = S*S
      U = R*R - 2.0_WP*R
      F = MAX(1.0_WP - 2.0_WP/R, FLOOR)
      H = MAX(1.0_WP + S2/U, FLOOR)
      ALPHA = Q*(2.0_WP + Q)

      ! logarithmic derivatives f'/f, h_r/h, h_th/h
      FLR = (2.0_WP/(R*R))/F
      HLR = (-S2*(2.0_WP*R - 2.0_WP)/(U*U))/H
      HLT = (2.0_WP*S*C/U)/H

      ! g^tt = -f^{-(1+q)}
      GU(1) = -F**(-(1.0_WP + Q))
      DGU_DR(1) = GU(1)*(-(1.0_WP + Q)*FLR)
      DGU_DTH(1) = 0.0_WP

      ! g^{t phi} = 0
      GU(2) = 0.0_WP
      DGU_DR(2) = 0.0_WP
      DGU_DTH(2) = 0.0_WP

      ! g^rr = f^{1+q} h^{alpha}
      GU(3) = F**(1.0_WP + Q)*H**ALPHA
      DGU_DR(3) = GU(3)*((1.0_WP + Q)*FLR + ALPHA*HLR)
      DGU_DTH(3) = GU(3)*(ALPHA*HLT)

      ! g^{theta theta} = f^q h^{alpha} / r^2
      GU(4) = F**Q*H**ALPHA/(R*R)
      DGU_DR(4) = GU(4)*(Q*FLR + ALPHA*HLR - 2.0_WP/R)
      DGU_DTH(4) = GU(4)*(ALPHA*HLT)

      ! g^{phi phi} = f^q / (r^2 sin^2)
      GU(5) = F**Q/(R*R*S2)
      DGU_DR(5) = GU(5)*(Q*FLR - 2.0_WP/R)
      DGU_DTH(5) = GU(5)*(-2.0_WP*C/S)
   END SUBROUTINE QMETRIC_CONTRA

END MODULE Q_METRIC
