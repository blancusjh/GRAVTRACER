MODULE KERR_METRIC
   ! Kerr metric in Boyer-Lindquist coordinates {t, r, theta, phi}.
   ! Geometrized units G = c = M = 1, signature (-,+,+,+).
   ! Component ordering used throughout (covariant and contravariant):
   !    (1) tt, (2) t-phi, (3) rr, (4) theta-theta, (5) phi-phi
   USE KIND_PARAMS, ONLY: WP
   IMPLICIT NONE
   PRIVATE

   PUBLIC :: METRIC_COV, METRIC_CONTRA, HORIZON_RADIUS, ISCO_RADIUS

CONTAINS

   PURE SUBROUTINE METRIC_COV(A, R, THETA, GD)
      ! Covariant components g_{mu nu}, eq. (15) of arXiv:2202.00086.
      REAL(WP), INTENT(IN)  :: A, R, THETA
      REAL(WP), INTENT(OUT) :: GD(5)
      REAL(WP) :: S2, SIGMA, DELTA

      S2 = SIN(THETA)**2
      SIGMA = R*R + A*A*COS(THETA)**2
      DELTA = R*R - 2.0_WP*R + A*A

      GD(1) = -(1.0_WP - 2.0_WP*R/SIGMA)
      GD(2) = -2.0_WP*A*R*S2/SIGMA
      GD(3) = SIGMA/DELTA
      GD(4) = SIGMA
      GD(5) = S2*(R*R + A*A + 2.0_WP*A*A*R*S2/SIGMA)
   END SUBROUTINE METRIC_COV

   PURE SUBROUTINE METRIC_CONTRA(A, R, THETA, GU, DGU_DR, DGU_DTH)
      ! Contravariant components g^{mu nu} and their r- and theta-derivatives,
      ! as needed by the Hamiltonian equations of motion.
      REAL(WP), INTENT(IN)  :: A, R, THETA
      REAL(WP), INTENT(OUT) :: GU(5), DGU_DR(5), DGU_DTH(5)
      REAL(WP) :: S, C, S2, SIN2T, SIGMA, DELTA, AF, D, W, N
      REAL(WP) :: DSIG_R, DSIG_TH, DDEL_R, DAF_R, DAF_TH
      REAL(WP) :: DD_R, DD_TH, DW_R, DW_TH, DN_R, DN_TH

      S = SIN(THETA)
      C = COS(THETA)
      S2 = S*S
      SIN2T = 2.0_WP*S*C

      SIGMA = R*R + A*A*C*C
      DELTA = R*R - 2.0_WP*R + A*A
      AF = (R*R + A*A)**2 - A*A*DELTA*S2

      DSIG_R = 2.0_WP*R
      DSIG_TH = -A*A*SIN2T
      DDEL_R = 2.0_WP*R - 2.0_WP
      DAF_R = 4.0_WP*R*(R*R + A*A) - A*A*S2*DDEL_R
      DAF_TH = -A*A*DELTA*SIN2T

      ! D = Sigma*Delta appears in the denominators of the t-phi block
      D = SIGMA*DELTA
      DD_R = DSIG_R*DELTA + SIGMA*DDEL_R
      DD_TH = DSIG_TH*DELTA

      ! g^tt = -AF/(Sigma*Delta)
      GU(1) = -AF/D
      DGU_DR(1) = -(DAF_R*D - AF*DD_R)/(D*D)
      DGU_DTH(1) = -(DAF_TH*D - AF*DD_TH)/(D*D)

      ! g^{t phi} = -2*a*r/(Sigma*Delta)
      GU(2) = -2.0_WP*A*R/D
      DGU_DR(2) = -2.0_WP*A*(D - R*DD_R)/(D*D)
      DGU_DTH(2) = 2.0_WP*A*R*DD_TH/(D*D)

      ! g^rr = Delta/Sigma
      GU(3) = DELTA/SIGMA
      DGU_DR(3) = (DDEL_R*SIGMA - DELTA*DSIG_R)/(SIGMA*SIGMA)
      DGU_DTH(3) = -DELTA*DSIG_TH/(SIGMA*SIGMA)

      ! g^{theta theta} = 1/Sigma
      GU(4) = 1.0_WP/SIGMA
      DGU_DR(4) = -DSIG_R/(SIGMA*SIGMA)
      DGU_DTH(4) = -DSIG_TH/(SIGMA*SIGMA)

      ! g^{phi phi} = (Delta - a^2 sin^2)/(Sigma*Delta*sin^2)
      N = DELTA - A*A*S2
      W = D*S2
      DN_R = DDEL_R
      DN_TH = -A*A*SIN2T
      DW_R = DD_R*S2
      DW_TH = DD_TH*S2 + D*SIN2T
      GU(5) = N/W
      DGU_DR(5) = (DN_R*W - N*DW_R)/(W*W)
      DGU_DTH(5) = (DN_TH*W - N*DW_TH)/(W*W)
   END SUBROUTINE METRIC_CONTRA

   PURE FUNCTION HORIZON_RADIUS(A) RESULT(RH)
      ! Outer event horizon, eq. (17): r_H = 1 + sqrt(1 - a^2).
      REAL(WP), INTENT(IN) :: A
      REAL(WP) :: RH
      RH = 1.0_WP + SQRT(MAX(1.0_WP - A*A, 0.0_WP))
   END FUNCTION HORIZON_RADIUS

   PURE FUNCTION ISCO_RADIUS(A) RESULT(RISCO)
      ! Prograde innermost stable circular orbit (Bardeen, Press & Teukolsky 1972).
      REAL(WP), INTENT(IN) :: A
      REAL(WP) :: RISCO, Z1, Z2
      Z1 = 1.0_WP + (1.0_WP - A*A)**(1.0_WP/3.0_WP) * &
           ((1.0_WP + A)**(1.0_WP/3.0_WP) + (1.0_WP - A)**(1.0_WP/3.0_WP))
      Z2 = SQRT(3.0_WP*A*A + Z1*Z1)
      RISCO = 3.0_WP + Z2 - SQRT((3.0_WP - Z1)*(3.0_WP + Z1 + 2.0_WP*Z2))
   END FUNCTION ISCO_RADIUS

END MODULE KERR_METRIC
