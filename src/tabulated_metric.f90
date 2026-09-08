MODULE TABULATED_METRIC
   ! A read-only table during each OpenMP ray batch. Python serializes table
   ! installation and tracing. Bicubic Hermite interpolation is C1; forces
   ! are derivatives of precisely the same interpolant used for the metric.
   USE KIND_PARAMS, ONLY: WP
   USE, INTRINSIC :: IEEE_ARITHMETIC, ONLY: IEEE_VALUE, IEEE_QUIET_NAN
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: SET_METRIC_TABLE, TABLE_CONTRA, TABLE_COV
   REAL(WP), ALLOCATABLE :: TAB(:, :, :, :)
   REAL(WP) :: X0, DX, DMU, ORIGIN
   INTEGER :: NR = 0, NM = 0
CONTAINS
   SUBROUTINE SET_METRIC_TABLE(DATA, NX, NY, XMIN, XSTEP, RORIGIN)
      INTEGER, INTENT(IN) :: NX, NY
      REAL(WP), INTENT(IN) :: DATA(5, 4, NX, NY), XMIN, XSTEP, RORIGIN
      IF (ALLOCATED(TAB)) DEALLOCATE(TAB)
      ALLOCATE(TAB(5, 4, NX, NY))
      TAB = DATA
      NR = NX
      NM = NY
      X0 = XMIN
      DX = XSTEP
      DMU = 2.0_WP/REAL(NY-1, WP)
      ORIGIN = RORIGIN
   END SUBROUTINE

   PURE SUBROUTINE BASIS(S, H, DH)
      REAL(WP), INTENT(IN) :: S
      REAL(WP), INTENT(OUT) :: H(4), DH(4)
      H = (/2*S**3-3*S**2+1, -2*S**3+3*S**2, S**3-2*S**2+S, S**3-S**2/)
      DH = (/6*S**2-6*S, -6*S**2+6*S, 3*S**2-4*S+1, 3*S**2-2*S/)
   END SUBROUTINE

   PURE SUBROUTINE TABLE_CONTRA(R, TH, GU, DR, DT)
      REAL(WP), INTENT(IN) :: R, TH
      REAL(WP), INTENT(OUT) :: GU(5), DR(5), DT(5)
      REAL(WP) :: X, U, HX(4), HY(4), DHX(4), DHY(4), F(5), FX(5), FY(5)
      REAL(WP) :: C(5), SCALE(5), S, CT
      INTEGER :: I, J, A, B, IA, JB, K
      GU = IEEE_VALUE(0.0_WP, IEEE_QUIET_NAN)
      DR = GU
      DT = GU
      IF (NR < 2 .OR. R <= ORIGIN) RETURN
      X = (LOG(R-ORIGIN)-X0)/DX
      IF (.NOT. (X >= 0.0_WP .AND. X <= REAL(NR-1, WP))) RETURN
      U = (COS(TH)+1.0_WP)/DMU
      I = MIN(NR-1, INT(X)+1)
      J = MIN(NM-1, MAX(1, INT(U)+1))
      CALL BASIS(X-REAL(I-1, WP), HX, DHX)
      CALL BASIS(U-REAL(J-1, WP), HY, DHY)
      F = 0
      FX = 0
      FY = 0
      DO A = 1, 4
         IA = I + MOD(A-1, 2)
         DO B = 1, 4
            JB = J + MOD(B-1, 2)
            K = 1
            IF (A > 2) K = K+1
            IF (B > 2) K = K+2
            C = TAB(:, K, IA, JB)
            F = F + C*HX(A)*HY(B)
            FX = FX + C*DHX(A)*HY(B)/DX
            FY = FY + C*HX(A)*DHY(B)/DMU
         END DO
      END DO
      ! Tables remove the spherical-coordinate poles analytically.
      S = SIN(TH)
      CT = COS(TH)
      SCALE = (/1.0_WP, 1.0_WP, 1.0_WP, R*R, R*R*S*S/)
      GU = F/SCALE
      DR = FX/((R-ORIGIN)*SCALE)
      DT = -S*FY/SCALE
      DR(4:5) = DR(4:5)-2*GU(4:5)/R
      DT(5) = DT(5)-2*CT*GU(5)/S
   END SUBROUTINE

   PURE SUBROUTINE TABLE_COV(R, TH, GD)
      REAL(WP), INTENT(IN) :: R, TH
      REAL(WP), INTENT(OUT) :: GD(5)
      REAL(WP) :: GU(5), DR(5), DT(5), DET
      CALL TABLE_CONTRA(R, TH, GU, DR, DT)
      DET = GU(1)*GU(5)-GU(2)**2
      GD = (/GU(5)/DET, -GU(2)/DET, 1/GU(3), 1/GU(4), GU(1)/DET/)
   END SUBROUTINE
END MODULE
