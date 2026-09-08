MODULE SPACETIME
   ! Spacetime dispatcher: one entry point for metric evaluation, keyed
   ! by a metric id MID and a parameter vector PAR(4).
   !
   !   MID_KERR    = 1 : Kerr in Boyer-Lindquist, PAR(1) = spin a
   !   MID_QMETRIC = 2 : q-metric (Zipoy-Voorhees), PAR(1) = quadrupole q
   !
   ! Everything downstream (equations of motion, integrators, tracers)
   ! is metric-agnostic and reaches the geometry only through this
   ! module. Adding a spacetime = one new module + one CASE line here.
   USE KIND_PARAMS, ONLY: WP
   USE KERR_METRIC, ONLY: KERR_COV => METRIC_COV, &
                          KERR_CONTRA => METRIC_CONTRA, HORIZON_RADIUS
   USE Q_METRIC, ONLY: QMETRIC_COV, QMETRIC_CONTRA
   USE TABULATED_METRIC, ONLY: TABLE_COV, TABLE_CONTRA
   IMPLICIT NONE
   PRIVATE

   PUBLIC :: METRIC_COV, METRIC_CONTRA, INNER_BOUNDARY
   INTEGER, PARAMETER, PUBLIC :: MID_KERR = 1
   INTEGER, PARAMETER, PUBLIC :: MID_QMETRIC = 2

CONTAINS

   PURE SUBROUTINE METRIC_COV(MID, PAR, R, THETA, GD)
      INTEGER, INTENT(IN)   :: MID
      REAL(WP), INTENT(IN)  :: PAR(4), R, THETA
      REAL(WP), INTENT(OUT) :: GD(5)
      SELECT CASE (MID)
      CASE (3)
         CALL TABLE_COV(R, THETA, GD)
      CASE (MID_QMETRIC)
         CALL QMETRIC_COV(PAR(1), R, THETA, GD)
      CASE DEFAULT
         CALL KERR_COV(PAR(1), R, THETA, GD)
      END SELECT
   END SUBROUTINE METRIC_COV

   PURE SUBROUTINE METRIC_CONTRA(MID, PAR, R, THETA, GU, DGU_DR, DGU_DTH)
      INTEGER, INTENT(IN)   :: MID
      REAL(WP), INTENT(IN)  :: PAR(4), R, THETA
      REAL(WP), INTENT(OUT) :: GU(5), DGU_DR(5), DGU_DTH(5)
      SELECT CASE (MID)
      CASE (3)
         CALL TABLE_CONTRA(R, THETA, GU, DGU_DR, DGU_DTH)
      CASE (MID_QMETRIC)
         CALL QMETRIC_CONTRA(PAR(1), R, THETA, GU, DGU_DR, DGU_DTH)
      CASE DEFAULT
         CALL KERR_CONTRA(PAR(1), R, THETA, GU, DGU_DR, DGU_DTH)
      END SELECT
   END SUBROUTINE METRIC_CONTRA

   PURE FUNCTION INNER_BOUNDARY(MID, PAR) RESULT(RB)
      ! Radius below which a ray counts as captured: the outer event
      ! horizon for Kerr, the r = 2m singular surface for the q-metric.
      INTEGER, INTENT(IN)  :: MID
      REAL(WP), INTENT(IN) :: PAR(4)
      REAL(WP) :: RB
      SELECT CASE (MID)
      CASE (3)
         RB = PAR(1)
      CASE (MID_QMETRIC)
         RB = 2.0_WP
      CASE DEFAULT
         RB = HORIZON_RADIUS(PAR(1))
      END SELECT
   END FUNCTION INNER_BOUNDARY

END MODULE SPACETIME
