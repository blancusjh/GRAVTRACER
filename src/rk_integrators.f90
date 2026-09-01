MODULE RK_INTEGRATORS
   ! Embedded Runge-Kutta 5(4) pairs with adaptive-step support.
   ! Methods: 1 = Dormand-Prince (RKDP45, default), 2 = Cash-Karp (RKCK45),
   !          3 = Fehlberg (RKF45).
   ! Each step advances with the 5th-order solution and returns the
   ! difference between the 5th- and 4th-order solutions as error estimate.
   USE KIND_PARAMS, ONLY: WP
   USE GEODESIC_EOM, ONLY: GEODESIC_RHS
   IMPLICIT NONE
   PRIVATE

   PUBLIC :: RK_EMBEDDED_STEP
   INTEGER, PARAMETER, PUBLIC :: METHOD_RKDP45 = 1
   INTEGER, PARAMETER, PUBLIC :: METHOD_RKCK45 = 2
   INTEGER, PARAMETER, PUBLIC :: METHOD_RKF45 = 3

CONTAINS

   PURE SUBROUTINE RK_EMBEDDED_STEP(METHOD, MID, PAR, PT, PPHI, Y, H, YNEW, ERRV)
      INTEGER, INTENT(IN)   :: METHOD, MID
      REAL(WP), INTENT(IN)  :: PAR(4), PT, PPHI, Y(6), H
      REAL(WP), INTENT(OUT) :: YNEW(6), ERRV(6)

      SELECT CASE (METHOD)
      CASE (METHOD_RKCK45)
         CALL STEP_RKCK45(MID, PAR, PT, PPHI, Y, H, YNEW, ERRV)
      CASE (METHOD_RKF45)
         CALL STEP_RKF45(MID, PAR, PT, PPHI, Y, H, YNEW, ERRV)
      CASE DEFAULT
         CALL STEP_RKDP45(MID, PAR, PT, PPHI, Y, H, YNEW, ERRV)
      END SELECT
   END SUBROUTINE RK_EMBEDDED_STEP

   PURE SUBROUTINE STEP_RKDP45(MID, PAR, PT, PPHI, Y, H, YNEW, ERRV)
      INTEGER, INTENT(IN)   :: MID
      REAL(WP), INTENT(IN)  :: PAR(4), PT, PPHI, Y(6), H
      REAL(WP), INTENT(OUT) :: YNEW(6), ERRV(6)
      REAL(WP) :: K1(6), K2(6), K3(6), K4(6), K5(6), K6(6), K7(6)

      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y, K1)
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*(0.2_WP*K1), K2)
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*((3.0_WP/40.0_WP)*K1 + &
           (9.0_WP/40.0_WP)*K2), K3)
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*((44.0_WP/45.0_WP)*K1 - &
           (56.0_WP/15.0_WP)*K2 + (32.0_WP/9.0_WP)*K3), K4)
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*((19372.0_WP/6561.0_WP)*K1 - &
           (25360.0_WP/2187.0_WP)*K2 + (64448.0_WP/6561.0_WP)*K3 - &
           (212.0_WP/729.0_WP)*K4), K5)
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*((9017.0_WP/3168.0_WP)*K1 - &
           (355.0_WP/33.0_WP)*K2 + (46732.0_WP/5247.0_WP)*K3 + &
           (49.0_WP/176.0_WP)*K4 - (5103.0_WP/18656.0_WP)*K5), K6)

      YNEW = Y + H*((35.0_WP/384.0_WP)*K1 + (500.0_WP/1113.0_WP)*K3 + &
             (125.0_WP/192.0_WP)*K4 - (2187.0_WP/6784.0_WP)*K5 + &
             (11.0_WP/84.0_WP)*K6)

      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, YNEW, K7)

      ERRV = H*((35.0_WP/384.0_WP - 5179.0_WP/57600.0_WP)*K1 + &
             (500.0_WP/1113.0_WP - 7571.0_WP/16695.0_WP)*K3 + &
             (125.0_WP/192.0_WP - 393.0_WP/640.0_WP)*K4 + &
             (-2187.0_WP/6784.0_WP + 92097.0_WP/339200.0_WP)*K5 + &
             (11.0_WP/84.0_WP - 187.0_WP/2100.0_WP)*K6 - &
             (1.0_WP/40.0_WP)*K7)
   END SUBROUTINE STEP_RKDP45

   PURE SUBROUTINE STEP_RKCK45(MID, PAR, PT, PPHI, Y, H, YNEW, ERRV)
      INTEGER, INTENT(IN)   :: MID
      REAL(WP), INTENT(IN)  :: PAR(4), PT, PPHI, Y(6), H
      REAL(WP), INTENT(OUT) :: YNEW(6), ERRV(6)
      REAL(WP) :: K1(6), K2(6), K3(6), K4(6), K5(6), K6(6)

      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y, K1)
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*(0.2_WP*K1), K2)
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*((3.0_WP/40.0_WP)*K1 + &
           (9.0_WP/40.0_WP)*K2), K3)
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*(0.3_WP*K1 - 0.9_WP*K2 + &
           1.2_WP*K3), K4)
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*((-11.0_WP/54.0_WP)*K1 + &
           2.5_WP*K2 - (70.0_WP/27.0_WP)*K3 + (35.0_WP/27.0_WP)*K4), K5)
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*((1631.0_WP/55296.0_WP)*K1 + &
           (175.0_WP/512.0_WP)*K2 + (575.0_WP/13824.0_WP)*K3 + &
           (44275.0_WP/110592.0_WP)*K4 + (253.0_WP/4096.0_WP)*K5), K6)

      YNEW = Y + H*((37.0_WP/378.0_WP)*K1 + (250.0_WP/621.0_WP)*K3 + &
             (125.0_WP/594.0_WP)*K4 + (512.0_WP/1771.0_WP)*K6)

      ERRV = H*((37.0_WP/378.0_WP - 2825.0_WP/27648.0_WP)*K1 + &
             (250.0_WP/621.0_WP - 18575.0_WP/48384.0_WP)*K3 + &
             (125.0_WP/594.0_WP - 13525.0_WP/55296.0_WP)*K4 - &
             (277.0_WP/14336.0_WP)*K5 + &
             (512.0_WP/1771.0_WP - 0.25_WP)*K6)
   END SUBROUTINE STEP_RKCK45

   PURE SUBROUTINE STEP_RKF45(MID, PAR, PT, PPHI, Y, H, YNEW, ERRV)
      INTEGER, INTENT(IN)   :: MID
      REAL(WP), INTENT(IN)  :: PAR(4), PT, PPHI, Y(6), H
      REAL(WP), INTENT(OUT) :: YNEW(6), ERRV(6)
      REAL(WP) :: K1(6), K2(6), K3(6), K4(6), K5(6), K6(6)

      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y, K1)
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*(0.25_WP*K1), K2)
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*((3.0_WP/32.0_WP)*K1 + &
           (9.0_WP/32.0_WP)*K2), K3)
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*((1932.0_WP/2197.0_WP)*K1 - &
           (7200.0_WP/2197.0_WP)*K2 + (7296.0_WP/2197.0_WP)*K3), K4)
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*((439.0_WP/216.0_WP)*K1 - &
           8.0_WP*K2 + (3680.0_WP/513.0_WP)*K3 - &
           (845.0_WP/4104.0_WP)*K4), K5)
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*((-8.0_WP/27.0_WP)*K1 + &
           2.0_WP*K2 - (3544.0_WP/2565.0_WP)*K3 + &
           (1859.0_WP/4104.0_WP)*K4 - (11.0_WP/40.0_WP)*K5), K6)

      YNEW = Y + H*((16.0_WP/135.0_WP)*K1 + (6656.0_WP/12825.0_WP)*K3 + &
             (28561.0_WP/56430.0_WP)*K4 - (9.0_WP/50.0_WP)*K5 + &
             (2.0_WP/55.0_WP)*K6)

      ERRV = H*((16.0_WP/135.0_WP - 25.0_WP/216.0_WP)*K1 + &
             (6656.0_WP/12825.0_WP - 1408.0_WP/2565.0_WP)*K3 + &
             (28561.0_WP/56430.0_WP - 2197.0_WP/4104.0_WP)*K4 + &
             (-9.0_WP/50.0_WP + 0.2_WP)*K5 + (2.0_WP/55.0_WP)*K6)
   END SUBROUTINE STEP_RKF45

END MODULE RK_INTEGRATORS
