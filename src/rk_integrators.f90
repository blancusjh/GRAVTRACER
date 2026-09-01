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

   PUBLIC :: RK_EMBEDDED_STEP, RKDP45_STEP_FSAL
   PUBLIC :: DP45_DENSE_PREP, DP45_DENSE_EVAL
   INTEGER, PARAMETER, PUBLIC :: METHOD_RKDP45 = 1
   INTEGER, PARAMETER, PUBLIC :: METHOD_RKCK45 = 2
   INTEGER, PARAMETER, PUBLIC :: METHOD_RKF45 = 3

   ! Dense-output coefficients of the Dormand-Prince 5(4) pair
   ! (Hairer, Norsett & Wanner, Solving ODEs I, DOPRI5 continuous
   ! extension; 4th-order accurate over the whole step).
   REAL(WP), PARAMETER :: D1 = -12715105075.0_WP/11282082432.0_WP
   REAL(WP), PARAMETER :: D3 = 87487479700.0_WP/32700410799.0_WP
   REAL(WP), PARAMETER :: D4 = -10690763975.0_WP/1880347072.0_WP
   REAL(WP), PARAMETER :: D5 = 701980252875.0_WP/199316789632.0_WP
   REAL(WP), PARAMETER :: D6 = -1453857185.0_WP/822651844.0_WP
   REAL(WP), PARAMETER :: D7 = 69997945.0_WP/29380423.0_WP

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
      ! Stateless Dormand-Prince step (no FSAL reuse, stages discarded).
      INTEGER, INTENT(IN)   :: MID
      REAL(WP), INTENT(IN)  :: PAR(4), PT, PPHI, Y(6), H
      REAL(WP), INTENT(OUT) :: YNEW(6), ERRV(6)
      REAL(WP) :: KS(6, 7)

      CALL RKDP45_STEP_FSAL(MID, PAR, PT, PPHI, Y, H, KS, .FALSE., YNEW, ERRV)
   END SUBROUTINE STEP_RKDP45

   PURE SUBROUTINE RKDP45_STEP_FSAL(MID, PAR, PT, PPHI, Y, H, KS, K1_VALID, &
                                    YNEW, ERRV)
      ! Dormand-Prince step exploiting the FSAL property. If K1_VALID,
      ! KS(:, 1) holds f(Y) on entry (the 7th stage of the previous
      ! accepted step, or the 1st stage of a rejected attempt at the
      ! same Y) and the first evaluation is skipped. On exit KS holds
      ! all seven stages of this attempt; KS(:, 7) = f(YNEW).
      INTEGER, INTENT(IN)     :: MID
      REAL(WP), INTENT(IN)    :: PAR(4), PT, PPHI, Y(6), H
      REAL(WP), INTENT(INOUT) :: KS(6, 7)
      LOGICAL, INTENT(IN)     :: K1_VALID
      REAL(WP), INTENT(OUT)   :: YNEW(6), ERRV(6)

      IF (.NOT. K1_VALID) CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y, KS(:, 1))
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*(0.2_WP*KS(:, 1)), KS(:, 2))
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*((3.0_WP/40.0_WP)*KS(:, 1) + &
           (9.0_WP/40.0_WP)*KS(:, 2)), KS(:, 3))
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*((44.0_WP/45.0_WP)*KS(:, 1) - &
           (56.0_WP/15.0_WP)*KS(:, 2) + (32.0_WP/9.0_WP)*KS(:, 3)), KS(:, 4))
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*((19372.0_WP/6561.0_WP)*KS(:, 1) - &
           (25360.0_WP/2187.0_WP)*KS(:, 2) + (64448.0_WP/6561.0_WP)*KS(:, 3) - &
           (212.0_WP/729.0_WP)*KS(:, 4)), KS(:, 5))
      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, Y + H*((9017.0_WP/3168.0_WP)*KS(:, 1) - &
           (355.0_WP/33.0_WP)*KS(:, 2) + (46732.0_WP/5247.0_WP)*KS(:, 3) + &
           (49.0_WP/176.0_WP)*KS(:, 4) - (5103.0_WP/18656.0_WP)*KS(:, 5)), KS(:, 6))

      YNEW = Y + H*((35.0_WP/384.0_WP)*KS(:, 1) + (500.0_WP/1113.0_WP)*KS(:, 3) + &
             (125.0_WP/192.0_WP)*KS(:, 4) - (2187.0_WP/6784.0_WP)*KS(:, 5) + &
             (11.0_WP/84.0_WP)*KS(:, 6))

      CALL GEODESIC_RHS(MID, PAR, PT, PPHI, YNEW, KS(:, 7))

      ERRV = H*((35.0_WP/384.0_WP - 5179.0_WP/57600.0_WP)*KS(:, 1) + &
             (500.0_WP/1113.0_WP - 7571.0_WP/16695.0_WP)*KS(:, 3) + &
             (125.0_WP/192.0_WP - 393.0_WP/640.0_WP)*KS(:, 4) + &
             (-2187.0_WP/6784.0_WP + 92097.0_WP/339200.0_WP)*KS(:, 5) + &
             (11.0_WP/84.0_WP - 187.0_WP/2100.0_WP)*KS(:, 6) - &
             (1.0_WP/40.0_WP)*KS(:, 7))
   END SUBROUTINE RKDP45_STEP_FSAL

   PURE SUBROUTINE DP45_DENSE_PREP(Y0, Y1, H, KS, RCONT)
      ! Coefficients of the 4th-order continuous extension of an
      ! accepted Dormand-Prince step Y0 -> Y1 of size H with stages KS.
      REAL(WP), INTENT(IN)  :: Y0(6), Y1(6), H, KS(6, 7)
      REAL(WP), INTENT(OUT) :: RCONT(6, 5)

      RCONT(:, 1) = Y0
      RCONT(:, 2) = Y1 - Y0
      RCONT(:, 3) = H*KS(:, 1) - RCONT(:, 2)
      RCONT(:, 4) = RCONT(:, 2) - H*KS(:, 7) - RCONT(:, 3)
      RCONT(:, 5) = H*(D1*KS(:, 1) + D3*KS(:, 3) + D4*KS(:, 4) + &
                       D5*KS(:, 5) + D6*KS(:, 6) + D7*KS(:, 7))
   END SUBROUTINE DP45_DENSE_PREP

   PURE SUBROUTINE DP45_DENSE_EVAL(RCONT, THETA, YOUT)
      ! Evaluate the continuous extension at fraction THETA in [0, 1]
      ! of the step (THETA = 0 -> Y0, THETA = 1 -> Y1).
      REAL(WP), INTENT(IN)  :: RCONT(6, 5), THETA
      REAL(WP), INTENT(OUT) :: YOUT(6)
      REAL(WP) :: T1

      T1 = 1.0_WP - THETA
      YOUT = RCONT(:, 1) + THETA*(RCONT(:, 2) + T1*(RCONT(:, 3) + &
             THETA*(RCONT(:, 4) + T1*RCONT(:, 5))))
   END SUBROUTINE DP45_DENSE_EVAL

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
