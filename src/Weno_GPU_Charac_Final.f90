! ==============================================================================
! MODULE: GPU_DATA
! 修改：移除了硬编码的 IIX, IIY，改为动态分配
! ==============================================================================
MODULE GPU_DATA
    USE ISO_C_BINDING
    USE CUDAFOR 
    IMPLICIT NONE

    INTEGER, PARAMETER :: DP = 8 
    
    ! --- [修改] 移除了 PARAMETER :: IIX, IIY ---
    ! 仅保留 Ghost Cell 层数 (MT) 和变量数 (MN)
    INTEGER, PARAMETER :: MT  = 3
    INTEGER, PARAMETER :: MN  = 4
    
    ! Block 尺寸 (GPU 线程块配置，保持不变)
    INTEGER, PARAMETER :: BDIMX = 32
    INTEGER, PARAMETER :: BDIMY = 4

    ! 设备端数组 (Device Arrays)
    REAL(KIND=DP), DEVICE, ALLOCATABLE :: D_d(:,:), U_d(:,:), V_d(:,:), P_d(:,:)
    REAL(KIND=DP), DEVICE, ALLOCATABLE :: UC_d(:,:,:,:) 
    REAL(KIND=DP), DEVICE, ALLOCATABLE :: RHS_d(:,:,:)  
    REAL(KIND=DP), DEVICE, ALLOCATABLE :: Flux_X_d(:,:,:), Flux_Y_d(:,:,:)
    
    ! 主机端数组 (Host Arrays)
    REAL(KIND=DP), ALLOCATABLE :: D_h(:,:), U_h(:,:), V_h(:,:), P_h(:,:)

CONTAINS
    ! --- [修改] 增加 NX, NY 作为参数 ---
    SUBROUTINE ALLOCATE_GPU(NX, NY)
        INTEGER, INTENT(IN) :: NX, NY
        INTEGER :: istat
        
        ! 根据传入的 NX, NY 分配显存
        ALLOCATE(D_d(-MT:NX+MT, -MT:NY+MT), STAT=istat)
        ALLOCATE(U_d(-MT:NX+MT, -MT:NY+MT), STAT=istat)
        ALLOCATE(V_d(-MT:NX+MT, -MT:NY+MT), STAT=istat)
        ALLOCATE(P_d(-MT:NX+MT, -MT:NY+MT), STAT=istat)
        
        ALLOCATE(UC_d(-MT:NX+MT, -MT:NY+MT, MN, 0:2), STAT=istat)
        ALLOCATE(RHS_d(-MT:NX+MT, -MT:NY+MT, MN), STAT=istat)
        
        ALLOCATE(Flux_X_d(-MT:NX+MT, -MT:NY+MT, MN), STAT=istat)
        ALLOCATE(Flux_Y_d(-MT:NX+MT, -MT:NY+MT, MN), STAT=istat)
        
        ! 分配主机端内存用于 IO
        ALLOCATE(D_h(-MT:NX+MT, -MT:NY+MT), U_h(-MT:NX+MT, -MT:NY+MT), &
                 V_h(-MT:NX+MT, -MT:NY+MT), P_h(-MT:NX+MT, -MT:NY+MT), STAT=istat)
                 
    END SUBROUTINE ALLOCATE_GPU
    
    SUBROUTINE FREE_GPU()
        DEALLOCATE(D_d, U_d, V_d, P_d, UC_d, RHS_d, Flux_X_d, Flux_Y_d)
        IF (ALLOCATED(D_h)) DEALLOCATE(D_h, U_h, V_h, P_h)
    END SUBROUTINE FREE_GPU
END MODULE GPU_DATA

! ==============================================================================
! MODULE: KERNELS
! 修改：所有 Kernel 增加 NX, NY 参数
! ==============================================================================
MODULE KERNELS
    USE GPU_DATA
    IMPLICIT NONE

CONTAINS

    ! --- WENO5-JS (标量函数，无需修改) ---
    ATTRIBUTES(DEVICE) REAL(KIND=DP) FUNCTION WENO5_JS(v1, v2, v3, v4, v5)
        IMPLICIT NONE
        REAL(KIND=DP), INTENT(IN) :: v1, v2, v3, v4, v5
        REAL(KIND=DP) :: b0, b1, b2, a0, a1, a2, s_inv
        REAL(KIND=DP), PARAMETER :: EPS = 1.0D-40
        REAL(KIND=DP), PARAMETER :: C1312 = 13.0_DP/12.0_DP, C14=0.25_DP, C16=1.0_DP/6.0_DP
        REAL(KIND=DP), PARAMETER :: D0=0.1_DP, D1=0.6_DP, D2=0.3_DP
    
        b0 = C1312*(v1 - 2.0_DP*v2 + v3)**2 + C14*(v1 - 4.0_DP*v2 + 3.0_DP*v3)**2
        b1 = C1312*(v2 - 2.0_DP*v3 + v4)**2 + C14*(v2 - v4)**2
        b2 = C1312*(v3 - 2.0_DP*v4 + v5)**2 + C14*(3.0_DP*v3 - 4.0_DP*v4 + v5)**2
        a0 = D0 / ((EPS + b0)**2); a1 = D1 / ((EPS + b1)**2); a2 = D2 / ((EPS + b2)**2)
        s_inv = 1.0_DP / (a0 + a1 + a2)
        WENO5_JS = (a0*(2.0_DP*v1-7.0_DP*v2+11.0_DP*v3) + a1*(-v2+5.0_DP*v3+2.0_DP*v4) + &
                    a2*(2.0_DP*v3+5.0_DP*v4-v5)) * C16 * s_inv
    END FUNCTION WENO5_JS

    ! --- 特征矩阵计算 (无需修改) ---
    ATTRIBUTES(DEVICE) SUBROUTINE GET_EIGEN_MATS(u, v, h, c, gamma, n1, n2, L, R)
        IMPLICIT NONE
        REAL(KIND=DP), INTENT(IN) :: u, v, h, c, gamma, n1, n2
        REAL(KIND=DP), INTENT(OUT) :: L(4,4), R(4,4)
        REAL(KIND=DP) :: b1, q2, gm1
        
        gm1 = gamma - 1.0_DP
        q2 = 0.5_DP * (u*u + v*v)
        b1 = gm1 / (c*c)
        
        R(1,1) = 1.0_DP; R(2,1) = u - c*n1; R(3,1) = v - c*n2; R(4,1) = h - c*(u*n1 + v*n2)
        R(1,2) = 1.0_DP; R(2,2) = u;        R(3,2) = v;        R(4,2) = q2
        R(1,3) = 1.0_DP; R(2,3) = u + c*n1; R(3,3) = v + c*n2; R(4,3) = h + c*(u*n1 + v*n2)
        R(1,4) = 0.0_DP; R(2,4) = -n2;      R(3,4) = n1;       R(4,4) = -(u*n2 - v*n1)
        
        L(1,1) = 0.5_DP * (b1*q2 + (u*n1+v*n2)/c)
        L(1,2) = -0.5_DP * (b1*u + n1/c)
        L(1,3) = -0.5_DP * (b1*v + n2/c)
        L(1,4) = 0.5_DP * b1
        
        L(2,1) = 1.0_DP - b1 * q2
        L(2,2) = b1 * u; L(2,3) = b1 * v; L(2,4) = -b1
        
        L(3,1) = 0.5_DP * (b1*q2 - (u*n1+v*n2)/c)
        L(3,2) = -0.5_DP * (b1*u - n1/c)
        L(3,3) = -0.5_DP * (b1*v - n2/c)
        L(3,4) = 0.5_DP * b1
        
        L(4,1) = (u*n2 - v*n1); L(4,2) = -n2; L(4,3) = n1; L(4,4) = 0.0_DP
    END SUBROUTINE GET_EIGEN_MATS

    ! --- X方向通量 (特征空间) ---
    ! [修改] 增加 NX, NY 参数，并将内部 IIX/IIY 替换为 NX/NY
    ATTRIBUTES(GLOBAL) SUBROUTINE KR_FLUX_CHAR_X(IRK, GAMMA, DX, DY, NX, NY)
        IMPLICIT NONE
        INTEGER, VALUE :: IRK, NX, NY
        REAL(KIND=DP), VALUE :: GAMMA, DX, DY
        INTEGER :: i, j, m, k, n, io
        REAL(KIND=DP) :: U_st(-2:3, 4), W_st(-2:3, 4), q_L(4), q_R(4), f_L(4), f_R(4)
        REAL(KIND=DP) :: L_mat(4,4), R_mat(4,4)
        REAL(KIND=DP) :: rL, uL, vL, hL, rR, uR, vR, hR, sqL, sqR, inv_sq
        REAL(KIND=DP) :: roe_u, roe_v, roe_h, roe_c, rho, u, v, p, E, c, am, inv_r, sum_val
        
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 2
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
        io = MOD(IRK, 3)

        IF (i >= -1 .AND. i <= NX .AND. j >= 0 .AND. j <= NY) THEN
            DO k = -2, 3
                DO m = 1, MN
                    U_st(k, m) = UC_d(i+k, j, m, io)
                END DO
            END DO

            ! Roe 平均
            rL = U_st(0,1); uL = U_st(0,2)/rL; vL = U_st(0,3)/rL
            p = (GAMMA-1.0_DP)*(U_st(0,4)-0.5_DP*rL*(uL**2+vL**2))
            hL = (U_st(0,4) + p) / rL
            
            rR = U_st(1,1); uR = U_st(1,2)/rR; vR = U_st(1,3)/rR
            p = (GAMMA-1.0_DP)*(U_st(1,4)-0.5_DP*rR*(uR**2+vR**2))
            hR = (U_st(1,4) + p) / rR
            
            sqL = SQRT(rL); sqR = SQRT(rR); inv_sq = 1.0_DP/(sqL + sqR)
            roe_u = (sqL*uL + sqR*uR) * inv_sq
            roe_v = (sqL*vL + sqR*vR) * inv_sq
            roe_h = (sqL*hL + sqR*hR) * inv_sq
            roe_c = SQRT((GAMMA-1.0_DP)*(roe_h - 0.5_DP*(roe_u**2 + roe_v**2)))
            
            CALL GET_EIGEN_MATS(roe_u, roe_v, roe_h, roe_c, GAMMA, 1.0_DP, 0.0_DP, L_mat, R_mat)
            
            ! 投影 U -> W
            DO k = -2, 3
                DO m = 1, MN
                    sum_val = 0.0_DP
                    DO n = 1, MN
                        sum_val = sum_val + L_mat(m, n) * U_st(k, n)
                    END DO
                    W_st(k, m) = sum_val
                END DO
            END DO
            
            ! WENO5 重构
            DO m = 1, MN
                q_L(m) = WENO5_JS(W_st(-2,m), W_st(-1,m), W_st(0,m), W_st(1,m), W_st(2,m))
                q_R(m) = WENO5_JS(W_st(3,m), W_st(2,m), W_st(1,m), W_st(0,m), W_st(-1,m))
            END DO
            
            ! 还原 W -> U
            DO m = 1, MN
                sum_val = 0.0_DP
                DO n = 1, MN
                    sum_val = sum_val + R_mat(m, n) * q_L(n)
                END DO
                f_L(m) = sum_val
                sum_val = 0.0_DP
                DO n = 1, MN
                    sum_val = sum_val + R_mat(m, n) * q_R(n)
                END DO
                f_R(m) = sum_val
            END DO
            q_L = f_L; q_R = f_R
            
            am = ABS(roe_u) + roe_c
            DO m = 1, MN
                rho=q_L(1); inv_r=1.0_DP/rho; u=q_L(2)*inv_r; v=q_L(3)*inv_r; E=q_L(4)
                p=(GAMMA-1.0_DP)*(E-0.5_DP*rho*(u**2+v**2))
                if(m==1) f_L(m)=rho*u; if(m==2) f_L(m)=rho*u**2+p; if(m==3) f_L(m)=rho*u*v; if(m==4) f_L(m)=u*(E+p)
                
                rho=q_R(1); inv_r=1.0_DP/rho; u=q_R(2)*inv_r; v=q_R(3)*inv_r; E=q_R(4)
                p=(GAMMA-1.0_DP)*(E-0.5_DP*rho*(u**2+v**2))
                if(m==1) f_R(m)=rho*u; if(m==2) f_R(m)=rho*u**2+p; if(m==3) f_R(m)=rho*u*v; if(m==4) f_R(m)=u*(E+p)
                
                Flux_X_d(i, j, m) = 0.5_DP * (f_L(m) + f_R(m) - am*(q_R(m)-q_L(m)))
            END DO
        END IF
    END SUBROUTINE KR_FLUX_CHAR_X

    ! --- Y方向通量 (特征空间) ---
    ! [修改] 增加 NX, NY 参数
    ATTRIBUTES(GLOBAL) SUBROUTINE KR_FLUX_CHAR_Y(IRK, GAMMA, DX, DY, NX, NY)
        IMPLICIT NONE
        INTEGER, VALUE :: IRK, NX, NY
        REAL(KIND=DP), VALUE :: GAMMA, DX, DY
        INTEGER :: i, j, m, k, n, io
        REAL(KIND=DP) :: U_st(-2:3, 4), W_st(-2:3, 4), q_L(4), q_R(4), g_L(4), g_R(4)
        REAL(KIND=DP) :: L_mat(4,4), R_mat(4,4)
        REAL(KIND=DP) :: rL, uL, vL, hL, rR, uR, vR, hR, sqL, sqR, inv_sq
        REAL(KIND=DP) :: roe_u, roe_v, roe_h, roe_c, rho, u, v, p, E, c, am, inv_r, sum_val
        
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 2
        io = MOD(IRK, 3)

        IF (i >= 0 .AND. i <= NX .AND. j >= -1 .AND. j <= NY) THEN
            DO k = -2, 3
                DO m = 1, MN
                    U_st(k, m) = UC_d(i, j+k, m, io)
                END DO
            END DO

            ! Roe 平均
            rL = U_st(0,1); uL = U_st(0,2)/rL; vL = U_st(0,3)/rL
            p = (GAMMA-1.0_DP)*(U_st(0,4)-0.5_DP*rL*(uL**2+vL**2))
            hL = (U_st(0,4) + p) / rL
            rR = U_st(1,1); uR = U_st(1,2)/rR; vR = U_st(1,3)/rR
            p = (GAMMA-1.0_DP)*(U_st(1,4)-0.5_DP*rR*(uR**2+vR**2))
            hR = (U_st(1,4) + p) / rR
            sqL = SQRT(rL); sqR = SQRT(rR); inv_sq = 1.0_DP/(sqL + sqR)
            roe_u = (sqL*uL + sqR*uR) * inv_sq
            roe_v = (sqL*vL + sqR*vR) * inv_sq
            roe_h = (sqL*hL + sqR*hR) * inv_sq
            roe_c = SQRT((GAMMA-1.0_DP)*(roe_h - 0.5_DP*(roe_u**2 + roe_v**2)))
            
            CALL GET_EIGEN_MATS(roe_u, roe_v, roe_h, roe_c, GAMMA, 0.0_DP, 1.0_DP, L_mat, R_mat)
            
            ! 投影 U -> W
            DO k = -2, 3
                DO m = 1, MN
                    sum_val = 0.0_DP
                    DO n = 1, MN
                        sum_val = sum_val + L_mat(m, n) * U_st(k, n)
                    END DO
                    W_st(k, m) = sum_val
                END DO
            END DO
            
            ! WENO5
            DO m = 1, MN
                q_L(m) = WENO5_JS(W_st(-2,m), W_st(-1,m), W_st(0,m), W_st(1,m), W_st(2,m))
                q_R(m) = WENO5_JS(W_st(3,m), W_st(2,m), W_st(1,m), W_st(0,m), W_st(-1,m))
            END DO
            
            ! 还原
            DO m = 1, MN
                sum_val = 0.0_DP
                DO n = 1, MN
                    sum_val = sum_val + R_mat(m, n) * q_L(n)
                END DO
                g_L(m) = sum_val
                sum_val = 0.0_DP
                DO n = 1, MN
                    sum_val = sum_val + R_mat(m, n) * q_R(n)
                END DO
                g_R(m) = sum_val
            END DO
            q_L = g_L; q_R = g_R
            
            am = ABS(roe_v) + roe_c
            DO m = 1, MN
                rho=q_L(1); inv_r=1.0_DP/rho; u=q_L(2)*inv_r; v=q_L(3)*inv_r; E=q_L(4)
                p=(GAMMA-1.0_DP)*(E-0.5_DP*rho*(u**2+v**2))
                if(m==1) g_L(m)=rho*v; if(m==2) g_L(m)=rho*u*v; if(m==3) g_L(m)=rho*v**2+p; if(m==4) g_L(m)=v*(E+p)
                
                rho=q_R(1); inv_r=1.0_DP/rho; u=q_R(2)*inv_r; v=q_R(3)*inv_r; E=q_R(4)
                p=(GAMMA-1.0_DP)*(E-0.5_DP*rho*(u**2+v**2))
                if(m==1) g_R(m)=rho*v; if(m==2) g_R(m)=rho*u*v; if(m==3) g_R(m)=rho*v**2+p; if(m==4) g_R(m)=v*(E+p)
                
                Flux_Y_d(i, j, m) = 0.5_DP * (g_L(m) + g_R(m) - am*(q_R(m)-q_L(m)))
            END DO
        END IF
    END SUBROUTINE KR_FLUX_CHAR_Y

    ! --- RHS 计算 ---
    ! [修改] 增加 NX, NY 参数
    ATTRIBUTES(GLOBAL) SUBROUTINE KR_COMPUTE_RHS(DX, DY, NX, NY)
        IMPLICIT NONE
        REAL(KIND=DP), VALUE :: DX, DY
        INTEGER, VALUE :: NX, NY
        INTEGER :: i, j, m
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
        IF (i >= 0 .AND. i <= NX .AND. j >= 0 .AND. j <= NY) THEN
            DO m = 1, MN
                RHS_d(i,j,m) = -((Flux_X_d(i,j,m)-Flux_X_d(i-1,j,m))/DX + (Flux_Y_d(i,j,m)-Flux_Y_d(i,j-1,m))/DY)
            END DO
        END IF
    END SUBROUTINE KR_COMPUTE_RHS

    ! --- TVD-RK3 时间推进 ---
    ! [修改] 增加 NX, NY 参数
    ATTRIBUTES(GLOBAL) SUBROUTINE KR_RK_UPDATE(DT, IRK, GAMMA, NX, NY)
        IMPLICIT NONE
        REAL(KIND=DP), VALUE :: DT, GAMMA
        INTEGER, VALUE :: IRK, NX, NY
        INTEGER :: i, j, m, idx_target
        REAL(KIND=DP) :: rho, u, v, E, p, kin_eng
        REAL(KIND=DP), PARAMETER :: MIN_EPS = 1.0D-6
        
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1

        IF (i >= 0 .AND. i <= NX .AND. j >= 0 .AND. j <= NY) THEN
            IF (IRK == 0) THEN
                idx_target = 1
            ELSE IF (IRK == 1) THEN
                idx_target = 2
            ELSE
                idx_target = 0
            END IF

            DO m = 1, MN
                IF (IRK == 0) THEN
                    UC_d(i,j,m,1) = UC_d(i,j,m,0) + DT*RHS_d(i,j,m)
                ELSE IF (IRK == 1) THEN
                    UC_d(i,j,m,2) = 0.75_DP*UC_d(i,j,m,0) + 0.25_DP*(UC_d(i,j,m,1) + DT*RHS_d(i,j,m))
                ELSE
                    UC_d(i,j,m,0) = (UC_d(i,j,m,0) + 2.0_DP*(UC_d(i,j,m,2) + DT*RHS_d(i,j,m)))/3.0_DP
                END IF
            END DO
            
            ! Positivity Protection
            rho = UC_d(i, j, 1, idx_target)
            IF (rho < MIN_EPS) THEN
                rho = MIN_EPS
                UC_d(i, j, 1, idx_target) = rho
                UC_d(i, j, 2, idx_target) = 0.0_DP
                UC_d(i, j, 3, idx_target) = 0.0_DP
            END IF
            u = UC_d(i, j, 2, idx_target) / rho
            v = UC_d(i, j, 3, idx_target) / rho
            E = UC_d(i, j, 4, idx_target)
            kin_eng = 0.5_DP * rho * (u**2 + v**2)
            p = (GAMMA - 1.0_DP) * (E - kin_eng)
            IF (p < MIN_EPS) THEN
                p = MIN_EPS
                E = p / (GAMMA - 1.0_DP) + kin_eng
                UC_d(i, j, 4, idx_target) = E
            END IF
        END IF
    END SUBROUTINE KR_RK_UPDATE

    ! --- 边界条件 (透射边界) ---
    ! [修改] 增加 NX, NY 参数，正确处理 Ghost Cells
    ATTRIBUTES(GLOBAL) SUBROUTINE KR_BOUNDARY(IRK, NX, NY)
        IMPLICIT NONE
        INTEGER, VALUE :: IRK, NX, NY
        INTEGER :: i, j, k, m, io
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
        io = MOD(IRK, 3)
        
        IF (j >= 0 .AND. j <= NY) THEN
            DO k = 1, MT
                DO m = 1, MN
                    ! Transmissive: 复制边界值 (0 -> -k, NX -> NX+k)
                    UC_d(-k, j, m, io) = UC_d(0, j, m, io)
                    UC_d(NX+k, j, m, io) = UC_d(NX, j, m, io)
                END DO
            END DO
        END IF
        IF (i >= -MT .AND. i <= NX+MT) THEN
            DO k = 1, MT
                DO m = 1, MN
                    ! Transmissive: 复制边界值 (0 -> -k, NY -> NY+k)
                    UC_d(i, -k, m, io) = UC_d(i, 0, m, io)
                    UC_d(i, NY+k, m, io) = UC_d(i, NY, m, io)
                END DO
            END DO
        END IF
    END SUBROUTINE KR_BOUNDARY

    ! --- 提取物理量 ---
    ATTRIBUTES(GLOBAL) SUBROUTINE KR_FINAL_PRIM(GAMMA, NX, NY)
        IMPLICIT NONE
        REAL(KIND=DP), VALUE :: GAMMA
        INTEGER, VALUE :: NX, NY
        INTEGER :: i, j
        REAL(KIND=DP) :: r, inv_r, u, v, E, p
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x - 1 
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y - 1
        IF (i >= 0 .AND. i <= NX .AND. j >= 0 .AND. j <= NY) THEN
            r = UC_d(i,j,1,0); inv_r = 1.0_DP/r
            u = UC_d(i,j,2,0)*inv_r; v = UC_d(i,j,3,0)*inv_r; E = UC_d(i,j,4,0)
            p = (GAMMA - 1.0_DP) * (E - 0.5_DP * r * (u**2 + v**2))
            D_d(i,j) = r; U_d(i,j) = u; V_d(i,j) = v; P_d(i,j) = p
        END IF
    END SUBROUTINE KR_FINAL_PRIM

END MODULE KERNELS

! ==============================================================================
! MAIN PROGRAM (Dynamic Grid Version)
! ==============================================================================
PROGRAM WENO_GPU_DYNAMIC
    USE GPU_DATA
    USE KERNELS
    USE CUDAFOR
    IMPLICIT NONE

    INTEGER :: i, j, N, IRK, istat, arg_len
    REAL(KIND=DP) :: TIME, x, y
    REAL(KIND=DP) :: DX_local, DY_local, GAMMA_local
    REAL(KIND=DP), ALLOCATABLE :: UC_temp(:,:,:,:) 
    TYPE(DIM3) :: dimGrid, dimBlock
    
    ! IO Vars
    CHARACTER(LEN=512) :: config_file   
    CHARACTER(LEN=512) :: output_file   
    
    ! NML Parameters
    REAL(KIND=DP) :: q1_r, q1_u, q1_v, q1_p
    REAL(KIND=DP) :: q2_r, q2_u, q2_v, q2_p
    REAL(KIND=DP) :: q3_r, q3_u, q3_v, q3_p
    REAL(KIND=DP) :: q4_r, q4_u, q4_v, q4_p
    REAL(KIND=DP) :: T_END, DT
    INTEGER :: NX, NY ! [修改] NX, NY 现在是 NML 读取的变量

    NAMELIST /CASE_PARAMS/ q1_r, q1_u, q1_v, q1_p, &
                           q2_r, q2_u, q2_v, q2_p, &
                           q3_r, q3_u, q3_v, q3_p, &
                           q4_r, q4_u, q4_v, q4_p, &
                           T_END, DT, NX, NY

    ! 1. Args
    CALL GET_COMMAND_ARGUMENT(1, config_file, length=arg_len)
    IF (arg_len == 0) STOP "Error: Missing input config file."
    CALL GET_COMMAND_ARGUMENT(2, output_file, length=arg_len)
    IF (arg_len == 0) STOP "Error: Missing output file path."

    ! 2. Read NML (Determine Grid Size First)
    NX = 0; NY = 0; DT = 0.0001_DP
    WRITE(*,*) ">>> Reading config:", TRIM(config_file)
    OPEN(UNIT=10, FILE=TRIM(config_file), STATUS='OLD', ACTION='READ')
    READ(10, NML=CASE_PARAMS)
    CLOSE(10)
    
    IF (NX == 0 .OR. NY == 0) STOP "Error: NX or NY not set in NML."
    WRITE(*,*) ">>> Grid Size:", NX, "x", NY

    ! 3. Init
    GAMMA_local = 1.4_DP
    DX_local = 1.0_DP / REAL(NX, KIND=DP)
    DY_local = 1.0_DP / REAL(NY, KIND=DP)

    ! [修改] 动态分配 GPU 内存
    CALL ALLOCATE_GPU(NX, NY) 
    
    ! Host Init using ALLOCATE_GPU's D_h etc. (Already allocated in module)
    DO j = -MT, NY+MT
      DO i = -MT, NX+MT
         x = REAL(i, KIND=DP)*DX_local
         y = REAL(j, KIND=DP)*DY_local
         
         IF (x >= 0.5_DP .AND. y >= 0.5_DP) THEN
             D_h(i,j)=q1_r; U_h(i,j)=q1_u; V_h(i,j)=q1_v; P_h(i,j)=q1_p
         ELSE IF (x < 0.5_DP .AND. y >= 0.5_DP) THEN
             D_h(i,j)=q2_r; U_h(i,j)=q2_u; V_h(i,j)=q2_v; P_h(i,j)=q2_p
         ELSE IF (x < 0.5_DP .AND. y < 0.5_DP) THEN
             D_h(i,j)=q3_r; U_h(i,j)=q3_u; V_h(i,j)=q3_v; P_h(i,j)=q3_p
         ELSE
             D_h(i,j)=q4_r; U_h(i,j)=q4_u; V_h(i,j)=q4_v; P_h(i,j)=q4_p
         END IF
      END DO
    END DO

    ! Upload
    ALLOCATE(UC_temp(-MT:NX+MT, -MT:NY+MT, MN, 0:2))
    DO j = -MT, NY+MT
        DO i = -MT, NX+MT
           UC_temp(i,j,1,0) = D_h(i,j)
           UC_temp(i,j,2,0) = D_h(i,j)*U_h(i,j)
           UC_temp(i,j,3,0) = D_h(i,j)*V_h(i,j)
           UC_temp(i,j,4,0) = P_h(i,j)/(GAMMA_local-1.0_DP) + 0.5_DP*D_h(i,j)*(U_h(i,j)**2+V_h(i,j)**2)
        END DO
    END DO
    UC_d = UC_temp
    DEALLOCATE(UC_temp)
    
    ! 4. Main Loop
    dimBlock = DIM3(BDIMX, BDIMY, 1)
    ! [修改] Grid 计算使用动态 NX, NY
    dimGrid = DIM3((NX+1+BDIMX-1)/BDIMX + 1, (NY+1+BDIMY-1)/BDIMY + 1, 1)
    
    TIME = 0.0_DP
    N = 0
    
    DO WHILE (TIME < T_END)
        N = N + 1
        IF (TIME + DT > T_END) DT = T_END - TIME
        TIME = TIME + DT

        DO IRK = 0, 2
            ! [修改] 传递 NX, NY 给所有 Kernel
            CALL KR_BOUNDARY<<<dimGrid, dimBlock>>>(IRK, NX, NY)
            CALL KR_FLUX_CHAR_X<<<dimGrid, dimBlock>>>(IRK, GAMMA_local, DX_local, DY_local, NX, NY)
            CALL KR_FLUX_CHAR_Y<<<dimGrid, dimBlock>>>(IRK, GAMMA_local, DX_local, DY_local, NX, NY)
            CALL KR_COMPUTE_RHS<<<dimGrid, dimBlock>>>(DX_local, DY_local, NX, NY)
            CALL KR_RK_UPDATE<<<dimGrid, dimBlock>>>(DT, IRK, GAMMA_local, NX, NY)
        END DO
        
        IF (MOD(N, 100) == 0) WRITE(*, '(A, I6, A, F8.5)') " Step:", N, "  Time:", TIME
    END DO

    WRITE(*,*) ">>> Finished. Saving to: ", TRIM(output_file)
    CALL KR_FINAL_PRIM<<<dimGrid, dimBlock>>>(GAMMA_local, NX, NY)
    CALL SAVE_TARGET(output_file, DX_local, DY_local, NX, NY)
    CALL FREE_GPU()

CONTAINS
    SUBROUTINE SAVE_TARGET(fpath, dx, dy, nx_in, ny_in)
        CHARACTER(LEN=*), INTENT(IN) :: fpath
        REAL(KIND=DP) :: dx, dy
        INTEGER :: nx_in, ny_in
        
        D_h = D_d; U_h = U_d; V_h = V_d; P_h = P_d
        istat = cudaDeviceSynchronize()
        
        OPEN(20, FILE=TRIM(fpath), STATUS='UNKNOWN')
        WRITE(20, *) 'TITLE="RIEMANN_2D_DYNAMIC"'
        WRITE(20, *) 'VARIABLES= ,"X" ,"Y" ,"D" ,"U" ,"V" ,"P"'
        WRITE(20, *) 'ZONE I=', nx_in+1, ' J=', ny_in+1, ' F=POINT'
        DO j = 0, ny_in
            DO i = 0, nx_in
                WRITE(20, '(2(E16.6E3,1X), 4(E16.6E3,1X))') REAL(i)*dx, REAL(j)*dy, D_h(i,j), U_h(i,j), V_h(i,j), P_h(i,j)
            END DO
        END DO
        CLOSE(20)
    END SUBROUTINE SAVE_TARGET

END PROGRAM WENO_GPU_DYNAMIC