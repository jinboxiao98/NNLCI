// benchmark_hpr.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

// #include <iostream>

// int main()
// {
//     std::cout << "Hello World!\n";
// }

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file


#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string>

#define nx          100   // n=L/dx, the number of intervals in one direction
#define ghst        3     //the number of ghost intervals at one boundary side
#define nxghst      106  // nxghst=nx+2*ghst, the number of intervals plus ghost intervals at the boundary
#define lbdry       0.0    //left boundry position
#define rbdry       1.0   //right boundry position
#define pi         3.1415926 
#define gam        1.4
#define CFL        0.3

void adv_fw(double, double, double);
double flux(double, double, double, int);
void init(double,double);
void compute_fx(double, double);
void compute_fx_2(double, double);
void correc_flux(double, double);
double max(double, double);
double min(double, double);
double sgn(double);
void limiting(double, double);
double minmod(double, double, double);
double minmod(double, double);
int MainRun(std::string fidelity, double rhoE, int i);
int mod(int);
void reconstruction(double);
void basis_funct(double, double, double, double, double, double*, double*, double*);
void bdry_cond_u(double);

double comp_max_time_step(double);
double LF_flux(double, double, double, double, double, double, int);
//Lax-Fridreich flux function
double LLF_flux(double, double, double, double, double, double, int);
//local Lax-Fridreich flux function
void limiting(double dx, double dt);

double u[nxghst][3][4];      // old solution,[][den,mom,en][0th moment, 1st moment, 2nd moment and mean]
double u_m[nxghst][3][4];      // middle stage solution
double u_n[nxghst][3][4];      // new solution
double char_speed;

std::string fidelity = "100";

int main() {
    double rhoE = 125;
    for (int i = 0; i < 500; i++) {
        rhoE = 125+i*2.5*(100-50)/500;
        std::cout << rhoE << std::endl;
        MainRun(fidelity, rhoE, i);
    }
}


int MainRun(std::string fidelity, double rhoE, int iiii) { // in domain [0,2]
    std::string Case = std::to_string(iiii);
    std::ofstream oden("../Data_Sod_MPR/" + fidelity + "/den_hpr_" + fidelity + "_" +Case, std::ios::out);
    // if (!oden.is_open()) {
    //     std::cerr << "Error: Could not open 'den' for writing." << std::endl;
    //     return 1;
    // }
    std::ofstream ovel("../Data_Sod_MPR/" + fidelity + "/vel_hpr_"+fidelity+"_" + Case, std::ios::out);
    std::ofstream opres("../Data_Sod_MPR/" + fidelity + "/pres_hpr_" + fidelity + "_" + Case, std::ios::out);
    std::ofstream oden2("../Data_Sod_MPR/" + fidelity + "/den2_hpr_" + fidelity + "_" + Case, std::ios::out);
    std::ofstream ox("../Data_Sod_MPR/" + fidelity + "/x_value_hpr_" + fidelity + "_" + Case, std::ios::out);
    std::ofstream ox2("../Data_Sod_MPR/" + fidelity + "/x_value2_hpr_" + fidelity + "_" + Case, std::ios::out);
    int i, j, k, iter, iii;
    double dx, dt, tau = 100000.0, T, Tfinal, theta;  // tau is the max time step
    double den, mom, en, vel, pres;

    dx = (rbdry - lbdry) / nx;
    T = 0.0;
    Tfinal = 0.1;
    init(dx, rhoE);

    for (i = 0; i < nxghst; i++)
        for (j = 0; j < 4; j++)
            for (k = 0; k < 3; k++)
            {
                u_n[i][k][j] = u[i][k][j];
            }

    dt = comp_max_time_step(dx);
    reconstruction(dx);
    limiting(dx, dt);

    iter = 1;
    while (T <= Tfinal && iter == 1)
        //for(iii=0;iii<1;iii++)
    {
        if (T + dt <= Tfinal)T += dt;
        else
        {
            dt = Tfinal - T;
            T = Tfinal;
            iter = 0;
        }

        for (i = 0; i < nxghst; i++)
            for (j = 0; j < 4; j++)
                for (k = 0; k < 3; k++)
                {
                    u_m[i][k][j] = u_n[i][k][j];
                    u[i][k][j] = u_m[i][k][j];
                }

        adv_fw(dx, dt, tau); //forward
        reconstruction(dx);
        limiting(dx, dt);

        for (i = 0; i < nxghst; i++)
            for (j = 0; j < 4; j++)
                for (k = 0; k < 3; k++)
                {
                    u_m[i][k][j] = u_n[i][k][j];
                }

        adv_fw(dx, dt, tau); //forward again

        //mid  averaging
        for (i = 0; i < nxghst; i++)
            for (j = 3; j < 4; j++)
                for (k = 0; k < 3; k++)
                {
                    u_n[i][k][j] = (3.0 * u[i][k][j] + u_n[i][k][j]) / 4.0;
                }
        reconstruction(dx);
        limiting(dx, dt);

        for (i = 0; i < nxghst; i++)
            for (j = 0; j < 4; j++)
                for (k = 0; k < 3; k++)
                {
                    u_m[i][k][j] = u_n[i][k][j];
                }

        adv_fw(dx, dt, tau); //forward again

        //final  averaging
        for (i = 0; i < nxghst; i++)
            for (j = 3; j < 4; j++)
                for (k = 0; k < 3; k++)
                {
                    u_n[i][k][j] = (u[i][k][j] + 2.0 * u_n[i][k][j]) / 3.0;
                }
        reconstruction(dx);
        limiting(dx, dt);

        //  cout<<T<<'\n';
        dt = comp_max_time_step(dx);
    }

    for (i = 0; i < nx; i++)
    {
        den = u_n[mod(i)][0][0] - 0.5 * dx * u_n[mod(i)][0][1] + 0.25 * dx * dx * u_n[mod(i)][0][2];
        mom = u_n[mod(i)][1][0] - 0.5 * dx * u_n[mod(i)][1][1] + 0.25 * dx * dx * u_n[mod(i)][1][2];
        en = u_n[mod(i)][2][0] - 0.5 * dx * u_n[mod(i)][2][1] + 0.25 * dx * dx * u_n[mod(i)][2][2];
        vel = mom / den;
        pres = (en - 0.5 * mom * mom / den) * (gam - 1.0);
        // std::cout << "Writing to 'den': " << den << std::endl;
        oden << den << '\n';
        ovel << vel << '\n';
        opres << pres << '\n';
        ox << lbdry + (i + 0.5) * dx << '\n';
        ox2 << lbdry + i * dx << '\n'; // all left boundaries of cells;
    }
    // calculate right boundary values
    den = u_n[mod(nx - 1)][0][0] + 0.5 * dx * u_n[mod(nx - 1)][0][1] + 0.25 * dx * dx * u_n[mod(nx - 1)][0][2];
    mom = u_n[mod(nx - 1)][1][0] + 0.5 * dx * u_n[mod(nx - 1)][1][1] + 0.25 * dx * dx * u_n[mod(nx - 1)][1][2];
    en = u_n[mod(nx - 1)][2][0] + 0.5 * dx * u_n[mod(nx - 1)][2][1] + 0.25 * dx * dx * u_n[mod(nx - 1)][2][2];
    // write these values
    oden << den << '\n';
    ovel << vel << '\n';
    opres << pres << '\n';
    ox2 << rbdry << '\n'; // add final right boundary

    std::cout << "Execution completed successfully. Check output files." << std::endl;
    return 0; // Explicit exit code 0
}

void adv_fw(double dx, double dt, double tau) { //advance u forward in time, L-F scheme
    int i, j, k;
    double lambda = dt / dx;
    double theta = dt / tau;

    //compute the cell average (mean)
    for (k = 0; k < 3; k++)
        for (i = 0; i < nx; i++)
        {
            u_n[mod(i)][k][3] = u_m[mod(i)][k][3]
                - lambda * (LF_flux(u_m[mod(i)][0][0] + 0.5 * dx * u_m[mod(i)][0][1] + 0.25 * dx * dx * u_m[mod(i)][0][2],
                    u_m[mod(i)][1][0] + 0.5 * dx * u_m[mod(i)][1][1] + 0.25 * dx * dx * u_m[mod(i)][1][2],
                    u_m[mod(i)][2][0] + 0.5 * dx * u_m[mod(i)][2][1] + 0.25 * dx * dx * u_m[mod(i)][2][2],
                    u_m[mod(i + 1)][0][0] - 0.5 * dx * u_m[mod(i + 1)][0][1] + 0.25 * dx * dx * u_m[mod(i + 1)][0][2],
                    u_m[mod(i + 1)][1][0] - 0.5 * dx * u_m[mod(i + 1)][1][1] + 0.25 * dx * dx * u_m[mod(i + 1)][1][2],
                    u_m[mod(i + 1)][2][0] - 0.5 * dx * u_m[mod(i + 1)][2][1] + 0.25 * dx * dx * u_m[mod(i + 1)][2][2],
                    k)
                    - LF_flux(u_m[mod(i - 1)][0][0] + 0.5 * dx * u_m[mod(i - 1)][0][1] + 0.25 * dx * dx * u_m[mod(i - 1)][0][2],
                        u_m[mod(i - 1)][1][0] + 0.5 * dx * u_m[mod(i - 1)][1][1] + 0.25 * dx * dx * u_m[mod(i - 1)][1][2],
                        u_m[mod(i - 1)][2][0] + 0.5 * dx * u_m[mod(i - 1)][2][1] + 0.25 * dx * dx * u_m[mod(i - 1)][2][2],
                        u_m[mod(i)][0][0] - 0.5 * dx * u_m[mod(i)][0][1] + 0.25 * dx * dx * u_m[mod(i)][0][2],
                        u_m[mod(i)][1][0] - 0.5 * dx * u_m[mod(i)][1][1] + 0.25 * dx * dx * u_m[mod(i)][1][2],
                        u_m[mod(i)][2][0] - 0.5 * dx * u_m[mod(i)][2][1] + 0.25 * dx * dx * u_m[mod(i)][2][2],
                        k));
        }
}

double comp_max_time_step(double dx) {
    double c, den, mom, en, p, vel;
    int i;
    char_speed = 0.0;
    for (i = 0; i < nx; i++)
    {
        den = u_n[mod(i)][0][3];
        mom = u_n[mod(i)][1][3];
        en = u_n[mod(i)][2][3];
        vel = mom / den;
        p = (gam - 1.0) * (en - mom * mom / (2.0 * den));
        c = sqrt(gam * p / den);  //sound speed
        char_speed = max(char_speed, max(fabs(vel + c), fabs(vel - c)));
    }
    return CFL * dx / char_speed;
}

double LF_flux(double denl, double moml, double enl, double denr, double momr, double enr, int k) {
    double alpha = 1.0 * char_speed;
    return 0.5 * (flux(denl, moml, enl, k) + flux(denr, momr, enr, k)) +
        0.5 * alpha * ((((k - 1) * (k - 2) / 2) * denl + (-k * (k - 2)) * moml + (k * (k - 1) / 2) * enl) -
            (((k - 1) * (k - 2) / 2) * denr + (-k * (k - 2)) * momr + (k * (k - 1) / 2) * enr));
}

double LLF_flux(double denl, double moml, double enl, double denr, double momr, double enr, int k) {
    double alpha, vell, velr, pl, pr, cl, cr, char_speedl, char_speedr;
    vell = moml / denl;
    pl = (gam - 1.0) * (enl - moml * moml / (2.0 * denl));
    cl = sqrt(gam * pl / denl);  //sound speed
    char_speedl = max(fabs(vell + cl), fabs(vell - cl));

    velr = momr / denr;
    pr = (gam - 1.0) * (enr - momr * momr / (2.0 * denr));
    cr = sqrt(gam * pr / denr);  //sound speed
    char_speedr = max(fabs(velr + cr), fabs(velr - cr));

    alpha = max(char_speedl, char_speedr);

    return 0.5 * (flux(denl, moml, enl, k) + flux(denr, momr, enr, k)) +
        0.5 * alpha * ((((k - 1) * (k - 2) / 2) * denl + (-k * (k - 2)) * moml + (k * (k - 1) / 2) * enl) -
            (((k - 1) * (k - 2) / 2) * denr + (-k * (k - 2)) * momr + (k * (k - 1) / 2) * enr));
}

void reconstruction(double dx) {//ENO polynomial reconstruction from p-w average
    int i, j, k, index;
    double x, fx, fx1, fx2;

    //set boundary condition for u_n
    bdry_cond_u(dx);

    //compute the 0th, 1st and 2nd moment of u_n
    for (k = 0; k < 3; k++)
        for (i = -1; i < nx + 1; i++)
        {
            index = i - 1;
            x = (i - index + 0.5) * dx;
            basis_funct(u_n[mod(index)][k][3] * dx, (u_n[mod(index)][k][3] + u_n[mod(index + 1)][k][3]) * dx,
                (u_n[mod(index)][k][3] + u_n[mod(index + 1)][k][3] + u_n[mod(index + 2)][k][3]) * dx,
                dx, x, &fx, &fx1, &fx2);
            u_n[mod(i)][k][0] = fx;
            u_n[mod(i)][k][1] = fx1;
            u_n[mod(i)][k][2] = 0.5 * fx2;
        }
    //set boundary condition for u_n
    bdry_cond_u(dx);
}

void limiting(double dx, double dt) { //limit the 1st and 2nd derivative, keep the volume unchanged
    int i, j, k, index;
    double tem;
    double uave[nxghst][3]; //cell averages of u_n 
    double epsilon2 = 5.0;
    double h, cell_l[2], cell_r[2], dist[2], c_l, c_r, c_0, d_l, d_r;
    double den_l, mom_l, en_l, p_l, den_r, mom_r, en_r, p_r;

    //define ghost cell values
    bdry_cond_u(dx);

    //compute the conservative average of u_n and v_n  
    for (i = 0; i < nxghst; i++)
        for (k = 0; k < 3; k++)
            uave[i][k] = u_n[i][k][3];

    //deriv limiter
    dist[0] = 0.75 * dx;
    dist[1] = 1.25 * dx;
    for (i = 0; i < nx; i++)
        for (k = 0; k < 3; k++)
        {
            cell_l[0] = u_n[mod(i - 1)][k][1] + 2.0 * u_n[mod(i - 1)][k][2] * 0.25 * dx;
            cell_l[1] = u_n[mod(i - 1)][k][1] - 2.0 * u_n[mod(i - 1)][k][2] * 0.25 * dx;
            cell_r[0] = u_n[mod(i + 1)][k][1] - 2.0 * u_n[mod(i + 1)][k][2] * 0.25 * dx;
            cell_r[1] = u_n[mod(i + 1)][k][1] + 2.0 * u_n[mod(i + 1)][k][2] * 0.25 * dx;
            c_0 = u_n[mod(i)][k][1];
            if (fabs(cell_l[0] - c_0) / dist[0] <= fabs(cell_l[1] - c_0) / dist[1])
            {
                c_l = cell_l[0];
                d_l = dist[0];
            }
            else
            {
                c_l = cell_l[1];
                d_l = dist[1];
            }

            if (fabs(cell_r[0] - c_0) / dist[0] <= fabs(cell_r[1] - c_0) / dist[1])
            {
                c_r = cell_r[0];
                d_r = dist[0];
            }
            else
            {
                c_r = cell_r[1];
                d_r = dist[1];
            }
            u_m[mod(i)][k][2] = (1.0 / 2.0) * minmod((c_0 - c_l) / d_l, (c_r - c_0) / d_r);

            cell_l[0] = u_n[mod(i - 1)][k][0] + u_n[mod(i - 1)][k][1] * 0.25 * dx + u_n[mod(i - 1)][k][2] * dx * dx / 12.0 -
                u_m[mod(i)][k][2] * 7.0 * dx * dx / 12.0;
            cell_l[1] = u_n[mod(i - 1)][k][0] - u_n[mod(i - 1)][k][1] * 0.25 * dx + u_n[mod(i - 1)][k][2] * dx * dx / 12.0 -
                u_m[mod(i)][k][2] * 19.0 * dx * dx / 12.0;
            cell_r[0] = u_n[mod(i + 1)][k][0] - u_n[mod(i + 1)][k][1] * 0.25 * dx + u_n[mod(i + 1)][k][2] * dx * dx / 12.0 -
                u_m[mod(i)][k][2] * 7.0 * dx * dx / 12.0;
            cell_r[1] = u_n[mod(i + 1)][k][0] + u_n[mod(i + 1)][k][1] * 0.25 * dx + u_n[mod(i + 1)][k][2] * dx * dx / 12.0 -
                u_m[mod(i)][k][2] * 19.0 * dx * dx / 12.0;
            c_0 = uave[mod(i)][k] - (u_m[mod(i)][k][2] * dx * dx / 12.0);
            if (fabs(cell_l[0] - c_0) / dist[0] <= fabs(cell_l[1] - c_0) / dist[1])
            {
                c_l = cell_l[0];
                d_l = dist[0];
            }
            else
            {
                c_l = cell_l[1];
                d_l = dist[1];
            }

            if (fabs(cell_r[0] - c_0) / dist[0] <= fabs(cell_r[1] - c_0) / dist[1])
            {
                c_r = cell_r[0];
                d_r = dist[0];
            }
            else
            {
                c_r = cell_r[1];
                d_r = dist[1];
            }
            u_m[mod(i)][k][1] = minmod((c_0 - c_l) / d_l, (c_r - c_0) / d_r);
            u_m[mod(i)][k][0] = c_0;
        }

    for (i = 0; i < nx; i++)
    {
        for (k = 0; k < 3; k++)
            for (j = 0; j < 3; j++)u_n[mod(i)][k][j] = u_m[mod(i)][k][j];

        //detect and fix negative pressure (Xu, Liu and Shu, JCP '09)
        index = 0;
        while (index == 0) {
            {
                den_r = u_n[mod(i)][0][0] + 0.5 * dx * u_n[mod(i)][0][1] + 0.25 * dx * dx * u_n[mod(i)][0][2];
                den_l = u_n[mod(i)][0][0] - 0.5 * dx * u_n[mod(i)][0][1] + 0.25 * dx * dx * u_n[mod(i)][0][2];
                mom_r = u_n[mod(i)][1][0] + 0.5 * dx * u_n[mod(i)][1][1] + 0.25 * dx * dx * u_n[mod(i)][1][2];
                mom_l = u_n[mod(i)][1][0] - 0.5 * dx * u_n[mod(i)][1][1] + 0.25 * dx * dx * u_n[mod(i)][1][2];
                en_r = u_n[mod(i)][2][0] + 0.5 * dx * u_n[mod(i)][2][1] + 0.25 * dx * dx * u_n[mod(i)][2][2];
                en_l = u_n[mod(i)][2][0] - 0.5 * dx * u_n[mod(i)][2][1] + 0.25 * dx * dx * u_n[mod(i)][2][2];
                p_r = (gam - 1.0) * (en_r - mom_r * mom_r / (2.0 * den_r));
                p_l = (gam - 1.0) * (en_l - mom_l * mom_l / (2.0 * den_l));
                if (p_r <= 0.0 || p_l <= 0.0)
                {
                    for (k = 0; k < 3; k++)
                    {
                        for (j = 1; j < 3; j++)u_n[mod(i)][k][j] = 0.5 * u_n[mod(i)][k][j];
                        u_n[mod(i)][k][0] = u_n[mod(i)][k][3] + 0.5 * (u_n[mod(i)][k][0] - u_n[mod(i)][k][3]);
                    }
                }
                else
                    index = 1;
            }
        }

        //define ghost cell values
        bdry_cond_u(dx);
    }
}

void bdry_cond_u(double dx) { // set the boundary condition for u_n's mean; constant BC
    int i;
    for (i = 1; i <= ghst; i++)
    {
        u_n[mod(-i)][0][3] = u_n[mod(0)][0][3];
        u_n[mod(-i)][1][3] = u_n[mod(0)][1][3];
        u_n[mod(-i)][2][3] = u_n[mod(0)][2][3];
        u_n[mod(nx - 1 + i)][0][3] = u_n[mod(nx - 1)][0][3];
        u_n[mod(nx - 1 + i)][1][3] = u_n[mod(nx - 1)][1][3];
        u_n[mod(nx - 1 + i)][2][3] = u_n[mod(nx - 1)][2][3];
    }
}

void basis_funct(double f1, double f2, double f3, double dx, double x, double* fx, double* fx1, double* fx2) {
    // (fx, fx1, fx2) = (u(x), u'(x), u"(x))
    double a = 0.0, b = dx, c = 2.0 * dx, d = 3.0 * dx;

    (*fx) = ((f1 / 2.0) * ((x - c) * (x - d) + (x - a) * (x - d) + (x - a) * (x - c)) +
        (-f2 / 2.0) * ((x - b) * (x - d) + (x - a) * (x - d) + (x - a) * (x - b)) +
        (f3 / 6.0) * ((x - b) * (x - c) + (x - a) * (x - c) + (x - a) * (x - b))) / (dx * dx * dx);
    (*fx1) = ((f1 + f3 / 3.0) * (x - c) + (f1 - f2) * (x - d) + (-f2 + f3 / 3.0) * (x - b) + (f1 - f2 + f3 / 3.0) * (x - a)) / (dx * dx * dx);
    (*fx2) = (3.0 * f1 - 3.0 * f2 + f3) / (dx * dx * dx);
}

double minmod(double a, double b, double c) { //minmod limiter
    if (a >= 0.0 && b >= 0.0 & c >= 0.0)return min(a, min(b, c));
    else if (a <= 0.0 && b <= 0.0 && c <= 0.0)return max(a, max(b, c));
    else return 0.0;
}

double minmod(double a, double b) { //minmod limiter
    if (a >= 0.0 && b >= 0.0)return min(a, b);
    else if (a <= 0.0 && b <= 0.0)return max(a, b);
    else return 0.0;
}

double flux(double den, double mom, double en, int k) {  // define the kth flux function
    if (k == 0)return mom;
    else if (k == 1)return mom * mom / den + (gam - 1.0) * (en - mom * mom / (2.0 * den));
    else return (mom / den) * (en + (gam - 1.0) * (en - mom * mom / (2.0 * den)));
}

double max(double a, double b) {
    if (a >= b)return a;
    else return b;
}

double min(double a, double b) {
    if (a >= b)return b;
    else return a;
}

int mod(int i) { //adjust the index to the coordinate with ghost boundary
    return ghst + i;
}

double sgn(double a) {
    if (a >= 0.0) return 1.0;
    else return -1.0;
}

//void init(double dx) {   // set initial value
//    int i, j, k;
//    double x;
//    for (i = 0;i < nx + 1;i++)
//    {
//        x = lbdry + dx * (i + 0.5);
//        if (x >= 0 && x < 0.1)
//        {
//            u[mod(i)][0][3] = 1.0;
//            u[mod(i)][1][3] = 0.0;
//            u[mod(i)][2][3] = 2500.0;
//        }
//        else if (x >= 0.1 && x < 0.9)
//        {
//            u[mod(i)][0][3] = 1.0;
//            u[mod(i)][1][3] = 0.0;
//            u[mod(i)][2][3] = 0.025;
//        }
//        else
//        {
//            u[mod(i)][0][3] = 1.0;
//            u[mod(i)][1][3] = 0.0;
//            u[mod(i)][2][3] = 250.0;
//        }
//    }
//}

void init(double dx, double rhoE) {   // set initial value
    int i, j, k;
    double x;
    for (i = 0; i < nx + 1; i++)
    {
        x = lbdry + dx * (i + 0.5);
        if (x <= 0.5)
        {
            u[mod(i)][0][3] = 1.0;
            u[mod(i)][1][3] = 0.0;
            u[mod(i)][2][3] = rhoE;
        }
        else
        {
            u[mod(i)][0][3] = 0.125;
            u[mod(i)][1][3] = 0.0;
            u[mod(i)][2][3] = 0.25;
        }
    }
}

// den = u_n[mod(i)][0][0];
// mom = u_n[mod(i)][1][0];
// en  = u_n[mod(i)][2][0];
// vel = mom / den;
// pres = (en - 0.5 * mom * mom / den) * (gam - 1.0);

// mom = vel * den;
// en  = (pres / (gam - 1.0) + 0.5 * mom * mom) / den

// 1. For x ≤ -10:
// - Density(rho) : 2.0
// - Velocity(u) : 0.0
// - Pressure(pl) : Varies between[10 ^ 9, 10 ^ 10](500 cases)

// 2. For x > -10:
// - Density(rho) : 0.001
// - Velocity(u) : 0.0
// - Pressure : 1.0