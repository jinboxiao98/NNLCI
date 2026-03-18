#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <sys/stat.h>

// --- Global Simulation Parameters ---
struct SimParams {
    int nx;
    int ghst = 3;
    int nxghst;
    // Lax problem domain usually matches Sod for code reuse, mapped to [0,1]
    double lbdry = 0.0;
    double rbdry = 1.0;
    double gam = 1.4;
    double CFL = 0.3; 
    double t_final = 0.14; // Lax problem time
    int order = 3;       
    double p_left = 3.528; // Standard Lax P_L baseline
};

SimParams params;
double char_speed;

std::vector<std::vector<std::vector<double>>> u;
std::vector<std::vector<std::vector<double>>> u_m;
std::vector<std::vector<std::vector<double>>> u_n;

// Function Prototypes
void allocate_memory();
void init(double dx);
void adv_fw(double dx, double dt, double tau);
double comp_max_time_step(double dx);
void reconstruction(double dx);
void limiting(double dx, double dt);
void bdry_cond_u();
void basis_funct(double, double, double, double, double, double*, double*, double*);
double minmod(double a, double b, double c);
double minmod(double a, double b);
double flux(double den, double mom, double en, int k);
double LF_flux(double denl, double moml, double enl, double denr, double momr, double enr, int k);
int mod(int i);

int main(int argc, char* argv[]) {
    // Arguments: ./solver_lax [nx] [p_left] [order] [output_id]
    std::string output_id = "test";
    if (argc >= 5) {
        params.nx = std::atoi(argv[1]);
        params.p_left = std::atof(argv[2]);
        params.order = std::atoi(argv[3]);
        output_id = argv[4];
    } else {
        std::cout << "Usage: ./solver_lax [nx] [p_left] [order] [output_id]" << std::endl;
        params.nx = 100;
    }
    
    params.nxghst = params.nx + 2 * params.ghst;
    allocate_memory();

    double dx = (params.rbdry - params.lbdry) / params.nx;
    double dt, tau = 100000.0, T = 0.0;
    
    init(dx);

    if (params.order == 3) {
        reconstruction(dx);
        dt = comp_max_time_step(dx);
        limiting(dx, dt);
    } else {
        // Fix for Order 1: Initialize 0th moment
        for(int i=0; i<params.nxghst; i++)
            for(int k=0; k<3; k++) { 
                u_n[i][k][0] = u_n[i][k][3]; // Value = Mean
                u_n[i][k][1] = 0.0; 
                u_n[i][k][2] = 0.0; 
            }
        bdry_cond_u();
        dt = comp_max_time_step(dx);
    }

    int iter = 0;
    while (T < params.t_final) {
        if (T + dt > params.t_final) dt = params.t_final - T;

        if (params.order == 3) {
            // SSP-RK3
            for (int i = 0; i < params.nxghst; i++)
                for (int j = 0; j < 4; j++)
                    for (int k = 0; k < 3; k++) { u_m[i][k][j] = u_n[i][k][j]; u[i][k][j] = u_m[i][k][j]; }
            adv_fw(dx, dt, tau); reconstruction(dx); limiting(dx, dt);

            for (int i = 0; i < params.nxghst; i++)
                for (int j = 0; j < 4; j++)
                    for (int k = 0; k < 3; k++) u_m[i][k][j] = u_n[i][k][j];
            adv_fw(dx, dt, tau);
            for (int i = 0; i < params.nxghst; i++)
                for (int j = 3; j < 4; j++)
                    for (int k = 0; k < 3; k++) u_n[i][k][j] = (3.0 * u[i][k][j] + u_n[i][k][j]) / 4.0;
            reconstruction(dx); limiting(dx, dt);

            for (int i = 0; i < params.nxghst; i++)
                for (int j = 0; j < 4; j++)
                    for (int k = 0; k < 3; k++) u_m[i][k][j] = u_n[i][k][j];
            adv_fw(dx, dt, tau);
            for (int i = 0; i < params.nxghst; i++)
                for (int j = 3; j < 4; j++)
                    for (int k = 0; k < 3; k++) u_n[i][k][j] = (u[i][k][j] + 2.0 * u_n[i][k][j]) / 3.0;
            reconstruction(dx); limiting(dx, dt);
        } else {
            // Forward Euler (Order 1) - FIXED
            for (int i = 0; i < params.nxghst; i++)
                for (int j = 0; j < 4; j++)
                    for (int k = 0; k < 3; k++) u_m[i][k][j] = u_n[i][k][j];
            
            // CRITICAL FIX: Set 0th moment to Mean (Prevent divide by zero)
            for (int i = 0; i < params.nxghst; i++)
                for (int k = 0; k < 3; k++) { 
                    u_m[i][k][0] = u_m[i][k][3]; // Fix
                    u_m[i][k][1] = 0.0; 
                    u_m[i][k][2] = 0.0; 
                }

            adv_fw(dx, dt, tau); 
            bdry_cond_u();
        }

        T += dt;
        dt = comp_max_time_step(dx);
        iter++;
    }

    // Write Output (Standard format, Python handles renaming)
    std::string filename = "data/sol_" + output_id + ".dat";
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        mkdir("data", 0777);
        outfile.open(filename);
    }
    
    outfile << "x den vel pres\n";

    for (int i = 0; i < params.nx; i++) {
        double den = u_n[mod(i)][0][3];
        double mom = u_n[mod(i)][1][3];
        double en  = u_n[mod(i)][2][3];
        double vel = mom / den;
        double pres = (en - 0.5 * mom * mom / den) * (params.gam - 1.0);
        double x = params.lbdry + (i + 0.5) * dx;
        outfile << x << " " << den << " " << vel << " " << pres << "\n";
    }
    outfile.close();

    return 0;
}

// --- Specific Logic for Lax Problem ---
void init(double dx) {
    // Lax Problem Constants
    // Left: rho=0.445, u=0.698, p=Variable (DoE)
    // Right: rho=0.5, u=0.0, p=0.571
    double rho_L = 0.445;
    double u_L   = 0.698;
    double p_L   = params.p_left; // Perturbed
    
    double rho_R = 0.5;
    double u_R   = 0.0;
    double p_R   = 0.571;

    for (int i = 0; i < params.nx + 1; i++) {
        double x = params.lbdry + dx * (i + 0.5);
        if (x <= 0.5) { 
            u[mod(i)][0][3] = rho_L;
            u[mod(i)][1][3] = rho_L * u_L;
            u[mod(i)][2][3] = p_L / (params.gam - 1.0) + 0.5 * rho_L * u_L * u_L;
        } else {
            u[mod(i)][0][3] = rho_R;
            u[mod(i)][1][3] = rho_R * u_R;
            u[mod(i)][2][3] = p_R / (params.gam - 1.0) + 0.5 * rho_R * u_R * u_R;
        }
    }
    for (int i = 0; i < params.nxghst; i++)
        for (int k = 0; k < 3; k++)
            for (int j = 0; j < 4; j++) u_n[i][k][j] = u[i][k][j];
            
    bdry_cond_u();
}

// --- Boilerplate Helpers (Same as Sod) ---
void allocate_memory() {
    u.resize(params.nxghst, std::vector<std::vector<double>>(3, std::vector<double>(4, 0.0)));
    u_m.resize(params.nxghst, std::vector<std::vector<double>>(3, std::vector<double>(4, 0.0)));
    u_n.resize(params.nxghst, std::vector<std::vector<double>>(3, std::vector<double>(4, 0.0)));
}
int mod(int i) { return params.ghst + i; }
void bdry_cond_u() {
    for (int i = 1; i <= params.ghst; i++) {
        for(int k=0; k<3; k++) {
            u_n[mod(-i)][k][3] = u_n[mod(0)][k][3];
            u_n[mod(params.nx - 1 + i)][k][3] = u_n[mod(params.nx - 1)][k][3];
        }
    }
}
double comp_max_time_step(double dx) {
    double c, den, mom, en, p, vel;
    char_speed = 0.0;
    for (int i = 0; i < params.nx; i++) {
        den = u_n[mod(i)][0][3];
        mom = u_n[mod(i)][1][3];
        en = u_n[mod(i)][2][3];
        vel = mom / den;
        p = (params.gam - 1.0) * (en - mom * mom / (2.0 * den));
        c = sqrt(params.gam * p / den);
        char_speed = std::max(char_speed, std::max(fabs(vel + c), fabs(vel - c)));
    }
    return params.CFL * dx / char_speed;
}
void adv_fw(double dx, double dt, double tau) {
    double lambda = dt / dx;
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < params.nx; i++) {
            double flux_diff = 
                LF_flux(
                    u_m[mod(i)][0][0] + 0.5 * dx * u_m[mod(i)][0][1] + 0.25 * dx * dx * u_m[mod(i)][0][2],
                    u_m[mod(i)][1][0] + 0.5 * dx * u_m[mod(i)][1][1] + 0.25 * dx * dx * u_m[mod(i)][1][2],
                    u_m[mod(i)][2][0] + 0.5 * dx * u_m[mod(i)][2][1] + 0.25 * dx * dx * u_m[mod(i)][2][2],
                    u_m[mod(i + 1)][0][0] - 0.5 * dx * u_m[mod(i + 1)][0][1] + 0.25 * dx * dx * u_m[mod(i + 1)][0][2],
                    u_m[mod(i + 1)][1][0] - 0.5 * dx * u_m[mod(i + 1)][1][1] + 0.25 * dx * dx * u_m[mod(i + 1)][1][2],
                    u_m[mod(i + 1)][2][0] - 0.5 * dx * u_m[mod(i + 1)][2][1] + 0.25 * dx * dx * u_m[mod(i + 1)][2][2],
                    k
                ) - 
                LF_flux(
                    u_m[mod(i - 1)][0][0] + 0.5 * dx * u_m[mod(i - 1)][0][1] + 0.25 * dx * dx * u_m[mod(i - 1)][0][2],
                    u_m[mod(i - 1)][1][0] + 0.5 * dx * u_m[mod(i - 1)][1][1] + 0.25 * dx * dx * u_m[mod(i - 1)][1][2],
                    u_m[mod(i - 1)][2][0] + 0.5 * dx * u_m[mod(i - 1)][2][1] + 0.25 * dx * dx * u_m[mod(i - 1)][2][2],
                    u_m[mod(i)][0][0] - 0.5 * dx * u_m[mod(i)][0][1] + 0.25 * dx * dx * u_m[mod(i)][0][2],
                    u_m[mod(i)][1][0] - 0.5 * dx * u_m[mod(i)][1][1] + 0.25 * dx * dx * u_m[mod(i)][1][2],
                    u_m[mod(i)][2][0] - 0.5 * dx * u_m[mod(i)][2][1] + 0.25 * dx * dx * u_m[mod(i)][2][2],
                    k
                );
            u_n[mod(i)][k][3] = u_m[mod(i)][k][3] - lambda * flux_diff;
        }
    }
}
double LF_flux(double denl, double moml, double enl, double denr, double momr, double enr, int k) {
    double alpha = 1.0 * char_speed;
    return 0.5 * (flux(denl, moml, enl, k) + flux(denr, momr, enr, k)) +
        0.5 * alpha * ((((k - 1) * (k - 2) / 2.0) * denl + (-k * (k - 2)) * moml + (k * (k - 1) / 2.0) * enl) -
            (((k - 1) * (k - 2) / 2.0) * denr + (-k * (k - 2)) * momr + (k * (k - 1) / 2.0) * enr));
}
double flux(double den, double mom, double en, int k) {
    if (k == 0) return mom;
    else if (k == 1) return mom * mom / den + (params.gam - 1.0) * (en - mom * mom / (2.0 * den));
    else return (mom / den) * (en + (params.gam - 1.0) * (en - mom * mom / (2.0 * den)));
}
void reconstruction(double dx) {
    int index;
    double x, fx, fx1, fx2;
    bdry_cond_u();
    for (int k = 0; k < 3; k++)
        for (int i = -1; i < params.nx + 1; i++) {
            index = i - 1;
            x = (i - index + 0.5) * dx;
            basis_funct(u_n[mod(index)][k][3] * dx, (u_n[mod(index)][k][3] + u_n[mod(index + 1)][k][3]) * dx,
                (u_n[mod(index)][k][3] + u_n[mod(index + 1)][k][3] + u_n[mod(index + 2)][k][3]) * dx,
                dx, x, &fx, &fx1, &fx2);
            u_n[mod(i)][k][0] = fx;
            u_n[mod(i)][k][1] = fx1;
            u_n[mod(i)][k][2] = 0.5 * fx2;
        }
    bdry_cond_u();
}
void limiting(double dx, double dt) {
    int index;
    double cell_l[2], cell_r[2], dist[2], c_l, c_r, c_0, d_l, d_r;
    double uave_k; 
    bdry_cond_u();
    dist[0] = 0.75 * dx; dist[1] = 1.25 * dx;
    for (int i = 0; i < params.nx; i++) {
        for (int k = 0; k < 3; k++) {
            uave_k = u_n[mod(i)][k][3];
            c_0 = u_n[mod(i)][k][1];
            cell_l[0] = u_n[mod(i - 1)][k][1] + 2.0 * u_n[mod(i - 1)][k][2] * 0.25 * dx;
            cell_l[1] = u_n[mod(i - 1)][k][1] - 2.0 * u_n[mod(i - 1)][k][2] * 0.25 * dx;
            cell_r[0] = u_n[mod(i + 1)][k][1] - 2.0 * u_n[mod(i + 1)][k][2] * 0.25 * dx;
            cell_r[1] = u_n[mod(i + 1)][k][1] + 2.0 * u_n[mod(i + 1)][k][2] * 0.25 * dx;
            if (fabs(cell_l[0] - c_0) / dist[0] <= fabs(cell_l[1] - c_0) / dist[1]) { c_l = cell_l[0]; d_l = dist[0]; } else { c_l = cell_l[1]; d_l = dist[1]; }
            if (fabs(cell_r[0] - c_0) / dist[0] <= fabs(cell_r[1] - c_0) / dist[1]) { c_r = cell_r[0]; d_r = dist[0]; } else { c_r = cell_r[1]; d_r = dist[1]; }
            u_m[mod(i)][k][2] = 0.5 * minmod((c_0 - c_l) / d_l, (c_r - c_0) / d_r);
            cell_l[0] = u_n[mod(i - 1)][k][0] + u_n[mod(i - 1)][k][1] * 0.25 * dx + u_n[mod(i - 1)][k][2] * dx * dx / 12.0 - u_m[mod(i)][k][2] * 7.0 * dx * dx / 12.0;
            cell_l[1] = u_n[mod(i - 1)][k][0] - u_n[mod(i - 1)][k][1] * 0.25 * dx + u_n[mod(i - 1)][k][2] * dx * dx / 12.0 - u_m[mod(i)][k][2] * 19.0 * dx * dx / 12.0;
            cell_r[0] = u_n[mod(i + 1)][k][0] - u_n[mod(i + 1)][k][1] * 0.25 * dx + u_n[mod(i + 1)][k][2] * dx * dx / 12.0 - u_m[mod(i)][k][2] * 7.0 * dx * dx / 12.0;
            cell_r[1] = u_n[mod(i + 1)][k][0] + u_n[mod(i + 1)][k][1] * 0.25 * dx + u_n[mod(i + 1)][k][2] * dx * dx / 12.0 - u_m[mod(i)][k][2] * 19.0 * dx * dx / 12.0;
            c_0 = uave_k - (u_m[mod(i)][k][2] * dx * dx / 12.0);
            if (fabs(cell_l[0] - c_0) / dist[0] <= fabs(cell_l[1] - c_0) / dist[1]) { c_l = cell_l[0]; d_l = dist[0]; } else { c_l = cell_l[1]; d_l = dist[1]; }
            if (fabs(cell_r[0] - c_0) / dist[0] <= fabs(cell_r[1] - c_0) / dist[1]) { c_r = cell_r[0]; d_r = dist[0]; } else { c_r = cell_r[1]; d_r = dist[1]; }
            u_m[mod(i)][k][1] = minmod((c_0 - c_l) / d_l, (c_r - c_0) / d_r);
            u_m[mod(i)][k][0] = c_0;
        }
    }
    for (int i = 0; i < params.nx; i++) 
        for (int k = 0; k < 3; k++)
            for (int j = 0; j < 3; j++) u_n[mod(i)][k][j] = u_m[mod(i)][k][j];
    bdry_cond_u();
}
void basis_funct(double f1, double f2, double f3, double dx, double x, double* fx, double* fx1, double* fx2) {
    double a = 0.0, b = dx, c = 2.0 * dx, d = 3.0 * dx;
    (*fx) = ((f1 / 2.0) * ((x - c) * (x - d) + (x - a) * (x - d) + (x - a) * (x - c)) +
        (-f2 / 2.0) * ((x - b) * (x - d) + (x - a) * (x - d) + (x - a) * (x - b)) +
        (f3 / 6.0) * ((x - b) * (x - c) + (x - a) * (x - c) + (x - a) * (x - b))) / (dx * dx * dx);
    (*fx1) = ((f1 + f3 / 3.0) * (x - c) + (f1 - f2) * (x - d) + (-f2 + f3 / 3.0) * (x - b) + (f1 - f2 + f3 / 3.0) * (x - a)) / (dx * dx * dx);
    (*fx2) = (3.0 * f1 - 3.0 * f2 + f3) / (dx * dx * dx);
}
double minmod(double a, double b, double c) {
    if (a >= 0.0 && b >= 0.0 && c >= 0.0) return std::min(a, std::min(b, c));
    else if (a <= 0.0 && b <= 0.0 && c <= 0.0) return std::max(a, std::max(b, c));
    else return 0.0;
}
double minmod(double a, double b) {
    if (a >= 0.0 && b >= 0.0) return std::min(a, b);
    else if (a <= 0.0 && b <= 0.0) return std::max(a, b);
    else return 0.0;
}