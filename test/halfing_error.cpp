#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <bitset>
#include <numeric>
#include "utils.hpp"
#include "Reconstructor/Reconstructor.hpp"
#include "Synthesizer4GE.hpp"

using namespace MDR;

// f(x) = x^2
template <class T>
inline double compute_bound_x_square(T x, T eb){
	return 2*fabs(x)*eb + eb*eb;
}

// f(x) = x^2
template <class T>
inline double compute_inverse_bound_x_square(T x, T eb, T tau){
	return sqrt(x*x + tau) - x*x;
}

// f(x) = sqrt(x)
template <class T>
inline double compute_bound_square_root_x(T x, T eb){
	if(x == 0) {
		return sqrt(eb);
	}
	if(x > eb){
		return eb / (sqrt(x - eb) + sqrt(x));
	}
	else{
		return eb / sqrt(x);
	}
}

// f(x) = sqrt(x)
template <class T>
inline double compute_inverse_bound_square_root_x(T x, T eb, T tau){
	if(x == 0){
		return tau * tau;
	}
	if(x > eb){
		return tau * (sqrt(x - eb) + sqrt(x));
	}
	else{
		return tau * sqrt(x);
	}
}

// f(x) = 1/(x+a)
template <class T>
inline double compute_bound_radical(T x, T a, T eb){
	if(fabs(x+a) > eb){
		if(x+a > 0){
			return eb / ((x+a-eb)*fabs(x+a));
		}
		else{
			return eb / ((x+a+eb)*fabs(x+a));
		}
	}
	else{
		std::cout << "Warning: cannot control error in 1/(x+a)\n";
		return 0;		
	}
}

// f(x) = 1/(x+a)
template <class T>
inline double compute_inverse_bound_radical(T x, T a, T eb, T tau){
	if(fabs(x+a) > eb){
		if(x+a > 0){
			return tau * ((x+a-eb)*fabs(x+a));
		}
		else{
			return tau * ((x+a+eb)*fabs(x+a));
		}
	}
	else{
		std::cout << "Warning: cannot control error in 1/(x+a)\n";
		return 0;		
	}
}

template <class T>
inline double compute_bound_multiplication(T x, T y, T eb_x, T eb_y){
	return fabs(x)*eb_y + fabs(y)*eb_x + eb_x*eb_y;
}

// f(x, y) = x/y
template <class T>
inline double compute_bound_division(T x, T y, T eb_x, T eb_y){
	if(eb_y < fabs(y)){
		double e = fabs(x)*eb_y + fabs(y)*eb_x;
		if(y > 0) return e / (y*(y - eb_y));
		else return e/ (y*(y + eb_y));
	}
	else{
		std::cout << "Warning: cannot control error in x/y\n";
		return 0;
	}
}

template <class T>
void print_error(std::string varname, T dec, T ori, T est){
	std::cout << varname << ": dec = " << dec << ", ori = " << ori << ", error = " << dec - ori << ", est = " << est << std::endl; 
}

template <class T>
double UniformTighteningVTOT(T Vx, T Vy, T Vz, T eb_Vx, T eb_Vy, T eb_Vz, T tau){
	// error of total velocity square
	double e_V_TOT_2 = compute_bound_x_square(Vx, eb_Vx) + compute_bound_x_square(Vy, eb_Vy) + compute_bound_x_square(Vz, eb_Vz);
	double V_TOT_2 = Vx*Vx + Vy*Vy + Vz*Vz;
	// error of total velocity
	double e_V_TOT = compute_bound_square_root_x(V_TOT_2, e_V_TOT_2);
	double V_TOT = sqrt(V_TOT_2);
	double new_e_V_TOT_2 = compute_inverse_bound_square_root_x(V_TOT, e_V_TOT, tau);
	// compute assignment on Vx, Vy, Vz
	double e1 = 2*fabs(Vx)*eb_Vx + 2*fabs(Vy)*eb_Vy + 2*fabs(Vz)*eb_Vz;
	double e2 = eb_Vx*eb_Vx + eb_Vy*eb_Vy  + eb_Vz*eb_Vz;
	// assuming the same scale on error bound deduction
	double scale = (-e1 + sqrt(e1*e1 + 4*e2*new_e_V_TOT_2))/(2*e2);
	return scale;
}

template <class T>
double UniformTighteningT(T P, T D, T eb_P, T eb_D, T c_1, T tau){
	double e_T = c_1 * compute_bound_division(P, D, eb_P, eb_D);
	return tau / e_T;	
}

std::vector<double> P_ori;
std::vector<double> D_ori;
std::vector<double> Vx_ori;
std::vector<double> Vy_ori;
std::vector<double> Vz_ori;
double * P_dec = NULL;
double * D_dec = NULL;
double * Vx_dec = NULL;
double * Vy_dec = NULL;
double * Vz_dec = NULL;
double * V_TOT_ori = NULL;
double * Temp_ori = NULL;
double * C_ori = NULL;
double * Mach_ori = NULL;
double * PT_ori = NULL;
double * mu_ori = NULL;
std::vector<double> error_V_TOT;
std::vector<double> error_Temp;
std::vector<double> error_C;
std::vector<double> error_Mach;
std::vector<double> error_PT;
std::vector<double> error_mu;
std::vector<double> error_est_V_TOT;
std::vector<double> error_est_Temp;
std::vector<double> error_est_C;
std::vector<double> error_est_Mach;
std::vector<double> error_est_PT;
std::vector<double> error_est_mu;
const std::vector<std::string> names{"V_TOT", "T", "C", "Mach", "PT", "mu"};
// estimate error from decompressed data and error bound
// return next eb
template <class T>
void estimate_error(const T * Vx, const T * Vy, const T * Vz, const T * P, const T * D, size_t n, const std::vector<double>& tau, std::vector<double>& ebs){

	double eb_Vx = ebs[0];
	double eb_Vy = ebs[1];
	double eb_Vz = ebs[2];
	double eb_P = ebs[3];
	double eb_D = ebs[4];
	double R = 287.1;
	double gamma = 1.4;
	double mi = 3.5;
	double mu_r = 1.716e-5;
	double T_r = 273.15;
	double S = 110.4;
	double c_1 = 1.0 / R;
	double c_2 = sqrt(gamma * R);
	double c_3 = T_r + S;
	int C7i[8] = {1, 7, 21, 35, 35, 21, 7, 1};
	std::vector<double> max_values(6);
	std::vector<int> max_index(6);
	for(int i=0; i<n; i++){
		// error of total velocity square
		double e_V_TOT_2 = compute_bound_x_square(Vx[i], eb_Vx) + compute_bound_x_square(Vy[i], eb_Vy) + compute_bound_x_square(Vz[i], eb_Vz);
		double V_TOT_2 = Vx[i]*Vx[i] + Vy[i]*Vy[i] + Vz[i]*Vz[i];
		// error of total velocity
		double e_V_TOT = compute_bound_square_root_x(V_TOT_2, e_V_TOT_2);
		double V_TOT = sqrt(V_TOT_2);
		// print_error("V_TOT", V_TOT, V_TOT_ori[i], e_V_TOT);
		// error of temperature
		double e_T = c_1 * compute_bound_division(P[i], D[i], eb_P, eb_D);
		double Temp = P[i] / (D[i] * R);
		// print_error("T", Temp, Temp_ori[i], e_T);
		// error of C
		double e_C = c_2*compute_bound_square_root_x(Temp, e_T);
		double C = c_2 * sqrt(Temp);
		// print_error("C", C, C_ori[i], e_C);
		double e_Mach = compute_bound_division(V_TOT, C, e_V_TOT, e_C);
		double Mach = V_TOT / C;
		// print_error("Mach", Mach, Mach_ori[i], e_Mach);
		// 1 + gamma/2 *Mach*Mach
		double e_Mach_tmp = (gamma-1) / 2 * compute_bound_x_square(Mach, e_Mach);
		double Mach_tmp = 1 + (gamma-1)/2 * Mach * Mach;
		double e_Mach_tmp_mi = 0;
		for(int i=1; i<=7; i++){
			e_Mach_tmp_mi += C7i[i] * pow(Mach_tmp, 7-i) * pow(e_Mach_tmp, i);
		}
		double Mach_tmp_mi = sqrt(pow(Mach_tmp, 7));
		double e_PT = compute_bound_multiplication(P[i], Mach_tmp_mi, eb_P, e_Mach_tmp_mi);
		double PT = P[i] * Mach_tmp_mi;
		// print_error("PT", PT, PT_ori[i], e_PT);
		double e_TrS_TS = c_3 * compute_bound_radical(Temp, S, e_T);
		double TrS_TS = c_3 / (Temp + S);
		double e_T_Tr_3 = 3*pow(Temp/T_r, 2)*(e_T/T_r) + 3*Temp/T_r*(e_T/T_r)*(e_T/T_r) + (e_T/T_r)*(e_T/T_r)*(e_T/T_r);
		double T_Tr_3 = pow(Temp/T_r, 3);
		double e_T_Tr_3_sqrt = compute_bound_square_root_x(T_Tr_3, e_T_Tr_3);
		double T_Tr_3_sqrt = sqrt(T_Tr_3);
		double e_mu = mu_r * compute_bound_multiplication(T_Tr_3_sqrt, TrS_TS, e_T_Tr_3_sqrt, e_TrS_TS);
		double mu = mu_r * T_Tr_3_sqrt * TrS_TS;
		// print_error("mu", mu, mu_ori[i], e_mu);
		// if(i == 7618){
		// 	std::cout << eb_Vx << " " << eb_Vy << " " << eb_Vz << " " << eb_P << " " << eb_D << std::endl;
		// 	std::cout << "ori data = " << Vx_ori[i] << ", " << Vy_ori[i] << ", " << Vz_ori[i] << ", " << P_ori[i] << ", " << D_ori[i] << std::endl;
		// 	std::cout << "dec data = " << Vx[i] << ", " << Vy[i] << ", " << Vz[i] << ", " << P[i] << ", " << D[i] << std::endl;
		// 	print_error("V_TOT", V_TOT, V_TOT_ori[i], e_V_TOT);
		// 	print_error("V_TOT_2", V_TOT_2, Vx_ori[i]*Vx_ori[i] + Vy_ori[i]*Vy_ori[i] + Vz_ori[i]*Vz_ori[i], e_V_TOT_2);
		// 	print_error("T", Temp, Temp_ori[i], e_T);
		// 	print_error("C", C, C_ori[i], e_C);
		// 	print_error("Mach", Mach, Mach_ori[i], e_Mach);
		// 	print_error("PT", PT, PT_ori[i], e_PT);
		// 	print_error("mu", mu, mu_ori[i], e_mu);
		// 	std::cout << "Mach_tmp: " << Mach_tmp << " " << e_Mach_tmp << std::endl;
		// 	std::cout << "Mach_tmp_mi: " << Mach_tmp_mi << " " << e_Mach_tmp_mi << std::endl;
		// 	// exit(-1);
		// }
		error_est_V_TOT[i] = e_V_TOT;
		error_est_Temp[i] = e_T;
		error_est_C[i] = e_C;
		error_est_Mach[i] = e_Mach;
		error_est_PT[i] = e_PT;
		error_est_mu[i] = e_mu;
		error_V_TOT[i] = V_TOT - V_TOT_ori[i];
		error_Temp[i] = Temp - Temp_ori[i];
		error_C[i] = C - C_ori[i];
		error_Mach[i] = Mach - Mach_ori[i];
		error_PT[i] = PT - PT_ori[i];
		error_mu[i] = mu - mu_ori[i];
		if(max_values[0] < error_est_V_TOT[i]){
			max_values[0] = error_est_V_TOT[i];
			max_index[0] = i;
		}
		if(max_values[1] < error_est_Temp[i]){
			max_values[1] = error_est_Temp[i];
			max_index[1] = i;
		}
		if(max_values[2] < error_est_C[i]){
			max_values[2] = error_est_C[i];
			max_index[2] = i;
		}
		if(max_values[3] < error_est_Mach[i]){
			max_values[3] = error_est_Mach[i];
			max_index[3] = i;
		}
		if(max_values[4] < error_est_PT[i]){
			max_values[4] = error_est_PT[i];
			max_index[4] = i;
		}
		if(max_values[5] < error_est_mu[i]){
			max_values[5] = error_est_mu[i];
			max_index[5] = i;
		}
	}
	for(int i=0; i<6; i++){
		std::cout << names[i] << ": max estimated error = " << max_values[i] << ", index = " << max_index[i] << std::endl;
	}

	// estimate error bound based on maximal errors
	{
		std::vector<double> new_ebs(5, 0);
		int i = max_index[0];
		// assuming the same scale on error bound deduction
		double scale = UniformTighteningVTOT(Vx[i], Vy[i], Vz[i], eb_Vx, eb_Vy, eb_Vz, tau[0]);
		if(scale < 1){
			new_ebs[0] = ebs[0] * scale;
			new_ebs[1] = ebs[1] * scale;
			new_ebs[2] = ebs[2] * scale;			
		}
		// error of temperature
		i = max_index[1];
		scale = UniformTighteningT(P[i], D[i], eb_P, eb_D, c_1, tau[1]);
		if(scale < 1){
			new_ebs[3] = ebs[3] * scale;
			new_ebs[4] = ebs[4] * scale;
		}
		// error of C
		// double e_C = c_2*compute_bound_square_root_x(Temp, e_T);
		// double C = c_2 * sqrt(Temp);
		// double new_C_2 = compute_inverse_bound_square_root_x(Temp, e_V_TOT, tau[0]);
		// // print_error("C", C, C_ori[i], e_C);
		// double e_Mach = compute_bound_division(V_TOT, C, e_V_TOT, e_C);
		// double Mach = V_TOT / C;
		// // print_error("Mach", Mach, Mach_ori[i], e_Mach);
		// // 1 + gamma/2 *Mach*Mach
		// double e_Mach_tmp = (gamma-1) / 2 * compute_bound_x_square(Mach, e_Mach);
		// double Mach_tmp = 1 + (gamma-1)/2 * Mach * Mach;
		// double e_Mach_tmp_mi = 0;
		// for(int i=1; i<=7; i++){
		// 	e_Mach_tmp_mi += C7i[i] * pow(Mach_tmp, 7-i) * pow(e_Mach_tmp, i);
		// }
		// double Mach_tmp_mi = sqrt(pow(Mach_tmp, 7));
		// double e_PT = compute_bound_multiplication(P[i], Mach_tmp_mi, eb_P, e_Mach_tmp_mi);
		// double PT = P[i] * Mach_tmp_mi;
		// // print_error("PT", PT, PT_ori[i], e_PT);
		// double e_TrS_TS = c_3 * compute_bound_radical(Temp, S, e_T);
		// double TrS_TS = c_3 / (Temp + S);
		// double e_T_Tr_3 = 3*pow(Temp/T_r, 2)*(e_T/T_r) + 3*Temp/T_r*(e_T/T_r)*(e_T/T_r) + (e_T/T_r)*(e_T/T_r)*(e_T/T_r);
		// double T_Tr_3 = pow(Temp/T_r, 3);
		// double e_T_Tr_3_sqrt = compute_bound_square_root_x(T_Tr_3, e_T_Tr_3);
		// double T_Tr_3_sqrt = sqrt(T_Tr_3);
		// double e_mu = mu_r * compute_bound_multiplication(T_Tr_3_sqrt, TrS_TS, e_T_Tr_3_sqrt, e_TrS_TS);
		// double mu = mu_r * T_Tr_3_sqrt * TrS_TS;	
		for(int i=0; i<5; i++){
			ebs[i] = new_ebs[i];
		}	
	}
}

template<class T>
void halfing_error(const T * Vx, const T * Vy, const T * Vz, const T * P, const T * D, size_t n, const std::vector<double>& tau, std::vector<double>& ebs){
	double eb_Vx = ebs[0];
	double eb_Vy = ebs[1];
	double eb_Vz = ebs[2];
	double eb_P = ebs[3];
	double eb_D = ebs[4];
	double R = 287.1;
	double gamma = 1.4;
	double mi = 3.5;
	double mu_r = 1.716e-5;
	double T_r = 273.15;
	double S = 110.4;
	double c_1 = 1.0 / R;
	double c_2 = sqrt(gamma * R);
	double c_3 = T_r + S;
	int C7i[8] = {1, 7, 21, 35, 35, 21, 7, 1};
	std::vector<double> max_values(6);
	std::vector<int> max_index(6);
	for(int i=0; i<n; i++){
		// error of total velocity square
		double e_V_TOT_2 = compute_bound_x_square(Vx[i], eb_Vx) + compute_bound_x_square(Vy[i], eb_Vy) + compute_bound_x_square(Vz[i], eb_Vz);
		double V_TOT_2 = Vx[i]*Vx[i] + Vy[i]*Vy[i] + Vz[i]*Vz[i];
		// error of total velocity
		double e_V_TOT = compute_bound_square_root_x(V_TOT_2, e_V_TOT_2);
		double V_TOT = sqrt(V_TOT_2);
		// print_error("V_TOT", V_TOT, V_TOT_ori[i], e_V_TOT);
		// error of temperature
		double e_T = c_1 * compute_bound_division(P[i], D[i], eb_P, eb_D);
		double Temp = P[i] / (D[i] * R);
		// print_error("T", Temp, Temp_ori[i], e_T);
		// error of C
		double e_C = c_2*compute_bound_square_root_x(Temp, e_T);
		double C = c_2 * sqrt(Temp);
		// print_error("C", C, C_ori[i], e_C);
		double e_Mach = compute_bound_division(V_TOT, C, e_V_TOT, e_C);
		double Mach = V_TOT / C;
		// print_error("Mach", Mach, Mach_ori[i], e_Mach);
		// 1 + gamma/2 *Mach*Mach
		double e_Mach_tmp = (gamma-1) / 2 * compute_bound_x_square(Mach, e_Mach);
		double Mach_tmp = 1 + (gamma-1)/2 * Mach * Mach;
		double e_Mach_tmp_mi = 0;
		for(int i=1; i<=7; i++){
			e_Mach_tmp_mi += C7i[i] * pow(Mach_tmp, 7-i) * pow(e_Mach_tmp, i);
		}
		double Mach_tmp_mi = sqrt(pow(Mach_tmp, 7));
		double e_PT = compute_bound_multiplication(P[i], Mach_tmp_mi, eb_P, e_Mach_tmp_mi);
		double PT = P[i] * Mach_tmp_mi;
		// print_error("PT", PT, PT_ori[i], e_PT);
		double e_TrS_TS = c_3 * compute_bound_radical(Temp, S, e_T);
		double TrS_TS = c_3 / (Temp + S);
		double e_T_Tr_3 = 3*pow(Temp/T_r, 2)*(e_T/T_r) + 3*Temp/T_r*(e_T/T_r)*(e_T/T_r) + (e_T/T_r)*(e_T/T_r)*(e_T/T_r);
		double T_Tr_3 = pow(Temp/T_r, 3);
		double e_T_Tr_3_sqrt = compute_bound_square_root_x(T_Tr_3, e_T_Tr_3);
		double T_Tr_3_sqrt = sqrt(T_Tr_3);
		double e_mu = mu_r * compute_bound_multiplication(T_Tr_3_sqrt, TrS_TS, e_T_Tr_3_sqrt, e_TrS_TS);
		double mu = mu_r * T_Tr_3_sqrt * TrS_TS;

		error_est_V_TOT[i] = e_V_TOT;
		error_est_Temp[i] = e_T;
		error_est_C[i] = e_C;
		error_est_Mach[i] = e_Mach;
		error_est_PT[i] = e_PT;
		error_est_mu[i] = e_mu;
		error_V_TOT[i] = V_TOT - V_TOT_ori[i];
		error_Temp[i] = Temp - Temp_ori[i];
		error_C[i] = C - C_ori[i];
		error_Mach[i] = Mach - Mach_ori[i];
		error_PT[i] = PT - PT_ori[i];
		error_mu[i] = mu - mu_ori[i];
		if(max_values[0] < error_est_V_TOT[i]){
			max_values[0] = error_est_V_TOT[i];
			max_index[0] = i;
		}
		if(max_values[1] < error_est_Temp[i]){
			max_values[1] = error_est_Temp[i];
			max_index[1] = i;
		}
		if(max_values[2] < error_est_C[i]){
			max_values[2] = error_est_C[i];
			max_index[2] = i;
		}
		if(max_values[3] < error_est_Mach[i]){
			max_values[3] = error_est_Mach[i];
			max_index[3] = i;
		}
		if(max_values[4] < error_est_PT[i]){
			max_values[4] = error_est_PT[i];
			max_index[4] = i;
		}
		if(max_values[5] < error_est_mu[i]){
			max_values[5] = error_est_mu[i];
			max_index[5] = i;
		}
	}
	for(int i=0; i<6; i++){
		std::cout << names[i] << ": max estimated error = " << max_values[i] << ", index = " << max_index[i] << std::endl;
	}

    bool reset_PD = false, reset_Va = false;
	// estimate error bound based on maximal errors
	{
		std::vector<double> new_ebs(5, 0);
		// error of Vtotal
        if(max_values[0] > tau[0]){
            reset_Va = true;
			new_ebs[0] = ebs[0] / 2;
			new_ebs[1] = ebs[1] / 2;
			new_ebs[2] = ebs[2] / 2;            
        }
		// error of temperature
        if(max_values[1] > tau[1]){
            reset_PD = true;
			new_ebs[3] = ebs[3] / 2;
			new_ebs[4] = ebs[4] / 2;
        }
		// error of C
        if(max_values[2] > tau[2] && !reset_PD){
            reset_PD = true;
			new_ebs[3] = ebs[3] / 2;
			new_ebs[4] = ebs[4] / 2;
        }
		// error of M
        if(max_values[3] > tau[3]){
            if(!reset_Va){
                reset_Va = true;
                new_ebs[0] = ebs[0] / 2;
                new_ebs[1] = ebs[1] / 2;
                new_ebs[2] = ebs[2] / 2;    
            }
            if(!reset_PD){
                reset_PD = true;
                new_ebs[3] = ebs[3] / 2;
                new_ebs[4] = ebs[4] / 2;
            }
        }
		// error of PT
        if(max_values[4] > tau[4]){
            if(!reset_Va){
                reset_Va = true;
                new_ebs[0] = ebs[0] / 2;
                new_ebs[1] = ebs[1] / 2;
                new_ebs[2] = ebs[2] / 2;    
            }
            if(!reset_PD){
                reset_PD = true;
                new_ebs[3] = ebs[3] / 2;
                new_ebs[4] = ebs[4] / 2;
            }
        }        
		// error of mu
        if(max_values[5] > tau[5] && !reset_PD){
            reset_PD = true;
            new_ebs[3] = ebs[3] / 2;
            new_ebs[4] = ebs[4] / 2;
        }	
		for(int i=0; i<5; i++){
			ebs[i] = new_ebs[i];
		}	
	}
}

template<class T>
bool halfing_error_V_TOT(const T * Vx, const T * Vy, const T * Vz, size_t n, const double tau, std::vector<double>& ebs){
	double eb_Vx = ebs[0];
	double eb_Vy = ebs[1];
	double eb_Vz = ebs[2];
	double max_value = 0;
	int max_index = 0;
	for(int i=0; i<n; i++){
		// error of total velocity square
		double e_V_TOT_2 = compute_bound_x_square(Vx[i], eb_Vx) + compute_bound_x_square(Vy[i], eb_Vy) + compute_bound_x_square(Vz[i], eb_Vz);
		double V_TOT_2 = Vx[i]*Vx[i] + Vy[i]*Vy[i] + Vz[i]*Vz[i];
		// error of total velocity
		double e_V_TOT = compute_bound_square_root_x(V_TOT_2, e_V_TOT_2);
		double V_TOT = sqrt(V_TOT_2);
		// print_error("V_TOT", V_TOT, V_TOT_ori[i], e_V_TOT);

		error_est_V_TOT[i] = e_V_TOT;
		error_V_TOT[i] = V_TOT - V_TOT_ori[i];

		if(max_value < error_est_V_TOT[i]){
			max_value = error_est_V_TOT[i];
			max_index = i;
		}

	}
	std::cout << "Vx = " << Vx[max_index] << " Vy = " << Vy[max_index] << " Vz = " << Vz[max_index] << std::endl;
	std::cout << "eb_Vx = " << eb_Vx << " eb_Vy = " << eb_Vy << " eb_Vz = " << eb_Vz << std::endl;
	std::cout << "coeff_x = " << fabs(Vx[max_index])*eb_Vx << " coeff_y = " << fabs(Vy[max_index])*eb_Vy <<  " coeff_z = " << fabs(Vz[max_index])*eb_Vz << std::endl;
	std::cout << names[0] << ": max estimated error = " << max_value << ", index = " << max_index << std::endl;
	// estimate error bound based on maximal errors
	if(max_value > tau){
		std::cout << "comparing: " << max_value << " " << tau << std::endl;
		std::vector<double> new_ebs(3, 0);
		// error of Vtotal
        {
			// new_ebs[0] = ebs[0] / 2;
			// new_ebs[1] = ebs[1] / 2;
			// new_ebs[2] = ebs[2] / 2;            
			// double coeff_0 = fabs(Vy[max_index])*eb_Vy * fabs(Vz[max_index])*eb_Vz;
			// double coeff_1 = fabs(Vx[max_index])*eb_Vx * fabs(Vz[max_index])*eb_Vz;
			// double coeff_2 = fabs(Vx[max_index])*eb_Vx * fabs(Vy[max_index])*eb_Vy;
			double coeff_0 = fabs(Vx[max_index])*eb_Vx;
			double coeff_1 = fabs(Vy[max_index])*eb_Vy;
			double coeff_2 = fabs(Vz[max_index])*eb_Vz;
			double average = (coeff_0 + coeff_1 + coeff_2)/3;
			new_ebs[0] = ebs[0] / coeff_0 * average / 2;
			new_ebs[1] = ebs[1] / coeff_1 * average / 2;
			new_ebs[2] = ebs[2] / coeff_2 * average / 2;            
        }
		for(int i=0; i<3; i++){
			ebs[i] = new_ebs[i];
		}
		return false;
	}
	return true;
}


template <class T>
void compute_QoIs(const T * Vx, const T * Vy, const T * Vz, const T * P, const T * D, size_t n,
					T * V_TOT_, T * Temp_, T * C_, T * Mach_, T * PT_, T * mu_){
	double R = 287.1;
	double gamma = 1.4;
	double mi = 3.5;
	double mu_r = 1.716e-5;
	double T_r = 273.15;
	double S = 110.4;
	for(int i=0; i<n; i++){
		double V_TOT_2 = Vx[i]*Vx[i] + Vy[i]*Vy[i] + Vz[i]*Vz[i];
		double V_TOT = sqrt(V_TOT_2);
		double Temp = P[i] / (D[i] * R);
		double C = sqrt(gamma * R * Temp);
		double Mach = V_TOT / C;
		double Mach_tmp = 1 + (gamma-1)/2 * Mach * Mach;
		double Mach_tmp_mi = sqrt(pow(Mach_tmp, 7));
		double PT = P[i] * Mach_tmp_mi;
		double TrS_TS = (T_r + S) / (Temp + S);
		double T_Tr_3 = pow(Temp/T_r, 3);
		double T_Tr_3_sqrt = sqrt(T_Tr_3);
		double mu = mu_r * T_Tr_3_sqrt * TrS_TS;
		V_TOT_[i] = V_TOT;
		Temp_[i] = Temp;
		C_[i] = C;
		Mach_[i] = Mach;
		PT_[i] = PT;
		mu_[i] = mu;
	}
}

template <class T>
T compute_value_range(const std::vector<T>& vec){
	T min = vec[0];
	T max = vec[0];
	for(int i=0; i<vec.size(); i++){
		if(vec[i] < min) min = vec[i];
		if(vec[i] > max) max = vec[i];
	}
	return max - min;
}

template <class T>
void print_info(const std::string& name, const std::vector<T>& vec){
	T max = vec[0];
	T min = vec[0];
	for(int i=1; i<vec.size(); i++){
		if(max < vec[i]) max = vec[i];
		if(min > vec[i]) min = vec[i];
	}
	std::cout << name << ": min value = " << min << ", max value = " << max << ", value range = " << max - min << std::endl;
}

template <class T>
void print_max_abs(const std::string& name, const std::vector<T>& vec){
	T max = fabs(vec[0]);
	for(int i=1; i<vec.size(); i++){
		if(max < fabs(vec[i])) max = fabs(vec[i]);
	}
	std::cout << name << ": max absolute value = " << max << std::endl;
}

int main(int argc, char ** argv){

    using T = double;
    int zone_id = 0;
    std::string id_str = std::to_string(zone_id);
    size_t num_elements = 0;
    std::string prefix = data_file_prefix + "zone_";
    P_ori = MGARD::readfile<T>((prefix + id_str + "_Pressure.dat").c_str(), num_elements);
    D_ori = MGARD::readfile<T>((prefix + id_str + "_Density.dat").c_str(), num_elements);
    Vx_ori = MGARD::readfile<T>((prefix + id_str + "_VelocityX.dat").c_str(), num_elements);
    Vy_ori = MGARD::readfile<T>((prefix + id_str + "_VelocityY.dat").c_str(), num_elements);
    Vz_ori = MGARD::readfile<T>((prefix + id_str + "_VelocityZ.dat").c_str(), num_elements);
    double target_rel_eb = 0.01;
    std::vector<double> ebs;
    ebs.push_back(compute_value_range(Vx_ori));
    ebs.push_back(compute_value_range(Vy_ori));
    ebs.push_back(compute_value_range(Vz_ori));
    // ebs.push_back(compute_value_range(P_ori));
    // ebs.push_back(compute_value_range(D_ori));
    for(int i=0; i<ebs.size(); i++){
    	ebs[i] *= target_rel_eb;
    }

	int nvar = ebs.size();
    std::vector<T> V_TOT(num_elements);
    std::vector<T> Temp(num_elements);
    std::vector<T> C(num_elements);
    std::vector<T> Mach(num_elements);
    std::vector<T> PT(num_elements);
    std::vector<T> mu(num_elements);
    compute_QoIs(Vx_ori.data(), Vy_ori.data(), Vz_ori.data(), P_ori.data(), D_ori.data(), num_elements, V_TOT.data(), Temp.data(), C.data(), Mach.data(), PT.data(), mu.data());

    std::vector<double> tau;
    tau.push_back(compute_value_range(V_TOT)*target_rel_eb);
    tau.push_back(compute_value_range(Temp)*target_rel_eb);
    tau.push_back(compute_value_range(C)*target_rel_eb);
    tau.push_back(compute_value_range(Mach)*target_rel_eb);
    tau.push_back(compute_value_range(PT)*target_rel_eb);
    tau.push_back(compute_value_range(mu)*target_rel_eb);
	V_TOT_ori = V_TOT.data();
	Temp_ori = Temp.data();
	C_ori = C.data();
	Mach_ori = Mach.data();
	PT_ori = PT.data();
	mu_ori = mu.data();

    std::string mask_file = rdata_file_prefix + "mask.bin";
    size_t num_valid_data = 0;
    auto mask = MGARD::readfile<unsigned char>(mask_file.c_str(), num_valid_data);
    std::vector<MDR::ComposedReconstructor<T, MGARDHierarchicalDecomposer<T>, DirectInterleaver<T>, PerBitBPEncoder<T, uint32_t>, AdaptiveLevelCompressor, SignExcludeGreedyBasedSizeInterpreter<MaxErrorEstimatorHB<T>>, MaxErrorEstimatorHB<T>, ConcatLevelFileRetriever>> reconstructors;
    for(int i=0; i<nvar; i++){
        // std::string rdir_prefix = rdata_file_prefix + "zone_" + id_str + "_" + varlist[i];
        std::string rdir_prefix = rdata_file_prefix + varlist[i];
        std::string metadata_file = rdir_prefix + "_refactored_data/metadata.bin";
        std::vector<std::string> files;
        int num_levels = 9;
        for(int i=0; i<num_levels; i++){
            std::string filename = rdir_prefix + "_refactored_data/level_" + std::to_string(i) + ".bin";
            files.push_back(filename);
        }
        auto decomposer = MGARDHierarchicalDecomposer<T>();
        auto interleaver = DirectInterleaver<T>();
        auto encoder = PerBitBPEncoder<T, uint32_t>();
        auto compressor = AdaptiveLevelCompressor(64);
        auto estimator = MaxErrorEstimatorHB<T>();
        auto interpreter = SignExcludeGreedyBasedSizeInterpreter<MaxErrorEstimatorHB<T>>(estimator);
        auto retriever = ConcatLevelFileRetriever(metadata_file, files);
        reconstructors.push_back(generateReconstructor<T>(decomposer, interleaver, encoder, compressor, estimator, interpreter, retriever));
        reconstructors.back().load_metadata();
    }    
    std::vector<std::vector<T>> reconstructed_vars(nvar, std::vector<double>(num_elements));

    int iter = 0;
    int max_iter = 5;
	bool tolerance_met = false;
    while((!tolerance_met) && (iter < max_iter)){
    	iter ++;
	    for(int i=0; i<nvar; i++){
	        auto reconstructed_data = reconstructors[i].progressive_reconstruct(ebs[i], -1);
	        if(i < 3){
	            // reconstruct with mask
	            int index = 0;
	            for(int j=0; j<num_elements; j++){
	                if(mask[j]){
	                    reconstructed_vars[i][j] = reconstructed_data[index ++];
	                }
	                else reconstructed_vars[i][j] = 0;
	            }
	        }
	        else{
	            memcpy(reconstructed_vars[i].data(), reconstructed_data, num_elements*sizeof(T));
	        }
	    }

	    Vx_dec = reconstructed_vars[0].data();
	    Vy_dec = reconstructed_vars[1].data();
	    Vz_dec = reconstructed_vars[2].data();
	    // P_dec = reconstructed_vars[3].data();
	    // D_dec = reconstructed_vars[4].data();
	    MGARD::print_statistics(Vx_ori.data(), Vx_dec, num_elements);
	    MGARD::print_statistics(Vy_ori.data(), Vy_dec, num_elements);
	    MGARD::print_statistics(Vz_ori.data(), Vz_dec, num_elements);
	    // MGARD::print_statistics(P_ori.data(), P_dec, num_elements);
	    // MGARD::print_statistics(D_ori.data(), D_dec, num_elements);
	    error_V_TOT = std::vector<double>(num_elements);
	    // error_Temp = std::vector<double>(num_elements);
	    // error_C = std::vector<double>(num_elements);
	    // error_Mach = std::vector<double>(num_elements);
	    // error_PT = std::vector<double>(num_elements);
	    // error_mu = std::vector<double>(num_elements);
	    error_est_V_TOT = std::vector<double>(num_elements);
	    // error_est_Temp = std::vector<double>(num_elements);
	    // error_est_C = std::vector<double>(num_elements);
	    // error_est_Mach = std::vector<double>(num_elements);
	    // error_est_PT = std::vector<double>(num_elements);
	    // error_est_mu = std::vector<double>(num_elements);
		std::cout << "iter" << iter << ": The old ebs are:" << std::endl;
	    MDR::print_vec(ebs);
	    tolerance_met = halfing_error_V_TOT(Vx_dec, Vy_dec, Vz_dec, num_elements, tau[0], ebs);
	    // halfing_error(Vx_dec, Vy_dec, Vz_dec, P_dec, D_dec, num_elements, tau, ebs);
		std::cout << "iter" << iter << ": The new ebs are:" << std::endl;
	    MDR::print_vec(ebs);
	    std::cout << names[0] << " requested error = " << tau[0] << std::endl;
	    print_max_abs(names[0] + " error", error_V_TOT);
	    print_max_abs(names[0] + " error_est", error_est_V_TOT);
	    // std::cout << names[1] << " requested error = " << tau[1] << std::endl;
	    // print_max_abs(names[1] + " error", error_Temp);
	    // print_max_abs(names[1] + " error_est", error_est_Temp);
	    // std::cout << names[2] << " requested error = " << tau[2] << std::endl;
	    // print_max_abs(names[2] + " error", error_C);
	    // print_max_abs(names[2] + " error_est", error_est_C);
	    // std::cout << names[3] << " requested error = " << tau[3] << std::endl;
	    // print_max_abs(names[3] + " error", error_Mach);
	    // print_max_abs(names[3] + " error_est", error_est_Mach);
	    // std::cout << names[4] << " requested error = " << tau[4] << std::endl;
	    // print_max_abs(names[4] + " error", error_PT);
	    // print_max_abs(names[4] + " error_est", error_est_PT);
	    // std::cout << names[5] << " requested error = " << tau[5] << std::endl;
	    // print_max_abs(names[5] + " error", error_mu);
	    // print_max_abs(names[5] + " error_est", error_est_mu);    	
    }
	std::cout << "iter = " << iter << std::endl;
   
    return 0;
}

