#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <bitset>
#include <numeric>
#include "utils.hpp"
#include "qoi_utils.hpp"
#include "Reconstructor/Reconstructor.hpp"
#include "Synthesizer4GE.hpp"

using namespace MDR;

std::vector<float> P_ori;
std::vector<float> D_ori;
std::vector<float> Vx_ori;
std::vector<float> Vy_ori;
std::vector<float> Vz_ori;
float * P_dec = NULL;
float * D_dec = NULL;
float * Vx_dec = NULL;
float * Vy_dec = NULL;
float * Vz_dec = NULL;
float * V_TOT_ori = NULL;
std::vector<float> error_V_TOT;
std::vector<float> error_est_V_TOT;


template<class T>
bool halfing_error_V_TOT_uniform(const T * Vx, const T * Vy, const T * Vz, size_t n, const std::vector<unsigned char>& mask, const float tau, std::vector<float>& ebs){
	float eb_Vx = ebs[0];
	float eb_Vy = ebs[1];
	float eb_Vz = ebs[2];
	float max_value = 0;
	int max_index = 0;
	for(int i=0; i<n; i++){
		// error of total velocity square
		float e_V_TOT_2 = 0;
		if(mask[i]) e_V_TOT_2 = compute_bound_x_square(Vx[i], eb_Vx) + compute_bound_x_square(Vy[i], eb_Vy) + compute_bound_x_square(Vz[i], eb_Vz);
		float V_TOT_2 = Vx[i]*Vx[i] + Vy[i]*Vy[i] + Vz[i]*Vz[i];
		// error of total velocity
		float e_V_TOT = 0;
		if(mask[i]) e_V_TOT = compute_bound_square_root_x(V_TOT_2, e_V_TOT_2);
		float V_TOT = sqrt(V_TOT_2);
		// print_error("V_TOT", V_TOT, V_TOT_ori[i], e_V_TOT);

		error_est_V_TOT[i] = e_V_TOT;
		error_V_TOT[i] = V_TOT - V_TOT_ori[i];
		if(e_V_TOT > tau) return false;
	}
	// std::cout << names[0] << ": max estimated error = " << max_value << ", index = " << max_index << std::endl;
	// estimate error bound based on maximal errors
	return true;
}


int main(int argc, char ** argv){

    using T = float;
	int argv_id = 1;
    float target_rel_eb = atof(argv[argv_id++]);
	std::string data_prefix_path = argv[argv_id++];
	std::string data_file_prefix = data_prefix_path + "/data/";
	std::string rdata_file_prefix = data_prefix_path + "/refactor/";

    size_t num_elements = 0;
    Vx_ori = MGARD::readfile<T>((data_file_prefix + "VelocityX.dat").c_str(), num_elements);
    Vy_ori = MGARD::readfile<T>((data_file_prefix + "VelocityY.dat").c_str(), num_elements);
    Vz_ori = MGARD::readfile<T>((data_file_prefix + "VelocityZ.dat").c_str(), num_elements);
    std::vector<float> ebs;
    ebs.push_back(compute_value_range(Vx_ori)*target_rel_eb);
    ebs.push_back(compute_value_range(Vy_ori)*target_rel_eb);
    ebs.push_back(compute_value_range(Vz_ori)*target_rel_eb);
	int n_variable = ebs.size();
    std::vector<std::vector<T>> vars_vec = {Vx_ori, Vy_ori, Vz_ori};
    std::vector<float> var_range(n_variable);
    for(int i=0; i<n_variable; i++){
        var_range[i] = compute_value_range(vars_vec[i]);
    } 

	struct timespec start, end;
	int err;
	float elapsed_time;

	err = clock_gettime(CLOCK_REALTIME, &start);

    std::vector<T> V_TOT(num_elements);
    compute_VTOT(Vx_ori.data(), Vy_ori.data(), Vz_ori.data(), num_elements, V_TOT.data());
	V_TOT_ori = V_TOT.data();
    float tau = compute_value_range(V_TOT)*target_rel_eb;

    std::string mask_file = rdata_file_prefix + "mask.bin";
    size_t num_valid_data = 0;
    auto mask = MGARD::readfile<unsigned char>(mask_file.c_str(), num_valid_data);
    std::vector<MDR::SegmentedReconstructor<T, MGARDHierarchicalDecomposer<T>, DirectInterleaver<T>, PerBitBPEncoder<T, uint32_t>, AdaptiveLevelCompressor, SignExcludeGreedyBasedSizeInterpreter<MaxErrorEstimatorHB<T>>, MaxErrorEstimatorHB<T>, ConcatLevelFileRetriever>> reconstructors;
    for(int i=0; i<n_variable; i++){
        std::string rdir_prefix = rdata_file_prefix + varlist[i];
        std::string metadata_file = rdir_prefix + "_refactored/metadata.bin";
        std::vector<std::string> files;
        int num_levels = 9;
        for(int i=0; i<num_levels; i++){
            std::string filename = rdir_prefix + "_refactored/level_" + std::to_string(i) + ".bin";
            files.push_back(filename);
        }
        auto decomposer = MGARDHierarchicalDecomposer<T>();
        auto interleaver = DirectInterleaver<T>();
        auto encoder = PerBitBPEncoder<T, uint32_t>();
        auto compressor = AdaptiveLevelCompressor(64);
        auto estimator = MaxErrorEstimatorHB<T>();
        auto interpreter = SignExcludeGreedyBasedSizeInterpreter<MaxErrorEstimatorHB<T>>(estimator);
        auto retriever = ConcatLevelFileRetriever(metadata_file, files);
        reconstructors.push_back(generateSegmentedReconstructor<T>(decomposer, interleaver, encoder, compressor, estimator, interpreter, retriever));
        reconstructors.back().load_metadata();
    }    
    std::vector<std::vector<T>> reconstructed_vars(n_variable, std::vector<float>(num_elements));

	std::vector<size_t> total_retrieved_size(n_variable, 0);

    int iter = 0;
    int max_iter = 100;
	bool tolerance_met = false;
	float max_act_error = 0, max_est_error = 0;
    while((!tolerance_met) && (iter < max_iter)){
    	iter ++;
	    for(int i=0; i<n_variable; i++){
	        auto reconstructed_data = reconstructors[i].progressive_reconstruct(50000000, -1);
			total_retrieved_size[i] = reconstructors[i].get_retrieved_size();
			ebs[i] = reconstructors[i].get_error_bound();
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
	    // MGARD::print_statistics(Vx_ori.data(), Vx_dec, num_elements);
	    // MGARD::print_statistics(Vy_ori.data(), Vy_dec, num_elements);
	    // MGARD::print_statistics(Vz_ori.data(), Vz_dec, num_elements);
	    error_V_TOT = std::vector<float>(num_elements);
	    error_est_V_TOT = std::vector<float>(num_elements);
		// std::cout << "iter" << iter << ": The old ebs are:" << std::endl;
	    // MDR::print_vec(ebs);
	    tolerance_met = halfing_error_V_TOT_uniform(Vx_dec, Vy_dec, Vz_dec, num_elements, mask, tau, ebs);
		// std::cout << "iter" << iter << ": The new ebs are:" << std::endl;
	    // MDR::print_vec(ebs);
	    // std::cout << names[0] << " requested error = " << tau << std::endl;
	    max_act_error = print_max_abs(names[0] + " error", error_V_TOT);
	    max_est_error = print_max_abs(names[0] + " error_est", error_est_V_TOT);  
    }
	err = clock_gettime(CLOCK_REALTIME, &end);
	elapsed_time = (float)(end.tv_sec - start.tv_sec) + (float)(end.tv_nsec - start.tv_nsec)/(float)1000000000;

	std::cout << "requested error = " << tau << std::endl;
	std::cout << "max_est_error = " << max_est_error << std::endl;
	std::cout << "max_act_error = " << max_act_error << std::endl;
	std::cout << "iter = " << iter << std::endl;

	size_t total_size = std::accumulate(total_retrieved_size.begin(), total_retrieved_size.end(), 0);
	// std::cout << "total_size: " << total_size << std::endl;
	// std::cout << "original_size: " << n_variable * num_elements * sizeof(T) << std::endl;
	float cr = n_variable * num_elements * sizeof(T) * 1.0 / total_size;
	std::cout << "each retrieved size:";
    for(int i=0; i<n_variable; i++){
        std::cout << total_retrieved_size[i] << ", ";
    }
    std::cout << std::endl;
	// MDR::print_vec(total_retrieved_size);
	std::cout << "aggregated cr = " << cr << std::endl;
	printf("elapsed_time = %.6f\n", elapsed_time);

    return 0;
}