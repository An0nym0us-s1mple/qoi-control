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

std::vector<double> X_ori;
double * x2_ori = NULL;
double * X_dec = NULL;
std::vector<double> error_x2;
std::vector<double> error_est_x2;

template<class T>
bool halfing_error_x_squared(const T * X, size_t n, const double tau, double& eb){
	double max_value = 0;
	int max_index = 0;
	for(int i=0; i<n; i++){
        double e_x2 = compute_bound_x_square(X[i], eb);
        double x2 = X[i] * X[i];

		error_est_x2[i] = e_x2;
		error_x2[i] = x2 - x2_ori[i];

		if(max_value < error_est_x2[i]){
			max_value = error_est_x2[i];
			max_index = i;
		}
	}
	std::cout << "X^2 : max estimated error = " << max_value << ", index = " << max_index << std::endl;
	// estimate error bound based on maximal errors
	if(max_value > tau){
		auto i = max_index;
		double estimate_error = max_value;
        double eb_x = eb;
		while(estimate_error > tau){
    		std::cout << "uniform decrease\n";
            eb_x = eb_x / 1.5;
			estimate_error = compute_bound_x_square(X[i], eb_x);
		}
        eb = eb_x;
		return false;
	}
	return true;
}


int main(int argc, char ** argv){

    using T = double;
	int argv_id = 1;
    std::string data_name = argv[argv_id++];
    double target_rel_eb = atof(argv[argv_id++]);
	std::string data_prefix_path = argv[argv_id++];
	std::string data_file_prefix = data_prefix_path + "/data/";
	std::string rdata_file_prefix = data_prefix_path + "/refactor/";

    size_t num_elements = 0;
    X_ori = MGARD::readfile<T>((data_file_prefix + data_name + ".dat").c_str(), num_elements);
    double eb = compute_value_range(X_ori) * target_rel_eb;

    std::vector<T> x2(num_elements);
    for(int i=0; i<num_elements; i++){
        x2[i] = X_ori[i] * X_ori[i];
    }

	struct timespec start, end;
	int err;
	double elapsed_time;

	err = clock_gettime(CLOCK_REALTIME, &start);

    double tau = compute_value_range(x2)*target_rel_eb;
    x2_ori = x2.data();

    std::string rdir_prefix = rdata_file_prefix + data_name;
    std::string metadata_file = rdir_prefix + "_refactored/metadata.bin";
    std::vector<std::string> files;
    int num_levels = 5;
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
    auto reconstructor = generateReconstructor<T>(decomposer, interleaver, encoder, compressor, estimator, interpreter, retriever);
    reconstructor.load_metadata();

    std::vector<T> reconstructed_X(num_elements);
	size_t total_retrieved_size = 0;

    int iter = 0;
    int max_iter = 5;
	bool tolerance_met = false;
    while((!tolerance_met) && (iter < max_iter)){
    	iter ++;
        auto reconstructed_data = reconstructor.progressive_reconstruct(eb, -1);
        total_retrieved_size = reconstructor.get_retrieved_size();
        memcpy(reconstructed_X.data(), reconstructed_data, num_elements*sizeof(T));

	    X_dec = reconstructed_X.data();
	    MGARD::print_statistics(X_ori.data(), X_dec, num_elements);
	    error_x2 = std::vector<double>(num_elements);
	    error_est_x2 = std::vector<double>(num_elements);
		std::cout << "iter" << iter << ": The old eb is:\n" << eb << std::endl;
	    tolerance_met = halfing_error_x_squared(X_dec, num_elements, tau, eb);
		std::cout << "iter" << iter << ": The new eb is:\n" << eb << std::endl;
	    std::cout << "X^2 requested error = " << tau << std::endl;
	    print_max_abs("X^2 error", error_x2);
	    print_max_abs("X^2 error_est", error_est_x2);   	
    }
	err = clock_gettime(CLOCK_REALTIME, &end);
	elapsed_time = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000;

	std::cout << "iter = " << iter << std::endl;
   
	double cr = num_elements * sizeof(T) * 1.0 / total_retrieved_size;
	std::cout << "retrieved size: " << total_retrieved_size << std::endl;
	std::cout << "aggregated cr = " << cr << std::endl;
	printf("elapsed_time = %.6f\n", elapsed_time);

    return 0;
}