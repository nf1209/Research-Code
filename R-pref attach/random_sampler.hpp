#include <random>
#include <stdexcept>

using namespace std;

class sampler_t {
private:
    default_random_engine random_engine; // seed
    uniform_real_distribution<double> unif_real_dist {0.0, 1.0};
    normal_distribution<double> std_normal_dist {0.0, 1.0};
    exponential_distribution<double> std_exponential_dist {1.0};
public:
    sampler_t () {default_random_engine r{1}; random_engine=r;} ;
    sampler_t (int64_t seed) {default_random_engine r{seed}; random_engine=r;} ;

    double unif_real_01() {return unif_real_dist(random_engine);} ;
    double normal_dist(double mean, double var) {if (var<0) throw runtime_error{"negative variance"}; return mean+sqrt(var)*std_normal_dist(random_engine);} ;
    double exp_dist(double lambda) {if (lambda<0) throw runtime_error{"negative mean"}; return std_exponential_dist(random_engine)/lambda;} ;
    double pareto(double alpha) {double u = unif_real_01(); return pow(u,-1.0/alpha);} ; // support (1,infty)
};
