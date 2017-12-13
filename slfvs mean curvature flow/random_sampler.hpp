#include <random>

using namespace std;

// thread safe rng(s)
// todo: have just one random_engine per thread

mt19937 seeded_engine() {
    random_device r;
    seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
    return mt19937(seed);
}

inline double sample_sign ()
{
	static mt19937 random_engine = seeded_engine();
	bernoulli_distribution distribution {0.5};
	return distribution(random_engine) ? 1.0 : -1.0;
}

inline double sample_bernoulli (const double p)
{
	static mt19937 random_engine = seeded_engine();
	bernoulli_distribution distribution {p};
	return distribution(random_engine) ? 1.0 : 0.0;
}

inline bool sample_boolean (const double p)
{
	static mt19937 random_engine = seeded_engine();
	bernoulli_distribution distribution {p};
	return distribution(random_engine);
}

inline double sample_uniform_R (const double a, const double b)
{
	static mt19937 random_engine = seeded_engine();
	uniform_real_distribution<double> distribution {a,b};
	return distribution(random_engine);
};

inline double sample_uniform_R ()
{
	static mt19937 random_engine = seeded_engine();
	uniform_real_distribution<double> distribution {0.0,1.0};
	return distribution(random_engine);
};

inline int sample_uniform_Z (const int a, const int b)
{
	static mt19937 random_engine = seeded_engine();
	uniform_int_distribution<int> distribution {a,b};
	return distribution(random_engine);
};

inline double sample_exponential (const double lambda)
{
	static mt19937 random_engine = seeded_engine();
	exponential_distribution<double> distribution {lambda};
	return distribution(random_engine);
}

inline double sample_poisson (const double lambda)
{
	static mt19937 random_engine = seeded_engine();
	poisson_distribution<int> distribution {lambda};
	return distribution(random_engine);
}

inline double sample_pareto (const double alpha, const double x_min)
{
	static mt19937 random_engine = seeded_engine();
	exponential_distribution<double> distribution {alpha};
	double y = distribution(random_engine);
	return x_min * exp(y);
}
