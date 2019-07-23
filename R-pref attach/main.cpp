#include <iostream>
#include <vector>
#include <list>
#include "random_sampler.hpp"
#include <stdexcept>

using namespace std;

// points are +ve integers

sampler_t sample {0};

const double alpha = 0.5; // power law tail of R

class graph_t
{
public:
    vector<double> fitness;
    vector<uint64_t> degree;
    vector<uint64_t> cluster; // label of the root of the cluster the particle joins
    vector<uint64_t> cluster_size; // 0 if not the root of the cluster

    void add_new_point ();
    uint64_t n () {return fitness.size();};
    graph_t ();

    void print_stat_line ();

private:
    uint64_t total_degree;
    vector<uint64_t> degree_biased_samples (uint64_t R);
};

graph_t::graph_t ()
{
    // initialize with a single point
    fitness.push_back(sample.unif_real_01());
    degree.push_back(0);
    cluster.push_back(1);
    cluster_size.push_back(1);

    // -> all points will be in cluster 1, but maybe one day it'll be useful to have new clusters appearing
}

vector<uint64_t> graph_t::degree_biased_samples (uint64_t R)
{
    // sample R (with replacement) points with prob proportional to current degrees
    uint64_t i;
    list<uint64_t> positions; // to size bias, think of the degrees as laid out, one by one, along N
    for (i=0; i<R; ++i) {
        double p = sample.unif_real_01();
        uint64_t pos = p * (double)total_degree; // FIXME: assumes there is only one point with zero degree, initially
        positions.push_back(pos);
    }
    positions.sort();

    vector<uint64_t> samples;
    if (positions.size()==0) return samples;

    uint64_t cur_point;
    uint64_t cur_pos = 0;
    uint64_t cur_target_pos = positions.front();
    for (cur_point=0; cur_point<degree.size(); ++cur_point) {
        if (cur_pos>=cur_target_pos) {
            samples.push_back(cur_point);
            positions.pop_front();
            if (positions.size()==0) break;
            cur_target_pos = positions.front();
        }
        else {
            cur_pos += degree[cur_point];
        }
    }

    return samples;
}

void graph_t::add_new_point()
{
    if (fitness.size()!=degree.size() || degree.size()!=cluster.size() || cluster.size()!=cluster_size.size()) throw runtime_error{"graph is corrupt"};

    // sample parent
    // sample R points with replacement w.p. size biased by degree, choose fittest
    uint64_t R = sample.pareto(alpha);
    //cout << R << endl;
    if (R==0) throw runtime_error{"no candidate parents"};
    vector<uint64_t> candidates;
    if (R<=n()*n()) {
        candidates = degree_biased_samples(R);
    }
    else{
        uint64_t i;
        for (i=0; i<n(); ++i) candidates.push_back(i);
    }
    uint64_t fittest_candidate;
    uint64_t best_fitness = 0;
    for(uint64_t this_candidate : candidates) {
        double this_fitness = fitness[this_candidate];
        if (this_fitness>best_fitness) {
            fittest_candidate = this_candidate;
            best_fitness = this_fitness;
        }
    }

    // construct the new point
    double new_fitness = sample.unif_real_01();
    uint64_t parent = fittest_candidate;
    uint64_t root = cluster[parent];
    uint64_t new_cluster = cluster[parent];

    // add the new point
    fitness.push_back(new_fitness);
    degree.push_back(1);
    ++degree[parent];
    total_degree += 2;
    cluster.push_back(new_cluster);
    cluster_size.push_back(0); // add as non-root
    ++cluster_size[root];
}

void graph_t::print_stat_line ()
{
    if (fitness.size()!=degree.size() || degree.size()!=cluster.size()) throw runtime_error{"graph is corrupt"};

    uint64_t i;
    uint64_t largest_vertex;
    uint64_t largest_vertex_degree = 0;
    for (i=0; i<degree.size(); ++i) {
        if (degree[i]>largest_vertex_degree || largest_vertex_degree==0) {
            largest_vertex = i;
            largest_vertex_degree = degree[i];
        }
    }
    cout << "n=" << n() << ", largest_vertex is " << largest_vertex << " with degree " << largest_vertex_degree << " and fitness " << fitness[largest_vertex] << endl;
}


int main()
{
    graph_t graph;

    uint64_t n;
    for (n=0; n<1e7; ++n) {
        graph.add_new_point();

        if (n%(10000)==0) {
            graph.print_stat_line();
        }
    }


    return 0;
}
