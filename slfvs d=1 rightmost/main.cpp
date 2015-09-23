#include <iostream>
#include <map>
#include <vector>
#include <random>
#include <sstream>
#include <cassert>
#include <thread>
#include <fstream>
#include <string>
#include <time.h>
#include <future>

using namespace std;

bool DEBUG {false};
bool PAUSE_DEBUG {false};

string format(double Value, int nPrecision){
    ostringstream ss;
    ss.precision(nPrecision);
    ss.setf(ios::fixed);
    ss << Value;
    return ss.str();
}

class datapoint_t
// record of where the left/right-most paths went
{
public:
    const double interval {0.05}; // interval between timepoints
    vector<double> leftmost;
    vector<double> rightmost;
    double final_particles {0.0}; // number of particles at final timepoint
    double n_events;
    double n_particles;
};

class particle {
public:
    double left_extent; // part of total extent closest and to the left of this particle
    double right_extent; // part of the total extent closest and to the right of this particle
    const double birth_time;
    const int64_t id;
    const double location;

    particle ();
    particle (const double, const int64_t, const double);
};

particle::particle ()
// keep gcc happy
:birth_time(0), id(-1), location(0)
{};


particle::particle (const double birth_time, const int64_t id, const double location)
:birth_time(birth_time), id(id), location(location)
{
    left_extent = 0.0;
    right_extent = 0.0; //SLFV manages these
};

enum event_type {
    neutral,
    selective
};

enum sampling_method_t {
    rejection,
    linear
};

class event_t {
public:
    event_type type;
    sampling_method_t sampling_method;

    double time;
    int64_t id;

    double center;
    double radius;
    double right_end;
    double left_end;

    double parent;
    double left_pre_parent;
    double right_pre_parent;

    map<double, particle>::iterator nearest_particle_it;
};

class SLFV {
public:
    const double R;
    const double u;
    const double s;
    const double n;

    SLFV (const double R, const double u, const double n, const int random_seed);
    void simulate (const double max_time, const bool verbose, datapoint_t& samples);

private:
    double current_time; // current time
    double wanted_time; // time to stop simulating

    default_random_engine random_engine;
    uniform_real_distribution<double> unif_dist {0.0,1.0};
    exponential_distribution<double> exp_dist {1.0};
    double sample_unif_dist();
    double sample_exp_dist();

    map<double, particle> cloud; // <location, particle>
    void insert_particle (const double location);
    bool remove_particles (const double location, const double radius);

    double total_extent; // area of the subset of R in which events may fall and affect one of the current particles
    void register_extents (map<double, particle>::iterator&);
    void deregister_extents (map<double, particle>::iterator&);

    particle& rightmost_particle ();
    particle& leftmost_particle ();

    void next_event (); // replace current_event with next event (based on current cloud)
    void update_particles (); // update cloud using current_event
    event_t current_event;

    int64_t cur_max_particle_id {0};

    void consistency_check ();
};

SLFV::SLFV (const double R_unscaled, const double u_unscaled, const double n, const int random_seed)
:R(R_unscaled/sqrt(n)), u(u_unscaled), s(1/sqrt(n)), n(n)
{
    random_engine.seed(random_seed);

    current_time = 0.0;
    wanted_time = 0.0;
    total_extent = 0.0;

    current_event.id = 0;

    insert_particle(0.0);
}

void SLFV::consistency_check ()
{
    assert(cloud.begin()!=cloud.end());

    // check extents
    // some (small) floating point errors will occur naturally
    map<double, particle>::iterator p_it {cloud.begin()};
    double seen_extent {0.0};
    while (p_it!=cloud.end())
    {
        assert(p_it->second.left_extent<=p_it->second.right_extent);
        assert(p_it->second.location<=p_it->second.right_extent);
        assert(p_it->second.left_extent<=p_it->second.location);
        seen_extent += p_it->second.right_extent - p_it->second.left_extent;
        map<double, particle>::iterator p2_it = p_it;
        ++p2_it;
        if (p2_it!=cloud.end())
        {
            assert(p2_it->second.left_extent >= p_it->second.right_extent - 0.01);
        }
        ++p_it;

    }
    assert(abs(seen_extent-total_extent)<=0.01);
}

particle& SLFV::rightmost_particle()
{
    //assert(cloud.begin()!=cloud.end());
    return (--cloud.end())->second;
}

particle& SLFV::leftmost_particle ()
{
    return cloud.begin()->second;
}

double SLFV::sample_exp_dist()
{
    exp_dist.reset();
    return exp_dist(random_engine);
}

double SLFV::sample_unif_dist()
{
    unif_dist.reset();
    return unif_dist(random_engine);
}


void SLFV::register_extents (map<double,particle>::iterator& p_it)
{
    // particle p is newly created; calculate its extents and update as appropriate
    particle& p = p_it->second;

    p.right_extent = p.location + R; //preliminary
    map<double,particle>::iterator next_right = next(p_it);
    if (next_right!=cloud.end() && p.right_extent > next_right->second.left_extent)
    {
        double mid_pos {(next_right->second.location+p.location)/2.0};

        total_extent -= (next_right->second.location-next_right->second.left_extent);
        next_right->second.left_extent = mid_pos;
        total_extent += (next_right->second.location-next_right->second.left_extent);

        p.right_extent = mid_pos;
        total_extent += (p.right_extent-p.location);
    }
    else
    {
        total_extent += R;
    }

    p.left_extent = p.location - R; //preliminary
    if (p_it!=cloud.begin())
    {
        map<double,particle>::iterator next_left = prev(p_it);
        if (next_left->second.right_extent > p.left_extent)
        {
            double mid_pos {(next_left->second.location+p.location)/2.0};

            total_extent -= (next_left->second.right_extent-next_left->second.location);
            next_left->second.right_extent = mid_pos;
            total_extent += (next_left->second.right_extent-next_left->second.location);

            p.left_extent = mid_pos;
            total_extent += (p.location-p.left_extent);
        }
        else
        {
            total_extent += R;
        }
    }
    else
    {
        total_extent += R;
    }
}

void SLFV::deregister_extents (map<double,particle>::iterator& p_it)
{
    // particle p exists and is about to be killed; remove its extents and update as appropriate
    particle& p = p_it->second;

    total_extent -= (p.right_extent - p.left_extent);

    map<double,particle>::iterator next_right = next(p_it);
    if (next_right==cloud.end())
    {
        if (cloud.begin()==cloud.end())
        {
            // p was the only particle
            total_extent = 0.0;
        }
        else
        {
            // p was the rightmost particle
            map<double,particle>::iterator next_left = prev(p_it);
            total_extent -= (next_left->second.right_extent - next_left->second.location);
            next_left->second.right_extent = next_left->second.location + R;
            total_extent += (next_left->second.right_extent - next_left->second.location);
        }
    }
    else
    {
        if (p_it==cloud.begin())
        {
            // p was the leftmost particle
            total_extent -= (next_right->second.location - next_right->second.left_extent);
            next_right->second.left_extent = next_right->second.location - R;
            total_extent += (next_right->second.location - next_right->second.left_extent);
        }
        else
        {
            // p was in the middle
            map<double,particle>::iterator next_left = prev(p_it);
            if (next_right->second.location-R < next_left->second.location+R)
            {
                // collision
                double mid_pos = (next_left->second.location + next_right->second.location)/2.0;

                total_extent -= (next_right->second.location - next_right->second.left_extent);
                next_right->second.left_extent = mid_pos;
                total_extent += (next_right->second.location - next_right->second.left_extent);

                total_extent -= (next_left->second.right_extent - next_left->second.location);
                next_left->second.right_extent = mid_pos;
                total_extent += (next_left->second.right_extent - next_left->second.location);

            }
            else
            {
                // no collision
                total_extent -= (next_right->second.location - next_right->second.left_extent);
                next_right->second.left_extent = next_right->second.location - R;
                total_extent += (next_right->second.location - next_right->second.left_extent);

                total_extent -= (next_left->second.right_extent - next_left->second.location);
                next_left->second.right_extent = next_left->second.location + R;
                total_extent += (next_left->second.right_extent - next_left->second.location);
            }

        }

    }

}

void SLFV::insert_particle (const double location)
// properly, we would make a wrapper class for std::map with this inside its insert (also SLFV::remove_particle), but it is not worth the effort
{
    // check we don't already have one at this location (!)
    map<double, particle>::iterator it = cloud.find(location);
    if (it!=cloud.end() && it->first==location) return;

    // insert
    particle p {current_time, cur_max_particle_id, location};
    auto ret = cloud.insert(make_pair(location,p));
    map<double, particle>::iterator p_it = ret.first;
    //map<double, particle>::iterator p_it = cloud.emplace_hint(hint, location, {current_time, cur_max_particle_id, location})); //better but requires C++14
    register_extents(p_it);

    ++cur_max_particle_id;

    if (DEBUG) cout << "  Created " << location << endl;
}

bool SLFV::remove_particles (const double location, const double radius)
{
    // remove particles within radius of location, each with probability u
    // start at the nearest particle and then move left/rightwards
    map<double,particle>::iterator leftmost_hit {current_event.nearest_particle_it};
    map<double,particle>::iterator rightmost_hit {current_event.nearest_particle_it};
    ++rightmost_hit;

    bool killed {false}; // did we kill at least one particle?

    // look leftwards (and at nearest)
    while (true)
    {
        if (leftmost_hit->first >= current_event.left_end) {
            if (leftmost_hit!=cloud.begin()) {
                if (sample_unif_dist()<u) {
                    if (DEBUG) cout << "  Killed " << leftmost_hit->first << " (L)" << endl;
                    deregister_extents(leftmost_hit);
                    cloud.erase(leftmost_hit--);
                    killed = true;
                }
                else
                {
                    if (DEBUG) cout << "  Passed " << leftmost_hit->first << " (L)" << endl;
                    leftmost_hit--;
                    continue;
                }
            }
            else
            {
                if (sample_unif_dist()<u) {
                    if (DEBUG) cout << "  Killed " << leftmost_hit->first << " (L)" << endl;
                    deregister_extents(leftmost_hit);
                    cloud.erase(leftmost_hit);
                    killed = true;
                }
                else
                {
                    if (DEBUG) cout << "  Passed " << leftmost_hit->first << " (L)" << endl;
                }
                break;
            }
        }
        else
        {
            break;
        }
    }

    // look rightwards
    while (true)
    {
        if(rightmost_hit!=cloud.end() && rightmost_hit->first <= current_event.right_end)
        {
            if (sample_unif_dist()<u) {
                if (DEBUG) cout << "  Killed " << rightmost_hit->first << " (R)" << endl;
                deregister_extents(rightmost_hit);
                cloud.erase(rightmost_hit++);
                killed = true;
                continue;
            }
            else
            {
                if (DEBUG) cout << "  Passed " << rightmost_hit->first << " (R)" << endl;
                rightmost_hit++;
                continue;
            }
        }
        else
        {
            break;
        }
    }

    return killed;
}

void SLFV::next_event () {
    event_t& e = current_event;

    // set type
    double type_dbl = sample_unif_dist();
    e.type = (type_dbl<=s) ? selective : neutral;
    if (DEBUG) e.type = (type_dbl<=0.5) ? selective : neutral;
    //e.type = neutral; //DEBUG

    // set time
    double rate = pow(n,1.5) * total_extent; // n from time, n^(0.5) from space
    double interarrival_time = sample_exp_dist() / rate;
    e.time = current_time + interarrival_time;

    // set center & nearest particle
    // we choose between two sampling methods
    double width = rightmost_particle().right_extent - leftmost_particle().left_extent;
    double density = total_extent / width;
    if (((log(cloud.size())) / density <= 0.5 * cloud.size())) //compare O(E[runtime])
    {
        // repeatedly sample a random point within [leftmost_particle().left_extent, rightmost_particle().right_extent]
        // reject until it intersects an extent of some particle
        e.sampling_method = rejection;
        while (true)
        {
            double location = leftmost_particle().left_extent + sample_unif_dist() * width;
            map<double, particle>::iterator r_it = cloud.upper_bound(location);
            map<double, particle>::iterator l_it = r_it;
            if (l_it!=cloud.begin()) --l_it;
            double l_dist = (l_it!=cloud.end() && location>=l_it->first) ? location - l_it->first : 2.0*R;
            double r_dist = (r_it!=cloud.end() && r_it->first>=location) ? r_it->first - location : 2.0*R;
            double dist = min(l_dist, r_dist);
            // cout << location << " " << l_it->first << " " << l_dist << " " << r_it->first << " " << r_dist << endl;
            if (dist<=R){
                if (l_dist<r_dist) {
                    // l_it is closest
                    e.center = location;
                    e.nearest_particle_it = l_it;
                    break;
                }
                else
                {
                    // r_it is closest
                    e.center = location;
                    e.nearest_particle_it = r_it;
                    break;
                }
            }
            // no particles were close, reject
        }
    }
    else
    {
        // first, decide how far along the extent the center will fall
        // second, match [0,extent] onto the (non-contiguous) area in which events affect particles
        e.sampling_method = linear;
        double location_within_extents = sample_unif_dist() * total_extent;
        map<double, particle>::iterator p_it {cloud.begin()};
        double seen_extent {0.0};
        while (true)
        {
            seen_extent += p_it->second.right_extent - p_it->second.left_extent;
            if (location_within_extents <= seen_extent) break;
            ++p_it;
            //assert(p_it!=cloud.end());
        }
        e.nearest_particle_it = p_it;
        e.center = p_it->second.left_extent + sample_unif_dist() * (p_it->second.right_extent - p_it->second.left_extent);
    }

    // set radius
    e.radius = R; // fixed radius
    e.left_end = e.center - e.radius;
    e.right_end = e.center + e.radius;

    // set parents
    if (e.type==neutral) {
        e.parent = e.left_end + sample_unif_dist() * (e.right_end - e.left_end);
    }
    else
    {
        double pre_parent_1 = e.left_end + sample_unif_dist() * (e.right_end - e.left_end);
        double pre_parent_2 = e.left_end + sample_unif_dist() * (e.right_end - e.left_end);
        e.left_pre_parent = min(pre_parent_1,pre_parent_2);
        e.right_pre_parent = max(pre_parent_1,pre_parent_2);
        e.parent = (sample_unif_dist()>0.5) ? pre_parent_1 : pre_parent_2;
    }

    // set id
    ++e.id;

    if (DEBUG)
    {
        cout << "Event: center " << e.center << ", radius " << e.radius << ", nearest " << e.nearest_particle_it->first << endl;
        if (e.type==neutral) {cout << " Neutral: " << e.parent << endl;} else {cout << " Selective: " << e.left_pre_parent << " " << e.right_pre_parent << endl;};
        cout << " Sampling: " << ((e.sampling_method==rejection) ? "rejection " : "linear ") << endl;
    }
}

void SLFV::update_particles ()
{
    // erase dead particles
    bool killed = remove_particles(current_event.center, current_event.radius);

    // insert new particles
    if (killed) {
        if (current_event.type==neutral)
        {
            insert_particle(current_event.parent);
        }else{ //selective
            insert_particle(current_event.left_pre_parent);
            insert_particle(current_event.right_pre_parent);
        }
    }
}


void SLFV::simulate (const double max_time, const bool verbose, datapoint_t& samples)
{
    if (max_time<=current_time) return;
    wanted_time = max_time;

    const double report_interval = 2.0;
    double report_time = -report_interval;

    double sample_time = 0.0;
    if (samples.interval < 1.0/n)
    {
        cout << "error: n is too small for sampling interval " << endl;
        return;
    }

    while (current_time<=wanted_time)
    {
        // sample an event
        next_event();
        current_time = current_event.time;

        // action the event
        update_particles();

        // print stuff
        if ((verbose && (current_time>=report_time+report_interval)) || DEBUG)
        {
            cout << "Time: " << format(current_time,1);
            if (DEBUG) cout << endl;
            cout << " L/R-most: " << format(leftmost_particle().location,4) << " " << format(rightmost_particle().location,4) << ", Total: " << cloud.size() << endl;
            if (DEBUG)
            {
                cout << " All Particles: ";
                for (auto it : cloud) cout << it.first << " ";
                cout << endl;
                if (PAUSE_DEBUG) cin.get();
            }
           report_time = current_time;
        }
        if (current_time>=sample_time)
        {
            // record datapoint
            samples.leftmost.push_back(leftmost_particle().location);
            samples.rightmost.push_back(rightmost_particle().location);
            sample_time += samples.interval;
        }

        if (DEBUG) consistency_check();
    }

    if (DEBUG || verbose) cout << "Total events: " << current_event.id << endl;
    samples.final_particles = cloud.size();
    samples.n_events = current_event.id;
    samples.n_particles = cur_max_particle_id;
}

datapoint_t spawn_model (const double R, const double u, const double n, const double max_time, const int i)
{
    datapoint_t this_datapoint;
    SLFV model {R,u,n,i};
    model.simulate(max_time, false, this_datapoint);
    return this_datapoint;
}

void collate_samples (const double u, vector<datapoint_t>& datapoints, map<double, datapoint_t>& avg, vector<double>::size_type& n_samples)
{
    // due to floating point, the datapoints may have different numbers of samples
    for(datapoint_t& datapoint : datapoints)
    {
        n_samples = min(n_samples, datapoint.leftmost.size());
        n_samples = min(n_samples, datapoint.rightmost.size());
    }
    avg[u].leftmost.resize(n_samples);
    avg[u].rightmost.resize(n_samples);

    vector<double>::size_type j;
    for (datapoint_t& datapoint : datapoints)
    {
        for (j=0; j<n_samples; ++j)
        {
            avg[u].leftmost[j] += datapoint.leftmost[j];
            avg[u].rightmost[j] += datapoint.rightmost[j];
        }
        avg[u].final_particles += datapoint.final_particles;
        avg[u].n_events += datapoint.n_events;
        avg[u].n_particles += datapoint.n_particles;
    }

    for (j=0; j<n_samples; ++j)
    {
        avg[u].leftmost[j] /= (double)datapoints.size();
        avg[u].rightmost[j] /= (double)datapoints.size();
    }
    avg[u].final_particles /= (double)datapoints.size();
    avg[u].n_events /= (double)datapoints.size();
    avg[u].n_particles /= (double)datapoints.size();
}

int main()
{
    DEBUG = false; // print debug info after each event
    PAUSE_DEBUG = false; // wait for keypress after each event (requires DEBUG)

    cout << "d=1 SLFV Genic Selection Simulation" << endl;

    double R {1.0};
    double u {1.0};
    double n {1e4};
    double max_time {15.0};

    // take max_datapoints (each of which is a pair of L/R paths) for each u-k*mu_step until we hit 0
    int wanted_datapoints = 2000;
    double u_step = 0.2;

    vector<double> u_values;
    while (u>0.001)
    {
        u_values.push_back(u);
        if (u_step==0) break;
        u -= u_step;
    }

    // special case if we only want one sample
    if (wanted_datapoints==1)
    {
        // take a sample of left/right-most paths
        datapoint_t this_datapoint;
        SLFV model {R,u_values.front(),n,0};
        model.simulate(max_time, true, this_datapoint);
        return 0;
    }

    // containers for the averaged samples
    // the datapoints may not all have the same number of samples (due to floating point errors)
    vector<datapoint_t> datapoints; // only for the current u (or we run out of mem)
    map<double, datapoint_t> avg; // one for each u
    vector<double>::size_type n_samples {HUGE_VAL};

    // asynchronous sampling
    int seed {0};
    const int max_worker_threads {7};
    const vector<future<datapoint_t>>::size_type n_futures = min(max_worker_threads, wanted_datapoints);

    for (const double u : u_values)
    {
        cout << "Taking " << wanted_datapoints << " samples for u=" << u << endl;
        vector<future<datapoint_t>> futures;
        futures.resize(n_futures);
        int datapoints_requested {0};
        int datapoints_recieved {0};

        // initialize worker threads with a future for each
        for (auto& fu : futures)
        {
            fu = async(launch::async, spawn_model, R, u, n, max_time, seed++);
            ++datapoints_requested;
        }

        // monitor worker threads
        vector<future<datapoint_t>>::iterator fu_it = futures.begin();
        while (datapoints_recieved<wanted_datapoints)
        {
            future<datapoint_t>& fu = *fu_it;

            if (fu.valid() && fu.wait_for(chrono::milliseconds(20))==future_status::ready)
            {
                datapoints.push_back(fu.get()); // save old
                ++datapoints_recieved;
                fu.~future();
                cout << ".";
                if (datapoints_requested<wanted_datapoints)
                {
                    fu = async(launch::async, spawn_model, R, u, n, max_time, seed++); // request new
                    ++datapoints_requested;
                }
            }

            ++fu_it;
            if (fu_it==futures.end()) fu_it = futures.begin();
        }
        cout << endl;

        // collate samples for this u
        if (datapoints.size()<=2)
        {
            cout << "Not enough samples to analyse data; increase max_time" << endl;
            return 0;
        }
        collate_samples(u, datapoints, avg, n_samples);
        datapoints.clear();
    }

    // initialize file output
    string outputfilename;
    stringstream out;
    time_t curtime;
    curtime = time(NULL);
    out << curtime;
    outputfilename = out.str() + ".csv"; //filename is current time + .csv
    ofstream outputfile;
    outputfile.open(outputfilename.c_str(), ofstream::out);

    // dump info
    cout << "Averages:" << endl;
    outputfile << "n = ," << n << ",R = ," << R << endl;
    outputfile << "Averages (" << wanted_datapoints << " samples)" << endl;
    outputfile << "Time,";
    for (auto& el : avg)
    {
        outputfile << "Left (u=" << el.first << "),Right (u=" << el.first << "),";
    }
    outputfile << endl;

    vector<double>::size_type j;
    for (j=0; j<n_samples; ++j)
    {
        outputfile << format((double)j*avg.begin()->second.interval,2) << ",";
        for (auto& el : avg)
        {
            datapoint_t avg_u = el.second;
            outputfile << avg_u.leftmost[j] << "," << avg_u.rightmost[j] << ",";
        }
        outputfile << endl;
    }

    outputfile << endl;
    outputfile << "u,drift" << endl;
    for (auto& el : avg)
    {
        double drift = (-el.second.leftmost[n_samples-1] + el.second.rightmost[n_samples-1])/(2.0*n_samples*el.second.interval);
        outputfile << el.first << "," << drift << endl;
        cout << "  u = " << el.first << ", drift = " << format(drift,3) << ", final particles = " << format(el.second.final_particles,2) << endl;
        cout << "     n_events = " << format(el.second.n_events,2) << ", n_particles = " << format(el.second.n_particles,2) << endl;
    }
    outputfile << endl;
    outputfile << "u,events,particles,final particles" << endl;
    for (auto& el : avg)
    {
        outputfile << el.first << "," << el.second.n_events << "," << el.second.n_particles << "," << el.second.final_particles << endl;
    }

    outputfile.close();

    return 0;
}
