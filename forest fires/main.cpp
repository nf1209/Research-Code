#include <iostream>
#include <vector>
#include <list>
#include <cassert>
#include <random>
#include <chrono>
#include <ctime>
#include <thread>
#include <SFML/Graphics.hpp>

using namespace std;
using namespace sf; //SFML graphics library
typedef chrono::duration<double> double_prec_seconds;


struct particle {
    // model stuff
    particle (int particle_id);
    const int id;
    int cluster; // the id of the particle that is the root of this particles cluster
    list<int> companions; //the ids of particles in this cluster, or empty if this particle is not the root of any cluster
    int n_companions; //size of the above vector; computed indirectly because it saves ALOT of time

    // other stuff
    double last_burned;
    bool infected;
};

particle::particle (int particle_id) :id(particle_id) {
    cluster = particle_id; //start as singletons
    companions.push_back(particle_id);
    n_companions = 1;

    last_burned = 0;
    infected = (particle_id==1);
}

enum event_type {growth, fire};

class forest_fire {
public:
    forest_fire (int n_particles);
    vector<particle> cloud; //a cloud of particles
    void simulate (double max_time);
    double time {0.0};
    int cluster_size (int particle_id);
    double speed {1.0};

    const int particles;
    const double E_explosion_size;
private:
    const double lambda;
    const double p_fire;
    void coalesce (int particle_id_1, int particle_id_2);
    void burn (int particle_id);

    void dump_info ();

    default_random_engine random_engine {0};
    uniform_int_distribution<int> unif_dist;
    exponential_distribution<double> exp_dist;

    int random_particle ();
    double event_time ();

    uniform_real_distribution<double> unif_real_dist {0.0, 1.0};
    event_type event ();
};

int forest_fire::cluster_size (int particle_id) {
    return cloud[particle_id].n_companions;
}

forest_fire::forest_fire (int n_particles) :
    particles(n_particles),
    lambda(1.0/sqrt((double)n_particles)),
    E_explosion_size(pow((double)n_particles, 0.5)),
    p_fire(lambda/(1.0+lambda))
{
    // populate with n_particles
    cloud = vector<particle> {};
    for (int i=0; i<n_particles; i++) {
        particle new_particle(i);
        cloud.push_back(new_particle);
    };

    // initialize distributions
    uniform_int_distribution<int> u_dist {0, n_particles-1};
    unif_dist = u_dist;
    exponential_distribution<double> e_dist ((double)n_particles * (1.0 + lambda));
    exp_dist = e_dist;
}

int forest_fire::random_particle () {
    // sample a random particle
    return unif_dist(random_engine);
}

double forest_fire::event_time () {
    // sample a random time until the next event
    return speed*exp_dist(random_engine);
}

event_type forest_fire::event () {
    // decide if we have a fire or a growth
    double ran = unif_real_dist(random_engine);
    if (ran<p_fire) {
        return fire;
    }
    return growth;
}

void forest_fire::dump_info () {
        // print the current state
        for (particle p : cloud) {
            cout << "{";
            for (int p_id : p.companions) {
                cout << p_id << ",";
            }
            cout << "} ";
        }
        cout << endl;
}

void forest_fire::burn (int particle_id) {
    // find the root, then burn the cluster
    int root_id = cloud[particle_id].cluster;
    particle& root = cloud[root_id];
    // first burn all except the root
    for (int id : root.companions) {
        if (id==root_id) continue;
        particle& p = cloud[id];
        assert(p.cluster==root_id);
        assert(p.companions.size()==0);
        p.cluster = p.id;
        p.companions.push_back(p.id);
        p.n_companions = 1;
        p.last_burned = time;
    }
    // now burn the root
    root.companions.clear();
    root.companions.push_back(root_id);
    root.n_companions = 1;
    root.last_burned = time;
}

void forest_fire::coalesce(int particle_1_id, int particle_2_id) {
    // work out which of the two roots is the new root, then coalesce
    if (particle_1_id==particle_2_id) return;
    particle& particle_1 = cloud[particle_1_id];
    particle& particle_2 = cloud[particle_2_id];
    if (particle_1.cluster==particle_2.cluster) return;

    int new_root_id, dead_root_id;
    if (cloud[particle_1.cluster].n_companions>cloud[particle_2.cluster].n_companions) {  // subsume smaller particle into larger
    //if (particle_1.cluster < particle_2.cluster) { //subsume into particle with lowest index
        new_root_id = particle_1.cluster;
        dead_root_id= particle_2.cluster;
    }else{
        new_root_id = particle_2.cluster;
        dead_root_id = particle_1.cluster;
    };

    particle& new_root = cloud[new_root_id];
    particle& dead_root = cloud[dead_root_id];

    // pass on infection, if needed
    if (new_root.infected && !dead_root.infected) {
        for (int id : dead_root.companions) {
            cloud[id].infected = true;
        }
    }
    if (dead_root.infected && !new_root.infected) {
        for (int id : new_root.companions) {
            cloud[id].infected = true;
        }
    }

    new_root.n_companions += dead_root.n_companions;
    for (int id : dead_root.companions) {
        particle& p = cloud[id];
        assert(p.cluster==dead_root_id);
        p.cluster = new_root_id;
        p.n_companions = 0;
    }
    new_root.companions.splice(new_root.companions.end(), dead_root.companions);
}

void forest_fire::simulate (double max_time) {
    // simulate the forest_fire until max_time
    while (time<max_time) {
        time += event_time();
        switch (event()) {
            case fire :
                burn(random_particle());
                break;
            case growth :
                coalesce(random_particle(), random_particle());
                break;
        }
    }
}

int main()
{
    cout << "Forest Fire simulation" << endl;

    // simulation of 1,000,000 particles for 1 second of model time takes about 1 seconds of real time, on my laptop, once compiled
    // runtime is linear in n_particles*max_time
    // memory use is linear in n_particles and constant in max_time; 25*10^6 particles uses about 2Gb of memory
    // this asymptotic is without graphics, i have no idea about those

    // model
    const int n_particles = 2000;
    forest_fire model(n_particles);
    const int n_watched = min(2000, n_particles);
    assert(model.particles>=n_watched);
    model.speed = 1.0; //inverse speed multiplier

    // visualization
    const int fps = 45;
    const double burn_duration = 0.02 * model.speed; //seconds that a particle is drawn as red `while it burns'

    // model timing
    double model_time = 0.0;
    const double grain = 1.0/(double)fps; //model time between two drawframes
    const double max_time = 10000.0;
    const double warning_time = 0.15; //tolerable lag per drawframe (can be high because printing warnings -> more lag -> more warnings)

    // drawframe timing
    chrono::duration<double> chrono_grain(grain), to_sleep;
    chrono::time_point<chrono::system_clock, double_prec_seconds> prev_drawframe, this_drawframe;
    prev_drawframe = chrono::system_clock::now();
    this_drawframe = prev_drawframe;

    // window
    const double window_diam = 1000;
    const int n_rows = (int)((double)n_watched/1000.0*3.0);
    assert(n_rows<=6); //we can fit at most 6 rows of particles on the screen (333 particles per row, 3 pixels for each particle)
    RenderWindow window(VideoMode(window_diam, 100.0*min(6,max(2, n_rows))), "MFFFM");

    // circles
    vector<CircleShape> circles;
    circles.resize(n_watched-1);
    for(CircleShape& circle : circles) {
        circle.setRadius(1.0);
    }

    // circle centres
    int x=0, y=0;
    for(CircleShape& circle : circles) {
        x += 3;
        if (x >=window_diam) {
            x = 0;
            y+=100;
        };
        circle.setPosition(x,y);
    }

    while (window.isOpen() && model_time<=max_time)
    {
        // increment model
        model_time += grain;
        model.simulate(model_time);
        //cout << model_time << endl;

        // get new circle radii & colours, set positions
        int cluster_size;
        int id = 0;
        for(CircleShape& circle : circles) {
            ++id;
            cluster_size = model.cluster_size(id);
            if ((model.cloud[id].last_burned + burn_duration >= model_time) && model_time > burn_duration) {
                // draw cluster as large and red for a short time to show burning
                circle.setFillColor(Color::Red);
            }else{
                circle.setRadius(pow((double)cluster_size,0.5)); //deliberately omit factor of pi, else the circles are too small
                    if (cluster_size<=20) {
                        circle.setFillColor(Color::Green);
                    }else{
                        circle.setFillColor(Color::Blue);
                    }
            }
        };

        // increment real_time , sleep if needed (TODO: handle model time lagging behind real time?)
        this_drawframe = prev_drawframe + chrono_grain;
        to_sleep = this_drawframe - chrono::system_clock::now();
        if (to_sleep.count()>=-warning_time) {
            this_thread::sleep_for(to_sleep);
        }else{
            cout << "Warning: model_time lagged behind real time by " << -to_sleep.count() << " seconds." << endl;
        }

        // was the window closed?
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // drawframe
        window.clear();
        for(CircleShape& circle : circles){
            window.draw(circle);
        }
        window.display();

        // record drawframe
        prev_drawframe = this_drawframe;
    }

    return 0;
}
