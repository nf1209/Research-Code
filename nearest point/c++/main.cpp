#include <iostream>
#include <random>
#include <algorithm>
#include <stdexcept>
#include "math.h"
#include "bitmap_image.hpp"
#include <set>
#include <vector>
#include <string>

using namespace std;

class sampler_t {
private:
    default_random_engine random_engine; // seed
    uniform_real_distribution<double> unif_real_dist {0.0, 1.0};
public:
    sampler_t () {default_random_engine r{1}; random_engine=r;} ;
    sampler_t (int64_t seed) {default_random_engine r{seed}; random_engine=r;} ;

    double unif_real_01() {return unif_real_dist(random_engine);} ;
};

class matched_y_coord {
public:
    double* x;
    double y;
    int64_t order; // position in the order in which the points fell
    int colour;
    matched_y_coord (double py, double* px_ptr, int col, int64_t ord) : y(py), x(px_ptr), colour(col), order(ord) {};
};

bool operator< (const matched_y_coord& a, const matched_y_coord& b)
{
    return a.y < b.y;
}

class point_t {
public:
    // bona fide coloured point (used for export only)
    double x;
    double y;
    int colour;
    point_t (double px, double py, int col) : x(px), y(py), colour(col) {};
};

class model_t {
public:
    // config
    const int64_t n_point;
    const int n_colour;
    const int64_t seed;

    // access
    vector<point_t> ordered_points ();

    // init
    model_t (const int64_t, const int, const int64_t);
    sampler_t sample;
 private:
     int64_t find_nearest_point_before(const int64_t id) const;

    // data
    vector<int64_t> y_coord_order;
    vector<matched_y_coord> y_coords;
    vector<double> x_coords;
};

int64_t model_t::find_nearest_point_before(const int64_t id) const
{
    if (id<0 or id>=x_coords.size()) throw runtime_error{"invalid id"};

    // "new" point
    const matched_y_coord& point = y_coords[id];
    double y = point.y;
    double x = *(point.x);
    int64_t order = point.order;

    // cycle over already existing points
    int dir;
    double cur_best_sqr_dist = 2.0; // assume we are working in [0,1]x[0,1]
    int64_t cur_best_id = -1;

    for (dir = -1; dir<2; dir+=2) // dir = -1 and 1
    {
        // go left or right, searching for closest point, until y coord dist is such that it can't be the closest point given the current best
        int64_t displacement = 0;
        while (true)
        {
            ++displacement;

            int64_t candidate_id = id + displacement*dir;
            if (candidate_id<0 || candidate_id>=y_coords.size()) break; // gone too far, reached begin/end of vector

            int64_t corder = y_coords[candidate_id].order;
            if (corder>=order) continue; // this candidate has not yet appeared

            double cy = y_coords[candidate_id].y;
            double sqr_y_dist = (y-cy)*(y-cy);
            if (sqr_y_dist>cur_best_sqr_dist) break; // all further points in this dir will have more y disp -> can't beat best

            double cx = *(y_coords[candidate_id].x);
            double sqr_x_dist = (x-cx)*(x-cx);
            if (sqr_x_dist>=cur_best_sqr_dist) continue;

            double sqr_dist = sqr_y_dist + sqr_x_dist;
            if (sqr_dist<cur_best_sqr_dist) {
                cur_best_sqr_dist = sqr_dist;
                cur_best_id = candidate_id;
            }
        }
    }

    if (cur_best_id<0) throw runtime_error{"no best found"};
    return cur_best_id;
}

model_t::model_t (const int64_t n, const int c, const int64_t s)
 : n_point(n), n_colour(c), seed(s)
{
    // RNG
    sampler_t sample(seed);

    // populate the x vector, then sort
    int64_t i;
    for(i=0; i<n_point; ++i)
        x_coords.push_back(sample.unif_real_01());
    sort(x_coords.begin(), x_coords.end());

    // construct uniform random permutation of {0,..,n_point}, using a Knuth shuffle
    vector<int64_t> perm;
    perm.resize(n_point);
    for (i=0; i<n_point; ++i) perm[i] = i;
    for (i=0; i<n_point-1; ++i)
    {
        int64_t j = i + (double)(n-i)*sample.unif_real_01();
        int64_t temp = perm[j];
        perm[j] = perm[i];
        perm[i] = temp;
    }

    // populate the y vector, with each point associated to next point from x vector, and order given by perm, then sort
    for (i=0; i<n_point; ++i)
        y_coords.emplace_back(sample.unif_real_01(), &x_coords[i], -1, perm[i]);
    sort(y_coords.begin(), y_coords.end());

    // map out which order the points, indexed by y.order, were added in
    y_coord_order.resize(n_point);
    for (i=0; i<n_point; ++i)
    {
        int64_t order = y_coords[i].order;
        y_coord_order[order] = i;
    }
    cout << "Generated " << n_point << " points." << endl;

    // colour the points
    for (i=0; i<n_colour; ++i) {
        y_coords[y_coord_order[i]].colour = i;
    }
    for (i=n_colour; i<n_point; ++i)
    {
       int64_t id = y_coord_order[i];
       int64_t nearest_id = find_nearest_point_before(id);
       y_coords[id].colour = y_coords[nearest_id].colour;
       if (i%1000==0) cout << "\r" << "Colouring points: " << i << "/" << n_point;
    }
    cout << "\r" << "Colouring points: " << n_point << "/" << n_point << endl;

    // consistency check
    for (i=0; i<n_point; ++i)
    {
        if (y_coords[i].colour<0) throw runtime_error{"invalid colour"};
    }

}

vector<point_t> model_t::ordered_points ()
{
    // repackage the points, in order of appearance, neatly into a (single) vector
    vector<point_t> points;
    int64_t i;
    for (i=0; i<n_point; ++i)
    {
        int64_t id = y_coord_order[i];
        matched_y_coord& point = y_coords[id];
        points.emplace_back(*(point.x), point.y, point.colour);
    }
    return points;
}

vector<unsigned char> RGB_colour (const int colour_id)
{
    vector<unsigned char> colour;
    colour.resize(3);
    unsigned char red,green,blue;
    switch (colour_id) {
        case 0:
            red=255; green=0; blue=0;
            break;
        case 1:
            red=0; green=255; blue=255;
            break;
        case 2:
            red=255; green=255; blue=0;
            break;
        case 3:
            red=255; green=0; blue=255;
            break;
        case 4:
            red=0; green=255; blue=0;
            break;
        case 5:
            red=255; green=128; blue=0;
            break;
        case 6:
            red=0; green=128; blue=255;
            break;
        case 7:
            red=255; green=153; blue=255;
            break;
        case 8:
            red=0; green=153; blue=76;
            break;
        case 9:
            red=0; green=0; blue=153;
            break;
        case 10:
            red=102; green=0; blue=204;
            break;
        case 11:
            red=153; green=153; blue=0;
            break;
        default:
            throw runtime_error{"invalid colour"};
    }
    colour[1] = red;
    colour[2] = green;
    colour[3] = blue;
    return colour;
}


int main()
{
    // model
    const int n_colour = 12; // 12 max
    const int64_t n_particle = 5e6; // 1e7 particles uses 800mb mem
    const int64_t seed = 998;
    model_t model(n_particle, n_colour, seed);
    vector<point_t> points = model.ordered_points();

    // init image
    const int res = 1200; // image is (res)x(res)
    bitmap_image image(res,res);
    image.set_all_channels(0,0,0); // black background

    // write images
    int64_t n_collision = 0; // number of pixels with >1 point
    int64_t n_colour_collison = 0; // number of pixels with points of >1 colour

    int64_t i;
    int64_t prev_sample = 0;
    for(i=0; i<points.size(); ++i)
    {
        // work out which pixel we want to place this point in
        point_t& point = points[i];
        int x = point.x*(double)res;
        int y = point.y*(double)res;
        const vector<unsigned char> colour = RGB_colour(point.colour);

        // assign colour to pixel, record collisions
        unsigned char r,g,b;
        image.get_pixel(x,y,r,g,b);
        if (r==0 && g==0 && b==0) {
            image.set_pixel(x,y,colour[1],colour[2],colour[3]); // was black, now coloured
        }
        else{
            if (r!=colour[1] || g!=colour[2] || b!=colour[3]) {
                image.set_pixel(x,y,255,255,255); // if we have points of multiple colours in this pixel, set it to be white
                ++n_colour_collison;
            }
            ++n_collision;
        }

        // save images as we go
        if (i+1==3*prev_sample || i+1==n_colour)
        {
            string filename = to_string(i+1) + ".bmp";
            image.save_image(filename);
            prev_sample = i+1;
        }
    }

    string filename = to_string(points.size()) + ".bmp";
    image.save_image(filename);

    cout << "Wrote " << res << "x" << res << " images. " << endl;
    cout << "Had " << n_collision << "/" << res*res << " pixel collisions, " << n_colour_collison << "/" << res*res << " colour collisions." << endl;

    return 0;
}
