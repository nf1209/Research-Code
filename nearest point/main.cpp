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
    double y;
    double* x;
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

    for (dir = -1; dir<2; dir+=2) // dir = -1 (left) and 1 (right)
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

    // populate the x vector
    int64_t i;
    for(i=0; i<n_point; ++i)
        x_coords.push_back(sample.unif_real_01());

    // populate the y vector, with each point associated to next point from x vector, and order given by perm, then sort
    for (i=0; i<n_point; ++i)
        y_coords.emplace_back(sample.unif_real_01(), &x_coords[i], -1, i);
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
    const int n_init = 1; // number of points of each colour to start with
    if (n_init*n_colour>n_point) throw runtime_error{"not enough points to initialize"};
    for (i=0; i<n_init*n_colour; ++i) {
        y_coords[y_coord_order[i]].colour = i % n_colour;
    }
    for (i=n_init*n_colour; i<n_point; ++i)
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
    // there is no better method of colouring than a hard coded list
    vector<string> colours = {
"FF0000","00FF00","01FFFE","FFA6FE","0000FF","010067","95003A","007DB5",
"FF00F6","FFEEE8","774D00","90FB92","0076FF","D5FF00","FF937E","6A826C",
"FF029D","FE8900","7A4782","7E2DD2","85A900","FF0056","A42400","00AE7E",
"683D3B","BDC6FF","263400","BDD393","00B917","9E008E","001544","C28C9F",
"FF74A3","01D0FF","004754","E56FFE","788231","0E4CA1","91D0CB","BE9970",
"968AE8","BB8800","43002C","DEFF74","00FFC6","FFE502","620E00","008F9C",
"98FF52","7544B1","B500FF","00FF78","FF6E41","005F39","6B6882","5FAD4E",
"A75740","A5FFD2","FFB167","009BFF","E85EBE"
    };
    if (colour_id<0 || colour_id>=colours.size()) throw runtime_error{"invalid colour id"};

    int num = stoi(colours[colour_id], 0, 16); // hex
    int r = num / 0x10000; // 0..255
    int g = (num / 0x100) % 0x100;
    int b = num % 0x100;

    vector<unsigned char> colour;
    colour.resize(3);
    colour[0] = r;
    colour[1] = g;
    colour[2] = b;
    return colour;
}


int main()
{
    // model
    const int n_colour = 61; // 61 max
    const int64_t n_particle = 5e6; // 1e7 particles uses 800mb mem
    const int64_t seed = 1001; // random seed

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
            image.set_pixel(x,y,colour[0],colour[1],colour[2]); // was black, now coloured
        }
        else{
            if (r!=colour[0] || g!=colour[1] || b!=colour[2]) {
                image.set_pixel(x,y,255,255,255); // if we have points of multiple colours in this pixel, set it to be white
                ++n_colour_collison;
            }
            ++n_collision;
        }

        // save images as we go, when i+1==(3^K)*n_colour
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
