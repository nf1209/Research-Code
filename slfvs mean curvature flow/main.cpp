#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>
#include <array>
#include <thread>
#include <mutex>
#include "bitmap_image.hpp"
#include "random_sampler.hpp"

/*
	event driven, forwards in time, multi-threaded SLFVS simulation

	this code will output *a lot* of bitmap files (150Gb+)
	which can then be stitched together into a video and compressed, use e.g. VirtualDub
	run it on a solid state drive, if possible

	the params for the SLFVS are consts in its class
	the params for the video frames are consts in main()

	each worker thread manages a different region of space (with book-keeping on the boundary regions)
	the state space is an array of floats

	the simulation runs *fast* and here's why:
	(1) the array of floats for the state space is properly packed
	(2) consequently, the mem footprint for each worker thread fits into the L1 cache of a modern processor core

	there is a slight approximation to simplify the book-keeping in regions of space shared by multiple threads
	see (*) below for details of what this approximation is
	this approximation could become bad if the number of worker threads exceeds the number of available cores, see below for details
	it could also become bad if the video frames are too long, because global synchronization only occurs at frame boundaries
	-> you should match WORKER_THREADS to cores, and leave enough cores free of other tasks whilst it runs
*/

using namespace std;

const int WORKER_THREADS = 7; // it will run for a long time, keep one core free for yourself!
array<mutex, WORKER_THREADS> mutexes;

class SLFVS_t {
public:
	const int64_t x_res = 3000;
	const int64_t y_res = 2000;

	const double event_rate = 1.0; // expected number of events per pixel (as centre) per unit time
	const int64_t max_events = 0;  // hard limit, 0 does nothing

	// two event radius modes: (1) fixed radius r, (2) pareto tails with exponent alpha
	const float fixed_r = 5.0;       // (1)
	const float alpha = 1.5;   // (2)
	const bool one_parent_fixed = false; // do one parent (neutral) events use (1) or (2)?
	const bool two_parent_fixed = true; // do two parent events use (1) or (2)?
	const bool three_parent_fixed = true; // do three parent use (1) or (2)?

	const float u = 0.02; // killing proportion per event

	const float s = 0.99; // proportion of three-parent (majority voting) selective events
	const float w = 0.00; // proportion of two-parent selective events favouring white
	const float b = 0.00; // proportion of two-parent selective events favouring black
	// it is required that s+w+b<1, and the remaining proportion of events will be neutral


	double model_time = 0.0;
	int64_t n_event = 0;

	SLFVS_t ();
	~SLFVS_t ();
	void execute (const double running_time);
	void execute_threaded (const double running_time);
	void write_to_bitmap (bitmap_image& image) const;

private:
	float* space; // use memory sparingly (-> fit into L1 cache!), use x as major coord

	// irrelevant speed-up
	const float prop_1 = s;
	const float prop_2 = s+w;
	const float prop_3 = s+w+b;

	// reproduction event helpers
	const double global_event_rate;
	float get_pixel (const int64_t x, const int64_t y) const;
	void set_pixel (const int64_t x, const int64_t y, const float val);
	void affect_pixel (const int64_t x, const int64_t y, const float parent);

	void action_event (const int64_t px, const int64_t py);
	bool sample_parent_type (const int64_t px, const int64_t py, const int64_t r, const int64_t sx, const int64_t ex, const int64_t sy, const int64_t ey);
	void clip_event (const int64_t px, const int64_t py, const double r, int64_t& sx, int64_t& sy, int64_t& ex, int64_t &ey);

	// threading
	void worker (const int worker_id, const int64_t my_events);
};

SLFVS_t::SLFVS_t ()
 : global_event_rate(x_res*y_res*event_rate)
{
	space = new float [x_res*y_res]; // guarantees alignment (stl does not!)

	// initial condition
	int64_t x,y;
	int64_t ox = x_res/2;
	int64_t oy = y_res/2;
	for (x=0; x<x_res; ++x) {
	for (y=0; y<y_res; ++y) {
		double r = sqrt(double((x-ox)*(x-ox)+(y-oy)*(y-oy)) / (double)(ox*ox+oy*oy)); // [0,1]
		double theta = atan ((double)(y-oy)/(double)(x-ox)); // (-pi/4,pi/4) or nan
		float val = (r==0.0 || r<=0.1+0.6*(1.0+cos(2.0*theta))*0.5) ? 1.0 : 0.0;
		set_pixel(x,y, val);
	}
	}
}

SLFVS_t::~SLFVS_t ()
{
	delete[] space;
}

void SLFVS_t::write_to_bitmap (bitmap_image& image) const
{
	const int64_t x_dim = image.width();
	const int64_t y_dim = image.height();
    if (x_dim==0 || x_dim>x_res || y_dim==0 || y_dim>y_res) throw runtime_error{"invalid image dimensions"};

    // set each pixel of the image to be the average of the corresponding pixels in space
    int64_t x,y,ix,iy;
    const double x_scale = (double)x_res/(double)x_dim;
    const double y_scale = (double)y_res/(double)y_dim;
    for (ix=0; ix<x_dim; ++ix) {
	for (iy=0; iy<y_dim; ++iy)
	{
		const int64_t sx = ix*x_scale;
		const int64_t sy = iy*y_scale;
		const int64_t ex = min((int64_t)((ix+1)*x_scale), x_res);
		const int64_t ey = min((int64_t)((iy+1)*y_scale), y_res);
		double sum  = 0.0;
		int64_t n = 0;
		for (x=sx; x<ex; ++x) {
		for (y=sy; y<ey; ++y) {
			sum += get_pixel(x,y);
			++n;
		}
		}
		unsigned char iv = (sum /= n) * 255;
		image.set_pixel(ix,iy, iv,iv,iv);
	}
	}
}

inline float SLFVS_t::get_pixel (const int64_t x, const int64_t y) const
{
	if (x<0 || y<0 || x>=x_res || y>=y_res) throw runtime_error{"out of range"};
	return space[x*y_res+y];
}

inline void SLFVS_t::set_pixel (const int64_t x, const int64_t y, const float  val)
{
	if (x<0 || y<0 || x>=x_res || y>=y_res) throw runtime_error{"out of range"};
	space[x*y_res+y] = val;
	return;
}

inline void SLFVS_t::affect_pixel (const int64_t x, const int64_t y, const float parent)
{
	// replace a fraction u of this pixel with the parent type
	if (x<0 || y<0 || x>=x_res || y>=y_res) throw runtime_error{"out of range"};
	space[x*y_res+y] *= (1.0-u);
	if (parent==0.0) return;
	if (parent!=1.0) throw runtime_error{"invalid parent type"};
	space[x*y_res+y] += u;
	return;
}

inline void SLFVS_t::clip_event (const int64_t px, const int64_t py, const double r, int64_t& sx, int64_t& sy, int64_t& ex, int64_t &ey)
{
	sx = (px-r<0) ? 0 : px-r;
	ex = (px+r+1>x_res) ? x_res : px+r+1; // one past end
	sy = (py-r<0) ? 0 : py-r;
	ey = (py+r+1>y_res) ? y_res : py+r+1;
}

inline bool SLFVS_t::sample_parent_type (const int64_t px, const int64_t py, const int64_t r, const int64_t sx, const int64_t ex, const int64_t sy, const int64_t ey)
{
	// return true if the parent is a 1 & false if its a 0
	// assume everything outside is 0s
	if (ex<=sx || ey<=sy) throw runtime_error{"invalid parent sampling region"};
	const float x = sample_uniform_Z(px-r,px+r);
	const float y = sample_uniform_Z(py-r,py+r);
	if (sx<=x && x<ex && sy<=y && y<ey)
	{
		const float p = get_pixel(x,y);
		return (sample_uniform_R()<p);
	}
	return 0;
}

inline void SLFVS_t::action_event (const int64_t px, const int64_t py)
{
	float r;
	float parent;
	int64_t sx,sy,ex,ey; // clipped extents

	// choose the type of this event
	const float prop = sample_uniform_R();
	if (prop<prop_1) // three parent majority vote
	{
		r = (three_parent_fixed) ? fixed_r : sample_pareto(alpha, 1.0);
		clip_event(px,py,r, sx,sy,ex,ey);
		const bool p1 = sample_parent_type(px,py,r, sx,ex,sy,ey);
		const bool p2 = sample_parent_type(px,py,r, sx,ex,sy,ey);
		const bool p3 = sample_parent_type(px,py,r, sx,ex,sy,ey);
		const int white_count = (p1 ? 1 : 0) + (p2 ? 1 : 0) + (p3 ? 1 : 0);
		parent = (white_count>=2) ? 1.0 : 0.0;
	}
	else if (prop<prop_2) // two parent, favour white
	{
		r = (two_parent_fixed) ? fixed_r : sample_pareto(alpha, 1.0);
		clip_event(px,py,r, sx,sy,ex,ey);
		const bool p1 = sample_parent_type(px,py,r, sx,ex,sy,ey);
		const bool p2 = sample_parent_type(px,py,r, sx,ex,sy,ey);
		parent = (p1 || p2) ? 1.0 : 0.0;
	}
	else if (prop<prop_3) // two parent, favour black
	{
		r = (two_parent_fixed) ? fixed_r : sample_pareto(alpha, 1.0);
		clip_event(px,py,r, sx,sy,ex,ey);
		const bool p1 = sample_parent_type(px,py,r, sx,ex,sy,ey);
		const bool p2 = sample_parent_type(px,py,r, sx,ex,sy,ey);
		parent = (p1 && p2) ? 1.0 : 0.0;
	}
	else // neutral
	{
		r = (one_parent_fixed) ? fixed_r : sample_pareto(alpha, 1.0);
		clip_event(px,py,r, sx,sy,ex,ey);
		const bool p1 = sample_parent_type(px,py,r, sx,ex,sy,ey);
		parent = p1 ? 1.0 : 0.0;
	}

	// action
	int64_t x,y;
	for (x=sx; x<ex; ++x) {
		for (y=sy; y<ey; ++y)
		{
			affect_pixel(x,y,parent);
		}
	}
}

void SLFVS_t::worker (const int my_id, const int64_t my_events)
{
	// work out the region of space that this worker thread owns/shares
	// divide space up with cuts of constant y
	// one big region exclusively for each thread, then small shared regions in between the big regions
	const int64_t my_start = (double)x_res/(double)WORKER_THREADS * (double)my_id;
	const int64_t my_end = (double)x_res/(double)WORKER_THREADS * (double)(my_id+1);
	const int64_t lower_shared = (my_id>0) ? my_start+fixed_r+1 : -1; // shared with my_id-1
	const int64_t upper_shared = (my_id<WORKER_THREADS-1) ? my_end-fixed_r-1 : x_res; // shared with my_id+1

	// (*) important note: in between frames we don't synchronize the event order in shared regions of space, we just mutex and hope!
	// assuming the worker threads execute independently and at the same rate, this approximation does not change the distribution of the process
	// use one mutex for each region of space that a (consecutive) pair of worker threads needs to access

	int64_t i;
	for (i=0; i<my_events; ++i)
	{
		// get our next event
		const int64_t px = sample_uniform_Z(my_start,my_end-1);
		const int64_t py = sample_uniform_Z(0,y_res-1);

		// action the event, lock the appropriate mutex if it is in a shared region
		if (px < lower_shared)
		{
			mutexes[my_id].lock();
			action_event(px,py);
			mutexes[my_id].unlock();
		}
		else if (px > upper_shared)
		{
			mutexes[my_id+1].lock();
			action_event(px,py);
			mutexes[my_id+1].unlock();
		}
		else
		{
			action_event(px,py);
		}
	}

}

void SLFVS_t::execute_threaded (const double running_time)
{
	// execution on WORKER_THREADS
	double max_time = model_time + running_time;
	const double worker_event_rate = global_event_rate / (double)WORKER_THREADS;
	const double E_worker_events = worker_event_rate * running_time;

	vector<thread> workers;
	int64_t worker_id;
	for (worker_id=0; worker_id<WORKER_THREADS; ++worker_id) {
		const int64_t worker_events = sample_poisson(E_worker_events);
		n_event += worker_events;
		workers.push_back(thread(&SLFVS_t::worker, this, worker_id, worker_events));
	}

	for (thread& t : workers)
		t.join();

	model_time = max_time;
}

void SLFVS_t::execute (const double running_time)
{
	// single threaded execution
	double max_time = model_time + running_time;
	while (model_time<max_time && (max_events==0 || n_event<max_events))
	{
		const int64_t px = sample_uniform_Z(0,x_res-1);
		const int64_t py = sample_uniform_Z(0,y_res-1);
		action_event(px,py);

		const double event_time = sample_exponential(global_event_rate);
		model_time += event_time;
		++n_event;
	}

}

int main()
{
	cout << setprecision(2) << fixed;
    cout << "SLFVS -> MCF simulation" << endl;

    SLFVS_t SLFVS;
    bitmap_image image {900,600};
    image.set_all_channels(0,0,0);

    SLFVS.write_to_bitmap(image);
    image.save_image("SLFVS_0.bmp");
    cout << "saved initial condition" << endl;

	const int fps = 30.0;
	const double frame_time = 1.0/fps;
	const double max_time = 3000.0;
	const int64_t frame_count = max_time/frame_time;
	cout << "running to time " << max_time << " (" << frame_count << " frames, " << fps << " fps)" << endl;

	const bool threaded = SLFVS.one_parent_fixed && SLFVS.two_parent_fixed && SLFVS.three_parent_fixed;

	int64_t frame = 0;
	while (frame<frame_count && (SLFVS.max_events==0 || SLFVS.n_event<SLFVS.max_events))
	{
		++frame;

		if (threaded)
			SLFVS.execute_threaded(frame_time);
		else
			SLFVS.execute(frame_time);

		cout << "\rmodel_time = " << SLFVS.model_time << ", n_events = " << SLFVS.n_event << ", n_frame = " << frame;

		SLFVS.write_to_bitmap(image);
		image.save_image("SLFVS_" + to_string(frame) + ".bmp");
	}

	cout << "finished" << endl;
    return 0;
}
