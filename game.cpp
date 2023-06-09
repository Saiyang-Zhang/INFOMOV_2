#include "precomp.h"
#include "game.h"

struct Point
{
	float2 pos;				// current position of the point
	float2 prev_pos;		// position of the point in the previous frame
	float2 fix;				// stationary position; used for the top line of points
	bool fixed;				// true if this is a point in the top line of the cloth
	float restlength[4];	// initial distance to neighbours
};

// grid access convenience
Point* pointGrid = new Point[GRIDSIZE * GRIDSIZE];
Point& grid(const uint x, const uint y) { return pointGrid[x + y * GRIDSIZE]; }

// grid offsets for the neighbours via the four links
int xoffset[4] = { 1, -1, 0, 0 }, yoffset[4] = { 0, 0, 1, -1 };

// initialization
void Game::Init()
{
	// create the cloth
	for (int y = 0; y < GRIDSIZE; y++) for (int x = 0; x < GRIDSIZE; x++)
	{
		grid(x, y).pos.x = 10 + (float)x * ((SCRWIDTH - 100) / GRIDSIZE) + y * 0.9f + Rand(2);
		grid(x, y).pos.y = 10 + (float)y * ((SCRHEIGHT - 180) / GRIDSIZE) + Rand(2);
		grid(x, y).prev_pos = grid(x, y).pos; // all points start stationary
		if (y == 0)
		{
			grid(x, y).fixed = true;
			grid(x, y).fix = grid(x, y).pos;
		}
		else
		{
			grid(x, y).fixed = false;
		}
	}
	for (int y = 1; y < GRIDSIZE - 1; y++) for (int x = 1; x < GRIDSIZE - 1; x++)
	{
		// calculate and store distance to four neighbours, allow 15% slack
		for (int c = 0; c < 4; c++)
		{
			grid(x, y).restlength[c] = length(grid(x, y).pos - grid(x + xoffset[c], y + yoffset[c]).pos) * 1.15f;
		}
	}

	// create the cloth
	for (int y = 0; y < GRIDSIZE; y++) for (int x = 0; x < GRIDSIZE; x++)
	{
		int index = x + y * GRIDSIZE;
		pos_x.floats[index] = 10 + (float)x * ((SCRWIDTH - 100) / GRIDSIZE) + y * 0.9f + Rand( 2 );
		pos_y.floats[index] = 10 + (float)y * ((SCRHEIGHT - 180) / GRIDSIZE) + Rand( 2 );
		prev_pos_x.floats[index] = pos_x.floats[index];
		prev_pos_y.floats[index] = pos_y.floats[index];
	}
	
	for (int i = 0; i < GRIDSIZE; i++)
	{
		fix_x[i] = pos_x.floats[i];
		fix_y[i] = pos_y.floats[i];
	}
	
	for (int y = 1; y < GRIDSIZE - 1; y++) for (int x = 1; x < GRIDSIZE - 1; x++)
	{
		// calculate and store distance to four neighbours, allow 15% slack
		
		int index = x + y * GRIDSIZE;
		float left_diff_x = pos_x.floats[index] - pos_x.floats[index - 1];
		float left_diff_y = pos_y.floats[index] - pos_y.floats[index - 1];
		float right_diff_x = pos_x.floats[index + 1] - pos_x.floats[index];
		float right_diff_y = pos_y.floats[index + 1] - pos_y.floats[index];
		float top_diff_x = pos_x.floats[index] - pos_x.floats[index - GRIDSIZE];
		float top_diff_y = pos_y.floats[index] - pos_y.floats[index - GRIDSIZE];
		float bot_diff_x = pos_x.floats[index + GRIDSIZE] - pos_x.floats[index];
		float bot_diff_y = pos_y.floats[index + GRIDSIZE] - pos_y.floats[index];
		length_left[index] = sqrt(left_diff_x * left_diff_x + left_diff_y * left_diff_y) * 1.15f;
		length_right[index] = sqrt(right_diff_x * right_diff_x + right_diff_y * right_diff_y) * 1.15f;
		length_top[index] = sqrt(top_diff_x * top_diff_x + top_diff_y * top_diff_y) * 1.15f;
		length_bot[index] = sqrt(bot_diff_x * bot_diff_x + bot_diff_y * bot_diff_y) * 1.15f;
	}
	
	kernel_gravity = new Kernel("kernel.cl", "gravity");
	kernel_pulll0 = new Kernel("kernel.cl", "pulll0");
	kernel_pulll1 = new Kernel("kernel.cl", "pulll1");
	kernel_pullr0 = new Kernel("kernel.cl", "pullr0");
	kernel_pullr1 = new Kernel("kernel.cl", "pullr1");
	kernel_pullt0 = new Kernel("kernel.cl", "pullt0");
	kernel_pullt1 = new Kernel("kernel.cl", "pullt1");
	kernel_pullb0 = new Kernel("kernel.cl", "pullb0");
	kernel_pullb1 = new Kernel("kernel.cl", "pullb1");
	buffer_pos_x = new Buffer(sizeof(unsigned int) * GRIDS, pos_x.uints, CL_MEM_READ_WRITE);
	buffer_pos_y = new Buffer(sizeof(unsigned int) * GRIDS, pos_y.uints, CL_MEM_READ_WRITE);
	buffer_prev_pos_x = new Buffer(sizeof(unsigned int) * GRIDS, prev_pos_x.uints, CL_MEM_READ_WRITE);
	buffer_prev_pos_y = new Buffer(sizeof(unsigned int) * GRIDS, prev_pos_y.uints, CL_MEM_READ_WRITE);
	buffer_length_left = new Buffer(sizeof(float) * GRIDS, length_left, CL_MEM_READ_ONLY);
	buffer_length_right = new Buffer(sizeof(float) * GRIDS, length_right, CL_MEM_READ_ONLY);
	buffer_length_top = new Buffer(sizeof(float) * GRIDS, length_top, CL_MEM_READ_ONLY);
	buffer_length_bot = new Buffer(sizeof(float) * GRIDS, length_bot, CL_MEM_READ_ONLY);

	buffer_length_left->CopyToDevice();
	buffer_length_right->CopyToDevice();
	buffer_length_top->CopyToDevice();
	buffer_length_bot->CopyToDevice();

	//kernel_test = new Kernel("kernel.cl", "test");

	//float a[1024];
	//float b[1024];
	//size_t global_size = 1024;
	//size_t local_size = 128;
	//
	//for (int i = 0; i < 1024; i++) {
	//	a[i] = i;
	//	b[i] = 2;
	//}
	//unsigned int flag[10];
	//Buffer* buffer_flag = new Buffer(sizeof(unsigned int) * 10, flag, CL_MEM_READ_WRITE);
	//Buffer* buffer_a = new Buffer(sizeof(float) * 1024, a, CL_MEM_READ_WRITE);
	//Buffer* buffer_b = new Buffer(sizeof(float) * 1024, b, CL_MEM_READ_WRITE);

	//buffer_flag->CopyToDevice();
	//buffer_a->CopyToDevice();
	//buffer_b->CopyToDevice();
	//kernel_test->SetArguments(buffer_flag, buffer_a, buffer_b);
	//kernel_test->Run(global_size, local_size);
	//buffer_a->CopyFromDevice();

	//for (int i = 1014; i < 1024; i++)
	//	printf("%f\n", a[i]);

}

// cloth rendering
// NOTE: For this assignment, please do not attempt to render directly on
// the GPU. Instead, if you use GPGPU, retrieve simulation results each frame
// and render using the function below. Do not modify / optimize it.
void Game::DrawGrid()
{
	// draw the grid
	screen->Clear( 0 );
	for (int y = 0; y < (GRIDSIZE - 1); y++) for (int x = 1; x < (GRIDSIZE - 2); x++)
	{
		int index = x + y * GRIDSIZE;
		const float p1x = pos_x.floats[index], p1y = pos_y.floats[index];
		const float p2x = pos_x.floats[index + 1], p2y = pos_y.floats[index + 1];
		const float p3x = pos_x.floats[index + GRIDSIZE], p3y = pos_y.floats[index + GRIDSIZE];
		screen->Line( p1x, p1y, p2x, p2y, 0xffffff );
		screen->Line( p1x, p1y, p3x, p3y, 0xffffff );
	}
	for (int y = 0; y < (GRIDSIZE - 1); y++)
	{
		int index = GRIDSIZE - 2 + y * GRIDSIZE;
		const float p1x = pos_x.floats[index], p1y = pos_y.floats[index];
		const float p2x = pos_x.floats[index + GRIDSIZE], p2y = pos_y.floats[index + GRIDSIZE];
		screen->Line( p1x, p1y, p2x, p2y, 0xffffff );
	}
}

// cloth simulation
// This function implements Verlet integration (see notes at top of file).
// Important: when constraints are applied, typically two points are
// drawn together to restore the rest length. When running on the GPU or
// when using SIMD, this will only work if the two vertices are not
// operated upon simultaneously (in a vector register, or in a warp).
float magic = 0.11f;
void Game::Simulation()
{

	// simulation is exected three times per frame; do not change this.
	for( int steps = 0; steps < 3; steps++ )
	{
		// verlet integration; apply gravity
		buffer_pos_x->CopyToDevice();
		buffer_pos_y->CopyToDevice();
		buffer_prev_pos_x->CopyToDevice();
		buffer_prev_pos_y->CopyToDevice();
		kernel_gravity->SetArguments(magic, buffer_pos_x, buffer_pos_y, buffer_prev_pos_x, buffer_prev_pos_y);
		kernel_gravity->Run(GRIDS, 1024);
		buffer_pos_x->CopyFromDevice();
		buffer_pos_y->CopyFromDevice();
		buffer_prev_pos_x->CopyFromDevice();
		buffer_prev_pos_y->CopyFromDevice();

		float extra, mov_x, mov_y;
		int flag = GRIDSIZE - 1;
		magic += 0.0002f; // slowly increases the chance of anomalies

		// apply constraints; 4 simulation steps: do not change this number.
		for (int iter = 0; iter < 4; iter++)
		{
			buffer_pos_x->CopyToDevice();
			buffer_pos_y->CopyToDevice();

			kernel_pulll0->SetArguments(
				buffer_pos_x, buffer_pos_y,
				buffer_length_left
			);
			kernel_pulll0->Run(GRIDS, 1024);
		
			kernel_pulll1->SetArguments(
				buffer_pos_x, buffer_pos_y,
				buffer_length_left
			);
			kernel_pulll1->Run(GRIDS, 1024);
 
			kernel_pullr0->SetArguments(
				buffer_pos_x, buffer_pos_y,
				buffer_length_right
			);
			kernel_pullr0->Run(GRIDS, 1024);
		
			kernel_pullr1->SetArguments(
				buffer_pos_x, buffer_pos_y,
				buffer_length_right
			);
			kernel_pullr1->Run(GRIDS, 1024);
 
			kernel_pullt0->SetArguments(
				buffer_pos_x, buffer_pos_y,
				buffer_length_top
			);
			kernel_pullt0->Run(GRIDS, 1024);

			kernel_pullt1->SetArguments(
				buffer_pos_x, buffer_pos_y,
				buffer_length_top
			);
			kernel_pullt1->Run(GRIDS, 1024);
 
			kernel_pullb0->SetArguments(
				buffer_pos_x, buffer_pos_y,
				buffer_length_bot
			);
			kernel_pullb0->Run(GRIDS, 1024);

			kernel_pullb1->SetArguments(
				buffer_pos_x, buffer_pos_y,
				buffer_length_bot
			);
			kernel_pullb1->Run(GRIDS, 1024);

			buffer_pos_x->CopyFromDevice();
			buffer_pos_y->CopyFromDevice();

			for (int i = 0; i < GRIDSIZE; i++)
			{
				pos_x.floats[i] = fix_x[i];
				pos_y.floats[i] = fix_y[i];
			}
		}
	}
}

void Game::Tick( float a_DT )
{
	// update the simulation
	Timer tm;
	tm.reset();
	Simulation();
	float elapsed1 = tm.elapsed();

	// draw the grid
	tm.reset();
	DrawGrid();
	float elapsed2 = tm.elapsed();

	// display statistics
	char t[128];
	sprintf( t, "ye olde ruggeth cloth simulation: %5.1f ms", elapsed1 * 1000 );
	screen->Print( t, 2, SCRHEIGHT - 24, 0xffffff );
	sprintf( t, "                       rendering: %5.1f ms", elapsed2 * 1000 );
	screen->Print( t, 2, SCRHEIGHT - 14, 0xffffff );
}