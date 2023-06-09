// Template, IGAD version 3
// Get the latest version from: https://github.com/jbikker/tmpl8
// IGAD/NHTV/UU - Jacco Bikker - 2006-2023

#pragma once

#define GRIDSIZE 256
#define GRIDS 65536

namespace Tmpl8
{

union AtomFloat
{
	float f;
	unsigned int u;
};

union AtomFloats
{
	float floats[GRIDS];
	unsigned int uints[GRIDS];
};

class Game : public TheApp
{

public:
	// game flow methods
	void Init();
	void DrawGrid();
	void Simulation();
	void Tick( float deltaTime );
	void Shutdown() { /* implement if you want to do something on exit */ }
	// input handling
	void MouseUp( int ) { /* implement if you want to detect mouse button presses */ }
	void MouseDown( int ) { /* implement if you want to detect mouse button presses */ }
	void MouseMove( int x, int y ) { mousePos.x = x, mousePos.y = y; }
	void MouseWheel( float ) { /* implement if you want to handle the mouse wheel */ }
	void KeyUp( int ) { /* implement if you want to handle keys */ }
	void KeyDown( int ) { /* implement if you want to handle keys */ }
	// data members
	int2 mousePos;

	Kernel* kernel_test = nullptr; 
	Kernel* kernel_gravity = nullptr; 
	
	Kernel* kernel_pulll0 = nullptr;
	Kernel* kernel_pulll1 = nullptr;
	Kernel* kernel_pullr0 = nullptr;
	Kernel* kernel_pullr1 = nullptr;
	Kernel* kernel_pullt0 = nullptr;
	Kernel* kernel_pullt1 = nullptr;
	Kernel* kernel_pullb0 = nullptr;
	Kernel* kernel_pullb1 = nullptr;
	
	Buffer* buffer_pos_x = nullptr;
	Buffer* buffer_pos_y = nullptr;
	Buffer* buffer_fix_x = nullptr;
	Buffer* buffer_fix_y = nullptr;
	Buffer* buffer_prev_pos_x = nullptr;
	Buffer* buffer_prev_pos_y = nullptr;
	Buffer* buffer_length_left = nullptr;
	Buffer* buffer_length_right = nullptr;
	Buffer* buffer_length_top = nullptr;
	Buffer* buffer_length_bot = nullptr;
	
	float fix_x[GRIDSIZE];
	float fix_y[GRIDSIZE];
	
	AtomFloats pos_x;
	AtomFloats pos_y;
	AtomFloats prev_pos_x;
	AtomFloats prev_pos_y;
	float length_left[GRIDS];
	float length_right[GRIDS];
	float length_top[GRIDS];
	float length_bot[GRIDS];
};

} // namespace Tmpl8