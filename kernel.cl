#define GRIDSIZE 256
#define GRIDS 65536

uint seed;

union AtomFloat {
	float f;
	unsigned int u;
};

uint RandomUInt()
{
	seed ^= seed << 13;
	seed ^= seed >> 17;
	seed ^= seed << 5;
	return seed;
}
float RandomFloat() { return RandomUInt() * 2.3283064365387e-10f; }
float Rand( float range ) { return RandomFloat() * range; }

__kernel void test(volatile __global uint* flag, __global float* a, __global float* b )
{
	int idx = get_global_id( 0 );
	atomic_inc(&flag[0]);
	if(isfinite(a[idx])) ;
	a[idx] *= b[idx];
	a[idx] = sqrt(a[idx]);
}

__kernel void gravity(float magic, 
	__global unsigned int* pos_x, __global unsigned int* pos_y, 
	__global unsigned int* prev_pos_x, __global unsigned int* prev_pos_y)
{
	int idx = get_global_id( 0 );
	
	union AtomFloat posx, posy, curx, cury, prevx, prevy;
	posx.u = pos_x[idx]; posy.u = pos_y[idx]; prevx.u = prev_pos_x[idx]; prevy.u = prev_pos_y[idx];
	float curpos_x = posx.f, curpos_y = posy.f, prevpos_x = prevx.f, prevpos_y = prevy.f;
	curx.f = curpos_x + curpos_x - prevpos_x;
	cury.f = curpos_y + curpos_y - prevpos_y + 0.003f;
	pos_x[idx] = curx.u;
	pos_y[idx] = cury.u;
	prev_pos_x[idx] = posx.u;
	prev_pos_y[idx] = posy.u;

	union AtomFloat a, b; 
	a.f = curpos_x; b.f = curpos_y; seed = a.u ^ ((b.u << 16) + (b.u >> 16)); 
	
	if (Rand(10) < 0.03f)
	{
		curx.f += Rand(0.02f + magic);
		pos_x[idx] = curx.u;
		
		cury.f += Rand(0.12f);
		pos_y[idx] = cury.u;
	}
}

__kernel void pulll0(
	volatile __global unsigned int* pos_x, volatile __global unsigned int* pos_y, __global float* length_left)
{
	int idx = get_global_id(0), col = 1, flag = GRIDSIZE - 1;
	if (idx < GRIDSIZE || idx >= GRIDS - GRIDSIZE || (idx & col) != 0) return;		
	if ((idx & flag) == 0 || (idx & flag) == flag) return;	
	
	float extra, mov_x, mov_y;
	union AtomFloat atom_pos_x, atom_pos_y;
	union AtomFloat atom_left_x, atom_left_y;

	atom_pos_x.u = pos_x[idx]; atom_pos_y.u = pos_y[idx];
	atom_left_x.u = pos_x[idx - 1]; atom_left_y.u = pos_y[idx - 1];

	float left_diff_x = atom_left_x.f - atom_pos_x.f, left_diff_y = atom_left_y.f - atom_pos_y.f;
	float left_len = sqrt(left_diff_x * left_diff_x + left_diff_y * left_diff_y);				

	if (isfinite(left_len) && left_len > length_left[idx])
	{
		extra = left_len / length_left[idx] - 1;
		mov_x = extra * left_diff_x * 0.5f;
		mov_y = extra * left_diff_y * 0.5f;
		atom_pos_x.f += mov_x; 
		atom_pos_y.f += mov_y;
		atom_left_x.f -= mov_x; 
		atom_left_y.f -= mov_y;
	}
	pos_x[idx] = atom_pos_x.u;
	pos_y[idx] = atom_pos_y.u;
	pos_x[idx - 1] = atom_left_x.u;
	pos_y[idx - 1] = atom_left_y.u;
}

__kernel void pulll1(volatile __global int* pos_x, volatile __global unsigned int* pos_y, __global float* length_left)
{
	int idx = get_global_id(0), col = 1, flag = GRIDSIZE - 1;
	if (idx < GRIDSIZE || idx >= GRIDS - GRIDSIZE || (idx & col) != 1) return;		
	if ((idx & flag) == 0 || (idx & flag) == flag) return;	
	
	float extra, mov_x, mov_y;
	union AtomFloat atom_pos_x, atom_pos_y;
	union AtomFloat atom_left_x, atom_left_y;

	atom_pos_x.u = pos_x[idx]; atom_pos_y.u = pos_y[idx];
	atom_left_x.u = pos_x[idx - 1]; atom_left_y.u = pos_y[idx - 1];

	float left_diff_x = atom_left_x.f - atom_pos_x.f, left_diff_y = atom_left_y.f - atom_pos_y.f;
	float left_len = sqrt(left_diff_x * left_diff_x + left_diff_y * left_diff_y);				

	if (isfinite(left_len) && left_len > length_left[idx])
	{
		extra = left_len / length_left[idx] - 1;
		mov_x = extra * left_diff_x * 0.5f;
		mov_y = extra * left_diff_y * 0.5f;
		atom_pos_x.f += mov_x; 
		atom_pos_y.f += mov_y;
		atom_left_x.f -= mov_x; 
		atom_left_y.f -= mov_y;
	}
	pos_x[idx] = atom_pos_x.u;
	pos_y[idx] = atom_pos_y.u;
	pos_x[idx - 1] = atom_left_x.u;
	pos_y[idx - 1] = atom_left_y.u;
}

__kernel void pullr0(volatile __global int* pos_x, volatile __global unsigned int* pos_y, __global float* length_right)
{
	int idx = get_global_id(0), col = 1, flag = GRIDSIZE - 1;
	if (idx < GRIDSIZE || idx >= GRIDS - GRIDSIZE || (idx & col) != 0 ) return;		
	if ((idx & flag) == 0 || (idx & flag) == flag) return;	
	
	float extra, mov_x, mov_y;
	union AtomFloat atom_pos_x, atom_pos_y;
	union AtomFloat atom_right_x, atom_right_y;

	atom_pos_x.u = pos_x[idx]; atom_pos_y.u = pos_y[idx];
	atom_right_x.u = pos_x[idx + 1]; atom_right_y.u = pos_y[idx + 1];

	float right_diff_x = atom_right_x.f - atom_pos_x.f, right_diff_y = atom_right_y.f - atom_pos_y.f;
	float right_len = sqrt(right_diff_x * right_diff_x + right_diff_y * right_diff_y);				

	if (isfinite(right_len) && right_len > length_right[idx])
	{
		extra = right_len / length_right[idx] - 1;
		mov_x = extra * right_diff_x * 0.5f;
		mov_y = extra * right_diff_y * 0.5f;
		atom_pos_x.f += mov_x; 
		atom_pos_y.f += mov_y;
		atom_right_x.f -= mov_x; 
		atom_right_y.f -= mov_y;
	}
	pos_x[idx] = atom_pos_x.u;
	pos_y[idx] = atom_pos_y.u;
	pos_x[idx + 1] = atom_right_x.u;
	pos_y[idx + 1] = atom_right_y.u;
}

__kernel void pullr1(volatile __global int* pos_x, volatile __global unsigned int* pos_y, __global float* length_right)
{
	int idx = get_global_id(0), col = 1, flag = GRIDSIZE - 1;
	if (idx < GRIDSIZE || idx >= GRIDS - GRIDSIZE || (idx & col) != 1) return;		
	if ((idx & flag) == 0 || (idx & flag) == flag) return;	
	
	float extra, mov_x, mov_y;
	union AtomFloat atom_pos_x, atom_pos_y;
	union AtomFloat atom_right_x, atom_right_y;

	atom_pos_x.u = pos_x[idx]; atom_pos_y.u = pos_y[idx];
	atom_right_x.u = pos_x[idx + 1]; atom_right_y.u = pos_y[idx + 1];

	float right_diff_x = atom_right_x.f - atom_pos_x.f, right_diff_y = atom_right_y.f - atom_pos_y.f;
	float right_len = sqrt(right_diff_x * right_diff_x + right_diff_y * right_diff_y);				

	if (isfinite(right_len) && right_len > length_right[idx])
	{
		extra = right_len / length_right[idx] - 1;
		mov_x = extra * right_diff_x * 0.5f;
		mov_y = extra * right_diff_y * 0.5f;
		atom_pos_x.f += mov_x; 
		atom_pos_y.f += mov_y;
		atom_right_x.f -= mov_x; 
		atom_right_y.f -= mov_y;
	}
	pos_x[idx] = atom_pos_x.u;
	pos_y[idx] = atom_pos_y.u;
	pos_x[idx + 1] = atom_right_x.u;
	pos_y[idx + 1] = atom_right_y.u;
}

__kernel void pullt0(volatile __global int* pos_x, volatile __global unsigned int* pos_y, __global float* length_top)
{
	int idx = get_global_id(0), row = 1, flag = GRIDSIZE - 1;
	if (idx < GRIDSIZE || idx >= GRIDS - GRIDSIZE || ((idx >> 8) & row) != 0) return;		
	if ((idx & flag) == 0 || (idx & flag) == flag) return;	
	

	float extra, mov_x, mov_y;
	union AtomFloat atom_pos_x, atom_pos_y;
	union AtomFloat atom_top_x, atom_top_y;

	atom_pos_x.u = pos_x[idx]; atom_pos_y.u = pos_y[idx];
	atom_top_x.u = pos_x[idx - GRIDSIZE]; atom_top_y.u = pos_y[idx - GRIDSIZE];

	float top_diff_x = atom_top_x.f - atom_pos_x.f, top_diff_y = atom_top_y.f - atom_pos_y.f;
	float top_len = sqrt(top_diff_x * top_diff_x + top_diff_y * top_diff_y);				

	if (isfinite(top_len) && top_len > length_top[idx])
	{
		extra = top_len / length_top[idx] - 1;
		mov_x = extra * top_diff_x * 0.5f;
		mov_y = extra * top_diff_y * 0.5f;
		atom_pos_x.f += mov_x; 
		atom_pos_y.f += mov_y;
		atom_top_x.f -= mov_x; 
		atom_top_y.f -= mov_y;
	}
	pos_x[idx] = atom_pos_x.u;
	pos_y[idx] = atom_pos_y.u;
	pos_x[idx - GRIDSIZE] = atom_top_x.u;
	pos_y[idx - GRIDSIZE] = atom_top_y.u;
}

__kernel void pullt1(volatile __global int* pos_x, volatile __global unsigned int* pos_y, __global float* length_top)
{
	int idx = get_global_id(0), row = 1, flag = GRIDSIZE - 1;
	if (idx < GRIDSIZE || idx >= GRIDS - GRIDSIZE || ((idx >> 8) & row) != 1) return;		
	if ((idx & flag) == 0 || (idx & flag) == flag) return;	
	
	float extra, mov_x, mov_y;
	union AtomFloat atom_pos_x, atom_pos_y;
	union AtomFloat atom_top_x, atom_top_y;

	atom_pos_x.u = pos_x[idx]; atom_pos_y.u = pos_y[idx];
	atom_top_x.u = pos_x[idx - GRIDSIZE]; atom_top_y.u = pos_y[idx - GRIDSIZE];

	float top_diff_x = atom_top_x.f - atom_pos_x.f, top_diff_y = atom_top_y.f - atom_pos_y.f;
	float top_len = sqrt(top_diff_x * top_diff_x + top_diff_y * top_diff_y);				

	if (isfinite(top_len) && top_len > length_top[idx])
	{
		extra = top_len / length_top[idx] - 1;
		mov_x = extra * top_diff_x * 0.5f;
		mov_y = extra * top_diff_y * 0.5f;
		atom_pos_x.f += mov_x; 
		atom_pos_y.f += mov_y;
		atom_top_x.f -= mov_x; 
		atom_top_y.f -= mov_y;
	}
	pos_x[idx] = atom_pos_x.u;
	pos_y[idx] = atom_pos_y.u;
	pos_x[idx - GRIDSIZE] = atom_top_x.u;
	pos_y[idx - GRIDSIZE] = atom_top_y.u;
}

__kernel void pullb0(volatile __global int* pos_x, volatile __global unsigned int* pos_y, __global float* length_bot)
{
	int idx = get_global_id(0), row = 1, flag = GRIDSIZE - 1;
	if (idx < GRIDSIZE || idx >= GRIDS - GRIDSIZE || ((idx >> 8) & row) != 0 ) return;		
	if ((idx & flag) == 0 || (idx & flag) == flag) return;	
	
	float extra, mov_x, mov_y;
	union AtomFloat atom_pos_x, atom_pos_y;
	union AtomFloat atom_bot_x, atom_bot_y;

	atom_pos_x.u = pos_x[idx]; atom_pos_y.u = pos_y[idx];
	atom_bot_x.u = pos_x[idx + GRIDSIZE]; atom_bot_y.u = pos_y[idx + GRIDSIZE];

	float bot_diff_x = atom_bot_x.f - atom_pos_x.f, bot_diff_y = atom_bot_y.f - atom_pos_y.f;
	float bot_len = sqrt(bot_diff_x * bot_diff_x + bot_diff_y * bot_diff_y);				

	if (isfinite(bot_len) && bot_len > length_bot[idx])
	{
		extra = bot_len / length_bot[idx] - 1;
		mov_x = extra * bot_diff_x * 0.5f;
		mov_y = extra * bot_diff_y * 0.5f;
		atom_pos_x.f += mov_x; 
		atom_pos_y.f += mov_y;
		atom_bot_x.f -= mov_x; 
		atom_bot_y.f -= mov_y;
	}
	pos_x[idx] = atom_pos_x.u;
	pos_y[idx] = atom_pos_y.u;
	pos_x[idx + GRIDSIZE] = atom_bot_x.u;
	pos_y[idx + GRIDSIZE] = atom_bot_y.u;
}

__kernel void pullb1(volatile __global int* pos_x, volatile __global unsigned int* pos_y, __global float* length_bot)
{
	int idx = get_global_id(0), row = 1, flag = GRIDSIZE - 1;
	if (idx < GRIDSIZE || idx >= GRIDS - GRIDSIZE || ((idx >> 8) & row) != 1) return;		
	if ((idx & flag) == 0 || (idx & flag) == flag) return;	
	
	float extra, mov_x, mov_y;
	union AtomFloat atom_pos_x, atom_pos_y;
	union AtomFloat atom_bot_x, atom_bot_y;

	atom_pos_x.u = pos_x[idx]; atom_pos_y.u = pos_y[idx];
	atom_bot_x.u = pos_x[idx + GRIDSIZE]; atom_bot_y.u = pos_y[idx + GRIDSIZE];

	float bot_diff_x = atom_bot_x.f - atom_pos_x.f, bot_diff_y = atom_bot_y.f - atom_pos_y.f;
	float bot_len = sqrt(bot_diff_x * bot_diff_x + bot_diff_y * bot_diff_y);				

	if (isfinite(bot_len) && bot_len > length_bot[idx])
	{
		extra = bot_len / length_bot[idx] - 1;
		mov_x = extra * bot_diff_x * 0.5f;
		mov_y = extra * bot_diff_y * 0.5f;
		atom_pos_x.f += mov_x; 
		atom_pos_y.f += mov_y;
		atom_bot_x.f -= mov_x; 
		atom_bot_y.f -= mov_y;
	}
	pos_x[idx] = atom_pos_x.u;
	pos_y[idx] = atom_pos_y.u;
	pos_x[idx + GRIDSIZE] = atom_bot_x.u;
	pos_y[idx + GRIDSIZE] = atom_bot_y.u;
}