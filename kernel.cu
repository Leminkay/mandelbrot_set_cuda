#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <helper_functions.h>

#include <stdio.h>
#include <complex>
#include <tuple>
#include <chrono>
#include <FreeImage.h>

#define iter_max 500
#define smooth_color false

using namespace std;

typedef float2 Complex;

static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b);
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b);
static __device__ __host__ inline float ComplexAbs(Complex a);

static __host__ __device__ inline Complex func(Complex z, Complex c) {
	return ComplexAdd(ComplexMul(z, z), c);
}

// Convert a pixel coordinate to the complex domain
static __device__ __host__ inline Complex scale(int* scr, float* fr, Complex c) {
	Complex aux;
	aux.x = c.x / (double)(scr[1] - scr[0]) * (fr[1] - fr[0]) + fr[0];
	aux.y = c.y / (double)(scr[3] - scr[2]) * (fr[3] - fr[2]) + fr[2];
	return aux;
}

// Check if a point is in the set or escapes to infinity, return the number if iterations
static __device__ __host__ inline int escape(Complex c) {
	Complex z;
	z.x = 0; z.y = 0;
	int iter = 0;

	while (ComplexAbs(z) < 2.0 && iter < iter_max) {
		z = func(z, c);
		iter++;
	}
	return iter;
}
 
// Loop over each pixel from our image and check if the points associated with this pixel escape to infinity
static __global__ void get_number_iterations(int* scr, float* fract, int* colors) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	// kernel ?
	if (i < scr[3] && j < scr[1] &&
		i >= scr[2] && j >= scr[0])
	{
		int k = (j - scr[0]) + (scr[1] - scr[0]) * (i - scr[2]);
		Complex c;
		c.x = (float)j;
		c.y = (float)i;
		c = scale(scr, fract, c);
		colors[k] = escape(c);
		
	}
}


__device__ __host__ void get_rgb_piecewise_linear(int n, int* rgb) {
	int N = 256; // colors per element
	int N3 = N * N * N;
	// map n on the 0..1 interval (real numbers)
	double t = (double)n / (double)iter_max;
	// expand n on the 0 .. 256^3 interval (integers)
	n = (int)(t * (double)N3);

	rgb[2] = n / (N * N);
	int nn = n - rgb[2] * N * N;
	rgb[0] = nn / N;
	rgb[1] = nn - rgb[0] * N;

}


__device__ __host__ void get_rgb_smooth(int n, int* rgb) {
	// map n on the 0..1 interval
	double t = (double)n / (double)iter_max;

	// Use smooth polynomials for r, g, b
	rgb[0] = (int)(9 * (1 - t) * t * t * t * 255);
	rgb[1] = (int)(15 * (1 - t) * (1 - t) * t * t * 255);
	rgb[2] = (int)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);

}

static __global__ void set_plot_color(int* scr, int* colors, int* col) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < scr[3] && j < scr[1] &&
		i >= scr[2] && j >= scr[0])
	{
		int k = (j - scr[0]) + (scr[1] - scr[0]) * (i - scr[2]);

		int rgb[3];
		int n = colors[k];
		if (!smooth_color) {
			get_rgb_piecewise_linear(n, rgb);
		}
		else {
			get_rgb_smooth(n, rgb);
		}

		col[3 * k] = rgb[0];
		col[3 * k + 1] = rgb[1];
		col[3 * k + 2] = rgb[2];

	}

}

void plot(int* h_scr, int* d_scr, int* d_colors, const char* fname) {
	// active only for static linking
#ifdef FREEIMAGE_LIB
	FreeImage_Initialise();
#endif

	unsigned int width = h_scr[1] - h_scr[0], height = h_scr[3] - h_scr[2];
	FIBITMAP* bitmap = FreeImage_Allocate(width, height, 32); // RGBA

	// temp array to get colors of pixel
	int* h_col;
	checkCudaErrors(cudaMallocHost((void**)&h_col, width * height * 3 * sizeof(int)));
	int* d_col;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**> (&d_col), width * height * 3 * sizeof(int)));

	//call kernel set_plot_color
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(((h_scr[3] - h_scr[2]) + threadsPerBlock.x - 1) / threadsPerBlock.x,
		((h_scr[1] - h_scr[0]) + threadsPerBlock.y - 1) / threadsPerBlock.y);

	set_plot_color <<< numBlocks, threadsPerBlock >>> (d_scr, d_colors, d_col);

	checkCudaErrors(cudaMemcpy(h_col, d_col, width * height * 3 * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = h_scr[2]; i < h_scr[3]; ++i) {
		for (int j = h_scr[0]; j < h_scr[1]; ++j) {
			int k = (j - h_scr[0]) + (h_scr[1] - h_scr[0]) * (i - h_scr[2]);

			RGBQUAD col;
			col.rgbRed = h_col[3 * k];
			col.rgbGreen = h_col[3 * k + 1];
			col.rgbBlue = h_col[3 * k + 2];
			col.rgbReserved = 255;

			FreeImage_SetPixelColor(bitmap, j, i, &col);
		}
	}

	// Save the image in a PNG file
	FreeImage_Save(FIF_PNG, bitmap, fname, PNG_DEFAULT);

	// Clean bitmap;
	FreeImage_Unload(bitmap);

	// active only for static linking
#ifdef FREEIMAGE_LIB
	FreeImage_DeInitialise();
#endif

	checkCudaErrors(cudaFree(d_col));
	checkCudaErrors(cudaFreeHost(h_col));
}


void fractal(int* h_scr, float *h_fract, int* d_scr, float* d_fract, int* d_colors, const char* fname) {
	auto start = chrono::steady_clock::now();

	//request as a kernel
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(((h_scr[3] - h_scr[2]) + threadsPerBlock.x - 1) / threadsPerBlock.x,
		((h_scr[1] - h_scr[0]) + threadsPerBlock.y - 1) / threadsPerBlock.y);

	get_number_iterations << < numBlocks, threadsPerBlock >> >(d_scr, d_fract, d_colors);
	checkCudaErrors(cudaDeviceSynchronize());

	auto end = chrono::steady_clock::now();
	cout << "Time to generate " << fname << " = " << chrono::duration <double, std::milli>(end - start).count() << " [ms]" << endl;
	
	// Save (show) the result as an image
	plot(h_scr, d_scr, d_colors, fname);
}


void mandelbrot() {

	int* h_scr;
	float* h_fract;
	checkCudaErrors(cudaMallocHost((void**)&h_scr, 4 * sizeof(int)));
	checkCudaErrors(cudaMallocHost((void**)&h_fract, 4 * sizeof(float)));

	int* d_scr;
	float* d_fract;
	checkCudaErrors(cudaMalloc(&d_scr, 4 * sizeof(int)));
	checkCudaErrors(cudaMalloc(&d_fract, 4 * sizeof(float)));
	
	h_scr[0] = 0; h_scr[1] = 1200; h_scr[2] = 0; h_scr[3] = 1200;
	h_fract[0] = -2.2; h_fract[1] = 1.2; h_fract[2] = -1.7; h_fract[3] = 1.7;

	int* d_colors;
	checkCudaErrors(cudaMalloc(&d_colors, (h_scr[1] - h_scr[0]) * (h_scr[3] - h_scr[2]) * sizeof(int)));

	checkCudaErrors(cudaMemcpy(d_scr, h_scr, 4 * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_fract, h_fract, 4 * sizeof(int), cudaMemcpyHostToDevice));

	const char* fname = "mandelbrot.png";

	fractal(h_scr, h_fract, d_scr, d_fract, d_colors, fname);

	checkCudaErrors(cudaFree(d_scr));
	checkCudaErrors(cudaFree(d_fract));
	checkCudaErrors(cudaFree(d_colors));
	checkCudaErrors(cudaFreeHost(h_scr));
	checkCudaErrors(cudaFreeHost(h_fract));
}


int main()
{
	mandelbrot();
}

static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
	Complex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
	Complex c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	return c;
}
static __device__ __host__ inline float ComplexAbs(Complex a) {
	return sqrt(a.x * a.x + a.y * a.y);
}

