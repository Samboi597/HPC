#define BITMAP_ID 0x4D42

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <Windows.h>
#include "freeglut.h"

cudaError_t mipmapping(float* in, float* out, int size);
BITMAPINFOHEADER imageInfoHeader;
unsigned char* imageBuffer;
float colourBuffer[1179648] = { 0.0 };
float* outBuffer;
float* inBuffer;
unsigned char *LoadBitmapFile(char *filename, BITMAPINFOHEADER *bitmapInfoHeader);
void drawToFrame(int y, int size);
void render(void);

__global__ void blur(float* in, int n, int length)
{
	int index = threadIdx.x + blockIdx.x * n;
	float filterMask[9] = { 0.0 };
	float pixelsSample[27] = { 0.0 };

	//how much each neighbouring value will contribute to the new pixel
	filterMask[0] = 0.0625f;
	filterMask[1] = 0.125f;
	filterMask[2] = 0.0625f;
	filterMask[3] = 0.125f;
	filterMask[4] = 0.25f;
	filterMask[5] = 0.125f;
	filterMask[6] = 0.0625f;
	filterMask[7] = 0.125f;
	filterMask[8] = 0.0625f;
	
	if (index > length && index < (length * length) - length - 1)
	{
		//collect neighbouring pixel values
		pixelsSample[0] = in[(index - length - 1) * 3]; //r
		pixelsSample[1] = in[(index - length - 1) * 3 + 1]; //g
		pixelsSample[2] = in[(index - length - 1) * 3 + 2]; //b

		pixelsSample[3] = in[(index - length) * 3]; //r
		pixelsSample[4] = in[(index - length) * 3 + 1]; //g
		pixelsSample[5] = in[(index - length) * 3 + 2];

		pixelsSample[6] = in[(index - length + 1) * 3]; //r
		pixelsSample[7] = in[(index - length + 1) * 3 + 1]; //g
		pixelsSample[8] = in[(index - length + 1) * 3 + 2]; //b

		pixelsSample[9] = in[(index - 1) * 3]; //r
		pixelsSample[10] = in[(index - 1) * 3 + 1]; //g
		pixelsSample[11] = in[(index - 1) * 3 + 2]; //b

		pixelsSample[12] = in[(index) * 3]; //r
		pixelsSample[13] = in[(index) * 3 + 1]; //g
		pixelsSample[14] = in[(index) * 3 + 2]; //b

		pixelsSample[15] = in[(index + 1) * 3]; //r
		pixelsSample[16] = in[(index + 1) * 3 + 1]; //g
		pixelsSample[17] = in[(index + 1) * 3 + 2]; //b

		pixelsSample[18] = in[(index + length - 1) * 3]; //r
		pixelsSample[19] = in[(index + length - 1) * 3 + 1]; //g
		pixelsSample[20] = in[(index + length - 1) * 3 + 2]; //b

		pixelsSample[21] = in[(index + length) * 3]; //r
		pixelsSample[22] = in[(index + length) * 3 + 1]; //g
		pixelsSample[23] = in[(index + length) * 3 + 2]; //b

		pixelsSample[24] = in[(index + length + 1) * 3]; //r
		pixelsSample[25] = in[(index + length + 1) * 3 + 1]; //g
		pixelsSample[26] = in[(index + length + 1) * 3 + 2]; //b

		//apply the filter mask
		pixelsSample[0] *= filterMask[0]; //r
		pixelsSample[1] *= filterMask[0]; //g
		pixelsSample[2] *= filterMask[0]; //b

		pixelsSample[3] *= filterMask[1]; //r
		pixelsSample[4] *= filterMask[1]; //g
		pixelsSample[5] *= filterMask[1]; //b

		pixelsSample[6] *= filterMask[2]; //r
		pixelsSample[7] *= filterMask[2]; //g
		pixelsSample[8] *= filterMask[2]; //b

		pixelsSample[9] *= filterMask[3]; //r
		pixelsSample[10] *= filterMask[3]; //g
		pixelsSample[11] *= filterMask[3]; //b

		pixelsSample[12] *= filterMask[4]; //r
		pixelsSample[13] *= filterMask[4]; //g
		pixelsSample[14] *= filterMask[4]; //b

		pixelsSample[15] *= filterMask[5]; //r
		pixelsSample[16] *= filterMask[5]; //g
		pixelsSample[17] *= filterMask[5]; //b

		pixelsSample[18] *= filterMask[6]; //r
		pixelsSample[19] *= filterMask[6]; //g
		pixelsSample[20] *= filterMask[6]; //b

		pixelsSample[21] *= filterMask[7]; //r
		pixelsSample[22] *= filterMask[7]; //g
		pixelsSample[23] *= filterMask[7]; //b

		pixelsSample[24] *= filterMask[8]; //r
		pixelsSample[25] *= filterMask[8]; //g
		pixelsSample[26] *= filterMask[8]; //b

		//add them all up
		float rAvg = 0.0f, gAvg = 0.0f, bAvg = 0.0f;
		for (int i = 0; i < 27; i += 3)
		{
			rAvg += pixelsSample[i];
			gAvg += pixelsSample[i + 1];
			bAvg += pixelsSample[i + 2];
		}
		__syncthreads();

		in[(index) * 3] = rAvg;
		in[(index) * 3 + 1] = gAvg;
		in[(index) * 3 + 2] = bAvg;
	}
}

__global__ void filterKernel(float* in, float* out, int n, int length)
{
	int index = threadIdx.x + blockIdx.x * n;

	float floorVal = floor((float)index / (float)length);
	int topQuarter = ((int)floorVal * length * 4) + ((index - (int)floorVal * length) * 2);
	float rAvg = ((in[topQuarter * 3]) + (in[(topQuarter + 1) * 3]) + 
		(in[(topQuarter + length * 2) * 3]) + (in[(topQuarter + length * 2 + 1) * 3])) / 4.0f;
	float gAvg = ((in[(topQuarter * 3) + 1]) + (in[((topQuarter + 1) * 3) + 1]) + 
		(in[((topQuarter + length * 2) * 3) + 1]) + (in[((topQuarter + length * 2 + 1) * 3) + 1])) / 4.0f;
	float bAvg = ((in[(topQuarter * 3) + 2]) + (in[((topQuarter + 1) * 3) + 2]) + 
		(in[((topQuarter + length * 2) * 3) + 2]) + (in[((topQuarter + length * 2 + 1) * 3) + 2])) / 4.0f;
	__syncthreads();

	out[index * 3] = rAvg; //r component of new pixel
	out[index * 3 + 1] = gAvg; //g component of new pixel
	out[index * 3 + 2] = bAvg; //b component of new pixel

//	printf("Colour value for pixel %d - %f, %f, %f\n", index, out[index * 3], out[index * 3 + 1], out[index * 3 + 2]);
}

int main(int argc, char** argv)
{
//	glutInit(&argc, argv);
//	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
//	glutInitWindowPosition(100, 100);
//	glutInitWindowSize(768, 512);
//	glutCreateWindow("Mipmapping");

	imageBuffer = LoadBitmapFile("lena.bmp", &imageInfoHeader);
	if (imageBuffer == NULL)
	{
		printf("File failed to load\n");
		return 1;
	}

	int currentLength = 512, drawY = 0;
	cudaError_t cudaStatus;
	while (currentLength > 1)
	{
		inBuffer = (float*)malloc((currentLength * currentLength * 3) * sizeof(float)); //new incoming image
		memcpy(inBuffer, outBuffer, (currentLength * currentLength * 3) * sizeof(float)); //copy old output into new input
		free(outBuffer); //goodbye, old output

		currentLength /= 2;
		outBuffer = (float*)malloc((currentLength * currentLength * 3) * sizeof(float)); //new outgoing image
		cudaStatus = mipmapping(inBuffer, outBuffer, currentLength); //mipmapping
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "mipmapping failed!");
			return 1;
		}

		drawToFrame(drawY, currentLength);
		drawY += currentLength;
		free(inBuffer); //goodbye, old input
	}
	free(outBuffer);

	//cudaDeviceReset must be called before exiting in order for profiling and
	//tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	//downsampling is performed here
//	glutDisplayFunc(render);
//	glutMainLoop();

	free(imageBuffer);
    return 0;
}

cudaError_t mipmapping(float* in, float* out, int size)
{
	float *dev_in = 0;
	float *dev_out = 0;
	cudaError_t cudaStatus;

	//setup the GPU on which we want to operate
	cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
	
	//allocate memory on the GPU buffers
    cudaStatus = cudaMalloc((void**)&dev_in, (size * size * 12) * sizeof(float)); //in buffer
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
//	printf("Input size: %d\n", (size * size * 12));

    cudaStatus = cudaMalloc((void**)&dev_out, (size * size * 3) * sizeof(float)); //out buffer
    if (cudaStatus != cudaSuccess) 
	{
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
//	printf("Output size: %d\n", (size * size * 3));
	
	//copy input buffer onto GPU memory
	cudaStatus = cudaMemcpy(dev_in, in, (size * size * 12) * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//launch kernel with specified number of blocks and threads
	int numBlocks, numThreads;
	if ((size * size * 4) <= 1024) //can you fit the entire input image onto one block?
	{
		numBlocks = 1;
		numThreads = (size * size * 4);
	}
	else //you can't?
	{
		numBlocks = (size * size * 4) / 1024;
		numThreads = 1024;
	}

//	printf("%d blocks, %d threads\n", numBlocks, numThreads);
	blur <<<numBlocks, numThreads >>> (dev_in, numThreads, (size * 2));

	if ((size * size * 4) <= 1024) //can you fit the entire input image onto one block?
	{
		numBlocks = 1;
		numThreads = (size * size);
	}
	else //you can't?
	{
		numBlocks = (size * size) / 1024;
		numThreads = 1024;
	}

//	printf("%d blocks, %d threads\n", numBlocks, numThreads);
	filterKernel <<<numBlocks, numThreads>>>(dev_in, dev_out, numThreads, size);

	//check for kernel launching errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	//cudaDeviceSynchronize waits for the kernel to finish, and returns
    //any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
		goto Error;
	}

	//copy output buffer to host memory
	cudaStatus = cudaMemcpy(out, dev_out, (size * size * 3) * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	Error:
    cudaFree(dev_in);
    cudaFree(dev_out);
    
    return cudaStatus;
}

unsigned char *LoadBitmapFile(char *filename, BITMAPINFOHEADER *bitmapInfoHeader)
{
	FILE *filePtr;							// the file pointer
	BITMAPFILEHEADER	bitmapFileHeader;		// bitmap file header
	unsigned char		*bitmapImage;			// bitmap image data
	int					imageIdx = 0;		// image index counter

												// open filename in "read binary" mode
	filePtr = fopen(filename, "rb");
	if (filePtr == NULL)
		return NULL;

	// read the bitmap file header
	fread(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);

	// verify that this is a bitmap by checking for the universal bitmap id
	if (bitmapFileHeader.bfType != BITMAP_ID)
	{
		fclose(filePtr);
		return NULL;
	}

	// read the bitmap information header
	fread(bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, filePtr);

	// move file pointer to beginning of bitmap data
	fseek(filePtr, bitmapFileHeader.bfOffBits, SEEK_SET);

	// allocate enough memory for the bitmap image data
	int imageSize = bitmapFileHeader.bfSize - bitmapFileHeader.bfOffBits;
	bitmapImage = (unsigned char*)malloc(imageSize);
	
	// verify memory allocation
	if (!bitmapImage)
	{
		free(bitmapImage);
		fclose(filePtr);
		return NULL;
	}

	// read in the bitmap image data
	fread(bitmapImage, 1, imageSize, filePtr);

	// make sure bitmap image data was read
	if (bitmapImage == NULL)
	{
		fclose(filePtr);
		return NULL;
	}

	// add the colours to the colour buffer
	outBuffer = (float*)malloc((512 * 512 * 3) * sizeof(float));
	int n, m;
	for (int i = 0; i < imageSize; i += 3)
	{
		//this will go into the mipmapping process
		outBuffer[i] = (float)bitmapImage[i + 2] / (float)255.0f;
		outBuffer[i + 1] = (float)bitmapImage[i + 1] / (float)255.0f;
		outBuffer[i + 2] = (float)bitmapImage[i] / (float)255.0f;
		
		n = floor(i / 1536); //row number
		m = n * 2304 + (i % 1536); //pixel number relative to row
		//this will be rendered onscreen
		colourBuffer[m] = (float)bitmapImage[i + 2] / (float)255.0f;
		colourBuffer[m + 1] = (float)bitmapImage[i + 1] / (float)255.0f;
		colourBuffer[m + 2] = (float)bitmapImage[i] / (float)255.0f;
	}

	// close the file and return the bitmap image data
	fclose(filePtr);
	return bitmapImage;
}

void drawToFrame(int y, int size)
{
	int imageSize = size * size * 3;
	int startPoint = y * 2304 + 1536; //starting point of new texture
	int endPoint = (y + size - 1) * 2304 + (1536 + size * 3);

//	printf("Start - %d, End - %d\n", startPoint, endPoint);

	int count = startPoint, row = 0, m;
	while (count < endPoint)
	{
		for (int i = 0; i < (size * 3); i += 3)
		{
			m = count + i; //how many pixels in from the edge
			
			colourBuffer[m] = outBuffer[row * (size * 3) + i];
			colourBuffer[m + 1] = outBuffer[row * (size * 3) + i + 1];
			colourBuffer[m + 2] = outBuffer[row * (size * 3) + i + 2];
		}

		count += 2304;
		row++;
	}
}

void render(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDrawPixels(768, 512, GL_RGB, GL_FLOAT, colourBuffer);
	glutSwapBuffers();
}