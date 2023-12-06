#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

__global__ void multiplyKernel_rowwise(int* a, int* b, int* c, int wa, int wb) {
	int ridA = threadIdx.x;
	int sum;

	for (int cidB = 0; cidB < wb; cidB++) {
		sum = 0;
		for (int k = 0; k < wa; k++) {
			sum += (a[ridA * wa + k] * b[k * wb + cidB]);
		}
		c[ridA * wb + cidB] = sum;
	}
}

__global__ void multiplyKernel_columnwise(int* a, int* b, int* c, int ha, int wa) {
	int cidB = threadIdx.x;
	int wb = blockDim.x;
	int sum, k;
	for (int ridA = 0; ridA < ha; ridA++) {
		sum = 0;
		for (k = 0; k < wa; k++)
			sum += (a[ridA * wa + k] * b[k * wb + cidB]);
		c[ridA * wb + cidB] = sum;
	}
}

__global__ void multiplyKernel_elementwise(int* a, int* b, int* c, int wa) {
	int ridA = threadIdx.y;
	int cidB = threadIdx.x;
	int wb = blockDim.x;
	int sum = 0, k;

	for (k = 0; k < wa; k++)
		sum += (a[ridA * wa + k] * b[k * wb + cidB]);
	
	c[ridA * wb + cidB] = sum;
}
int main() {
	int* a, * b, * c;
	int* da, * db, * dc;
	int rowsA, colsA, rowsB, colsB;

	printf("Enter rows and columns of matrix A:");
	scanf("%d %d", &rowsA, &colsA);

	printf("Enter rows and columns of matrix B:");
	scanf("%d %d", &rowsB, &colsB);

	if (colsA != rowsB) {
		printf("Matrix multiplication not possible");
		return 1;
	}

	int sizeA = sizeof(int) * rowsA * colsA;
	int sizeB = sizeof(int) * rowsB * colsB;
	int sizeC = sizeof(int) * rowsA * colsB;

	a = (int*)malloc(sizeA);
	b = (int*)malloc(sizeB);
	c = (int*)malloc(sizeC);

	printf("Enter the elements of matrix A: ");
	for (int i = 0; i < rowsA * colsA; i++)
		scanf("%d", &a[i]);

	printf("Enter the elements of matrix B: ");
	for (int i = 0; i < rowsB * colsB; i++)
		scanf("%d", &b[i]);

	cudaMalloc((void**)&da, sizeA);
	cudaMalloc((void**)&db, sizeB);
	cudaMalloc((void**)&dc, sizeC);

	cudaMemcpy(da, a, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, sizeB, cudaMemcpyHostToDevice);

	//multiplyKernel_rowwise << <1, rowsA >> > (da, db, dc, colsA, colsB);
	//multiplyKernel_columnwise << < 1, colsB >> > (da, db, dc, rowsA, colsA);
	multiplyKernel_elementwise << <dim3(1, 1), dim3(colsB, rowsA) >> > (da, db, dc, colsA);
	cudaMemcpy(c, dc, sizeC, cudaMemcpyDeviceToHost);
	
	printf("resultant matrix: \n");
	for (int i = 0; i < rowsA; i++) {
		for (int j = 0; j < colsB; j++)
			printf("%d\t", c[i * colsB + j]);
		printf("\n");
	}

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	
	return 0;
}