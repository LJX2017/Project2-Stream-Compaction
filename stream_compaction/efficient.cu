#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define threadsPerBlock 256

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

        __global__ void kernUpSweep(int N, int* odata, int d) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= N) {
                return;
            }
            if (k % (1 << (d + 1)) == 0) {
                odata[k + (1 << (d + 1)) - 1] += odata[k + (1 << d) - 1];
            }
        }

        __global__ void kernDownSweep(int N, int* x, int d) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= N) {
                return;
            }
            if (k % (1 << (d + 1)) == 0) {
                int t = x[k + (1 << d) - 1];
                x[k + (1 << d) - 1] = x[k + (1 << (d + 1)) - 1];
                x[k + (1 << (d + 1)) - 1] += t;
            }
        }

        void scan(int n, int *odata, const int *idata) {
            int* dev_odata;
            int log2N = ilog2ceil(n);
            int N = (1 << log2N);
            int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
            cudaMalloc(&dev_odata, N * sizeof(int));
            cudaMemset(dev_odata, 0, N * sizeof(int));
            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            // Up-Sweep
            for (int d = 0; d < log2N; d++) {
                kernUpSweep <<< blocksPerGrid, threadsPerBlock >>> (N, dev_odata, d);
                cudaDeviceSynchronize();
                checkCUDAError("kernUpSweep");
            }

            // Down-Sweep
            cudaMemset(dev_odata + (N - 1), 0, sizeof(int));
            for (int d = log2N - 1; d >= 0; d--) {
                kernDownSweep <<< blocksPerGrid, threadsPerBlock >>> (N, dev_odata, d);
                cudaDeviceSynchronize();
                checkCUDAError("kernDownSweep");
            }
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_odata);
        }


        void scanNoTiming(int n, int* odata, const int* idata) {
            int* dev_odata;
            int log2N = ilog2ceil(n);
            int N = (1 << log2N);
            int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
            cudaMalloc(&dev_odata, N * sizeof(int));
            cudaMemset(dev_odata, 0, N * sizeof(int));
            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            // Up-Sweep
            for (int d = 0; d < log2N; d++) {
                kernUpSweep << < blocksPerGrid, threadsPerBlock >> > (N, dev_odata, d);
                cudaDeviceSynchronize();
                checkCUDAError("kernUpSweep");
            }

            // Down-Sweep
            cudaMemset(dev_odata + (N - 1), 0, sizeof(int));
            for (int d = log2N - 1; d >= 0; d--) {
                kernDownSweep << < blocksPerGrid, threadsPerBlock >> > (N, dev_odata, d);
                cudaDeviceSynchronize();
                checkCUDAError("kernDownSweep");
            }
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_odata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
   //     __global__ void label_data(int N, int* idata, int* odata) {
   //         int k = blockIdx.x * blockDim.x + threadIdx.x;
   //         if (k >= N) {
   //             return;
   //         }
   //         odata[k] = idata[k] != 0;
   //     }
   //     __global__ void scatter(int N, int* idata, int* odata, int* sum_data) {
   //         int k = blockIdx.x * blockDim.x + threadIdx.x;
   //         if (k >= N) {
   //             return;
   //         }
   //         if (idata[k] != 0) {
   //             odata[sum_data[k]] = idata[k];
			//}
   //     }
        int compact(int n, int *odata, const int *idata) {

            int* dev_odata, *dev_idata, *dev_sumdata, *dev_bool;
            int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
            cudaMalloc(&dev_idata, n * sizeof(int));
            cudaMalloc(&dev_odata, n * sizeof(int));
            cudaMalloc(&dev_sumdata, n * sizeof(int));
            cudaMalloc(&dev_bool, n * sizeof(int));

            cudaMemset(dev_odata, 0, n * sizeof(int));
            cudaMemset(dev_sumdata, 0, n * sizeof(int));
            cudaMemset(dev_bool, 0, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();

            Common::kernMapToBoolean << < blocksPerGrid, threadsPerBlock >> > (n, dev_bool, dev_idata);
			scanNoTiming(n, dev_sumdata, dev_bool);

            int sum = 0;
			cudaMemcpy(&sum, dev_sumdata + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
			printf("sum: %d\n", sum);

            Common::kernScatter << < blocksPerGrid, threadsPerBlock >> > (n, dev_odata, dev_idata, dev_bool, dev_sumdata);
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            timer().endGpuTimer();
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_sumdata);
            cudaFree(dev_bool);
			return sum + (idata[n - 1] != 0);
        }
    }
}
