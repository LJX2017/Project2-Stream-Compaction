#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

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

        __global__ void kernDownSweep(int N, int* odata, int d) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= N) {
                return;
            }
            if (k % (1 << (d + 1)) == 0) {
                int t = odata[k + (1 << d) - 1];
                //odata[k + (1 << d) - 1] = odata[k + (1 << (d + 1)) - 1];  // Set left child to this node¡¯s value
                odata[k + (1 << (d + 1)) - 1] += t;
            }
        }

        void scan(int n, int *odata, const int *idata) {
            
            int* dev_odata;
            int log2N = ilog2ceil(n);
            int N = (1 << log2N);
            int threadsPerBlock = 32;
            int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
            cudaMalloc(&dev_odata, N * sizeof(int));
            cudaMemset(dev_odata, 0, N * sizeof(int));
            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            //timer().startGpuTimer();
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
            //timer().endGpuTimer();
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
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
