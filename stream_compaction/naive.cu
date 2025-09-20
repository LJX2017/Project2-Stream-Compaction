#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"


namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: 

        __global__ void kernNaiveScan(int N, int* odata, int *tempdata, int d) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= N) {
                return;
            }
            if (k >= (1 << (d - 1))) {
                tempdata[k] = odata[k - (1 << (d - 1))] + odata[k];
            }
            else {
                tempdata[k] = odata[k];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            int * dev_odata, * dev_temp;
            cudaMalloc(&dev_odata, n * sizeof(int));
            cudaMalloc(&dev_temp, n * sizeof(int));
            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int threadsPerBlock = 32;
            int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

            for (int d = 1; d <= ilog2ceil(n); d++) {
                kernNaiveScan << <blocksPerGrid, threadsPerBlock >> > (n, dev_odata, dev_temp, d);
                checkCUDAError("Naive went wrong at" + d);
                int* temp = dev_odata;
                dev_odata = dev_temp;
                dev_temp = temp;
            }
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_odata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_odata);
            cudaFree(dev_temp);
            timer().endGpuTimer();
        }
    }
}
