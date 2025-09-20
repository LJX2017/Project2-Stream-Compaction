#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                //printf("%d %d\n", i, idata[i]);
                odata[i] = odata[i - 1] + idata[i - 1];
            }
            timer().endCpuTimer();
        }

        void scanNoTimer(int n, int* odata, const int* idata) {
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                //printf("%d %d\n", i, idata[i]);
                odata[i] = odata[i - 1] + idata[i - 1];
            }
        }
        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[count++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* temp = new int[n];
            int* sum = new int[n];
            for (int i = 0; i < n; i++) {
                temp[i] = (idata[i] != 0);
            }
            scanNoTimer(n, sum, temp);
            for (int i = 0; i < n; i++) {
                odata[sum[i]] = idata[i];
            }
            int count = sum[n - 1];
            delete[] temp;
            delete[] sum;
            timer().endCpuTimer();
            return count;
        }
    }
}
