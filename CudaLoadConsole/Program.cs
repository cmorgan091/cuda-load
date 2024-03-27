using System.Diagnostics;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace CudaLoadConsole
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("GPU Load Test");

            Console.Write("Creating 100MB matrix... ");

            var depth = 500_000;
            var dimensions = 50;

            var matrix = new float[depth, dimensions];

            var rng = new Random();
            for (var x = 0; x < matrix.GetLength(0); x++)
            for (var y = 0; y < matrix.GetLength(1); y++)
            {
                matrix[x, y] = rng.NextSingle();
            }

            var vector = Enumerable.Range(0, dimensions).Select(_ => rng.NextSingle()).ToArray();
            
            Console.WriteLine("Done");

            // grab gpu context
            var context = Context.CreateDefault();

            foreach (var device in context.Devices)
            {
                Console.WriteLine();
                Console.WriteLine(device);

                var accelerator = device.CreateAccelerator(context);

                // load the main matrix into memory
                var loadStopwatch = Stopwatch.StartNew();

                using var matrixBuffer = accelerator.Allocate2DDenseY<float>(matrix.GetExtent());
                matrixBuffer.View.CopyFromCPU(matrix);

                loadStopwatch.Stop();
                Console.WriteLine($"Load time: {loadStopwatch.ElapsedMilliseconds:N0}ms");

                // load a vector
                var deviceVector = accelerator.Allocate1D(vector);

                // create a array for output
                var deviceOutput = accelerator.Allocate1D<float>(depth);

                // precompile kernel
                var loadedKernel =
                    accelerator
                        .LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView2D<float, Stride2D.DenseY>,
                            ArrayView<float>>(EuclideanSquareKernel);

                var runStopwatch = Stopwatch.StartNew();

                // compile and start work on gpu
                loadedKernel(depth, deviceVector.View, matrixBuffer.View, deviceOutput.View);

                // sync and copy back to cpu
                accelerator.Synchronize();
                var output = deviceOutput.GetAsArray1D();

                runStopwatch.Stop();

                Console.WriteLine($"Run time: {runStopwatch.ElapsedMilliseconds:N0}ms");
                Console.WriteLine($"Last value = {output[^1]}");

            }
        }

        static void EuclideanSquareKernel(Index1D index, ArrayView<float> vector,
            ArrayView2D<float, Stride2D.DenseY> matrix, ArrayView<float> output)
        {
            float squaredDistance = 0;
            for (var i = 0; i < vector.Length; i++)
            {
                var diff = vector[i] - matrix[index.X, i];
                squaredDistance += diff * diff;
            }

            output[index] = squaredDistance;
        }
    }
}
