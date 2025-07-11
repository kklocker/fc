﻿using fc.Abstractions;
using fc.pc.app;

//var numberOfNodes = 5;
//var network = new Network(numberOfNodes, new SigmoidActivationFunction());

//int iterations = 1000;
//for (int i = 0; i < iterations; i++)
//{
//    network.Next();
//    var totalEnergy = network.TotalEnergy;
//    if (totalEnergy < 0.001)
//    {
//        Console.WriteLine($"Converged after {i} iterations with total energy: {totalEnergy}");
//        break;
//    }

//    if (i % 10 == 0)
//    {

//        Console.WriteLine($"Iteration {i}, Total Energy: {totalEnergy}");
//    }
//}

// XOR problem:
var xorNetwork = new Network(6, new SigmoidActivationFunction());
xorNetwork.SetDataDimensions(2, 1);

// Train:
(double[] Input, double[] Output)[] trainingData =
[
    (Input: [0.0, 0.0], Output: [0.0]),
    (Input: [0.0, 1.0], Output: [1.0]),
    (Input: [1.0, 0.0], Output: [1.0]),
    (Input: [1.0, 1.0], Output: [0.0])
];


// Train

var epochs = 5;
for (int epoch = 0; epoch < epochs; epoch++)
{
    Console.WriteLine($"Epoch {epoch + 1}/{epochs}");
    int epochIterationCount = 0;
    foreach (var (input, output) in trainingData)
    {
        xorNetwork.SetAndLockInputActivations(input);
        xorNetwork.SetAndLockOutputActivations(output);

        for (int i = 0; i < 100; i++)
        {
            epochIterationCount++;
            xorNetwork.Next();
            var totalEnergy = xorNetwork.TotalEnergy;
            if (totalEnergy < 0.0001)
            {
                Console.WriteLine($"XOR Network converged after {i} iterations with total energy: {totalEnergy}");
                break;
            }
            if (i % 10 == 0)
            {

                Console.WriteLine($"Iteration {i}, Total Energy: {totalEnergy}");
            }
        }
    }

    Console.WriteLine($"Epoch {epoch + 1} completed in {epochIterationCount} iterations.");
}

// Test:
Console.WriteLine();
Console.WriteLine("Locking weights");

xorNetwork.LockWeights();
xorNetwork.UnlockOutputActivations();


foreach (var (input, _) in trainingData)
{
    xorNetwork.SetAndLockInputActivations(input);

    for (int i = 0; i < 1000; i++)
    {
        var previousTotalEnergy = xorNetwork.TotalEnergy;

        xorNetwork.Next();
        var totalEnergy = xorNetwork.TotalEnergy;

        if(Math.Abs(totalEnergy - previousTotalEnergy) < 0.0001)
        {
            Console.WriteLine($"Network seemed to reach a steady state after {i} iterations with total energy: {totalEnergy}");
            break;
        }

        if (i % 100 == 0)
        {
            Console.WriteLine($"Iteration {i}, Total Energy: {totalEnergy}");
        }
    }

    var result = xorNetwork.GetOutputActivations();
    Console.WriteLine($"Input: {string.Join(", ", input)} => Output: {string.Join(", ", result)}. Total energy: {xorNetwork.TotalEnergy}");
}