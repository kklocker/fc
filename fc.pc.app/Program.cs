

using fc.Abstractions;
using fc.pc.app;

var numberOfNodes = 5;
var network = new Network(numberOfNodes, new SigmoidActivationFunction());

int iterations = 1000;
for (int i = 0; i < iterations; i++)
{
    network.Next();
    var totalEnergy = network.TotalEnergy;
    if (totalEnergy < 0.001)
    {
        Console.WriteLine($"Converged after {i} iterations with total energy: {totalEnergy}");
        break;
    }

    if (i % 10 == 0)
    {

        Console.WriteLine($"Iteration {i}, Total Energy: {totalEnergy}");
    }

}
