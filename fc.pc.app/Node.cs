using System.Text;

using fc.Abstractions;

namespace fc.pc.app;

public class Node : INode
{
    public double Activation { get; set; }

    public double Error { get; set; }

    public bool IsLocked { get; set; } = false;
}

public class Matrix2D<T> where T : struct, IComparable<T>
{
    private readonly T[][] _data;

    public int Rows { get; }
    public int Columns { get; }

    public Matrix2D(int rows, int? columns = null)
    {
        Rows = rows;
        Columns = columns ?? rows;
        _data = new T[rows][];
        for (int i = 0; i < rows; i++)
        {
            _data[i] = new T[Columns];
        }
    }

    public T this[int row, int column]
    {
        get => _data[row][column];
        set => _data[row][column] = value;
    }

    public override string ToString()
    {
        var sb = new StringBuilder();

        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Columns; j++)
            {
                sb.Append(_data[i][j] + " ");

            }
            sb.AppendLine();
        }

        return sb.ToString();
    }
}


public static class Matrix2DExtensions
{
    public static Matrix2D<T> Apply<T>(this Matrix2D<T> matrix, Func<T, T> func) where T : struct, IComparable<T>
    {
        foreach (var i in Enumerable.Range(0, matrix.Rows))
            foreach (var j in Enumerable.Range(0, matrix.Columns))
                matrix[i, j] = func(matrix[i, j]);

        return matrix;
    }

    public static double Sum(this Matrix2D<double> matrix)
    {
        double sum = 0.0;
        for (var i = 0; i < matrix.Rows; i++)
            for (var j = 0; j < matrix.Columns; j++)
                sum += matrix[i, j];
        return sum;
    }

}