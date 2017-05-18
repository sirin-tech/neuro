defmodule Mix.Tasks.Neuro.Benchmark do
  alias Neuro.Layers.Convolution
  alias Neuro.Layers.FullyConnected
  alias Neuro.Network

  defmodule MNISTNetwork do
    use Network

    input  {28, 28}
    output 10

    def graph(graph) do
      graph
      |> chain(:conv1, Convolution, kernel_size: {5, 5, 20}, pooling: {2, 2})
      |> chain(:conv2, Convolution, kernel_size: {5, 5, 50}, pooling: {2, 2})
      |> chain(:fc1,   FullyConnected, out_size: 500)
      |> chain(:fc2,   FullyConnected, out_size: 10)
      |> close()
    end
  end

  defp weights(num) do
    for _ <- 1..num, do: 0.1
  end

  def run(_) do
    weights = %{
      conv1: weights(5 * 5 * 20),
      conv2: weights(5 * 5 * 50),
      fc1:   weights(50 * 4 * 4 * 500),
      fc2:   weights(500 * 10)
    }
    biases = %{
      conv1: weights(5 * 5 * 20),
      conv2: weights(5 * 5 * 50),
      fc1:   weights(500),
      fc2:   weights(10)
    }

    {:ok, _} = MNISTNetwork.start_link(weights: weights, biases: biases)
    i = weights(28 * 28)
    #i = weights(2 * 2)

    start = System.monotonic_time(:microseconds)
    1..200 |> Enum.each(fn _ ->
      MNISTNetwork.run(%{input: i})
    end)
    finish = System.monotonic_time(:microseconds)
    IO.inspect(finish - start)

    #opts = [size: {3, 3},
    #        kernel_size: {2, 2, 2},
    #        padding: {1, 1},
    #        pooling: [pooling: {2, 2}, stride: {2, 2}]]
    #graph = Factory.new(%Graph{}, :conv, Convolution, opts)

    #i = [0.1, 0.2, 0.3,
    #     0.5, 0.6, 0.8,
    #     1.0, 0.1, 0.2]
    #w = [[1.0, 2.0,
    #      3.0, 4.0],
    #     [5.0, 6.0,
    #      7.0, 8.0]]
    #w = w |> List.flatten |> Enum.reduce(<<>>, & &2 <> <<&1::float-little-32>>)

    #{:ok, graph} = Unit.compile(graph, context())

    #{:ok, cuda} = Cuda.start_link()
    #{:ok, mw}   = Cuda.memory_load(cuda, w)

    #opts = [cuda: cuda, args: %{w: mw}]
    #{:ok, graph} = Runner.load(graph, opts)
    #start = System.monotonic_time(:microseconds)
    #1..10_000 |> Enum.each(fn _ ->
    #  Runner.run(graph, %{input: i}, cuda: cuda)
    #end)
    #finish = System.monotonic_time(:microseconds)
    #IO.inspect(finish - start)
  end
end
