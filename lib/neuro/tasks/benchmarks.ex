defmodule Mix.Tasks.Neuro.Benchmark do
  alias Neuro.Layers.Convolution
  alias Cuda.Graph
  alias Cuda.Graph.Factory
  alias Cuda.Compiler.Unit
  alias Cuda.Runner
  import Neuro.Test.CudaHelpers

  def run(_) do
    opts = [size: {3, 3},
            kernel_size: {2, 2, 2},
            padding: {1, 1},
            pooling: [pooling: {2, 2}, stride: {2, 2}]]
    graph = Factory.new(%Graph{}, :conv, Convolution, opts)

    i = [0.1, 0.2, 0.3,
         0.5, 0.6, 0.8,
         1.0, 0.1, 0.2]
    w = [[1.0, 2.0,
          3.0, 4.0],
         [5.0, 6.0,
          7.0, 8.0]]
    w = w |> List.flatten |> Enum.reduce(<<>>, & &2 <> <<&1::float-little-32>>)

    {:ok, graph} = Unit.compile(graph, context())

    {:ok, cuda} = Cuda.start_link()
    {:ok, mw}   = Cuda.memory_load(cuda, w)

    opts = [cuda: cuda, args: %{w: mw}]
    start = System.monotonic_time(:microseconds)
    1..10_000 |> Enum.each(fn _ ->
      Runner.run(graph, %{input: i}, opts)
    end)
    finish = System.monotonic_time(:microseconds)
    IO.inspect(finish - start)
  end
end
