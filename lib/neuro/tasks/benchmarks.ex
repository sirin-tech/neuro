defmodule Mix.Tasks.Neuro.Benchmark do
  alias Neuro.Layers.Convolution
  alias Cuda.Graph
  alias Cuda.Graph.Factory
  alias Cuda.Compiler.Unit
  import Neuro.Test.CudaHelpers

  def run(_) do
    opts = [size: {4, 4}, kernel_size: {2, 2, 2}]
    graph = Factory.new(%Graph{}, :conv, Convolution, opts)
    l = Logger.level()
    Logger.configure(level: :error)
    {:ok, %{nodes: [cg]} = graph} = Unit.compile(graph, context())
    Logger.configure(level: l)

    i = [0.1, 0.2, 0.3, 0.4,
         0.5, 0.6, 0.7, 0.8,
         1.0, 0.1, 0.2, 0.3,
         0.4, 0.5, 0.6, 0.7]
    w = [[1.0, 2.0,
          3.0, 4.0],
         [5.0, 13.0,
          7.0, 8.0]]
    i = i |> Enum.reduce(<<>>, & &2 <> <<&1::float-little-32>>)
    w = w |> List.flatten |> Enum.reduce(<<>>, & &2 <> <<&1::float-little-32>>)

    o_size = cg.assigns.pin_size - byte_size(i)
    pins = i <> <<0::unit(8)-size(o_size)>>

    {:ok, cuda}   = Cuda.start_link()
    {:ok, mw}     = Cuda.memory_load(cuda, w)
    {:ok, mpins}  = Cuda.memory_load(cuda, pins)
    {:ok, module} = Cuda.module_load(cuda, cg.assigns.cubin)

    batch = [{"conv__conv", {3, 3, 2}, {1, 1, 1}, [mpins, mw]}]

    start = System.monotonic_time(:microseconds)
    1..10_000 |> Enum.each(fn _ ->
      :ok = Cuda.stream(cuda, module, batch)
      {:ok, o} = Cuda.memory_read(cuda, mpins)
    end)
    IO.inspect(System.monotonic_time(:microseconds) - start)
  end
end
