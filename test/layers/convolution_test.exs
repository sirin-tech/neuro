defmodule Neuro.Layers.ConvolutionTest do
  use ExUnit.Case
  require Logger
  alias Neuro.Layers.Convolution
  alias Cuda.Graph
  alias Cuda.Graph.Factory
  alias Cuda.Compiler.Unit
  import Neuro.Test.CudaHelpers

  describe "convolution layer" do
    test "simple convolution" do
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
      IO.inspect(for <<x::float-little-32 <- pins>>, do: x)

      {:ok, cuda}   = Cuda.start_link()
      {:ok, mw}     = Cuda.memory_load(cuda, w)
      {:ok, mpins}  = Cuda.memory_load(cuda, pins)
      {:ok, module} = Cuda.module_load(cuda, cg.assigns.cubin)

      batch = [{"conv__conv", {3, 3, 2}, {1, 1, 1}, [mpins, mw]}]

      :ok = Cuda.stream(cuda, module, batch)
      {:ok, o} = Cuda.memory_read(cuda, mpins)

      offset = graph.assigns.pin_offsets.output
      <<_::unit(8)-size(offset), o::binary-size(72), _::binary>> = o

      o = for <<x::float-little-32 <- o>>, do: x
      o = Enum.map(o, & Float.round(&1, 1))

      # 4.4 = 0.1 * 1.0 + 0.2 * 2.0 + 0.5 * 3.0 + 0.6 * 4.0
      # 5.4 = 0.2 * 1.0 + 0.3 * 2.0 + 0.6 * 3.0 + 0.7 * 4.0
      # ...
      assert o == [
        4.4, 5.4, 6.4,
        5.1, 3.1, 4.1,
        4.4, 4.4, 5.4,

        10.0, 12.6, 15.2,
        13.9,  9.5, 12.1,
        12.4, 10.0, 12.6
      ]
    end
  end
end
