defmodule Neuro.NetworkTest do
  use ExUnit.Case
  alias Neuro.Network
  alias Neuro.Layers

  defmodule SimpleNetwork do
    use Network

    input  {4, 4}
    output 5

    def graph(graph) do
      graph
      |> chain(:conv, Layers.Convolution, kernel_size: {2, 2})
      |> chain(:fc,   Layers.FullyConnected, out_size: 5)
      |> close()
    end
  end

  describe "Network" do
    test "simple network" do
      graph = Cuda.Graph.Factory.new(%Cuda.Graph{}, :n, SimpleNetwork, [])
      assert %Cuda.Graph{} = graph
      # IO.inspect(graph)
    end
  end
end
