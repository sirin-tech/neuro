defmodule Neuro.Nodes.FullyConnectedTest do
  use ExUnit.Case
  require Logger
  alias Neuro.Nodes.FullyConnected
  alias Cuda.Graph.GPUNode
  alias Cuda.Graph.Factory
  import Neuro.Test.NodesHelpers

  describe "fully connected node" do
    test "simple fully connected" do
      opts = %{size: 8, out_size: 2}
      graph = Factory.new(%GPUNode{}, :fully, FullyConnected, opts)

      i = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
      b = [1.0, 0.0]
      w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
           1.5, 0.7, 6.4, 3.2, 2.1, 1.1, 0.1, 9.0]

      o = run(graph, %{input: i}, %{w: w, b: b})

      assert o.output == [21.4, 12.5]
    end

    test "relu activation" do
      opts = %{size: 8, out_size: 2}
      graph = Factory.new(%GPUNode{}, :fully, FullyConnected, opts)

      i = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
      b = [0.0, 0.0]
      w = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0,
           1.5, 0.7, 6.4, 3.2, 2.1, 1.1, 0.1, 9.0]

      o = run(graph, %{input: i}, %{w: w, b: b})

      assert o.output == [0.0, 12.5]
    end
  end
end
