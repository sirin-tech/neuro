defmodule Neuro.Nodes.ConvolutionTest do
  use ExUnit.Case
  require Logger
  alias Neuro.Nodes.Convolution
  alias Cuda.Graph.GPUNode
  alias Cuda.Graph.Factory
  import Neuro.Test.NodesHelpers

  describe "convolution node" do
    test "simple convolution" do
      opts = %{size: {4, 4}, kernel_size: {2, 2, 2}}
      graph = Factory.new(%GPUNode{}, :conv, Convolution, opts)

      i = [0.1, 0.2, 0.3, 0.4,
           0.5, 0.6, 0.7, 0.8,
           1.0, 0.1, 0.2, 0.3,
           0.4, 0.5, 0.6, 0.7]
      w = [[1.0, 2.0,
            3.0, 4.0],
           [5.0, 6.0,
            7.0, 8.0]]

      o = run(graph, %{input: i}, %{w: w})

      # 4.4 = 0.1 * 1.0 + 0.2 * 2.0 + 0.5 * 3.0 + 0.6 * 4.0
      # 5.4 = 0.2 * 1.0 + 0.3 * 2.0 + 0.6 * 3.0 + 0.7 * 4.0
      # ...
      assert o.output == [
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
