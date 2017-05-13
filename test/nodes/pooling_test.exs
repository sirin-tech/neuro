defmodule Neuro.Nodes.PoolingTest do
  use ExUnit.Case
  require Logger
  alias Neuro.Nodes.Pooling
  alias Cuda.Graph.GPUNode
  alias Cuda.Graph.Factory
  import Neuro.Test.NodesHelpers

  describe "pooling node" do
    test "pooling: 2x2, stride: 2x2" do
      opts = %{size: {4, 4}, pooling: {2, 2}, stride: {2, 2}}
      node = Factory.new(%GPUNode{}, :node, Pooling, opts)

      i = [0.1, 0.2, 0.3, 0.4,
           0.5, 0.6, 0.7, 0.8,
           1.0, 0.1, 0.2, 0.3,
           0.4, 0.5, 0.6, 0.7]

      o = run(node, %{input: i})

      assert o.output == [
        0.6, 0.8,
        1.0, 0.7
      ]
    end

    test "pooling: 2x2, stride: 1x1" do
      opts = %{size: {4, 4}, pooling: 2, stride: 1}
      node = Factory.new(%GPUNode{}, :node, Pooling, opts)

      i = [0.1, 0.2, 0.3, 0.4,
           0.5, 0.6, 0.7, 0.8,
           1.0, 0.1, 0.2, 0.3,
           0.4, 0.5, 0.6, 0.7]

      o = run(node, %{input: i})

      assert o.output == [
        0.6, 0.7, 0.8,
        1.0, 0.7, 0.8,
        1.0, 0.6, 0.7
      ]
    end
  end
end
