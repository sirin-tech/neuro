defmodule Neuro.Nodes.PaddingTest do
  use ExUnit.Case
  require Logger
  alias Neuro.Nodes.Padding
  alias Cuda.Graph.GPUNode
  alias Cuda.Graph.Factory
  import Neuro.Test.NodesHelpers

  describe "padding node" do
    test "simple padding" do
      opts = %{size: {3, 3}, padding_size: {1, 1}, padding: 1.0}
      node = Factory.new(%GPUNode{}, :node, Padding, opts)

      i = [0.1, 0.2, 0.3,
           0.5, 0.6, 0.7,
           1.0, 0.1, 0.2]

      o = run(node, %{input: i})

      # 4.4 = 0.1 * 1.0 + 0.2 * 2.0 + 0.5 * 3.0 + 0.6 * 4.0
      # 5.4 = 0.2 * 1.0 + 0.3 * 2.0 + 0.6 * 3.0 + 0.7 * 4.0
      # ...
      assert o.output == [
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 0.1, 0.2, 0.3, 1.0,
        1.0, 0.5, 0.6, 0.7, 1.0,
        1.0, 1.0, 0.1, 0.2, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
      ]
    end
  end
end
