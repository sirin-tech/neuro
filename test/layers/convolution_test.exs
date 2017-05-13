defmodule Neuro.Layers.ConvolutionTest do
  use ExUnit.Case
  require Logger
  alias Neuro.Layers.Convolution
  alias Cuda.Graph
  alias Cuda.Graph.Factory
  import Neuro.Test.NodesHelpers

  describe "convolution layer" do
    test "simple convolution" do
      opts = [size: {4, 4}, kernel_size: {2, 2, 2}]
      graph = Factory.new(%Graph{}, :conv, Convolution, opts)

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

    test "convolution with padding" do
      opts = [size: {2, 2}, kernel_size: {2, 2, 2}, padding: {1, 1}]
      graph = Factory.new(%Graph{}, :conv, Convolution, opts)

      i = [0.1, 0.2,
           0.5, 0.6]
      w = [[1.0, 2.0,
            3.0, 4.0],
           [5.0, 6.0,
            7.0, 8.0]]

      o = run(graph, %{input: i}, %{w: w})

      assert o.output == [
        0.4, 1.1, 0.6,
        2.2, 4.4, 2.0,
        1.0, 1.7, 0.6,

        0.8,  2.3, 1.4,
        4.6, 10.0, 5.2,
        3.0,  6.1, 3.0
      ]
    end

    test "convolution with padding and pooling" do
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

      o = run(graph, %{input: i}, %{w: w})

      # Without pooling:
      #
      # 0.4, 1.1, 1.8, 0.9,
      # 2.2, 4.4, 5.8, 2.7,
      # 5.0, 5.1, 3.3, 1.2,
      # 2.0, 1.2, 0.5, 0.2,
      #
      # 0.8,   2.3,  3.8, 2.1,
      # 4.6,  10.0, 12.4, 7.1,
      # 11.0, 13.9, 10.1, 5.4,
      # 6.0,   5.6,  1.7, 1.0

      assert o.output == [
        4.4, 5.8,
        5.1, 3.3,

        10.0, 13.4,
        13.9, 10.1
      ]
    end
  end
end
