defmodule Neuro.Layers.ConvolutionTest do
  use ExUnit.Case
  require Logger
  alias Neuro.Layers.Convolution
  import Neuro.Test.NodesHelpers

  defmodule Wrapper do
    use Cuda.Graph

    def __graph__(graph) do
      graph |> chain(:conv, Convolution, graph.assigns.options) |> close()
    end

    def __assigns__(opts, env) do
      m = Convolution.__assigns__(opts, env)
      Map.drop(m, [:shared])
    end

    defdelegate __pins__(assings), to: Convolution
  end

  describe "convolution layer" do
    setup do
      {:ok, shared} = Cuda.Shared.start_link()
      log_level = Logger.level()
      Logger.configure(level: :error)
      on_exit(fn -> Logger.configure(level: log_level) end)
      [shared: shared]
    end

    test "simple convolution", ctx do
      i = [0.1, 0.2, 0.3, 0.4,
           0.5, 0.6, 0.7, 0.8,
           1.0, 0.1, 0.2, 0.3,
           0.4, 0.5, 0.6, 0.7]
      w = [[1.0, 2.0,
            3.0, 4.0],
           [5.0, 6.0,
            7.0, 8.0]]
      b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

      opts = [network: Wrapper,
              shared_pid: ctx[:shared],
              shared: %{weights: %{conv: w}, biases: %{conv: b}},
              network_options: [size: {4, 4}, kernel_size: {2, 2, 2}]]
      {:ok, worker} = Cuda.Worker.start_link(opts)
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

      # 4.4 = 0.1 * 1.0 + 0.2 * 2.0 + 0.5 * 3.0 + 0.6 * 4.0
      # 5.4 = 0.2 * 1.0 + 0.3 * 2.0 + 0.6 * 3.0 + 0.7 * 4.0
      # ...
      assert o.output |> round!() == [
        4.4, 5.4, 6.4,
        5.1, 3.1, 4.1,
        4.4, 4.4, 5.4,

        10.0, 12.6, 15.2,
        13.9,  9.5, 12.1,
        12.4, 10.0, 12.6
      ]
    end

    test "convolution with padding", ctx do
      i = [0.1, 0.2,
           0.5, 0.6]
      w = [[1.0, 2.0,
            3.0, 4.0],
           [5.0, 6.0,
            7.0, 8.0]]
      b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

      opts = [network: Wrapper,
              shared_pid: ctx[:shared],
              shared: %{weights: %{conv: w}, biases: %{conv: b}},
              network_options: [size: {2, 2}, kernel_size: {2, 2, 2}, padding: {1, 1}]]
      {:ok, worker} = Cuda.Worker.start_link(opts)
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

      assert o.output |> round!() == [
        0.4, 1.1, 0.6,
        2.2, 4.4, 2.0,
        1.0, 1.7, 0.6,

        0.8,  2.3, 1.4,
        4.6, 10.0, 5.2,
        3.0,  6.1, 3.0
      ]
    end

    test "convolution with padding and pooling", ctx do
      i = [0.1, 0.2, 0.3,
           0.5, 0.6, 0.8,
           1.0, 0.1, 0.2]
      w = [[1.0, 2.0,
            3.0, 4.0],
           [5.0, 6.0,
            7.0, 8.0]]
      b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

      opts = [network: Wrapper,
              shared_pid: ctx[:shared],
              shared: %{weights: %{conv: w}, biases: %{conv: b}},
              network_options: [
                size: {3, 3},
                kernel_size: {2, 2, 2},
                padding: {1, 1},
                pooling: [pooling: {2, 2}, stride: {2, 2}
              ]]]
      {:ok, worker} = Cuda.Worker.start_link(opts)
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

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

      assert o.output |> round!() == [
        4.4, 5.8,
        5.1, 3.3,

        10.0, 13.4,
        13.9, 10.1
      ]
    end
  end
end
