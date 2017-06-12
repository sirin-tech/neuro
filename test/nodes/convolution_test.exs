defmodule Neuro.Nodes.ConvolutionTest do
  use ExUnit.Case
  require Logger
  alias Neuro.Nodes.Convolution
  alias Cuda.Shared
  import Neuro.Test.NodesHelpers

  defmodule Inference do
    use Neuro.Layers.Base

    def __graph__(graph) do
      graph |> chain(:conv, Convolution, graph.assigns.options) |> close()
    end

    defdelegate vars(opts, env), to: Convolution
  end

  defmodule BackPropagation do
    use Neuro.Layers.Base

    def __graph__(graph) do
      graph
      |> add(:conv, Convolution, graph.assigns.options)
      |> link(:output, {:conv, :output})
      |> link(:inference, {:conv, :inference})
      |> close()
    end

    defdelegate vars(opts, env), to: Convolution
  end

  describe "convolution node" do
    setup ~w(disable_logging load_graph)a

    @tag graph: Inference
    @tag options: [size: {4, 4}, kernel_size: {2, 2, 2}]
    @tag shared: %{
      weights: %{network: [[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0]]},
      biases:  %{network: [0.0, 0.0]}
    }
    test "simple convolution", ctx do
      i = [0.1, 0.2, 0.3, 0.4,
           0.5, 0.6, 0.7, 0.8,
           1.0, 0.1, 0.2, 0.3,
           0.4, 0.5, 0.6, 0.7]

      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

      # 4.4 = 0.1 * 1.0 + 0.2 * 2.0 + 0.5 * 3.0 + 0.6 * 4.0
      # 5.4 = 0.2 * 1.0 + 0.3 * 2.0 + 0.6 * 3.0 + 0.7 * 4.0
      # ...
      assert o.output |> round!() == [
        [[4.4, 5.4, 6.4], [5.1, 3.1, 4.1], [4.4, 4.4, 5.4]],
        [[10.0, 12.6, 15.2], [13.9,  9.5, 12.1], [12.4, 10.0, 12.6]]
      ]
    end

    @tag graph: Inference
    @tag options: [size: {2, 2}, kernel_size: {2, 2}, padding: 1]
    @tag shared: %{
      weights: %{network: [1.0, 2.0, 3.0, 4.0]},
      biases:  %{network: [0.0]}
    }
    test "padding", ctx do
      i = [0.1, 0.2, 0.3, 0.4]

      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

      assert o.output |> round!() == [[0.4, 1.1, 0.6], [1.4, 3.0, 1.4], [0.6, 1.1, 0.4]]
    end

    @tag graph: Inference
    @tag options: [size: {4, 4}, kernel_size: {2, 2, 2}]
    @tag shared: %{
      weights: %{network: [[1.0, 2.0, 3.0, 4.0],
                           [-5.0, -6.0, -7.0, -8.0]]},
      biases:  %{network: [0.0, 0.0]}
    }
    test "relu activation", ctx do
      i = [0.1, 0.2, 0.3, 0.4,
           0.5, 0.6, 0.7, 0.8,
           1.0, 0.1, 0.2, 0.3,
           0.4, 0.5, 0.6, 0.7]

      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

      assert o.output |> round!() == [
        [[4.4, 5.4, 6.4], [5.1, 3.1, 4.1], [4.4, 4.4, 5.4]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
      ]
    end

    @tag graph: Inference
    @tag options: [size: {3, 3}, kernel_size: {2, 2}, training: true]
    @tag shared: %{
      weights: %{network: [[1.0, 2.0], [3.0, 4.0]]},
      biases:  %{network: [0.0]},
      states:  %{network: [0.0, 0.0, 0.0, 0.0]}
    }
    test "saves neuron states in training mode", ctx do
      i = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, _o} = Cuda.Worker.run(worker, %{input: i})
      {:ok, shared} = Shared.vars(ctx[:shared_pid])

      assert shared.states.network |> round!() == [3.7, 4.7, 6.7, 7.7]
    end

    @tag graph: Inference
    @tag options: [size: {3, 3}, kernel_size: {2, 2, 2}, padding: 1]
    @tag shared: %{
      weights: %{network: [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]},
      biases:  %{network: [0.0, 0.0]}
    }
    test "bigger convolution with padding", ctx do
      i = [0.1, 0.2, 0.3,
           0.5, 0.6, 0.8,
           1.0, 0.1, 0.2]

      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

      assert o.output |> round!() == [
        [[0.4, 1.1, 1.8, 0.9],
         [2.2, 4.4, 5.8, 2.7],
         [5.0, 5.1, 3.3, 1.4],
         [2.0, 1.2, 0.5, 0.2]],

        [[0.8,   2.3,  3.8, 2.1],
         [4.6,  10.0, 13.4, 7.1],
         [11.0, 13.9, 10.1, 5.4],
         [6.0,   5.6,  1.7, 1.0]]
      ]
    end

    @tag graph: BackPropagation
    @tag options: [size: {3, 3}, kernel_size: {2, 2}, back_propagation: true]
    @tag shared: %{
      weights: %{network: [1.0, 2.0, 3.0, 4.0]},
      biases:  %{network: [0.0]},
      states:  %{network: [5.0, 6.0, 7.0, 8.0]},
      dw:      %{network: [0.0, 0.0, 0.0, 0.0]},
      db:      %{network: [0.0]}
    }
    test "back propagation", ctx do
      i = [0.1, 0.2, 0.3, 0.4]
      inf = [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]

      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, o} = Cuda.Worker.run(worker, %{output: i, inference: inf})
      assert o.input |> round! == [[0.1, 0.4, 0.4],
                                   [0.6, 2.0, 1.6],
                                   [0.9, 2.4, 1.6]]

      {:ok, shared} = Shared.vars(ctx[:shared_pid])
      # it accumulates delta * activation for weight correction
      assert shared.dw.network |> round!(3) == [3.175, 3.425, 3.925, 4.175]
      # it accumulates delta for bias correction
      assert shared.db.network |> round!(2) == [0.25]

      {:ok, _o} = Cuda.Worker.run(worker, %{output: i, inference: inf})
      {:ok, shared} = Shared.vars(ctx[:shared_pid])
      # it accumulates delta * activation for weight correction
      assert shared.dw.network |> round!(2) == [6.35, 6.85, 7.85, 8.35]
      # it accumulates delta for bias correction
      assert shared.db.network |> round!() == [0.5]
    end

    @tag graph: BackPropagation
    @tag options: [size: {2, 2}, kernel_size: {2, 2}, padding: 1, back_propagation: true]
    @tag shared: %{
      weights: %{network: [1.0, 2.0,
                           3.0, 4.0]},
      biases:  %{network: [0.0]},
      states:  %{network: [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]},
      dw:      %{network: [0.0, 0.0, 0.0, 0.0]},
      db:      %{network: [0.0]}
    }
    test "back propagation with padding", ctx do
      i = [0.1, 0.2, 0.3,
           0.4, 0.5, 0.6,
           0.7, 0.8, 0.9]
      inf = [[10.0, 11.0],
             [13.0, 14.0]]

      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, o} = Cuda.Worker.run(worker, %{output: i, inference: inf})
      assert o.input |> round! == [[2.3, 3.3], [5.3, 6.3]]

      {:ok, shared} = Shared.vars(ctx[:shared_pid])
      # it accumulates delta * activation for weight correction
      assert shared.dw.network |> round!(2) == [8.65, 7.45, 5.05, 3.85]
      # it accumulates delta for bias correction
      assert shared.db.network |> round!(2) == [0.5]
    end
  end
end
