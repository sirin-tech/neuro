defmodule Neuro.Nodes.FullyConnectedTest do
  use ExUnit.Case
  require Logger
  alias Neuro.Nodes.FullyConnected
  alias Cuda.Shared
  import Neuro.Test.NodesHelpers

  defmodule Inference do
    use Neuro.Layers.Base

    def __graph__(graph) do
      graph |> chain(:fc, FullyConnected, graph.assigns.options) |> close()
    end

    defdelegate vars(opts, env), to: FullyConnected
  end

  defmodule BackPropagation do
    use Neuro.Layers.Base

    def __graph__(graph) do
      graph
      |> add(:fc, FullyConnected, graph.assigns.options)
      |> link(:output, {:fc, :output})
      |> link(:inference, {:fc, :inference})
      |> close()
    end

    defdelegate vars(opts, env), to: FullyConnected
  end

  describe "fully connected node" do
    setup ~w(disable_logging load_graph)a

    @tag graph: Inference
    @tag options: [size: 8, out_size: 2]
    @tag shared: %{
      weights: %{network: [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                            1.5, 0.7, 6.4, 3.2, 2.1, 1.1, 0.1, 9.0]]},
      biases:  %{network: [1.0, 0.0]}
    }
    test "simple fully connected", ctx do
      i = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

      assert o.output |> round!() == [21.4, 12.5]
    end

    @tag graph: Inference
    @tag options: [size: 8, out_size: 2]
    @tag shared: %{
      weights: %{network: [[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0,
                             1.5,  0.7,  6.4,  3.2,  2.1,  1.1,  0.1,  9.0]]},
      biases:  %{network: [0.0, 0.0]}
    }
    test "relu activation", ctx do
      i = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, o} = Cuda.Worker.run(worker, %{input: i})

      assert o.output |> round!() == [0.0, 12.5]
    end

    @tag graph: Inference
    @tag options: [size: 8, out_size: 2, training: true]
    @tag shared: %{
      weights: %{network: [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                            1.5, 0.7, 6.4, 3.2, 2.1, 1.1, 0.1, 9.0]]},
      biases:  %{network: [1.0, 0.0]},
      states:  %{network: [0.0, 0.0]}
    }
    test "saves neuron states in training mode", ctx do
      i = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, _o} = Cuda.Worker.run(worker, %{input: i})
      {:ok, shared} = Shared.vars(ctx[:shared_pid])

      assert shared.states.network |> round!() == [21.4, 12.5]
    end

    @tag graph: BackPropagation
    @tag options: [size: 4, out_size: 3, back_propagation: true]
    @tag shared: %{
      weights: %{network: [[1.0, 2.0, 3.0, 4.0],
                           [0.1, 0.2, 0.3, 0.4],
                           [0.5, 0.6, 0.7, 0.8]]},
      biases:  %{network: [0.0, 0.0, 0.0]},
      states:  %{network: [30.0, 0.0, 40.0]},
      dw:      %{network: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
      db:      %{network: [0.0, 0.0, 0.0]}
    }
    test "back propagation", ctx do
      loss = [10.0, 15.0, 20.0]
      inf  = [11.0, 12.0, 13.0, 14.0]

      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, o} = Cuda.Worker.run(worker, %{output: loss, inference: inf})
      # it calculates deltas (errors)
      assert o.input |> round! == [20.0, 32.0, 44.0, 56.0]

      {:ok, shared} = Shared.vars(ctx[:shared_pid])
      # it accumulates delta * activation for weight correction
      assert shared.dw.network |> round!() == [110.0, 120.0, 130.0, 140.0,
                                               165.0, 180.0, 195.0, 210.0,
                                               220.0, 240.0, 260.0, 280.0]
      # it accumulates delta for bias correction
      assert shared.db.network |> round!() == [10.0, 15.0, 20.0]

      {:ok, _o} = Cuda.Worker.run(worker, %{output: loss, inference: inf})
      {:ok, shared} = Shared.vars(ctx[:shared_pid])
      # it accumulates delta * activation for weight correction
      assert shared.dw.network |> round!() == [220.0, 240.0, 260.0, 280.0,
                                               330.0, 360.0, 390.0, 420.0,
                                               440.0, 480.0, 520.0, 560.0]
      # it accumulates delta for bias correction
      assert shared.db.network |> round!() == [20.0, 30.0, 40.0]
    end
  end
end
