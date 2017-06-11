defmodule Neuro.Nodes.CorrectionTest do
  use ExUnit.Case
  require Logger
  alias Neuro.Nodes.Correction
  alias Cuda.Shared
  import Neuro.Test.NodesHelpers

  defmodule Graph do
    use Cuda.Graph

    def __graph__(graph) do
      graph
      |> chain(:correction, Correction, graph.assigns.options)
      |> link(:input, :output)
    end

    def __assigns__(_id, opts, _env) do
      %{
        options: opts,
        input: Keyword.get(opts, :input),
        shared: %{
          shared: %{
            weights: %{network: {:f32, 5}},
            biases:  %{network: {:f32, 2}},
            dw:      %{network: {:f32, 5}},
            db:      %{network: {:f32, 2}}
          }
        }
      }
    end

    def __pins__(%{input: input}) do
      [input(:input, input), output(:output, input)]
    end

    defdelegate vars(opts, env), to: Correction
  end

  describe "correction node" do
    setup ~w(disable_logging load_graph)a

    @tag graph: Graph
    @tag options: [input: {:f32, 3}, weights: 5, biases: 2]
    @tag shared: %{
      weights: %{network: [1.0, 2.0, 3.0, 4.0, 5.0]},
      biases:  %{network: [10.0, 20.0]},
      dw:      %{network: [30.0, 40.0, 50.0, 60.0, 70.0]},
      db:      %{network: [80.0, 90.0]}
    }
    test "corrects weights and biases", ctx do
      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, _o} = Cuda.Worker.run(worker, %{input: [0.0, 0.0, 0.0]}, %{speed: 0.5})
      {:ok, shared} = Shared.vars(ctx[:shared_pid])
      assert shared.weights.network |> round!() == [16.0, 22.0, 28.0, 34.0, 40.0]
      assert shared.biases.network |> round!()  == [50.0, 65.0]
    end
  end
end
