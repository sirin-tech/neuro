defmodule Neuro.Nodes.ErrorTest do
  use ExUnit.Case
  require Logger
  alias Neuro.Nodes.Error
  alias Cuda.Graph.Pin
  import Neuro.Test.NodesHelpers

  defmodule Wrapper do
    use Cuda.Graph

    def __graph__(graph) do
      graph
      |> add(:error, Error, graph.assigns.options)
      |> link(:input, {:error, :input})
      |> link(:reply, {:error, :reply})
      |> link({:error, :output}, :output)
      |> link({:error, :error}, :error)
    end

    defdelegate __assigns__(id, opts, env), to: Error
    defdelegate __pins__(assigns), to: Error
  end

  describe "error node" do
    setup ~w(disable_logging load_graph)a

    @tag graph: Wrapper
    @tag options: [size: 3, input: %Pin{data_type: {:f32, 3}}]
    test "error calculation", ctx do
      i = [0.1, 0.2, 0.3]
      r = [0.0, 1.0, 0.2]

      {:ok, worker} = Cuda.Worker.start_link(ctx[:worker_options])
      {:ok, o} = Cuda.Worker.run(worker, %{input: i, reply: r})

      assert o.output |> round!(2) == [-0.23, -0.13, -0.03]
      assert o.error |> round!(2) == 0.33
    end
  end
end
