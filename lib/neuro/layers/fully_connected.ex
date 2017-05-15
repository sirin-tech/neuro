defmodule Neuro.Layers.FullyConnected do
  use Cuda.Graph
  alias Neuro.Nodes
  alias Neuro.Nodes.Base
  import Base, only: [input_type: 1, output_type: 1]

  def __assigns__(opts, _env) do
    %{vars: vars(opts |> Enum.into(%{}))}
  end

  def __pins__(assigns) do
    i = input_type(assigns.vars)
    o = output_type(assigns.vars)
    [input(:input, i), output(:output, o)]
  end

  def __graph__(graph) do
    vars = graph.assigns.options
    graph = graph |> chain(:fully, Nodes.FullyConnected, vars)
    graph = case Map.get(vars, :softmax, false) do
      false   -> graph
      _softmax -> graph
    end
    graph |> close()
  end

  defp vars(opts) do
    Nodes.FullyConnected.vars(opts)
  end
end
