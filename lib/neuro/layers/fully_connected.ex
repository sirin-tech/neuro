defmodule Neuro.Layers.FullyConnected do
  alias Neuro.Nodes
  alias Neuro.Nodes.Base
  use Base, proto: Cuda.Graph

  def __graph__(%{assigns: %{back_propagation: true}} = graph) do
    vars = graph.assigns.vars
    graph = case Map.get(vars, :softmax, false) do
      false    -> graph
      _softmax -> graph
    end
    graph = graph |> chain(:fully, Nodes.FullyConnected)
    graph |> close()
  end
  def __graph__(graph) do
    vars = graph.assigns.vars
    graph = graph |> chain(:fully, Nodes.FullyConnected)
    graph = case Map.get(vars, :softmax, false) do
      false    -> graph
      _softmax -> graph
    end
    graph |> close()
  end

  def __child_options__(:fully, _, %{assigns: %{options: opts}}), do: opts
  def __child_options__(_, _, _), do: []

  def vars(opts, env) do
    Nodes.FullyConnected.vars(opts, env)
  end
end
