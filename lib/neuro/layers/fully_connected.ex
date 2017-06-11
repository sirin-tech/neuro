defmodule Neuro.Layers.FullyConnected do
  alias Neuro.Nodes
  use Neuro.Layers.Base

  def __graph__(%{assigns: %{back_propagation: true}} = graph) do
    #vars = graph.assigns.vars
    #graph = case Map.get(vars, :softmax, false) do
    #  false    -> graph
    #  _softmax -> graph
    #end
    #graph = graph |> chain(:fc_node, Nodes.FullyConnected)
    #graph |> close()
    graph
    |> add(:fc_node, Nodes.FullyConnected)
    |> link(:output, {:fc_node, :output})
    #|> link(:result, {:fc_node, :result})
    |> link(:inference, {:fc_node, :inference})
    |> close()
  end
  def __graph__(graph) do
    vars = graph.assigns.vars
    graph = graph |> chain(:fc_node, Nodes.FullyConnected)
    graph = case Map.get(vars, :softmax, false) do
      false    -> graph
      _softmax -> graph
    end
    graph |> close()
  end

  def __child_options__(id, module, %{assigns: %{options: opts}} = graph) do
    Keyword.merge(super(id, module, graph), child_options(id, opts))
  end

  defp child_options(:fc_node, opts), do: opts
  defp child_options(_, _), do: []

  def vars(opts, env) do
    Nodes.FullyConnected.vars(opts, env)
  end
end
