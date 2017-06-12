defmodule Neuro.Layers.Convolution do
  alias Neuro.Nodes
  use Neuro.Layers.Base

  def __pins__(assigns) do
    i = input_type(assigns.vars, assigns.env)
    o = case assigns.vars do
      %{pooling_vars: pv} -> pv
      %{conv_vars: cv}    -> cv
    end
    o = output_type(o, assigns.env)
    group = case Map.get(assigns, :training) do
      true -> :activation
      _    -> nil
    end
    case Map.get(assigns, :back_propagation) do
      true -> [input(:output, o),
               output(:input, i),
               input(:inference, i, :activation)]
      _    -> [input(:input, i, group),
               output(:output, o, group)]
    end
  end

  def __graph__(%{assigns: %{back_propagation: true}} = graph) do
    vars = graph.assigns.vars
    {graph, next} = case vars.pooling do
      false -> {graph, :output}
      _     -> {graph |> chain(:pooling_node, Nodes.Pooling), {:pooling_node, :input}}
    end
    graph = graph
            |> add(:conv_node, Nodes.Convolution)
            |> link(next, {:conv_node, :output})
            #|> link(:result, {:conv_node, :result})
            |> link(:inference, {:conv_node, :inference})
    graph |> close()
  end
  def __graph__(graph) do
    vars = graph.assigns.vars
    graph = graph |> chain(:conv_node, Nodes.Convolution)
    graph = case vars.pooling do
      false -> graph
      _     -> graph |> chain(:pooling_node, Nodes.Pooling)
    end
    graph |> close()
  end

  def __child_options__(id, module, %{assigns: %{vars: vars}} = graph) do
    Keyword.merge(super(id, module, graph), child_options(id, vars))
  end

  defp child_options(:pooling_node, %{pooling: opts}), do: opts
  defp child_options(:conv_node,    %{conv: opts}), do: opts
  defp child_options(_, _), do: []

  def vars(opts, env) do
    {x, y, z}  = opts |> Keyword.get(:size) |> Neuro.Layers.Base.triple_size()
    opts
    |> Enum.into(%{})
    |> Map.merge(%{x: x, y: y, z: z})
    |> process_conv(env)
    |> process_pooling(env)
    |> Map.drop([:size])
  end

  defp process_conv(vars, env) do
    c = vars
        |> Map.drop([:pooling, :shared])
        |> Enum.into([])
        |> Keyword.merge(size: {vars.x, vars.y, vars.z})
    vars
    |> Map.put(:conv, c)
    |> Map.put(:conv_vars, Nodes.Convolution.vars(c, env))
  end

  defp process_pooling(%{pooling: p, conv_vars: cv} = vars, env) do
    with p when is_list(p) <- pooling(p) do
      p = Keyword.merge(p, size: {cv.ox, cv.oy, cv.oz})
      vars
      |> Map.put(:pooling, p)
      |> Map.put(:pooling_vars, Nodes.Pooling.vars(p, env))
    else
      _ -> %{vars | pooling: false}
    end
  end
  defp process_pooling(vars, _env) do
    Map.put(vars, :pooling, false)
  end

  defp pooling(nil), do: false
  defp pooling(n) when is_integer(n), do: [pooling: {n, n}, stride: {n, n}]
  defp pooling({_, _} = p), do: [pooling: p, stride: p]
  defp pooling(p) when is_list(p) do
    if Keyword.keyword?(p), do: p, else: false
  end
  defp pooling(_), do: false
end
