defmodule Neuro.Layers.Convolution do
  alias Neuro.Nodes
  alias Neuro.Nodes.Base
  use Base, proto: Cuda.Graph

  def __pins__(assigns) do
    i = input_type(assigns.vars)
    o = case assigns.vars do
      %{pooling_vars: pv} -> pv
      %{conv_vars: cv}    -> cv
    end
    o = output_type(o)
    [input(:input, i), output(:output, o)]
  end

  def __graph__(graph) do
    vars = graph.assigns.vars
    graph = case vars.padding do
      false -> graph
      _     -> graph |> chain(:padding, Nodes.Padding)
    end
    graph = graph |> chain(:conv, Nodes.Convolution)
    graph = case vars.pooling do
      false -> graph
      _     -> graph |> chain(:pooling, Nodes.Pooling)
    end
    graph |> close()
  end

  def __child_options__(:pooling, _, %{assigns: %{vars: %{pooling: opts}}}), do: opts
  def __child_options__(:padding, _, %{assigns: %{vars: %{padding: opts}}}), do: opts
  def __child_options__(:conv,    _, %{assigns: %{vars: %{conv: opts}}}), do: opts
  def __child_options__(_, _, _), do: []

  def vars(opts, env) do
    {x, y, z}  = opts |> Keyword.get(:size) |> Base.triple_size()
    vars = opts
           |> Enum.into(%{})
           |> Map.merge(%{x: x, y: y, z: z})
           |> process_padding(env)
           |> process_conv(env)
           |> process_pooling(env)
           |> Map.drop([:size])
    conv = Map.get(vars, :conv_vars)
    Map.merge(vars, Map.take(conv, ~w(weights neurons)a))
  end

  defp process_padding(%{padding: p} = vars, env) do
    with p when is_list(p) <- padding(p) do
      p = Keyword.merge(p, size: {vars.x, vars.y, vars.z},
                           float_size: vars.float_size)
      vars
      |> Map.put(:padding, p)
      |> Map.put(:padding_vars, Nodes.Padding.vars(p, env))
    else
      _ -> %{vars | padding: false}
    end
  end
  defp process_padding(vars, _env) do
    Map.put(vars, :padding, false)
  end

  defp process_conv(%{padding_vars: pv} = vars, env) do
    c = vars
        |> Map.drop([:padding, :padding_vars, :pooling])
        |> Enum.into([])
        |> Keyword.merge(size: {pv.ox, pv.oy, pv.oz},
                         float_size: vars.float_size)
    vars
    |> Map.put(:conv, c)
    |> Map.put(:conv_vars, Nodes.Convolution.vars(c, env))
  end
  defp process_conv(vars, env) do
    c = vars
        |> Enum.into([])
        |> Keyword.merge(size: {vars.x, vars.y, vars.z},
                         float_size: vars.float_size)
    vars
    |> Map.put(:conv, c)
    |> Map.put(:conv_vars, Nodes.Convolution.vars(c, env))
  end

  defp process_pooling(%{pooling: p, conv_vars: cv} = vars, env) do
    with p when is_list(p) <- pooling(p) do
      p = Keyword.merge(p, size: {cv.ox, cv.oy, cv.oz},
                           float_size: vars.float_size)
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

  defp padding(nil), do: false
  defp padding(n) when is_integer(n), do: [padding_size: {n, n}]
  defp padding({_, _} = p), do: [padding_size: p]
  defp padding(p) when is_list(p) do
    if Keyword.keyword?(p), do: p, else: false
  end
  defp padding(_), do: false

  defp pooling(nil), do: false
  defp pooling(n) when is_integer(n), do: [pooling: {n, n}, stride: {n, n}]
  defp pooling({_, _} = p), do: [pooling: p, stride: p]
  defp pooling(p) when is_list(p) do
    if Keyword.keyword?(p), do: p, else: false
  end
  defp pooling(_), do: false
end
