defmodule Neuro.Layers.Convolution do
  use Cuda.Graph
  alias Neuro.Nodes
  alias Neuro.Nodes.Base
  import Base, only: [input_type: 1, output_type: 1]

  def __assigns__(opts, _env) do
    %{vars: vars(opts |> Enum.into(%{}))}
  end

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
    {graph, next} = case vars.padding do
      false ->
        {graph, :input}
      padding ->
        graph = graph
                |> add(:padding, Nodes.Padding, padding)
                |> link(:input, {:padding, :input})
        {graph, {:padding, :output}}
    end
    graph = graph
            |> add(:conv, Nodes.Convolution, vars.conv)
            |> link(next, {:conv, :input})
    {graph, next} = case vars.pooling do
      false ->
        {graph, {:conv, :output}}
      pooling ->
        graph = graph
                |> add(:pooling, Nodes.Pooling, pooling)
                |> link({:conv, :output}, {:pooling, :input})
        {graph, {:pooling, :output}}
    end
    graph
    |> link(next, :output)
  end

  def __run__(_) do
    [:conv]
  end

  defp vars(opts) do
    float_size = opts |> Map.get(:float_size) |> Base.float_size()
    {x, y, z}  = opts |> Map.get(:size) |> Base.triple_size()
    opts
    |> Map.merge(%{x: x, y: y, z: z, float_size: float_size})
    |> process_padding()
    |> process_conv()
    |> process_pooling()
    |> Map.drop([:size])
    |> Map.put(:f, "f#{float_size * 8}")
  end

  defp process_padding(%{padding: p} = vars) do
    with p when is_map(p) <- padding(p) do
      p = Map.merge(p, %{size: {vars.x, vars.y, vars.z},
                         float_size: vars.float_size})
      vars
      |> Map.put(:padding, p)
      |> Map.put(:padding_vars, Nodes.Padding.vars(p))
    else
      _ -> %{vars | padding: false}
    end
  end
  defp process_padding(vars) do
    Map.put(vars, :padding, false)
  end

  defp process_conv(%{padding_vars: pv} = vars) do
    c = vars
        |> Map.drop([:padding, :padding_vars, :pooling])
        |> Map.merge(%{size: {pv.ox, pv.oy, pv.oz},
                       float_size: vars.float_size})
    vars
    |> Map.put(:conv, c)
    |> Map.put(:conv_vars, Nodes.Convolution.vars(c))
  end
  defp process_conv(vars) do
    c = Map.merge(vars, %{size: {vars.x, vars.y, vars.z},
                          float_size: vars.float_size})
    vars
    |> Map.put(:conv, c)
    |> Map.put(:conv_vars, Nodes.Convolution.vars(c))
  end

  defp process_pooling(%{pooling: p, conv_vars: cv} = vars) do
    with p when is_map(p) <- pooling(p) do
      p = Map.merge(p, %{size: {cv.ox, cv.oy, cv.oz},
                         float_size: vars.float_size})
      vars
      |> Map.put(:pooling, p)
      |> Map.put(:pooling_vars, Nodes.Pooling.vars(p))
    else
      _ -> %{vars | pooling: false}
    end
  end
  defp process_pooling(vars) do
    Map.put(vars, :pooling, false)
  end

  defp padding(nil), do: false
  defp padding(n) when is_integer(n), do: %{padding_size: {n, n}}
  defp padding({_, _} = p), do: %{padding_size: p}
  defp padding(p) when is_map(p), do: p
  defp padding(list) when is_list(list) do
    case Keyword.keyword?(list) do
      true -> list |> Enum.into(%{})
      _    -> false
    end
  end

  defp pooling(nil), do: false
  defp pooling(n) when is_integer(n), do: %{pooling_size: {n, n}}
  defp pooling({_, _} = p), do: %{pooling_size: p}
  defp pooling(p) when is_map(p), do: p
  defp pooling(list) when is_list(list) do
    case Keyword.keyword?(list) do
      true -> list |> Enum.into(%{})
      _    -> false
    end
  end
end
