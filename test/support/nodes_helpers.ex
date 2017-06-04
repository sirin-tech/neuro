defmodule Neuro.Test.NodesHelpers do
  import Neuro.Test.CudaHelpers
  alias Cuda.Compiler.Unit
  alias Cuda.Graph.{Factory, NodeProto}
  alias Cuda.{Memory, Runner}

  def disable_logging(_) do
    log_level = Logger.level()
    Logger.configure(level: :warn)
    ExUnit.Callbacks.on_exit(fn -> Logger.configure(level: log_level) end)
    :ok
  end

  def load_graph(ctx) do
    {:ok, load_graph(ctx[:graph], ctx[:options], ctx[:shared])}
  end

  def load_graph(module, opts, shared_values) do
    {:ok, cuda} = Cuda.start_link()
    opts = [cuda: cuda, network_options: opts]
    graph = Neuro.Network.load_network(module, opts)
    {shared_opts, ctx} = if is_nil(shared_values) do
      {[], []}
    else
      {:ok, shared} = Neuro.Network.collect_shared(graph)
      {:ok, shared_pid} = Cuda.Shared.start_link()
      memory = Memory.new(shared.shared, :shared)
      {:ok, _} = Cuda.Shared.load(shared_pid, memory, shared_values)
      memory = graph.assigns
               |> Map.get(:memory, %{})
               |> Map.put(:shared, memory)
      g = NodeProto.assign(graph, :memory, memory)
      {[graph: g, shared: %{shared: shared_pid}], [shared_pid: shared_pid]}
    end
    opts = [cuda: cuda, graph: graph] |> Keyword.merge(shared_opts)
    Keyword.merge(ctx, worker_options: opts)
  end

  defp make_binary(v) when is_list(v) do
    v |> List.flatten |> Enum.reduce(<<>>, & &2 <> <<&1::float-little-32>>)
  end

  def round!(x, prec \\ 1)
  def round!(l, prec) when is_list(l), do: Enum.map(l, &round!(&1, prec))
  def round!(f, prec) when is_float(f), do: Float.round(f, prec)
  def round!(x, _prec), do: x

  def run(proto, id, module, opts, inputs, args \\ %{}) do
    args = args |> Enum.map(fn
      {k, v} -> {k, v |> make_binary}
      x      -> x
    end)

    l = Logger.level()
    Logger.configure(level: :error)

    {:ok, cuda} = Cuda.start_link()
    {:ok, info} = Cuda.device_info(cuda)

    node = Factory.new(proto, id, module, opts, %Cuda.Env{gpu_info: info})

    {:ok, node} = Unit.compile(node, context())
    Logger.configure(level: l)

    args = Enum.reduce(args, %{}, fn
      {k, v}, acc ->
        with {:ok, m} <- Cuda.memory_load(cuda, v) do
          Map.put(acc, k, m)
        else
          _ -> acc
        end
      _, acc ->
        acc
    end)

    with {:ok, node}    <- Runner.load(node, cuda: cuda),
         {:ok, outputs} <- Runner.run(node, inputs, cuda: cuda, args: args) do
      outputs
      |> Enum.map(fn {k, v} -> {k, round!(v)} end)
      |> Enum.into(%{})
    else
      _ -> %{}
    end
  end
end
