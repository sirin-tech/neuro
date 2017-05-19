defmodule Neuro.Worker do
  use GenServer
  alias Cuda.Shared
  alias Cuda.Graph
  alias Cuda.Graph.Factory
  alias Cuda.Graph.Processing
  alias Cuda.Compiler.Unit
  alias Cuda.Compiler.Context
  alias Cuda.Runner

  def start_link(opts \\ []) do
    {opts, args} = Keyword.split(opts, ~w(name)a)
    GenServer.start_link(__MODULE__, args, opts)
  end

  def init(opts) do
    with {:ok, module} <- Keyword.fetch(opts, :network),
         {:ok, vars} <- Keyword.fetch(opts, :vars),
         {:ok, cuda} <- Cuda.start_link(),
         {:ok, info} <- Cuda.device_info(cuda) do
      st = %{cuda: cuda, vars: vars, module: module, info: info,
             graph: nil, shared: nil, shared_offsets: nil}
      load_network(st, opts)
    end
  end

  def run(pid, input) do
    GenServer.call(pid, {:run, input})
  end

  def gpu_info(pid) do
    GenServer.call(pid, :info)
  end

  def handle_call({:run, input}, _from, st) do
    opts = [cuda: st.cuda, args: %{shared: st.shared}]
    # [x] = st.graph.nodes
    # IO.inspect(x.assigns.batch)
    result = Runner.run(st.graph, input, opts)
    {:reply, result, st}
  end

  def handle_call(:info, _from, st) do
    {:reply, {:ok, st.info}, st}
  end

  defp load_network(st, opts) do
    with {:ok, weights} <- Keyword.fetch(opts, :weights),
         {:ok, biases} <- Keyword.fetch(opts, :biases),
         # prepare compile environment
         env = %Cuda.Env{gpu_info: st.info},
         # create new graph
         %{} = graph <- Factory.new(%Graph{}, :network, st.module, [], env),
         ctx = %Context{},
         # we need to precompile graph to calculate CTA parameters and
         # shared variables sizes (weights and biases)
         {:ok, precompiled} <- Unit.compile(graph, %{ctx | assigns: %{compile_sources: false}}),
         # prepare shared variables
         {:ok, shared} <- collect_shared(precompiled),
         vars = %{weights: {shared.weights, weights}, biases: {shared.biases, biases}},
         # load shared variables
         {:ok, _} <- Shared.load(st.vars, vars),
         {:ok, shared} <- Shared.share(st.vars),
         {:ok, shared} <- Cuda.memory_load(st.cuda, shared),
         # retrieve shared variables offset for compilation
         {:ok, offsets} <- Shared.offsets(st.vars),
         # compile sources into cubin
         {:ok, graph} <- Unit.compile(graph, %{ctx | assigns: %{shared_offsets: offsets}}),
         # load compiled cubins into GPU
         {:ok, graph} <- Runner.load(graph, shared: shared, cuda: st.cuda) do
      {:ok, %{st | graph: graph, shared: shared, shared_offsets: offsets}}
    end
  end

  defp collect_shared(graph) do
    Processing.dfs(graph, fn
      :enter, {%{type: :computation_graph, nodes: nodes}, _}, st ->
        st = nodes |> Enum.reduce(st, fn
          %{id: {id, _}, assigns: %{vars: %{neurons: n, weights: w, f: f}}}, st ->
            st
            |> Map.put(:weights, Map.put(st.weights, id, {f, w}))
            |> Map.put(:biases, Map.put(st.biases, id, {f, n}))
          _, st ->
            st
        end)
        {:ok, st}
      _, _, st ->
        {:ok, st}
    end, %{weights: %{}, biases: %{}})
  end
end
