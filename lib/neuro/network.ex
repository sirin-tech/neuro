defmodule Neuro.Network do
  # alias Cuda.Graph.Factory
  # alias Cuda.Graph.Node

  @pins     __MODULE__.Pins
  @f        __MODULE__.F
  @vars     __MODULE__.Vars
  @exports  input: 1, input: 2, output: 1, output: 2

  defmacro __using__(opts) do
    float_size = opts |> Keyword.get(:float_size) |> Neuro.Nodes.Base.float_size
    f = "f#{float_size * 8}"

    quote do
      use Supervisor
      use Cuda.Graph

      @before_compile unquote(__MODULE__)
      import unquote(__MODULE__), only: unquote(@exports)

      Module.register_attribute(__MODULE__, unquote(@pins), accumulate: true)
      Module.register_attribute(__MODULE__, unquote(@f), [])
      Module.put_attribute(__MODULE__, unquote(@f), unquote(f))

      def start_link(opts \\ []) do
        {name, opts} = Keyword.pop(opts, :name, __MODULE__)
        Supervisor.start_link(__MODULE__, opts, name: name)
      end

      # Cuda.Graph behaviour
      def __assigns__(opts, _env) do
        opts |> Enum.into(%{}) |> unquote(__MODULE__).vars
      end
      def __graph__(%{assigns: %{options: opts}} = graph) do
        case Keyword.get(opts, :type) do
          :training         -> unquote(__MODULE__).make_training_graph(graph)
          :back_propagation -> unquote(__MODULE__).make_back_propagation_graph(graph)
          _                 -> graph(graph)
        end
      end
      def __child_options__(_id, _module, %{nodes: [], assigns: %{options: opts}} = graph) do
        import Cuda.Graph.Node, only: [input_pin_types: 0]
        opts = [float_size: Keyword.get(opts, :float_size, unquote(float_size))]
        with [input | _] <- graph |> Cuda.Graph.NodeProto.pins(input_pin_types()),
             {_, type} = input.data_type do
          Keyword.put(opts, :size, type)
        else
          _ -> opts
        end
      end
      def __child_options__(_id, _module, %{nodes: [last | _], assigns: %{options: opts}}) do
        import Cuda.Graph.Node, only: [output_pin_types: 0]
        opts = [float_size: Keyword.get(opts, :float_size, unquote(float_size))]
        with [output | _] <- last |> Cuda.Graph.NodeProto.pins(output_pin_types()),
             {_, type} = output.data_type do
          Keyword.put(opts, :size, type)
        else
          _ -> opts
        end
      end
      def __child_options__(_id, _module, _graph) do
        []
      end

      def init(opts) do
        opts = Keyword.merge(opts, network: __MODULE__,
                                   shared_pid: unquote(@vars),
                                   name: __MODULE__.Worker)
        children = [
          worker(Cuda.Shared, [[name: unquote(@vars)]]),
          worker(Cuda.Worker, [opts])
        ]
        supervise(children, strategy: :one_for_one)
      end

      def graph(graph), do: graph

      def run(input) do
        Cuda.Worker.run(__MODULE__.Worker, input)
      end

      def gpu_info() do
        Cuda.Worker.gpu_info(__MODULE__.Worker)
      end

      defoverridable graph: 1
    end
  end

  defmacro __before_compile__(env) do
    pins = env.module |> Module.get_attribute(@pins) |> Macro.escape
    quote do
      def __pins__(assigns) do
        import Cuda.Graph.Node, only: [output_pin_types: 0]
        case Keyword.get(assigns.options, :type) do
          :training ->
            pins = unquote(pins)
            output = pins |> Enum.find(& &1.type in output_pin_types())
            reply = %{output | id: :reply, type: :input}
            pins ++ [reply]
          :back_propagation ->
            unquote(pins) |> Enum.reduce([], fn
              %{type: :output} = pin, pins -> [%{pin | type: :input} | pins]
              %{type: :input} = pin, pins -> [%{pin | type: :output} | pins]
              _, pins -> pins
            end)
          _ ->
            unquote(pins)
        end
      end
    end
  end

  defmacro input(arity) do
    quote do
      f = Module.get_attribute(__MODULE__, unquote(@f))
      pin = Cuda.Graph.Node.input(:input, {f, unquote(arity)})
      Module.put_attribute(__MODULE__, unquote(@pins), pin)
    end
  end
  defmacro input(id, arity) do
    quote do
      f = Module.get_attribute(__MODULE__, unquote(@f))
      pin = Cuda.Graph.Node.input(unquote(id), {f, unquote(arity)})
      Module.put_attribute(__MODULE__, unquote(@pins), pin)
    end
  end

  defmacro output(arity) do
    quote do
      f = Module.get_attribute(__MODULE__, unquote(@f))
      pin = Cuda.Graph.Node.output(:output, {f, unquote(arity)})
      Module.put_attribute(__MODULE__, unquote(@pins), pin)
    end
  end
  defmacro output(id, arity) do
    quote do
      f = Module.get_attribute(__MODULE__, unquote(@f))
      pin = Cuda.Graph.Node.output(unquote(id), {f, unquote(arity)})
      Module.put_attribute(__MODULE__, unquote(@pins), pin)
    end
  end

  def make_training_graph(%{assigns: %{options: options}} = graph) do
    iopts = Keyword.drop(options, [:type])
    graph = graph |> Cuda.Graph.add(:inference, graph.module, iopts)
    inference = Cuda.Graph.GraphProto.node(graph, :inference)

    bopts = Keyword.merge(iopts, type: :back_propagation, inference: inference)
    graph = graph |> Cuda.Graph.add(:back_propagation, graph.module, bopts)
    back = Cuda.Graph.GraphProto.node(graph, :back_propagation)

    eopts = [input: Cuda.Graph.NodeProto.pin(inference, :output),
             output: Cuda.Graph.NodeProto.pin(back, :input)]
    graph = graph
    |> Cuda.Graph.add(:error, Neuro.Nodes.Error, eopts)
    |> Cuda.Graph.link(:input, {:inference, :input})
    |> Cuda.Graph.link(:reply, {:error, :reply})
    |> Cuda.Graph.link({:inference, :output}, {:error, :input})
    |> Cuda.Graph.link({:error, :output}, {:back_propagation, :output})
    |> Cuda.Graph.link({:back_propagation, :input}, :output)

    shared = graph |> collect_shared(%{})

    graph = graph
    |> Cuda.Graph.Processing.expand
    graph = graph.nodes |> Enum.reduce(graph, fn
      %{id: id, type: :gpu}, graph when is_tuple(id) and elem(id, 0) == :inference ->
        back_id = put_elem(id, 0, :back_propagation)
        #IO.inspect({id, back_id})
        graph |> Cuda.Graph.link({id, :output}, {back_id, :result})
      _, graph ->
        graph
    end)
    #|> IO.inspect
    # IO.puts(dump(graph |> Cuda.Graph.Processing.expand))
    #graph
    Cuda.Graph.NodeProto.assign(graph, :shared, shared)
  end

  defp collect_shared(%{nodes: nodes} = graph, shared) do
    with {:ok, graph} <- graph.module.__compile__(graph) do
      shared = Map.merge(shared, Map.get(graph.assigns, :shared, %{}))
      Enum.reduce(nodes, shared, &collect_shared/2)
    else
      _ -> shared
    end
  end
  defp collect_shared(_node, shared), do: shared

  def dump(graph, indent \\ "") do
    nodes = case Map.get(graph, :nodes) do
      nil   -> ""
      nodes -> indent <> "nodes:\n" <> (nodes |> Enum.map(& dump(&1, indent <> "  ")) |> Enum.join("\n"))
    end
    links = case Map.get(graph, :links) do
      nil   -> ""
      links -> indent <> "links:\n#{indent}  " <>
               (links |> Enum.map(&"#{inspect &1}") |> Enum.join("\n#{indent}  "))
    end
    pins = indent <> "pins:\n#{indent}  " <>
           (graph.pins |> Enum.map(&"#{inspect &1}") |> Enum.join("\n#{indent}  "))
    [indent <> "id: #{Cuda.Graph.Node.string_id(graph.id)}",
     "#{indent}type: #{graph.type}",
     pins,
     nodes,
     links] |> Enum.join("\n")
  end

  def make_back_propagation_graph(%{assigns: %{options: options}} = graph) do
    inference = Keyword.get(options, :inference)
    result = Cuda.Graph.Processing.dfs_reverse(inference, fn
      :enter, {%{id: :inference}, _}, graph ->
        {:ok, graph}
      :enter, {node, _}, graph ->
        opts = node.assigns.options |> Keyword.put(:back_propagation, true)
        graph = graph |> Cuda.Graph.add(node.id, node.module, opts)
        {:ok, graph}
      :move, {{src, src_pin}, {dst, dst_pin}}, graph ->
        src_id = if src.id == :inference, do: :__self__, else: src.id
        dst_id = if dst.id == :inference, do: :__self__, else: dst.id
        link = {{src_id, src_pin.id}, {dst_id, dst_pin.id}}
        {:ok, %{graph | links: [link | graph.links]}}
      _, _, graph ->
        {:ok, graph}
    end, graph)
    with {:ok, graph} <- result do
      graph
    end
  end

  def vars(opts) do
    opts
  end
end
