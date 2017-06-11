defmodule Neuro.Network do
  use Supervisor
  alias Cuda.{Graph, Graph.GPUNode, Memory}

  @pins     __MODULE__.Pins
  @f        __MODULE__.F
  @exports  collect_shared: 1, input: 1, input: 2, load_network: 2,
            output: 1, output: 2

  defmacro __using__(opts) do
    float_size = opts |> Keyword.get(:float_size) |> Neuro.Nodes.Base.float_size
    f = "f#{float_size * 8}"

    quote do
      use Supervisor
      use Cuda.Graph

      @before_compile unquote(__MODULE__)
      @cuda   __MODULE__.Cuda
      @shared __MODULE__.Shared
      @worker __MODULE__.Worker
      import unquote(__MODULE__), only: unquote(@exports)

      Module.register_attribute(__MODULE__, unquote(@pins), accumulate: true)
      Module.register_attribute(__MODULE__, unquote(@f), [])
      Module.put_attribute(__MODULE__, unquote(@f), unquote(f))

      def __cuda__(), do: @cuda

      def start_link(opts \\ []) do
        {name, opts} = Keyword.pop(opts, :name, __MODULE__)
        Supervisor.start_link(__MODULE__, opts, name: name)
      end

      def init(opts) do
        opts = Keyword.merge(opts, cuda: @cuda)
        shared_values = Keyword.get(opts, :shared, %{})
        with %{} = graph   <- load_network(__MODULE__, opts),
             {:ok, shared} <- collect_shared(graph) do
          children = [supervisor(Registry, [:unique, @shared])]
          {children, shared} = Enum.reduce(shared, {children, %{}}, fn {k, memory}, {children, shared} ->
            name = {:via, Registry, {@shared, k}}
            memory = Memory.new(memory, :shared)
            shared_opts = case Map.get(shared_values, k) do
              nil    -> []
              values -> [vars: values]
            end
            shared_opts = shared_opts
                          |> Keyword.merge(memory: memory, name: name)
            {children ++ [worker(Cuda.Shared, [shared_opts])], Map.put(shared, k, memory)}
          end)
          memory = graph.assigns |> Map.get(:memory, %{}) |> Map.merge(shared)
          graph = Cuda.Graph.NodeProto.assign(graph, :memory, memory)
          shared = shared
                   |> Enum.map(fn {k, _} -> {k, {:via, Registry, {@shared, k}}} end)
                   |> Enum.into(%{})
          opts = Keyword.merge(opts, graph: graph, shared: shared, name: @worker)
          children = children ++ [worker(Cuda.Worker, [opts])]
          supervise(children, strategy: :one_for_one)
        end
      end

      # Cuda.Graph behaviour
      def __assigns__(opts, _env) do
        opts |> Enum.into(%{}) |> unquote(__MODULE__).vars
      end
      def __graph__(%{assigns: %{options: opts}} = graph) do
        #IO.inspect(opts)
        case Keyword.get(opts, :type) do
          :training         -> unquote(__MODULE__).make_training_graph(graph)
          :back_propagation -> unquote(__MODULE__).make_back_propagation_graph(graph)
          _                 -> graph(graph)
        end
      end
      def __child_options__(_id, _module, %{nodes: [], assigns: %{options: opts}} = graph) do
        import Cuda.Graph.Node, only: [input_pin_types: 0]
        opts = [float_size: Keyword.get(opts, :float_size, unquote(float_size)),
                training: Keyword.get(opts, :training)]
        with [input | _] <- graph |> Cuda.Graph.NodeProto.pins(input_pin_types()),
             {_, type} = input.data_type do
          Keyword.put(opts, :size, type)
        else
          _ -> opts
        end
      end
      def __child_options__(_id, _module, %{nodes: [last | _], assigns: %{options: opts}}) do
        import Cuda.Graph.Node, only: [output_pin_types: 0]
        opts = [float_size: Keyword.get(opts, :float_size, unquote(float_size)),
                training: Keyword.get(opts, :training)]
        with [output | _] <- last |> Cuda.Graph.NodeProto.pins(output_pin_types()),
             {_, type} = output.data_type do
          Keyword.put(opts, :size, type)
        else
          _ -> opts
        end
      end
      def __child_options__(_id, _module, %{assigns: %{options: opts}}) do
        [training: Keyword.get(opts, :training)]
      end

      def graph(graph), do: graph

      def run(input) do
        Cuda.Worker.run(@worker, input)
      end

      def gpu_info() do
        Cuda.device_info(@cuda)
      end

      defoverridable graph: 1
    end
  end

  defmacro __before_compile__(env) do
    pins = env.module |> Module.get_attribute(@pins) |> Macro.escape
    quote do
      def __pins__(assigns) do
        import Cuda.Graph.Node, only: [output_pin_types: 0]
        pins = unquote(pins)
        type = Keyword.get(assigns.options, :type)
        f = Cuda.Env.f(assigns.env)
        pins = if type != :back_propagation do
          pins |> Enum.map(fn
            %{type: :input} = pin -> %{pin | group: :activation}
            pin -> pin
          end)
        else
          pins
        end
        case type do
          :training ->
            output = pins |> Enum.find(& &1.type in output_pin_types())
            pins = pins |> Enum.reject(& &1.type in output_pin_types())
            reply = %{output | id: :reply, type: :input}
            error = Cuda.Graph.Node.output(:error, f)
            pins ++ [reply, error]
          :back_propagation ->
            unquote(pins) |> Enum.reduce([], fn
              %{type: :output} = pin, pins -> [%{pin | type: :input} | pins]
              %{type: :input} = pin, pins -> [%{pin | type: :output} | pins]
              _, pins -> pins
            end)
          _ ->
            pins
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
    iopts = options
            |> Keyword.drop([:type])
            |> Keyword.merge(training: true)
    graph = graph |> Cuda.Graph.add(:inference, graph.module, iopts)
    inference = Cuda.Graph.GraphProto.node(graph, :inference)

    bopts = Keyword.merge(iopts, type: :back_propagation, inference: inference)
    graph = graph |> Cuda.Graph.add(:back_propagation, graph.module, bopts)
    back = Cuda.Graph.GraphProto.node(graph, :back_propagation)

    e_size = Cuda.Graph.Pin.data_arity(Cuda.Graph.NodeProto.pin(inference, :output))
    t_size = Cuda.Graph.Pin.data_arity(Cuda.Graph.NodeProto.pin(back, :input))

    graph = graph
    |> Cuda.Graph.add(:error, Neuro.Nodes.Error, size: e_size)
    |> Cuda.Graph.add(:terminator, Neuro.Nodes.Terminator, size: t_size)
    |> Cuda.Graph.link(:input, {:inference, :input})
    |> Cuda.Graph.link(:reply, {:error, :reply})
    |> Cuda.Graph.link({:inference, :output}, {:error, :input})
    |> Cuda.Graph.link({:error, :output}, {:back_propagation, :output})
    |> Cuda.Graph.link({:error, :error}, :error)
    |> Cuda.Graph.link({:back_propagation, :input}, {:terminator, :input})

    graph = graph |> Cuda.Graph.Processing.expand

    graph = graph.links |> Enum.reduce(graph, fn
      {src, {dst, _}}, graph when is_tuple(dst) and elem(dst, 0) == :inference ->
        src = case src do
          {:__self__, src} -> src
          src              -> src
        end
        back_id = put_elem(dst, 0, :back_propagation)
        graph |> Cuda.Graph.link(src, {back_id, :inference})
      _, graph ->
        graph
    end)

    graph
  end

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

  def start_link(module, opts \\ []) do
    Supervisor.start_link(__MODULE__, {module, opts})
  end

  @networks __MODULE__.Networks
  def run(module, inputs) do
    module.run(inputs)
  end

  def init({module, opts}) do
    # TODO: create a wrapper supervisor to restart all others when Cuda crashed
    opts = Keyword.merge(opts, name: {:via, Registry, {@networks, module}})
    children = [
      supervisor(Registry, [:unique, @networks]),
      worker(Cuda, [[name: module.__cuda__()]]),
      worker(module, [opts])
    ]
    supervise(children, strategy: :one_for_one)
  end

  def load_network(module, opts) do
    cuda = Keyword.fetch!(opts, :cuda)
    with {:ok, info} <- Cuda.device_info(cuda) do
      nopts = Keyword.get(opts, :network_options, [])
      proto = struct(Cuda.Graph.Node.proto(module))
      env = %Cuda.Env{gpu_info: info}
      Cuda.Graph.Factory.new(proto, :network, module, nopts, env)
    end
  end

  defp merge_shared(_, a, b) when is_map(a) and is_map(b) do
    Map.merge(a, b, &merge_shared/3)
  end
  defp merge_shared(_, _, b) do
    b
  end

  def collect_shared(%GPUNode{assigns: %{shared: shared}}) do
    {:ok, Map.merge(%{}, shared)}
  end
  def collect_shared(%{id: gid, nodes: _} = graph) do
    Cuda.Graph.Processing.dfs(graph, fn
      :enter, {%Graph{id: ^gid, assigns: assigns}, _}, st ->
        {:ok, Map.merge(st, Map.get(assigns, :shared, %{}), &merge_shared/3)}
      :enter, {%Graph{assigns: assigns} = g, _}, st ->
        with shared <- Map.merge(st, Map.get(assigns, :shared, %{}), &merge_shared/3),
             {:ok, nested} <- collect_shared(g) do
          {:ok, Map.merge(shared, nested, &merge_shared/3)}
        end
      :enter, {%{assigns: %{shared: shared}}, _}, st ->
        {:ok, Map.merge(st, shared, &merge_shared/3)}
      _, _, st ->
        {:ok, st}
    end, %{})
  end
  def collect_shared(_), do: {:ok, %{}}
end
