defmodule Neuro.Network do
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
        GenServer.start_link(__MODULE__, opts, name: name)
      end

      # Cuda.Graph behaviour
      def __assigns__(opts, _env) do
        opts |> Enum.into(%{}) |> unquote(__MODULE__).vars
      end
      def __graph__(graph) do
        graph(graph)
      end
      def __child_options__(_id, _module, %{nodes: [], assigns: %{options: opts}} = graph) do
        import Cuda.Graph.Node, only: [input_pin_types: 0]
        {_, input} = graph
                     |> Cuda.Graph.NodeProto.pins(input_pin_types())
                     |> List.first
                     |> Map.get(:data_type)
        [size: input,
         float_size: Keyword.get(opts, :float_size, unquote(float_size))]
      end
      def __child_options__(_id, _module, %{nodes: [last | _], assigns: %{options: opts}}) do
        import Cuda.Graph.Node, only: [output_pin_types: 0]
        {_, output} = last
                      |> Cuda.Graph.NodeProto.pins(output_pin_types())
                      |> List.first
                      |> Map.get(:data_type)
        [size: output,
         float_size: Keyword.get(opts, :float_size, unquote(float_size))]
      end
      def __child_options__(_id, _module, _graph), do: []

      def init(opts) do
        opts = Keyword.merge(opts, network: __MODULE__, vars: unquote(@vars))
        children = [
          worker(Neuro.Variables, [name: unquote(@vars)]),
          worker(Neuro.Worker, opts)
        ]
        supervise(children, startegy: :one_for_one)
      end

      def graph(graph), do: graph

      defoverridable graph: 1
    end
  end

  defmacro __before_compile__(env) do
    pins = env.module |> Module.get_attribute(@pins) |> Macro.escape
    quote do
      def __pins__(_assigns) do
        unquote(pins)
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

  def vars(opts) do
    opts
  end
end
