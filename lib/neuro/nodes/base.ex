defmodule Neuro.Nodes.Base do
  defmodule Helpers do
    #def shared_offset(ctx, name) do
    #  id = with {parent_id, _} <- ctx.node.id, do: parent_id
    #  id = Cuda.Graph.Node.string_id(id)
    #  # IO.inspect({id, name, ctx.assigns.shared_offsets})
    #  ctx.assigns.shared_offsets[name][id]
    #end
  end

  defmacro __using__(opts) do
    proto = Keyword.get(opts, :proto, Cuda.Graph.GPUNode)
    quote do
      use unquote(proto)
      import unquote(__MODULE__)

      def vars(_opts, _env), do: %{}
      def shared(_vars), do: %{}

      def __assigns__(opts, env) do
        float_size = opts |> Keyword.get(:float_size) |> float_size()
        f = "f#{float_size * 8}"
        vars = opts
               |> Keyword.merge(float_size: float_size, f: f)
               |> vars(env)
               |> Enum.into(%{})
               |> Map.merge(%{float_size: float_size, f: f})
        assigns = %{vars: vars, helpers: [Neuro.Nodes.Base.Helpers]}
        case Keyword.get(opts, :back_propagation) do
          true -> Map.merge(assigns, %{back_propagation: true})
          _    -> Map.merge(assigns, %{shared: shared(vars)})
        end
      end

      def __compile__(%{id: id, assigns: %{back_propagation: true}} = node) do
        case id |> Cuda.Graph.Node.string_id() |> String.split("__") do
          ["back_propagation" | id] ->
            id = ["inference" | id] |> Enum.join("__")
            {:ok, Cuda.Graph.NodeProto.assign(node, :alias, id)}
          _ ->
            {:ok, node}
        end
      end
      def __compile__(node) do
        case Keyword.get(node.assigns.options, :alias) do
          nil -> {:ok, node}
          a   -> {:ok, Cuda.Graph.NodeProto.assign(node, :alias, a)}
        end
      end

      def __pins__(%{back_propagation: true} = assigns) do
        #IO.inspect({:!, output_type(assigns.vars)})
        [output(:input, input_type(assigns.vars)),
         input(:output, output_type(assigns.vars))]
      end
      def __pins__(assigns) do
        [input(:input,   input_type(assigns.vars)),
         output(:output, output_type(assigns.vars))]
      end

      defoverridable __compile__: 1, __pins__: 1, shared: 1, vars: 2
    end
  end

  def input_type(%{z: 1, x: x, y: 1, f: f}), do: {f, x}
  def input_type(%{z: 1, x: x, y: y, f: f}), do: {f, {x, y}}
  def input_type(%{z: z, x: x, y: y, f: f}), do: {f, {x, y, z}}

  def output_type(%{oz: 1, ox: x, oy: 1, f: f}), do: {f, x}
  def output_type(%{oz: 1, ox: x, oy: y, f: f}), do: {f, {x, y}}
  def output_type(%{oz: z, ox: x, oy: y, f: f}), do: {f, {x, y, z}}

  def triple_size({x, y}), do: {x, y, 1}
  def triple_size({_, _, _} = tuple), do: tuple
  def triple_size(x) when is_integer(x), do: {x, x, 1}
  def triple_size(x) do
    raise CompileError, description: "Invalid size specified: #{inspect x}"
  end

  def plane_size({x, y}),               do: x * y
  def plane_size({x, y, z}),            do: x * y * z
  def plane_size(x) when is_integer(x), do: x
  def plane_size(x) do
    raise CompileError, description: "Invalid size specified: #{inspect x}"
  end

  def stride({_, _} = tuple), do: tuple
  def stride(x) when is_integer(x), do: {x, x}
  def stride(_), do: {1, 1}

  def float_size(x) when x in [16, 32, 64], do: x / 8
  def float_size(_), do: 4

  def input(assigns) do
    f = assigns.vars.f
    x = assigns.vars.x
    y = assigns.vars.y
    z = assigns.vars.z
    cond do
      z == 1 and y == 1 -> {f, x}
      z == 1            -> {{f, x}, {f, y}}
      true              -> {{f, x}, {f, y}, {f, z}}
    end
  end

  def activation(:relu), do: Neuro.Nodes.Activation.Relu
  def activation(_) do
    raise CompileError, description: "Invalid activation function"
  end
end
