defmodule Neuro.Layers.Base do
  defmacro __using__(opts) do
    proto = Keyword.get(opts, :proto, Cuda.Graph)
    quote do
      use unquote(proto)
      import unquote(__MODULE__)

      def vars(_opts, _env), do: %{}

      def __assigns__(id, opts, env) do
        vars = opts |> vars(env) |> Map.merge(%{layer: id})
        back_propagation = Keyword.get(opts, :back_propagation) == true
        %{vars: vars,
          back_propagation: back_propagation,
          training: back_propagation || Keyword.get(opts, :training) == true}
      end

      def __child_options__(_, _, %{assigns: %{vars: %{layer: layer} = assigns}}) do
        assigns
        |> Map.take(~w(back_propagation training)a)
        |> Enum.into([])
        |> Keyword.merge(layer: layer)
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
      def __compile__(%{id: id} = node) do
        assigns = case Keyword.get(node.assigns.options, :alias) do
          nil -> %{}
          a   -> %{alias: a}
        end
        {:ok, Cuda.Graph.NodeProto.assign(node, assigns)}
      end

      def __pins__(%{back_propagation: true, env: env, vars: vars}) do
        [output(:input, input_type(vars, env)),
         input(:output, output_type(vars, env)),
         #input(:result, output_type(vars, env), :fixed),
         input(:inference, input_type(vars, env), :activation)]
      end
      def __pins__(%{env: env, vars: vars} = assigns) do
        group = case Map.get(assigns, :training) do
          true -> :activation
          _    -> nil
        end
        [input(:input,   input_type(vars, env), group),
         output(:output, output_type(vars, env), group)]
      end

      defoverridable __child_options__: 3, __compile__: 1, __pins__: 1, vars: 2
    end
  end

  def input_type(%{z: 1, x: x, y: 1}, env), do: {Cuda.Env.f(env), x}
  def input_type(%{z: 1, x: x, y: y}, env), do: {Cuda.Env.f(env), {x, y}}
  def input_type(%{z: z, x: x, y: y}, env), do: {Cuda.Env.f(env), {x, y, z}}

  def output_type(%{oz: 1, ox: x, oy: 1}, env), do: {Cuda.Env.f(env), x}
  def output_type(%{oz: 1, ox: x, oy: y}, env), do: {Cuda.Env.f(env), {x, y}}
  def output_type(%{oz: z, ox: x, oy: y}, env), do: {Cuda.Env.f(env), {x, y, z}}

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

  def activation(:relu), do: Neuro.Nodes.Activation.Relu
  def activation(_) do
    raise CompileError, description: "Invalid activation function"
  end
end
