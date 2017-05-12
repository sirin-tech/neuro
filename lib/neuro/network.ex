defmodule Neuro.Network do
  defmacro __using__(_opts) do
    quote do
      use Supervisor
      use Cuda.Graph

      def start_link(opts \\ []) do
        {name, opts} = Keyword.pop(opts, :name, __MODULE__)
        GenServer.start_link(__MODULE__, opts, name: name)
      end

      # Cuda.Graph behaviour
      def __assigns__(opts, _env) do
        {:ok, opts |> Enum.into(%{}) |> unquote(__MODULE__).vars}
      end
      def __graph__(graph) do
        graph(graph)
      end
      def __pins__(assigns) do
        []
      end

      def init(opts) do
        opts = Keyword.merge(opts, network: __MODULE__)
        children = [worker(Neuro.Worker, opts)]
        supervise(children, startegy: :one_for_one)
      end

      def graph(graph), do: graph

      defoverridable init: 1, graph: 1
    end
  end

  def vars(opts) do
    opts
  end
end
