defmodule Neuro.Stream.Producer do
  defmacro __using__(_opts) do
    quote do
      use unquote(GenStage)

      def start_link({args, consumer, arity}) do
        IO.inspect args, label: :start_link
        {name, args} = Keyword.pop(args, :name)

        case name do
          nil  -> GenStage.start_link(__MODULE__, {args, consumer, arity})
          name -> GenStage.start_link(__MODULE__, {args, consumer, arity}, name: name)
        end
      end

      def init({opts, consumer, arity}) do
        IO.inspect opts
        case handle_init(opts, arity) do
          {:ok, state} ->
            st = %{
              state:  state
            }
            GenStage.sync_subscribe(consumer, to: self())
            {:producer, st}
          error        ->
            {:error, error}
        end
      end

      def handle_demand(demand, %{state: state}) do
        case handle_read(demand, state) do
          {:reply, data, state}  ->
            data = is_list(data) && data || [data]
            {:noreply, data, %{state: state}}
          {:error, error, state} ->
            {:ok, state} = handle_exit({:error, error}, state)
            {:stop, error, %{state: state}}
          error                  ->
            {:ok, state} = handle_exit(error, state)
            {:stop, error, %{state: state}}
        end
      end

      def handle_cast(:exit, state) do
        {:ok, state} = handle_exit({:stop, :normal}, state)
        {:stop, :normal, state}
      end

      defp handle_read(_number, state), do: {:reply, [], state}

      defp handle_init(_opts, _arity), do: {:ok, nil}

      defp handle_exit(_reason, state), do: {:ok, state}

      defoverridable handle_exit: 2, handle_init: 2, handle_read: 2
    end
  end
end
