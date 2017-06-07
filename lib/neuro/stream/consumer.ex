defmodule Neuro.Stream.Consumer do
  defmacro __using__(_opts) do
    quote do
      use unquote(GenStage)

      def start_link(opts \\ nil) do
        GenStage.start_link(__MODULE__, opts)
      end

      def init(opts) do
        case handle_init(opts) do
          {:ok, state} ->
            {:consumer, %{state: state}}
          {:error, _} = error ->
            error
          error ->
            error
        end
      end

      def handle_subscribe(:producer, _opts, from, state) do
        {:manual, Map.put(state, :producer, from)}
      end

      # def handle_info({:data_reader_ask, num}, %{producer: p} = state) do
      #   GenStage.ask(p, num)
      #   {:noreply, [], state}
      # end
      def handle_cast({:ask, num}, %{producer: p} = state) do
        GenStage.ask(p, num)
        {:noreply, [], state}
      end

      def handle_cast({:stream_register, pid}, state) do
        {:noreply, [], Map.put(state, :stream, pid)}
      end

      def handle_events(events, _from, %{state: state} = st) do
        st = events
        |> Enum.reduce_while(st, fn
          {:data, _} = data, st ->
            data
            |> handle_data(state)
            |> parse_reply(st)
          {:eof, _} = eof, st   ->
            eof
            |> handle_eof(state)
            |> parse_reply(st)
          error, st             ->
            error
            |>  handle_errors(state)
            |> parse_reply(st)
        end)
        {:noreply, [], state}
      end

      defp handle_init(_opts), do: {:ok, nil}

      defp handle_data(_data, _state), do: :ok

      defp handle_errors(_errors, _state), do: :ok

      defp handle_eof(_eof, _state), do: :ok

      defp parse_reply(reply, %{state: state, stream: stream} = st) do
        case reply do
          :ok          ->
            {:cont, st}
          {:ok, state} ->
            {:cont, %{st | state: state}}
          :stop ->
            if is_pid(stream), do: Neuro.Stream.close(stream)
            {:halt, %{st | stream: nil}}
          {:stop, state} ->
            if is_pid(stream), do: Neuro.Stream.close(stream)
            {:halt, %{st | state: state, stream: nil}}
          _              ->
            {:cont, st}
        end
      end

      defoverridable handle_data: 2, handle_eof: 2, handle_errors: 2, handle_init: 1
    end
  end
end
