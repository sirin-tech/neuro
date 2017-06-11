defmodule Neuro.Stream.TestConsumer do
  require Logger
  use Neuro.Stream.Consumer

  defp handle_data({:data, data}, _state) do
    IO.inspect :received, label: :handle_data
    :ok
  end

  defp handle_eof({:eof, eof}, _state) do
    IO.inspect eof, label: :handle_eof
    :ok
  end

  defp handle_errors(error, _state) do
    IO.inspect error, label: :handle_error
    :ok
  end

  def start() do
    Logger.info("TestConsumer starting...")
    {:ok, c} = start_link()
    Logger.info("Stream starting...")
    {:ok, s} = Neuro.Stream.start_link([
      consumer: c,
      module: Neuro.Stream.MNISTReader,
      module_opts: [images: ~s[/projects/ai/train-images.idx3-ubyte], labels: ~s[/projects/ai/train-labels.idx1-ubyte]],
      arity: 4
      ])
    Logger.info("Send 5 requests...")
    Neuro.Stream.read(c, 5)
  end

end
