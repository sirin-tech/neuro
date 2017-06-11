defmodule Neuro.Stream.MNISTReader do
  require Logger

  use Neuro.Stream.Producer

  @lbl_hdr_offset 8
  @img_hdr_offset 16

  defp handle_init(opts, arity) do
    img_path = Keyword.get(opts, :images)
    lbl_path = Keyword.get(opts, :labels)

    {:ok, fimg} = File.open(img_path, [:binary])
    {:ok, flbl} = File.open(lbl_path, [:binary])
    {:ok, img_count} = count(fimg)
    {:ok, lbl_count} = count(flbl)
    {:ok, {rows, columns}} = image_size(fimg)
    block_offset = rows * columns

    ret_value = cond do
      img_count != lbl_count ->
        {:error, :wrong_sources}
      Cuda.Memory.size(arity) !=  block_offset ->
        {:error, :wrong_arity_size}
      true ->
        :rand.seed(:exsplus, :os.timestamp())

        Logger.info("Loading MNIST data...")
        values = read_all(flbl, fimg, rows, block_offset, img_count)
        Logger.info("MNIST data loaded successfully")
        
        {:ok, values}
    end

    File.close(flbl)
    File.close(fimg)

    ret_value
  end

  defp handle_read(number, state) do
    {state, answer} = read_random(number, state)
    {:reply, answer, state}
  end

  defp handle_exit(_, state) do
    {:ok, state}
  end

  defp read_random(number, values, result \\ [])
  defp read_random(_, [], result),     do: {[], result ++ [{:eof, nil}]}
  defp read_random(0, values, result), do: {values, result}
  defp read_random(number, values, result) do
    lng = length(values)
    index = if lng > 1 do
      :rand.uniform(lng) - 1
    else
      0
    end
    {rand, values} = List.pop_at(values, index)
    read_random(number - 1, values, result ++ [{:data, rand}])
  end

  defp read_all(flbl, fimg, rows, block_offset, count) do
    :file.position(flbl, @lbl_hdr_offset)
    :file.position(fimg, @img_hdr_offset)
    Enum.map(0..(count - 1), fn _ ->
      {:ok, lbl} = lbl_read(flbl)
      {:ok, img} = img_read(fimg, rows, block_offset)
      {lbl, img}
    end)
  end

  defp lbl_read(flbl, index \\ nil) do
    if is_integer(index), do:
      :file.position(flbl, @lbl_hdr_offset + index)
    case IO.binread(flbl, 1) do
      :eof           -> {:error, :eof}
      <<i::integer>> -> {:ok, i}
    end
  end

  defp img_read(fimg, rows, block_offset, index \\ nil) do
    if is_integer(index) do
      idx_offset = index * block_offset
      :file.position(fimg, @img_hdr_offset + idx_offset)
    end
    case IO.binread(fimg, block_offset) do
      :eof -> {:error, :eof}
      b    -> {:ok, to_int(b, rows)}
    end
  end

  defp to_int(binary, rsize, raw \\ [], result \\ [])
  defp to_int(<<>>, _, _, result), do: result
  defp to_int(<<px::big-size(8),rest::binary>>, rsize, raw, result) do
    raw = raw ++ [px / 255]
    if length(raw) == rsize do
      result = List.insert_at(result, length(result) , raw)
      to_int(rest, rsize, [], result)
    else
      to_int(rest, rsize, raw, result)
    end
  end

  defp count(file) do
    :file.position(file, 4)
    <<count::big-size(32)>> = IO.binread(file, 4)
    {:ok, count}
  end

  defp image_size(fimg) do
    :file.position(fimg, 8)
    <<rows::big-size(32),columns::big-size(32)>> = IO.binread(fimg, 8)
    {:ok, {rows, columns}}
  end
end
