defmodule ConvolutionTest do
  use ExUnit.Case

  test "sample layer" do
    i = [0.1, 0.2, 0.3, 0.4,
         0.5, 0.6, 0.7, 0.8,
         1.0, 0.1, 0.2, 0.3,
         0.4, 0.5, 0.6, 0.7]
    w = [[1.0, 2.0,
          3.0, 4.0],
         [5.0, 6.0,
          7.0, 8.0]]
    i = i |> Enum.reduce(<<>>, & &2 <> <<&1::float-little-32>>)
    w = w |> List.flatten |> Enum.reduce(<<>>, & &2 <> <<&1::float-little-32>>)
    o = Neuro.Convolution.run({4, 4, i}, {2, 2, 2, w}, {1, 1})
    o = Enum.map(o, & Float.round(&1, 1))

    # 4.4 = 0.1 * 1.0 + 0.2 * 2.0 + 0.5 * 3.0 + 0.6 * 4.0
    # 5.4 = 0.2 * 1.0 + 0.3 * 2.0 + 0.6 * 3.0 + 0.7 * 4.0
    # ...
    assert o == [
      4.4, 5.4, 6.4,
      5.1, 3.1, 4.1,
      4.4, 4.4, 5.4,

      10.0, 12.6, 15.2,
      13.9,  9.5, 12.1,
      12.4, 10.0, 12.6
    ]
  end
end
