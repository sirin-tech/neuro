defmodule FullyConnectedTest do
  use ExUnit.Case

  test "sample  fully connected layer" do
    i = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
         1.5, 0.7, 6.4, 3.2, 2.1, 1.1, 0.1, 9.0]

    o = Neuro.FullyConnected.run({i, w, 8}, 2)
    o = Enum.map(o, & Float.round(&1, 1))

    # 20.4 = relu(0.1 * 1.0 + 0.2 * 2.0 + ... + 0.8 * 8.0)
    # 12.5 = relu(0.1 * 1.5 + 0.2 * 0.7 + ... + 0.8 * 9.0)
    # ...
    assert o == [20.4, 12.5]
  end
end
